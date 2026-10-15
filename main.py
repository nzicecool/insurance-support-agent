import json
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, Field

app = FastAPI(title="Support Agent for Insurance Products")
ROLE = "an insurance support assistant for read-only product guidance"
DESIGN_CONTEXT = "The agent supports customers seeking insurance product guidance. Treat official product documentation and current terms as the source of truth. Do not claim to browse or retrieve live content from https://www.aami.com.au/. Do not invent products, quotes, coverage, premiums, eligibility, legal advice, or policy terms. If current product information is required, explain that the customer should verify it through the official insurer channel. Provide concise, clearly qualified guidance only and do not perform purchases, policy changes, claims actions, or customer-account actions."
MODEL = "gemini-3-flash-preview"
TRACER_NAME = "agent-studio"

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"status": "ok", "agent": "supportagent", "managedModelConfigured": bool(os.getenv("LLM_PROVIDER_URL") and os.getenv("LLM_PROVIDER_KEY")), "model": MODEL}

def trace_content_enabled() -> bool:
    return os.getenv("AMP_TRACE_CONTENT", "true").lower() != "false"

def trace_messages(messages: list[dict[str, Any]]) -> str:
    if trace_content_enabled():
        return json.dumps(messages, ensure_ascii=False)
    return json.dumps([{**message, "content": "[redacted]"} for message in messages], ensure_ascii=False)

def finops_post(suffix: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    base_url = os.getenv("FINOPS_API_BASE_URL", "").rstrip("/")
    token = os.getenv("FINOPS_RUNTIME_TOKEN", "")
    if not base_url or not token:
        return None
    request = Request(
        base_url + suffix,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-FinOps-Token": token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=2.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None

def finops_preflight(messages: list[dict[str, Any]]) -> tuple[str, bool, str]:
    request_id = str(uuid.uuid4())
    if not os.getenv("FINOPS_API_BASE_URL", "") or not os.getenv("FINOPS_RUNTIME_TOKEN", ""):
        return request_id, True, "FinOps runtime is not configured"
    estimated_input_tokens = max(1, sum(len(str(message.get("content", ""))) for message in messages) // 4)
    response = finops_post("/finops/runtime/preflight", {
        "agentName": os.getenv("FINOPS_AGENT_NAME", "supportagent"),
        "projectName": os.getenv("FINOPS_PROJECT_NAME", "default"),
        "model": MODEL,
        "expectedInputTokens": estimated_input_tokens,
        "expectedOutputTokens": max(64, estimated_input_tokens // 2),
        "requestId": request_id,
    })
    if not response:
        return request_id, False, "Governance preflight is unavailable"
    return request_id, bool(response.get("allowed")), str(response.get("reason") or "Governance preflight denied the model call")

def record_finops_usage(request_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
    finops_post("/finops/runtime/usage", {
        "agentName": os.getenv("FINOPS_AGENT_NAME", "supportagent"),
        "projectName": os.getenv("FINOPS_PROJECT_NAME", "default"),
        "model": model,
        "inputTokens": max(0, input_tokens),
        "outputTokens": max(0, output_tokens),
        "requestId": request_id,
    })

def build_messages(request: ChatRequest) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": f"You are {ROLE}. Follow this approved design context:\n{DESIGN_CONTEXT}\n\nDo not claim to take actions outside the information supplied by the caller."},
        {"role": "user", "content": request.message},
    ]
    if request.context:
        messages[-1]["content"] += "\n\nCaller context:\n" + json.dumps(request.context, ensure_ascii=False)
    return messages

def managed_completion(messages: list[dict[str, Any]]) -> tuple[str, str, int, int]:
    base_url = os.getenv("LLM_PROVIDER_URL", "").rstrip("/")
    api_key = os.getenv("LLM_PROVIDER_KEY", "")
    if not base_url or not api_key:
        raise HTTPException(status_code=503, detail="Agent Manager managed model binding is not available")
    payload = json.dumps({"model": MODEL, "messages": messages}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if os.getenv("LLM_PROVIDER_AUTH_STYLE", "").lower() == "api-key":
        headers["API-Key"] = api_key
    else:
        headers["Authorization"] = "Bearer " + api_key
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span("chat", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "agent-manager-managed-provider")
        span.set_attribute("gen_ai.request.model", MODEL)
        span.set_attribute("gen_ai.input.messages", trace_messages(messages))
        upstream = Request(base_url + "/chat/completions", data=payload, headers=headers, method="POST")
        try:
            with urlopen(upstream, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            span.set_status(Status(StatusCode.ERROR, "managed model request failed"))
            span.set_attribute("error.type", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Managed model request failed") from exc
        choices = body.get("choices") if isinstance(body, dict) else None
        content = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(content, str) or not content.strip():
            span.set_status(Status(StatusCode.ERROR, "managed model returned no response"))
            raise HTTPException(status_code=502, detail="Managed model returned no response")
        usage = body.get("usage") if isinstance(body, dict) else {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0
        output_tokens = int(usage.get("completion_tokens", 0) or 0) if isinstance(usage, dict) else 0
        response_model = str(body.get("model") or MODEL) if isinstance(body, dict) else MODEL
        span.set_attribute("gen_ai.response.model", response_model)
        span.set_attribute("gen_ai.output.messages", trace_messages([{"role": "assistant", "content": content.strip()}]))
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        return content.strip(), response_model, input_tokens, output_tokens

@app.post("/chat")
def chat(request: ChatRequest):
    messages = build_messages(request)
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span("invoke_agent", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.system", "agent-studio")
        span.set_attribute("gen_ai.agent.name", "supportagent")
        span.set_attribute("gen_ai.agent.description", ROLE)
        span.set_attribute("gen_ai.request.model", MODEL)
        span.set_attribute("gen_ai.system_instructions", DESIGN_CONTEXT if trace_content_enabled() else "[redacted]")
        span.set_attribute("gen_ai.conversation.id", request.session_id or "")
        span.set_attribute("gen_ai.input.messages", trace_messages(messages))
        request_id, allowed, reason = finops_preflight(messages)
        if not allowed:
            span.set_status(Status(StatusCode.ERROR, reason))
            span.set_attribute("error.type", "GovernancePreflightError")
            raise HTTPException(status_code=503, detail=reason)
        try:
            content, response_model, input_tokens, output_tokens = managed_completion(messages)
        except HTTPException as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc.detail)))
            span.set_attribute("error.type", "ManagedModelRequestError")
            raise
        span.set_attribute("gen_ai.response.model", response_model)
        span.set_attribute("gen_ai.output.messages", trace_messages([{"role": "assistant", "content": content}]))
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        record_finops_usage(request_id, response_model, input_tokens, output_tokens)
        return {"response": content, "sessionId": request.session_id, "model": MODEL}
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )

