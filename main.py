import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Support Agent for Insurance Products")
ROLE = "an insurance support assistant for read-only product guidance"
DESIGN_CONTEXT = """The agent supports customers seeking insurance product guidance. Treat official product documentation and current terms as the source of truth. Do not claim to browse or retrieve live content from https://www.aami.com.au/. Do not invent products, quotes, coverage, premiums, eligibility, legal advice, or policy terms. If current product information is required, explain that the customer should verify it through the official insurer channel. Provide concise, clearly qualified guidance only and do not perform purchases, policy changes, claims actions, or customer-account actions."""
MODEL = "gemini-3-flash-preview"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent": "supportagent",
        "managedModelConfigured": bool(os.getenv("LLM_PROVIDER_URL") and os.getenv("LLM_PROVIDER_KEY")),
        "model": MODEL,
    }


def managed_completion(request: ChatRequest) -> str:
    base_url = os.getenv("LLM_PROVIDER_URL", "").rstrip("/")
    api_key = os.getenv("LLM_PROVIDER_KEY", "")
    if not base_url or not api_key:
        raise HTTPException(status_code=503, detail="Agent Manager managed model binding is not available")
    messages = [
        {
            "role": "system",
            "content": f"You are {ROLE}. Follow this approved design context:\n{DESIGN_CONTEXT}\n\nDo not claim to take actions outside the information supplied by the caller.",
        },
        {"role": "user", "content": request.message},
    ]
    if request.context:
        messages[-1]["content"] += "\n\nCaller context:\n" + json.dumps(request.context, ensure_ascii=False)
    payload = json.dumps({"model": MODEL, "messages": messages}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if os.getenv("LLM_PROVIDER_AUTH_STYLE", "").lower() == "api-key":
        headers["API-Key"] = api_key
    else:
        headers["Authorization"] = "Bearer " + api_key
    upstream = Request(base_url + "/chat/completions", data=payload, headers=headers, method="POST")
    try:
        with urlopen(upstream, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Managed model request failed") from exc
    choices = body.get("choices") if isinstance(body, dict) else None
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=502, detail="Managed model returned no response")
    return content.strip()


@app.post("/chat")
def chat(request: ChatRequest):
    return {"response": managed_completion(request), "sessionId": request.session_id, "model": MODEL}
