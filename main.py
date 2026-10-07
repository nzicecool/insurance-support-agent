from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Support Agent for Insurance Products")
ROLE = "a helpful enterprise assistant"
DESIGN_CONTEXT = "The agent listens to what the customers wants and provide an answer on suitable policies based on what the company offers.\nYou are an Insurance Support Agent in a read-only simulation.\n\nFor this example use the website https://www.aami.com.au/ and get the product details from the website. Provide only what's available on the website, don't come up with make up products that are not available on the website."

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.get("/health")
def health():
    return {"status": "ok", "agent": "supportagent"}

@app.post("/chat")
def chat(request: ChatRequest):
    # Replace this deterministic scaffold with an Agent Manager managed-model call.
    # The platform injects provider configuration rather than source-held keys.
    return {"answer": f"{ROLE}: {request.message}", "designContext": DESIGN_CONTEXT, "sessionId": request.session_id}
