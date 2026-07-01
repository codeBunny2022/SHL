import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.agent import Agent
from app.schemas import ChatRequest, ChatResponse, Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent: Agent | None = None
UI_PATH = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    try:
        agent = Agent()
        logger.info("Agent ready with %s catalog items", len(agent.index.items))
    except Exception as exc:
        logger.error("Failed to load agent: %s", exc)
        agent = None
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return UI_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        return ChatResponse(
            reply="Tell me about the role or skills you are hiring for.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Normalize and validate roles alternate user/assistant is not required by spec,
    # but empty content is dropped.
    messages = [
        Message(role=m.role, content=m.content.strip())
        for m in request.messages
        if m.content and m.content.strip()
    ]
    if not messages or messages[-1].role != "user":
        return ChatResponse(
            reply="Please send a user message to continue.",
            recommendations=[],
            end_of_conversation=False,
        )

    if agent is None:
        return ChatResponse(
            reply="Service is starting up. Please try again shortly.",
            recommendations=[],
            end_of_conversation=False,
        )
    try:
        return agent.chat(messages)
    except Exception as exc:
        logger.exception("chat failed: %s", exc)
        return ChatResponse(
            reply="Something went wrong. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
