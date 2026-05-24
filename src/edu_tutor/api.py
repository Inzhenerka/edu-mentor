from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from edu_tutor.agent import TutorResponse
from edu_tutor.dependencies import AgentDependency, init_global_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_global_dependencies(app)
    logger.info("Starting tutor")
    yield
    logger.info("Stopping tutor")


app = FastAPI(lifespan=lifespan)


@app.get("/demo")
def demo():
    """Simple demo of web UI."""
    return FileResponse("templates/demo.html")


@app.post("/ask", response_model=TutorResponse)
def ask(
    agent: AgentDependency,
    question: str = Form(
        description="Student question",
        examples=["Что такое Python?"],
    ),
    thread_id: str | None = Form(default=None, description="Chat thread id"),
) -> TutorResponse:
    """Ask question to tutor agent."""
    try:
        return agent.invoke(prompt=question, thread_id=thread_id)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
