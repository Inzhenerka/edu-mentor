import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain.messages import HumanMessage
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from edu_mentor.chat_model import load_chat_model
from edu_mentor.config import Config
from edu_mentor.middleware.rag import RAGMiddleware
from edu_mentor.prompts import MentorPrompt
from edu_mentor.rag.chunk import ChunkMetadata
from edu_mentor.rag.provenance import parse_referenced_chunks
from edu_mentor.rag.retriever import get_retriever
from edu_mentor.tools.load_tools import load_tools


class MentorResponse(BaseModel):
    content: str
    thread_id: str
    sources: dict[int, ChunkMetadata]


class Mentor:
    _agent: CompiledStateGraph

    def __init__(self, llm_key: str, config: Config, vector_store: VectorStore, debug: bool = False):
        """Инициализация обертки агента."""

        # Создаем модель чата
        chat_model = load_chat_model(config.llms[llm_key])

        # Создаем ретривер
        retriever = get_retriever(config.rag.retriever, vector_store=vector_store)

        self._agent = create_agent(
            model=chat_model,
            tools=load_tools(config.tools),
            system_prompt=MentorPrompt().render_prompt(),
            checkpointer=InMemorySaver(),
            middleware=[
                ModelRetryMiddleware(max_retries=2, initial_delay=1),
                ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
                RAGMiddleware(retriever=retriever),
            ],
            debug=debug,
        )

    def invoke(
        self,
        prompt: str,
        thread_id: str | None = None,
    ) -> MentorResponse:
        """Вызываем агента и извлекаем ответ из массива сообщений"""

        # Ограничиваем длину запроса для экономии токенов
        prompt = prompt.strip()[:4000]

        # Используем принятый или создаем новый ID чата
        effective_thread_id = thread_id or str(uuid.uuid4())

        # Формируем и передаем агенту сообщение
        response = self._agent.invoke(
            input={"messages": HumanMessage(prompt)},
            config={"configurable": {"thread_id": effective_thread_id}},
        )

        # Извлекаем из состояния ответ LLM и исходные чанки для обработки
        response_content = response["messages"][-1].content
        chunks = response.get("chunks") or []

        # Обрабатываем сноски в ответе, формируем из чанков словарь источников
        sources = parse_referenced_chunks(
            content=response_content,
            chunks=chunks,
        )

        # Формируем ответ, включая источники и вызванные инструменты
        return MentorResponse(
            content=response_content,
            sources=sources,
            thread_id=effective_thread_id,
        )
