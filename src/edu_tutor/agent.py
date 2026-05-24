import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, filter_messages
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from edu_tutor.chat_model import load_chat_model
from edu_tutor.config import Config
from edu_tutor.middleware.rag import RAGMiddleware
from edu_tutor.prompts import TutorPrompt
from edu_tutor.rag.chunk import ChunkMetadata
from edu_tutor.rag.provenance import parse_referenced_chunks
from edu_tutor.rag.retriever import get_retriever
from edu_tutor.tools.load_tools import load_tools


class TutorResponse(BaseModel):
    content: str
    thread_id: str
    sources: dict[int, ChunkMetadata]
    tool_calls: set[str]


class Tutor:
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
            system_prompt=TutorPrompt().render_prompt(),
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
    ) -> TutorResponse:
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

        # Собираем имена вызванных тулов из истории сообщений для отображения в демо.
        ai_messages = filter_messages(response["messages"], include_types=AIMessage)
        tool_calls = set([call["name"] for m in ai_messages for call in m.tool_calls])

        # Формируем ответ, включая источники и вызванные инструменты
        return TutorResponse(
            content=response_content,
            sources=sources,
            thread_id=effective_thread_id,
            tool_calls=tool_calls,
        )
