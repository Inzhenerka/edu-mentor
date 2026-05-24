from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import HumanMessage
from langchain_core.messages import filter_messages
from langchain_core.retrievers import BaseRetriever
from langgraph.runtime import Runtime

from edu_tutor.state import TutorAgentState
from edu_tutor.prompts import TutorHumanMessage


class RAGMiddleware(AgentMiddleware[TutorAgentState]):
    """Готовая RAG-обвязка для агента. Подключается одной строкой через передачу ретривера."""

    state_schema = TutorAgentState

    def __init__(self, retriever: BaseRetriever):
        super().__init__()
        self._retriever = retriever

    def before_agent(
        self,
        state: TutorAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Готовим чанки под текущую тему беседы."""

        # Создаем запрос для ретривера из свежей истории сообщений
        query = build_retrieval_query(state, max_messages=3)

        # Получаем чанки с помощью ретривера
        chunks = self._retriever.invoke(query)

        # Обновляем состояние
        return {"chunks": chunks}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Подмешиваем чанки в последнее сообщение пользователя, не оставляя их в состоянии.

        В данном агенте важно правильно найти сообщение пользователя в истории, где присутствуют tool messages.
        """

        # Извлекаем подготовленные чанки из состояния
        chunks = request.state.get("chunks") or []

        # Извлекаем все сообщения человека
        human_messages = filter_messages(request.messages, include_types=HumanMessage)
        # Берем последнее сообщение пользователя
        last_message = human_messages[-1]
        # Находим его индекс для последующей замены сообщения
        last_idx = next(i for i, m in enumerate(request.messages) if m is last_message)

        # Рендерим новое сообщение пользователя
        enriched_message = TutorHumanMessage(
            chunks=chunks,
            question=str(last_message.content),
        ).render_human_message()

        # Подменяем сообщение пользователя, не трогая более поздние tool messages
        new_messages = list(request.messages)
        new_messages[last_idx] = enriched_message

        return handler(request.override(messages=new_messages))


def build_retrieval_query(state: TutorAgentState, max_messages: int = 3) -> str:
    """Собрать запрос для ретривера из последних N сообщений пользователя, чтобы не терять контекст."""
    human_messages = filter_messages(state["messages"], include_types=HumanMessage)
    return "\n".join(str(m.content) for m in human_messages[-max_messages:])
