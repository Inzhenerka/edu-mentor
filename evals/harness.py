import os
from typing import Any, Callable

from dotenv import load_dotenv

from edu_mentor.config import Config
from edu_mentor.rag.embedder import get_embedder
from edu_mentor.rag.retriever import get_retriever
from edu_mentor.rag.vector_store import get_vector_store
from edu_mentor.observability import setup_observability
from edu_mentor.rag.chunk import ChunkMetadata
from edu_mentor.agent import Mentor
from evals.contracts import ExampleInput, RetrievalOutput, AgentOutput


def prepare_environment() -> Config:
    """Загрузка конфигурации приложения, переменных окружения и инструментация."""
    load_dotenv()
    config = Config.from_yaml_file("config.yml")
    setup_observability(config.observability)
    return config


def get_user_id() -> str:
    """Получаем имейл пользователя Phoenix и вычленяем первую часть в качестве ID."""
    return os.environ.get("PHOENIX_USER", "UNKNOWN_USER").split("@", 1)[0] or "UNKNOWN_USER"


def make_retrieval_task(config: Config) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Фабрика задачи для тестирования ретривера.
    Создаем сам ретривер и оборачивает его вызов для Phoenix."""

    # Создание ретривера из конфига через цепочку зависимостей
    embedder = get_embedder(config.rag.embedder)
    vector_store = get_vector_store(config.rag.store, embedder=embedder)
    retriever = get_retriever(config.rag.retriever, vector_store=vector_store)

    def retrieval_task(input: dict[str, Any]) -> dict[str, Any]:
        """Задача Phoenix для вызова ретривера."""

        # Парсим вход примера из словаря
        example_input = ExampleInput.model_validate(input)
        # Вызываем ретривер для тестового запроса пользователя
        documents = retriever.invoke(example_input.question)

        # Формируем выход задачи из выхода ретривера для последующего сравнения с эталонным ответом
        return RetrievalOutput(
            chunk_ids=[ChunkMetadata.from_document(doc).chunk_id for doc in documents]
        ).model_dump()

    return retrieval_task


def make_agent_task(config: Config) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Фабрика задачи для тестирования агента.
    Создаем полноценного агента и оборачиваем его вызов для Phoenix."""

    # Создание агента из конфига через цепочку зависимостей
    embedder = get_embedder(config.rag.embedder)
    vector_store = get_vector_store(config.rag.store, embedder=embedder)
    agent = Mentor(llm_key="api", config=config, vector_store=vector_store)

    def agent_task(input: dict[str, Any]) -> dict[str, Any]:
        """Задача для Phoenix для вызова агента."""

        # Парсим входные данные примера из словаря
        parsed_input = ExampleInput.model_validate(input)
        # Вызываем агента для тестового запроса пользователя
        response = agent.invoke(parsed_input.question)

        # Формируем выход задачи из выхода агента для последующей передачи LLM-судье
        return AgentOutput(answer=response.content).model_dump()

    return agent_task
