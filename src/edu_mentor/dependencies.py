import sys
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from loguru import logger

from edu_mentor.agent import Mentor
from edu_mentor.config import Config
from edu_mentor.observability import setup_observability
from edu_mentor.rag.embedder import get_embedder
from edu_mentor.rag.vector_store import get_vector_store


def init_global_dependencies(app: FastAPI) -> None:
    """Инициализация и подготовка всех зависимостей агента."""
    # Загрузка переменных из .env
    load_dotenv()

    # Загрузка конфигурации
    config = Config.from_yaml_file("config.yml")

    # Настройка глобального логгера из конфига
    logger.remove()
    logger.add(sys.stdout, level=config.app.log_level)

    # Подключение к Phoenix для мониторинга - до создания агента
    setup_observability(config.observability)
    logger.info(f"Phoenix observability ready: project={config.observability.project_name}")

    # Подключение к существующей коллекции Qdrant
    embedder = get_embedder(config.rag.embedder)
    vector_store = get_vector_store(config.rag.store, embedder)
    logger.info(f"Vector store ready: {config.rag.store.collection} @ {config.rag.store.location}")

    # Создание агента.
    app.state.agent = Mentor(
        llm_key="api",
        config=config,
        vector_store=vector_store,
        debug=False,
    )


def get_agent(request: Request) -> Mentor:
    return request.app.state.agent


type AgentDependency = Annotated[Mentor, Depends(get_agent)]
