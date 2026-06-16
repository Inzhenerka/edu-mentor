from urllib.parse import urlparse

from loguru import logger
from phoenix.client import Client
from phoenix.evals import LLM

from src.edu_mentor.config import LLMConfig
from evals.contracts import ExampleDict


def get_phoenix_client(endpoint: str) -> Client:
    """Создаем REST-клиент Phoenix."""
    # Определяем базовый адрес из произвольного входного адреса
    parsed_endpoint = urlparse(endpoint)
    base_url = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}"
    # Создаем клиент
    return Client(base_url=base_url)


def ensure_dataset(
    client: Client,
    name: str,
    examples: list[ExampleDict],
    description: str | None = None
):
    """Находим имеющийся датасет по названию или создаем новый."""
    try:
        # Пытаемся найти датасет по названию
        dataset = client.datasets.get_dataset(dataset=name)
        logger.info(f"Reusing Phoenix Dataset: {name}")
        return dataset
    except ValueError:
        pass

    # Создаем новый датасет
    dataset = client.datasets.create_dataset(
        name=name,
        examples=examples,
        dataset_description=description,
    )
    logger.info(f"New Phoenix Dataset created: {name}")
    return dataset


def get_llm(judge_llm_config: LLMConfig) -> LLM:
    """Подключаемся к LLM для работы судей в рамках Phoenix Evals."""
    return LLM(
        provider=judge_llm_config.provider,
        model=judge_llm_config.model,
        base_url=judge_llm_config.base_url,
        timeout=judge_llm_config.timeout,
    )
