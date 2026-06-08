from urllib.parse import urlparse

from loguru import logger
from phoenix.client import Client

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
