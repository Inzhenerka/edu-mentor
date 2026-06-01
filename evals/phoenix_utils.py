from urllib.parse import urlparse

from phoenix.client import Client


def get_phoenix_client(endpoint: str) -> Client:
    """Создаем REST-клиент Phoenix."""
    # Определяем базовый адрес из произвольного входного адреса
    parsed_endpoint = urlparse(endpoint)
    base_url = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}"
    # Создаем клиент
    return Client(base_url=base_url)
