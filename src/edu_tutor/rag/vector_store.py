import os

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from edu_tutor.config import StoreConfig


def get_vector_store(
    config: StoreConfig,
    embedder: OpenAIEmbeddings,
) -> QdrantVectorStore:
    """Подключаемся к хранилищу Qdrant, проверяем наличие коллекции, привязываем embedder."""

    # Создаем базу в памяти, на диске или подключаемся к удаленному сервису
    location = config.location
    if location == ":memory:":
        client = QdrantClient(location=location)
    elif location.startswith(("http://", "https://")):
        client = QdrantClient(url=location, api_key=os.environ["QDRANT_API_KEY"])
    else:
        client = QdrantClient(path=location)

    # Проверяем наличие коллекции в базе
    collection_already_exists = client.collection_exists(config.collection)
    if not collection_already_exists:
        raise RuntimeError(f"Collection {config.collection} does not exist in Qdrant")

    # Создаем объект для работы с векторной базой
    store = QdrantVectorStore(
        client=client,
        collection_name=config.collection,
        embedding=embedder,
    )
    return store
