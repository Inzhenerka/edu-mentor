from typing import Any, TypedDict

from pydantic import BaseModel


class ExampleDict(TypedDict):
    """Сериализованный пример для датасета Phoenix."""
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]


class ExampleInput(BaseModel):
    """Вход для тестирования агента/ретривера."""
    question: str


class RetrievalOutput(BaseModel):
    """Ожидаемый/фактический выход ретривера."""
    chunk_ids: list[str]


class EmptyOutput(BaseModel):
    """Пустой ответ для случаев, когда ответ не предполагается."""


class AgentOutput(BaseModel):
    """Выход задачи агента."""
    answer: str
