from typing import Self

from langchain_core.documents import Document
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Метаданные чанка: паспорт документа, позиция фрагмента, уникальный id."""

    # Поля из паспорта корпуса
    doc_id: str
    file: str
    title: str
    author: str | None = None
    source: str | None = None
    source_url: str | None = None
    section: str | None = None
    content_scope: str | None = None

    # Поля, появляющиеся при нарезке текста на чанки
    start_index: int
    chunk_id: str

    @classmethod
    def from_document(cls, document: Document) -> Self:
        return cls.model_validate(document.metadata)
