from datetime import date
from typing import ClassVar
from pathlib import Path

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.messages import HumanMessage


class BasePrompt(BaseModel):
    __file_path__: ClassVar[str | Path]

    def render_prompt(self) -> str:
        text = Path(self.__file_path__).read_text(encoding="utf-8")
        template = Template(text, undefined=StrictUndefined)
        return template.render(self.model_dump()).strip()

    def render_human_message(self) -> HumanMessage:
        return HumanMessage(content=self.render_prompt())


class TutorPrompt(BasePrompt):
    __file_path__ = "prompts/templates/tutor.jinja"
    today: str = Field(default_factory=lambda: date.today().isoformat())  # Текущая дата


class TutorHumanMessage(BasePrompt):
    __file_path__ = "prompts/messages/tutor_human.jinja"
    question: str
    chunks: list[Document]
