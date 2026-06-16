from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import BaseModel
import yaml

from evals.contracts import ExampleInput, ExampleDict, RetrievalOutput, EmptyOutput


class BaseExample[OutputT: BaseModel](BaseModel):
    """Базовый эталонный датасет, экспортируемый в формат Phoenix."""
    example_id: str
    input: ExampleInput
    output: OutputT
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> ExampleDict:
        """Создание полноценного примера для датасета Phoenix."""
        return ExampleDict(
            input=self.input.model_dump(),
            output=self.output.model_dump(),
            metadata=(self.metadata or {}) | {"example_id": self.example_id},
        )


class BaseDataset[ExampleT: BaseExample](BaseModel):
    """Базовая модель эталонного датасета с параметрической моделью кейса.
    Позволяет транслировать пример из файла в подготовленный словарь для Phoenix."""
    __file_path__: ClassVar[str]

    version: str
    examples: list[ExampleT]

    @classmethod
    def from_yaml_file(cls) -> Self:
        """Загрузка набора данных из YAML-файла."""
        return cls.model_validate(yaml.safe_load(Path(cls.__file_path__).read_text(encoding="utf-8")))

    def to_example_dicts(self) -> list[ExampleDict]:
        """Создание полноценных Phoenix-примеров для датасета."""
        return [example.to_dict() for example in self.examples]


class RetrievalExample(BaseExample[RetrievalOutput]):
    """Модель кейса для тестирования ретривера. Задает специфичную модель output."""


class RetrievalDataset(BaseDataset[RetrievalExample]):
    """Модель датасета для тестирования ретривера. Связывает файл на диске и модель кейса."""
    __file_path__ = "evals/datasets/retrieval.yml"


class SecurityExample(BaseExample[EmptyOutput]):
    """Модель кейса для тестирования агента на безопасность."""
    output: EmptyOutput = EmptyOutput()  # Эталонный ответ не предполагается


class SecurityDataset(BaseDataset[SecurityExample]):
    """Модель датасета для тестирования агента на безопасность."""
    __file_path__ = "evals/datasets/security.yml"
