from typing import Literal, Self
import yaml
from pathlib import Path
from pydantic import BaseModel

type LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppConfig(BaseModel):
    log_level: LogLevelType = "INFO"


class LLMConfig(BaseModel):
    model: str
    provider: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_output_tokens: int = 1024


class EmbedderConfig(BaseModel):
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    dimensions: int = 1536
    timeout: float = 30.0


class StoreConfig(BaseModel):
    collection: str
    location: str = ":memory:"


class RetrieverConfig(BaseModel):
    k: int = 3
    score_threshold: float = 0.6


class RAGConfig(BaseModel):
    embedder: EmbedderConfig
    store: StoreConfig
    retriever: RetrieverConfig


class ToolsConfig(BaseModel):
    exercises_file: Path = Path("data/exercises.yml")


class Config(BaseModel):
    app: AppConfig
    llms: dict[str, LLMConfig]
    rag: RAGConfig
    tools: ToolsConfig

    @classmethod
    def from_yaml_file(cls, config_path: str | Path = "config.yml") -> Self:
        config_str = Path(config_path).read_text(encoding="utf-8")
        config_dict = yaml.safe_load(config_str)
        return cls.model_validate(config_dict)
