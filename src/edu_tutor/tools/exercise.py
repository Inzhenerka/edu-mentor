from pathlib import Path
import random
from typing import Literal, Self

import yaml
from langchain.tools import BaseTool, tool
from pydantic import BaseModel

type TopicType = Literal["python", "git", "shell", "sql"]
type DifficultyType = Literal["easy", "medium"]


class Exercise(BaseModel):
    """Одно практическое задание из банка."""
    id: str
    topic: TopicType
    difficulty: DifficultyType
    statement: str

    @classmethod
    def list_from_yaml_file(cls, exercises_file: Path) -> list[Self]:
        raw = yaml.safe_load(exercises_file.read_text(encoding="utf-8"))
        return [cls.model_validate(item) for item in raw]


def make_suggest_exercise(data_file: Path) -> BaseTool:
    """Фабрика тула: загружает банк задач и создает готовый к привязке к агенту BaseTool."""

    # Загрузка списка задач из файла
    exercises = Exercise.list_from_yaml_file(data_file)

    @tool
    def suggest_exercise(topic: TopicType, difficulty: DifficultyType = "easy") -> str:
        """Подбирает практическое задание для студента по заданной теме и уровню сложности.

        Используй, когда студент просит "дай задачу", "хочу попрактиковаться", "есть задание на ...".
        """

        # Фильтруем упражнения по теме и сложности
        candidates = [ex for ex in exercises if ex.topic == topic and ex.difficulty == difficulty]
        if not candidates:
            return f"В банке нет задач по теме {topic!r} с уровнем {difficulty!r}."

        # Выбираем упражнение случайным образом
        picked = random.choice(candidates)
        return (
            f"Задание (#{picked.id}, тема: {picked.topic}, уровень: {picked.difficulty}):\n\n"
            f"{picked.statement}"
        )

    return suggest_exercise
