from pathlib import Path

from langchain.tools import BaseTool

from edu_mentor.tools.exercise import make_suggest_exercise
from edu_mentor.tools.consultation import book_consultation


def load_tools() -> list[BaseTool]:
    return [
        make_suggest_exercise(Path("data/exercises.yml")),
        book_consultation,
    ]
