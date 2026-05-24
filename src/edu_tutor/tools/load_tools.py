from langchain.tools import BaseTool

from edu_tutor.config import ToolsConfig
from edu_tutor.tools.exercise import make_suggest_exercise
from edu_tutor.tools.consultation import book_consultation


def load_tools(config: ToolsConfig) -> list[BaseTool]:
    return [
        make_suggest_exercise(config.exercises_file),
        book_consultation,
    ]
