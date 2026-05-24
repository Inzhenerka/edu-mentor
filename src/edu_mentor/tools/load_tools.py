from langchain.tools import BaseTool

from edu_mentor.config import ToolsConfig
from edu_mentor.tools.exercise import make_suggest_exercise
from edu_mentor.tools.consultation import book_consultation


def load_tools(config: ToolsConfig) -> list[BaseTool]:
    return [
        make_suggest_exercise(config.exercises_file),
        book_consultation,
    ]
