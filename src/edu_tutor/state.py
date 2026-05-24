from typing import NotRequired

from langchain.agents.middleware import AgentState
from langchain_core.documents import Document


class TutorAgentState(AgentState):
    chunks: NotRequired[list[Document]]
