from typing import NotRequired

from langchain.agents.middleware import AgentState
from langchain_core.documents import Document


class MentorAgentState(AgentState):
    chunks: NotRequired[list[Document]]
