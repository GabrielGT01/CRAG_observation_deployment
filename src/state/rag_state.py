

"""RAG state definition for LangGraph"""

from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of retrieved documents
        messages: running chat history, auto-accumulated across turns via add_messages
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    messages: Annotated[list, add_messages]
    faithfulness: bool
    answer_relevancy: bool
    context_precision: float
