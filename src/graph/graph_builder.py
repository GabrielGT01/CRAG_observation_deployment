
"""Graph builder for the Corrective RAG LangGraph workflow"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.rag_state import GraphState
from src.node.node import RAGNodes


class GraphBuilder:
    """Builds and manages the Corrective RAG LangGraph workflow"""

    def __init__(self, retriever):
        """
        Initialize graph builder
        Args:
            retriever: Document retriever instance — built upstream from any
                       mix of PDF, TXT, or URL sources via DocumentProcessor + VectorStore
        """
        self.nodes = RAGNodes(retriever)
        self.memory = MemorySaver()
        self.graph = None

    def build(self):
        """
        Build the Corrective RAG workflow graph
        Returns:
            Compiled graph instance
        """
        builder = StateGraph(GraphState)

        # Register nodes
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("grade_documents", self.nodes.grade_documents)
        builder.add_node("generate", self.nodes.generate_answer)
        builder.add_node("transform_query", self.nodes.transform_query)
        builder.add_node("web_search", self.nodes.web_search)

        # Fixed edges: START -> retrieve -> grade_documents -> (conditional)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade_documents")

        # Conditional edge: after grading, decide_to_generate picks the next node
        builder.add_conditional_edges(
            "grade_documents",
            self.nodes.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        # Correction path: rewrite question -> web search -> generate -> END
        builder.add_edge("transform_query", "web_search")
        builder.add_edge("web_search", "generate")

        # Direct path
        builder.add_edge("generate", END)

        self.graph = builder.compile(checkpointer=self.memory)
        return self.graph

    def run(self, question: str, thread_id: str = "default") -> dict:
        """
        Run the Corrective RAG workflow
        Args:
            question: User question
            thread_id: Conversation thread ID — reuse the same ID across calls
                       so the checkpointer carries chat history forward
        Returns:
            Final state, including 'generation' (the answer) and 'messages'
        """
        if self.graph is None:
            self.build()

        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke({"question": question}, config=config)
