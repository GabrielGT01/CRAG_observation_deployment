


import os
from dotenv import load_dotenv


load_dotenv()  # pulls OPENAI_API_KEY / TAVILY_API_KEY from .env / LANGSMITH tracing


"""LangGraph nodes for the Corrective RAG workflow"""

from typing import Literal
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_PROJECT"] = "TESTING-phase"

from src.state.rag_state import GraphState
from src.node.functions import (
    format_docs,
    retrieval_grader,
    rag_chain,
    question_rewriter,
    web_search_tool,
)
from src.rag_metric.metrics import score_faithfulness, score_answer_relevancy, score_context_precision




class RAGNodes:
    """Contains node functions for the Corrective RAG workflow"""

    def __init__(self, retriever):
        """
        Initialize RAG nodes
        Args:
            retriever: Document retriever instance (built from any source type —
                       PDF, TXT, or URL — via DocumentProcessor + VectorStore)
        """
        self.retriever = retriever
        # Chains/tools are shared singletons defined once in functions.py —
        # imported directly here rather than re-passed through every constructor
        self.rag_chain = rag_chain
        self.retrieval_grader = retrieval_grader
        self.question_rewriter = question_rewriter
        self.web_search_tool = web_search_tool

    @traceable(name="retrieve_chunks", run_type="retriever")
    def retrieve_docs(self, state: GraphState) -> dict:
        """
        Retrieve relevant documents node
        Args:
            state: Current graph state
        Returns:
            Partial state update with retrieved documents
        """
        print("RETRIEVING FROM DATABASE")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state: GraphState) -> dict:
        """
        Grade each retrieved document for relevance.
        Web search is triggered only when none of the retrieved
        documents are relevant.
        Args:
            state: Current graph state with retrieved documents
        Returns:
            Partial state update with filtered documents and web_search flag
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for document in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": document.page_content}
            )
            grade = score.binary_score.strip().lower()

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(document)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        web_search = "Yes" if len(filtered_docs) == 0 else "No"

        if web_search == "Yes":
            print("---NO RELEVANT DOCUMENTS: WEB SEARCH REQUIRED---")
        else:
            print(f"---RELEVANT DOCUMENTS FOUND: {len(filtered_docs)}---")

        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    @traceable(name="generate_answer")
    def generate_answer(self, state: GraphState) -> dict:
        """
        Generate answer from retrieved documents node
        Args:
            state: Current graph state with (graded) retrieved documents
        Returns:
            Partial state update with the generated answer, RAGAS scores, and chat history
        """
        print("Generating the answer")
        question = state["question"]
        documents = state["documents"]
        messages = state.get("messages", [])

        generation = self.rag_chain.invoke(
            {
                "context": format_docs(documents),
                "question": question,
                "chat_history": messages,
            }
        )

        # Score against the same documents that were used to generate the answer.
        # All three are @traceable, so they nest under this node's own trace span
        # automatically -- no extra wiring needed.
        chunks_for_scoring = [{"content": d.page_content} for d in documents]
        faithful = score_faithfulness(question, generation, chunks_for_scoring)
        relevant = score_answer_relevancy(question, generation)
        precision = score_context_precision(question, chunks_for_scoring)

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "faithfulness": faithful,
            "answer_relevancy": relevant,
            "context_precision": precision,
            "messages": [HumanMessage(content=question), AIMessage(content=generation)],
        }

    def transform_query(self, state: GraphState) -> dict:
        """
        Transform the query to produce a better question.
        Args:
            state: Current graph state
        Returns:
            Partial state update with the rewritten question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state: GraphState) -> dict:
        """
        Web search based on the re-phrased question.
        Args:
            state: Current graph state
        Returns:
            Partial state update with web results appended to documents
        """
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        response = self.web_search_tool.invoke({"query": question})
        results = response["results"]
        top_results = results[:3]
        web_results = "\n\n".join(r["content"] for r in top_results)

        documents.append(Document(page_content=web_results))
        return {"documents": documents, "question": question}

    def decide_to_generate(self, state: GraphState) -> Literal["transform_query", "generate"]:
        """
        Route to query transformation when no relevant local
        documents remain. Otherwise, generate an answer.
        Args:
            state: Current graph state
        Returns:
            Name of the next node to route to
        """
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"].strip().lower()

        if web_search == "yes":
            print("---DECISION: TRANSFORM QUERY---")
            return "transform_query"

        print("---DECISION: GENERATE---")
        return "generate"
