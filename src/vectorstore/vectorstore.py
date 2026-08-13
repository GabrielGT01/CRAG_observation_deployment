
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)


class VectorStore:
    """Creating VectorStore and its operations"""

    def __init__(self):
        """Initialise vectorstore with OpenAI embeddings"""
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document], k: int = 4):
        """
        Receive documents and create the vector store
        k: Number of documents to retrieve
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retriever(self):
        """
        Get the retriever instance
        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        Args:
            query: Search query
        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)
