
"""End-to-end ingestion: load + chunk + embed any mix of PDF/TXT/URL sources
into a retriever, ready to hand to GraphBuilder."""

from typing import List, Union

from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore


class IngestionPipeline:
    """Turns a list of sources (PDF paths, TXT paths, or URLs) into a retriever"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, k: int = 4):
        """
        Args:
            chunk_size: characters per chunk
            chunk_overlap: overlap between chunks
            k: number of documents the resulting retriever returns per query
        """
        self.processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore()
        self.k = k

    def build_retriever(self, sources: Union[str, List[str]]):
        """
        Load, chunk, and embed the given sources; return a ready-to-use retriever.
        Args:
            sources: a single source or list of sources — any mix of .pdf, .txt
                     file paths, or http(s):// URLs
        Returns:
            A retriever instance (FAISS-backed)
        """
        chunks = self.processor.process_documents(sources)
        self.vector_store.create_vectorstore(chunks, k=self.k)
        return self.vector_store.get_retriever()
