
from typing import List, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    WebBaseLoader,
    TextLoader,
)


class DocumentProcessor:
    """Handles the document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialise document processor"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document from a URL or web address"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load PDF file"""
        loader = PyMuPDFLoader(str(file_path))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """Load documents from a mix of URLs, .txt files, or PDF files and return them"""
        ## append = add one object
        ## extend = add many items
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            elif src.endswith(".pdf"):
                docs.extend(self.load_from_pdf(src))
            elif src.endswith(".txt"):
                docs.extend(self.load_from_txt(src))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use a URL, .txt file, or .pdf file."
                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        Args:
            documents: List of documents to split
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)

    def process_documents(self, sources: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents from any supported source type
        Args:
            sources: List of URLs, .txt file paths, or .pdf file paths to process
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)
