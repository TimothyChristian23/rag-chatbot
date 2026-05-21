"""
Document loader: reads PDFs and .txt files from the data/documents folder,
splits them into chunks, and returns LangChain Document objects.
"""
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DOCUMENTS_DIR = Path("data/documents")


def load_documents(directory: Path = DOCUMENTS_DIR) -> list:
    """Load all PDFs and .txt files from a directory."""
    docs = []
    for file in directory.iterdir():
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} raw document pages from {directory}")
    return docs


def split_documents(docs: list) -> list:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
