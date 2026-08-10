"""
Document loader: reads PDFs and .txt files from the data/documents folder,
splits them into chunks, and returns LangChain Document objects.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DOCUMENTS_DIR = Path("data/documents")


def load_documents(directory: Path = DOCUMENTS_DIR) -> list:
    """Load all PDFs and .txt files from a directory."""
    if not directory.exists():
        print(f"No document directory found at {directory}")
        return []

    docs = []
    for file in sorted(directory.iterdir()):
        if file.suffix.lower() == ".pdf":
            docs.extend(_load_pdf(file))
        elif file.suffix.lower() == ".txt":
            docs.append(_load_text(file))

    print(f"Loaded {len(docs)} raw document pages from {directory}")
    return docs


def _load_pdf(file: Path) -> list[Document]:
    """Load a PDF as one Document per page."""
    reader = PdfReader(str(file))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name, "page": page_number},
                )
            )
    return pages


def _load_text(file: Path) -> Document:
    """Load a text file as a single Document."""
    return Document(
        page_content=file.read_text(encoding="utf-8"),
        metadata={"source": file.name, "page": 1},
    )


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
