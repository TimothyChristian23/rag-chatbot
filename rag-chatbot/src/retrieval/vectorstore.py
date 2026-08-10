"""
Vector store: embeds document chunks and stores them in ChromaDB.
Also handles similarity search at query time.
"""
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "opt_assistant")
TOP_K = int(os.getenv("TOP_K_RESULTS", 5))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings():
    """Return the embedding model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and rebuild the persisted ChromaDB collection."""
    if not chunks:
        raise ValueError("Cannot build a vector store with zero document chunks.")

    print("Building vector store - this may take a moment...")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks, ids=_document_ids(chunks))
    print(f"Vector store saved to {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    if not Path(CHROMA_PERSIST_DIR).exists():
        raise FileNotFoundError(
            f"No vector store found at '{CHROMA_PERSIST_DIR}'. "
            "Run `python ingest.py` first."
        )
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def retrieve(query: str, vectorstore: Chroma) -> list:
    """Return the top-K most relevant chunks for a query."""
    return vectorstore.similarity_search(query, k=TOP_K)


def _document_ids(chunks: list) -> list[str]:
    """Build deterministic IDs so repeated ingestion rewrites the same logical chunks."""
    ids = []
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "unknown")
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{source}:{page}:{index}:{digest}")
    return ids
