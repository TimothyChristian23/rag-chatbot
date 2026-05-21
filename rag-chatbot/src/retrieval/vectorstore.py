"""
Vector store: embeds document chunks and stores them in ChromaDB.
Also handles similarity search at query time.
"""
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
TOP_K = int(os.getenv("TOP_K_RESULTS", 5))


def get_embeddings():
    """Return the embedding model (swap this out to change providers)."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and persist them to ChromaDB."""
    print("Building vector store — this may take a moment...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
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
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def retrieve(query: str, vectorstore: Chroma) -> list:
    """Return the top-K most relevant chunks for a query."""
    return vectorstore.similarity_search(query, k=TOP_K)
