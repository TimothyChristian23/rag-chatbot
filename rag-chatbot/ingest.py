"""
Run this script once to ingest all documents in data/documents/
and build the ChromaDB vector store.

Usage:
    python ingest.py
"""
from src.ingestion.loader import load_documents, split_documents
from src.retrieval.vectorstore import build_vectorstore

if __name__ == "__main__":
    print("=== OPT Assistant - Document Ingestion ===")
    docs = load_documents()
    if not docs:
        print("No documents found in data/documents/. Add some PDFs or .txt files first.")
        exit(1)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Done! Run `uvicorn src.api.main:app --reload` to start the API.")
