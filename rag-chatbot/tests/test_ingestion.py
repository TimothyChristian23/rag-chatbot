"""Basic tests for the ingestion pipeline."""
from langchain_core.documents import Document

from src.ingestion.loader import split_documents


def test_split_documents_basic():
    docs = [Document(page_content="Hello world. " * 100, metadata={"source": "test.txt"})]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # chunk_size + some tolerance


def test_split_documents_preserves_metadata():
    docs = [Document(page_content="Test content. " * 50, metadata={"source": "myfile.pdf", "page": 1})]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "myfile.pdf"
