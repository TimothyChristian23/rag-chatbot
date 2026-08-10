"""Tests for vector store helper behavior."""
import pytest
from langchain_core.documents import Document

from src.retrieval.vectorstore import _document_ids, build_vectorstore


def test_build_vectorstore_rejects_empty_chunks():
    with pytest.raises(ValueError, match="zero document chunks"):
        build_vectorstore([])


def test_document_ids_are_deterministic_and_unique():
    chunks = [
        Document(page_content="same content", metadata={"source": "guide.txt", "page": 1}),
        Document(page_content="same content", metadata={"source": "guide.txt", "page": 1}),
        Document(page_content="different content", metadata={"source": "guide.txt", "page": 2}),
    ]

    first_ids = _document_ids(chunks)
    second_ids = _document_ids(chunks)

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first_ids[0].startswith("guide.txt:1:0:")
