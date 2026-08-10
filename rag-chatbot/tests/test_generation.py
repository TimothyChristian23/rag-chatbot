"""Tests for the OPT assistant prompt and context formatting."""
from langchain_core.documents import Document

from src.generation.chain import (
    LEGAL_DISCLAIMER,
    SYSTEM_PROMPT,
    collect_sources,
    format_context,
)


def test_prompt_contains_opt_guardrails():
    assert "OPT-focused assistant" in SYSTEM_PROMPT
    assert "Do not invent deadlines" in SYSTEM_PROMPT
    assert "DSO" in SYSTEM_PROMPT
    assert "qualified immigration attorney" in SYSTEM_PROMPT
    assert "not legal advice" in LEGAL_DISCLAIMER


def test_format_context_includes_source_and_page():
    docs = [
        Document(
            page_content="OPT content",
            metadata={"source": "uscis-practical-training.txt", "page": 2},
        )
    ]

    context = format_context(docs)

    assert "[Source: uscis-practical-training.txt, page 2]" in context
    assert "OPT content" in context


def test_format_context_handles_empty_docs():
    assert format_context([]) == "No retrieved context."


def test_collect_sources_returns_sorted_unique_sources():
    docs = [
        Document(page_content="A", metadata={"source": "z.txt"}),
        Document(page_content="B", metadata={"source": "a.txt"}),
        Document(page_content="C", metadata={"source": "z.txt"}),
    ]

    assert collect_sources(docs) == ["a.txt", "z.txt"]
