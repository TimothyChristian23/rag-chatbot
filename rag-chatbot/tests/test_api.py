"""Smoke tests for the FastAPI endpoints."""
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import src.api.main as api
from src.api.main import app

client = TestClient(app)


def setup_function():
    api._memory.clear_all()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_rejects_blank_question():
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_uses_same_docs_for_answer_and_sources(monkeypatch):
    docs = [
        Document(
            page_content="OPT source context",
            metadata={"source": "school-opt-guide.txt", "page": 1},
        )
    ]
    captured = {}

    monkeypatch.setattr(api, "_vectorstore", object())
    monkeypatch.setattr(api, "_get_answer_chain", lambda: object())

    def fake_retrieve(question, vectorstore):
        captured["retrieval_query"] = question
        return docs

    monkeypatch.setattr(api, "retrieve", fake_retrieve)

    def fake_generate_answer(question, retrieved_docs, answer_chain, chat_history=None):
        captured["question"] = question
        captured["docs"] = retrieved_docs
        captured["chat_history"] = chat_history
        return "Use your school OPT guide."

    monkeypatch.setattr(api, "generate_answer", fake_generate_answer)

    api._memory.add_exchange("demo", "What is CPT?", "CPT answer")

    response = client.post("/chat", json={"session_id": "demo", "question": "  What is OPT?  "})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "demo"
    assert payload["answer"] == "Use your school OPT guide."
    assert payload["sources"] == ["school-opt-guide.txt"]
    assert payload["source_snippets"] == [
        {
            "source": "school-opt-guide.txt",
            "page": 1,
            "snippet": "OPT source context",
            "rank": 1,
        }
    ]
    assert payload["history"][-2:] == [
        {"role": "user", "content": "What is OPT?"},
        {"role": "assistant", "content": "Use your school OPT guide."},
    ]
    assert captured["question"] == "What is OPT?"
    assert captured["docs"] is docs
    assert captured["chat_history"][0].content == "What is CPT?"
    assert "What is CPT?" in captured["retrieval_query"]
    assert "What is OPT?" in captured["retrieval_query"]


def test_chat_history_endpoints_clear_session():
    api._memory.add_exchange("demo", "What is OPT?", "OPT answer")

    response = client.get("/chat/sessions/demo")

    assert response.status_code == 200
    assert response.json()["history"] == [
        {"role": "user", "content": "What is OPT?"},
        {"role": "assistant", "content": "OPT answer"},
    ]

    response = client.delete("/chat/sessions/demo")

    assert response.status_code == 200
    assert response.json()["history"] == []
