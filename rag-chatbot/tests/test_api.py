"""Smoke tests for the FastAPI endpoints."""
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import src.api.main as api
from src.api.main import app

client = TestClient(app)


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
    monkeypatch.setattr(api, "retrieve", lambda question, vectorstore: docs)

    def fake_generate_answer(question, retrieved_docs, answer_chain):
        captured["question"] = question
        captured["docs"] = retrieved_docs
        return "Use your school OPT guide."

    monkeypatch.setattr(api, "generate_answer", fake_generate_answer)

    response = client.post("/chat", json={"question": "  What is OPT?  "})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use your school OPT guide."
    assert response.json()["sources"] == ["school-opt-guide.txt"]
    assert captured["question"] == "What is OPT?"
    assert captured["docs"] is docs
