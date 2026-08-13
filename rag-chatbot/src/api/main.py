"""
FastAPI backend for the OPT assistant. It exposes three endpoints:
  POST /ingest   -> upload a document and build/update the vector store
  POST /chat     -> answer a question using the RAG chain
  GET  /health   -> check server status
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.conversation.memory import (
    ChatMessage,
    ConversationMemory,
    build_retrieval_query,
    normalize_session_id,
)
from src.generation.chain import (
    LEGAL_DISCLAIMER,
    build_answer_chain,
    collect_source_snippets,
    collect_sources,
    generate_answer,
)
from src.ingestion.loader import DOCUMENTS_DIR, load_documents, split_documents
from src.retrieval.vectorstore import CHROMA_PERSIST_DIR, build_vectorstore, load_vectorstore, retrieve

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="International Student OPT RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Module-level state for a simple single-process demo app.
_answer_chain = None
_vectorstore = None
_memory = ConversationMemory()


class ChatMessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SourceSnippetResponse(BaseModel):
    source: str
    page: str | int
    snippet: str
    rank: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Trim questions and reject whitespace-only input."""
        question = value.strip()
        if not question:
            raise ValueError("Question cannot be blank.")
        return question

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session(cls, value: str | None) -> str:
        """Trim session IDs and use a stable default for blank values."""
        return normalize_session_id(value)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    source_snippets: list[SourceSnippetResponse]
    disclaimer: str = LEGAL_DISCLAIMER
    history: list[ChatMessageResponse]


class ChatHistoryResponse(BaseModel):
    session_id: str
    history: list[ChatMessageResponse]


def _get_answer_chain():
    """Lazily create the answer chain so tests can import the app without an API key."""
    global _answer_chain
    if _answer_chain is None:
        _answer_chain = build_answer_chain()
    return _answer_chain


def _get_vectorstore():
    """Lazily load the vector store and reuse it for this process."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def _serialize_history(history: list[ChatMessage]) -> list[ChatMessageResponse]:
    """Convert stored dataclasses into API response models."""
    return [
        ChatMessageResponse(role=message.role, content=message.content)
        for message in history
    ]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def frontend():
    """Serve the browser chat UI."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend files were not found.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF or .txt file and add it to the vector store."""
    filename = Path(file.filename or "uploaded_document").name
    if Path(filename).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and .txt files are supported.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCUMENTS_DIR / filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_documents(dest.parent)
    chunks = split_documents(docs)
    global _answer_chain, _vectorstore
    _vectorstore = build_vectorstore(chunks)
    _answer_chain = build_answer_chain()
    return {"message": f"Ingested '{filename}' successfully", "chunks": len(chunks)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using the RAG pipeline."""
    try:
        vectorstore = _get_vectorstore()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    history = _memory.get_history(request.session_id)
    retrieval_query = build_retrieval_query(request.question, history)
    docs = retrieve(retrieval_query, vectorstore)
    answer = generate_answer(
        request.question,
        docs,
        _get_answer_chain(),
        chat_history=history,
    )
    sources = collect_sources(docs)
    source_snippets = collect_source_snippets(docs)
    updated_history = _memory.add_exchange(request.session_id, request.question, answer)
    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        sources=sources,
        source_snippets=source_snippets,
        disclaimer=LEGAL_DISCLAIMER,
        history=_serialize_history(updated_history),
    )


@app.get("/chat/sessions/{session_id}", response_model=ChatHistoryResponse)
async def chat_history(session_id: str):
    """Return the stored history for one chat session."""
    normalized_session_id = normalize_session_id(session_id)
    return ChatHistoryResponse(
        session_id=normalized_session_id,
        history=_serialize_history(_memory.get_history(normalized_session_id)),
    )


@app.delete("/chat/sessions/{session_id}", response_model=ChatHistoryResponse)
async def clear_chat_history(session_id: str):
    """Clear the stored history for one chat session."""
    normalized_session_id = normalize_session_id(session_id)
    _memory.clear_session(normalized_session_id)
    return ChatHistoryResponse(session_id=normalized_session_id, history=[])


@app.get("/health")
async def health():
    vectorstore_available = Path(CHROMA_PERSIST_DIR).exists()
    return {
        "status": "ok",
        "chain_loaded": _answer_chain is not None,
        "vectorstore_loaded": _vectorstore is not None,
        "vectorstore_available": vectorstore_available,
        "memory_sessions": _memory.session_count(),
    }
