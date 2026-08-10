"""
FastAPI backend for the OPT assistant. It exposes three endpoints:
  POST /ingest   -> upload a document and build/update the vector store
  POST /chat     -> answer a question using the RAG chain
  GET  /health   -> check server status
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.generation.chain import (
    LEGAL_DISCLAIMER,
    build_answer_chain,
    collect_sources,
    generate_answer,
)
from src.ingestion.loader import DOCUMENTS_DIR, load_documents, split_documents
from src.retrieval.vectorstore import build_vectorstore, load_vectorstore, retrieve

app = FastAPI(title="International Student OPT RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level state for a simple single-process demo app.
_answer_chain = None
_vectorstore = None


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


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    disclaimer: str = LEGAL_DISCLAIMER


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

    docs = retrieve(request.question, vectorstore)
    answer = generate_answer(request.question, docs, _get_answer_chain())
    sources = collect_sources(docs)
    return ChatResponse(answer=answer, sources=sources, disclaimer=LEGAL_DISCLAIMER)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chain_loaded": _answer_chain is not None,
        "vectorstore_loaded": _vectorstore is not None,
    }
