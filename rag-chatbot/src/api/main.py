"""
FastAPI backend — exposes two endpoints:
  POST /ingest   -> upload a document and build/update the vector store
  POST /chat     -> answer a question using the RAG chain
  GET  /health   -> check server status
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from pathlib import Path

from src.ingestion.loader import load_documents, split_documents
from src.retrieval.vectorstore import build_vectorstore, load_vectorstore
from src.generation.chain import build_rag_chain

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level chain (swap for proper state management in production)
_chain = None


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF or .txt file and add it to the vector store."""
    dest = Path("data/documents") / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_documents(dest.parent)
    chunks = split_documents(docs)
    build_vectorstore(chunks)

    global _chain
    _chain = build_rag_chain(load_vectorstore())
    return {"message": f"Ingested '{file.filename}' successfully", "chunks": len(chunks)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using the RAG pipeline."""
    global _chain
    if _chain is None:
        try:
            _chain = build_rag_chain(load_vectorstore())
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    answer = _chain.invoke(request.question)

    vs = load_vectorstore()
    docs = vs.similarity_search(request.question, k=3)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok", "chain_loaded": _chain is not None}
