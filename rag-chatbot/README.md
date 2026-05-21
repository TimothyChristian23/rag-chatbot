# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions over your own documents.

## Architecture

```
Documents (PDF/TXT)
      ↓
  [Ingestion]  →  Chunk + Embed  →  ChromaDB (Vector Store)
                                          ↓
User Query  →  [Retrieval]  →  Top-K chunks
                                          ↓
                          [Generation]  →  LLM  →  Answer + Sources
```

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add your documents
Drop PDFs or `.txt` files into `data/documents/`.

### 4. Ingest documents
```bash
python ingest.py
```

### 5a. Chat via CLI
```bash
python chat_cli.py
```

### 5b. Start the API
```bash
uvicorn src.api.main:app --reload
# Visit http://localhost:8000/docs for the interactive API explorer
```

## Project Structure
```
rag-chatbot/
├── data/
│   └── documents/          ← drop your PDFs/TXTs here
├── src/
│   ├── ingestion/
│   │   └── loader.py       ← document loading & chunking
│   ├── retrieval/
│   │   └── vectorstore.py  ← ChromaDB embedding & search
│   ├── generation/
│   │   └── chain.py        ← LangChain RAG chain
│   └── api/
│       └── main.py         ← FastAPI endpoints
├── tests/
├── ingest.py               ← one-time ingestion script
├── chat_cli.py             ← CLI for quick testing
├── requirements.txt
└── .env.example
```

## Next Steps (Week 4–6)
- [ ] Add conversation memory (store chat history per session)
- [ ] Build a Streamlit or React frontend
- [ ] Add source citation highlighting in the UI
- [ ] Dockerize the app
- [ ] Deploy to Railway or Hugging Face Spaces
