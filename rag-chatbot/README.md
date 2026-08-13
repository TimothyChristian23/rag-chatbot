# International Student OPT RAG Assistant

A retrieval-augmented generation chatbot that helps international students understand
OPT, STEM OPT, CPT, and related F-1 practical training topics using trusted source
documents.

This project is built as a portfolio-ready AI application: it combines document
ingestion, vector search, a guarded generation prompt, FastAPI endpoints, and tests.

> This assistant is for general education only and is not legal advice. Students should
> confirm their situation with their DSO or a qualified immigration attorney.

## Why This Project

International students often need quick, readable answers about practical training, but
OPT rules depend on official guidance, school processes, and individual student records.
This app uses RAG so answers can be grounded in documents instead of relying only on the
model's memory.

Good source candidates include:

- [USCIS Practical Training guidance](https://www.uscis.gov/node/92821)
- [ICE SEVIS Practical Training page](https://www.ice.gov/sevis/practical-training)
- [Study in the States Form I-983 overview](https://studyinthestates.dhs.gov/form-i-983-overview)
- School international office handbooks, OPT checklists, and policy PDFs

## Features

- PDF and `.txt` ingestion
- Chunking and source metadata normalization
- ChromaDB vector storage
- OpenAI embeddings and chat model integration
- FastAPI `/ingest`, `/chat`, and `/health` endpoints
- Session-based conversation memory for follow-up questions
- Browser chat frontend served by FastAPI
- Domain-specific prompt guardrails for OPT questions
- Source list from the same retrieved chunks used for generation
- Highlighted retrieved source snippets for answer traceability
- Disclaimer in every chat response
- CLI for local testing
- Pytest coverage for API behavior, generation helpers, ingestion, and vectorstore helpers

## Architecture

```text
Documents
   |
   v
Ingestion -> Chunking -> Embeddings -> ChromaDB
                                      |
Student question -> Single Retrieval -+
                                      |
                                      v
                            Guarded RAG Prompt
                                      |
                                      v
                              Answer + Sources
```

See [docs/architecture.md](docs/architecture.md) for the detailed design.

## Tech Stack

- Python
- FastAPI
- LangChain
- OpenAI
- ChromaDB
- Pytest

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_key_here
```

### 4. Add documents

Place official guidance PDFs, school OPT checklists, or `.txt` files in:

```text
data/documents/
```

### 5. Build the vector store

```bash
python ingest.py
```

### 6. Run the API

```bash
uvicorn src.api.main:app --reload
```

Open the API docs at:

```text
http://localhost:8000/docs
```

Open the browser chat UI at:

```text
http://localhost:8000
```

## API Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"demo-student\",\"question\":\"When can I apply for post-completion OPT?\"}"
```

Example response shape:

```json
{
  "session_id": "demo-student",
  "answer": "Based on the retrieved context...",
  "sources": ["uscis-practical-training.txt"],
  "source_snippets": [
    {
      "source": "uscis-practical-training.txt",
      "page": 1,
      "snippet": "Students may apply for post-completion OPT...",
      "rank": 1
    }
  ],
  "disclaimer": "This information is for general education only and is not legal advice. Students should confirm their situation with their DSO or a qualified immigration attorney.",
  "history": [
    {"role": "user", "content": "When can I apply for post-completion OPT?"},
    {"role": "assistant", "content": "Based on the retrieved context..."}
  ]
}
```

Session history can be fetched or cleared with:

```bash
curl http://localhost:8000/chat/sessions/demo-student
curl -X DELETE http://localhost:8000/chat/sessions/demo-student
```

## CLI Usage

```bash
python chat_cli.py
```

## Tests

```bash
pytest
```

The tests avoid live model calls, so they can run without an OpenAI API key.

## Project Structure

```text
rag-chatbot/
|-- data/
|   `-- documents/
|-- docs/
|   `-- architecture.md
|-- src/
|   |-- api/
|   |   `-- main.py
|   |-- generation/
|   |   `-- chain.py
|   |-- conversation/
|   |   `-- memory.py
|   |-- ingestion/
|   |   `-- loader.py
|   `-- retrieval/
|       `-- vectorstore.py
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|-- tests/
|-- chat_cli.py
|-- ingest.py
|-- requirements.txt
`-- README.md
```

## Roadmap

- Add Docker support
- Add CI with GitHub Actions
- Add a curated official-source ingestion script
- Deploy a demo API or web app
