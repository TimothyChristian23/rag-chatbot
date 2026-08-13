# Architecture

This project is a retrieval-augmented generation assistant for international students
researching OPT and related F-1 practical training questions.

## System Flow

1. Student support documents are placed in `data/documents/`.
2. `ingest.py` loads supported files, normalizes source metadata, and chunks the text.
3. `src/retrieval/vectorstore.py` embeds the chunks with OpenAI embeddings and rebuilds the ChromaDB collection.
4. `/chat` loads recent messages for the requested `session_id`.
5. Retrieval uses the current question plus recent student questions so follow-ups can stay grounded.
6. `src/generation/chain.py` builds a guarded prompt from those exact chunks, conversation context, and citation requirements.
7. The API stores the new exchange and returns an answer, source list, highlighted source snippets from the same chunks, history, and education-only disclaimer.
8. The static frontend at `/` calls the API for chat, uploads, session loading, and session clearing.

## Domain Design

The assistant is designed for questions about:

- Post-completion OPT
- Pre-completion OPT
- STEM OPT extensions
- CPT versus OPT distinctions
- EAD and work authorization concepts
- DSO reporting and school process questions
- Form I-765 and Form I-983 guidance, when supported by the ingested documents

The assistant is not designed to make legal decisions, predict case outcomes, or replace a
Designated School Official (DSO) or immigration attorney.

## Safety Approach

- Answers must be grounded in retrieved documents.
- Returned sources come from the same retrieved chunks used to generate the answer.
- Source snippets expose the retrieved text that supported the answer.
- Conversation history is used only to interpret follow-up questions, not as evidence for rules or policies.
- Unsupported questions receive an "I do not have enough information" style response.
- The prompt prohibits invented deadlines, forms, fees, policies, and legal conclusions.
- Every API response includes a disclaimer.
- Uploads are limited to PDF and `.txt` files.

## Portfolio Notes

This repository demonstrates a practical AI engineering pattern:

- Domain-specific RAG
- Local vector search with ChromaDB
- FastAPI service design
- Session-based conversation memory
- Dependency-free static browser frontend
- Prompt guardrails for a higher-stakes use case
- Testable API, ingestion, generation, and vectorstore helpers
- Environment-based model configuration
