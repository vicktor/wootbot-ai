# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this

WootBot AI — an open-source AI support bot for Chatwoot Community Edition. Receives webhooks from Chatwoot, searches a knowledge base via RAG (pgvector), generates responses with an LLM, and replies or hands off to a human agent.

## Commands

```bash
# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8200

# Production (matches Dockerfile and systemd service)
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8200 app.main:app

# Install dependencies (uses uv)
uv pip install -e .

# Ingest documents via CLI
python -m app.rag.ingest url <URL> "<Title>"
python -m app.rag.ingest file <path> "<Title>"
python -m app.rag.ingest tickets <count>

# Health check
curl http://127.0.0.1:8200/health
```

No test suite exists yet.

## Architecture

```
Chatwoot webhook → POST /webhook (app/main.py)
                      ↓
              process_message() [BackgroundTask]
              ├── get conversation history (Chatwoot API)
              ├── search_documents() → pgvector cosine similarity (app/rag/search.py)
              ├── LLM generate with context + history (app/llm/provider.py)
              ├── confidence check → handoff or respond
              └── log to conversation_logs table
```

### Key modules

- **app/main.py** — FastAPI app, webhook handler, orchestrates the RAG→LLM→response flow
- **app/llm/provider.py** — Abstract `LLMProvider` with Gemini, OpenAI, Anthropic implementations. All providers use `gemini-embedding-001` for embeddings (3072 dimensions)
- **app/rag/search.py** — Semantic search via pgvector cosine distance, returns top-k documents
- **app/rag/ingest.py** — Document ingestion: URL, PDF, DOCX, MD, TXT, CSV, resolved Chatwoot tickets. Chunks text (500 words, 50 overlap) and embeds
- **app/chatwoot/client.py** — HTTP client for Chatwoot API (send messages, handoff, get history, labels)
- **app/admin/router.py** — Admin panel with HTML UI + API endpoints for knowledge base management and analytics
- **app/database.py** — SQLAlchemy models: `Document` (with pgvector Vector(3072) column) and `ConversationLog`
- **app/config.py** — Pydantic Settings loaded from `.env`

### Embedding design decision

All three LLM providers use Google's `gemini-embedding-001` for embeddings regardless of the generation provider. This keeps vectors in a single semantic space but means the Google API key is always required for embeddings, even when using OpenAI or Anthropic for generation. The `AnthropicProvider` and `OpenAIProvider` embedding methods have coupling issues with `llm_api_key` — see provider.py lines 182-215.

### LLM response format

The LLM returns JSON: `{"reasoning", "response", "confidence", "detected_language"}`. Confidence below `CONFIDENCE_THRESHOLD` (default 0.65) triggers handoff to a human agent.

### Database

PostgreSQL with pgvector extension. Two tables: `documents` (knowledge base chunks + embeddings) and `conversation_logs` (metrics/feedback). Connection via `DATABASE_URL` env var.

## Environment variables

Configured via `.env` (see `.env.example`). Key vars: `CHATWOOT_URL`, `CHATWOOT_BOT_TOKEN`, `LLM_PROVIDER` (gemini/openai/anthropic), `LLM_API_KEY`, `LLM_MODEL`, `DATABASE_URL`, `CONFIDENCE_THRESHOLD`.

## Language

The bot auto-detects customer language and responds accordingly. Default is Spanish (`BOT_LANGUAGE=es`). Handoff messages are multilingual (es, ca, en, pt, fr, de).
