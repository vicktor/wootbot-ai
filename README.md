# WootBot AI - Open Source AI Agent for Chatwoot Community Edition

An open-source AI-powered support bot for Chatwoot that provides RAG-based automatic responses using your knowledge base, documents, and past conversations. Works with Chatwoot Community Edition — no Enterprise license required.

## Features

- **RAG-powered responses** from documents, URLs, and past tickets
- **LLM-agnostic**: Gemini, Claude, OpenAI, or any OpenAI-compatible endpoint
- **Smart handoff** to human agents when confidence is low
- **CSAT tracking** with response quality feedback
- **Multi-language** support (Spanish, Catalan, English)
- **Document ingestion** from PDFs, URLs, and plain text
- **Conversation memory** for context-aware responses
- **pgvector** for semantic search (same DB as Chatwoot)

## Architecture

```
Customer → Chatwoot Widget → Chatwoot Server
                                    ↓ webhook
                              WootBot AI (FastAPI)
                              ├── RAG Search (pgvector)
                              ├── LLM Generation (Gemini/Claude)
                              └── Chatwoot API → Response to customer
```

## Quick Start

```bash
cp .env.example .env
# Edit .env with your config
pip install -r requirements.txt
python -m app.ingest  # Ingest your docs
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

Then register the bot in Chatwoot: Settings → Bots → Add Bot → Webhook URL: `http://localhost:8200/webhook`

## License

MIT
