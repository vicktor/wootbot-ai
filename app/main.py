import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel

from app.config import get_settings
from app.database import init_db, get_session, ConversationLog
from app.chatwoot.client import ChatwootClient
from app.llm.provider import get_llm_provider
from app.rag.search import search_documents, format_context
from app.rag.ingest import ingest_url, ingest_text

logger = structlog.get_logger()
chatwoot = ChatwootClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("wootbot_starting")
    init_db()
    yield
    logger.info("wootbot_stopping")


app = FastAPI(
    title="WootBot AI",
    description="Open-source AI Agent Bot for Chatwoot Community Edition",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Webhook Handler ──────────────────────────────────────────────────

async def process_message(conversation_id: int, message_content: str):
    """Core bot logic: RAG search → LLM generation → response."""
    settings = get_settings()
    llm = get_llm_provider()

    try:
        # 1. Get conversation history for context
        history = await chatwoot.get_messages(conversation_id)

        # 2. Search knowledge base
        documents = await search_documents(message_content, top_k=5)
        context = format_context(documents)

        # 3. Check if we have any relevant docs
        top_similarity = documents[0]["similarity"] if documents else 0.0

        # 4. Generate response with LLM
        result = await llm.generate(
            question=message_content,
            context=context,
            history=history,
        )

        answer = result.get("answer", "")
        confidence = result.get("confidence", "LOW")
        needs_handoff = result.get("needs_handoff", False)
        handoff_reason = result.get("handoff_reason")

        # 5. Confidence check — handoff if too low
        confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.3}
        confidence_score = confidence_map.get(confidence, 0.3)

        detected_lang = result.get("detected_language", "en")
        sources = ", ".join(d.get("title", d["source"])[:50] for d in documents[:3]) if documents else ""

        if needs_handoff or (confidence_score < settings.confidence_threshold and top_similarity < 0.5):
            reason = handoff_reason or f"Low confidence ({confidence}), similarity={top_similarity:.2f}"
            await chatwoot.handoff_to_agent(conversation_id, reason=reason, language=detected_lang)
            logger.info("handoff", conversation_id=conversation_id, reason=reason, lang=detected_lang)
        else:
            # 6. Send AI response
            await chatwoot.send_message(conversation_id, answer)

            # 7. Add private note with metadata for agents
            note = f"🤖 Confidence: {confidence} | Similarity: {top_similarity:.2f} | Sources: {sources}"
            await chatwoot.send_message(conversation_id, note, private=True)

            # 8. Label conversation
            labels = ["bot-handled"]
            if confidence == "HIGH":
                labels.append("bot-confident")
            await chatwoot.set_conversation_labels(conversation_id, labels)

        # 9. Log for analytics
        session = get_session()
        try:
            log = ConversationLog(
                conversation_id=conversation_id,
                question=message_content[:2000],
                answer=answer[:2000],
                confidence=confidence_score,
                sources_used=sources or None,
            )
            session.add(log)
            session.commit()
        finally:
            session.close()

    except Exception as e:
        logger.error("process_message_error", conversation_id=conversation_id, error=str(e))
        try:
            await chatwoot.handoff_to_agent(conversation_id, reason=f"Bot error: {str(e)[:100]}")
        except Exception:
            logger.error("handoff_fallback_error", conversation_id=conversation_id)


@app.post("/webhook")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive webhook events from Chatwoot."""
    payload = await request.json()

    event_type = payload.get("event")

    # Only process incoming customer messages
    if event_type == "message_created":
        message = payload.get("content", "")
        message_type = payload.get("message_type")
        conversation = payload.get("conversation", {})
        conversation_id = conversation.get("id")
        sender = payload.get("sender", {})

        # Skip: outgoing messages (from agents/bot), private notes, empty messages
        if message_type != "incoming" or not message or not conversation_id:
            return {"status": "skipped"}

        # Skip if sender is an agent (not a contact)
        if sender.get("type") == "user":
            return {"status": "skipped_agent"}

        logger.info("incoming_message", conversation_id=conversation_id, message=message[:80])

        # Process in background to return 200 quickly
        background_tasks.add_task(process_message, conversation_id, message)

    elif event_type == "conversation_created":
        conversation = payload.get("conversation", {})
        conversation_id = conversation.get("id")
        if conversation_id:
            settings = get_settings()
            background_tasks.add_task(
                chatwoot.send_message, conversation_id, settings.greeting_message
            )

    return {"status": "ok"}


# ── Admin API ────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
    title: str | None = None


class IngestTextRequest(BaseModel):
    content: str
    source: str
    title: str | None = None


@app.post("/admin/ingest/url")
async def admin_ingest_url(req: IngestURLRequest):
    """Ingest a URL into the knowledge base."""
    count = await ingest_url(req.url, req.title)
    return {"status": "ok", "chunks_ingested": count}


@app.post("/admin/ingest/text")
async def admin_ingest_text(req: IngestTextRequest):
    """Ingest plain text into the knowledge base."""
    count = await ingest_text(req.content, req.source, req.title)
    return {"status": "ok", "chunks_ingested": count}


@app.get("/admin/stats")
async def admin_stats():
    """Get bot statistics."""
    session = get_session()
    try:
        from sqlalchemy import func, text
        total_logs = session.query(ConversationLog).count()
        avg_confidence = session.query(func.avg(ConversationLog.confidence)).scalar()
        handoff_count = session.query(ConversationLog).filter(
            ConversationLog.confidence < get_settings().confidence_threshold
        ).count()

        from app.database import Document
        total_docs = session.query(Document).count()

        return {
            "total_conversations_handled": total_logs,
            "average_confidence": round(float(avg_confidence or 0), 3),
            "handoff_count": handoff_count,
            "knowledge_base_chunks": total_docs,
        }
    finally:
        session.close()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "wootbot-ai"}
