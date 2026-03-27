import hmac
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException

from app.config import get_settings
from app.database import init_db, get_session, ConversationLog, get_bot_setting
from app.chatwoot.client import ChatwootClient
from app.llm.provider import get_llm_provider
from app.rag.search import search_documents, format_context
from app.admin.router import router as admin_router, public_router as admin_public_router

logger = structlog.get_logger()
chatwoot = ChatwootClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # #3: Warn loudly if secrets are not configured
    settings = get_settings()
    if not settings.webhook_secret:
        logger.critical("WEBHOOK_SECRET is not set — webhook endpoint is unauthenticated!")
    if not settings.admin_secret:
        logger.critical("ADMIN_SECRET is not set — admin panel is unauthenticated!")
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

async def process_message(conversation_id: int, message_content: str, contact_info: dict = None, channel: str = None):
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
            contact_info=contact_info,
            channel=channel,
        )

        response_text = result.get("response", "")
        confidence = result.get("confidence", "LOW")
        reasoning = result.get("reasoning", "")
        detected_lang = result.get("detected_language", "en")
        sources = ", ".join(d.get("title", d["source"])[:50] for d in documents[:3]) if documents else ""

        # 5. Confidence check — handoff if needed
        confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.3}
        confidence_score = confidence_map.get(confidence, 0.3)

        is_email = channel and "email" in channel.lower()

        if is_email:
            # Email: be conservative — skip if similarity OR confidence is low
            is_handoff = (
                response_text == "conversation_handoff"
                or top_similarity < 0.5
                or confidence_score < settings.confidence_threshold
            )
        else:
            # Chat: handoff only when both confidence AND similarity are low
            is_handoff = (
                response_text == "conversation_handoff"
                or (confidence_score < settings.confidence_threshold and top_similarity < 0.5)
            )

        if is_handoff:
            reason = reasoning or f"Low confidence ({confidence}), similarity={top_similarity:.2f}"
            if is_email:
                await chatwoot.silent_handoff(conversation_id, reason=reason)
                logger.info("email_silent_handoff", conversation_id=conversation_id, reason=reason)
            else:
                await chatwoot.handoff_to_agent(conversation_id, reason=reason, language=detected_lang)
                logger.info("handoff", conversation_id=conversation_id, reason=reason, lang=detected_lang)
        else:
            # 6. Format email with greeting/closing if email channel
            if is_email:
                greeting = get_bot_setting("email_greeting", settings.email_greeting)
                closing = get_bot_setting("email_closing", settings.email_closing)
                closing = closing.replace("\\n", "\n") if closing else ""
                if greeting or closing:
                    translated_greeting = await llm.translate(greeting, detected_lang) if greeting else ""
                    translated_closing = await llm.translate(closing, detected_lang) if closing else ""
                    translated_closing = translated_closing.replace("\n", "\n\n") if translated_closing else ""
                    parts = []
                    if translated_greeting:
                        parts.append(translated_greeting)
                    parts.append(response_text)
                    if translated_closing:
                        parts.append(translated_closing)
                    response_text = "\n\n".join(parts)

            # 7. Send AI response
            await chatwoot.send_message(conversation_id, response_text)

            # 8. Add private note with metadata for agents
            note = f"Confidence: {confidence} | Similarity: {top_similarity:.2f} | Sources: {sources}"
            if reasoning:
                note += f"\nReasoning: {reasoning}"
            await chatwoot.send_message(conversation_id, note, private=True)

            # 9. Label conversation
            labels = ["bot-handled"]
            if confidence == "HIGH":
                labels.append("bot-confident")
            await chatwoot.set_conversation_labels(conversation_id, labels)

        # 10. Log for analytics
        session = get_session()
        try:
            log = ConversationLog(
                conversation_id=conversation_id,
                question=message_content[:2000],
                answer=response_text[:2000],
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
            # #8: Generic error message — don't leak internal details
            await chatwoot.handoff_to_agent(conversation_id, reason="Bot encountered an internal error")
        except Exception:
            logger.error("handoff_fallback_error", conversation_id=conversation_id)


@app.post("/webhook")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive webhook events from Chatwoot."""
    settings = get_settings()

    # #3 + #5: Validate webhook secret with constant-time comparison
    if settings.webhook_secret:
        token = request.headers.get("X-Webhook-Secret") or request.query_params.get("secret") or ""
        if not hmac.compare_digest(token, settings.webhook_secret):
            logger.warning("webhook_unauthorized", ip=request.client.host)
            raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    event_type = payload.get("event")

    if event_type == "message_created":
        message = payload.get("content", "")
        message_type = payload.get("message_type")
        conversation = payload.get("conversation", {})
        conversation_id = conversation.get("id")
        sender = payload.get("sender", {})

        # #21: Validate conversation_id type
        if not isinstance(conversation_id, int) or conversation_id <= 0:
            return {"status": "skipped"}

        if message_type != "incoming" or not message:
            return {"status": "skipped"}

        if sender.get("type") == "user":
            return {"status": "skipped_agent"}

        channel = conversation.get("channel")
        logger.info("incoming_message", conversation_id=conversation_id, channel=channel, message=message[:80])

        contact_info = {
            "name": sender.get("name"),
            "email": sender.get("email"),
            "phone": sender.get("phone_number"),
        }

        background_tasks.add_task(process_message, conversation_id, message, contact_info, channel)

    elif event_type == "conversation_created":
        conversation = payload.get("conversation", {})
        conversation_id = conversation.get("id")
        if isinstance(conversation_id, int) and conversation_id > 0:
            background_tasks.add_task(
                chatwoot.send_message, conversation_id, settings.greeting_message
            )

    return {"status": "ok"}


# ── Admin Panel (embeddable as Chatwoot Dashboard App) ───────────────
app.include_router(admin_public_router)
app.include_router(admin_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "wootbot-ai"}
