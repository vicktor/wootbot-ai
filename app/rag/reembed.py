"""Re-embedding background task.

Re-embeds all documents when the embedding provider/model changes.
"""

import structlog
from sqlalchemy import text

from app.database import get_session, get_engine, Document, set_bot_setting
from app.rag.embeddings import get_embedding_provider, get_embedding_dimensions

logger = structlog.get_logger()

reembed_status: dict = {
    "running": False,
    "total": 0,
    "done": 0,
    "error": None,
}


def get_reembed_status() -> dict:
    return dict(reembed_status)


async def reembed_all(provider: str, model: str, api_key: str) -> None:
    """Re-embed all documents with a new provider/model.

    1. Save new config to bot_settings
    2. Resize vector column if dimensions changed
    3. Re-generate embeddings for all documents
    """
    if reembed_status["running"]:
        raise RuntimeError("Re-embed already in progress")

    reembed_status.update(running=True, total=0, done=0, error=None)

    try:
        # 1. Save new config
        set_bot_setting("embedding_provider", provider)
        set_bot_setting("embedding_model", model)
        if api_key:
            set_bot_setting("embedding_api_key", api_key)

        # 2. Resize vector column if needed
        new_dims = get_embedding_dimensions(provider, model)
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE documents ALTER COLUMN embedding TYPE vector({new_dims})"))
            conn.commit()

        # 3. Fetch all documents
        session = get_session()
        try:
            docs = session.query(Document).all()
            reembed_status["total"] = len(docs)
        finally:
            session.close()

        # 4. Re-embed each document
        embedder = get_embedding_provider()
        for doc in docs:
            try:
                embedding = await embedder.embed(doc.content)
                session = get_session()
                try:
                    session.execute(
                        text("UPDATE documents SET embedding = cast(:emb as vector) WHERE id = :id"),
                        {"emb": "[" + ",".join(str(x) for x in embedding) + "]", "id": doc.id}
                    )
                    session.commit()
                finally:
                    session.close()
                reembed_status["done"] += 1
            except Exception as e:
                logger.error("reembed_chunk_error", doc_id=doc.id, error=str(e))
                reembed_status["done"] += 1

        logger.info("reembed_complete", total=reembed_status["total"], provider=provider, model=model)

    except Exception as e:
        logger.error("reembed_error", error=str(e))
        reembed_status["error"] = str(e)
    finally:
        reembed_status["running"] = False
