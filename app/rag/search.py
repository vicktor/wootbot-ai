import structlog
from sqlalchemy import text
from app.database import get_session, Document
from app.llm.provider import get_llm_provider

logger = structlog.get_logger()


async def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search knowledge base using semantic similarity with pgvector."""
    llm = get_llm_provider()

    try:
        query_embedding = await llm.get_embedding(query)
    except Exception as e:
        logger.error("embedding_error", error=str(e))
        return []

    session = get_session()
    try:
        # pgvector cosine distance search
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        results = session.execute(
            text("""
                SELECT id, source, title, content,
                       1 - (embedding <=> cast(:embedding as vector)) as similarity
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> cast(:embedding as vector)
                LIMIT :top_k
            """),
            {"embedding": embedding_str, "top_k": top_k},
        ).fetchall()

        docs = []
        for row in results:
            docs.append({
                "id": row[0],
                "source": row[1],
                "title": row[2],
                "content": row[3],
                "similarity": float(row[4]),
            })

        logger.info("rag_search", query=query[:80], results=len(docs),
                     top_similarity=docs[0]["similarity"] if docs else 0)
        return docs

    except Exception as e:
        logger.error("search_error", error=str(e))
        return []
    finally:
        session.close()


def format_context(documents: list[dict]) -> str:
    """Format retrieved documents into context string for LLM."""
    if not documents:
        return "No relevant documents found in the knowledge base."

    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.get("title") or doc.get("source", "Unknown")
        parts.append(f"[Source {i}: {source}]\n{doc['content']}\n")

    return "\n".join(parts)
