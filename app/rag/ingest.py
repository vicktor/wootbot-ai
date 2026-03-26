import structlog
import requests
from bs4 import BeautifulSoup

from app.config import get_settings
from app.database import get_session, Document, init_db
from app.llm.provider import get_llm_provider

logger = structlog.get_logger()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


async def ingest_text(content: str, source: str, title: str = None) -> int:
    """Ingest plain text into the knowledge base."""
    settings = get_settings()
    llm = get_llm_provider()
    session = get_session()
    count = 0

    try:
        chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            embedding = await llm.get_embedding(chunk)
            doc = Document(
                source=source,
                title=title or source,
                content=chunk,
                embedding=embedding,
                chunk_index=i,
            )
            session.add(doc)
            count += 1

        session.commit()
        logger.info("ingested_text", source=source, chunks=count)
        return count
    except Exception as e:
        session.rollback()
        logger.error("ingest_text_error", error=str(e))
        raise
    finally:
        session.close()


async def ingest_url(url: str, title: str = None) -> int:
    """Fetch and ingest content from a URL."""
    try:
        response = requests.get(url, timeout=30, headers={
            "User-Agent": "WootBot/1.0 Knowledge Ingester"
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        page_title = title or (soup.title.string if soup.title else url)

        return await ingest_text(text, source=url, title=page_title)
    except Exception as e:
        logger.error("ingest_url_error", url=url, error=str(e))
        raise


async def ingest_pdf(filepath: str, title: str = None) -> int:
    """Ingest a PDF file (requires pdfplumber or pymupdf)."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        return await ingest_text(full_text, source=filepath, title=title or filepath)
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        raise


# CLI entrypoint for ingestion
if __name__ == "__main__":
    import asyncio
    import sys

    init_db()

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m app.rag.ingest url https://docs.example.com 'My Docs'")
        print("  python -m app.rag.ingest text 'Your text here' 'Source name'")
        print("  python -m app.rag.ingest pdf /path/to/file.pdf 'Doc title'")
        sys.exit(1)

    mode = sys.argv[1]
    content = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else None

    async def main():
        if mode == "url":
            count = await ingest_url(content, title)
        elif mode == "text":
            count = await ingest_text(content, source="manual", title=title)
        elif mode == "pdf":
            count = await ingest_pdf(content, title)
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
        print(f"Ingested {count} chunks")

    asyncio.run(main())
