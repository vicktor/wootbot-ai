from functools import lru_cache

from sqlalchemy import text
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector

from app.config import get_settings

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(500), nullable=False)  # URL, filename, etc.
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=True)
    chunk_index = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())


class BotSetting(Base):
    __tablename__ = "bot_settings"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)


class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    feedback = Column(String(20), nullable=True)  # positive, negative, null
    sources_used = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


# #7: Cache engine and session factory to avoid connection leaks
@lru_cache
def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def _get_session_factory():
    return sessionmaker(bind=get_engine())


def get_session() -> Session:
    return _get_session_factory()()


def get_bot_setting(key: str, default: str = "") -> str:
    """Get a bot setting from DB, falling back to default."""
    session = get_session()
    try:
        row = session.query(BotSetting).filter_by(key=key).first()
        return row.value if row else default
    finally:
        session.close()


def set_bot_setting(key: str, value: str):
    """Upsert a bot setting."""
    session = get_session()
    try:
        row = session.query(BotSetting).filter_by(key=key).first()
        if row:
            row.value = value
        else:
            session.add(BotSetting(key=key, value=value))
        session.commit()
    finally:
        session.close()


def init_db():
    """Create tables, enable pgvector extension, and create HNSW index."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents USING hnsw (embedding vector_cosine_ops)
        """))
        conn.commit()
