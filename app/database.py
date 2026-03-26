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


def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url)


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db():
    """Create tables and enable pgvector extension."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
