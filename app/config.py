from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Chatwoot
    chatwoot_url: str = "https://support.listen.doctor"
    chatwoot_bot_token: str = ""
    chatwoot_account_id: int = 1

    # LLM
    llm_provider: str = "gemini"  # gemini, openai, anthropic
    llm_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    llm_base_url: str | None = None

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/wootbot"

    # Bot behavior
    confidence_threshold: float = 0.65
    max_context_messages: int = 10
    bot_language: str = "es"
    handoff_message: str = "Te transfiero con un agente humano. Un momento por favor."
    greeting_message: str = "¡Hola! Soy el asistente de soporte. ¿En qué puedo ayudarte?"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
