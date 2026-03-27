from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Chatwoot
    chatwoot_url: str = ""
    chatwoot_bot_token: str = ""
    chatwoot_account_id: int = 1

    # Security
    webhook_secret: str = ""
    admin_secret: str = ""
    allowed_origins: str = ""  # comma-separated origins for iframe CSP

    # LLM
    llm_provider: str = "gemini"  # gemini, openai, anthropic
    llm_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    llm_base_url: str | None = None

    # Bot identity (#16: configurable, not hardcoded)
    company_name: str = "Your Company"
    assistant_name: str = "Support Assistant"

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/wootbot"

    # Bot behavior
    confidence_threshold: float = 0.65
    max_context_messages: int = 10
    handoff_message: str = "I'm transferring you to a human agent. One moment please. / Te transfiero con un agente humano. Un momento por favor."
    greeting_message: str = "Hello! I'm the support assistant. How can I help you? / Hola! Soy el asistente de soporte. En que puedo ayudarte?"

    # Email formatting
    email_greeting: str = "Hola, gracias por contactar con nosotros."
    email_closing: str = "Esperamos haber sido de ayuda. Si necesitas cualquier otra cosa, estaremos encantados de ayudarte.\nUn cordial saludo,\nEquipo de Soporte"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
