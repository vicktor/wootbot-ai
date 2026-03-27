"""Standalone embedding provider module.

Supports Gemini and OpenAI embedding models independently of the LLM provider
used for generation. Configuration is read from bot_settings (DB) with fallback
to app/config.py defaults.
"""

import asyncio
import structlog
from abc import ABC, abstractmethod

from app.config import get_settings
from app.database import get_bot_setting

logger = structlog.get_logger()

# Registry: provider -> model -> dimensions
EMBEDDING_MODELS: dict[str, dict[str, int]] = {
    "gemini": {
        "gemini-embedding-001": 3072,
    },
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    },
}

# Default model per provider
_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}

# Default provider
_DEFAULT_PROVIDER = "gemini"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------

class GeminiEmbedding(EmbeddingProvider):
    """Embed using google-genai (sync SDK wrapped with asyncio.to_thread)."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model,
            contents=text,
        )
        return result.embeddings[0].values


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIEmbedding(EmbeddingProvider):
    """Embed using the AsyncOpenAI client."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None):
        from openai import AsyncOpenAI
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_embedding_config() -> dict:
    """Return the active embedding configuration.

    Reads from bot_settings (DB) first, falls back to app config / sensible
    defaults.  Keys returned:
      - provider: str
      - model: str
      - api_key: str
      - base_url: str | None
    """
    settings = get_settings()

    provider = get_bot_setting("embedding_provider", "") or settings.llm_provider or _DEFAULT_PROVIDER
    # Normalise to a known provider; unknown values fall back to default
    if provider not in EMBEDDING_MODELS:
        logger.warning("unknown_embedding_provider", provider=provider, fallback=_DEFAULT_PROVIDER)
        provider = _DEFAULT_PROVIDER

    default_model = _DEFAULT_MODELS[provider]
    model = get_bot_setting("embedding_model", "") or default_model
    # Ensure the chosen model is valid for the provider
    if model not in EMBEDDING_MODELS[provider]:
        logger.warning(
            "unknown_embedding_model",
            provider=provider,
            model=model,
            fallback=default_model,
        )
        model = default_model

    api_key = get_bot_setting("embedding_api_key", "") or settings.llm_api_key
    base_url = get_bot_setting("embedding_base_url", "") or None

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }


# ---------------------------------------------------------------------------
# Factory with config-aware caching
# ---------------------------------------------------------------------------

# Module-level cache: (provider, model, api_key, base_url) -> EmbeddingProvider
_provider_cache: dict[tuple, EmbeddingProvider] = {}


def get_embedding_provider() -> EmbeddingProvider:
    """Return a cached EmbeddingProvider instance.

    The cache is keyed on the full config tuple, so it is automatically
    invalidated whenever any setting changes (e.g. after an admin update).
    """
    cfg = get_embedding_config()
    cache_key = (cfg["provider"], cfg["model"], cfg["api_key"], cfg["base_url"])

    if cache_key not in _provider_cache:
        provider_name = cfg["provider"]
        logger.info(
            "creating_embedding_provider",
            provider=provider_name,
            model=cfg["model"],
        )
        if provider_name == "gemini":
            instance: EmbeddingProvider = GeminiEmbedding(
                api_key=cfg["api_key"],
                model=cfg["model"],
            )
        elif provider_name == "openai":
            instance = OpenAIEmbedding(
                api_key=cfg["api_key"],
                model=cfg["model"],
                base_url=cfg["base_url"],
            )
        else:
            # Should never reach here because get_embedding_config normalises
            raise ValueError(f"Unsupported embedding provider: {provider_name}")

        _provider_cache[cache_key] = instance

    return _provider_cache[cache_key]


# ---------------------------------------------------------------------------
# Dimension helper
# ---------------------------------------------------------------------------

def get_embedding_dimensions(provider: str, model: str) -> int:
    """Return the output dimension for a given provider/model pair.

    Raises KeyError if the combination is not in EMBEDDING_MODELS.
    """
    return EMBEDDING_MODELS[provider][model]
