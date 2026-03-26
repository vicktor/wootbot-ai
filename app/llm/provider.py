import json
import structlog
from abc import ABC, abstractmethod
from app.config import get_settings

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a helpful customer support assistant for {company}.
You answer questions based ONLY on the provided context from our knowledge base.

Rules:
- Answer in the same language the customer writes in.
- If the context doesn't contain enough information to answer confidently, say so honestly and set confidence to LOW.
- Be concise and helpful. Don't repeat the question back.
- If the customer seems frustrated or the issue is complex, recommend handoff to a human agent.
- Never invent information not present in the context.

Respond in this JSON format:
{{
    "answer": "your response to the customer",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "needs_handoff": false,
    "handoff_reason": null
}}
"""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, question: str, context: str, history: list[dict] = None) -> dict:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        settings = get_settings()
        genai.configure(api_key=settings.llm_api_key)
        self.model = genai.GenerativeModel(settings.llm_model)
        self.embed_model = "models/text-embedding-004"

    async def generate(self, question: str, context: str, history: list[dict] = None) -> dict:
        settings = get_settings()
        system = SYSTEM_PROMPT.format(company="Listen.Doctor")

        history_text = ""
        if history:
            for msg in history[-settings.max_context_messages:]:
                role = "Customer" if msg["role"] == "user" else "Agent"
                history_text += f"{role}: {msg['content']}\n"

        prompt = f"""{system}

Context from knowledge base:
---
{context}
---

Conversation history:
{history_text}

Customer question: {question}

Respond with valid JSON only."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Clean markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("llm_generation_error", error=str(e))
            return {
                "answer": "Lo siento, ha ocurrido un error procesando tu consulta. Un agente te atenderá en breve.",
                "confidence": "LOW",
                "needs_handoff": True,
                "handoff_reason": "LLM error",
            }

    async def get_embedding(self, text: str) -> list[float]:
        import google.generativeai as genai
        result = genai.embed_content(model=self.embed_model, content=text)
        return result["embedding"]


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI
        settings = get_settings()
        kwargs = {"api_key": settings.llm_api_key}
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = settings.llm_model

    async def generate(self, question: str, context: str, history: list[dict] = None) -> dict:
        settings = get_settings()
        system = SYSTEM_PROMPT.format(company="Listen.Doctor")

        messages = [{"role": "system", "content": system}]

        if history:
            for msg in history[-settings.max_context_messages:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nRespond with valid JSON only.",
        })

        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            logger.error("llm_generation_error", error=str(e))
            return {
                "answer": "Lo siento, ha ocurrido un error. Un agente te atenderá en breve.",
                "confidence": "LOW",
                "needs_handoff": True,
                "handoff_reason": "LLM error",
            }

    async def get_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding


class AnthropicProvider(LLMProvider):
    def __init__(self):
        import anthropic
        settings = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
        self.model = settings.llm_model or "claude-sonnet-4-20250514"
        # Use OpenAI for embeddings since Anthropic doesn't have an embedding API
        from openai import AsyncOpenAI
        self._openai = AsyncOpenAI(api_key=settings.llm_api_key)

    async def generate(self, question: str, context: str, history: list[dict] = None) -> dict:
        settings = get_settings()
        system = SYSTEM_PROMPT.format(company="Listen.Doctor")

        messages = []
        if history:
            for msg in history[-settings.max_context_messages:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nRespond with valid JSON only.",
        })

        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=1024, system=system, messages=messages
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            logger.error("llm_generation_error", error=str(e))
            return {
                "answer": "Lo siento, ha ocurrido un error. Un agente te atenderá en breve.",
                "confidence": "LOW",
                "needs_handoff": True,
                "handoff_reason": "LLM error",
            }

    async def get_embedding(self, text: str) -> list[float]:
        # Anthropic doesn't provide embeddings, fall back to Gemini
        import google.generativeai as genai
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        return result["embedding"]


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    provider_class = providers.get(settings.llm_provider)
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
    return provider_class()
