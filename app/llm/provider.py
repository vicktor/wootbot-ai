import json
import asyncio
import structlog
from abc import ABC, abstractmethod
from functools import lru_cache
from app.config import get_settings

logger = structlog.get_logger()

SYSTEM_PROMPT = """[Identity]
Your name is {assistant_name}, a helpful, friendly, and knowledgeable support assistant for {company}.
You will not answer anything about other products or events outside of {company}.

{contact_context}[Response Guidelines]
- Always detect the customer's language and reply in that SAME language. Never switch languages unless the customer does.
- Use natural, polite conversational language that is clear and easy to follow. Use short sentences and simple words.
- Be concise: respond in 1 to 3 sentences maximum, unless the customer asks for more detail.
- Do NOT use lists, markdown, bullet points, headers, or any formatting not typically spoken in a chat.
- Do NOT rush giving a response. If there are multiple steps, provide only ONE step at a time. Ask the customer to confirm they completed it before continuing with the next step.
- Do NOT use your own training data or assumptions. Base responses strictly on the provided context.
- When there is ambiguity, ask a clarifying question rather than making assumptions.
- Do NOT try to end the conversation (avoid phrases like "Is there anything else I can help with?" or "Talk soon!").
- Keep the conversation flowing naturally. Ask relevant follow-up questions when appropriate.
- Do NOT repeat the customer's question back to them.
- If the answer is not in the provided context, tell the customer you'll connect them with a support agent.
- Include a brief reasoning of how you arrived at the answer.
{custom_instructions}
[Task]
When the customer asks a question, use the provided context from the knowledge base to give a helpful response following the steps below:
1. Check if the context contains relevant information to answer the question.
2. If yes, provide the answer step by step (one step at a time), in plain conversational text.
3. If the context does not contain the answer, set response to "conversation_handoff".

[Output Format]
Always respond with valid JSON only, no other text:
{{
    "reasoning": "Brief explanation of why you chose this response based on the context",
    "response": "Your conversational response in the customer's language, or 'conversation_handoff' if you cannot answer",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "detected_language": "es" or "en" or "ca" or other ISO code
}}
"""


def build_contact_context(contact_info: dict = None) -> str:
    """Build contact context string from contact info."""
    if not contact_info:
        return ""
    lines = []
    if contact_info.get("name"):
        lines.append(f"- Name: {contact_info['name']}")
    if contact_info.get("email"):
        lines.append(f"- Email: {contact_info['email']}")
    if contact_info.get("phone"):
        lines.append(f"- Phone: {contact_info['phone']}")
    if not lines:
        return ""
    return "[Contact Information]\n" + "\n".join(lines) + "\n\n"


TRANSLATE_PROMPT = """Translate the following text to {lang}. Return ONLY the translated text, nothing else. Preserve line breaks exactly.

{text}"""


def _build_prompt(question: str, context: str, history: list[dict] = None, contact_info: dict = None, channel: str = None) -> str:
    """Build the full prompt with system message, context, history and question."""
    settings = get_settings()

    # #16: Use configurable company/assistant names
    system = SYSTEM_PROMPT.format(
        assistant_name=settings.assistant_name,
        company=settings.company_name,
        contact_context=build_contact_context(contact_info),
        custom_instructions="",
    )

    history_text = ""
    if history:
        for msg in history[-(settings.max_context_messages):]:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    return f"""{system}

[Knowledge Base Context]
{context}

[Conversation History]
{history_text}

[Customer Message]
{question}

Respond with valid JSON only."""


def _parse_llm_response(text: str) -> dict:
    """Parse LLM response, handling markdown fences and old format compatibility."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
        # Normalize old format to new format
        if "answer" in result and "response" not in result:
            result["response"] = result.pop("answer")
        if "needs_handoff" in result:
            if result.get("needs_handoff"):
                result["response"] = "conversation_handoff"
            del result["needs_handoff"]
        if "handoff_reason" in result and "reasoning" not in result:
            result["reasoning"] = result.get("handoff_reason", "")
        # Ensure all required fields exist
        result.setdefault("reasoning", "")
        result.setdefault("response", "conversation_handoff")
        result.setdefault("confidence", "LOW")
        result.setdefault("detected_language", "en")
        return result
    except json.JSONDecodeError as e:
        logger.error("json_parse_error", error=str(e), raw=text[:200])
        return ERROR_RESPONSE


ERROR_RESPONSE = {
    "reasoning": "LLM error",
    "response": "conversation_handoff",
    "confidence": "LOW",
    "detected_language": "en",
}


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, question: str, context: str, history: list[dict] = None, contact_info: dict = None, channel: str = None) -> dict:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def translate(self, text: str, target_lang: str) -> str:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai
        settings = get_settings()
        self.client = genai.Client(api_key=settings.llm_api_key)
        self.model_name = settings.llm_model
        self.embed_model = "gemini-embedding-001"

    # #6: Wrap sync google-genai calls with asyncio.to_thread
    async def generate(self, question: str, context: str, history: list[dict] = None, contact_info: dict = None, channel: str = None) -> dict:
        prompt = _build_prompt(question, context, history, contact_info, channel)
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name, contents=prompt
            )
            return _parse_llm_response(response.text)
        except Exception as e:
            logger.error("llm_generation_error", error=str(e))
            return ERROR_RESPONSE

    async def get_embedding(self, text: str) -> list[float]:
        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.embed_model, contents=text
        )
        return result.embeddings[0].values

    async def translate(self, text: str, target_lang: str) -> str:
        prompt = TRANSLATE_PROMPT.format(lang=target_lang, text=text)
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error("translate_error", error=str(e))
            return text


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI
        settings = get_settings()
        kwargs = {"api_key": settings.llm_api_key}
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = settings.llm_model

    async def generate(self, question: str, context: str, history: list[dict] = None, contact_info: dict = None, channel: str = None) -> dict:
        prompt = _build_prompt(question, context, history, contact_info, channel)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3
            )
            return _parse_llm_response(response.choices[0].message.content)
        except Exception as e:
            logger.error("llm_generation_error", error=str(e))
            return ERROR_RESPONSE

    # #11: Use OpenAI's own embedding model, not gemini
    async def get_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-large", input=text
        )
        return response.data[0].embedding

    async def translate(self, text: str, target_lang: str) -> str:
        prompt = TRANSLATE_PROMPT.format(lang=target_lang, text=text)
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("translate_error", error=str(e))
            return text


class AnthropicProvider(LLMProvider):
    def __init__(self):
        import anthropic
        from google import genai
        settings = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
        self.model = settings.llm_model or "claude-sonnet-4-20250514"
        # #12: Embeddings need a Google API key — use embedding_api_key or warn
        embedding_key = settings.llm_base_url or settings.llm_api_key
        self._genai_client = genai.Client(api_key=embedding_key)

    async def generate(self, question: str, context: str, history: list[dict] = None, contact_info: dict = None, channel: str = None) -> dict:
        prompt = _build_prompt(question, context, history, contact_info, channel)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=1024, messages=messages
            )
            return _parse_llm_response(response.content[0].text)
        except Exception as e:
            logger.error("llm_generation_error", error=str(e))
            return ERROR_RESPONSE

    # #6 + #12: Wrap sync genai call, reuse client from __init__
    async def get_embedding(self, text: str) -> list[float]:
        result = await asyncio.to_thread(
            self._genai_client.models.embed_content,
            model="gemini-embedding-001", contents=text
        )
        return result.embeddings[0].values

    async def translate(self, text: str, target_lang: str) -> str:
        prompt = TRANSLATE_PROMPT.format(lang=target_lang, text=text)
        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=256, messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error("translate_error", error=str(e))
            return text


# #20: Cache provider instance to avoid re-creating clients on every request
@lru_cache
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
