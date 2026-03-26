import httpx
import structlog
from app.config import get_settings

logger = structlog.get_logger()


class ChatwootClient:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.chatwoot_url.rstrip("/")
        self.token = settings.chatwoot_bot_token
        self.account_id = settings.chatwoot_account_id
        self.headers = {
            "api_access_token": self.token,
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1/accounts/{self.account_id}{path}"

    async def send_message(self, conversation_id: int, content: str, private: bool = False) -> dict:
        """Send a message to a conversation."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url(f"/conversations/{conversation_id}/messages"),
                headers=self.headers,
                json={
                    "content": content,
                    "message_type": "outgoing",
                    "private": private,
                },
            )
            response.raise_for_status()
            logger.info("message_sent", conversation_id=conversation_id, private=private)
            return response.json()

    HANDOFF_MESSAGES = {
        "es": "Te transfiero con un agente humano. Un momento por favor.",
        "ca": "Et transfereixo amb un agent humà. Un moment si us plau.",
        "en": "I'm transferring you to a human agent. One moment please.",
        "pt": "Estou transferindo você para um agente humano. Um momento, por favor.",
        "fr": "Je vous transfère à un agent humain. Un moment s'il vous plaît.",
        "de": "Ich verbinde Sie mit einem menschlichen Agenten. Einen Moment bitte.",
    }

    async def handoff_to_agent(self, conversation_id: int, reason: str = None, language: str = "en") -> dict:
        """Toggle conversation status to 'open' for human agent pickup."""
        handoff_msg = self.HANDOFF_MESSAGES.get(language, self.HANDOFF_MESSAGES["en"])

        # Send handoff message to customer in their language
        await self.send_message(conversation_id, handoff_msg)

        # Add private note with reason for agents
        if reason:
            await self.send_message(
                conversation_id,
                f"🤖 Bot handoff reason: {reason}",
                private=True,
            )

        # Change status to open (triggers agent assignment)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url(f"/conversations/{conversation_id}/toggle_status"),
                headers=self.headers,
                json={"status": "open"},
            )
            response.raise_for_status()
            logger.info("handoff_to_agent", conversation_id=conversation_id, reason=reason)
            return response.json()

    async def get_messages(self, conversation_id: int) -> list[dict]:
        """Get conversation messages for context."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._url(f"/conversations/{conversation_id}/messages"),
                headers=self.headers,
            )
            response.raise_for_status()
            payload = response.json().get("payload", [])

            messages = []
            for msg in payload:
                if msg.get("content"):
                    role = "user" if msg["message_type"] == 0 else "assistant"
                    messages.append({"role": role, "content": msg["content"]})
            return messages

    async def list_resolved_conversations(self, page: int = 1) -> dict:
        """List resolved conversations (paginated, 25 per page)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._url("/conversations"),
                headers=self.headers,
                params={"status": "resolved", "page": page},
            )
            response.raise_for_status()
            return response.json()

    async def set_conversation_labels(self, conversation_id: int, labels: list[str]) -> dict:
        """Add labels to a conversation."""
        async with httpx.AsyncClient() as client:
            # Get current labels first
            conv_response = await client.get(
                self._url(f"/conversations/{conversation_id}"),
                headers=self.headers,
            )
            conv_response.raise_for_status()
            current_labels = conv_response.json().get("labels", [])

            all_labels = list(set(current_labels + labels))
            response = await client.post(
                self._url(f"/conversations/{conversation_id}/labels"),
                headers=self.headers,
                json={"labels": all_labels},
            )
            response.raise_for_status()
            return response.json()
