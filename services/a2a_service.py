"""A2A Service – Versand und Empfang von Agent-to-Agent Nachrichten über KEI-Bus.

Implementiert ein minimales, robustes Interface für A2A-Interop mit
idempotentem Versand, Korrelation und W3C Trace-Propagation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from services.core.constants import (
    A2A_INVALID_REPLY_ERROR,
    A2A_MESSAGE_VALIDATION_ERROR,
    A2A_REQUEST_TYPE,
    DEFAULT_A2A_TIMEOUT_SECONDS,
    DEFAULT_A2A_VERSION,
)
from services.messaging import get_messaging_service
from services.messaging.envelope import BusEnvelope
from services.messaging.metrics import inject_trace
from services.messaging.naming import subject_for_a2a

if TYPE_CHECKING:
    from data_models.a2a import A2AMessage

logger = get_logger(__name__)


class A2AService:
    """Sende-/Empfangs-Fassade für A2A-Nachrichten."""

    def __init__(self) -> None:
        self._bus = get_messaging_service()

    async def send_message(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: A2AMessage,
        tenant: str | None = None,
    ) -> str:
        """Versendet eine A2A-Nachricht an einen Ziel-Agenten.

        Args:
            from_agent_id: Absender-Agent-ID
            to_agent_id: Empfänger-Agent-ID
            message: A2AMessage Instanz
            tenant: Optionaler Tenant-Name

        Returns:
            Message-ID des Bus-Events
        """
        await self._bus.initialize()
        # Schema-Validierung
        try:
            message.validate()
        except Exception as e:
            raise ValueError(f"{A2A_MESSAGE_VALIDATION_ERROR}: {e}")

        subject = subject_for_a2a(to_agent_id=to_agent_id, version=DEFAULT_A2A_VERSION, tenant=tenant)
        payload = {
            "from_agent_id": from_agent_id,
            "to_agent_id": to_agent_id,
            "message": _serialize_message(message),
        }
        env = BusEnvelope(
            type="a2a_message",
            subject=subject,
            payload=payload,
            tenant=tenant,
        )
        env.headers = inject_trace(env.headers)
        if message.corr_id:
            env.headers["correlation_id"] = message.corr_id
        if message.traceparent:
            env.headers["traceparent"] = message.traceparent

        await self._bus.publish(env)
        return env.id

    async def request_reply(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: A2AMessage,
        tenant: str | None = None,
        timeout_seconds: float = DEFAULT_A2A_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Sendet A2A Request und wartet auf Antwort (Request/Reply Pattern).

        Hinweis: Implementiert per Inbox-Subject und temporärem Subscription-Handler.
        """
        await self._bus.initialize()

        # Inbox Subject (Reply-To) ableiten – hier vereinfachte Variante
        inbox_subject = subject_for_a2a(to_agent_id=f"inbox.{from_agent_id}", version=DEFAULT_A2A_VERSION, tenant=tenant)

        # Reply-Correlation
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()

        async def _on_reply(env: BusEnvelope) -> None:
            try:
                if not future.done():
                    future.set_result(env.payload)
            except Exception:
                if not future.done():
                    future.set_result({"error": A2A_INVALID_REPLY_ERROR})

        # Temporär abonnieren
        await self._bus.subscribe(inbox_subject, queue=None, handler=_on_reply, durable=None)

        # Schema-Validierung
        try:
            message.validate()
        except Exception as e:
            raise ValueError(f"{A2A_MESSAGE_VALIDATION_ERROR}: {e}")

        # Request senden (Reply-To in Headers)
        subject = subject_for_a2a(to_agent_id=to_agent_id, version=DEFAULT_A2A_VERSION, tenant=tenant)
        payload = {
            "from_agent_id": from_agent_id,
            "to_agent_id": to_agent_id,
            "message": _serialize_message(message),
        }
        env = BusEnvelope(
            type=A2A_REQUEST_TYPE,
            subject=subject,
            payload=payload,
            tenant=tenant,
        )
        env.headers["reply_to"] = inbox_subject
        await self._bus.publish(env)

        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except Exception:
            return {"error": "timeout"}


def _serialize_message(message: A2AMessage) -> dict[str, Any]:
    """Serialisiert `A2AMessage` zu einem JSON-fähigen Dict."""
    # Dataclasses per asdict vermeiden, da Enums/Datetimes angepasst werden
    return {
        "protocol_version": message.protocol_version,
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [
            {
                "name": c.name,
                "arguments": c.arguments,
                "call_id": c.call_id,
            }
            for c in message.tool_calls
        ],
        "attachments": [
            {
                "id": a.id,
                "type": a.type,
                "uri": a.uri,
                "description": a.description,
                "metadata": a.metadata,
            }
            for a in message.attachments
        ],
        "corr_id": message.corr_id,
        "causation_id": message.causation_id,
        "traceparent": message.traceparent,
        "timestamp": message.timestamp.isoformat(),
        "headers": message.headers,
        "metadata": message.metadata,
    }


__all__ = ["A2AService"]
