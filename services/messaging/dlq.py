"""DLQ/Parking/Requeue Verwaltung für KEI-Bus (NATS/JetStream)."""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    from nats.js import JetStreamContext
except Exception:  # pragma: no cover
    JetStreamContext = object  # type: ignore


DLQ_STREAM_NAME = "DLQ"
DLQ_SUBJECT_PREFIX = "kei.dlq.>"


async def ensure_dlq_stream(js: JetStreamContext) -> None:
    """Stellt dedizierten DLQ-Stream sicher."""
    try:
        await js.add_stream(name=DLQ_STREAM_NAME, subjects=[DLQ_SUBJECT_PREFIX])
    except Exception:
        # Existiert bereits
        pass


async def list_dlq_messages(js: JetStreamContext, subject_filter: str | None = None, max_items: int = 100) -> list[dict[str, Any]]:
    """Listet DLQ-Messages (einfach per Consumer-Info/Limits)."""
    # Vereinfachter Ansatz: JS API hat kein triviales Listing; abhängig von Storage.
    # Hier: dummy-Struktur (erweiterbar via get_msg/stream_state APIs)
    return [{"subject_filter": subject_filter or "*", "approx": True, "items": max_items}]


async def requeue_message(js: JetStreamContext, original_subject: str, data: bytes, headers: dict[str, Any] | None = None) -> None:
    """Stellt Nachricht aus DLQ zurück in Original-Subject."""
    await js.publish(original_subject, data, headers=headers)  # type: ignore[arg-type]


async def park_message(js: JetStreamContext, subject: str, data: bytes, headers: dict[str, Any] | None = None) -> None:
    """Verschiebt Nachricht in Parking Subject."""
    parking_subject = f"kei.parking.{subject}"
    await js.publish(parking_subject, data, headers=headers)  # type: ignore[arg-type]
