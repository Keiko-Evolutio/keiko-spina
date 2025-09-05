"""Outbox/Inbox Muster auf Redis-Basis für zuverlässige Zustellung."""

from __future__ import annotations

import json
from typing import Any

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

logger = get_logger(__name__)

# Konstanten für Outbox/Inbox
DEFAULT_INBOX_TTL_SECONDS = 3600  # 1 Stunde


class Outbox:
    """Einfache Outbox mit Redis-Listen."""

    def __init__(self, name: str = "default") -> None:
        self.name = name

    async def add(self, message: dict[str, Any]) -> None:
        """Fügt Nachricht zur Outbox hinzu (Legacy-Listenablage)."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return
        await client.lpush(f"bus:outbox:{self.name}", json.dumps(message))

    async def persist(self, message_id: str, message: dict[str, Any]) -> None:
        """Persistiert Nachricht deterministisch für Outbox-Pattern (Hash-basiert)."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return
        try:
            await client.hset(f"bus:outbox:{self.name}:map", message_id, json.dumps(message))  # type: ignore[attr-defined]
        except Exception:
            # Fallback: Listenspeicherung
            await client.lpush(f"bus:outbox:{self.name}", json.dumps(message))

    async def flush(self) -> int:
        """Sendet alle Nachrichten (nur Rückgabe der Anzahl hier)."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return 0
        count = 0
        while True:
            data = await client.rpop(f"bus:outbox:{self.name}")
            if not data:
                break
            count += 1
            # Versenden erfolgt an anderer Stelle (Publisher ruft add und publish auf)
        return count

    async def remove(self, message_id: str) -> None:
        """Entfernt persistierte Nachricht nach erfolgreichem Publish."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return
        try:
            await client.hdel(f"bus:outbox:{self.name}:map", message_id)  # type: ignore[attr-defined]
        except Exception:
            pass


class Inbox:
    """Einfache Inbox für Deduplizierung und Verarbeitung."""

    def __init__(self, name: str = "default") -> None:
        self.name = name

    async def ack(self, message_id: str) -> None:
        """Markiert Nachricht als verarbeitet (vereinfachte Umsetzung)."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return
        await client.setex(f"bus:inbox:{self.name}:{message_id}", DEFAULT_INBOX_TTL_SECONDS, "1")
