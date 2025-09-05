"""Event-History Storage für KEI-Webhook mittels Redis Streams.

Speichert alle ausgehenden Events in Tages-Streams mit automatischer
Ablaufzeit (TTL) und erlaubt gefiltertes Listing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

if TYPE_CHECKING:
    from .models import DeliveryRecord, WebhookEvent, WebhookTarget

logger = get_logger(__name__)


def _bucket_key(for_dt: datetime) -> str:
    """Erzeugt den Stream-Key für das Datum (UTC)."""
    d = for_dt.astimezone(UTC)
    return f"kei:webhook:history:{d.strftime('%Y%m%d')}"


class EventHistoryStore:
    """Speichert und listet Event-Historie in Redis Streams.

    Die Daten werden pro Tag in einem separaten Stream gespeichert und der
    Stream-Schlüssel erhält eine TTL gemäß `retention_days`.
    """

    def __init__(self, *, retention_days: int = 7) -> None:
        self.retention_days = retention_days

    async def write_event(self, *, record: DeliveryRecord, target: WebhookTarget, event: WebhookEvent) -> str | None:
        """Schreibt ein Event in den Tages-Stream und setzt TTL.

        Returns:
            Stream-ID des gespeicherten Eintrags oder None bei NoOp.
        """
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return None
        key = _bucket_key(datetime.now(UTC))
        fields = {
            "record": json.dumps(record.model_dump(mode="json")),
            "target": json.dumps(target.model_dump(mode="json")),
            "event": json.dumps(event.model_dump(mode="json")),
        }
        try:
            entry_id: str = await client.xadd(key, fields)  # type: ignore[attr-defined]
            # TTL für den Stream setzen (best effort)
            await client.expire(key, int(self.retention_days * 24 * 3600))  # type: ignore[attr-defined]
            return entry_id
        except Exception as exc:  # pragma: no cover
            logger.debug(f"History write failed: {exc}")
            return None

    async def list_events(
        self,
        *,
        limit: int = 50,
        target_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Listet die letzten N Events gefiltert nach Target/Event-Type.

        Iteriert rückwärts über Tages-Streams bis `limit` erreicht ist oder
        die Retentionsgrenze erreicht wurde.
        """
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return []
        now = datetime.now(UTC)
        results: list[dict[str, Any]] = []
        for day in range(self.retention_days):
            key = _bucket_key(now - timedelta(days=day))
            try:
                # XRANGE von + bis - (neueste zuerst via XREVRANGE)
                entries = await client.xrevrange(key, max="+", min="-", count=limit)  # type: ignore[attr-defined]
            except Exception:
                entries = []
            for entry_id, fields in entries or []:  # type: ignore[assignment]
                try:
                    rec = json.loads(fields.get("record", "{}"))
                    tgt = json.loads(fields.get("target", "{}"))
                    evt = json.loads(fields.get("event", "{}"))
                    if target_id and rec.get("target_id") != target_id:
                        continue
                    if event_type and evt.get("event_type") != event_type:
                        continue
                    results.append({"id": entry_id, "record": rec, "target": tgt, "event": evt})
                    if len(results) >= limit:
                        return results
                except Exception:
                    continue
            if len(results) >= limit:
                break
        return results
