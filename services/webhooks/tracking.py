"""Delivery-Tracking für KEI-Webhook.

Persistiert Delivery-Objekte mit TTL, erlaubt Listing/Detail, Retry (rate-
limited) und Cancel. Nutzt Redis als Speicher.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

from .constants import REDIS_DELIVERY_PREFIX, REDIS_KEY_PREFIX, REDIS_OUTBOX_PREFIX
from .keys import deliveries_recent_key, delivery_key, outbox_key, retry_rate_key

logger = get_logger(__name__)


# Konstanten für Kompatibilität mit __all__
DELIVERIES_RECENT_KEY = f"{REDIS_KEY_PREFIX}:deliveries:recent"
DELIVERY_KEY_PREFIX = REDIS_DELIVERY_PREFIX
OUTBOX_KEY = REDIS_OUTBOX_PREFIX
RETRY_RATE_KEY_PREFIX = f"{REDIS_KEY_PREFIX}:retry_rate"


# Schlüssel werden tenant‑spezifisch über .keys generiert


class DeliveryTracker:
    """Kapselt die Delivery-Tracking-Funktionalität in Redis."""

    def __init__(self, *, ttl_seconds: int = 30 * 24 * 3600) -> None:
        self.ttl_seconds = ttl_seconds

    async def save_initial(self, obj: dict[str, Any]) -> None:
        """Speichert ein neues Delivery-Objekt (mit Index) mit TTL."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return
        delivery_id = obj.get("record", {}).get("delivery_id")
        if not delivery_id:
            return
        tenant_id = (((obj.get("event", {}) or {}).get("meta", {}) or {}).get("tenant"))
        key = delivery_key(tenant_id, str(delivery_id))
        await client.set(key, json.dumps(obj), ex=self.ttl_seconds)
        await client.lpush(deliveries_recent_key(tenant_id), delivery_id)

    async def update(self, obj: dict[str, Any]) -> None:
        """Aktualisiert ein bestehendes Delivery-Objekt (erneuert TTL)."""
        await self.save_initial(obj)

    async def list_deliveries(
        self,
        *,
        tenant_id: str | None = None,
        page: int = 1,
        limit: int = 50,
        target_id: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Listet Deliveries mit einfachen Filtern und Pagination."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return {"items": [], "total": 0, "page": page, "limit": limit}
        ids: list[str] = await client.lrange(deliveries_recent_key(tenant_id), 0, 5000)  # type: ignore[attr-defined]
        results: list[dict[str, Any]] = []
        for did in ids:
            raw = await client.get(delivery_key(tenant_id, str(did)))
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                rec = obj.get("record", {})
                evt = obj.get("event", {})
                created = rec.get("created_at")
                if target_id and rec.get("target_id") != target_id:
                    continue
                if event_type and evt.get("event_type") != event_type:
                    continue
                if status and rec.get("status") != status:
                    continue
                if from_date and created and created < from_date:
                    continue
                if to_date and created and created > to_date:
                    continue
                results.append(obj)
            except Exception:  # pragma: no cover - toleranter Parser
                continue
        total = len(results)
        start = max(0, (page - 1) * limit)
        end = start + limit
        return {"items": results[start:end], "total": total, "page": page, "limit": limit}

    async def get_detail(self, *, tenant_id: str | None, delivery_id: str) -> dict[str, Any] | None:
        """Lädt die Details eines Delivery-Eintrags."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return None
        raw = await client.get(delivery_key(tenant_id, str(delivery_id)))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:  # pragma: no cover
            return None

    async def retry(self, *, tenant_id: str | None, delivery_id: str) -> str | None:
        """Erzeugt einen neuen Delivery-Versuch (Rate-Limit 5/min pro Target)."""
        from uuid import uuid4

        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return None
        detail = await self.get_detail(tenant_id=tenant_id, delivery_id=delivery_id)
        if not detail:
            return None
        rec = detail.get("record", {})
        tgt = detail.get("target", {})
        evt = detail.get("event", {})
        if rec.get("status") not in {"failed", "dlq"}:
            return None
        tenant_id = (evt.get("meta", {}) or {}).get("tenant")
        rl_key = retry_rate_key(tenant_id, rec.get("target_id",""))
        try:
            count = await client.incr(rl_key)  # type: ignore[attr-defined]
            if count == 1:
                await client.expire(rl_key, 60)  # type: ignore[attr-defined]
            if count > 5:
                return None
        except Exception:  # pragma: no cover - best effort
            pass
        new_delivery_id = uuid4().hex
        new_record = {
            **rec,
            "delivery_id": new_delivery_id,
            "status": "pending",
            "attempt": 0,
            "last_error": None,
        }
        obj = {"record": new_record, "target": tgt, "event": evt}
        await self.save_initial(obj)
        await client.lpush(outbox_key(tenant_id), json.dumps(obj))
        return new_delivery_id

    async def cancel(self, *, tenant_id: str | None, delivery_id: str) -> bool:
        """Entfernt eine geplante Delivery aus der Outbox und markiert als cancelled."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return False
        length = int(await client.llen(outbox_key(tenant_id)) or 0)
        found = False
        for _ in range(length):
            raw = await client.rpop(outbox_key(tenant_id))
            if not raw:
                break
            try:
                obj = json.loads(raw)
                if obj.get("record", {}).get("delivery_id") == delivery_id:
                    found = True
                    rec = obj.get("record", {})
                    rec["status"] = "failed"
                    rec["last_error"] = "cancelled"
                    obj["record"] = rec
                    await self.update(obj)
                    continue
            except Exception:
                pass
            await client.lpush(outbox_key(tenant_id), raw)
        return found

    async def stats(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        """Berechnet einfache Aggregatmetriken für Deliveries."""
        listing = await self.list_deliveries(tenant_id=tenant_id, page=1, limit=1000)
        items = listing.get("items", [])
        total = len(items)
        if total == 0:
            return {"total": 0, "success_rate": 0.0, "avg_latency_ms": 0.0, "retry_rate": 0.0}
        success = 0
        latency_sum_ms = 0.0
        extra_attempts_sum = 0
        for it in items:
            rec = it.get("record", {})
            if rec.get("status") == "success":
                success += 1
                c = rec.get("created_at")
                d = rec.get("delivered_at")
                if c and d:
                    try:
                        cdt = datetime.fromisoformat(c)
                        ddt = datetime.fromisoformat(d)
                        latency_sum_ms += max(0.0, (ddt - cdt).total_seconds() * 1000.0)
                    except Exception:
                        pass
            attempts = int(rec.get("attempt", 0))
            extra_attempts_sum += max(0, attempts - 1)
        success_rate = success / total
        avg_latency_ms = (latency_sum_ms / max(1, success)) if success else 0.0
        retry_rate = extra_attempts_sum / total
        return {
            "total": total,
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "retry_rate": round(retry_rate, 4),
        }


__all__ = [
    "DELIVERIES_RECENT_KEY",
    "DELIVERY_KEY_PREFIX",
    "OUTBOX_KEY",
    "RETRY_RATE_KEY_PREFIX",
    "DeliveryTracker",
]
