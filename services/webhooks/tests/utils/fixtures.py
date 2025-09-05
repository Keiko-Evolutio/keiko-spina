"""Test-Fixtures für das KEI-Webhook System."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from ...models import (
    DeliveryRecord,
    DeliveryStatus,
    WebhookEvent,
    WebhookEventMeta,
    WebhookTarget,
)


class MockRedisClient:
    """Mock Redis-Client für Tests."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._expirations: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        """Mock GET-Operation."""
        return self._data.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None, nx: bool = False) -> bool:
        """Mock SET-Operation."""
        if nx and key in self._data:
            return False
        self._data[key] = value
        if ex:
            self._expirations[key] = ex
        return True

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Mock SETEX-Operation."""
        self._data[key] = value
        self._expirations[key] = seconds
        return True

    async def delete(self, *keys: str) -> int:
        """Mock DELETE-Operation."""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
            if key in self._lists:
                del self._lists[key]
                count += 1
            if key in self._hashes:
                del self._hashes[key]
                count += 1
            self._expirations.pop(key, None)
        return count

    async def lpush(self, key: str, *values: str) -> int:
        """Mock LPUSH-Operation."""
        if key not in self._lists:
            self._lists[key] = []
        for value in reversed(values):
            self._lists[key].insert(0, value)
        return len(self._lists[key])

    async def rpop(self, key: str) -> str | None:
        """Mock RPOP-Operation."""
        if key not in self._lists or not self._lists[key]:
            return None
        return self._lists[key].pop()

    async def llen(self, key: str) -> int:
        """Mock LLEN-Operation."""
        return len(self._lists.get(key, []))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        """Mock LRANGE-Operation."""
        if key not in self._lists:
            return []
        return self._lists[key][start:end+1 if end >= 0 else None]

    async def hset(self, key: str, field: str, value: str) -> bool:
        """Mock HSET-Operation."""
        if key not in self._hashes:
            self._hashes[key] = {}
        existed = field in self._hashes[key]
        self._hashes[key][field] = value
        return not existed

    async def hget(self, key: str, field: str) -> str | None:
        """Mock HGET-Operation."""
        return self._hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[str, str]:
        """Mock HGETALL-Operation."""
        return self._hashes.get(key, {}).copy()

    async def hdel(self, key: str, *fields: str) -> int:
        """Mock HDEL-Operation."""
        if key not in self._hashes:
            return 0
        count = 0
        for field in fields:
            if field in self._hashes[key]:
                del self._hashes[key][field]
                count += 1
        return count

    async def expire(self, key: str, seconds: int) -> bool:
        """Mock EXPIRE-Operation."""
        if key in self._data or key in self._lists or key in self._hashes:
            self._expirations[key] = seconds
            return True
        return False

    async def xadd(self, key: str, fields: dict[str, str]) -> str:
        """Mock XADD-Operation."""
        entry_id = f"{int(datetime.now().timestamp() * 1000)}-0"
        if key not in self._lists:
            self._lists[key] = []
        self._lists[key].append(json.dumps({"id": entry_id, "fields": fields}))
        return entry_id

    def clear(self) -> None:
        """Löscht alle Mock-Daten."""
        self._data.clear()
        self._lists.clear()
        self._hashes.clear()
        self._expirations.clear()


def create_webhook_event(
    *,
    event_type: str = "test.event",
    payload: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    correlation_id: str | None = None,
    occurred_at: datetime | None = None,
) -> WebhookEvent:
    """Erstellt ein Test-WebhookEvent."""
    return WebhookEvent(
        event_type=event_type,
        payload=payload or {"test": "data"},
        meta=WebhookEventMeta(
            tenant=tenant_id,
            correlation_id=correlation_id or str(uuid.uuid4()),
            source="test",
        ),
        occurred_at=occurred_at or datetime.now(UTC),
    )


def create_webhook_target(
    *,
    target_id: str | None = None,
    url: str = "https://example.com/webhook",
    enabled: bool = True,
    secret_key_name: str | None = None,
    legacy_secret: str | None = None,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    headers: dict[str, str] | None = None,
) -> WebhookTarget:
    """Erstellt ein Test-WebhookTarget."""
    return WebhookTarget(
        id=target_id or str(uuid.uuid4()),
        url=url,
        enabled=enabled,
        secret_key_name=secret_key_name,
        legacy_secret=legacy_secret,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        headers=headers,
    )


def create_delivery_record(
    *,
    delivery_id: str | None = None,
    target_id: str | None = None,
    correlation_id: str | None = None,
    status: DeliveryStatus = DeliveryStatus.pending,
    attempt: int = 0,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    delivered_at: datetime | None = None,
    last_error: str | None = None,
) -> DeliveryRecord:
    """Erstellt ein Test-DeliveryRecord."""
    now = datetime.now(UTC)
    return DeliveryRecord(
        delivery_id=delivery_id or str(uuid.uuid4()),
        target_id=target_id or str(uuid.uuid4()),
        correlation_id=correlation_id or str(uuid.uuid4()),
        status=status,
        attempt=attempt,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        created_at=created_at or now,
        updated_at=updated_at or now,
        delivered_at=delivered_at,
        last_error=last_error,
    )


__all__ = [
    "MockRedisClient",
    "create_delivery_record",
    "create_webhook_event",
    "create_webhook_target",
]
