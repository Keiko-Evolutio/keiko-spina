"""Zentrale Redis‑Key‑Funktionen für Kei‑Webhook (mandantenspezifisch).

Alle Schlüssel werden mit `tenant_id` namespaced, um strikte Tenant‑Isolation
zu gewährleisten.
"""

from __future__ import annotations

from .constants import (
    REDIS_DELIVERY_PREFIX,
    REDIS_DLQ_PREFIX,
    REDIS_KEY_PREFIX,
    REDIS_OUTBOX_PREFIX,
    REDIS_TARGETS_PREFIX,
    get_redis_key,
    get_tenant_normalized,
)


def normalize_tenant(tenant_id: str | None) -> str:
    """Normalisiert Tenant‑IDs (None → "default").

    Deprecated: Verwende get_tenant_normalized aus constants.py
    """
    return get_tenant_normalized(tenant_id)


def outbox_key(tenant_id: str | None, queue_name: str = "default") -> str:
    """Berechnet den Outbox‑Key für einen Tenant und eine Queue."""
    tenant = get_tenant_normalized(tenant_id)
    return get_redis_key(REDIS_OUTBOX_PREFIX, tenant, queue_name)


def dlq_key(tenant_id: str | None) -> str:
    """Berechnet den DLQ‑Key für einen Tenant."""
    tenant = get_tenant_normalized(tenant_id)
    return get_redis_key(REDIS_DLQ_PREFIX, tenant)


def deliveries_recent_key(tenant_id: str | None) -> str:
    """Liste der zuletzt verarbeiteten Deliveries je Tenant."""
    tenant = get_tenant_normalized(tenant_id)
    return get_redis_key(REDIS_KEY_PREFIX, tenant, "deliveries", "recent")


def delivery_key(tenant_id: str | None, delivery_id: str) -> str:
    """Detail‑Key eines Delivery‑Objekts je Tenant."""
    tenant = get_tenant_normalized(tenant_id)
    return get_redis_key(REDIS_DELIVERY_PREFIX, tenant, delivery_id)


def retry_rate_key(tenant_id: str | None, target_id: str) -> str:
    """Rate‑Limit Key für Retry‑Operationen je Tenant und Target."""
    tenant = get_tenant_normalized(tenant_id)
    return get_redis_key(REDIS_KEY_PREFIX, tenant, "retryrl", target_id)


def targets_hash_key(tenant_id: str | None, registry_name: str = "default") -> str:
    """Hash‑Key für Webhook‑Targets pro Tenant und Registry."""
    tenant = get_tenant_normalized(tenant_id)
    return get_redis_key(REDIS_TARGETS_PREFIX, tenant, registry_name)


__all__ = [
    "deliveries_recent_key",
    "delivery_key",
    "dlq_key",
    "normalize_tenant",
    "outbox_key",
    "retry_rate_key",
    "targets_hash_key",
]
