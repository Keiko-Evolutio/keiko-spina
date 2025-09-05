"""Prometheus‑Metriken für Webhook‑Subsystem.

Stellt Counter, Histogram und Gauge zur Verfügung und integriert einfache
Helper‑Funktionen zum Setzen/Aktualisieren. Alle Metriken verwenden Labels
(`target_id`, `event_type`, `tenant_id`, `status`, `error_type`) wo sinnvoll.
"""

from __future__ import annotations

import asyncio
import os

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from config.settings import settings

from .alerting import emit_info, emit_warning

_DISABLE_PROM = os.getenv("KEI_DISABLE_PROMETHEUS", "0") in {"1", "true", "yes"}
_REGISTRY = None if not _DISABLE_PROM else CollectorRegistry()

# Counter (optionale Registry in Tests, verhindert Duplikate)
WEBHOOK_REQUESTS_TOTAL = Counter(
    "webhook_requests_total",
    "Gesamtzahl eingehender Webhook‑Requests",
    labelnames=("target_id", "event_type", "tenant_id", "status"),
    registry=_REGISTRY,
)
WEBHOOK_DELIVERIES_TOTAL = Counter(
    "webhook_deliveries_total",
    "Gesamtzahl Outbound‑Zustellungen",
    labelnames=("target_id", "event_type", "tenant_id", "status"),
    registry=_REGISTRY,
)
WEBHOOK_ERRORS_TOTAL = Counter(
    "webhook_errors_total",
    "Gesamtzahl aufgetretener Webhook‑Fehler",
    labelnames=("target_id", "event_type", "tenant_id", "error_type"),
    registry=_REGISTRY,
)

# Zusätzliche Counter für Enqueue/DLQ
WEBHOOK_ENQUEUED_TOTAL = Counter(
    "webhook_enqueued_total",
    "Gesamtzahl eingeplanter Outbound‑Events",
    labelnames=("target_id", "event_type", "tenant_id"),
    registry=_REGISTRY,
)
WEBHOOK_DLQ_TOTAL = Counter(
    "webhook_dlq_total",
    "Gesamtzahl in die DLQ verschobener Zustellungen",
    labelnames=("target_id", "event_type", "tenant_id"),
    registry=_REGISTRY,
)

# Histogramme
WEBHOOK_DELIVERY_DURATION = Histogram(
    "webhook_delivery_duration_seconds",
    "Dauer einer einzelnen Outbound‑Zustellung in Sekunden",
    labelnames=("target_id", "event_type", "tenant_id", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10),
    registry=_REGISTRY,
)
WEBHOOK_PROCESSING_DURATION = Histogram(
    "webhook_processing_duration_seconds",
    "Gesamte Verarbeitungsdauer pro Outbound‑Job",
    labelnames=("target_id", "event_type", "tenant_id", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10, 30),
    registry=_REGISTRY,
)

# Gauges
WEBHOOK_ACTIVE_WORKERS = Gauge(
    "webhook_active_workers",
    "Anzahl aktiver Delivery‑Worker",
    registry=_REGISTRY,
)
WEBHOOK_QUEUE_DEPTH = Gauge(
    "webhook_queue_depth",
    "Aktuelle Tiefe der Outbox Queue",
    labelnames=("queue_name",),
    registry=_REGISTRY,
)
WEBHOOK_CIRCUIT_STATE = Gauge(
    "webhook_circuit_breaker_state",
    "Circuit Breaker Zustand (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
    labelnames=("target_id", "tenant_id"),
    registry=_REGISTRY,
)

# HTTP/Version Metriken
WEBHOOK_HTTP_VERSION_CONNECTIONS = Gauge(
    "webhook_http_version_connections",
    "Aktive HTTP Verbindungen nach Protokollversion",
    labelnames=("version",),
    registry=_REGISTRY,
)


def set_circuit_state(*, target_id: str, tenant_id: str | None, state: str) -> None:
    """Setzt Gauge‑Zustand für Circuit Breaker.

    Args:
        target_id: Ziel‑ID
        tenant_id: Tenant ID oder None
        state: Zustand als String (closed/open/half_open)
    """
    mapping = {"closed": 0, "open": 1, "half_open": 2}
    WEBHOOK_CIRCUIT_STATE.labels(target_id=target_id, tenant_id=tenant_id or "").set(mapping.get(state, -1))
    # Optional: Info-Alert bei State-Wechsel (nur Production empfohlen)
    try:
        if settings.alerting_enabled and state in ("open", "half_open"):
            title = "Circuit Breaker Status geändert"
            message = {"target_id": target_id, "tenant_id": tenant_id, "state": state}
            if state == "open":
                # Öffnen ist kritisch für Zustellungen
                asyncio.create_task(emit_warning(title, message))
            else:
                asyncio.create_task(emit_info(title, message))
    except Exception:
        pass


__all__ = [
    "WEBHOOK_ACTIVE_WORKERS",
    "WEBHOOK_CIRCUIT_STATE",
    "WEBHOOK_DELIVERIES_TOTAL",
    "WEBHOOK_DELIVERY_DURATION",
    "WEBHOOK_DLQ_TOTAL",
    "WEBHOOK_ENQUEUED_TOTAL",
    "WEBHOOK_ERRORS_TOTAL",
    "WEBHOOK_PROCESSING_DURATION",
    "WEBHOOK_QUEUE_DEPTH",
    "WEBHOOK_REQUESTS_TOTAL",
    "set_circuit_state",
]
