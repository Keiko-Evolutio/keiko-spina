"""Bus-Metriken und -Tracing Integrationen."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger
from monitoring import get_monitoring_manager
from observability.tracing import ensure_traceparent

from .constants import (
    MAX_LATENCY_SAMPLES,
    METRIC_CONSUME_COUNT,
    METRIC_DLQ_COUNT,
    METRIC_ERRORS_COUNT,
    METRIC_LATENCY_MS,
    METRIC_PUBLISH_COUNT,
    METRIC_REDELIVERIES_COUNT,
    METRIC_TIMEOUTS_COUNT,
)

logger = get_logger(__name__)


@dataclass
class BusMetricNames:
    """Zentrale Metriknamen für KEI-Bus."""

    publish_count: str = METRIC_PUBLISH_COUNT
    consume_count: str = METRIC_CONSUME_COUNT
    redeliveries: str = METRIC_REDELIVERIES_COUNT
    dlq_count: str = METRIC_DLQ_COUNT
    latency_ms: str = METRIC_LATENCY_MS
    errors: str = METRIC_ERRORS_COUNT
    timeouts: str = METRIC_TIMEOUTS_COUNT


class BusMetrics:
    """Einfache Metrik-Wrapper basierend auf vorhandenem Monitoring-Manager."""

    def __init__(self) -> None:
        self.monitoring = get_monitoring_manager()
        self.names = BusMetricNames()
        self._latency_samples: dict[str, list[float]] = {}
        self._latency_max_points: int = MAX_LATENCY_SAMPLES

    def mark_publish(self, subject: str, tenant: str | None = None) -> None:
        """Erhöht Publish-Counter."""
        tags = {"subject": subject}
        if tenant:
            tags["tenant"] = tenant
        try:
            self.monitoring.record_metric(self.names.publish_count, 1, tags)
        except Exception:  # pragma: no cover - defensiv
            pass

    def mark_consume(self, subject: str, tenant: str | None = None) -> None:
        """Erhöht Consume-Counter."""
        tags = {"subject": subject}
        if tenant:
            tags["tenant"] = tenant
        with contextlib.suppress(Exception):
            self.monitoring.record_metric(self.names.consume_count, 1, tags)

    def record_latency(self, subject: str, started_at: float, tenant: str | None = None) -> None:
        """Erfasst End-to-End-Latenz in Millisekunden."""
        duration_ms = (time.time() - started_at) * 1000.0
        tags = {"subject": subject}
        if tenant:
            tags["tenant"] = tenant
        with contextlib.suppress(Exception):
            self.monitoring.record_metric(self.names.latency_ms, duration_ms, tags)
        # Samples für Perzentile puffern (pro Subject)
        samples = self._latency_samples.setdefault(subject, [])
        samples.append(duration_ms)
        if len(samples) > self._latency_max_points:
            del samples[0: len(samples) - self._latency_max_points]

    def mark_dlq(self, subject: str, tenant: str | None = None) -> None:
        """Erhöht DLQ-Counter."""
        tags = {"subject": subject}
        if tenant:
            tags["tenant"] = tenant
        with contextlib.suppress(Exception):
            self.monitoring.record_metric(self.names.dlq_count, 1, tags)

    def mark_redelivery(self, subject: str, tenant: str | None = None) -> None:
        """Erhöht Redelivery-Counter."""
        tags = {"subject": subject}
        if tenant:
            tags["tenant"] = tenant
        with contextlib.suppress(Exception):
            self.monitoring.record_metric(self.names.redeliveries, 1, tags)

    def mark_error(self, subject: str, tenant: str | None = None) -> None:
        """Erhöht Error-Counter."""
        tags = {"subject": subject}
        if tenant:
            tags["tenant"] = tenant
        with contextlib.suppress(Exception):
            self.monitoring.record_metric(self.names.errors, 1, tags)

    def mark_timeout(self, subject: str) -> None:
        """Erhöht Timeout-Counter für Pull-Fetch."""
        with contextlib.suppress(Exception):
            self.monitoring.record_metric(self.names.timeouts, 1, {"subject": subject})

    def get_latency_percentiles(self, subject: str | None = None) -> dict[str, float]:
        """Berechnet p50/p95/p99 für gespeicherte Latenzen (letzte N Proben)."""
        import math

        def percentiles(values: list[float]) -> dict[str, float]:
            if not values:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            arr = sorted(values)
            def pct(p: float) -> float:
                k = (len(arr) - 1) * p
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return arr[int(k)]
                return arr[f] + (arr[c] - arr[f]) * (k - f)
            return {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)}

        if subject:
            return percentiles(self._latency_samples.get(subject, []))
        # Gesamtaggregation
        all_vals: list[float] = []
        for vals in self._latency_samples.values():
            all_vals.extend(vals)
        return percentiles(all_vals)




def inject_trace(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sorgt für vorhandenen `traceparent` Header für Bus-Nachrichten."""
    return ensure_traceparent(headers)
