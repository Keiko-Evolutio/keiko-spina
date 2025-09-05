# backend/observability/base_metrics.py
"""Base-Klassen und gemeinsame Patterns für Observability-Metriken.

Stellt wiederverwendbare Basisklassen und Utility-Funktionen für alle
Metrics Collectors bereit. Implementiert Standard-Patterns für Thread-Safety,
Metriken-Recording und -Aggregation.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from kei_logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# KONSTANTEN
# ============================================================================

class MetricsConstants:
    """Zentrale Konstanten für Observability-Metriken."""

    # Zeitkonstanten
    DEFAULT_SAFETY_MARGIN_MS = 50
    DEFAULT_COLLECTION_INTERVAL_SECONDS = 30
    DEFAULT_RETENTION_HOURS = 168  # 1 Woche
    DEFAULT_TIMEOUT_SECONDS = 30.0

    # Größenkonstanten
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_MAX_QUEUE_SIZE = 10000
    DEFAULT_MAX_SAMPLES = 1000

    # String-Konstanten
    PROMETHEUS_FORMAT = "prometheus"
    JSON_FORMAT = "json"
    CSV_FORMAT = "csv"
    INFLUXDB_FORMAT = "influxdb"

    # Header-Konstanten
    TIME_BUDGET_HEADER = "X-Time-Budget-Ms"
    TOKEN_BUDGET_HEADER = "X-Token-Budget"
    COST_BUDGET_HEADER = "X-Cost-Budget-Usd"


# ============================================================================
# PROTOCOLS UND INTERFACES
# ============================================================================

class MetricsCollectorProtocol(Protocol):
    """Protocol für Metrics Collectors."""

    def record_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None
    ) -> None:
        """Zeichnet eine Metrik auf."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Gibt alle Metriken zurück."""
        ...

    def reset_metrics(self) -> None:
        """Setzt alle Metriken zurück."""
        ...


class LatencyTrackerProtocol(Protocol):
    """Protocol für Latenz-Tracking."""

    def record_latency(self, latency_ms: float) -> None:
        """Zeichnet Latenz auf."""
        ...

    def get_percentiles(self) -> dict[str, float]:
        """Gibt Latenz-Perzentile zurück."""
        ...


# ============================================================================
# BASE CLASSES
# ============================================================================

@dataclass
class MetricDataPoint:
    """Einzelner Metriken-Datenpunkt."""
    timestamp: float
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags
        }


class BaseMetricsCollector(ABC):
    """Basisklasse für alle Metrics Collectors.

    Stellt gemeinsame Funktionalität für Thread-Safety, Metriken-Recording
    und Standard-Operationen bereit.
    """

    def __init__(self, collector_name: str):
        """Initialisiert Base Metrics Collector.

        Args:
            collector_name: Name des Collectors für Logging und Metriken
        """
        self.collector_name = collector_name
        self._lock = threading.RLock()
        self._metrics: dict[str, Any] = {}
        self._start_time = time.time()
        self._last_reset = time.time()
        self._total_metrics_recorded = 0

        logger.debug("BaseMetricsCollector '%s' initialisiert", collector_name)

    def record_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        timestamp: float | None = None
    ) -> None:
        """Zeichnet eine Metrik auf.

        Args:
            name: Metrik-Name
            value: Metrik-Wert
            tags: Optional Tags
            timestamp: Optional Timestamp (default: aktuell)
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []

            data_point = MetricDataPoint(
                timestamp=timestamp or time.time(),
                value=value,
                tags=tags or {}
            )

            self._metrics[name].append(data_point)
            self._total_metrics_recorded += 1

            # Begrenze Anzahl der Datenpunkte
            if len(self._metrics[name]) > MetricsConstants.DEFAULT_MAX_SAMPLES:
                self._metrics[name] = self._metrics[name][-MetricsConstants.DEFAULT_MAX_SAMPLES:]

    def get_metrics(self) -> dict[str, Any]:
        """Gibt alle Metriken zurück.

        Returns:
            Dictionary mit allen Metriken
        """
        with self._lock:
            return {
                "collector_name": self.collector_name,
                "start_time": self._start_time,
                "last_reset": self._last_reset,
                "uptime_seconds": time.time() - self._start_time,
                "total_metrics_recorded": self._total_metrics_recorded,
                "metrics": {
                    name: [dp.to_dict() for dp in data_points]
                    for name, data_points in self._metrics.items()
                }
            }

    def reset_metrics(self) -> None:
        """Setzt alle Metriken zurück."""
        with self._lock:
            self._metrics.clear()
            self._last_reset = time.time()
            self._total_metrics_recorded = 0

        logger.debug("Metriken für '%s' zurückgesetzt", self.collector_name)

    def get_metric_summary(self, metric_name: str) -> dict[str, Any] | None:
        """Gibt Zusammenfassung für eine spezifische Metrik zurück.

        Args:
            metric_name: Name der Metrik

        Returns:
            Metriken-Zusammenfassung oder None
        """
        with self._lock:
            if metric_name not in self._metrics:
                return None

            data_points = self._metrics[metric_name]
            if not data_points:
                return None

            values = [dp.value for dp in data_points]

            return {
                "name": metric_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None,
                "latest_timestamp": data_points[-1].timestamp if data_points else None
            }

    @abstractmethod
    def get_collector_specific_metrics(self) -> dict[str, Any]:
        """Gibt collector-spezifische Metriken zurück.

        Muss von Subklassen implementiert werden.
        """
        raise NotImplementedError("Subclasses must implement get_collector_specific_metrics")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_headers_with_context(
    existing_headers: dict[str, str] | None = None,
    time_budget_ms: int | None = None,
    token_budget: int | None = None,
    cost_budget_usd: float | None = None,
    safety_margin_ms: int = MetricsConstants.DEFAULT_SAFETY_MARGIN_MS
) -> dict[str, str]:
    """Erstellt Header-Dictionary mit Budget-Kontext.

    Args:
        existing_headers: Bestehende Headers
        time_budget_ms: Zeit-Budget in Millisekunden
        token_budget: Token-Budget
        cost_budget_usd: Kosten-Budget in USD
        safety_margin_ms: Sicherheitsabschlag in Millisekunden

    Returns:
        Header-Dictionary mit Budget-Informationen
    """
    headers = {}
    if existing_headers:
        headers.update(existing_headers)

    if time_budget_ms is not None:
        budget = max(0, time_budget_ms - safety_margin_ms)
        headers[MetricsConstants.TIME_BUDGET_HEADER] = str(budget)

    if token_budget is not None:
        headers[MetricsConstants.TOKEN_BUDGET_HEADER] = str(max(0, token_budget))

    if cost_budget_usd is not None:
        headers[MetricsConstants.COST_BUDGET_HEADER] = str(max(0.0, cost_budget_usd))

    return headers


def calculate_percentiles(values: list[int | float]) -> dict[str, float]:
    """Berechnet Perzentile für eine Liste von Werten.

    Args:
        values: Liste von numerischen Werten

    Returns:
        Dictionary mit Perzentilen (p50, p95, p99, p999)
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}

    sorted_values = sorted(values)
    count = len(sorted_values)

    def get_percentile(percentile: float) -> float:
        if count == 1:
            return float(sorted_values[0])

        index = (percentile / 100.0) * (count - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, count - 1)

        if lower_index == upper_index:
            return float(sorted_values[lower_index])

        weight = index - lower_index
        return float(
            sorted_values[lower_index] * (1 - weight) +
            sorted_values[upper_index] * weight
        )

    return {
        "p50": get_percentile(50.0),
        "p95": get_percentile(95.0),
        "p99": get_percentile(99.0),
        "p999": get_percentile(99.9)
    }


def sanitize_metric_name(name: str) -> str:
    """Sanitisiert Metrik-Namen für verschiedene Export-Formate.

    Args:
        name: Original-Metrik-Name

    Returns:
        Sanitisierter Metrik-Name
    """
    return name.replace(".", "_").replace("-", "_").replace(" ", "_").lower()


# ============================================================================
# SPECIALIZED BASE CLASSES
# ============================================================================

class BaseLatencyTracker(BaseMetricsCollector):
    """Spezialisierte Basisklasse für Latenz-Tracking.

    Implementiert Perzentil-Berechnung und Latenz-spezifische Metriken.
    """

    def __init__(self, collector_name: str):
        """Initialisiert Latency Tracker."""
        super().__init__(collector_name)
        self._latency_samples: list[float] = []

    def record_latency(self, latency_ms: float) -> None:
        """Zeichnet Latenz-Messung auf.

        Args:
            latency_ms: Latenz in Millisekunden
        """
        with self._lock:
            self._latency_samples.append(latency_ms)

            # Begrenze Anzahl der Samples
            max_samples = MetricsConstants.DEFAULT_MAX_SAMPLES
            if len(self._latency_samples) > max_samples:
                self._latency_samples = self._latency_samples[-max_samples:]

        # Zeichne als Standard-Metrik auf
        self.record_metric("latency_ms", latency_ms)

    def get_latency_percentiles(self) -> dict[str, float]:
        """Gibt Latenz-Perzentile zurück.

        Returns:
            Dictionary mit Perzentilen
        """
        with self._lock:
            return calculate_percentiles(self._latency_samples)

    def get_collector_specific_metrics(self) -> dict[str, Any]:
        """Gibt latenz-spezifische Metriken zurück."""
        with self._lock:
            percentiles = self.get_latency_percentiles()

            return {
                "latency_percentiles": percentiles,
                "total_samples": len(self._latency_samples),
                "avg_latency_ms": (
                    sum(self._latency_samples) / len(self._latency_samples)
                    if self._latency_samples else 0.0
                )
            }


class BaseRateTracker(BaseMetricsCollector):
    """Spezialisierte Basisklasse für Rate-Tracking.

    Implementiert RPS-Berechnung und Rate-spezifische Metriken.
    """

    def __init__(self, collector_name: str, window_seconds: int = 60):
        """Initialisiert Rate Tracker.

        Args:
            collector_name: Name des Collectors
            window_seconds: Zeitfenster für Rate-Berechnung
        """
        super().__init__(collector_name)
        self._window_seconds = window_seconds
        self._event_timestamps: list[float] = []

    def record_event(self, timestamp: float | None = None) -> None:
        """Zeichnet Event für Rate-Berechnung auf.

        Args:
            timestamp: Optional Timestamp (default: aktuell)
        """
        event_time = timestamp or time.time()

        with self._lock:
            self._event_timestamps.append(event_time)

            # Entferne alte Events außerhalb des Zeitfensters
            cutoff_time = event_time - self._window_seconds
            self._event_timestamps = [
                ts for ts in self._event_timestamps if ts >= cutoff_time
            ]

        # Zeichne als Standard-Metrik auf
        self.record_metric("events_total", 1, timestamp=event_time)

    def get_current_rate(self) -> float:
        """Gibt aktuelle Rate (Events pro Sekunde) zurück.

        Returns:
            Rate in Events/Sekunde
        """
        current_time = time.time()
        cutoff_time = current_time - self._window_seconds

        with self._lock:
            recent_events = [
                ts for ts in self._event_timestamps if ts >= cutoff_time
            ]

            return len(recent_events) / self._window_seconds

    def get_collector_specific_metrics(self) -> dict[str, Any]:
        """Gibt rate-spezifische Metriken zurück."""
        return {
            "current_rate_per_second": self.get_current_rate(),
            "window_seconds": self._window_seconds,
            "total_events_in_window": len(self._event_timestamps)
        }


class BaseErrorTracker(BaseMetricsCollector):
    """Spezialisierte Basisklasse für Error-Tracking.

    Implementiert Error-Kategorisierung und Error-Rate-Berechnung.
    """

    def __init__(self, collector_name: str):
        """Initialisiert Error Tracker."""
        super().__init__(collector_name)
        self._error_counts: dict[str, int] = {}
        self._total_operations = 0
        self._total_errors = 0

    def record_operation(self, success: bool, error_category: str | None = None) -> None:
        """Zeichnet Operation mit Erfolg/Fehler auf.

        Args:
            success: Ob Operation erfolgreich war
            error_category: Fehler-Kategorie bei Misserfolg
        """
        with self._lock:
            self._total_operations += 1

            if not success:
                self._total_errors += 1

                if error_category:
                    current_count = self._error_counts.get(error_category, 0)
                    self._error_counts[error_category] = current_count + 1

        # Zeichne als Standard-Metriken auf
        self.record_metric("operations_total", 1)
        if not success:
            self.record_metric("errors_total", 1, tags={"category": error_category or "unknown"})

    def get_error_rate(self) -> float:
        """Gibt aktuelle Error-Rate zurück.

        Returns:
            Error-Rate (0.0 - 1.0)
        """
        with self._lock:
            if self._total_operations == 0:
                return 0.0
            return self._total_errors / self._total_operations

    def get_collector_specific_metrics(self) -> dict[str, Any]:
        """Gibt error-spezifische Metriken zurück."""
        with self._lock:
            return {
                "error_rate": self.get_error_rate(),
                "total_operations": self._total_operations,
                "total_errors": self._total_errors,
                "error_counts_by_category": self._error_counts.copy()
            }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BaseErrorTracker",
    "BaseLatencyTracker",
    # Base Classes
    "BaseMetricsCollector",
    "BaseRateTracker",
    "LatencyTrackerProtocol",
    "MetricDataPoint",
    # Protocols
    "MetricsCollectorProtocol",
    # Konstanten
    "MetricsConstants",
    # Utility Functions
    "build_headers_with_context",
    "calculate_percentiles",
    "sanitize_metric_name",
]
