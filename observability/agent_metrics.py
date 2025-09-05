# backend/observability/agent_metrics.py
"""Agent-spezifische Metriken-Erweiterung für Keiko Personal Assistant

Implementiert Task-RPS-Tracking, Latenz-Perzentile, Erfolg/Fehler-Raten,
Queue-Depth-Monitoring und Tool-Call-Rate-Tracking.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .base_metrics import (
    BaseLatencyTracker,
    BaseRateTracker,
    MetricsConstants,
    calculate_percentiles,
)

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Typen von Metriken."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class ErrorCategory(str, Enum):
    """Kategorien von Fehlern für detailliertes Tracking."""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_ERROR = "resource_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class PercentileMetrics:
    """Perzentil-Metriken für Latenz-Tracking."""
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    min_value: float = float("inf")
    max_value: float = 0.0
    mean: float = 0.0
    count: int = 0

    def to_dict(self) -> dict[str, float]:
        """Konvertiert zu Dictionary."""
        return {
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "p999": self.p999,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
            "count": self.count
        }


@dataclass
class RateMetrics:
    """Rate-Metriken für RPS-Tracking."""
    current_rate: float = 0.0
    peak_rate: float = 0.0
    total_count: int = 0
    window_size_seconds: int = 60

    def to_dict(self) -> dict[str, float | int]:
        """Konvertiert zu Dictionary."""
        return {
            "current_rate": self.current_rate,
            "peak_rate": self.peak_rate,
            "total_count": self.total_count,
            "window_size_seconds": self.window_size_seconds
        }


@dataclass
class ErrorMetrics:
    """Fehler-Metriken mit detaillierten Kategorien."""
    total_errors: int = 0
    error_rate: float = 0.0
    errors_by_category: dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    last_error_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "total_errors": self.total_errors,
            "error_rate": self.error_rate,
            "errors_by_category": {cat.value: count for cat, count in self.errors_by_category.items()},
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None
        }


@dataclass
class QueueMetrics:
    """Queue-Depth-Metriken."""
    current_depth: int = 0
    peak_depth: int = 0
    average_depth: float = 0.0
    total_enqueued: int = 0
    total_dequeued: int = 0
    average_wait_time_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        """Konvertiert zu Dictionary."""
        return {
            "current_depth": self.current_depth,
            "peak_depth": self.peak_depth,
            "average_depth": self.average_depth,
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "average_wait_time_ms": self.average_wait_time_ms
        }


@dataclass
class ToolCallMetrics:
    """Tool-Call-spezifische Metriken."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    success_rate: float = 0.0
    average_duration_ms: float = 0.0
    calls_per_second: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        """Konvertiert zu Dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "average_duration_ms": self.average_duration_ms,
            "calls_per_second": self.calls_per_second
        }


class LatencyTracker(BaseLatencyTracker):
    """Thread-safe Latenz-Tracker für Perzentil-Berechnung."""

    def __init__(self, max_samples: int = MetricsConstants.DEFAULT_MAX_SAMPLES):
        """Initialisiert Latency Tracker.

        Args:
            max_samples: Maximale Anzahl Samples für Perzentil-Berechnung
        """
        super().__init__("agent_latency_tracker")
        self.max_samples = max_samples
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._total_count = 0
        self._total_sum = 0.0

    def record_latency(self, latency_ms: float) -> None:
        """Zeichnet Latenz-Sample auf.

        Args:
            latency_ms: Latenz in Millisekunden
        """
        # Verwende die Base-Implementierung
        super().record_latency(latency_ms)

        # Zusätzliche Agent-spezifische Tracking
        with self._lock:
            self._samples.append(latency_ms)
            self._total_count += 1
            self._total_sum += latency_ms

    def get_percentiles(self) -> PercentileMetrics:
        """Berechnet aktuelle Perzentile.

        Returns:
            Perzentil-Metriken
        """
        with self._lock:
            if not self._samples:
                return PercentileMetrics()

            # Verwende die gemeinsame Utility-Funktion
            percentiles = calculate_percentiles(list(self._samples))

            return PercentileMetrics(
                p50=percentiles["p50"],
                p95=percentiles["p95"],
                p99=percentiles["p99"],
                p999=percentiles["p999"],
                min_value=min(self._samples),
                max_value=max(self._samples),
                mean=self._total_sum / self._total_count if self._total_count > 0 else 0.0,
                count=self._total_count
            )


class RateTracker(BaseRateTracker):
    """Thread-safe Rate-Tracker für RPS-Berechnung."""

    def __init__(self, window_size_seconds: int = 60):
        """Initialisiert Rate Tracker.

        Args:
            window_size_seconds: Zeitfenster für Rate-Berechnung
        """
        super().__init__("agent_rate_tracker", window_size_seconds)
        self.window_size_seconds = window_size_seconds
        self._timestamps: deque[float] = deque()
        self._total_count = 0
        self._peak_rate = 0.0

    def record_event(self, timestamp: float | None = None) -> None:
        """Zeichnet Event für Rate-Berechnung auf."""
        current_time = timestamp or time.time()

        # Verwende die Base-Implementierung
        super().record_event(current_time)

        # Zusätzliche Agent-spezifische Tracking
        with self._lock:
            self._timestamps.append(current_time)
            self._total_count += 1

            # Entferne alte Timestamps
            cutoff_time = current_time - self.window_size_seconds
            while self._timestamps and self._timestamps[0] < cutoff_time:
                self._timestamps.popleft()

            # Aktualisiere Peak-Rate
            current_rate = len(self._timestamps) / self.window_size_seconds
            self._peak_rate = max(self._peak_rate, current_rate)

    def get_current_rate(self) -> float:
        """Gibt aktuelle Rate zurück.

        Returns:
            Aktuelle Rate (Events pro Sekunde)
        """
        current_time = time.time()

        with self._lock:
            # Entferne alte Timestamps
            cutoff_time = current_time - self.window_size_seconds
            while self._timestamps and self._timestamps[0] < cutoff_time:
                self._timestamps.popleft()

            return len(self._timestamps) / self.window_size_seconds

    def get_rate_metrics(self) -> RateMetrics:
        """Gibt Rate-Metriken zurück.

        Returns:
            Rate-Metriken
        """
        return RateMetrics(
            current_rate=self.get_current_rate(),
            peak_rate=self._peak_rate,
            total_count=self._total_count,
            window_size_seconds=self.window_size_seconds
        )


class AgentMetricsCollector:
    """Agent-spezifischer Metriken-Collector."""

    def __init__(self, agent_id: str):
        """Initialisiert Agent Metrics Collector.

        Args:
            agent_id: Agent-ID
        """
        self.agent_id = agent_id

        # Latenz-Tracking
        self.task_latency_tracker = LatencyTracker()
        self.tool_call_latency_tracker = LatencyTracker()

        # Rate-Tracking
        self.task_rate_tracker = RateTracker()
        self.tool_call_rate_tracker = RateTracker()

        # Error-Tracking
        self.error_metrics = ErrorMetrics()
        self._error_lock = threading.RLock()

        # Queue-Tracking
        self.queue_metrics = QueueMetrics()
        self._queue_lock = threading.RLock()

        # Tool-Call-Tracking
        self.tool_call_metrics: dict[str, ToolCallMetrics] = defaultdict(ToolCallMetrics)
        self._tool_call_lock = threading.RLock()

        # Success/Failure-Tracking
        self._success_count = 0
        self._failure_count = 0
        self._success_failure_lock = threading.RLock()

        # Timestamps
        self.created_at = datetime.now(UTC)
        self.last_activity = datetime.now(UTC)

    @trace_function("agent_metrics.record_task_execution")
    def record_task_execution(
        self,
        duration_ms: float,
        success: bool,
        error_category: ErrorCategory | None = None
    ) -> None:
        """Zeichnet Task-Execution auf.

        Args:
            duration_ms: Ausführungsdauer in Millisekunden
            success: Erfolg der Ausführung
            error_category: Fehler-Kategorie bei Misserfolg
        """
        # Latenz aufzeichnen
        self.task_latency_tracker.record_latency(duration_ms)

        # Rate aufzeichnen
        self.task_rate_tracker.record_event()

        # Success/Failure aufzeichnen
        with self._success_failure_lock:
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1

        # Error aufzeichnen
        if not success and error_category:
            self.record_error(error_category)

        # Letzte Aktivität aktualisieren
        self.last_activity = datetime.now(UTC)

    @trace_function("agent_metrics.record_tool_call")
    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error_category: ErrorCategory | None = None
    ) -> None:
        """Zeichnet Tool-Call auf.

        Args:
            tool_name: Name des Tools
            duration_ms: Ausführungsdauer in Millisekunden
            success: Erfolg des Calls
            error_category: Fehler-Kategorie bei Misserfolg
        """
        # Latenz aufzeichnen
        self.tool_call_latency_tracker.record_latency(duration_ms)

        # Rate aufzeichnen
        self.tool_call_rate_tracker.record_event()

        # Tool-spezifische Metriken
        with self._tool_call_lock:
            tool_metrics = self.tool_call_metrics[tool_name]
            tool_metrics.total_calls += 1

            if success:
                tool_metrics.successful_calls += 1
            else:
                tool_metrics.failed_calls += 1

            # Aktualisiere abgeleitete Metriken
            tool_metrics.success_rate = tool_metrics.successful_calls / tool_metrics.total_calls

            # Aktualisiere durchschnittliche Dauer (vereinfacht)
            if tool_metrics.total_calls == 1:
                tool_metrics.average_duration_ms = duration_ms
            else:
                tool_metrics.average_duration_ms = (
                    (tool_metrics.average_duration_ms * (tool_metrics.total_calls - 1) + duration_ms) /
                    tool_metrics.total_calls
                )

            # Aktualisiere Calls per Second (basierend auf Rate-Tracker)
            tool_metrics.calls_per_second = self.tool_call_rate_tracker.get_current_rate()

        # Error aufzeichnen
        if not success and error_category:
            self.record_error(error_category)

        # Letzte Aktivität aktualisieren
        self.last_activity = datetime.now(UTC)

    def record_error(self, error_category: ErrorCategory) -> None:
        """Zeichnet Fehler auf.

        Args:
            error_category: Fehler-Kategorie
        """
        with self._error_lock:
            self.error_metrics.total_errors += 1
            self.error_metrics.errors_by_category[error_category] += 1
            self.error_metrics.last_error_time = datetime.now(UTC)

            # Aktualisiere Error-Rate
            total_operations = self._success_count + self._failure_count
            if total_operations > 0:
                self.error_metrics.error_rate = self._failure_count / total_operations

    def record_queue_depth(self, depth: int) -> None:
        """Zeichnet Queue-Depth auf.

        Args:
            depth: Aktuelle Queue-Tiefe
        """
        with self._queue_lock:
            self.queue_metrics.current_depth = depth

            self.queue_metrics.peak_depth = max(self.queue_metrics.peak_depth, depth)

            # Vereinfachte durchschnittliche Tiefe (exponential moving average)
            alpha = 0.1
            self.queue_metrics.average_depth = (
                alpha * depth + (1 - alpha) * self.queue_metrics.average_depth
            )

    def record_queue_enqueue(self) -> None:
        """Zeichnet Queue-Enqueue auf."""
        with self._queue_lock:
            self.queue_metrics.total_enqueued += 1

    def record_queue_dequeue(self, wait_time_ms: float) -> None:
        """Zeichnet Queue-Dequeue auf.

        Args:
            wait_time_ms: Wartezeit in der Queue in Millisekunden
        """
        with self._queue_lock:
            self.queue_metrics.total_dequeued += 1

            # Vereinfachte durchschnittliche Wartezeit (exponential moving average)
            alpha = 0.1
            self.queue_metrics.average_wait_time_ms = (
                alpha * wait_time_ms + (1 - alpha) * self.queue_metrics.average_wait_time_ms
            )

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Gibt umfassende Metriken zurück.

        Returns:
            Vollständige Metriken-Dictionary
        """
        return {
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "uptime_seconds": (datetime.now(UTC) - self.created_at).total_seconds(),

            # Task-Metriken
            "task_metrics": {
                "latency": self.task_latency_tracker.get_percentiles().to_dict(),
                "rate": self.task_rate_tracker.get_rate_metrics().to_dict(),
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "success_rate": self._success_count / max(self._success_count + self._failure_count, 1)
            },

            # Tool-Call-Metriken
            "tool_call_metrics": {
                "overall_latency": self.tool_call_latency_tracker.get_percentiles().to_dict(),
                "overall_rate": self.tool_call_rate_tracker.get_rate_metrics().to_dict(),
                "by_tool": {
                    tool_name: metrics.to_dict()
                    for tool_name, metrics in self.tool_call_metrics.items()
                }
            },

            # Error-Metriken
            "error_metrics": self.error_metrics.to_dict(),

            # Queue-Metriken
            "queue_metrics": self.queue_metrics.to_dict()
        }

    def reset_metrics(self) -> None:
        """Setzt alle Metriken zurück."""
        # Latenz-Tracker zurücksetzen
        self.task_latency_tracker = LatencyTracker()
        self.tool_call_latency_tracker = LatencyTracker()

        # Rate-Tracker zurücksetzen
        self.task_rate_tracker = RateTracker()
        self.tool_call_rate_tracker = RateTracker()

        # Error-Metriken zurücksetzen
        with self._error_lock:
            self.error_metrics = ErrorMetrics()

        # Queue-Metriken zurücksetzen
        with self._queue_lock:
            self.queue_metrics = QueueMetrics()

        # Tool-Call-Metriken zurücksetzen
        with self._tool_call_lock:
            self.tool_call_metrics.clear()

        # Success/Failure zurücksetzen
        with self._success_failure_lock:
            self._success_count = 0
            self._failure_count = 0

        # Timestamps aktualisieren
        self.created_at = datetime.now(UTC)
        self.last_activity = datetime.now(UTC)

        logger.info(f"Metriken für Agent {self.agent_id} zurückgesetzt")


# Globaler Agent-Metriken-Registry
_agent_metrics_registry: dict[str, AgentMetricsCollector] = {}
_registry_lock = threading.RLock()


def get_agent_metrics_collector(agent_id: str) -> AgentMetricsCollector:
    """Holt oder erstellt Agent-Metriken-Collector.

    Args:
        agent_id: Agent-ID

    Returns:
        Agent-Metriken-Collector
    """
    with _registry_lock:
        if agent_id not in _agent_metrics_registry:
            _agent_metrics_registry[agent_id] = AgentMetricsCollector(agent_id)

        return _agent_metrics_registry[agent_id]


def get_all_agent_metrics() -> dict[str, dict[str, Any]]:
    """Gibt Metriken für alle Agenten zurück.

    Returns:
        Metriken-Dictionary für alle Agenten
    """
    with _registry_lock:
        return {
            agent_id: collector.get_comprehensive_metrics()
            for agent_id, collector in _agent_metrics_registry.items()
        }


def reset_all_agent_metrics() -> None:
    """Setzt Metriken für alle Agenten zurück."""
    with _registry_lock:
        for collector in _agent_metrics_registry.values():
            collector.reset_metrics()

        logger.info(f"Metriken für {len(_agent_metrics_registry)} Agenten zurückgesetzt")
