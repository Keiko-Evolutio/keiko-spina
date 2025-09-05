# backend/monitoring/custom_metrics.py
"""Erweiterte Custom Metrics für KEI Agent Framework.

Implementiert umfassende Metriken-Sammlung mit Agent-spezifischen Erweiterungen,
Task-RPS-Tracking, Latenz-Perzentilen und Integration mit dem neuen Observability-System.
"""

import threading
import time
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from kei_logging.pii_redaction import redact_tags
from observability import trace_function

# Import der gemeinsamen Error-Handling-Patterns
from .error_handling import (
    log_and_ignore_error,
    safe_async_operation,
)

logger = get_logger(__name__)

# Import der neuen Observability-Komponenten (lazy import um Circular Dependencies zu vermeiden)
try:
    from observability.agent_metrics import (
        AgentMetricsCollector,
        ErrorCategory,
        get_agent_metrics_collector,
        get_all_agent_metrics,
    )
    from observability.metrics_aggregator import (
        AggregationType,
        AggregationWindow,
        metrics_aggregator,
    )
    from observability.system_integration_metrics import system_integration_metrics
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    logger.debug("Erweiterte Observability-Komponenten nicht verfügbar - verwende Legacy-Modus")

    # Fallback-Definitionen für fehlende Observability-Komponenten
    from enum import Enum

    class ErrorCategory(str, Enum):
        """Fallback ErrorCategory für Legacy-Modus."""
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

    def get_agent_metrics_collector(agent_id: str):
        """Fallback get_agent_metrics_collector für Legacy-Modus."""
        class NoOpAgentMetricsCollector:
            """No-Op Agent Metrics Collector für Legacy-Modus."""
            def record_task_execution(self, duration_ms: float, success: bool, error_category=None):
                pass
            def record_tool_call(self, tool_name: str, duration_ms: float, success: bool, error_category=None):
                pass
            def record_queue_enqueue(self, queue_depth: int):
                pass
            def record_queue_dequeue(self, duration_ms: float):
                pass
        return NoOpAgentMetricsCollector()

    class NoOpMetricsAggregator:
        """No-Op Metrics Aggregator für Legacy-Modus."""
        def start(self):
            pass
        async def stop(self):
            pass
        def collect_metric(self, name: str, value: float, tags: dict = None, timestamp: float = None):
            pass

    metrics_aggregator = NoOpMetricsAggregator()

    class NoOpSystemIntegrationMetrics:
        """No-Op System Integration Metrics für Legacy-Modus."""
        def start(self):
            pass
        async def stop(self):
            pass

    system_integration_metrics = NoOpSystemIntegrationMetrics()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Import der gemeinsamen Konfiguration
from .config_base import (
    MAX_POINTS_PER_METRIC,
    RETAIN_POINTS_AFTER_TRIM,
    MetricsConfig,
)

# Legacy-Konstanten für Backward-Compatibility
_MAX_POINTS_PER_METRIC: int = MAX_POINTS_PER_METRIC
_RETAIN_POINTS_AFTER_TRIM: int = RETAIN_POINTS_AFTER_TRIM


# ============================================================================
# METRICS COLLECTOR - VEREINFACHT
# ============================================================================

class MetricsCollector:
    """Erweiterte Metrics Collector mit Observability-Integration."""

    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self._metrics: dict[str, list[dict]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self.logger = logger

        # Neue Observability-Integration
        self._observability_enabled = OBSERVABILITY_AVAILABLE and self.config.enable_agent_metrics
        self._agent_collectors: dict[str, Any] = {}  # AgentMetricsCollector per Agent

        # Performance-Tracking
        self._metrics_collected_count = 0
        self._last_rps_calculation = time.time()
        self._current_rps = 0.0

    @safe_async_operation(
        error_message="Metrics Collection Start fehlgeschlagen",
        raise_on_error=False
    )
    async def start(self) -> None:
        """Startet erweiterte Metrics Collection."""
        if self._observability_enabled:
            # Starte Observability-Komponenten mit Error-Handling
            observability_started = log_and_ignore_error(
                lambda: metrics_aggregator.start(),
                "Metrics Aggregator Start"
            )

            if observability_started:
                log_and_ignore_error(
                    lambda: system_integration_metrics.start(),
                    "System Integration Metrics Start"
                )
                self.logger.info("Observability-Integration aktiviert")
            else:
                self.logger.warning("Observability-Integration fehlgeschlagen")
                self._observability_enabled = False

        self.logger.info("Erweiterte Metrics Collection gestartet")

    async def stop(self) -> None:
        """Stoppt erweiterte Metrics Collection."""
        if self._observability_enabled:
            try:
                await metrics_aggregator.stop()
                await system_integration_metrics.stop()
            except Exception as e:
                self.logger.warning(f"Observability-Shutdown-Fehler: {e}")

        with self._lock:
            self._metrics.clear()
            self._agent_collectors.clear()

        self.logger.info("Erweiterte Metrics Collection gestoppt")

    @trace_function("metrics_collector.record_metric")
    def record_metric(self, name: str, value: int | float, tags: dict[str, Any] | None = None) -> None:
        """Zeichnet erweiterte Metrik mit Observability-Integration auf."""
        if not self.config.enabled:
            return

        # Sampling-Check für High-Volume-Scenarios
        if not self._should_sample_metric():
            return

        with self._lock:
            # Cleanup bei Bedarf
            self._cleanup_if_needed()

            # Legacy-Metrik hinzufügen
            if name not in self._metrics:
                self._metrics[name] = []

            safe_tags = redact_tags(tags)
            current_time = time.time()

            metric_point = {
                "value": value,
                "timestamp": current_time,
                "tags": safe_tags or {}
            }

            self._metrics[name].append(metric_point)

            # Memory-Limit prüfen
            if len(self._metrics[name]) > _MAX_POINTS_PER_METRIC:
                self._metrics[name] = self._metrics[name][-_RETAIN_POINTS_AFTER_TRIM:]

            # Observability-Integration mit Error-Handling
            if self._observability_enabled:
                # Sammle in neuem Aggregator
                log_and_ignore_error(
                    lambda: metrics_aggregator.collect_metric(name, value, safe_tags, current_time),
                    "Metrics Aggregator Collection"
                )

                # Agent-spezifische Metriken
                agent_id = safe_tags.get("agent_id") if safe_tags else None
                if agent_id and self.config.enable_agent_metrics:
                    log_and_ignore_error(
                        lambda: self._record_agent_metric(agent_id, name, value, safe_tags),
                        f"Agent Metrics Recording für {agent_id}"
                    )

            # Performance-Tracking
            self._metrics_collected_count += 1
            self._update_rps()

    def _should_sample_metric(self) -> bool:
        """Prüft, ob Metrik gesampelt werden soll."""
        if not self.config.enable_agent_metrics:
            return True

        # High-Volume-Sampling
        time.time()
        rps_threshold = self.config.high_volume_threshold / 60.0  # Convert to RPS
        if self._current_rps > rps_threshold:
            import random
            return random.random() < self.config.metrics_sampling_rate

        return True

    def _update_rps(self) -> None:
        """Aktualisiert RPS-Berechnung."""
        current_time = time.time()
        time_diff = current_time - self._last_rps_calculation

        rps_update_interval = 60.0  # Sekunden
        if time_diff >= rps_update_interval:
            self._current_rps = self._metrics_collected_count / time_diff
            self._metrics_collected_count = 0
            self._last_rps_calculation = current_time

    def _record_agent_metric(self, agent_id: str, metric_name: str, value: int | float, tags: dict[str, Any]) -> None:
        """Zeichnet Agent-spezifische Metrik auf."""
        try:
            if agent_id not in self._agent_collectors:
                self._agent_collectors[agent_id] = get_agent_metrics_collector(agent_id)

            agent_collector = self._agent_collectors[agent_id]

            # Task-Execution-Metriken
            if "task_duration" in metric_name:
                success = tags.get("status") == "success"
                error_category = None
                if not success:
                    error_category = ErrorCategory.BUSINESS_LOGIC_ERROR  # Default
                    if "timeout" in tags.get("error_type", ""):
                        error_category = ErrorCategory.TIMEOUT_ERROR
                    elif "auth" in tags.get("error_type", ""):
                        error_category = ErrorCategory.AUTHENTICATION_ERROR

                agent_collector.record_task_execution(value, success, error_category)

            # Tool-Call-Metriken
            elif "tool_call" in metric_name:
                tool_name = tags.get("tool_name", "unknown")
                success = tags.get("status") == "success"
                error_category = None
                if not success:
                    error_category = ErrorCategory.EXTERNAL_SERVICE_ERROR

                agent_collector.record_tool_call(tool_name, value, success, error_category)

            # Queue-Depth-Metriken
            elif "queue_depth" in metric_name:
                agent_collector.record_queue_depth(int(value))

        except Exception as e:
            self.logger.warning(f"Agent-Metrik-Recording fehlgeschlagen für {agent_id}: {e}")

    def increment_counter(self, name: str, value: int = 1, tags: dict[str, Any] | None = None) -> None:
        """Erhöht Counter-Metrik."""
        self.record_metric(f"counter.{name}", value, tags)

    def record_gauge(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
        """Zeichnet Gauge-Metrik auf."""
        self.record_metric(f"gauge.{name}", value, tags)

    def record_histogram(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
        """Zeichnet Histogram-Metrik auf."""
        self.record_metric(f"histogram.{name}", value, tags)

    def get_metrics(self) -> dict[str, Any]:
        """Gibt alle Metriken zurück."""
        with self._lock:
            return {
                "metrics_count": len(self._metrics),
                "total_data_points": sum(len(points) for points in self._metrics.values()),
                "last_cleanup": datetime.fromtimestamp(self._last_cleanup).isoformat(),
                "config": {
                    "enabled": self.config.enabled,
                    "max_metrics": self.config.max_metrics_in_memory,
                    "retention_hours": self.config.retention_hours
                }
            }

    def _cleanup_if_needed(self) -> None:
        """Cleanup alter Metriken basierend auf Konfiguration."""
        now = time.time()

        # Nur alle X Minuten cleanup
        if now - self._last_cleanup < (self.config.cleanup_interval_minutes * 60):
            return

        cutoff_time = now - (self.config.retention_hours * 3600)

        for name in list(self._metrics.keys()):
            # Entferne alte Datenpunkte
            self._metrics[name] = [
                point for point in self._metrics[name]
                if point["timestamp"] > cutoff_time
            ]

            # Entferne leere Metriken
            if not self._metrics[name]:
                del self._metrics[name]

        self._last_cleanup = now

    def health_check(self) -> dict[str, Any]:
        """Health Check für Metrics."""
        with self._lock:
            total_points = sum(len(points) for points in self._metrics.values())
            healthy = total_points < self.config.max_metrics_in_memory

            availability_ratio = 1.0
            try:
                # Einfache Verfügbarkeitsabschätzung: completed / started
                completed = len(self._metrics.get("counter.performance.completed", []))
                started = len(self._metrics.get("counter.performance.started", []))
                availability_ratio = (completed / max(1, started)) if started else 1.0
            except Exception:
                availability_ratio = 1.0

            return {
                "status": "healthy" if healthy else "warning",
                "total_metrics": len(self._metrics),
                "total_data_points": total_points,
                "memory_ok": healthy,
                # Zusätzliche Felder zur Exposition (können via /metrics exportiert werden)
                "health_availability_ratio": availability_ratio,
            }


# ============================================================================
# PERFORMANCE TRACKER - VEREINFACHT
# ============================================================================

class PerformanceTracker:
    """Performance Tracker."""

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        self.metrics = metrics_collector
        self._active_operations: dict[str, float] = {}
        self._lock = threading.RLock()

    def start_tracking(self, operation: str) -> str:
        """Startet Performance-Tracking."""
        tracking_id = f"{operation}_{int(time.time() * 1000)}"

        with self._lock:
            self._active_operations[tracking_id] = time.time()

        if self.metrics:
            self.metrics.increment_counter("performance.started", tags={"operation": operation})

        return tracking_id

    def stop_tracking(self, tracking_id: str) -> dict[str, Any] | None:
        """Stoppt Performance-Tracking."""
        with self._lock:
            start_time = self._active_operations.pop(tracking_id, None)

        if start_time is None:
            return None

        duration = time.time() - start_time
        operation = tracking_id.split("_")[0]

        if self.metrics:
            # Dauer als Histogramm für p95/p99-Auswertung aufnehmen
            self.metrics.record_histogram("performance.duration", duration, {"operation": operation})
            # p95 Hilfsmetrik: Sekunden → Millisekunden
            self.metrics.record_metric("performance.duration_ms", duration * 1000.0, {"operation": operation})
            self.metrics.increment_counter("performance.completed", tags={"operation": operation})

        return {
            "tracking_id": tracking_id,
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }


# Dead Code entfernt - leere Convenience-Funktionen eliminiert


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MetricsCollector",
    "MetricsConfig",
    "PerformanceTracker",
]
