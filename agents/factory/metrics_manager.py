# backend/agents/factory/metrics_manager.py
"""Konsolidiertes Performance-Metrics-System für das Factory-Modul.

Merge aller Performance-Metrics aus verschiedenen Komponenten zu einem
einheitlichen, enterprise-grade Monitoring-System.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import mean, median
from typing import Any

from kei_logging import get_logger

from .constants import (
    MAX_REQUEST_COUNT,
    PERFORMANCE_SAMPLE_SIZE,
    LogLevel,
    MetricType,
)
from .singleton_mixin import SingletonMixin

logger = get_logger(__name__)


# =============================================================================
# Metric Data Classes
# =============================================================================

@dataclass
class MetricSample:
    """Einzelne Metric-Messung."""
    timestamp: datetime
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_seconds(self) -> float:
        """Gibt das Alter der Messung in Sekunden zurück."""
        return (datetime.now(UTC) - self.timestamp).total_seconds()


@dataclass
class AggregatedMetrics:
    """Aggregierte Metriken für einen bestimmten Zeitraum."""
    metric_type: MetricType
    sample_count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    percentile_95: float
    percentile_99: float
    total_value: float
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung."""
        return {
            "metric_type": self.metric_type.value,
            "sample_count": self.sample_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "median_value": self.median_value,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "total_value": self.total_value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
        }


@dataclass
class ComponentMetrics:
    """Metriken für eine spezifische Komponente (Agent, MCP Client, etc.)."""
    component_id: str
    component_type: str
    creation_time: datetime
    last_activity: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latency_samples: deque[MetricSample] = field(default_factory=lambda: deque(maxlen=PERFORMANCE_SAMPLE_SIZE))
    throughput_samples: deque[MetricSample] = field(default_factory=lambda: deque(maxlen=PERFORMANCE_SAMPLE_SIZE))
    error_samples: deque[MetricSample] = field(default_factory=lambda: deque(maxlen=PERFORMANCE_SAMPLE_SIZE))

    @property
    def success_rate(self) -> float:
        """Berechnet die Erfolgsrate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Berechnet die Fehlerrate."""
        return 1.0 - self.success_rate

    @property
    def average_latency(self) -> float:
        """Berechnet die durchschnittliche Latenz."""
        if not self.latency_samples:
            return 0.0
        return mean(sample.value for sample in self.latency_samples)

    def record_request(self, latency: float, success: bool = True) -> None:
        """Zeichnet eine Request-Metrik auf."""
        now = datetime.now(UTC)
        self.last_activity = now
        self.total_requests = min(self.total_requests + 1, MAX_REQUEST_COUNT)

        if success:
            self.successful_requests = min(self.successful_requests + 1, MAX_REQUEST_COUNT)
        else:
            self.failed_requests = min(self.failed_requests + 1, MAX_REQUEST_COUNT)

        # Latenz-Sample hinzufügen
        self.latency_samples.append(MetricSample(
            timestamp=now,
            value=latency,
            metadata={"success": success}
        ))

        # Error-Sample bei Fehler
        if not success:
            self.error_samples.append(MetricSample(
                timestamp=now,
                value=1.0,
                metadata={"error_type": "request_failed"}
            ))


# =============================================================================
# Metrics Manager
# =============================================================================

class MetricsManager(SingletonMixin):
    """Zentraler Manager für alle Performance-Metriken.

    Konsolidiert Metriken von Agents, MCP Clients, Sessions und Factory-Operationen
    zu einem einheitlichen Monitoring-System.
    """

    def _initialize_singleton(self, *args, **kwargs) -> None:
        """Initialisiert den Metrics Manager."""
        self._components: dict[str, ComponentMetrics] = {}
        self._global_metrics: dict[MetricType, deque[MetricSample]] = {
            metric_type: deque(maxlen=PERFORMANCE_SAMPLE_SIZE)
            for metric_type in MetricType
        }
        self._lock = threading.RLock()
        self._cleanup_interval = timedelta(hours=1)
        self._last_cleanup = datetime.now(UTC)

        logger.info(
            "MetricsManager initialisiert",
            extra={
                "component": "MetricsManager",
                "log_level": LogLevel.INFO
            }
        )

    def register_component(
        self,
        component_id: str,
        component_type: str
    ) -> ComponentMetrics:
        """Registriert eine neue Komponente für Metrics-Tracking.

        Args:
            component_id: Eindeutige ID der Komponente
            component_type: Typ der Komponente (agent, mcp_client, session, etc.)

        Returns:
            ComponentMetrics-Objekt für die Komponente
        """
        with self._lock:
            if component_id in self._components:
                logger.debug(
                    f"Komponente bereits registriert: {component_id}",
                    extra={
                        "component_id": component_id,
                        "component_type": component_type,
                        "log_level": LogLevel.DEBUG
                    }
                )
                return self._components[component_id]

            metrics = ComponentMetrics(
                component_id=component_id,
                component_type=component_type,
                creation_time=datetime.now(UTC),
                last_activity=datetime.now(UTC)
            )

            self._components[component_id] = metrics

            logger.debug(
                f"Komponente registriert: {component_id}",
                extra={
                    "component_id": component_id,
                    "component_type": component_type,
                    "log_level": LogLevel.DEBUG
                }
            )

            return metrics

    def record_metric(
        self,
        component_id: str,
        metric_type: MetricType,
        value: float,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Zeichnet eine Metrik für eine Komponente auf.

        Args:
            component_id: ID der Komponente
            metric_type: Typ der Metrik
            value: Metrik-Wert
            metadata: Zusätzliche Metadaten
        """
        with self._lock:
            now = datetime.now(UTC)
            sample = MetricSample(
                timestamp=now,
                value=value,
                metadata=metadata or {}
            )

            # Global Metrics
            self._global_metrics[metric_type].append(sample)

            # Component-spezifische Metrics
            if component_id in self._components:
                component = self._components[component_id]
                component.last_activity = now

                if metric_type == MetricType.LATENCY:
                    component.latency_samples.append(sample)
                elif metric_type == MetricType.THROUGHPUT:
                    component.throughput_samples.append(sample)
                elif metric_type == MetricType.ERROR_RATE:
                    component.error_samples.append(sample)

            self._maybe_cleanup()

    def record_request(
        self,
        component_id: str,
        latency: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Zeichnet eine Request-Metrik auf (Convenience-Methode).

        Args:
            component_id: ID der Komponente
            latency: Request-Latenz in Sekunden
            success: Ob der Request erfolgreich war
            metadata: Zusätzliche Metadaten
        """
        with self._lock:
            if component_id in self._components:
                self._components[component_id].record_request(latency, success)

            # Global Metrics
            self.record_metric(component_id, MetricType.LATENCY, latency, metadata)

            if not success:
                self.record_metric(component_id, MetricType.ERROR_RATE, 1.0, metadata)

    def get_component_metrics(self, component_id: str) -> ComponentMetrics | None:
        """Gibt Metriken für eine spezifische Komponente zurück."""
        with self._lock:
            return self._components.get(component_id)

    def get_aggregated_metrics(
        self,
        metric_type: MetricType,
        component_id: str | None = None,
        time_window: timedelta | None = None
    ) -> AggregatedMetrics | None:
        """Gibt aggregierte Metriken zurück.

        Args:
            metric_type: Typ der Metrik
            component_id: Optionale Komponenten-ID (None für globale Metriken)
            time_window: Zeitfenster für Aggregation (None für alle Samples)

        Returns:
            AggregatedMetrics oder None wenn keine Daten vorhanden
        """
        with self._lock:
            samples = []

            if component_id and component_id in self._components:
                component = self._components[component_id]
                if metric_type == MetricType.LATENCY:
                    samples = list(component.latency_samples)
                elif metric_type == MetricType.THROUGHPUT:
                    samples = list(component.throughput_samples)
                elif metric_type == MetricType.ERROR_RATE:
                    samples = list(component.error_samples)
            else:
                samples = list(self._global_metrics[metric_type])

            if not samples:
                return None

            # Zeitfenster-Filter anwenden
            if time_window:
                cutoff_time = datetime.now(UTC) - time_window
                samples = [s for s in samples if s.timestamp >= cutoff_time]

            if not samples:
                return None

            values = [s.value for s in samples]
            values.sort()

            return AggregatedMetrics(
                metric_type=metric_type,
                sample_count=len(values),
                min_value=min(values),
                max_value=max(values),
                mean_value=mean(values),
                median_value=median(values),
                percentile_95=values[int(len(values) * 0.95)] if values else 0.0,
                percentile_99=values[int(len(values) * 0.99)] if values else 0.0,
                total_value=sum(values),
                start_time=samples[0].timestamp,
                end_time=samples[-1].timestamp
            )

    def get_all_component_stats(self) -> dict[str, dict[str, Any]]:
        """Gibt Statistiken für alle Komponenten zurück."""
        with self._lock:
            stats = {}

            for component_id, metrics in self._components.items():
                stats[component_id] = {
                    "component_type": metrics.component_type,
                    "creation_time": metrics.creation_time.isoformat(),
                    "last_activity": metrics.last_activity.isoformat(),
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "average_latency": metrics.average_latency,
                    "latency_samples": len(metrics.latency_samples),
                    "throughput_samples": len(metrics.throughput_samples),
                    "error_samples": len(metrics.error_samples),
                }

            return stats

    def cleanup_component(self, component_id: str) -> bool:
        """Entfernt eine Komponente und ihre Metriken.

        Args:
            component_id: ID der zu entfernenden Komponente

        Returns:
            True wenn Komponente entfernt wurde, False wenn nicht gefunden
        """
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                logger.debug(
                    f"Komponenten-Metriken entfernt: {component_id}",
                    extra={
                        "component_id": component_id,
                        "log_level": LogLevel.DEBUG
                    }
                )
                return True
            return False

    def _maybe_cleanup(self) -> None:
        """Führt periodische Cleanup-Operationen durch."""
        now = datetime.now(UTC)
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_samples()
            self._last_cleanup = now

    def _cleanup_old_samples(self) -> None:
        """Entfernt alte Samples um Memory-Usage zu begrenzen."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        # Global Metrics cleanup ist automatisch durch deque maxlen

        # Component Metrics cleanup
        inactive_components = []
        for component_id, metrics in self._components.items():
            if metrics.last_activity < cutoff_time:
                inactive_components.append(component_id)

        for component_id in inactive_components:
            self.cleanup_component(component_id)

        if inactive_components:
            logger.info(
                f"Cleanup: {len(inactive_components)} inaktive Komponenten entfernt",
                extra={
                    "cleanup_count": len(inactive_components),
                    "log_level": LogLevel.INFO
                }
            )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_metrics_manager() -> MetricsManager:
    """Gibt die Singleton-Instanz des MetricsManager zurück."""
    return MetricsManager()


# =============================================================================
# Export für einfachen Import
# =============================================================================

__all__ = [
    "AggregatedMetrics",
    "ComponentMetrics",
    "MetricSample",
    "MetricsManager",
    "get_metrics_manager",
]
