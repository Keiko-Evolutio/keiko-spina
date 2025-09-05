# backend/agents/monitoring/performance_monitor.py
"""Performance Monitor für das Agent-Framework.

Performance-Monitoring mit:
- Real-time Performance-Metriken
- Automatische Baseline-Erkennung
- Performance-Anomalie-Erkennung
- Adaptive Threshold-Management
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..logging_utils import StructuredLogger

# Fallback für fehlende Module
try:
    from observability import trace_function
except ImportError:
    def trace_function(_name):
        def decorator(func):
            return func
        return decorator

# Alert-Integration
try:
    from .alert_manager import AlertLevel, AlertManager
except ImportError:
    AlertManager = None
    class AlertLevel:
        WARNING = "warning"

logger = StructuredLogger("performance_monitor")


class PerformanceConstants:
    """Konstanten für Performance Monitor."""

    # Sampling und Limits
    DEFAULT_SAMPLE_RATE = 1.0  # 100% Sampling
    DEFAULT_MAX_SAMPLES = 1000
    DEFAULT_BASELINE_WINDOW = 100
    DEFAULT_BASELINE_UPDATE_INTERVAL = 50
    MIN_BASELINE_SAMPLES = 10
    MAX_ANOMALY_HISTORY = 100

    # Thresholds
    DEFAULT_RESPONSE_TIME_THRESHOLD = 5.0  # Sekunden
    DEFAULT_THROUGHPUT_THRESHOLD = 10.0
    DEFAULT_ERROR_RATE_THRESHOLD = 0.05  # 5%
    DEFAULT_ANOMALY_MULTIPLIER = 2.0
    DEFAULT_ANOMALY_WINDOW = 20

    # Resilience Alert Thresholds
    RESILIENCE_ERROR_RATE_THRESHOLD = 0.05  # 5%
    RESILIENCE_RESPONSE_TIME_THRESHOLD = 5.0  # 5 Sekunden
    RESILIENCE_CIRCUIT_BREAKER_TRIPS = 3
    RESILIENCE_BUDGET_EXHAUSTIONS = 5

    # Retention
    DEFAULT_RETENTION_HOURS = 24

    # Defaults
    DEFAULT_FLOAT_VALUE = 0.0
    DEFAULT_COUNTER_VALUE = 1.0

    # Percentiles
    P95_PERCENTILE = 0.95
    P99_PERCENTILE = 0.99


class MetricType(Enum):
    """Typen von Performance-Metriken."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class PerformanceConfig:
    """Konfiguration für Performance Monitor."""

    # Sampling-Konfiguration
    enable_sampling: bool = True
    sample_rate: float = PerformanceConstants.DEFAULT_SAMPLE_RATE
    max_samples_per_metric: int = PerformanceConstants.DEFAULT_MAX_SAMPLES

    # Baseline-Erkennung
    enable_baseline_detection: bool = True
    baseline_window_size: int = PerformanceConstants.DEFAULT_BASELINE_WINDOW
    baseline_update_interval: int = PerformanceConstants.DEFAULT_BASELINE_UPDATE_INTERVAL

    # Anomalie-Erkennung
    enable_anomaly_detection: bool = True
    anomaly_threshold_multiplier: float = PerformanceConstants.DEFAULT_ANOMALY_MULTIPLIER
    anomaly_detection_window: int = PerformanceConstants.DEFAULT_ANOMALY_WINDOW

    # Performance-Thresholds
    default_response_time_threshold: float = PerformanceConstants.DEFAULT_RESPONSE_TIME_THRESHOLD
    default_throughput_threshold: float = PerformanceConstants.DEFAULT_THROUGHPUT_THRESHOLD
    default_error_rate_threshold: float = PerformanceConstants.DEFAULT_ERROR_RATE_THRESHOLD

    # Retention
    metric_retention_hours: int = PerformanceConstants.DEFAULT_RETENTION_HOURS
    aggregation_intervals: list[int] = field(default_factory=lambda: [60, 300, 3600])  # 1min, 5min, 1h


@dataclass
class PerformanceMetric:
    """Performance-Metrik mit Metadaten."""

    name: str
    metric_type: MetricType
    value: float | int
    timestamp: float = field(default_factory=time.time)

    # Kontext
    agent_id: str | None = None
    operation: str | None = None
    component: str | None = None

    # Zusätzliche Dimensionen
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance-Bericht mit Aggregationen."""

    report_id: str
    generated_at: float = field(default_factory=time.time)
    period_start: float = PerformanceConstants.DEFAULT_FLOAT_VALUE
    period_end: float = PerformanceConstants.DEFAULT_FLOAT_VALUE

    # Aggregierte Metriken
    response_time_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    throughput_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    error_rate_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Anomalien
    detected_anomalies: list[dict[str, Any]] = field(default_factory=list)

    # Empfehlungen
    performance_recommendations: list[str] = field(default_factory=list)

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Performance Monitor für das Agent-Framework."""

    def __init__(self, config: PerformanceConfig):
        """Initialisiert Performance Monitor.

        Args:
            config: Performance-Konfiguration
        """
        self.config = config

        # Metrik-Speicher
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.max_samples_per_metric))
        self._metric_metadata: dict[str, dict[str, Any]] = {}

        # Baseline-Tracking
        self._baselines: dict[str, dict[str, float]] = {}
        self._baseline_update_counters: dict[str, int] = defaultdict(int)

        # Anomalie-Erkennung
        self._anomaly_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Performance-Tracking
        self._operation_timers: dict[str, float] = {}
        self._throughput_counters: dict[str, int] = defaultdict(int)
        self._error_counters: dict[str, int] = defaultdict(int)

        # Aggregation-Cache
        self._aggregation_cache: dict[str, dict[str, Any]] = {}
        self._last_aggregation: float = time.time()

        # Capability- und Upstream-Metriken
        self._capability_metrics: dict[str, dict[str, Any]] = {}
        self._upstream_metrics: dict[str, dict[str, Any]] = {}
        self._resilience_metrics: dict[str, Any] = {}

        # Alert-Integration
        self._alert_manager: AlertManager | None = None
        self._alert_thresholds: dict[str, float] = {
            "error_rate": PerformanceConstants.RESILIENCE_ERROR_RATE_THRESHOLD,
            "response_time": PerformanceConstants.RESILIENCE_RESPONSE_TIME_THRESHOLD,
            "circuit_breaker_trips": PerformanceConstants.RESILIENCE_CIRCUIT_BREAKER_TRIPS,
            "budget_exhaustions": PerformanceConstants.RESILIENCE_BUDGET_EXHAUSTIONS,
        }

        logger.info("Performance Monitor initialisiert")

    def set_alert_manager(self, alert_manager: AlertManager) -> None:
        """Setzt Alert Manager für Performance-Alerts.

        Args:
            alert_manager: Alert Manager Instanz
        """
        self._alert_manager = alert_manager
        logger.info("Alert Manager für Performance Monitor gesetzt")

    def update_alert_thresholds(self, thresholds: dict[str, float]) -> None:
        """Aktualisiert Alert-Schwellenwerte.

        Args:
            thresholds: Neue Schwellenwerte
        """
        self._alert_thresholds.update(thresholds)
        logger.info(f"Alert-Schwellenwerte aktualisiert: {thresholds}")

    @trace_function("performance.record_metric")
    def record_metric(
        self,
        name: str,
        value: float | int,
        metric_type: MetricType = MetricType.GAUGE,
        agent_id: str | None = None,
        operation: str | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Zeichnet Performance-Metrik auf.

        Args:
            name: Name der Metrik
            value: Metrik-Wert
            metric_type: Typ der Metrik
            agent_id: Agent-ID
            operation: Operation-Name
            tags: Zusätzliche Tags
        """
        # Sampling prüfen
        if self.config.enable_sampling and self._should_sample():
            return

        # Metrik erstellen
        metric = PerformanceMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            agent_id=agent_id,
            operation=operation,
            tags=tags or {}
        )

        # Metrik speichern
        metric_key = self._get_metric_key(name, agent_id, operation)
        self._metrics[metric_key].append(metric)

        # Metadaten aktualisieren
        self._update_metric_metadata(metric_key, metric)

        # Baseline aktualisieren
        if self.config.enable_baseline_detection:
            self._update_baseline(metric_key, value)

        # Anomalie-Erkennung
        if self.config.enable_anomaly_detection:
            self._detect_anomaly(metric_key, value)

        logger.debug(f"Metrik aufgezeichnet: {name}={value} ({metric_type.value})")

    @trace_function("performance.start_timer")
    def start_timer(self, operation: str, agent_id: str | None = None) -> str:
        """Startet Timer für Operation.

        Args:
            operation: Operation-Name
            agent_id: Agent-ID

        Returns:
            Timer-ID
        """
        timer_id = f"{operation}_{agent_id}_{time.time()}"
        self._operation_timers[timer_id] = time.time()

        logger.debug(f"Timer gestartet: {timer_id}")
        return timer_id

    @trace_function("performance.stop_timer")
    def stop_timer(self, timer_id: str) -> float:
        """Stoppt Timer und zeichnet Dauer auf.

        Args:
            timer_id: Timer-ID

        Returns:
            Gemessene Dauer in Sekunden
        """
        if timer_id not in self._operation_timers:
            logger.warning(f"Timer nicht gefunden: {timer_id}")
            return 0.0

        start_time = self._operation_timers.pop(timer_id)
        duration = time.time() - start_time

        # Dauer als Metrik aufzeichnen
        parts = timer_id.split("_")
        operation = parts[0]
        agent_id = parts[1] if len(parts) > 1 and parts[1] != "None" else None

        self.record_metric(
            name=f"{operation}_duration",
            value=duration,
            metric_type=MetricType.TIMER,
            agent_id=agent_id,
            operation=operation
        )

        logger.debug(f"Timer gestoppt: {timer_id}, Dauer: {duration:.3f}s")
        return duration

    def record_throughput(
        self,
        operation: str,
        count: int = 1,
        agent_id: str | None = None
    ) -> None:
        """Zeichnet Durchsatz-Metrik auf.

        Args:
            operation: Operation-Name
            count: Anzahl der Operationen
            agent_id: Agent-ID
        """
        key = f"{operation}_{agent_id}"
        self._throughput_counters[key] += count

        self.record_metric(
            name=f"{operation}_throughput",
            value=count,
            metric_type=MetricType.COUNTER,
            agent_id=agent_id,
            operation=operation
        )

    def record_error(
        self,
        operation: str,
        error_type: str = "general",
        agent_id: str | None = None
    ) -> None:
        """Zeichnet Fehler-Metrik auf.

        Args:
            operation: Operation-Name
            error_type: Typ des Fehlers
            agent_id: Agent-ID
        """
        key = f"{operation}_{agent_id}"
        self._error_counters[key] += 1

        self.record_metric(
            name=f"{operation}_errors",
            value=1,
            metric_type=MetricType.COUNTER,
            agent_id=agent_id,
            operation=operation,
            tags={"error_type": error_type}
        )

    def get_metric_statistics(
        self,
        metric_name: str,
        agent_id: str | None = None,
        operation: str | None = None,
        time_window: float | None = None
    ) -> dict[str, float]:
        """Gibt Statistiken für Metrik zurück.

        Args:
            metric_name: Name der Metrik
            agent_id: Agent-ID Filter
            operation: Operation Filter
            time_window: Zeitfenster in Sekunden

        Returns:
            Dictionary mit Statistiken
        """
        metric_key = self._get_metric_key(metric_name, agent_id, operation)

        if metric_key not in self._metrics:
            return {}

        metrics = list(self._metrics[metric_key])

        # Zeitfenster-Filter
        if time_window:
            cutoff_time = time.time() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": PerformanceMonitor._percentile(values, 0.95),
            "p99": PerformanceMonitor._percentile(values, 0.99)
        }

    def generate_performance_report(
        self,
        start_time: float,
        end_time: float,
        agent_id: str | None = None
    ) -> PerformanceReport:
        """Generiert Performance-Bericht.

        Args:
            start_time: Start-Zeitstempel
            end_time: End-Zeitstempel
            agent_id: Agent-ID Filter

        Returns:
            Performance-Bericht
        """
        report = PerformanceReport(
            report_id=f"perf_report_{int(time.time())}",
            period_start=start_time,
            period_end=end_time
        )

        # Response Time Statistiken
        response_time_stats = {}
        throughput_stats = {}
        error_rate_stats = {}

        for metric_key, metrics in self._metrics.items():
            # Zeitfenster-Filter
            filtered_metrics = [
                m for m in metrics
                if start_time <= m.timestamp <= end_time
            ]

            if not filtered_metrics:
                continue

            # Agent-Filter
            if agent_id:
                filtered_metrics = [m for m in filtered_metrics if m.agent_id == agent_id]

            if not filtered_metrics:
                continue

            # Statistiken nach Metrik-Typ
            metric_name = filtered_metrics[0].name
            values = [m.value for m in filtered_metrics]

            if "duration" in metric_name or "response_time" in metric_name:
                response_time_stats[metric_name] = {
                    "mean": statistics.mean(values),
                    "p95": PerformanceMonitor._percentile(values, 0.95),
                    "p99": PerformanceMonitor._percentile(values, 0.99)
                }
            elif "throughput" in metric_name:
                throughput_stats[metric_name] = {
                    "total": sum(values),
                    "rate": sum(values) / (end_time - start_time)
                }
            elif "error" in metric_name:
                error_rate_stats[metric_name] = {
                    "total": sum(values),
                    "rate": sum(values) / len(filtered_metrics)
                }

        report.response_time_stats = response_time_stats
        report.throughput_stats = throughput_stats
        report.error_rate_stats = error_rate_stats

        # Anomalien für Zeitraum
        report.detected_anomalies = self._get_anomalies_for_period(start_time, end_time)

        # Performance-Empfehlungen
        report.performance_recommendations = self._generate_performance_recommendations(report)

        return report

    def _should_sample(self) -> bool:
        """Prüft ob Metrik gesampelt werden soll."""
        import random
        return random.random() > self.config.sample_rate

    @staticmethod
    def _get_metric_key(
        name: str,
        agent_id: str | None,
        operation: str | None
    ) -> str:
        """Generiert Metrik-Schlüssel.

        Args:
            name: Metrik-Name
            agent_id: Optional Agent-ID
            operation: Optional Operation-Name

        Returns:
            Eindeutiger Metrik-Schlüssel
        """
        parts = [name]
        if agent_id:
            parts.append(f"agent:{agent_id}")
        if operation:
            parts.append(f"op:{operation}")
        return "|".join(parts)

    def _update_metric_metadata(self, metric_key: str, metric: PerformanceMetric) -> None:
        """Aktualisiert Metrik-Metadaten."""
        if metric_key not in self._metric_metadata:
            self._metric_metadata[metric_key] = {
                "first_seen": metric.timestamp,
                "metric_type": metric.metric_type.value,
                "sample_count": 0
            }

        self._metric_metadata[metric_key]["last_seen"] = metric.timestamp
        self._metric_metadata[metric_key]["sample_count"] += 1

    def _update_baseline(self, metric_key: str, _value: float) -> None:
        """Aktualisiert Baseline für Metrik."""
        self._baseline_update_counters[metric_key] += 1

        # Baseline nur periodisch aktualisieren
        if self._baseline_update_counters[metric_key] % self.config.baseline_update_interval != 0:
            return

        # Baseline-Statistiken berechnen
        if metric_key in self._metrics:
            recent_metrics = list(self._metrics[metric_key])[-self.config.baseline_window_size:]
            values = [m.value for m in recent_metrics]

            if len(values) >= 10:  # Mindestens 10 Samples für Baseline
                self._baselines[metric_key] = {
                    "mean": statistics.mean(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "updated_at": time.time()
                }

    def _detect_anomaly(self, metric_key: str, value: float) -> None:
        """Erkennt Anomalien in Metrik."""
        if metric_key not in self._baselines:
            return

        baseline = self._baselines[metric_key]
        mean = baseline["mean"]
        std_dev = baseline["std_dev"]

        # Anomalie-Schwellwert
        threshold = std_dev * self.config.anomaly_threshold_multiplier

        if abs(value - mean) > threshold:
            anomaly = {
                "metric_key": metric_key,
                "value": value,
                "baseline_mean": mean,
                "baseline_std_dev": std_dev,
                "deviation": abs(value - mean),
                "threshold": threshold,
                "timestamp": time.time(),
                "severity": "high" if abs(value - mean) > threshold * 2 else "medium"
            }

            self._anomaly_history[metric_key].append(anomaly)

            # Nur letzte N Anomalien behalten
            if len(self._anomaly_history[metric_key]) > 100:
                self._anomaly_history[metric_key] = self._anomaly_history[metric_key][-100:]

            logger.warning(
                f"Performance-Anomalie erkannt: {metric_key}={value} "
                f"(Baseline: {mean:.2f}±{std_dev:.2f})"
            )

    def _get_anomalies_for_period(
        self,
        start_time: float,
        end_time: float
    ) -> list[dict[str, Any]]:
        """Gibt Anomalien für Zeitraum zurück."""
        anomalies = []

        for metric_key, metric_anomalies in self._anomaly_history.items():
            period_anomalies = [
                a for a in metric_anomalies
                if start_time <= a["timestamp"] <= end_time
            ]
            anomalies.extend(period_anomalies)

        # Nach Zeitstempel sortieren
        anomalies.sort(key=lambda a: a["timestamp"])

        return anomalies

    def _generate_performance_recommendations(
        self,
        report: PerformanceReport
    ) -> list[str]:
        """Generiert Performance-Empfehlungen."""
        recommendations = []

        # Response Time Empfehlungen
        for metric_name, stats in report.response_time_stats.items():
            if stats["p95"] > self.config.default_response_time_threshold:
                recommendations.append(
                    f"Hohe Response Time bei {metric_name}: "
                    f"P95={stats['p95']:.2f}s (Schwellwert: {self.config.default_response_time_threshold}s)"
                )

        # Error Rate Empfehlungen
        for metric_name, stats in report.error_rate_stats.items():
            if stats["rate"] > self.config.default_error_rate_threshold:
                recommendations.append(
                    f"Hohe Fehlerrate bei {metric_name}: "
                    f"{stats['rate']:.2%} (Schwellwert: {self.config.default_error_rate_threshold:.2%})"
                )

        # Anomalie-Empfehlungen
        if report.detected_anomalies:
            high_severity_anomalies = [a for a in report.detected_anomalies if a["severity"] == "high"]
            if high_severity_anomalies:
                recommendations.append(
                    f"{len(high_severity_anomalies)} kritische Performance-Anomalien erkannt"
                )

        return recommendations

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Berechnet Perzentil.

        Args:
            values: Liste von Werten
            percentile: Perzentil (0.0 - 1.0)

        Returns:
            Perzentil-Wert
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]

    def get_performance_summary(self) -> dict[str, Any]:
        """Gibt Performance-Zusammenfassung zurück."""
        current_time = time.time()

        return {
            "total_metrics": sum(len(metrics) for metrics in self._metrics.values()),
            "unique_metrics": len(self._metrics),
            "baselines_established": len(self._baselines),
            "total_anomalies": sum(len(anomalies) for anomalies in self._anomaly_history.values()),
            "active_timers": len(self._operation_timers),
            "config": {
                "sampling_enabled": self.config.enable_sampling,
                "sample_rate": self.config.sample_rate,
                "baseline_detection": self.config.enable_baseline_detection,
                "anomaly_detection": self.config.enable_anomaly_detection
            },
            "last_updated": current_time
        }

    def record_capability_event(
        self,
        agent_id: str,
        capability: str,
        event_type: str,
        response_time: float = 0.0,
        success: bool = True,
        error_type: str | None = None
    ) -> None:
        """Zeichnet Capability-Event auf.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            event_type: Event-Typ (request, response, error)
            response_time: Response-Zeit in Sekunden
            success: Erfolg-Status
            error_type: Fehler-Typ bei Fehlern
        """
        key = f"{agent_id}.{capability}"

        if key not in self._capability_metrics:
            self._capability_metrics[key] = {
                "agent_id": agent_id,
                "capability": capability,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0.0,
                "avg_response_time": 0.0,
                "error_rate": 0.0,
                "last_updated": time.time()
            }

        metrics = self._capability_metrics[key]

        # Metriken aktualisieren
        if event_type in ["request", "response"]:
            metrics["total_requests"] += 1
            if success:
                metrics["successful_requests"] += 1
            else:
                metrics["failed_requests"] += 1

            if response_time > 0:
                metrics["total_response_time"] += response_time
                metrics["avg_response_time"] = (
                    metrics["total_response_time"] / metrics["total_requests"]
                )

        # Error-Rate berechnen
        if metrics["total_requests"] > 0:
            metrics["error_rate"] = metrics["failed_requests"] / metrics["total_requests"]

        metrics["last_updated"] = time.time()

        # Performance-Metrik für Enterprise-System aufzeichnen
        self.record_metric(
            name=f"capability.{event_type}",
            value=response_time if response_time > 0 else 1.0,
            metric_type=MetricType.HISTOGRAM if response_time > 0 else MetricType.COUNTER,
            agent_id=agent_id,
            operation=capability,
            tags={"success": str(success), "error_type": error_type or "none"}
        )

    async def check_capability_alerts(self, agent_id: str, capability: str) -> None:
        """Prüft Capability-Metriken auf Alert-Bedingungen.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
        """
        if not self._alert_manager:
            return

        key = f"{agent_id}.{capability}"
        metrics = self._capability_metrics.get(key)

        if not metrics:
            return

        # Error-Rate-Check
        if metrics["error_rate"] > self._alert_thresholds["error_rate"]:
            await self._alert_manager.create_alert(
                title=f"High Error Rate: {agent_id}.{capability}",
                message=f"Error-Rate bei {metrics['error_rate']:.1%}",
                level=AlertLevel.WARNING,
                component="performance_monitor",
                agent_id=agent_id,
                capability=capability,
                metric_name="error_rate",
                metric_value=metrics["error_rate"],
                threshold=self._alert_thresholds["error_rate"]
            )

        # Response-Time-Check
        if metrics["avg_response_time"] > self._alert_thresholds["response_time"]:
            await self._alert_manager.create_alert(
                title=f"Slow Response: {agent_id}.{capability}",
                message=f"Durchschnittliche Response-Zeit: {metrics['avg_response_time']:.2f}s",
                level=AlertLevel.WARNING,
                component="performance_monitor",
                agent_id=agent_id,
                capability=capability,
                metric_name="avg_response_time",
                metric_value=metrics["avg_response_time"],
                threshold=self._alert_thresholds["response_time"]
            )

    def get_capability_metrics(self, agent_id: str | None = None) -> dict[str, Any]:
        """Gibt Capability-Metriken zurück.

        Args:
            agent_id: Optional Agent-ID Filter

        Returns:
            Capability-Metriken
        """
        if agent_id:
            return {
                key: metrics for key, metrics in self._capability_metrics.items()
                if metrics["agent_id"] == agent_id
            }
        return self._capability_metrics.copy()

    def get_resilience_summary(self) -> dict[str, Any]:
        """Gibt Resilience-Metriken-Zusammenfassung zurück."""
        return {
            "capabilities": self._capability_metrics,
            "upstreams": self._upstream_metrics,
            "resilience": self._resilience_metrics,
            "alert_thresholds": self._alert_thresholds,
            "alert_manager_configured": self._alert_manager is not None
        }
