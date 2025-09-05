# backend/agents/slo_sla/monitor.py
"""Real-time SLO/SLA Monitoring System

Implementiert SLO/SLA-Tracking mit P95-Latency-Berechnung,
Error-Rate-Tracking und Violation-Detection.
"""

import asyncio
import statistics
import threading
import time
from collections import defaultdict, deque
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

from .config import SLOSLAConfig
from .constants import (
    DEFAULT_ERROR_RATE_WINDOW_SECONDS,
    LARGE_SLIDING_WINDOW_SIZE,
)
from .models import (
    SLADefinition,
    SLAMetrics,
    SLODefinition,
    SLOMetrics,
    SLOType,
    SLOViolation,
)
from .utils import ThreadSafeManager, get_current_timestamp

logger = get_logger(__name__)


class PercentileCalculator:
    """Effiziente Percentile-Berechnung mit Sliding-Window."""

    def __init__(self, window_size: int = LARGE_SLIDING_WINDOW_SIZE):
        """Initialisiert Percentile-Calculator.

        Args:
            window_size: Maximale Anzahl Werte im Sliding-Window
        """
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)
        self.sorted_values: list[float] = []
        self._needs_sort = False
        self._lock = threading.RLock()

    def add_value(self, value: float):
        """Fügt Wert hinzu."""
        with self._lock:
            self.values.append(value)
            self._needs_sort = True

    def get_percentile(self, percentile: float) -> float:
        """Berechnet Percentile (0.0 - 1.0).

        Args:
            percentile: Percentile (z.B. 0.95 für P95)

        Returns:
            Percentile-Wert
        """
        with self._lock:
            if not self.values:
                return 0.0

            if self._needs_sort:
                self.sorted_values = sorted(self.values)
                self._needs_sort = False

            if not self.sorted_values:
                return 0.0

            index = int(percentile * (len(self.sorted_values) - 1))
            return self.sorted_values[index]

    def get_statistics(self) -> dict[str, float]:
        """Holt umfassende Statistiken."""
        with self._lock:
            if not self.values:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            if self._needs_sort:
                self.sorted_values = sorted(self.values)
                self._needs_sort = False

            return {
                "count": len(self.values),
                "min": self.sorted_values[0],
                "max": self.sorted_values[-1],
                "mean": statistics.mean(self.values),
                "p50": self.get_percentile(0.5),
                "p95": self.get_percentile(0.95),
                "p99": self.get_percentile(0.99),
            }


class ErrorRateTracker(ThreadSafeManager):
    """Error-Rate-Tracking mit konfigurierbaren Zeitfenstern."""

    def __init__(self, window_duration_seconds: float = DEFAULT_ERROR_RATE_WINDOW_SECONDS):
        """Initialisiert Error-Rate-Tracker.

        Args:
            window_duration_seconds: Zeitfenster-Dauer in Sekunden
        """
        super().__init__()
        self.window_duration = window_duration_seconds
        self.events: deque[dict[str, Any]] = deque()

    def record_event(self, success: bool, timestamp: float | None = None):
        """Zeichnet Event auf.

        Args:
            success: Ob Event erfolgreich war
            timestamp: Optional Timestamp
        """
        if timestamp is None:
            timestamp = get_current_timestamp()

        event = {"success": success, "timestamp": timestamp}

        with self._lock:
            self.events.append(event)
            self._cleanup_old_events()

    def _cleanup_old_events(self):
        """Entfernt Events außerhalb des Zeitfensters."""
        cutoff_time = time.time() - self.window_duration

        while self.events and self.events[0]["timestamp"] < cutoff_time:
            self.events.popleft()

    def get_error_rate(self) -> float:
        """Berechnet aktuelle Error-Rate (0.0 - 1.0).

        Returns:
            Error-Rate als Dezimalzahl
        """
        with self._lock:
            self._cleanup_old_events()

            if not self.events:
                return 0.0

            total_events = len(self.events)
            failed_events = sum(1 for event in self.events if not event["success"])

            return failed_events / total_events

    def get_statistics(self) -> dict[str, Any]:
        """Holt Error-Rate-Statistiken."""
        with self._lock:
            self._cleanup_old_events()

            total_events = len(self.events)
            if total_events == 0:
                return {
                    "total_events": 0,
                    "successful_events": 0,
                    "failed_events": 0,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                }

            failed_events = sum(1 for event in self.events if not event["success"])
            successful_events = total_events - failed_events
            error_rate = failed_events / total_events

            return {
                "total_events": total_events,
                "successful_events": successful_events,
                "failed_events": failed_events,
                "error_rate": error_rate,
                "success_rate": 1.0 - error_rate,
            }


class SlidingWindowStats:
    """Sliding-Window-Statistiken für verschiedene Metriken."""

    def __init__(self, window_duration_seconds: float = 300.0):  # 5 Minuten Default
        """Initialisiert Sliding-Window-Stats.

        Args:
            window_duration_seconds: Zeitfenster-Dauer in Sekunden
        """
        self.window_duration = window_duration_seconds
        self.percentile_calculator = PercentileCalculator()
        self.error_rate_tracker = ErrorRateTracker(window_duration_seconds)
        self.throughput_events: deque[float] = deque()
        self._lock = threading.RLock()

    def record_request(
        self, response_time: float, success: bool, timestamp: float | None = None
    ):
        """Zeichnet Request auf.

        Args:
            response_time: Response-Zeit in Sekunden
            success: Ob Request erfolgreich war
            timestamp: Optional Timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Response-Zeit für Percentile-Berechnung
            self.percentile_calculator.add_value(response_time)

            # Error-Rate-Tracking
            self.error_rate_tracker.record_event(success, timestamp)

            # Throughput-Tracking
            self.throughput_events.append(timestamp)
            self._cleanup_throughput_events()

    def _cleanup_throughput_events(self):
        """Entfernt alte Throughput-Events."""
        cutoff_time = time.time() - self.window_duration

        while self.throughput_events and self.throughput_events[0] < cutoff_time:
            self.throughput_events.popleft()

    def get_throughput_rps(self) -> float:
        """Berechnet Throughput in Requests per Second."""
        with self._lock:
            self._cleanup_throughput_events()

            if not self.throughput_events:
                return 0.0

            return len(self.throughput_events) / self.window_duration

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Holt umfassende Statistiken."""
        with self._lock:
            percentile_stats = self.percentile_calculator.get_statistics()
            error_stats = self.error_rate_tracker.get_statistics()
            throughput = self.get_throughput_rps()

            return {
                "latency": percentile_stats,
                "error_rate": error_stats,
                "throughput_rps": throughput,
                "window_duration_seconds": self.window_duration,
            }


class SLOSLAMonitor:
    """Haupt-Monitor für SLO/SLA-Tracking."""

    def __init__(self, config: SLOSLAConfig):
        """Initialisiert SLO/SLA-Monitor.

        Args:
            config: SLO/SLA-Konfiguration
        """
        self.config = config

        # SLO/SLA-Definitionen und Metriken
        self.slo_definitions: dict[str, SLODefinition] = {}
        self.sla_definitions: dict[str, SLADefinition] = {}
        self.slo_metrics: dict[str, SLOMetrics] = {}
        self.sla_metrics: dict[str, SLAMetrics] = {}

        # Sliding-Window-Stats pro Capability
        self.capability_stats: dict[str, SlidingWindowStats] = {}

        # Parallel-Execution-Tracking
        self.parallel_requests: dict[str, list[float]] = defaultdict(list)

        # Thread-Safety
        self._lock = threading.RLock()

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None

        # Metrics
        self._metrics_collector = MetricsCollector()

        # Setup
        self._setup_slo_sla_definitions()

    def _setup_slo_sla_definitions(self):
        """Richtet SLO/SLA-Definitionen ein."""
        # Erstelle SLO-Definitionen für alle konfigurierten Capabilities
        for capability, slo_config in self.config.slo_config.capability_slo_configs.items():
            slo_definitions = slo_config.create_slo_definitions()

            for slo_def in slo_definitions:
                key = self._get_slo_key(slo_def)
                self.slo_definitions[key] = slo_def

                # Erstelle SLO-Metriken
                self.slo_metrics[key] = SLOMetrics(slo_definition=slo_def)

                # Erstelle Sliding-Window-Stats
                capability_key = f"{slo_def.agent_id or 'global'}.{slo_def.capability}"
                if capability_key not in self.capability_stats:
                    window_duration = slo_def.get_time_window_seconds()
                    self.capability_stats[capability_key] = SlidingWindowStats(window_duration)

        # Erstelle SLA-Definitionen
        for capability, sla_config in self.config.sla_config.capability_sla_configs.items():
            sla_def = sla_config.create_sla_definition()
            key = self._get_sla_key(sla_def)

            self.sla_definitions[key] = sla_def
            self.sla_metrics[key] = SLAMetrics(sla_definition=sla_def)

    def _get_slo_key(self, slo_def: SLODefinition) -> str:
        """Generiert eindeutigen Key für SLO."""
        return f"{slo_def.agent_id or 'global'}.{slo_def.capability or 'global'}.{slo_def.name}"

    def _get_sla_key(self, sla_def: SLADefinition) -> str:
        """Generiert eindeutigen Key für SLA."""
        return f"{sla_def.service or 'global'}.{sla_def.name}"

    async def record_capability_request(
        self,
        agent_id: str,
        capability: str,
        response_time: float,
        success: bool,
        parallel_requests: int = 1,
        timestamp: float | None = None,
    ):
        """Zeichnet Capability-Request für SLO/SLA-Tracking auf.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            response_time: Response-Zeit in Sekunden
            success: Ob Request erfolgreich war
            parallel_requests: Anzahl paralleler Requests
            timestamp: Optional Timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Sliding-Window-Stats aktualisieren
            capability_key = f"{agent_id}.{capability}"
            if capability_key not in self.capability_stats:
                self.capability_stats[capability_key] = SlidingWindowStats()

            self.capability_stats[capability_key].record_request(response_time, success, timestamp)

            # Parallel-Execution-Tracking
            if self.config.slo_config.enable_parallel_execution_tracking:
                self._track_parallel_execution(capability_key, parallel_requests, timestamp)

            # SLO-Metriken aktualisieren
            await self._update_slo_metrics(agent_id, capability, response_time, success, timestamp)

            # SLA-Metriken aktualisieren
            await self._update_sla_metrics(capability)

    def _track_parallel_execution(
        self, capability_key: str, parallel_requests: int, timestamp: float
    ):
        """Trackt parallele Ausführung für P95-Berechnung."""
        if parallel_requests >= self.config.slo_config.parallel_execution_threshold:
            self.parallel_requests[capability_key].append(timestamp)

            # Cleanup alte Einträge (letzte Stunde)
            cutoff_time = timestamp - 3600.0
            self.parallel_requests[capability_key] = [
                t for t in self.parallel_requests[capability_key] if t >= cutoff_time
            ]

    async def _update_slo_metrics(
        self, agent_id: str, capability: str, response_time: float, success: bool, timestamp: float
    ):
        """Aktualisiert SLO-Metriken."""
        # Finde relevante SLOs
        relevant_slos = [
            (key, slo_def)
            for key, slo_def in self.slo_definitions.items()
            if (slo_def.agent_id is None or slo_def.agent_id == agent_id)
            and (slo_def.capability is None or slo_def.capability == capability)
        ]

        for slo_key, slo_def in relevant_slos:
            metrics = self.slo_metrics[slo_key]

            # Berechne SLO-Wert basierend auf Typ
            slo_value = self._calculate_slo_value(
                slo_def, agent_id, capability, response_time, success
            )

            # Füge Messung hinzu
            metrics.add_measurement(slo_value, timestamp)

            # Prüfe Violation
            if not metrics.is_compliant(slo_value):
                await self._handle_slo_violation(slo_key, slo_def, metrics, slo_value, timestamp)

            # Metrics an Collector senden
            self._metrics_collector.record_gauge(
                "slo.compliance_percentage",
                metrics.compliance_percentage,
                tags={
                    "agent_id": agent_id,
                    "capability": capability,
                    "slo_name": slo_def.name,
                    "slo_type": slo_def.slo_type.value,
                },
            )

            self._metrics_collector.record_gauge(
                "slo.error_budget_consumed",
                metrics.error_budget_consumed,
                tags={"agent_id": agent_id, "capability": capability, "slo_name": slo_def.name},
            )

    def _calculate_slo_value(
        self,
        slo_def: SLODefinition,
        agent_id: str,
        capability: str,
        _response_time: float,
        _success: bool,
    ) -> float:
        """Berechnet SLO-Wert basierend auf SLO-Typ."""
        capability_key = f"{agent_id}.{capability}"
        stats = self.capability_stats.get(capability_key)

        if not stats:
            return 0.0

        if slo_def.slo_type == SLOType.LATENCY_P95:
            return stats.percentile_calculator.get_percentile(0.95)
        if slo_def.slo_type == SLOType.LATENCY_P99:
            return stats.percentile_calculator.get_percentile(0.99)
        if slo_def.slo_type == SLOType.LATENCY_P50:
            return stats.percentile_calculator.get_percentile(0.5)
        if slo_def.slo_type == SLOType.ERROR_RATE:
            return stats.error_rate_tracker.get_error_rate()
        if slo_def.slo_type == SLOType.AVAILABILITY or slo_def.slo_type == SLOType.SUCCESS_RATE:
            return 1.0 - stats.error_rate_tracker.get_error_rate()
        if slo_def.slo_type == SLOType.THROUGHPUT:
            return stats.get_throughput_rps()
        return 0.0  # Custom SLOs

    async def _handle_slo_violation(
        self,
        _slo_key: str,
        slo_def: SLODefinition,
        metrics: SLOMetrics,
        violation_value: float,
        timestamp: float,
    ):
        """Behandelt SLO-Violation."""
        # Prüfe Grace-Period
        if metrics.last_violation_time:
            time_since_last = timestamp - metrics.last_violation_time
            if time_since_last < slo_def.grace_period_seconds:
                return  # Noch in Grace-Period

        # Erstelle Violation
        violation = SLOViolation(
            slo_name=slo_def.name,
            violation_value=violation_value,
            threshold=slo_def.threshold,
            timestamp=timestamp,
            severity=metrics.calculate_violation_severity(violation_value),
            agent_id=slo_def.agent_id,
            capability=slo_def.capability,
        )

        logger.warning(
            f"SLO-Violation: {slo_def.name} - "
            f"Value: {violation_value:.4f}, Threshold: {slo_def.threshold:.4f}, "
            f"Severity: {violation.severity.value}"
        )

        # Metrics
        self._metrics_collector.increment_counter(
            "slo.violations",
            tags={
                "agent_id": slo_def.agent_id or "global",
                "capability": slo_def.capability or "global",
                "slo_name": slo_def.name,
                "slo_type": slo_def.slo_type.value,
                "severity": violation.severity.value,
            },
        )

        # Alert-Handling (wird vom SLOSLACoordinator behandelt)
        if slo_def.alert_on_violation:
            # Event für Alert-System
            pass

    async def _update_sla_metrics(self, capability: str):
        """Aktualisiert SLA-Metriken."""
        # Finde relevante SLAs
        relevant_slas = [
            (key, sla_def)
            for key, sla_def in self.sla_definitions.items()
            if sla_def.service == capability
            or any(slo.capability == capability for slo in sla_def.slo_definitions)
        ]

        for sla_key, sla_def in relevant_slas:
            metrics = self.sla_metrics[sla_key]

            # Sammle relevante SLO-Metriken
            relevant_slo_metrics = {}
            for slo_def in sla_def.slo_definitions:
                slo_key = self._get_slo_key(slo_def)
                if slo_key in self.slo_metrics:
                    relevant_slo_metrics[slo_key] = self.slo_metrics[slo_key]

            metrics.slo_metrics = relevant_slo_metrics

            # Aktualisiere SLA-Compliance
            metrics.update_compliance()

            # Metrics an Collector senden
            self._metrics_collector.record_gauge(
                "sla.compliance",
                metrics.current_compliance,
                tags={
                    "sla_name": sla_def.name,
                    "sla_type": sla_def.sla_type.value,
                    "customer": sla_def.customer or "internal",
                },
            )

            if metrics.is_breached:
                self._metrics_collector.increment_counter(
                    "sla.breaches",
                    tags={"sla_name": sla_def.name, "customer": sla_def.customer or "internal"},
                )

    def start_monitoring(self):
        """Startet SLO/SLA-Monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """Stoppt SLO/SLA-Monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()

    async def _monitoring_loop(self):
        """Monitoring-Loop für kontinuierliche SLO/SLA-Überwachung."""
        while True:
            try:
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                await self._periodic_slo_sla_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im SLO/SLA-Monitoring: {e}")

    async def _periodic_slo_sla_check(self):
        """Periodische SLO/SLA-Checks."""
        # Prüfe Error-Budget-Exhaustion
        for slo_key, metrics in self.slo_metrics.items():
            if metrics.is_error_budget_exhausted():
                logger.warning(f"Error-Budget erschöpft für SLO: {slo_key}")

                self._metrics_collector.increment_counter(
                    "slo.error_budget_exhausted",
                    tags={
                        "slo_name": metrics.slo_definition.name,
                        "agent_id": metrics.slo_definition.agent_id or "global",
                        "capability": metrics.slo_definition.capability or "global",
                    },
                )

            if metrics.is_burn_rate_critical():
                logger.warning(f"Kritische Burn-Rate für SLO: {slo_key}")

                self._metrics_collector.increment_counter(
                    "slo.burn_rate_critical",
                    tags={
                        "slo_name": metrics.slo_definition.name,
                        "agent_id": metrics.slo_definition.agent_id or "global",
                        "capability": metrics.slo_definition.capability or "global",
                    },
                )

    def get_slo_metrics(self) -> dict[str, Any]:
        """Holt SLO-Metriken für Health-Check-Integration."""
        with self._lock:
            metrics = {}
            for key, stats in self.capability_stats.items():
                comprehensive_stats = stats.get_comprehensive_stats()
                metrics[key] = {
                    "p95_ms": comprehensive_stats.get("latency", {}).get("p95", 0.0),
                    "error_rate_pct": comprehensive_stats.get("error_rate", {}).get("rate", 0.0) * 100.0,
                    "throughput_rps": comprehensive_stats.get("throughput_rps", 0.0)
                }
            return metrics

    def get_slo_metrics_summary(self) -> dict[str, Any]:
        """Holt SLO-Metriken-Zusammenfassung."""
        with self._lock:
            return {
                "total_slos": len(self.slo_definitions),
                "slo_metrics": {
                    key: metrics.to_dict() for key, metrics in self.slo_metrics.items()
                },
                "capability_stats": {
                    key: stats.get_comprehensive_stats()
                    for key, stats in self.capability_stats.items()
                },
            }

    def get_sla_metrics_summary(self) -> dict[str, Any]:
        """Holt SLA-Metriken-Zusammenfassung."""
        with self._lock:
            return {
                "total_slas": len(self.sla_definitions),
                "sla_metrics": {
                    key: metrics.to_dict() for key, metrics in self.sla_metrics.items()
                },
            }

    def get_comprehensive_summary(self) -> dict[str, Any]:
        """Holt umfassende SLO/SLA-Zusammenfassung."""
        return {
            "slo_summary": self.get_slo_metrics_summary(),
            "sla_summary": self.get_sla_metrics_summary(),
            "config": self.config.to_dict(),
            "monitoring_active": self._monitoring_task is not None
            and not self._monitoring_task.done(),
        }
