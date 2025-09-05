# backend/agents/resilience/performance_monitor.py
"""Performance-Monitoring für Resilience-Features.

Implementiert Real-time-Performance-Tracking für alle Resilience-Komponenten
mit Capability-spezifischen Metriken und Alerting.
"""

import asyncio
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert-Schweregrade."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Typen von Metriken."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class CapabilityMetrics:
    """Metriken für spezifische Agent-Capability."""

    capability: str
    agent_id: str

    # Request-Metriken
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Timing-Metriken
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0

    # Circuit Breaker-Metriken
    circuit_breaker_state: str = "closed"
    circuit_breaker_trips: int = 0
    circuit_breaker_recoveries: int = 0

    # Retry-Metriken
    total_retries: int = 0
    retry_success_rate: float = 1.0

    # Budget-Metriken
    budget_warnings: int = 0
    budget_exhaustions: int = 0
    deadline_violations: int = 0

    # Resource-Utilization
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    avg_network_utilization: float = 0.0

    def add_response_time(self, response_time: float):
        """Fügt Response-Zeit hinzu und aktualisiert Statistiken."""
        self.response_times.append(response_time)

        if len(self.response_times) > 0:
            times = list(self.response_times)
            self.avg_response_time = statistics.mean(times)

            if len(times) >= 20:  # Mindestens 20 Samples für Perzentile
                sorted_times = sorted(times)
                self.p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                self.p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]

    def get_success_rate(self) -> float:
        """Berechnet Success-Rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def get_error_rate(self) -> float:
        """Berechnet Error-Rate."""
        return 1.0 - self.get_success_rate()

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "capability": self.capability,
            "agent_id": self.agent_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "error_rate": self.get_error_rate(),
            "avg_response_time": self.avg_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "circuit_breaker_state": self.circuit_breaker_state,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "circuit_breaker_recoveries": self.circuit_breaker_recoveries,
            "total_retries": self.total_retries,
            "retry_success_rate": self.retry_success_rate,
            "budget_warnings": self.budget_warnings,
            "budget_exhaustions": self.budget_exhaustions,
            "deadline_violations": self.deadline_violations,
            "avg_cpu_utilization": self.avg_cpu_utilization,
            "avg_memory_utilization": self.avg_memory_utilization,
            "avg_network_utilization": self.avg_network_utilization,
        }


@dataclass
class UpstreamMetrics:
    """Metriken für Upstream-Services."""

    upstream_id: str

    # Request-Metriken
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Retry-Metriken
    total_retries: int = 0
    retry_attempts: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_retry_attempts: float = 0.0

    # Performance-Metriken
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_response_time: float = 0.0
    current_base_delay: float = 1.0

    # Health-Metriken
    health_score: float = 1.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0

    def add_request_result(self, success: bool, response_time: float, retry_attempts: int = 0):
        """Fügt Request-Ergebnis hinzu."""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
            self.last_success_time = time.time()
        else:
            self.failed_requests += 1
            self.last_failure_time = time.time()

        self.response_times.append(response_time)
        if len(self.response_times) > 0:
            self.avg_response_time = statistics.mean(self.response_times)

        if retry_attempts > 0:
            self.total_retries += retry_attempts
            self.retry_attempts.append(retry_attempts)
            if len(self.retry_attempts) > 0:
                self.avg_retry_attempts = statistics.mean(self.retry_attempts)

        # Health-Score aktualisieren (exponential moving average)
        alpha = 0.1
        new_score = 1.0 if success else 0.0
        self.health_score = (alpha * new_score) + ((1 - alpha) * self.health_score)

    def get_success_rate(self) -> float:
        """Berechnet Success-Rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "upstream_id": self.upstream_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "total_retries": self.total_retries,
            "avg_retry_attempts": self.avg_retry_attempts,
            "avg_response_time": self.avg_response_time,
            "current_base_delay": self.current_base_delay,
            "health_score": self.health_score,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
        }


@dataclass
class ResilienceMetrics:
    """Gesamte Resilience-Metriken."""

    # Circuit Breaker-Metriken
    total_circuit_breakers: int = 0
    open_circuit_breakers: int = 0
    half_open_circuit_breakers: int = 0

    # Retry-Metriken
    total_retry_policies: int = 0
    total_retries_executed: int = 0
    retry_success_rate: float = 1.0

    # Budget-Metriken
    active_budgets: int = 0
    budget_warnings_total: int = 0
    budget_exhaustions_total: int = 0
    deadline_violations_total: int = 0

    # Performance-Metriken
    avg_system_response_time: float = 0.0
    system_success_rate: float = 1.0
    system_health_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "circuit_breakers": {
                "total": self.total_circuit_breakers,
                "open": self.open_circuit_breakers,
                "half_open": self.half_open_circuit_breakers,
                "closed": self.total_circuit_breakers
                - self.open_circuit_breakers
                - self.half_open_circuit_breakers,
            },
            "retries": {
                "total_policies": self.total_retry_policies,
                "total_executed": self.total_retries_executed,
                "success_rate": self.retry_success_rate,
            },
            "budgets": {
                "active": self.active_budgets,
                "warnings_total": self.budget_warnings_total,
                "exhaustions_total": self.budget_exhaustions_total,
                "deadline_violations_total": self.deadline_violations_total,
            },
            "performance": {
                "avg_response_time": self.avg_system_response_time,
                "success_rate": self.system_success_rate,
                "health_score": self.system_health_score,
            },
        }


@dataclass
class Alert:
    """Alert für Performance-/Resilience-Ereignisse."""

    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float = field(default_factory=time.time)

    # Context
    agent_id: str | None = None
    capability: str | None = None
    upstream_id: str | None = None

    # Metrics
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "capability": self.capability,
            "upstream_id": self.upstream_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "tags": self.tags,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }


class AlertManager:
    """Manager für Performance-/Resilience-Alerts."""

    def __init__(self):
        """Initialisiert Alert-Manager."""
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._lock = threading.RLock()

        # Alert-Callbacks
        self._alert_callbacks: dict[AlertSeverity, list[Callable[[Alert], Awaitable[None]]]] = {
            severity: [] for severity in AlertSeverity
        }

        # Metrics
        self._metrics_collector = MetricsCollector()

    def register_alert_callback(
        self, severity: AlertSeverity, callback: Callable[[Alert], Awaitable[None]]
    ):
        """Registriert Callback für Alert-Severity.

        Args:
            severity: Alert-Schweregrad
            callback: Callback-Funktion
        """
        self._alert_callbacks[severity].append(callback)

    async def create_alert(
        self, alert_id: str, severity: AlertSeverity, title: str, description: str, **kwargs
    ) -> Alert:
        """Erstellt neuen Alert.

        Args:
            alert_id: Eindeutige Alert-ID
            severity: Alert-Schweregrad
            title: Alert-Titel
            description: Alert-Beschreibung
            **kwargs: Zusätzliche Alert-Attribute

        Returns:
            Erstellter Alert
        """
        alert = Alert(
            alert_id=alert_id, severity=severity, title=title, description=description, **kwargs
        )

        with self._lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)

        logger.warning(f"Alert erstellt: {alert.title} ({alert.severity.value})")

        # Callbacks ausführen
        callbacks = self._alert_callbacks.get(severity, [])
        for callback in callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert-Callback fehlgeschlagen: {e}")

        # Metrics
        self._metrics_collector.increment_counter(
            "alerts.created",
            tags={
                "severity": severity.value,
                "agent_id": alert.agent_id or "unknown",
                "capability": alert.capability or "unknown",
            },
        )

        return alert

    async def resolve_alert(self, alert_id: str) -> Alert | None:
        """Löst Alert auf.

        Args:
            alert_id: Alert-ID

        Returns:
            Aufgelöster Alert oder None
        """
        with self._lock:
            alert = self._active_alerts.pop(alert_id, None)

            if alert:
                alert.resolved = True
                alert.resolved_at = time.time()

                logger.info(f"Alert aufgelöst: {alert.title}")

                # Metrics
                self._metrics_collector.increment_counter(
                    "alerts.resolved",
                    tags={
                        "severity": alert.severity.value,
                        "agent_id": alert.agent_id or "unknown",
                        "capability": alert.capability or "unknown",
                    },
                )

        return alert

    def get_active_alerts(self) -> list[Alert]:
        """Holt alle aktiven Alerts.

        Returns:
            Liste aktiver Alerts
        """
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Holt Alert-History.

        Args:
            limit: Maximale Anzahl Alerts

        Returns:
            Alert-History
        """
        with self._lock:
            history = list(self._alert_history)
            return history[-limit:] if len(history) > limit else history

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt Alert-Metriken-Zusammenfassung.

        Returns:
            Alert-Metriken
        """
        with self._lock:
            active_alerts = list(self._active_alerts.values())

            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1

            return {
                "active_alerts": len(active_alerts),
                "total_alerts_in_history": len(self._alert_history),
                "severity_breakdown": dict(severity_counts),
                "alerts": [alert.to_dict() for alert in active_alerts],
            }


class PerformanceMonitor:
    """Haupt-Performance-Monitor für Resilience-Features."""

    def __init__(self):
        """Initialisiert Performance-Monitor."""
        self._capability_metrics: dict[str, CapabilityMetrics] = {}
        self._upstream_metrics: dict[str, UpstreamMetrics] = {}
        self._resilience_metrics = ResilienceMetrics()
        self._alert_manager = AlertManager()
        self._lock = threading.RLock()

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 10.0  # 10 Sekunden

        # Thresholds für Alerts
        self._alert_thresholds = {
            "error_rate": 0.05,  # 5% Error-Rate
            "response_time": 5.0,  # 5 Sekunden
            "circuit_breaker_trips": 3,  # 3 Trips
            "budget_exhaustions": 5,  # 5 Budget-Exhaustions
        }

        # Metrics
        self._metrics_collector = MetricsCollector()

    def get_capability_metrics(self, agent_id: str, capability: str) -> CapabilityMetrics:
        """Holt oder erstellt Capability-Metriken.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name

        Returns:
            Capability-Metriken
        """
        key = f"{agent_id}.{capability}"

        with self._lock:
            if key not in self._capability_metrics:
                self._capability_metrics[key] = CapabilityMetrics(
                    capability=capability, agent_id=agent_id
                )

            return self._capability_metrics[key]

    def get_upstream_metrics(self, upstream_id: str) -> UpstreamMetrics:
        """Holt oder erstellt Upstream-Metriken.

        Args:
            upstream_id: Upstream-Service-ID

        Returns:
            Upstream-Metriken
        """
        with self._lock:
            if upstream_id not in self._upstream_metrics:
                self._upstream_metrics[upstream_id] = UpstreamMetrics(upstream_id=upstream_id)

            return self._upstream_metrics[upstream_id]

    async def record_capability_request(
        self, agent_id: str, capability: str, success: bool, response_time: float, **kwargs
    ):
        """Zeichnet Capability-Request auf.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            success: Ob Request erfolgreich war
            response_time: Response-Zeit in Sekunden
            **kwargs: Zusätzliche Metriken
        """
        metrics = self.get_capability_metrics(agent_id, capability)

        with self._lock:
            metrics.total_requests += 1

            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

            metrics.add_response_time(response_time)

            # Zusätzliche Metriken
            if "circuit_breaker_state" in kwargs:
                metrics.circuit_breaker_state = kwargs["circuit_breaker_state"]

            if "retries" in kwargs:
                metrics.total_retries += kwargs["retries"]

        # Metrics an Collector senden
        self._metrics_collector.increment_counter(
            "capability.requests",
            tags={"agent_id": agent_id, "capability": capability, "success": str(success)},
        )

        self._metrics_collector.record_histogram(
            "capability.response_time",
            response_time,
            tags={"agent_id": agent_id, "capability": capability},
        )

        # Alert-Checks
        await self._check_capability_alerts(metrics)

    async def record_upstream_request(
        self, upstream_id: str, success: bool, response_time: float, retry_attempts: int = 0
    ):
        """Zeichnet Upstream-Request auf.

        Args:
            upstream_id: Upstream-Service-ID
            success: Ob Request erfolgreich war
            response_time: Response-Zeit in Sekunden
            retry_attempts: Anzahl Retry-Versuche
        """
        metrics = self.get_upstream_metrics(upstream_id)

        with self._lock:
            metrics.add_request_result(success, response_time, retry_attempts)

        # Metrics an Collector senden
        self._metrics_collector.increment_counter(
            "upstream.requests", tags={"upstream_id": upstream_id, "success": str(success)}
        )

        if retry_attempts > 0:
            self._metrics_collector.increment_counter(
                "upstream.retries", tags={"upstream_id": upstream_id}
            )

    async def record_circuit_breaker_event(
        self,
        agent_id: str,
        capability: str,
        event_type: str,  # "trip", "recovery", "call_rejected"
        **_,
    ):
        """Zeichnet Circuit Breaker-Event auf.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            event_type: Event-Typ
            **kwargs: Zusätzliche Event-Daten
        """
        metrics = self.get_capability_metrics(agent_id, capability)

        with self._lock:
            if event_type == "trip":
                metrics.circuit_breaker_trips += 1
                metrics.circuit_breaker_state = "open"
            elif event_type == "recovery":
                metrics.circuit_breaker_recoveries += 1
                metrics.circuit_breaker_state = "closed"
            elif event_type == "half_open":
                metrics.circuit_breaker_state = "half_open"

        # Metrics
        self._metrics_collector.increment_counter(
            "circuit_breaker.events",
            tags={"agent_id": agent_id, "capability": capability, "event_type": event_type},
        )

        # Alert bei Circuit Breaker-Trip
        if event_type == "trip":
            await self._alert_manager.create_alert(
                alert_id=f"cb_trip_{agent_id}_{capability}_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                title=f"Circuit Breaker Trip: {agent_id}.{capability}",
                description=f"Circuit Breaker für {agent_id}.{capability} ist geöffnet",
                agent_id=agent_id,
                capability=capability,
                metric_name="circuit_breaker_trips",
                metric_value=metrics.circuit_breaker_trips,
            )

    async def record_budget_event(
        self,
        agent_id: str,
        capability: str,
        event_type: str,  # "warning", "exhausted", "deadline_exceeded"
        resource_type: str = "",
        **_,
    ):
        """Zeichnet Budget-Event auf.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            event_type: Event-Typ
            resource_type: Ressourcen-Typ
            **kwargs: Zusätzliche Event-Daten
        """
        metrics = self.get_capability_metrics(agent_id, capability)

        with self._lock:
            if event_type == "warning":
                metrics.budget_warnings += 1
            elif event_type == "exhausted":
                metrics.budget_exhaustions += 1
            elif event_type == "deadline_exceeded":
                metrics.deadline_violations += 1

        # Metrics
        self._metrics_collector.increment_counter(
            "budget.events",
            tags={
                "agent_id": agent_id,
                "capability": capability,
                "event_type": event_type,
                "resource_type": resource_type,
            },
        )

        # Alerts
        if event_type == "exhausted":
            await self._alert_manager.create_alert(
                alert_id=f"budget_exhausted_{agent_id}_{capability}_{int(time.time())}",
                severity=AlertSeverity.ERROR,
                title=f"Budget Exhausted: {agent_id}.{capability}",
                description=f"Budget für {resource_type} erschöpft",
                agent_id=agent_id,
                capability=capability,
                metric_name="budget_exhaustions",
                metric_value=metrics.budget_exhaustions,
            )
        elif event_type == "deadline_exceeded":
            await self._alert_manager.create_alert(
                alert_id=f"deadline_exceeded_{agent_id}_{capability}_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                title=f"Deadline Exceeded: {agent_id}.{capability}",
                description="Request-Deadline überschritten",
                agent_id=agent_id,
                capability=capability,
                metric_name="deadline_violations",
                metric_value=metrics.deadline_violations,
            )

    async def _check_capability_alerts(self, metrics: CapabilityMetrics):
        """Prüft Capability-Metriken auf Alert-Bedingungen.

        Args:
            metrics: Capability-Metriken
        """
        # Error-Rate-Check
        error_rate = metrics.get_error_rate()
        if error_rate > self._alert_thresholds["error_rate"]:
            await self._alert_manager.create_alert(
                alert_id=f"high_error_rate_{metrics.agent_id}_{metrics.capability}",
                severity=AlertSeverity.WARNING,
                title=f"High Error Rate: {metrics.agent_id}.{metrics.capability}",
                description=f"Error-Rate bei {error_rate:.1%}",
                agent_id=metrics.agent_id,
                capability=metrics.capability,
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=self._alert_thresholds["error_rate"],
            )

        # Response-Time-Check
        if metrics.avg_response_time > self._alert_thresholds["response_time"]:
            await self._alert_manager.create_alert(
                alert_id=f"slow_response_{metrics.agent_id}_{metrics.capability}",
                severity=AlertSeverity.WARNING,
                title=f"Slow Response: {metrics.agent_id}.{metrics.capability}",
                description=f"Durchschnittliche Response-Zeit: {metrics.avg_response_time:.2f}s",
                agent_id=metrics.agent_id,
                capability=metrics.capability,
                metric_name="avg_response_time",
                metric_value=metrics.avg_response_time,
                threshold=self._alert_thresholds["response_time"],
            )

    def start_monitoring(self):
        """Startet Performance-Monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """Stoppt Performance-Monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()

    async def _monitoring_loop(self):
        """Monitoring-Loop für kontinuierliche Überwachung."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)
                await self._update_resilience_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Performance-Monitoring: {e}")

    async def _update_resilience_metrics(self):
        """Aktualisiert Gesamte Resilience-Metriken."""
        with self._lock:
            # Circuit Breaker-Statistiken
            cb_states = defaultdict(int)
            for metrics in self._capability_metrics.values():
                cb_states[metrics.circuit_breaker_state] += 1

            self._resilience_metrics.total_circuit_breakers = len(self._capability_metrics)
            self._resilience_metrics.open_circuit_breakers = cb_states["open"]
            self._resilience_metrics.half_open_circuit_breakers = cb_states["half_open"]

            # System-Performance
            if self._capability_metrics:
                total_requests = sum(m.total_requests for m in self._capability_metrics.values())
                successful_requests = sum(
                    m.successful_requests for m in self._capability_metrics.values()
                )

                if total_requests > 0:
                    self._resilience_metrics.system_success_rate = (
                        successful_requests / total_requests
                    )

                response_times = []
                for metrics in self._capability_metrics.values():
                    if metrics.response_times:
                        response_times.extend(metrics.response_times)

                if response_times:
                    self._resilience_metrics.avg_system_response_time = statistics.mean(
                        response_times
                    )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt vollständige Metriken-Zusammenfassung.

        Returns:
            Vollständige Metriken-Zusammenfassung
        """
        with self._lock:
            return {
                "resilience": self._resilience_metrics.to_dict(),
                "capabilities": {
                    key: metrics.to_dict() for key, metrics in self._capability_metrics.items()
                },
                "upstreams": {
                    key: metrics.to_dict() for key, metrics in self._upstream_metrics.items()
                },
                "alerts": self._alert_manager.get_metrics_summary(),
            }

    def get_alert_manager(self) -> AlertManager:
        """Holt Alert-Manager.

        Returns:
            Alert-Manager
        """
        return self._alert_manager
