# backend/agents/common/monitoring_base.py
"""Gemeinsame Monitoring-Basis-Implementierung für Keiko Personal Assistant

Konsolidiert alle Monitoring- und Alerting-Funktionalitäten in eine einheitliche,
wiederverwendbare Basis-Klasse für Performance-Tracking und Alerting.
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)

# Konstanten für bessere Wartbarkeit
DEFAULT_MONITORING_INTERVAL = 10.0
DEFAULT_METRICS_RETENTION_SIZE = 1000
DEFAULT_ALERT_COOLDOWN = 300.0  # 5 Minuten
DEFAULT_ERROR_RATE_THRESHOLD = 0.05  # 5%
DEFAULT_RESPONSE_TIME_THRESHOLD = 5.0  # 5 Sekunden


class AlertSeverity(str, Enum):
    """Einheitliche Alert-Severity-Level."""

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
class Alert:
    """Einheitliche Alert-Struktur."""

    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float = field(default_factory=time.time)

    # Kontext-Informationen
    agent_id: str = ""
    capability: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0

    # Zusätzliche Metadaten
    tags: dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: float | None = None

    def resolve(self) -> None:
        """Markiert Alert als gelöst."""
        self.resolved = True
        self.resolved_timestamp = time.time()


@dataclass
class MetricSnapshot:
    """Snapshot einer Metrik zu einem bestimmten Zeitpunkt."""

    timestamp: float
    value: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Basis-Performance-Metriken."""

    # Request-Metriken
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Timing-Metriken
    total_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0

    # Fehler-Tracking
    error_count_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_error_time: float = 0.0

    # Zeitstempel
    first_request_time: float = 0.0
    last_request_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Berechnet Fehlerrate."""
        return 1.0 - self.success_rate

    @property
    def avg_response_time(self) -> float:
        """Berechnet durchschnittliche Response-Zeit."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    @property
    def requests_per_second(self) -> float:
        """Berechnet Requests pro Sekunde."""
        if self.first_request_time == 0 or self.last_request_time == 0:
            return 0.0

        duration = self.last_request_time - self.first_request_time
        if duration <= 0:
            return 0.0

        return self.total_requests / duration


class BaseAlertManager:
    """Basis-Klasse für Alert-Management."""

    def __init__(self):
        """Initialisiert Alert-Manager."""
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: deque[Alert] = deque(maxlen=DEFAULT_METRICS_RETENTION_SIZE)
        self._alert_callbacks: dict[AlertSeverity, list[Callable[[Alert], Any]]] = defaultdict(list)
        self._alert_cooldowns: dict[str, float] = {}
        self._metrics_collector = MetricsCollector()

    async def create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        agent_id: str = "",
        capability: str = "",
        metric_name: str = "",
        metric_value: float = 0.0,
        threshold: float = 0.0,
        tags: dict[str, str] | None = None
    ) -> Alert:
        """Erstellt neuen Alert.
        
        Args:
            alert_id: Eindeutige Alert-ID
            severity: Alert-Severity
            title: Alert-Titel
            description: Alert-Beschreibung
            agent_id: Betroffener Agent
            capability: Betroffene Capability
            metric_name: Metrik-Name
            metric_value: Aktueller Metrik-Wert
            threshold: Überschrittener Threshold
            tags: Zusätzliche Tags
            
        Returns:
            Erstellter Alert
        """
        # Prüfe Cooldown
        if self._is_in_cooldown(alert_id):
            logger.debug(f"Alert '{alert_id}' ist noch im Cooldown")
            return self._active_alerts.get(alert_id)

        # Erstelle Alert
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            agent_id=agent_id,
            capability=capability,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            tags=tags or {}
        )

        # Speichere Alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        self._alert_cooldowns[alert_id] = time.time()

        # Metrics
        self._metrics_collector.increment_counter(
            "alerts.created",
            tags={
                "severity": severity.value,
                "agent_id": agent_id,
                "capability": capability,
                "metric_name": metric_name
            }
        )

        # Callbacks ausführen
        await self._execute_alert_callbacks(alert)

        logger.info(f"Alert erstellt: {severity.value} - {title}")
        return alert

    async def resolve_alert(self, alert_id: str) -> bool:
        """Löst Alert auf.
        
        Args:
            alert_id: Alert-ID
            
        Returns:
            True wenn Alert gelöst wurde
        """
        if alert_id not in self._active_alerts:
            return False

        alert = self._active_alerts[alert_id]
        alert.resolve()
        del self._active_alerts[alert_id]

        # Metrics
        self._metrics_collector.increment_counter(
            "alerts.resolved",
            tags={
                "severity": alert.severity.value,
                "agent_id": alert.agent_id,
                "capability": alert.capability
            }
        )

        logger.info(f"Alert gelöst: {alert.title}")
        return True

    def register_alert_callback(
        self,
        severity: AlertSeverity,
        callback: Callable[[Alert], Any]
    ) -> None:
        """Registriert Alert-Callback.
        
        Args:
            severity: Alert-Severity für Callback
            callback: Callback-Funktion
        """
        self._alert_callbacks[severity].append(callback)

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Gibt aktive Alerts zurück.
        
        Args:
            severity: Optional Severity-Filter
            
        Returns:
            Liste aktiver Alerts
        """
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Gibt Alert-Historie zurück.
        
        Args:
            limit: Maximale Anzahl Alerts
            
        Returns:
            Liste historischer Alerts
        """
        return list(self._alert_history)[-limit:]

    def _is_in_cooldown(self, alert_id: str) -> bool:
        """Prüft ob Alert im Cooldown ist."""
        if alert_id not in self._alert_cooldowns:
            return False

        last_alert_time = self._alert_cooldowns[alert_id]
        return (time.time() - last_alert_time) < DEFAULT_ALERT_COOLDOWN

    async def _execute_alert_callbacks(self, alert: Alert) -> None:
        """Führt Alert-Callbacks aus."""
        callbacks = self._alert_callbacks.get(alert.severity, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Fehler beim Ausführen von Alert-Callback: {e}")


class BasePerformanceMonitor:
    """Basis-Klasse für Performance-Monitoring."""

    def __init__(self):
        """Initialisiert Performance-Monitor."""
        self._metrics: dict[str, PerformanceMetrics] = {}
        self._metric_snapshots: dict[str, deque[MetricSnapshot]] = defaultdict(
            lambda: deque(maxlen=DEFAULT_METRICS_RETENTION_SIZE)
        )
        self._alert_manager = BaseAlertManager()
        self._metrics_collector = MetricsCollector()

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = DEFAULT_MONITORING_INTERVAL
        self._running = False

        # Alert-Thresholds
        self._alert_thresholds = {
            "error_rate": DEFAULT_ERROR_RATE_THRESHOLD,
            "response_time": DEFAULT_RESPONSE_TIME_THRESHOLD,
        }

    async def record_request(
        self,
        identifier: str,
        success: bool,
        response_time: float,
        error_type: str | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Zeichnet Request-Metrik auf.
        
        Args:
            identifier: Eindeutiger Identifier (z.B. "agent_id.capability")
            success: Ob Request erfolgreich war
            response_time: Response-Zeit in Sekunden
            error_type: Typ des Fehlers (falls vorhanden)
            tags: Zusätzliche Tags
        """
        # Hole oder erstelle Metriken
        if identifier not in self._metrics:
            self._metrics[identifier] = PerformanceMetrics()

        metrics = self._metrics[identifier]
        current_time = time.time()

        # Aktualisiere Metriken
        metrics.total_requests += 1
        metrics.last_request_time = current_time

        if metrics.first_request_time == 0:
            metrics.first_request_time = current_time

        if success:
            metrics.successful_requests += 1
            metrics.total_response_time += response_time
            metrics.min_response_time = min(metrics.min_response_time, response_time)
            metrics.max_response_time = max(metrics.max_response_time, response_time)
        else:
            metrics.failed_requests += 1
            metrics.last_error_time = current_time

            if error_type:
                metrics.error_count_by_type[error_type] += 1

        # Snapshot erstellen
        snapshot = MetricSnapshot(
            timestamp=current_time,
            value=response_time,
            tags=tags or {}
        )
        self._metric_snapshots[identifier].append(snapshot)

        # Metrics an Collector senden
        self._metrics_collector.record_histogram(
            "performance.response_time",
            response_time,
            tags={
                "identifier": identifier,
                "success": str(success),
                **(tags or {})
            }
        )

        # Alert-Prüfung
        await self._check_alerts(identifier, metrics)

    async def _check_alerts(self, identifier: str, metrics: PerformanceMetrics) -> None:
        """Prüft ob Alerts erstellt werden müssen."""
        # Error-Rate-Check
        if metrics.error_rate > self._alert_thresholds["error_rate"]:
            await self._alert_manager.create_alert(
                alert_id=f"high_error_rate_{identifier}",
                severity=AlertSeverity.WARNING,
                title=f"Hohe Fehlerrate: {identifier}",
                description=f"Fehlerrate: {metrics.error_rate:.2%}",
                metric_name="error_rate",
                metric_value=metrics.error_rate,
                threshold=self._alert_thresholds["error_rate"]
            )

        # Response-Time-Check
        if metrics.avg_response_time > self._alert_thresholds["response_time"]:
            await self._alert_manager.create_alert(
                alert_id=f"slow_response_{identifier}",
                severity=AlertSeverity.WARNING,
                title=f"Langsame Response: {identifier}",
                description=f"Durchschnittliche Response-Zeit: {metrics.avg_response_time:.2f}s",
                metric_name="avg_response_time",
                metric_value=metrics.avg_response_time,
                threshold=self._alert_thresholds["response_time"]
            )

    def get_metrics(self, identifier: str) -> PerformanceMetrics | None:
        """Gibt Metriken für Identifier zurück."""
        return self._metrics.get(identifier)

    def get_all_metrics(self) -> dict[str, PerformanceMetrics]:
        """Gibt alle Metriken zurück."""
        return self._metrics.copy()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Gibt Metriken-Zusammenfassung zurück."""
        return {
            "total_identifiers": len(self._metrics),
            "total_requests": sum(m.total_requests for m in self._metrics.values()),
            "total_successful_requests": sum(m.successful_requests for m in self._metrics.values()),
            "total_failed_requests": sum(m.failed_requests for m in self._metrics.values()),
            "overall_success_rate": self._calculate_overall_success_rate(),
            "active_alerts": len(self._alert_manager.get_active_alerts()),
            "identifiers": {
                identifier: {
                    "total_requests": metrics.total_requests,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "requests_per_second": metrics.requests_per_second
                }
                for identifier, metrics in self._metrics.items()
            }
        }

    def _calculate_overall_success_rate(self) -> float:
        """Berechnet Gesamt-Erfolgsrate."""
        total_requests = sum(m.total_requests for m in self._metrics.values())
        total_successful = sum(m.successful_requests for m in self._metrics.values())

        if total_requests == 0:
            return 1.0

        return total_successful / total_requests

    def start_monitoring(self) -> None:
        """Startet Performance-Monitoring."""
        if not self._running:
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self) -> None:
        """Stoppt Performance-Monitoring."""
        self._running = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()

    async def _monitoring_loop(self) -> None:
        """Monitoring-Loop für kontinuierliche Überwachung."""
        while self._running:
            try:
                await asyncio.sleep(self._monitoring_interval)
                await self._update_monitoring_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Performance-Monitoring: {e}")

    async def _update_monitoring_metrics(self) -> None:
        """Aktualisiert Monitoring-Metriken."""
        # Sende aggregierte Metriken an Collector
        for identifier, metrics in self._metrics.items():
            self._metrics_collector.record_gauge(
                "performance.success_rate",
                metrics.success_rate,
                tags={"identifier": identifier}
            )

            self._metrics_collector.record_gauge(
                "performance.requests_per_second",
                metrics.requests_per_second,
                tags={"identifier": identifier}
            )
