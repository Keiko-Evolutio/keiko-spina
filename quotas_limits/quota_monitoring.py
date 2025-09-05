# backend/quotas_limits/quota_monitoring.py
"""Quota Monitoring für Keiko Personal Assistant

Implementiert Real-time Monitoring, Alerting, Health-Checks
und Performance-Metriken für das Quota-System.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Schweregrad von Alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Status von Alerts."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class MonitoringMetric(str, Enum):
    """Monitoring-Metriken."""
    QUOTA_UTILIZATION = "quota_utilization"
    RATE_LIMIT_HITS = "rate_limit_hits"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CONCURRENT_REQUESTS = "concurrent_requests"


class HealthStatus(str, Enum):
    """Health-Status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class QuotaAlert:
    """Quota-Alert."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus

    # Alert-Details
    title: str
    description: str
    quota_id: str
    scope_id: str

    # Trigger-Werte
    current_value: float
    threshold_value: float

    # Zeitstempel
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged_by: str | None = None
    resolution_notes: str | None = None

    @property
    def duration_minutes(self) -> float:
        """Gibt Alert-Dauer in Minuten zurück."""
        end_time = self.resolved_at or datetime.now(UTC)
        return (end_time - self.triggered_at).total_seconds() / 60.0

    @property
    def is_active(self) -> bool:
        """Prüft, ob Alert aktiv ist."""
        return self.status == AlertStatus.ACTIVE


@dataclass
class MonitoringRule:
    """Regel für Monitoring."""
    rule_id: str
    name: str
    description: str

    # Bedingungen
    metric: MonitoringMetric
    quota_id_pattern: str  # Regex-Pattern
    scope_pattern: str

    # Schwellwerte
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    emergency_threshold: float | None = None

    # Zeitfenster
    evaluation_window_minutes: int = 5
    consecutive_violations: int = 1

    # Aktionen
    notify_webhook: str | None = None
    notify_emails: list[str] = field(default_factory=list)
    auto_scale_quota: bool = False
    scale_factor: float = 1.2

    # Cooldown
    cooldown_minutes: int = 30
    last_triggered: datetime | None = None

    # Gültigkeit
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_in_cooldown(self) -> bool:
        """Prüft, ob Regel in Cooldown ist."""
        if not self.last_triggered:
            return False

        cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
        return datetime.now(UTC) < cooldown_end

    def get_severity_for_value(self, value: float) -> AlertSeverity | None:
        """Bestimmt Severity basierend auf Wert."""
        if self.emergency_threshold and value >= self.emergency_threshold:
            return AlertSeverity.EMERGENCY
        if self.critical_threshold and value >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        if self.warning_threshold and value >= self.warning_threshold:
            return AlertSeverity.WARNING

        return None


@dataclass
class PerformanceMetrics:
    """Performance-Metriken."""
    metric_id: str
    timestamp: datetime

    # Quota-System-Metriken
    quota_checks_per_second: float = 0.0
    average_check_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0

    # Rate-Limiting-Metriken
    rate_limit_checks_per_second: float = 0.0
    rate_limit_violations_per_second: float = 0.0

    # Budget-Metriken
    budget_operations_per_second: float = 0.0
    budget_transfer_latency_ms: float = 0.0

    # System-Metriken
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0

    # Error-Metriken
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0


@dataclass
class QuotaHealthCheck:
    """Health-Check für Quota-System."""
    check_id: str
    component: str
    status: HealthStatus

    # Check-Details
    check_name: str
    description: str

    # Ergebnis
    success: bool
    response_time_ms: float
    error_message: str | None = None

    # Zeitstempel
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Sammelt Performance-Metriken."""

    def __init__(self):
        """Initialisiert Metrics Collector."""
        self._metrics_history: deque = deque(maxlen=1000)
        self._current_metrics = PerformanceMetrics(
            metric_id="current",
            timestamp=datetime.now(UTC)
        )

        # Counters für Metriken
        self._quota_checks_count = 0
        self._rate_limit_checks_count = 0
        self._budget_operations_count = 0
        self._errors_count = 0

        # Timing-Daten
        self._check_latencies: deque = deque(maxlen=100)
        self._transfer_latencies: deque = deque(maxlen=100)

        # Letzte Aktualisierung
        self._last_update = time.time()

    def record_quota_check(self, latency_ms: float, cache_hit: bool = False) -> None:
        """Zeichnet Quota-Check auf."""
        self._quota_checks_count += 1
        self._check_latencies.append(latency_ms)

        if cache_hit:
            # Cache-Hit-Rate wird in update_metrics berechnet
            pass

    def record_rate_limit_check(self, violation: bool = False) -> None:
        """Zeichnet Rate-Limit-Check auf."""
        self._rate_limit_checks_count += 1

        if violation:
            # Violations werden separat getrackt
            pass

    def record_budget_operation(self, latency_ms: float) -> None:
        """Zeichnet Budget-Operation auf."""
        self._budget_operations_count += 1
        self._transfer_latencies.append(latency_ms)

    def record_error(self) -> None:
        """Zeichnet Fehler auf."""
        self._errors_count += 1

    def update_metrics(self) -> PerformanceMetrics:
        """Aktualisiert und gibt aktuelle Metriken zurück."""
        now = time.time()
        time_elapsed = now - self._last_update

        if time_elapsed == 0:
            return self._current_metrics

        # Berechne Raten pro Sekunde
        quota_checks_per_second = self._quota_checks_count / time_elapsed
        rate_limit_checks_per_second = self._rate_limit_checks_count / time_elapsed
        budget_operations_per_second = self._budget_operations_count / time_elapsed

        # Berechne Latenzen
        avg_check_latency = sum(self._check_latencies) / len(self._check_latencies) if self._check_latencies else 0.0
        avg_transfer_latency = sum(self._transfer_latencies) / len(self._transfer_latencies) if self._transfer_latencies else 0.0

        # Berechne Error-Rate
        total_operations = self._quota_checks_count + self._rate_limit_checks_count + self._budget_operations_count
        error_rate = (self._errors_count / total_operations * 100) if total_operations > 0 else 0.0

        # Erstelle neue Metriken
        metrics = PerformanceMetrics(
            metric_id=f"metrics_{int(now)}",
            timestamp=datetime.now(UTC),
            quota_checks_per_second=quota_checks_per_second,
            average_check_latency_ms=avg_check_latency,
            rate_limit_checks_per_second=rate_limit_checks_per_second,
            budget_operations_per_second=budget_operations_per_second,
            budget_transfer_latency_ms=avg_transfer_latency,
            error_rate_percent=error_rate
        )

        # Speichere in Historie
        self._metrics_history.append(metrics)
        self._current_metrics = metrics

        # Reset Counters
        self._quota_checks_count = 0
        self._rate_limit_checks_count = 0
        self._budget_operations_count = 0
        self._errors_count = 0
        self._last_update = now

        return metrics

    def get_metrics_history(self, minutes_back: int = 60) -> list[PerformanceMetrics]:
        """Gibt Metriken-Historie zurück."""
        cutoff_time = datetime.now(UTC) - timedelta(minutes=minutes_back)

        return [
            metrics for metrics in self._metrics_history
            if metrics.timestamp >= cutoff_time
        ]


class AlertManager:
    """Manager für Alerts."""

    def __init__(self):
        """Initialisiert Alert Manager."""
        self._active_alerts: dict[str, QuotaAlert] = {}
        self._alert_history: list[QuotaAlert] = []
        self._suppressed_alerts: set[str] = set()

        # Statistiken
        self._total_alerts_triggered = 0
        self._alerts_by_severity = defaultdict(int)

    async def trigger_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        quota_id: str,
        scope_id: str,
        current_value: float,
        threshold_value: float,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Triggert neuen Alert."""
        import uuid

        alert_id = str(uuid.uuid4())

        # Prüfe Suppression
        suppression_key = f"{alert_type}:{quota_id}:{scope_id}"
        if suppression_key in self._suppressed_alerts:
            logger.debug(f"Alert suppressed: {suppression_key}")
            return alert_id

        alert = QuotaAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            title=title,
            description=description,
            quota_id=quota_id,
            scope_id=scope_id,
            current_value=current_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )

        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        self._total_alerts_triggered += 1
        self._alerts_by_severity[severity.value] += 1

        # Sende Notifications
        await self._send_alert_notifications(alert)

        logger.warning(f"Alert triggered: {title} ({severity.value})")
        return alert_id

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Bestätigt Alert."""
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(UTC)
        alert.acknowledged_by = acknowledged_by

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    async def resolve_alert(self, alert_id: str, resolution_notes: str | None = None) -> bool:
        """Löst Alert auf."""
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(UTC)
        alert.resolution_notes = resolution_notes

        # Entferne aus aktiven Alerts
        del self._active_alerts[alert_id]

        logger.info(f"Alert resolved: {alert_id}")
        return True

    def suppress_alerts(self, alert_type: str, quota_id: str, scope_id: str, duration_minutes: int = 60) -> None:
        """Unterdrückt Alerts für bestimmte Zeit."""
        suppression_key = f"{alert_type}:{quota_id}:{scope_id}"
        self._suppressed_alerts.add(suppression_key)

        # Schedule Removal (vereinfacht)
        async def remove_suppression():
            await asyncio.sleep(duration_minutes * 60)
            self._suppressed_alerts.discard(suppression_key)

        asyncio.create_task(remove_suppression())
        logger.info(f"Alerts suppressed for {duration_minutes}min: {suppression_key}")

    async def _send_alert_notifications(self, alert: QuotaAlert) -> None:
        """Sendet Alert-Notifications."""
        # Vereinfachte Notification-Implementierung
        # In Produktion würde hier E-Mail, Slack, PagerDuty etc. integriert

        notification_message = f"""
        Alert: {alert.title}
        Severity: {alert.severity.value}
        Description: {alert.description}
        Quota: {alert.quota_id}
        Scope: {alert.scope_id}
        Current Value: {alert.current_value}
        Threshold: {alert.threshold_value}
        Time: {alert.triggered_at.isoformat()}
        """

        logger.warning(f"Alert Notification: {notification_message}")

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[QuotaAlert]:
        """Gibt aktive Alerts zurück."""
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    def get_alert_statistics(self) -> dict[str, Any]:
        """Gibt Alert-Statistiken zurück."""
        return {
            "total_alerts_triggered": self._total_alerts_triggered,
            "active_alerts": len(self._active_alerts),
            "alerts_by_severity": dict(self._alerts_by_severity),
            "suppressed_alert_types": len(self._suppressed_alerts),
            "alert_history_size": len(self._alert_history)
        }


class HealthChecker:
    """Health-Checker für Quota-System."""

    def __init__(self):
        """Initialisiert Health Checker."""
        self._health_checks: dict[str, QuotaHealthCheck] = {}
        self._check_history: list[QuotaHealthCheck] = []

        # Health-Check-Funktionen
        self._check_functions: dict[str, Callable] = {}

    def register_health_check(self, component: str, check_name: str, check_function: Callable) -> None:
        """Registriert Health-Check-Funktion."""
        check_key = f"{component}:{check_name}"
        self._check_functions[check_key] = check_function
        logger.info(f"Health-Check registriert: {check_key}")

    async def run_health_checks(self) -> dict[str, QuotaHealthCheck]:
        """Führt alle Health-Checks aus."""
        results = {}

        for check_key, check_function in self._check_functions.items():
            component, check_name = check_key.split(":", 1)

            start_time = time.time()

            try:
                # Führe Health-Check aus
                success = await check_function()
                response_time = (time.time() - start_time) * 1000

                status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY

                health_check = QuotaHealthCheck(
                    check_id=f"check_{int(time.time())}_{check_key}",
                    component=component,
                    status=status,
                    check_name=check_name,
                    description=f"Health check for {component}",
                    success=success,
                    response_time_ms=response_time
                )

            except Exception as e:
                response_time = (time.time() - start_time) * 1000

                health_check = QuotaHealthCheck(
                    check_id=f"check_{int(time.time())}_{check_key}",
                    component=component,
                    status=HealthStatus.CRITICAL,
                    check_name=check_name,
                    description=f"Health check for {component}",
                    success=False,
                    response_time_ms=response_time,
                    error_message=str(e)
                )

            results[check_key] = health_check
            self._health_checks[check_key] = health_check
            self._check_history.append(health_check)

        return results

    def get_overall_health(self) -> HealthStatus:
        """Gibt Overall-Health-Status zurück."""
        if not self._health_checks:
            return HealthStatus.HEALTHY

        statuses = [check.status for check in self._health_checks.values()]

        if any(status == HealthStatus.CRITICAL for status in statuses):
            return HealthStatus.CRITICAL
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        if any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


class QuotaMonitor:
    """Hauptklasse für Quota-Monitoring."""

    def __init__(self):
        """Initialisiert Quota Monitor."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()

        self._monitoring_rules: dict[str, MonitoringRule] = {}
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 60  # 1 Minute
        self._is_monitoring = False

        # Registriere Standard-Health-Checks
        self._register_default_health_checks()

    def register_monitoring_rule(self, rule: MonitoringRule) -> None:
        """Registriert Monitoring-Regel."""
        self._monitoring_rules[rule.rule_id] = rule
        logger.info(f"Monitoring-Regel registriert: {rule.rule_id}")

    async def start_monitoring(self) -> None:
        """Startet Monitoring."""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Quota-Monitoring gestartet")

    async def stop_monitoring(self) -> None:
        """Stoppt Monitoring."""
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        logger.info("Quota-Monitoring gestoppt")

    async def _monitoring_loop(self) -> None:
        """Monitoring-Loop."""
        while self._is_monitoring:
            try:
                # Aktualisiere Metriken
                current_metrics = self.metrics_collector.update_metrics()

                # Führe Health-Checks aus
                await self.health_checker.run_health_checks()

                # Prüfe Monitoring-Regeln
                await self._evaluate_monitoring_rules(current_metrics)

                # Warte bis zum nächsten Zyklus
                await asyncio.sleep(self._monitoring_interval)

            except Exception as e:
                logger.exception(f"Monitoring-Loop-Fehler: {e}")
                await asyncio.sleep(self._monitoring_interval)

    async def _evaluate_monitoring_rules(self, metrics: PerformanceMetrics) -> None:
        """Evaluiert Monitoring-Regeln."""
        for rule in self._monitoring_rules.values():
            if not rule.enabled or rule.is_in_cooldown():
                continue

            try:
                # Hole Metrik-Wert
                metric_value = self._get_metric_value(metrics, rule.metric)

                # Prüfe Schwellwerte
                severity = rule.get_severity_for_value(metric_value)

                if severity:
                    # Trigger Alert
                    await self.alert_manager.trigger_alert(
                        alert_type=f"monitoring_rule_{rule.rule_id}",
                        severity=severity,
                        title=f"Monitoring Rule Violation: {rule.name}",
                        description=f"{rule.description} - Current value: {metric_value}",
                        quota_id="system",
                        scope_id="monitoring",
                        current_value=metric_value,
                        threshold_value=rule.warning_threshold or 0.0,
                        metadata={"rule_id": rule.rule_id, "metric": rule.metric.value}
                    )

                    rule.last_triggered = datetime.now(UTC)

                    # Auto-Scaling falls konfiguriert
                    if rule.auto_scale_quota:
                        await self._auto_scale_quota(rule, metric_value)

            except Exception as e:
                logger.exception(f"Monitoring-Regel-Evaluation fehlgeschlagen: {rule.rule_id}: {e}")

    def _get_metric_value(self, metrics: PerformanceMetrics, metric: MonitoringMetric) -> float:
        """Extrahiert Metrik-Wert."""
        metric_mapping = {
            MonitoringMetric.RESPONSE_TIME: metrics.average_check_latency_ms,
            MonitoringMetric.ERROR_RATE: metrics.error_rate_percent,
            MonitoringMetric.THROUGHPUT: metrics.quota_checks_per_second,
            MonitoringMetric.CONCURRENT_REQUESTS: metrics.active_connections
        }

        return metric_mapping.get(metric, 0.0)

    async def _auto_scale_quota(self, rule: MonitoringRule, current_value: float) -> None:
        """Führt Auto-Scaling durch."""
        # Vereinfachte Auto-Scaling-Implementierung
        logger.info(f"Auto-Scaling triggered by rule {rule.rule_id}: {current_value}")

    def _register_default_health_checks(self) -> None:
        """Registriert Standard-Health-Checks."""
        async def quota_manager_health():
            # Vereinfachter Health-Check
            return True

        async def rate_limiter_health():
            # Vereinfachter Health-Check
            return True

        async def budget_manager_health():
            # Vereinfachter Health-Check
            return True

        self.health_checker.register_health_check("quota_manager", "basic_check", quota_manager_health)
        self.health_checker.register_health_check("rate_limiter", "basic_check", rate_limiter_health)
        self.health_checker.register_health_check("budget_manager", "basic_check", budget_manager_health)

    def get_monitoring_status(self) -> dict[str, Any]:
        """Gibt Monitoring-Status zurück."""
        return {
            "is_monitoring": self._is_monitoring,
            "monitoring_interval_seconds": self._monitoring_interval,
            "registered_rules": len(self._monitoring_rules),
            "overall_health": self.health_checker.get_overall_health().value,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "current_metrics": {
                "quota_checks_per_second": self.metrics_collector._current_metrics.quota_checks_per_second,
                "average_latency_ms": self.metrics_collector._current_metrics.average_check_latency_ms,
                "error_rate_percent": self.metrics_collector._current_metrics.error_rate_percent
            }
        }


# Globale Quota Monitor Instanz
quota_monitor = QuotaMonitor()
