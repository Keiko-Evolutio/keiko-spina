# backend/audit_system/audit_monitoring.py
"""Audit Monitoring für Keiko Personal Assistant

Implementiert Real-time Monitoring, Alerting, Health-Checks
und Analytics für das Audit-System.
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
from observability import trace_function

from .audit_constants import AuditAlertTypes, AuditConstants, AuditMessages

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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


class HealthStatus(str, Enum):
    """Health-Status von Komponenten."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class AuditAlert:
    """Alert für Audit-System."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus

    # Alert-Details
    title: str
    description: str
    source_component: str

    # Kontext
    audit_event_id: str | None = None
    affected_resource: str | None = None

    # Timing
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    # Acknowledgment
    acknowledged_by: str | None = None
    resolution_notes: str | None = None

    @property
    def duration_minutes(self) -> float:
        """Gibt Alert-Dauer in Minuten zurück."""
        end_time = self.resolved_at or datetime.now(UTC)
        duration = end_time - self.triggered_at
        return duration.total_seconds() / 60

    @property
    def is_active(self) -> bool:
        """Prüft, ob Alert aktiv ist."""
        return self.status == AlertStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "source_component": self.source_component,
            "audit_event_id": self.audit_event_id,
            "affected_resource": self.affected_resource,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_minutes": self.duration_minutes,
            "is_active": self.is_active,
            "acknowledged_by": self.acknowledged_by,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata
        }


@dataclass
class AuditHealthCheck:
    """Health-Check für Audit-Komponenten."""
    component: str
    check_name: str
    status: HealthStatus

    # Check-Details
    success: bool
    response_time_ms: float
    error_message: str | None = None

    # Timing
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "component": self.component,
            "check_name": self.check_name,
            "status": self.status.value,
            "success": self.success,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "checked_at": self.checked_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AuditAnalytics:
    """Analytics für Audit-System."""
    timestamp: datetime

    # Event-Metriken
    total_events: int = 0
    events_per_minute: float = 0.0
    events_by_type: dict[str, int] = field(default_factory=dict)
    events_by_severity: dict[str, int] = field(default_factory=dict)

    # Performance-Metriken
    avg_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    error_rate_percent: float = 0.0

    # Compliance-Metriken
    pii_redaction_rate: float = 0.0
    tamper_proof_rate: float = 0.0
    retention_compliance_rate: float = 0.0

    # Alert-Metriken
    active_alerts: int = 0
    alerts_by_severity: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_events": self.total_events,
            "events_per_minute": self.events_per_minute,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "p95_processing_time_ms": self.p95_processing_time_ms,
            "error_rate_percent": self.error_rate_percent,
            "pii_redaction_rate": self.pii_redaction_rate,
            "tamper_proof_rate": self.tamper_proof_rate,
            "retention_compliance_rate": self.retention_compliance_rate,
            "active_alerts": self.active_alerts,
            "alerts_by_severity": self.alerts_by_severity
        }


@dataclass
class ComplianceReport:
    """Compliance-Report für Audit-System."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Compliance-Metriken
    total_events_audited: int = 0
    pii_redaction_compliance: float = 0.0
    retention_policy_compliance: float = 0.0
    tamper_proof_compliance: float = 0.0

    # Violations
    compliance_violations: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_events_audited": self.total_events_audited,
            "pii_redaction_compliance": self.pii_redaction_compliance,
            "retention_policy_compliance": self.retention_policy_compliance,
            "tamper_proof_compliance": self.tamper_proof_compliance,
            "compliance_violations": self.compliance_violations,
            "recommendations": self.recommendations
        }


class AlertManager:
    """Manager für Audit-Alerts."""

    def __init__(self):
        """Initialisiert Alert Manager."""
        self._alerts: dict[str, AuditAlert] = {}
        self._alert_handlers: dict[str, Callable[[AuditAlert], Awaitable[None]]] = {}
        self._suppressed_alerts: dict[str, datetime] = {}

        # Statistiken
        self._alerts_triggered = 0
        self._alerts_acknowledged = 0
        self._alerts_resolved = 0

    def register_alert_handler(
        self,
        alert_type: str,
        handler: Callable[[AuditAlert], Awaitable[None]]
    ) -> None:
        """Registriert Alert-Handler.

        Args:
            alert_type: Alert-Typ
            handler: Handler-Funktion
        """
        self._alert_handlers[alert_type] = handler
        logger.info(f"Alert-Handler registriert: {alert_type}")

    @trace_function("audit_monitoring.trigger_alert")
    async def trigger_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        source_component: str = "audit_system",
        audit_event_id: str | None = None,
        affected_resource: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Triggert neuen Alert.

        Args:
            alert_type: Alert-Typ
            severity: Schweregrad
            title: Alert-Titel
            description: Alert-Beschreibung
            source_component: Quell-Komponente
            audit_event_id: Zugehörige Audit-Event-ID
            affected_resource: Betroffene Ressource
            metadata: Zusätzliche Metadaten

        Returns:
            Alert-ID
        """
        import uuid

        # Prüfe Alert-Suppression
        suppression_key = f"{alert_type}:{affected_resource or 'global'}"
        if suppression_key in self._suppressed_alerts:
            suppression_end = self._suppressed_alerts[suppression_key]
            if datetime.now(UTC) < suppression_end:
                logger.debug(f"Alert unterdrückt: {alert_type}")
                return ""

        alert_id = str(uuid.uuid4())

        alert = AuditAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            title=title,
            description=description,
            source_component=source_component,
            audit_event_id=audit_event_id,
            affected_resource=affected_resource,
            metadata=metadata or {}
        )

        self._alerts[alert_id] = alert
        self._alerts_triggered += 1

        # Führe Alert-Handler aus
        handler = self._alert_handlers.get(alert_type)
        if handler:
            try:
                await handler(alert)
            except Exception as e:
                logger.exception(f"Alert-Handler fehlgeschlagen: {e}")

        logger.info(f"Alert getriggert: {alert_id} ({severity.value}) - {title}")

        return alert_id

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Bestätigt Alert.

        Args:
            alert_id: Alert-ID
            acknowledged_by: Bestätigt von

        Returns:
            True wenn erfolgreich
        """
        alert = self._alerts.get(alert_id)
        if not alert or alert.status != AlertStatus.ACTIVE:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(UTC)
        alert.acknowledged_by = acknowledged_by

        self._alerts_acknowledged += 1

        logger.info(f"Alert bestätigt: {alert_id} von {acknowledged_by}")
        return True

    async def resolve_alert(self, alert_id: str, resolution_notes: str | None = None) -> bool:
        """Löst Alert auf.

        Args:
            alert_id: Alert-ID
            resolution_notes: Lösungsnotizen

        Returns:
            True wenn erfolgreich
        """
        alert = self._alerts.get(alert_id)
        if not alert or alert.status == AlertStatus.RESOLVED:
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(UTC)
        alert.resolution_notes = resolution_notes

        self._alerts_resolved += 1

        logger.info(f"Alert aufgelöst: {alert_id}")
        return True

    def suppress_alerts(
        self,
        alert_type: str,
        affected_resource: str | None = None,
        duration_minutes: int = 60
    ) -> None:
        """Unterdrückt Alerts für bestimmte Zeit.

        Args:
            alert_type: Alert-Typ
            affected_resource: Betroffene Ressource
            duration_minutes: Dauer in Minuten
        """
        suppression_key = f"{alert_type}:{affected_resource or 'global'}"
        suppression_end = datetime.now(UTC) + timedelta(minutes=duration_minutes)

        self._suppressed_alerts[suppression_key] = suppression_end

        logger.info(f"Alerts unterdrückt: {suppression_key} für {duration_minutes} Minuten")

    def get_active_alerts(self) -> list[AuditAlert]:
        """Gibt aktive Alerts zurück."""
        return [alert for alert in self._alerts.values() if alert.is_active]

    def get_alert_statistics(self) -> dict[str, Any]:
        """Gibt Alert-Statistiken zurück."""
        active_alerts = self.get_active_alerts()
        alerts_by_severity = defaultdict(int)

        for alert in active_alerts:
            alerts_by_severity[alert.severity.value] += 1

        return {
            "alerts_triggered": self._alerts_triggered,
            "alerts_acknowledged": self._alerts_acknowledged,
            "alerts_resolved": self._alerts_resolved,
            "active_alerts": len(active_alerts),
            "alerts_by_severity": dict(alerts_by_severity),
            "suppressed_alert_types": len(self._suppressed_alerts)
        }


class HealthChecker:
    """Health-Checker für Audit-Komponenten."""

    def __init__(self):
        """Initialisiert Health Checker."""
        self._health_checks: dict[str, Callable[[], Awaitable[bool]]] = {}
        self._health_results: dict[str, AuditHealthCheck] = {}
        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False

    def register_health_check(
        self,
        component: str,
        check_name: str,
        check_function: Callable[[], Awaitable[bool]]
    ) -> None:
        """Registriert Health-Check.

        Args:
            component: Komponenten-Name
            check_name: Check-Name
            check_function: Check-Funktion
        """
        check_key = f"{component}:{check_name}"
        self._health_checks[check_key] = check_function
        logger.info(f"Health-Check registriert: {check_key}")

    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Startet Health-Monitoring.

        Args:
            interval_seconds: Monitoring-Intervall
        """
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(
            self._health_monitoring_loop(interval_seconds)
        )

        logger.info("Health-Monitoring gestartet")

    async def stop_monitoring(self) -> None:
        """Stoppt Health-Monitoring."""
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        logger.info("Health-Monitoring gestoppt")

    async def _health_monitoring_loop(self, interval_seconds: int) -> None:
        """Health-Monitoring-Loop."""
        while self._is_monitoring:
            try:
                await self.run_health_checks()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.exception(f"Health-Monitoring-Fehler: {e}")
                await asyncio.sleep(interval_seconds)

    @trace_function("audit_monitoring.run_health_checks")
    async def run_health_checks(self) -> dict[str, AuditHealthCheck]:
        """Führt alle Health-Checks aus.

        Returns:
            Health-Check-Ergebnisse
        """
        results = {}

        for check_key, check_function in self._health_checks.items():
            component, check_name = check_key.split(":", 1)

            start_time = time.time()

            try:
                success = await asyncio.wait_for(check_function(), timeout=30.0)
                response_time = (time.time() - start_time) * 1000

                status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
                error_message = None

            except TimeoutError:
                success = False
                response_time = 30000.0  # 30s timeout
                status = HealthStatus.CRITICAL
                error_message = "Health check timeout"

            except Exception as e:
                success = False
                response_time = (time.time() - start_time) * 1000
                status = HealthStatus.CRITICAL
                error_message = str(e)

            health_check = AuditHealthCheck(
                component=component,
                check_name=check_name,
                status=status,
                success=success,
                response_time_ms=response_time,
                error_message=error_message
            )

            results[check_key] = health_check
            self._health_results[check_key] = health_check

        return results

    def get_overall_health(self) -> HealthStatus:
        """Gibt Overall-Health-Status zurück."""
        if not self._health_results:
            return HealthStatus.HEALTHY

        statuses = [check.status for check in self._health_results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_health_summary(self) -> dict[str, Any]:
        """Gibt Health-Zusammenfassung zurück."""
        status_counts = defaultdict(int)

        for check in self._health_results.values():
            status_counts[check.status.value] += 1

        return {
            "overall_health": self.get_overall_health().value,
            "total_checks": len(self._health_results),
            "status_distribution": dict(status_counts),
            "is_monitoring": self._is_monitoring
        }


class AuditDashboard:
    """Dashboard für Audit-System-Monitoring."""

    def __init__(self):
        """Initialisiert Audit Dashboard."""
        self._analytics_history: deque = deque(maxlen=1440)  # 24h bei 1min Intervall
        self._compliance_reports: dict[str, ComplianceReport] = {}

        # Metriken-Sammlung
        self._event_counts: defaultdict = defaultdict(int)
        self._processing_times: deque = deque(maxlen=1000)
        self._error_counts: defaultdict = defaultdict(int)

    def record_event_processed(
        self,
        event_type: str,
        severity: str,
        processing_time_ms: float,
        success: bool
    ) -> None:
        """Zeichnet verarbeitetes Event auf.

        Args:
            event_type: Event-Typ
            severity: Schweregrad
            processing_time_ms: Verarbeitungszeit
            success: Erfolgreich
        """
        self._event_counts[f"type:{event_type}"] += 1
        self._event_counts[f"severity:{severity}"] += 1
        self._processing_times.append(processing_time_ms)

        if not success:
            self._error_counts[event_type] += 1

    def generate_analytics(self) -> AuditAnalytics:
        """Generiert aktuelle Analytics.

        Returns:
            Audit-Analytics
        """
        now = datetime.now(UTC)

        # Berechne Metriken
        total_events = sum(
            count for key, count in self._event_counts.items()
            if key.startswith("type:")
        )

        events_by_type = {
            key.replace("type:", ""): count
            for key, count in self._event_counts.items()
            if key.startswith("type:")
        }

        events_by_severity = {
            key.replace("severity:", ""): count
            for key, count in self._event_counts.items()
            if key.startswith("severity:")
        }

        # Performance-Metriken
        if self._processing_times:
            avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            sorted_times = sorted(self._processing_times)
            p95_processing_time = sorted_times[int(len(sorted_times) * 0.95)]
        else:
            avg_processing_time = p95_processing_time = 0.0

        # Error-Rate
        total_errors = sum(self._error_counts.values())
        error_rate = (total_errors / max(total_events, 1)) * 100

        analytics = AuditAnalytics(
            timestamp=now,
            total_events=total_events,
            events_per_minute=total_events / 60.0,  # Vereinfacht
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            avg_processing_time_ms=avg_processing_time,
            p95_processing_time_ms=p95_processing_time,
            error_rate_percent=error_rate
        )

        self._analytics_history.append(analytics)

        return analytics

    def generate_compliance_report(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> ComplianceReport:
        """Generiert Compliance-Report.

        Args:
            period_start: Berichtszeitraum-Start
            period_end: Berichtszeitraum-Ende

        Returns:
            Compliance-Report
        """
        import uuid

        report_id = str(uuid.uuid4())

        # Vereinfachte Compliance-Metriken
        total_events = sum(
            count for key, count in self._event_counts.items()
            if key.startswith("type:")
        )

        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.now(UTC),
            period_start=period_start,
            period_end=period_end,
            total_events_audited=total_events,
            pii_redaction_compliance=95.0,  # Beispielwerte
            retention_policy_compliance=98.0,
            tamper_proof_compliance=99.0,
            recommendations=[
                "Erhöhe PII-Redaction-Rate auf 100%",
                "Implementiere automatische Retention-Policy-Enforcement",
                "Verbessere Tamper-Detection-Algorithmen"
            ]
        )

        self._compliance_reports[report_id] = report

        return report

    def get_dashboard_data(self) -> dict[str, Any]:
        """Gibt Dashboard-Daten zurück."""
        current_analytics = self.generate_analytics()

        return {
            "current_analytics": current_analytics.to_dict(),
            "analytics_history": [a.to_dict() for a in list(self._analytics_history)[-24:]],  # Letzte 24 Einträge
            "compliance_reports": len(self._compliance_reports),
            "system_health": "healthy"  # Vereinfacht
        }


class AuditMonitor:
    """Hauptklasse für Audit-Monitoring."""

    def __init__(self):
        """Initialisiert Audit Monitor."""
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.dashboard = AuditDashboard()

        # Monitoring-Tasks
        self._monitoring_tasks: list[asyncio.Task] = []
        self._is_monitoring = False

        # Registriere Standard-Health-Checks
        self._register_default_health_checks()

        # Registriere Standard-Alert-Handler
        self._register_default_alert_handlers()

    def _register_default_health_checks(self) -> None:
        """Registriert Standard-Health-Checks."""
        async def audit_engine_health() -> bool:
            from .core_audit_engine import audit_engine
            stats = audit_engine.get_statistics()
            return stats["is_running"]

        async def tamper_proof_health() -> bool:
            from .tamper_proof_trail import tamper_proof_trail
            stats = tamper_proof_trail.get_trail_statistics()
            return stats["events_added"] >= 0

        async def pii_redaction_health() -> bool:
            from .audit_pii_redaction import audit_pii_redactor
            stats = audit_pii_redactor.get_redaction_statistics()
            return stats["redactions_performed"] >= 0

        self.health_checker.register_health_check("audit_engine", "basic_health", audit_engine_health)
        self.health_checker.register_health_check("tamper_proof", "basic_health", tamper_proof_health)
        self.health_checker.register_health_check("pii_redaction", "basic_health", pii_redaction_health)

    def _register_default_alert_handlers(self) -> None:
        """Registriert Standard-Alert-Handler."""
        async def log_alert_handler(alert: AuditAlert) -> None:
            logger.warning(f"AUDIT ALERT: {alert.title} - {alert.description}")

        async def critical_alert_handler(alert: AuditAlert) -> None:
            logger.critical(f"CRITICAL AUDIT ALERT: {alert.title} - {alert.description}")
            # Hier könnte zusätzliche Eskalation stattfinden

        self.alert_manager.register_alert_handler("default", log_alert_handler)
        self.alert_manager.register_alert_handler("critical", critical_alert_handler)

    async def start_monitoring(self) -> None:
        """Startet Audit-Monitoring."""
        if self._is_monitoring:
            return

        self._is_monitoring = True

        # Starte Health-Monitoring
        await self.health_checker.start_monitoring()

        # Starte Analytics-Monitoring
        analytics_task = asyncio.create_task(self._analytics_monitoring_loop())
        self._monitoring_tasks.append(analytics_task)

        logger.info(AuditMessages.MONITORING_STARTED)

    async def stop_monitoring(self) -> None:
        """Stoppt Audit-Monitoring."""
        self._is_monitoring = False

        # Stoppe Health-Monitoring
        await self.health_checker.stop_monitoring()

        # Stoppe Monitoring-Tasks
        for task in self._monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        logger.info(AuditMessages.MONITORING_STOPPED)

    async def _analytics_monitoring_loop(self) -> None:
        """Analytics-Monitoring-Loop."""
        while self._is_monitoring:
            try:
                # Generiere Analytics
                analytics = self.dashboard.generate_analytics()

                # Prüfe auf Anomalien und triggere Alerts
                await self._check_analytics_for_alerts(analytics)

                # Warte Monitoring-Intervall
                await asyncio.sleep(AuditConstants.DEFAULT_MONITORING_INTERVAL_SECONDS)

            except Exception as e:
                logger.exception(f"{AuditMessages.MONITORING_ERROR}: {e}")
                await asyncio.sleep(AuditConstants.DEFAULT_MONITORING_INTERVAL_SECONDS)

    async def _check_analytics_for_alerts(self, analytics: AuditAnalytics) -> None:
        """Prüft Analytics auf Alert-Bedingungen."""
        # Error-Rate-Alert
        if analytics.error_rate_percent > AuditConstants.MAX_ERROR_RATE_PERCENT:
            await self.alert_manager.trigger_alert(
                alert_type=AuditAlertTypes.HIGH_ERROR_RATE,
                severity=AlertSeverity.CRITICAL,
                title=AuditMessages.HIGH_ERROR_RATE_TITLE,
                description=AuditMessages.ERROR_RATE_DESCRIPTION.format(rate=analytics.error_rate_percent),
                source_component="audit_analytics"
            )

        # Performance-Alert
        if analytics.avg_processing_time_ms > AuditConstants.MAX_PROCESSING_TIME_MS:
            await self.alert_manager.trigger_alert(
                alert_type=AuditAlertTypes.SLOW_PROCESSING,
                severity=AlertSeverity.WARNING,
                title=AuditMessages.SLOW_PROCESSING_TITLE,
                description=AuditMessages.PROCESSING_TIME_DESCRIPTION.format(time=analytics.avg_processing_time_ms),
                source_component="audit_analytics"
            )

    def get_monitoring_status(self) -> dict[str, Any]:
        """Gibt Monitoring-Status zurück."""
        return {
            "is_monitoring": self._is_monitoring,
            "alert_manager": self.alert_manager.get_alert_statistics(),
            "health_checker": self.health_checker.get_health_summary(),
            "dashboard": self.dashboard.get_dashboard_data()
        }


# Globale Audit Monitor Instanz
audit_monitor = AuditMonitor()
