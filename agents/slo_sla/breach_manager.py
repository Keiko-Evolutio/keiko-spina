# backend/agents/slo_sla/breach_manager.py
"""SLA Breach Management System

Implementiert SLA-Breach-Detection, automatische Escalation-Workflows
und Recovery-Tracking mit AlertManager-Integration.
"""

import asyncio
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

from ..resilience.performance_monitor import AlertManager, AlertSeverity
from .models import SLABreach, SLADefinition, SLAMetrics, ViolationSeverity

logger = get_logger(__name__)


class EscalationLevel(str, Enum):
    """Escalation-Level für SLA-Breaches."""

    LEVEL_0 = "level_0"  # Initial Alert
    LEVEL_1 = "level_1"  # Team Lead
    LEVEL_2 = "level_2"  # Manager
    LEVEL_3 = "level_3"  # Director
    LEVEL_4 = "level_4"  # Executive


@dataclass
class BreachNotification:
    """Notification für SLA-Breach."""

    breach_id: str
    sla_name: str
    severity: ViolationSeverity
    escalation_level: EscalationLevel

    # Notification-Details
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)

    # Recipients
    recipients: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)  # email, slack, pagerduty, etc.

    # Status
    sent: bool = False
    sent_at: float | None = None
    acknowledged: bool = False
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "breach_id": self.breach_id,
            "sla_name": self.sla_name,
            "severity": self.severity.value,
            "escalation_level": self.escalation_level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "recipients": self.recipients,
            "channels": self.channels,
            "sent": self.sent,
            "sent_at": self.sent_at,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class EscalationWorkflow:
    """Escalation-Workflow für SLA-Breaches."""

    sla_name: str

    # Escalation-Konfiguration
    escalation_delays: dict[EscalationLevel, float] = field(
        default_factory=lambda: {
            EscalationLevel.LEVEL_0: 0.0,  # Sofort
            EscalationLevel.LEVEL_1: 300.0,  # 5 Minuten
            EscalationLevel.LEVEL_2: 900.0,  # 15 Minuten
            EscalationLevel.LEVEL_3: 1800.0,  # 30 Minuten
            EscalationLevel.LEVEL_4: 3600.0,  # 1 Stunde
        }
    )

    # Kontakte pro Escalation-Level
    escalation_contacts: dict[EscalationLevel, list[str]] = field(default_factory=dict)

    # Notification-Channels pro Level
    escalation_channels: dict[EscalationLevel, list[str]] = field(
        default_factory=lambda: {
            EscalationLevel.LEVEL_0: ["email"],
            EscalationLevel.LEVEL_1: ["email", "slack"],
            EscalationLevel.LEVEL_2: ["email", "slack", "pagerduty"],
            EscalationLevel.LEVEL_3: ["email", "slack", "pagerduty", "phone"],
            EscalationLevel.LEVEL_4: ["email", "slack", "pagerduty", "phone", "sms"],
        }
    )

    # Auto-Escalation-Einstellungen
    auto_escalation_enabled: bool = True
    max_escalation_level: EscalationLevel = EscalationLevel.LEVEL_4

    def get_escalation_delay(self, level: EscalationLevel) -> float:
        """Holt Escalation-Delay für Level."""
        return self.escalation_delays.get(level, 0.0)

    def get_escalation_contacts(self, level: EscalationLevel) -> list[str]:
        """Holt Escalation-Kontakte für Level."""
        return self.escalation_contacts.get(level, [])

    def get_escalation_channels(self, level: EscalationLevel) -> list[str]:
        """Holt Escalation-Channels für Level."""
        return self.escalation_channels.get(level, ["email"])

    def get_next_escalation_level(
        self, current_level: EscalationLevel
    ) -> EscalationLevel | None:
        """Holt nächstes Escalation-Level."""
        levels = list(EscalationLevel)
        try:
            current_index = levels.index(current_level)
            if current_index < len(levels) - 1:
                next_level = levels[current_index + 1]
                if levels.index(next_level) <= levels.index(self.max_escalation_level):
                    return next_level
        except ValueError:
            pass
        return None


class RecoveryTracker:
    """Tracker für SLA-Recovery-Metriken."""

    def __init__(self):
        """Initialisiert Recovery-Tracker."""
        self.recovery_times: dict[str, list[float]] = defaultdict(list)
        self.mttr_cache: dict[str, float] = {}
        self._lock = threading.RLock()

    def record_recovery(self, sla_name: str, recovery_time: float):
        """Zeichnet SLA-Recovery auf.

        Args:
            sla_name: SLA-Name
            recovery_time: Recovery-Zeit in Sekunden
        """
        with self._lock:
            self.recovery_times[sla_name].append(recovery_time)

            # Behalte nur letzte 100 Recovery-Times
            if len(self.recovery_times[sla_name]) > 100:
                self.recovery_times[sla_name] = self.recovery_times[sla_name][-100:]

            # Invalidiere MTTR-Cache
            if sla_name in self.mttr_cache:
                del self.mttr_cache[sla_name]

    def get_mttr(self, sla_name: str) -> float:
        """Berechnet Mean Time To Recovery für SLA.

        Args:
            sla_name: SLA-Name

        Returns:
            MTTR in Sekunden
        """
        with self._lock:
            if sla_name in self.mttr_cache:
                return self.mttr_cache[sla_name]

            recovery_times = self.recovery_times.get(sla_name, [])
            if not recovery_times:
                return 0.0

            mttr = sum(recovery_times) / len(recovery_times)
            self.mttr_cache[sla_name] = mttr
            return mttr

    def get_recovery_statistics(self, sla_name: str) -> dict[str, Any]:
        """Holt Recovery-Statistiken für SLA.

        Args:
            sla_name: SLA-Name

        Returns:
            Recovery-Statistiken
        """
        with self._lock:
            recovery_times = self.recovery_times.get(sla_name, [])

            if not recovery_times:
                return {
                    "mttr": 0.0,
                    "min_recovery_time": 0.0,
                    "max_recovery_time": 0.0,
                    "total_recoveries": 0,
                }

            return {
                "mttr": self.get_mttr(sla_name),
                "min_recovery_time": min(recovery_times),
                "max_recovery_time": max(recovery_times),
                "total_recoveries": len(recovery_times),
            }


class SLABreachManager:
    """Manager für SLA-Breach-Detection und -Handling."""

    def __init__(self, alert_manager: AlertManager):
        """Initialisiert SLA-Breach-Manager.

        Args:
            alert_manager: Alert-Manager für Notifications
        """
        self.alert_manager = alert_manager

        # Breach-Tracking
        self.active_breaches: dict[str, SLABreach] = {}
        self.breach_history: list[SLABreach] = []

        # Escalation-Workflows
        self.escalation_workflows: dict[str, EscalationWorkflow] = {}

        # Recovery-Tracking
        self.recovery_tracker = RecoveryTracker()

        # Notification-Tracking
        self.pending_notifications: dict[str, list[BreachNotification]] = defaultdict(list)
        self.sent_notifications: list[BreachNotification] = []

        # Thread-Safety
        self._lock = threading.RLock()

        # Escalation-Tasks
        self._escalation_tasks: dict[str, asyncio.Task] = {}

        # Metrics
        self._metrics_collector = MetricsCollector()

        # Notification-Callbacks
        self._notification_callbacks: dict[
            str, list[Callable[[BreachNotification], Awaitable[None]]]
        ] = defaultdict(list)

    def register_escalation_workflow(self, sla_name: str, workflow: EscalationWorkflow):
        """Registriert Escalation-Workflow für SLA.

        Args:
            sla_name: SLA-Name
            workflow: Escalation-Workflow
        """
        with self._lock:
            self.escalation_workflows[sla_name] = workflow

    def register_notification_callback(
        self, channel: str, callback: Callable[[BreachNotification], Awaitable[None]]
    ):
        """Registriert Notification-Callback für Channel.

        Args:
            channel: Notification-Channel (email, slack, pagerduty, etc.)
            callback: Callback-Funktion
        """
        self._notification_callbacks[channel].append(callback)

    async def handle_sla_breach(self, sla_metrics: SLAMetrics) -> SLABreach | None:
        """Behandelt SLA-Breach.

        Args:
            sla_metrics: SLA-Metriken

        Returns:
            SLA-Breach falls einer aufgetreten ist
        """
        sla_def = sla_metrics.sla_definition

        if not sla_metrics.is_breached:
            # Prüfe ob Recovery von aktiver Breach
            if sla_def.name in self.active_breaches:
                await self._handle_sla_recovery(sla_def.name, sla_metrics)
            return None

        # Prüfe ob bereits aktive Breach
        if sla_def.name in self.active_breaches:
            return self.active_breaches[sla_def.name]

        # Neue SLA-Breach
        breach = SLABreach(
            sla_name=sla_def.name,
            breach_timestamp=time.time(),
            compliance_at_breach=sla_metrics.current_compliance,
            affected_slos=[slo.name for slo in sla_def.slo_definitions],
            severity=self._calculate_breach_severity(sla_metrics),
            customer_impact=self._assess_customer_impact(sla_def),
            business_impact=self._assess_business_impact(sla_def),
        )

        with self._lock:
            self.active_breaches[sla_def.name] = breach
            self.breach_history.append(breach)

            # Behalte nur letzte 1000 Breaches
            if len(self.breach_history) > 1000:
                self.breach_history = self.breach_history[-1000:]

        logger.error(
            f"SLA-Breach detected: {sla_def.name} - "
            f"Compliance: {sla_metrics.current_compliance:.2f}%, "
            f"Severity: {breach.severity.value}"
        )

        # Metrics
        self._metrics_collector.increment_counter(
            "sla.breaches.detected",
            tags={
                "sla_name": sla_def.name,
                "customer": sla_def.customer or "internal",
                "severity": breach.severity.value,
            },
        )

        # Starte Escalation-Workflow
        await self._start_escalation_workflow(breach)

        # Erstelle Alert
        await self._create_breach_alert(breach, sla_def)

        return breach

    async def _handle_sla_recovery(self, sla_name: str, sla_metrics: SLAMetrics):
        """Behandelt SLA-Recovery."""
        with self._lock:
            if sla_name not in self.active_breaches:
                return

            breach = self.active_breaches[sla_name]
            del self.active_breaches[sla_name]

        # Berechne Recovery-Zeit
        recovery_time = time.time() - breach.breach_timestamp
        breach.resolved = True
        breach.resolution_timestamp = time.time()
        breach.recovery_time = recovery_time

        # Recovery-Tracking
        self.recovery_tracker.record_recovery(sla_name, recovery_time)

        logger.info(
            f"SLA-Recovery: {sla_name} - "
            f"Recovery-Zeit: {recovery_time:.2f}s, "
            f"Compliance: {sla_metrics.current_compliance:.2f}%"
        )

        # Metrics
        self._metrics_collector.record_histogram(
            "sla.recovery_time",
            recovery_time,
            tags={
                "sla_name": sla_name,
                "customer": sla_metrics.sla_definition.customer or "internal",
            },
        )

        # Stoppe Escalation-Workflow
        await self._stop_escalation_workflow(sla_name)

        # Recovery-Alert
        await self._create_recovery_alert(breach, sla_metrics.sla_definition)

    def _calculate_breach_severity(self, sla_metrics: SLAMetrics) -> ViolationSeverity:
        """Berechnet Breach-Severity basierend auf Compliance."""
        compliance = sla_metrics.current_compliance

        if compliance < 50.0:  # < 50% Compliance
            return ViolationSeverity.CRITICAL
        if compliance < 70.0:  # < 70% Compliance
            return ViolationSeverity.HIGH
        if compliance < 85.0:  # < 85% Compliance
            return ViolationSeverity.MEDIUM
        return ViolationSeverity.LOW

    def _assess_customer_impact(self, sla_def: SLADefinition) -> str:
        """Bewertet Customer-Impact."""
        if sla_def.customer:
            if sla_def.priority == "high":
                return "High customer impact - Premium customer affected"
            if sla_def.priority == "medium":
                return "Medium customer impact - Standard customer affected"
            return "Low customer impact - Basic customer affected"
        return "Internal service impact - No direct customer impact"

    def _assess_business_impact(self, sla_def: SLADefinition) -> str:
        """Bewertet Business-Impact."""
        if sla_def.penalty_enabled and sla_def.penalty_amount > 0:
            return f"Financial impact - Potential penalty: {sla_def.penalty_amount} {sla_def.penalty_currency}"
        return "Reputational impact - Service quality degradation"

    async def _start_escalation_workflow(self, breach: SLABreach):
        """Startet Escalation-Workflow für Breach."""
        workflow = self.escalation_workflows.get(breach.sla_name)
        if not workflow or not workflow.auto_escalation_enabled:
            return

        # Starte Escalation-Task
        task = asyncio.create_task(self._escalation_workflow_task(breach, workflow))

        self._escalation_tasks[breach.sla_name] = task

    async def _stop_escalation_workflow(self, sla_name: str):
        """Stoppt Escalation-Workflow."""
        if sla_name in self._escalation_tasks:
            task = self._escalation_tasks[sla_name]
            if not task.done():
                task.cancel()
            del self._escalation_tasks[sla_name]

    async def _escalation_workflow_task(self, breach: SLABreach, workflow: EscalationWorkflow):
        """Escalation-Workflow-Task."""
        current_level = EscalationLevel.LEVEL_0

        try:
            while True:
                # Sende Notification für aktuelles Level
                await self._send_escalation_notification(breach, workflow, current_level)

                # Prüfe ob Breach noch aktiv ist
                if breach.sla_name not in self.active_breaches:
                    break

                # Nächstes Escalation-Level
                next_level = workflow.get_next_escalation_level(current_level)
                if not next_level:
                    break

                # Warte auf Escalation-Delay
                delay = workflow.get_escalation_delay(next_level)
                await asyncio.sleep(delay)

                current_level = next_level

                # Update Breach-Escalation-Info
                with self._lock:
                    if breach.sla_name in self.active_breaches:
                        active_breach = self.active_breaches[breach.sla_name]
                        active_breach.escalated = True
                        active_breach.escalation_timestamp = time.time()
                        active_breach.escalation_level = list(EscalationLevel).index(current_level)

        except asyncio.CancelledError:
            logger.info(f"Escalation-Workflow für {breach.sla_name} gestoppt")
        except Exception as e:
            logger.error(f"Fehler im Escalation-Workflow für {breach.sla_name}: {e}")

    async def _send_escalation_notification(
        self, breach: SLABreach, workflow: EscalationWorkflow, level: EscalationLevel
    ):
        """Sendet Escalation-Notification."""
        notification = BreachNotification(
            breach_id=f"{breach.sla_name}_{int(breach.breach_timestamp)}",
            sla_name=breach.sla_name,
            severity=breach.severity,
            escalation_level=level,
            title=f"SLA Breach - {breach.sla_name} ({level.value.upper()})",
            message=self._create_breach_message(breach, level),
            recipients=workflow.get_escalation_contacts(level),
            channels=workflow.get_escalation_channels(level),
        )

        # Sende Notification über alle Channels
        for channel in notification.channels:
            await self._send_notification_via_channel(notification, channel)

        # Tracking
        with self._lock:
            self.pending_notifications[breach.sla_name].append(notification)
            self.sent_notifications.append(notification)

            # Behalte nur letzte 1000 Notifications
            if len(self.sent_notifications) > 1000:
                self.sent_notifications = self.sent_notifications[-1000:]

        notification.sent = True
        notification.sent_at = time.time()

        logger.info(
            f"Escalation-Notification gesendet: {breach.sla_name} - "
            f"Level: {level.value}, Channels: {notification.channels}"
        )

    def _create_breach_message(self, breach: SLABreach, level: EscalationLevel) -> str:
        """Erstellt Breach-Message."""
        return f"""
SLA Breach Alert - Escalation Level {level.value.upper()}

SLA: {breach.sla_name}
Breach Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(breach.breach_timestamp))}
Severity: {breach.severity.value.upper()}
Compliance at Breach: {breach.compliance_at_breach:.2f}%

Affected SLOs: {', '.join(breach.affected_slos)}

Customer Impact: {breach.customer_impact}
Business Impact: {breach.business_impact}

Please investigate and resolve immediately.
        """.strip()

    async def _send_notification_via_channel(self, notification: BreachNotification, channel: str):
        """Sendet Notification über spezifischen Channel."""
        callbacks = self._notification_callbacks.get(channel, [])

        for callback in callbacks:
            try:
                await callback(notification)
            except Exception as e:
                logger.error(f"Fehler beim Senden von Notification über {channel}: {e}")

    async def _create_breach_alert(self, breach: SLABreach, sla_def: SLADefinition):
        """Erstellt Alert für SLA-Breach."""
        severity_map = {
            ViolationSeverity.LOW: AlertSeverity.INFO,
            ViolationSeverity.MEDIUM: AlertSeverity.WARNING,
            ViolationSeverity.HIGH: AlertSeverity.ERROR,
            ViolationSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }

        alert_severity = severity_map.get(breach.severity, AlertSeverity.ERROR)

        await self.alert_manager.create_alert(
            alert_id=f"sla_breach_{breach.sla_name}_{int(breach.breach_timestamp)}",
            severity=alert_severity,
            title=f"SLA Breach: {breach.sla_name}",
            description=f"SLA {breach.sla_name} breached with {breach.compliance_at_breach:.2f}% compliance",
            metric_name="sla_compliance",
            metric_value=breach.compliance_at_breach,
            threshold=sla_def.penalty_threshold,
            tags={
                "sla_name": breach.sla_name,
                "customer": sla_def.customer or "internal",
                "severity": breach.severity.value,
            },
        )

    async def _create_recovery_alert(self, breach: SLABreach, sla_def: SLADefinition):
        """Erstellt Alert für SLA-Recovery."""
        await self.alert_manager.create_alert(
            alert_id=f"sla_recovery_{breach.sla_name}_{int(time.time())}",
            severity=AlertSeverity.INFO,
            title=f"SLA Recovery: {breach.sla_name}",
            description=f"SLA {breach.sla_name} recovered after {breach.recovery_time:.2f}s",
            metric_name="sla_recovery_time",
            metric_value=breach.recovery_time or 0.0,
            tags={"sla_name": breach.sla_name, "customer": sla_def.customer or "internal"},
        )

    def get_active_breaches(self) -> list[SLABreach]:
        """Holt alle aktiven SLA-Breaches."""
        with self._lock:
            return list(self.active_breaches.values())

    def get_breach_history(self, hours: float = 24.0) -> list[SLABreach]:
        """Holt Breach-History der letzten N Stunden."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            return [
                breach for breach in self.breach_history if breach.breach_timestamp >= cutoff_time
            ]

    def get_recovery_statistics(self, sla_name: str | None = None) -> dict[str, Any]:
        """Holt Recovery-Statistiken."""
        if sla_name:
            return self.recovery_tracker.get_recovery_statistics(sla_name)
        # Alle SLAs
        all_stats = {}
        for sla in self.escalation_workflows.keys():
            all_stats[sla] = self.recovery_tracker.get_recovery_statistics(sla)
        return all_stats

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt Breach-Manager-Metriken-Zusammenfassung."""
        with self._lock:
            return {
                "active_breaches": len(self.active_breaches),
                "total_breaches_24h": len(self.get_breach_history(24.0)),
                "escalation_workflows": len(self.escalation_workflows),
                "pending_notifications": sum(
                    len(notifications) for notifications in self.pending_notifications.values()
                ),
                "sent_notifications_24h": len(
                    [n for n in self.sent_notifications if n.timestamp >= time.time() - 86400]
                ),
                "recovery_statistics": self.get_recovery_statistics(),
            }
