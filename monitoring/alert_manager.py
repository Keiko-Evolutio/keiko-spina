"""Alert Manager Implementation.
Verwaltet Alerts und Benachrichtigungen für das Monitoring-System.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .interfaces import Alert, AlertSeverity, IAlertManager, IMetricsCollector

logger = get_logger(__name__)


@dataclass
class AlertRule:
    """Alert-Regel Definition."""
    name: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    message_template: str
    labels: dict[str, str]
    cooldown_seconds: int = 300  # 5-minute default
    enabled: bool = True


@dataclass
class NotificationChannel:
    """Benachrichtigungs-Kanal."""
    name: str
    type: str  # webhook, email, slack, etc.
    config: dict[str, Any]
    enabled: bool = True


class AlertManager(IAlertManager):
    """Alert Manager Implementation.
    Verwaltet Alert-Regeln, Benachrichtigungen und Alert-Lifecycle.
    """

    def __init__(self, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector

        # Alert-Storage
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._alert_rules: dict[str, AlertRule] = {}
        self._notification_channels: dict[str, NotificationChannel] = {}

        # Alert-Tracking
        self._last_alert_times: dict[str, datetime] = {}
        self._alert_counts: dict[str, int] = defaultdict(int)

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

        # Standard Alert-Regeln registrieren
        self._register_default_alert_rules()

        logger.info("Alert manager initialized")

    def _register_default_alert_rules(self) -> None:
        """Registriert Standard-Alert-Regeln."""
        # High CPU Usage
        self.register_alert_rule(
            name="high_cpu_usage",
            condition=lambda: self._check_high_cpu(),
            severity=AlertSeverity.HIGH,
            message_template="High CPU usage detected: {cpu_percent}%",
            labels={"component": "system", "type": "resource"}
        )

        # High Memory Usage
        self.register_alert_rule(
            name="high_memory_usage",
            condition=lambda: self._check_high_memory(),
            severity=AlertSeverity.HIGH,
            message_template="High memory usage detected: {memory_percent}%",
            labels={"component": "system", "type": "resource"}
        )

        # Voice Workflow Failure Rate
        self.register_alert_rule(
            name="voice_workflow_high_failure_rate",
            condition=lambda: self._check_voice_failure_rate(),
            severity=AlertSeverity.MEDIUM,
            message_template="Voice workflow failure rate is high: {failure_rate}%",
            labels={"component": "voice", "type": "business"}
        )

        # Service Health Check Failures
        self.register_alert_rule(
            name="service_health_check_failure",
            condition=lambda: self._check_service_health(),
            severity=AlertSeverity.CRITICAL,
            message_template="Critical service health check failed: {service_name}",
            labels={"component": "health", "type": "availability"}
        )

        logger.debug("Default alert rules registered")

    def register_alert_rule(self, name: str, condition: Callable[[], bool],
                          severity: AlertSeverity, message_template: str,
                          labels: dict[str, str] = None, cooldown_seconds: int = 300) -> None:
        """Registriert neue Alert-Regel."""
        rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            message_template=message_template,
            labels=labels or {},
            cooldown_seconds=cooldown_seconds
        )

        self._alert_rules[name] = rule
        logger.info(f"Registered alert rule: {name} (severity: {severity.value})")

    def register_notification_channel(self, name: str, channel_type: str, config: dict[str, Any]) -> None:
        """Registriert Benachrichtigungs-Kanal."""
        channel = NotificationChannel(
            name=name,
            type=channel_type,
            config=config
        )

        self._notification_channels[name] = channel
        logger.info(f"Registered notification channel: {name} (type: {channel_type})")

    async def send_alert(self, alert: Alert) -> None:
        """Sendet einen Alert."""
        # Alert zu aktiven Alerts hinzufügen
        self._active_alerts[alert.name] = alert
        self._alert_history.append(alert)

        # Alert-History begrenzen (letzte 1000)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]

        # Metriken aktualisieren
        self.metrics_collector.increment_counter(
            "alerts_total",
            labels={
                "alert_name": alert.name,
                "severity": alert.severity.value,
                **alert.labels
            }
        )

        self.metrics_collector.set_gauge(
            "active_alerts",
            len(self._active_alerts)
        )

        # Benachrichtigungen senden
        await self._send_notifications(alert)

        logger.warning(f"Alert sent: {alert.name} (severity: {alert.severity.value}) - {alert.message}")

    async def resolve_alert(self, alert_name: str) -> None:
        """Markiert Alert als gelöst."""
        if alert_name in self._active_alerts:
            alert = self._active_alerts[alert_name]
            alert.resolved = True

            # Aus aktiven Alerts entfernen
            del self._active_alerts[alert_name]

            # Metriken aktualisieren
            self.metrics_collector.increment_counter(
                "alerts_resolved_total",
                labels={
                    "alert_name": alert_name,
                    "severity": alert.severity.value
                }
            )

            self.metrics_collector.set_gauge(
                "active_alerts",
                len(self._active_alerts)
            )

            logger.info(f"Alert resolved: {alert_name}")

    def get_active_alerts(self) -> list[Alert]:
        """Gibt aktive Alerts zurück."""
        return list(self._active_alerts.values())

    async def start_alert_monitoring(self) -> None:
        """Startet Alert-Monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._alert_monitoring_loop())
        logger.info("Started alert monitoring")

    async def stop_alert_monitoring(self) -> None:
        """Stoppt Alert-Monitoring."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped alert monitoring")

    async def _alert_monitoring_loop(self) -> None:
        """Alert-Monitoring-Schleife."""
        while self._running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _evaluate_alert_rules(self) -> None:
        """Evaluiert alle Alert-Regeln."""
        current_time = datetime.utcnow()

        for rule_name, rule in self._alert_rules.items():
            if not rule.enabled:
                continue

            try:
                # Cooldown prüfen
                last_alert_time = self._last_alert_times.get(rule_name)
                if (last_alert_time and
                    current_time - last_alert_time < timedelta(seconds=rule.cooldown_seconds)):
                    continue

                # Regel-Condition evaluieren
                if rule.condition():
                    # Alert erstellen und senden
                    alert = Alert(
                        name=rule_name,
                        severity=rule.severity,
                        message=rule.message_template,  # Template wird später gefüllt
                        labels=rule.labels,
                        timestamp=current_time
                    )

                    await self.send_alert(alert)
                    self._last_alert_times[rule_name] = current_time
                    self._alert_counts[rule_name] += 1

                # Prüfe ob Alert gelöst werden kann
                elif rule_name in self._active_alerts:
                    await self.resolve_alert(rule_name)

            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")

    async def _send_notifications(self, alert: Alert) -> None:
        """Sendet Benachrichtigungen für Alert."""
        for channel_name, channel in self._notification_channels.items():
            if not channel.enabled:
                continue

            try:
                await self._send_notification_to_channel(alert, channel)
            except Exception as e:
                logger.error(f"Failed to send notification to {channel_name}: {e}")

    async def _send_notification_to_channel(self, alert: Alert, channel: NotificationChannel) -> None:
        """Sendet Benachrichtigung an spezifischen Kanal."""
        if channel.type == "webhook":
            await self._send_webhook_notification(alert, channel)
        elif channel.type == "email":
            await self._send_email_notification(alert, channel)
        elif channel.type == "slack":
            await self._send_slack_notification(alert, channel)
        else:
            logger.warning(f"Unknown notification channel type: {channel.type}")

    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Sendet Webhook-Benachrichtigung."""
        try:
            import aiohttp

            webhook_url = channel.config.get("url")
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return

            payload = {
                "alert": asdict(alert),
                "timestamp": alert.timestamp.isoformat(),
                "source": "keiko-monitoring"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.debug(f"Webhook notification sent for alert {alert.name}")
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")

    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Sendet E-Mail-Benachrichtigung."""
        # Placeholder für E-Mail-Implementation
        logger.debug(f"Email notification would be sent for alert {alert.name}")

    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Sendet Slack-Benachrichtigung."""
        # Placeholder für Slack-Implementation
        logger.debug(f"Slack notification would be sent for alert {alert.name}")

    def get_alert_statistics(self) -> dict[str, Any]:
        """Gibt Alert-Statistiken zurück."""
        return {
            "active_alerts": len(self._active_alerts),
            "total_alerts_sent": len(self._alert_history),
            "alert_counts_by_rule": dict(self._alert_counts),
            "alerts_by_severity": self._get_alerts_by_severity(),
            "notification_channels": len(self._notification_channels),
            "alert_rules": len(self._alert_rules)
        }

    def _get_alerts_by_severity(self) -> dict[str, int]:
        """Gibt Alert-Anzahl nach Schweregrad zurück."""
        severity_counts = defaultdict(int)

        for alert in self._alert_history[-100:]:  # Letzte 100 Alerts
            severity_counts[alert.severity.value] += 1

        return dict(severity_counts)

    # Alert-Condition-Funktionen
    def _check_high_cpu(self) -> bool:
        """Prüft hohe CPU-Nutzung."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent > 80.0
        except Exception:
            return False

    def _check_high_memory(self) -> bool:
        """Prüft hohe Speicher-Nutzung."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent > 80.0
        except Exception:
            return False

    def _check_voice_failure_rate(self) -> bool:
        """Prüft Voice-Workflow-Fehlerrate."""
        # Placeholder - würde echte Metriken prüfen
        return False

    def _check_service_health(self) -> bool:
        """Prüft Service-Health."""
        # Placeholder - würde echte Health-Checks prüfen
        return False
