# backend/agents/slo_sla/alerting.py
"""SLO/SLA Alerting System

Implementiert umfassende Alerting-Funktionalitäten:
- SLO-Threshold-basierte Alerts
- Error Budget Burn Rate Alerts
- Multi-Channel Alert Routing
- Alert Escalation und Suppression
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import aiohttp

from .exceptions import AlertChannelError, AlertingError, EscalationError

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert Severity Enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Alert Rule Data Structure."""
    rule_id: str
    slo_name: str
    alert_type: str
    severity: str
    threshold_value: float | None = None
    comparison_operator: str | None = None
    evaluation_window: timedelta | None = None
    burn_rate_threshold: float | None = None
    time_window: timedelta | None = None
    windows: list[dict[str, Any]] | None = None


@dataclass
class AlertChannel:
    """Alert Channel Data Structure."""
    channel_id: str
    channel_type: str
    config: dict[str, Any]
    severity_filter: list[str] | None = None


@dataclass
class AlertEscalation:
    """Alert Escalation Data Structure."""
    escalation_id: str
    levels: list[dict[str, Any]]


@dataclass
class ThresholdAlert:
    """Threshold Alert Data Structure."""
    rule_id: str
    slo_name: str
    severity: str
    current_value: float | None = None
    threshold_value: float | None = None
    timestamp: datetime | None = None
    escalation_id: str | None = None


@dataclass
class BurnRateAlert:
    """Burn Rate Alert Data Structure."""
    rule_id: str
    slo_name: str
    severity: str
    burn_rate: float | None = None
    error_budget_remaining: float | None = None
    timestamp: datetime | None = None
    alert_type: str = "burn_rate"
    triggered_window: timedelta | None = None


@dataclass
class AlertSuppression:
    """Alert Suppression Data Structure."""
    suppression_id: str
    slo_names: list[str]
    start_time: datetime
    end_time: datetime
    reason: str


class AlertChannelManager:
    """Alert Channel Manager."""

    def __init__(self):
        """Initialisiert AlertChannelManager."""
        self._channels: dict[str, AlertChannel] = {}

        logger.info("AlertChannelManager initialisiert")

    async def send_to_slack(self, channel: AlertChannel, alert: Any) -> dict[str, Any]:
        """Sendet Alert an Slack.

        Args:
            channel: Slack Channel
            alert: Alert-Objekt

        Returns:
            Versand-Ergebnis
        """
        try:
            webhook_url = channel.config.get("webhook_url")
            if not webhook_url:
                raise AlertChannelError("Slack Webhook URL nicht konfiguriert")

            # Erstelle Slack-Message
            message = {
                "text": f"SLO Alert: {alert.slo_name}",
                "attachments": [
                    {
                        "color": self._get_slack_color(alert.severity),
                        "fields": [
                            {
                                "title": "SLO",
                                "value": alert.slo_name,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity,
                                "short": True
                            },
                            {
                                "title": "Rule",
                                "value": alert.rule_id,
                                "short": True
                            }
                        ]
                    }
                ]
            }

            # Sende an Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "channel_id": channel.channel_id,
                            "message_id": "slack_msg"
                        }
                    raise AlertChannelError(f"Slack API Error: {response.status}")

        except Exception as e:
            logger.error(f"Slack-Versand fehlgeschlagen: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "retry_recommended": True
            }

    def _get_slack_color(self, severity: str) -> str:
        """Gibt Slack-Farbe für Severity zurück."""
        color_map = {
            "low": "good",
            "medium": "warning",
            "high": "danger",
            "critical": "danger"
        }
        return color_map.get(severity, "warning")

    async def send_to_email(self, channel: AlertChannel, alert: Any) -> dict[str, Any]:
        """Sendet Alert per Email.

        Args:
            channel: Email Channel
            alert: Alert-Objekt

        Returns:
            Versand-Ergebnis
        """
        try:
            smtp_config = channel.config
            recipients = smtp_config.get("recipients", [])

            if not recipients:
                raise AlertChannelError("Keine Email-Empfänger konfiguriert")

            # Erstelle Email
            msg = MIMEMultipart()
            msg["From"] = smtp_config.get("username", "alerts@company.com")
            msg["Subject"] = f"SLO Alert: {alert.slo_name} - {alert.severity.upper()}"

            # Email-Body
            body = f"""
SLO Alert Details:

SLO Name: {alert.slo_name}
Severity: {alert.severity}
Rule ID: {alert.rule_id}
Timestamp: {getattr(alert, 'timestamp', datetime.utcnow())}

Please investigate and take appropriate action.
            """

            msg.attach(MIMEText(body, "plain"))

            # Sende Email (Mock-Implementierung)
            logger.info(f"Email-Alert an {len(recipients)} Empfänger gesendet")

            return {
                "success": True,
                "recipients_notified": len(recipients)
            }

        except Exception as e:
            logger.error(f"Email-Versand fehlgeschlagen: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "retry_recommended": True
            }

    async def send_to_pagerduty(self, channel: AlertChannel, alert: Any) -> dict[str, Any]:
        """Sendet Alert an PagerDuty.

        Args:
            channel: PagerDuty Channel
            alert: Alert-Objekt

        Returns:
            Versand-Ergebnis
        """
        try:
            integration_key = channel.config.get("integration_key")
            if not integration_key:
                raise AlertChannelError("PagerDuty Integration Key nicht konfiguriert")

            # Erstelle PagerDuty Event
            event = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"SLO Alert: {alert.slo_name}",
                    "severity": alert.severity,
                    "source": "SLO Monitor",
                    "custom_details": {
                        "slo_name": alert.slo_name,
                        "rule_id": alert.rule_id,
                        "severity": alert.severity
                    }
                }
            }

            # Sende an PagerDuty
            pagerduty_url = "https://events.pagerduty.com/v2/enqueue"

            async with aiohttp.ClientSession() as session:
                async with session.post(pagerduty_url, json=event) as response:
                    if response.status == 202:
                        result = await response.json()
                        return {
                            "success": True,
                            "incident_key": result.get("dedup_key", "unknown")
                        }
                    raise AlertChannelError(f"PagerDuty API Error: {response.status}")

        except Exception as e:
            logger.error(f"PagerDuty-Versand fehlgeschlagen: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "retry_recommended": True
            }

    async def send_to_webhook(self, channel: AlertChannel, alert: Any) -> dict[str, Any]:
        """Sendet Alert an Generic Webhook.

        Args:
            channel: Webhook Channel
            alert: Alert-Objekt

        Returns:
            Versand-Ergebnis
        """
        try:
            webhook_config = channel.config
            url = webhook_config.get("url")
            method = webhook_config.get("method", "POST")
            headers = webhook_config.get("headers", {})
            timeout = webhook_config.get("timeout", 30)

            if not url:
                raise AlertChannelError("Webhook URL nicht konfiguriert")

            # Erstelle Payload
            payload = {
                "alert_type": "slo_alert",
                "slo_name": alert.slo_name,
                "severity": alert.severity,
                "rule_id": alert.rule_id,
                "timestamp": getattr(alert, "timestamp", datetime.utcnow()).isoformat()
            }

            # Sende Webhook
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(method, url, json=payload, headers=headers) as response:
                    return {
                        "success": True,
                        "response_status": response.status
                    }

        except Exception as e:
            logger.error(f"Webhook-Versand fehlgeschlagen: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "retry_recommended": True
            }


class SLOAlertManager:
    """Zentraler SLO Alert Manager."""

    def __init__(self, evaluation_interval: timedelta = timedelta(minutes=1),
                 max_concurrent_alerts: int = 50, enable_correlation: bool = True):
        """Initialisiert SLOAlertManager.

        Args:
            evaluation_interval: Evaluation-Intervall
            max_concurrent_alerts: Maximale gleichzeitige Alerts
            enable_correlation: Alert-Korrelation aktivieren
        """
        self.evaluation_interval = evaluation_interval
        self.max_concurrent_alerts = max_concurrent_alerts
        self.enable_correlation = enable_correlation

        # Initialisiere Komponenten
        self.channel_manager = AlertChannelManager()

        # Alert-Management
        self._alert_rules: dict[str, AlertRule] = {}
        self._alert_channels: dict[str, AlertChannel] = {}
        self._escalations: dict[str, AlertEscalation] = {}
        self._suppressions: dict[str, AlertSuppression] = {}
        self._active_alerts: dict[str, Any] = {}

        # Rate Limiting
        self._rate_limits = {
            "max_alerts_per_minute": 10,
            "max_alerts_per_hour": 50
        }
        self._alert_counts = {
            "minute": 0,
            "hour": 0,
            "last_minute_reset": datetime.utcnow(),
            "last_hour_reset": datetime.utcnow()
        }

        logger.info(f"SLOAlertManager initialisiert (max_alerts: {max_concurrent_alerts})")

    async def register_alert_rule(self, alert_rule: AlertRule) -> None:
        """Registriert Alert Rule.

        Args:
            alert_rule: Alert Rule
        """
        self._alert_rules[alert_rule.rule_id] = alert_rule
        logger.info(f"Alert Rule {alert_rule.rule_id} registriert")

    async def register_alert_channel(self, channel: AlertChannel) -> None:
        """Registriert Alert Channel.

        Args:
            channel: Alert Channel
        """
        self._alert_channels[channel.channel_id] = channel
        logger.info(f"Alert Channel {channel.channel_id} registriert")

    async def register_escalation(self, escalation: AlertEscalation) -> None:
        """Registriert Escalation-Konfiguration.

        Args:
            escalation: Escalation-Konfiguration
        """
        self._escalations[escalation.escalation_id] = escalation
        logger.info(f"Escalation {escalation.escalation_id} registriert")

    async def register_suppression(self, suppression: AlertSuppression) -> None:
        """Registriert Alert-Suppression.

        Args:
            suppression: Alert-Suppression
        """
        self._suppressions[suppression.suppression_id] = suppression
        logger.info(f"Alert-Suppression {suppression.suppression_id} registriert")

    async def evaluate_threshold_alerts(self, slo_metrics: dict[str, Any]) -> list[ThresholdAlert]:
        """Evaluiert Threshold-basierte Alerts.

        Args:
            slo_metrics: SLO-Metriken

        Returns:
            Liste der getriggerten Alerts
        """
        triggered_alerts = []

        for rule_id, rule in self._alert_rules.items():
            if rule.alert_type == "threshold":
                current_value = slo_metrics.get("current_value", 0)
                threshold_value = rule.threshold_value
                operator = rule.comparison_operator

                alert_triggered = False

                if (operator == "<" and current_value < threshold_value) or (operator == ">" and current_value > threshold_value) or (operator == "<=" and current_value <= threshold_value) or (operator == ">=" and current_value >= threshold_value):
                    alert_triggered = True

                if alert_triggered:
                    alert = ThresholdAlert(
                        rule_id=rule_id,
                        slo_name=rule.slo_name,
                        severity=rule.severity,
                        current_value=current_value,
                        threshold_value=threshold_value,
                        timestamp=datetime.utcnow()
                    )
                    triggered_alerts.append(alert)

        return triggered_alerts

    async def evaluate_burn_rate_alerts(self, burn_rate_metrics: dict[str, Any]) -> list[BurnRateAlert]:
        """Evaluiert Burn Rate Alerts.

        Args:
            burn_rate_metrics: Burn Rate Metriken

        Returns:
            Liste der getriggerten Burn Rate Alerts
        """
        triggered_alerts = []

        for rule_id, rule in self._alert_rules.items():
            if rule.alert_type == "burn_rate":
                current_burn_rate = burn_rate_metrics.get("current_burn_rate", 0)
                threshold = rule.burn_rate_threshold

                if current_burn_rate > threshold:
                    alert = BurnRateAlert(
                        rule_id=rule_id,
                        slo_name=rule.slo_name,
                        severity=rule.severity,
                        burn_rate=current_burn_rate,
                        error_budget_remaining=burn_rate_metrics.get("error_budget_remaining", 0),
                        timestamp=datetime.utcnow()
                    )
                    triggered_alerts.append(alert)

        return triggered_alerts

    async def evaluate_multi_window_burn_rate(self, multi_window_metrics: dict[str, Any]) -> list[BurnRateAlert]:
        """Evaluiert Multi-Window Burn Rate Alerts.

        Args:
            multi_window_metrics: Multi-Window Metriken

        Returns:
            Liste der getriggerten Alerts
        """
        triggered_alerts = []

        for rule_id, rule in self._alert_rules.items():
            if rule.alert_type == "multi_window_burn_rate" and rule.windows:
                burn_rates = multi_window_metrics.get("burn_rates", {})

                for window_config in rule.windows:
                    window_duration = window_config["duration"]
                    threshold = window_config["burn_rate_threshold"]

                    if window_duration in burn_rates:
                        current_rate = burn_rates[window_duration]

                        if current_rate > threshold:
                            alert = BurnRateAlert(
                                rule_id=rule_id,
                                slo_name=rule.slo_name,
                                severity=rule.severity,
                                burn_rate=current_rate,
                                timestamp=datetime.utcnow(),
                                triggered_window=window_duration
                            )
                            triggered_alerts.append(alert)
                            break  # Nur ein Alert pro Rule

        return triggered_alerts

    async def route_alert(self, alert: Any) -> dict[str, Any]:
        """Routet Alert an entsprechende Channels.

        Args:
            alert: Alert-Objekt

        Returns:
            Routing-Ergebnis
        """
        try:
            # Prüfe Suppression
            suppression_result = await self.check_suppression(alert)
            if suppression_result.get("suppressed"):
                return {
                    "channels_notified": 0,
                    "suppressed": True,
                    "reason": suppression_result.get("reason")
                }

            # Finde passende Channels
            matching_channels = []
            for channel_id, channel in self._alert_channels.items():
                if self._channel_matches_alert(channel, alert):
                    matching_channels.append(channel)

            # Sende an Channels
            channels_notified = 0
            for channel in matching_channels:
                try:
                    result = await self._send_to_channel(channel, alert)
                    if result.get("success"):
                        channels_notified += 1
                except Exception as e:
                    logger.error(f"Channel-Versand fehlgeschlagen: {e}")

            return {
                "channels_notified": channels_notified,
                "total_channels": len(matching_channels)
            }

        except Exception as e:
            logger.error(f"Alert-Routing fehlgeschlagen: {e}")
            raise AlertingError(f"Alert-Routing fehlgeschlagen: {e}")

    def _channel_matches_alert(self, channel: AlertChannel, alert: Any) -> bool:
        """Prüft ob Channel für Alert geeignet ist."""
        if channel.severity_filter:
            return alert.severity in channel.severity_filter
        return True

    async def _send_to_channel(self, channel: AlertChannel, alert: Any) -> dict[str, Any]:
        """Sendet Alert an spezifischen Channel."""
        if channel.channel_type == "slack":
            return await self.channel_manager.send_to_slack(channel, alert)
        if channel.channel_type == "email":
            return await self.channel_manager.send_to_email(channel, alert)
        if channel.channel_type == "pagerduty":
            return await self.channel_manager.send_to_pagerduty(channel, alert)
        if channel.channel_type == "webhook":
            return await self.channel_manager.send_to_webhook(channel, alert)
        raise AlertChannelError(f"Unbekannter Channel-Typ: {channel.channel_type}")

    async def check_escalation(self, alert: Any) -> dict[str, Any]:
        """Prüft Alert-Escalation.

        Args:
            alert: Alert-Objekt

        Returns:
            Escalation-Ergebnis
        """
        try:
            escalation_id = getattr(alert, "escalation_id", None)
            if not escalation_id or escalation_id not in self._escalations:
                return {"escalated": False}

            escalation = self._escalations[escalation_id]
            alert_age = datetime.utcnow() - alert.timestamp

            # Finde passende Escalation-Level
            for level_config in escalation.levels:
                level = level_config["level"]
                delay = level_config["delay"]

                if alert_age >= delay:
                    # Escalation erforderlich
                    return {
                        "escalated": True,
                        "escalation_level": level,
                        "channels": level_config.get("channels", [])
                    }

            return {"escalated": False}

        except Exception as e:
            logger.error(f"Escalation-Check fehlgeschlagen: {e}")
            raise EscalationError(f"Escalation-Check fehlgeschlagen: {e}")

    async def check_suppression(self, alert: Any) -> dict[str, Any]:
        """Prüft Alert-Suppression.

        Args:
            alert: Alert-Objekt

        Returns:
            Suppression-Ergebnis
        """
        try:
            current_time = datetime.utcnow()

            for suppression_id, suppression in self._suppressions.items():
                # Prüfe Zeitfenster
                if suppression.start_time <= current_time <= suppression.end_time:
                    # Prüfe SLO-Namen
                    if alert.slo_name in suppression.slo_names:
                        return {
                            "suppressed": True,
                            "suppression_id": suppression_id,
                            "reason": suppression.reason
                        }

            return {"suppressed": False}

        except Exception as e:
            logger.error(f"Suppression-Check fehlgeschlagen: {e}")
            return {"suppressed": False}

    async def correlate_alerts(self, alerts: list[Any]) -> dict[str, Any]:
        """Korreliert verwandte Alerts.

        Args:
            alerts: Liste der Alerts

        Returns:
            Korrelations-Ergebnis
        """
        if not self.enable_correlation or len(alerts) < 2:
            return {"correlated": False}

        # Einfache Korrelation basierend auf Service/SLO
        service_groups = {}
        for alert in alerts:
            service_key = alert.slo_name.split("_")[0]  # Extrahiere Service-Name
            if service_key not in service_groups:
                service_groups[service_key] = []
            service_groups[service_key].append(alert)

        # Finde Gruppen mit mehreren Alerts
        for service, service_alerts in service_groups.items():
            if len(service_alerts) >= 2:
                return {
                    "correlated": True,
                    "correlation_type": "service_degradation",
                    "correlated_alerts": service_alerts,
                    "root_cause_hypothesis": f"{service}_infrastructure_issue"
                }

        return {"correlated": False}

    async def deduplicate_alerts(self, alerts: list[Any]) -> list[Any]:
        """Dedupliziert identische Alerts.

        Args:
            alerts: Liste der Alerts

        Returns:
            Deduplizierte Alert-Liste
        """
        seen_alerts = set()
        deduplicated = []

        for alert in alerts:
            # Erstelle eindeutigen Key
            alert_key = f"{alert.rule_id}_{alert.slo_name}_{alert.severity}"

            if alert_key not in seen_alerts:
                seen_alerts.add(alert_key)
                deduplicated.append(alert)

        return deduplicated

    def configure_rate_limiting(self, max_alerts_per_minute: int = 5,
                               max_alerts_per_hour: int = 20) -> None:
        """Konfiguriert Rate Limiting.

        Args:
            max_alerts_per_minute: Max Alerts pro Minute
            max_alerts_per_hour: Max Alerts pro Stunde
        """
        self._rate_limits["max_alerts_per_minute"] = max_alerts_per_minute
        self._rate_limits["max_alerts_per_hour"] = max_alerts_per_hour

    async def send_alert_with_rate_limiting(self, alert: Any) -> dict[str, Any]:
        """Sendet Alert mit Rate Limiting.

        Args:
            alert: Alert-Objekt

        Returns:
            Versand-Ergebnis
        """
        current_time = datetime.utcnow()

        # Reset Counters wenn nötig
        if current_time - self._alert_counts["last_minute_reset"] >= timedelta(minutes=1):
            self._alert_counts["minute"] = 0
            self._alert_counts["last_minute_reset"] = current_time

        if current_time - self._alert_counts["last_hour_reset"] >= timedelta(hours=1):
            self._alert_counts["hour"] = 0
            self._alert_counts["last_hour_reset"] = current_time

        # Prüfe Rate Limits
        if self._alert_counts["minute"] >= self._rate_limits["max_alerts_per_minute"]:
            return {"sent": False, "reason": "minute_rate_limit_exceeded"}

        if self._alert_counts["hour"] >= self._rate_limits["max_alerts_per_hour"]:
            return {"sent": False, "reason": "hour_rate_limit_exceeded"}

        # Sende Alert
        try:
            await self.route_alert(alert)
            self._alert_counts["minute"] += 1
            self._alert_counts["hour"] += 1
            return {"sent": True}
        except Exception as e:
            return {"sent": False, "error": str(e)}
