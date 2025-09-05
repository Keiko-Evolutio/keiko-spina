# backend/agents/monitoring/alert_manager.py
"""Alert Manager für das Agent-Framework.

Alerting-System mit:
- Flexible Alert-Regeln und -Channels
- Multi-Channel-Benachrichtigungen
- Alert-Deduplizierung und Rate-Limiting
- Callback-System für externe Integration
- Umfassende Metriken und Statistiken
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..logging_utils import StructuredLogger

logger = StructuredLogger("alert_manager")


class AlertConstants:
    """Konstanten für Alert Manager."""

    # Zeitkonstanten (in Sekunden)
    DEFAULT_DEDUPLICATION_WINDOW = 300.0  # 5 Minuten
    DEFAULT_ESCALATION_TIMEOUT = 1800.0  # 30 Minuten
    DEFAULT_EVALUATION_WINDOW = 60.0  # 1 Minute
    RATE_LIMIT_WINDOW = 60.0  # 1 Minute
    HOURS_TO_SECONDS = 3600

    # Limits
    DEFAULT_MAX_ALERTS_PER_MINUTE = 10
    DEFAULT_ALERT_RETENTION_HOURS = 168  # 1 Woche
    DEFAULT_MAX_ALERT_HISTORY = 10000

    # Defaults
    DEFAULT_METRIC_VALUE = 0.0
    DEFAULT_THRESHOLD = 0.0

    # ID-Generation
    TIMESTAMP_MULTIPLIER = 1000


class AlertLevel(Enum):
    """Alert-Schweregrade."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Konfiguration für Alert Manager."""

    # Alert-Verhalten
    enable_alerts: bool = True
    enable_deduplication: bool = True
    deduplication_window: float = AlertConstants.DEFAULT_DEDUPLICATION_WINDOW

    # Escalation
    enable_escalation: bool = True
    escalation_timeout: float = AlertConstants.DEFAULT_ESCALATION_TIMEOUT

    # Rate Limiting
    enable_rate_limiting: bool = True
    max_alerts_per_minute: int = AlertConstants.DEFAULT_MAX_ALERTS_PER_MINUTE

    # Retention und History
    alert_retention_hours: int = AlertConstants.DEFAULT_ALERT_RETENTION_HOURS
    max_alert_history: int = AlertConstants.DEFAULT_MAX_ALERT_HISTORY

    # Callback-System
    enable_callbacks: bool = True


@dataclass
class Alert:
    """Alert-Datenstruktur."""

    alert_id: str
    title: str
    message: str
    level: AlertLevel
    timestamp: float = field(default_factory=time.time)

    # Kontext
    component: str | None = None
    agent_id: str | None = None
    operation: str | None = None
    capability: str | None = None
    upstream_id: str | None = None

    # Status
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    resolved_at: float | None = None

    # Metriken
    metric_name: str = ""
    metric_value: float = AlertConstants.DEFAULT_METRIC_VALUE
    threshold: float = AlertConstants.DEFAULT_THRESHOLD

    # Metadaten
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Alert zu Dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "component": self.component,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "capability": self.capability,
            "upstream_id": self.upstream_id,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "escalated": self.escalated,
            "resolved_at": self.resolved_at,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class AlertRule:
    """Alert-Regel-Definition."""

    rule_id: str
    name: str
    condition: str
    level: AlertLevel
    enabled: bool = True

    # Trigger-Konfiguration
    threshold: float | None = None
    evaluation_window: float = AlertConstants.DEFAULT_EVALUATION_WINDOW

    # Benachrichtigung
    channels: list[str] = field(default_factory=list)

    # Metadaten
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)


class AlertChannel(ABC):
    """Abstrakte Basis-Klasse für Alert-Channels."""

    def __init__(self, channel_id: str, name: str):
        self.channel_id = channel_id
        self.name = name
        self.enabled = True

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Sendet Alert über Channel."""


class LogAlertChannel(AlertChannel):
    """Log-basierter Alert-Channel."""

    async def send_alert(self, alert: Alert) -> bool:
        """Sendet Alert an Log."""
        try:
            log_level = {
                AlertLevel.INFO: "info",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "error",
                AlertLevel.CRITICAL: "critical"
            }.get(alert.level, "info")

            log_method = getattr(logger, log_level)
            log_method(
                f"ALERT: {alert.title}",
                extra_data={
                    "alert_id": alert.alert_id,
                    "message": alert.message,
                    "level": alert.level.value,
                    "component": alert.component,
                    "agent_id": alert.agent_id,
                    "tags": alert.tags
                }
            )
            return True
        except Exception as e:
            logger.error("Fehler beim Senden von Log-Alert", error=e)
            return False


class AlertManager:
    """Alert Manager für das Agent-Framework."""

    def __init__(self, config: AlertConfig):
        """Initialisiert Alert Manager.

        Args:
            config: Alert-Manager-Konfiguration
        """
        self.config = config

        # Alert-Speicher
        self._alerts: dict[str, Alert] = {}
        self._alert_rules: dict[str, AlertRule] = {}
        self._alert_channels: dict[str, AlertChannel] = {}
        self._alert_history: deque[Alert] = deque(maxlen=config.max_alert_history)

        # Deduplication
        self._alert_fingerprints: dict[str, float] = {}

        # Rate Limiting
        self._alert_counts: dict[str, list[float]] = {}

        # Callback-System
        self._alert_callbacks: dict[AlertLevel, list[Callable[[Alert], Awaitable[None]]]] = {
            level: [] for level in AlertLevel
        }

        # Legacy-Kompatibilität
        self._last_alert_times: dict[str, float] = {}
        self._alert_count_by_rule: dict[str, int] = defaultdict(int)

        # Standard Log-Channel registrieren
        self.register_channel(LogAlertChannel("log", "Log Channel"))

        logger.info("Alert Manager initialisiert")

    def register_channel(self, channel: AlertChannel) -> None:
        """Registriert Alert-Channel.

        Args:
            channel: Alert-Channel-Instanz
        """
        self._alert_channels[channel.channel_id] = channel
        logger.info(f"Alert-Channel registriert: {channel.name}")

    def register_rule(self, rule: AlertRule) -> None:
        """Registriert Alert-Regel.

        Args:
            rule: Alert-Regel
        """
        self._alert_rules[rule.rule_id] = rule
        logger.info(f"Alert-Regel registriert: {rule.name}")

    def register_alert_callback(
        self,
        level: AlertLevel,
        callback: Callable[[Alert], Awaitable[None]]
    ) -> None:
        """Registriert Callback für Alert-Level.

        Args:
            level: Alert-Level
            callback: Async Callback-Funktion
        """
        if not self.config.enable_callbacks:
            logger.warning("Callbacks sind deaktiviert")
            return

        self._alert_callbacks[level].append(callback)
        logger.info(f"Alert-Callback für Level {level.value} registriert")

    async def create_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        component: str | None = None,
        agent_id: str | None = None,
        operation: str | None = None,
        capability: str | None = None,
        upstream_id: str | None = None,
        metric_name: str = "",
        metric_value: float = AlertConstants.DEFAULT_METRIC_VALUE,
        threshold: float = AlertConstants.DEFAULT_THRESHOLD,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """Erstellt neuen Alert.

        Args:
            title: Alert-Titel
            message: Alert-Nachricht
            level: Alert-Level
            component: Komponente
            agent_id: Agent-ID
            operation: Operation
            capability: Capability
            upstream_id: Upstream-ID
            metric_name: Metrik-Name
            metric_value: Metrik-Wert
            threshold: Schwellenwert
            tags: Tags
            metadata: Metadaten

        Returns:
            Alert-ID wenn erstellt, None wenn dedupliziert
        """
        if not self.config.enable_alerts:
            return None

        # Rate Limiting prüfen
        if self._is_rate_limited():
            logger.warning("Alert-Rate-Limit erreicht, Alert verworfen")
            return None

        # Alert-ID generieren
        alert_id = f"alert_{int(time.time() * AlertConstants.TIMESTAMP_MULTIPLIER)}"

        # Deduplication prüfen
        fingerprint = AlertManager._generate_fingerprint(title, component, agent_id)
        if self._is_duplicate(fingerprint):
            logger.debug(f"Alert dedupliziert: {title}")
            return None

        # Alert erstellen
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            level=level,
            component=component,
            agent_id=agent_id,
            operation=operation,
            capability=capability,
            upstream_id=upstream_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            tags=tags or {},
            metadata=metadata or {}
        )

        # Alert speichern
        self._alerts[alert_id] = alert
        self._alert_history.append(alert)

        # Fingerprint für Deduplication speichern
        if self.config.enable_deduplication:
            self._alert_fingerprints[fingerprint] = time.time()

        # Rate Limiting aktualisieren
        self._update_rate_limiting()

        # Alert senden
        await self._send_alert(alert)

        # Callbacks ausführen
        await self._execute_alert_callbacks(alert)

        logger.info(
            f"Alert erstellt: {title}",
            extra_data={
                "alert_id": alert_id,
                "level": level.value,
                "component": component,
                "agent_id": agent_id,
                "capability": capability,
                "metric_name": metric_name
            }
        )

        return alert_id

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Bestätigt Alert.

        Args:
            alert_id: Alert-ID
            user_id: Benutzer-ID

        Returns:
            True wenn erfolgreich bestätigt
        """
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        alert.acknowledged = True
        alert.metadata["acknowledged_by"] = user_id
        alert.metadata["acknowledged_at"] = time.time()

        logger.info(
            f"Alert bestätigt: {alert.title}",
            extra_data={
                "alert_id": alert_id,
                "acknowledged_by": user_id
            }
        )

        return True

    async def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Löst Alert auf.

        Args:
            alert_id: Alert-ID
            user_id: Benutzer-ID

        Returns:
            True wenn erfolgreich aufgelöst
        """
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        alert.resolved = True
        alert.metadata["resolved_by"] = user_id
        alert.metadata["resolved_at"] = time.time()

        logger.info(
            f"Alert aufgelöst: {alert.title}",
            extra_data={
                "alert_id": alert_id,
                "resolved_by": user_id
            }
        )

        return True

    def get_active_alerts(self) -> list[Alert]:
        """Gibt aktive Alerts zurück."""
        return [
            alert for alert in self._alerts.values()
            if not alert.resolved
        ]

    def get_alerts_by_level(self, level: AlertLevel) -> list[Alert]:
        """Gibt Alerts nach Level zurück."""
        return [
            alert for alert in self._alerts.values()
            if alert.level == level
        ]

    def get_alert_statistics(self) -> dict[str, Any]:
        """Gibt Alert-Statistiken zurück."""
        alerts = list(self._alerts.values())
        history = list(self._alert_history)

        if not alerts:
            return {
                "total_alerts": 0,
                "active_alerts": 0,
                "by_level": {},
                "by_component": {},
                "by_capability": {},
                "history_size": len(history)
            }

        active_alerts = [a for a in alerts if not a.resolved]

        # Nach Level gruppieren
        by_level = {}
        for level in AlertLevel:
            count = len([a for a in alerts if a.level == level])
            by_level[level.value] = count

        # Nach Komponente gruppieren
        by_component = {}
        for alert in alerts:
            component = alert.component or "unknown"
            by_component[component] = by_component.get(component, 0) + 1

        # Nach Capability gruppieren
        by_capability = {}
        for alert in alerts:
            if alert.capability:
                by_capability[alert.capability] = by_capability.get(alert.capability, 0) + 1

        # Statistiken
        return {
            "total_alerts": len(alerts),
            "active_alerts": len(active_alerts),
            "acknowledged_alerts": len([a for a in alerts if a.acknowledged]),
            "resolved_alerts": len([a for a in alerts if a.resolved]),
            "escalated_alerts": len([a for a in alerts if a.escalated]),
            "by_level": by_level,
            "by_component": by_component,
            "by_capability": by_capability,
            "channels_count": len(self._alert_channels),
            "rules_count": len(self._alert_rules),
            "callbacks_count": sum(len(callbacks) for callbacks in self._alert_callbacks.values()),
            "history_size": len(history),
            "deduplication_enabled": self.config.enable_deduplication,
            "rate_limiting_enabled": self.config.enable_rate_limiting
        }

    async def _send_alert(self, alert: Alert) -> None:
        """Sendet Alert über alle Channels."""
        if not self._alert_channels:
            logger.warning("Keine Alert-Channels konfiguriert")
            return

        # Alert über alle aktiven Channels senden
        for channel in self._alert_channels.values():
            if not channel.enabled:
                continue

            try:
                success = await channel.send_alert(alert)
                if not success:
                    logger.warning(f"Alert-Versand fehlgeschlagen: {channel.name}")
            except Exception as e:
                logger.error(f"Fehler beim Alert-Versand über {channel.name}", error=e)

    @staticmethod
    def _generate_fingerprint(
        title: str,
        component: str | None,
        agent_id: str | None
    ) -> str:
        """Generiert Fingerprint für Deduplication.

        Args:
            title: Alert-Titel
            component: Komponenten-Name
            agent_id: Agent-ID

        Returns:
            Eindeutiger Fingerprint-String
        """
        parts = [title]
        if component:
            parts.append(f"comp:{component}")
        if agent_id:
            parts.append(f"agent:{agent_id}")
        return "|".join(parts)

    def _is_duplicate(self, fingerprint: str) -> bool:
        """Prüft ob Alert ein Duplikat ist."""
        if not self.config.enable_deduplication:
            return False

        if fingerprint not in self._alert_fingerprints:
            return False

        last_time = self._alert_fingerprints[fingerprint]
        return (time.time() - last_time) < self.config.deduplication_window

    def _is_rate_limited(self) -> bool:
        """Prüft Rate Limiting."""
        if not self.config.enable_rate_limiting:
            return False

        current_time = time.time()
        minute_ago = current_time - AlertConstants.RATE_LIMIT_WINDOW

        # Alte Einträge entfernen
        if "global" not in self._alert_counts:
            self._alert_counts["global"] = []

        self._alert_counts["global"] = [
            t for t in self._alert_counts["global"]
            if t > minute_ago
        ]

        return len(self._alert_counts["global"]) >= self.config.max_alerts_per_minute

    def _update_rate_limiting(self) -> None:
        """Aktualisiert Rate Limiting Counter."""
        if not self.config.enable_rate_limiting:
            return

        if "global" not in self._alert_counts:
            self._alert_counts["global"] = []

        self._alert_counts["global"].append(time.time())

    def cleanup_old_alerts(self) -> int:
        """Bereinigt alte Alerts.

        Returns:
            Anzahl der bereinigten Alerts
        """
        if not self.config.alert_retention_hours:
            return 0

        cutoff_time = time.time() - (self.config.alert_retention_hours * AlertConstants.HOURS_TO_SECONDS)
        old_alert_ids = [
            alert_id for alert_id, alert in self._alerts.items()
            if alert.timestamp < cutoff_time
        ]

        for alert_id in old_alert_ids:
            del self._alerts[alert_id]

        # Alte Fingerprints bereinigen
        old_fingerprints = [
            fp for fp, timestamp in self._alert_fingerprints.items()
            if timestamp < cutoff_time
        ]

        for fp in old_fingerprints:
            del self._alert_fingerprints[fp]

        if old_alert_ids:
            logger.info(f"{len(old_alert_ids)} alte Alerts bereinigt")

        return len(old_alert_ids)

    async def _execute_alert_callbacks(self, alert: Alert) -> None:
        """Führt registrierte Callbacks für Alert-Level aus.

        Args:
            alert: Alert-Instanz
        """
        if not self.config.enable_callbacks:
            return

        callbacks = self._alert_callbacks.get(alert.level, [])
        if not callbacks:
            return

        for callback in callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(
                    "Fehler beim Ausführen von Alert-Callback",
                    error=e,
                    extra_data={
                        "alert_id": alert.alert_id,
                        "level": alert.level.value,
                        "callback": str(callback)
                    }
                )

    def get_alert_history(self) -> list[Alert]:
        """Gibt Alert-History zurück."""
        return list(self._alert_history)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Gibt Metriken-Zusammenfassung zurück."""
        return self.get_alert_statistics()
