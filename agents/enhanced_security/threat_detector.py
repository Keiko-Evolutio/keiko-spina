# backend/agents/enhanced_security/threat_detector.py
"""Threat Detector

Enterprise-Grade Bedrohungserkennung mit:
- Anomalie-Erkennung
- Pattern-basierte Threat Detection
- Real-time Alerting
- Automated Response
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Bedrohungsstufen."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Typen von Bedrohungen."""

    BRUTE_FORCE = "brute_force"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_PAYLOAD = "malicious_payload"


@dataclass
class ThreatEvent:
    """Bedrohungs-Event."""

    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    threat_type: ThreatType = ThreatType.SUSPICIOUS_PATTERN
    threat_level: ThreatLevel = ThreatLevel.MEDIUM

    # Kontext
    agent_id: str | None = None
    user_id: str | None = None
    source_ip: str | None = None
    operation: str | None = None
    resource: str | None = None

    # Details
    description: str = ""
    indicators: list[str] = field(default_factory=list)
    confidence_score: float = 0.5

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    # Response
    auto_response_triggered: bool = False
    response_actions: list[str] = field(default_factory=list)


@dataclass
class SecurityAlert:
    """Sicherheitsalarm."""

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    threat_event: ThreatEvent = field(default_factory=lambda: ThreatEvent())

    # Alert-Details
    title: str = ""
    message: str = ""
    severity: ThreatLevel = ThreatLevel.MEDIUM

    # Status
    acknowledged: bool = False
    resolved: bool = False
    assigned_to: str | None = None

    # Response
    recommended_actions: list[str] = field(default_factory=list)
    automated_actions: list[str] = field(default_factory=list)


@dataclass
class ThreatResponse:
    """Bedrohungsreaktion."""

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Response-Details
    action_type: str = ""
    description: str = ""
    success: bool = False

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)


class ThreatDetector:
    """Enterprise Threat Detector"""

    def __init__(
        self,
        enable_anomaly_detection: bool = True,
        enable_pattern_detection: bool = True,
        enable_auto_response: bool = False,
        alert_threshold: float = 0.7
    ):
        """Initialisiert Threat Detector.

        Args:
            enable_anomaly_detection: Anomalie-Erkennung aktivieren
            enable_pattern_detection: Pattern-Erkennung aktivieren
            enable_auto_response: Automatische Reaktion aktivieren
            alert_threshold: Schwellwert für Alerts
        """
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_auto_response = enable_auto_response
        self.alert_threshold = alert_threshold

        # Threat-Tracking
        self._threat_events: list[ThreatEvent] = []
        self._active_alerts: dict[str, SecurityAlert] = {}
        self._response_history: list[ThreatResponse] = []

        # Anomalie-Erkennung
        self._baseline_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
        self._recent_activities: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Pattern-Erkennung
        self._suspicious_patterns: dict[str, dict[str, Any]] = {}
        self._pattern_counters: dict[str, int] = defaultdict(int)

        # Response-Handler
        self._response_handlers: dict[ThreatType, list[Callable]] = defaultdict(list)

        # Statistiken
        self._stats = {
            "threats_detected": 0,
            "alerts_generated": 0,
            "responses_triggered": 0,
            "false_positives": 0
        }

        # Initialisierung
        self._initialize_patterns()

        logger.info("Threat Detector initialisiert")

    def _initialize_patterns(self) -> None:
        """Initialisiert bekannte Bedrohungsmuster."""
        # Brute Force Pattern
        self._suspicious_patterns["brute_force"] = {
            "max_failures": 5,
            "time_window": 300,  # 5 Minuten
            "threat_type": ThreatType.BRUTE_FORCE,
            "threat_level": ThreatLevel.HIGH
        }

        # Rate Limit Abuse
        self._suspicious_patterns["rate_abuse"] = {
            "max_requests": 100,
            "time_window": 60,  # 1 Minute
            "threat_type": ThreatType.RATE_LIMIT_ABUSE,
            "threat_level": ThreatLevel.MEDIUM
        }

        # Privilege Escalation
        self._suspicious_patterns["privilege_escalation"] = {
            "suspicious_operations": [
                "admin_access", "user_creation", "permission_change"
            ],
            "threat_type": ThreatType.PRIVILEGE_ESCALATION,
            "threat_level": ThreatLevel.CRITICAL
        }

    @trace_function("threat.analyze_activity")
    async def analyze_activity(
        self,
        agent_id: str,
        operation: str,
        metadata: dict[str, Any] | None = None
    ) -> ThreatEvent | None:
        """Analysiert Aktivität auf Bedrohungen.

        Args:
            agent_id: Agent-ID
            operation: Durchgeführte Operation
            metadata: Zusätzliche Metadaten

        Returns:
            Erkannte Bedrohung oder None
        """
        try:
            # Aktivität aufzeichnen
            activity = {
                "timestamp": time.time(),
                "operation": operation,
                "metadata": metadata or {}
            }
            self._recent_activities[agent_id].append(activity)

            # Anomalie-Erkennung
            if self.enable_anomaly_detection:
                anomaly_threat = await self._detect_anomalies(agent_id, activity)
                if anomaly_threat:
                    return await self._process_threat(anomaly_threat)

            # Pattern-Erkennung
            if self.enable_pattern_detection:
                pattern_threat = await self._detect_patterns(agent_id, activity)
                if pattern_threat:
                    return await self._process_threat(pattern_threat)

            return None

        except Exception as e:
            logger.error(f"Threat-Analyse fehlgeschlagen: {e}")
            return None

    async def _detect_anomalies(
        self,
        agent_id: str,
        activity: dict[str, Any]
    ) -> ThreatEvent | None:
        """Erkennt Anomalien in Agent-Verhalten."""
        # Baseline-Metriken aktualisieren
        await self._update_baseline(agent_id, activity)

        # Anomalie-Checks
        recent_activities = list(self._recent_activities[agent_id])

        # Ungewöhnliche Aktivitätsfrequenz
        if len(recent_activities) >= 10:
            recent_timestamps = [a["timestamp"] for a in recent_activities[-10:]]
            time_span = max(recent_timestamps) - min(recent_timestamps)

            if time_span < 10:  # 10 Aktivitäten in weniger als 10 Sekunden
                return ThreatEvent(
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    agent_id=agent_id,
                    description="Ungewöhnlich hohe Aktivitätsfrequenz erkannt",
                    indicators=["high_frequency"],
                    confidence_score=0.6,
                    metadata={"time_span": time_span, "activity_count": 10}
                )

        # Ungewöhnliche Operationen
        operation = activity["operation"]
        baseline = self._baseline_metrics[agent_id]

        if "operations" in baseline:
            operation_frequency = baseline["operations"].get(operation, 0)
            total_operations = sum(baseline["operations"].values())

            if total_operations > 50 and operation_frequency / total_operations < 0.01:
                # Operation ist sehr selten für diesen Agent
                return ThreatEvent(
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.LOW,
                    agent_id=agent_id,
                    operation=operation,
                    description="Ungewöhnliche Operation für Agent erkannt",
                    indicators=["rare_operation"],
                    confidence_score=0.4,
                    metadata={"operation_frequency": operation_frequency}
                )

        return None

    async def _detect_patterns(
        self,
        agent_id: str,
        activity: dict[str, Any]
    ) -> ThreatEvent | None:
        """Erkennt bekannte Bedrohungsmuster."""
        operation = activity["operation"]
        timestamp = activity["timestamp"]

        # Brute Force Detection
        if "auth" in operation.lower() or "login" in operation.lower():
            if activity["metadata"].get("success") is False:
                pattern_key = f"brute_force_{agent_id}"
                self._pattern_counters[pattern_key] += 1

                pattern = self._suspicious_patterns["brute_force"]
                if self._pattern_counters[pattern_key] >= pattern["max_failures"]:
                    return ThreatEvent(
                        threat_type=pattern["threat_type"],
                        threat_level=pattern["threat_level"],
                        agent_id=agent_id,
                        operation=operation,
                        description="Brute Force Angriff erkannt",
                        indicators=["multiple_auth_failures"],
                        confidence_score=0.8,
                        metadata={"failure_count": self._pattern_counters[pattern_key]}
                    )

        # Rate Limit Abuse Detection
        recent_activities = [
            a for a in self._recent_activities[agent_id]
            if timestamp - a["timestamp"] <= 60  # Letzte Minute
        ]

        if len(recent_activities) >= self._suspicious_patterns["rate_abuse"]["max_requests"]:
            return ThreatEvent(
                threat_type=ThreatType.RATE_LIMIT_ABUSE,
                threat_level=ThreatLevel.MEDIUM,
                agent_id=agent_id,
                operation=operation,
                description="Rate Limit Missbrauch erkannt",
                indicators=["high_request_rate"],
                confidence_score=0.7,
                metadata={"request_count": len(recent_activities)}
            )

        # Privilege Escalation Detection
        escalation_ops = self._suspicious_patterns["privilege_escalation"]["suspicious_operations"]
        if any(op in operation.lower() for op in escalation_ops):
            return ThreatEvent(
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                threat_level=ThreatLevel.CRITICAL,
                agent_id=agent_id,
                operation=operation,
                description="Potentielle Privilege Escalation erkannt",
                indicators=["suspicious_admin_operation"],
                confidence_score=0.6,
                metadata={"operation": operation}
            )

        return None

    async def _update_baseline(
        self,
        agent_id: str,
        activity: dict[str, Any]
    ) -> None:
        """Aktualisiert Baseline-Metriken für Agent."""
        baseline = self._baseline_metrics[agent_id]

        # Operations-Häufigkeit
        if "operations" not in baseline:
            baseline["operations"] = defaultdict(int)
        baseline["operations"][activity["operation"]] += 1

        # Zeitbasierte Metriken
        if "hourly_activity" not in baseline:
            baseline["hourly_activity"] = defaultdict(int)

        hour = int(activity["timestamp"]) // 3600
        baseline["hourly_activity"][hour] += 1

        # Letzte Aktivität
        baseline["last_activity"] = activity["timestamp"]

    async def _process_threat(self, threat: ThreatEvent) -> ThreatEvent:
        """Verarbeitet erkannte Bedrohung."""
        # Threat speichern
        self._threat_events.append(threat)
        self._stats["threats_detected"] += 1

        logger.warning(
            f"Bedrohung erkannt: {threat.threat_type.value} "
            f"(Level: {threat.threat_level.value}, Agent: {threat.agent_id})"
        )

        # Alert generieren (falls Schwellwert erreicht)
        if threat.confidence_score >= self.alert_threshold:
            await self._generate_alert(threat)

        # Automatische Reaktion (falls aktiviert)
        if self.enable_auto_response and threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._trigger_auto_response(threat)

        return threat

    async def _generate_alert(self, threat: ThreatEvent) -> SecurityAlert:
        """Generiert Sicherheitsalarm."""
        alert = SecurityAlert(
            threat_event=threat,
            title=f"{threat.threat_type.value.replace('_', ' ').title()} erkannt",
            message=threat.description,
            severity=threat.threat_level,
            recommended_actions=self._get_recommended_actions(threat)
        )

        self._active_alerts[alert.alert_id] = alert
        self._stats["alerts_generated"] += 1

        logger.error(f"Sicherheitsalarm generiert: {alert.title} (ID: {alert.alert_id})")

        return alert

    async def _trigger_auto_response(self, threat: ThreatEvent) -> None:
        """Löst automatische Reaktion aus."""
        response_actions = []

        # Threat-spezifische Reaktionen
        if threat.threat_type == ThreatType.BRUTE_FORCE:
            response_actions.extend([
                "temporary_lockout",
                "increase_auth_requirements"
            ])
        elif threat.threat_type == ThreatType.RATE_LIMIT_ABUSE:
            response_actions.extend([
                "rate_limit_enforcement",
                "temporary_throttling"
            ])
        elif threat.threat_type == ThreatType.PRIVILEGE_ESCALATION:
            response_actions.extend([
                "revoke_elevated_permissions",
                "audit_recent_actions"
            ])

        # Reaktionen ausführen
        for action in response_actions:
            try:
                await self._execute_response_action(threat, action)
                threat.response_actions.append(action)
            except Exception as e:
                logger.error(f"Auto-Response-Aktion fehlgeschlagen ({action}): {e}")

        threat.auto_response_triggered = True
        self._stats["responses_triggered"] += 1

    async def _execute_response_action(
        self,
        threat: ThreatEvent,
        action: str
    ) -> None:
        """Führt Response-Aktion aus."""
        response = ThreatResponse(
            threat_id=threat.threat_id,
            action_type=action,
            description=f"Automatische Reaktion auf {threat.threat_type.value}"
        )

        # Vereinfachte Implementierung - in Produktion würden hier
        # echte Sicherheitsmaßnahmen implementiert
        if action == "temporary_lockout":
            # Agent temporär sperren
            logger.info(f"Agent {threat.agent_id} temporär gesperrt")
            response.success = True
        elif action == "rate_limit_enforcement":
            # Rate Limiting verstärken
            logger.info(f"Rate Limiting für Agent {threat.agent_id} verstärkt")
            response.success = True
        elif action == "revoke_elevated_permissions":
            # Erhöhte Berechtigungen entziehen
            logger.info(f"Erhöhte Berechtigungen für Agent {threat.agent_id} entzogen")
            response.success = True
        else:
            response.success = False

        self._response_history.append(response)

    @staticmethod
    def _get_recommended_actions(threat: ThreatEvent) -> list[str]:
        """Gibt empfohlene Aktionen für Bedrohung zurück.

        Args:
            threat: Erkannte Bedrohung

        Returns:
            Liste empfohlener Aktionen
        """
        actions = []

        if threat.threat_type == ThreatType.BRUTE_FORCE:
            actions.extend([
                "Agent-Zugang überprüfen",
                "Authentifizierungsrichtlinien verstärken",
                "Monitoring erhöhen"
            ])
        elif threat.threat_type == ThreatType.ANOMALOUS_BEHAVIOR:
            actions.extend([
                "Agent-Verhalten analysieren",
                "Baseline-Metriken überprüfen",
                "Manuell validieren"
            ])
        elif threat.threat_type == ThreatType.PRIVILEGE_ESCALATION:
            actions.extend([
                "Berechtigungen sofort überprüfen",
                "Audit-Log analysieren",
                "Incident Response einleiten"
            ])

        return actions

    async def process_violation(self, violation) -> None:
        """Verarbeitet Sicherheitsverletzung vom Security Manager."""
        # Violation in ThreatEvent konvertieren
        threat = ThreatEvent(
            threat_type=ThreatType.SUSPICIOUS_PATTERN,
            threat_level=ThreatLevel.MEDIUM,
            agent_id=violation.agent_id,
            operation=violation.operation,
            description=violation.description,
            indicators=["security_violation"],
            confidence_score=0.7,
            metadata={"violation_type": violation.violation_type.value}
        )

        await self._process_threat(threat)

    def get_threat_statistics(self) -> dict[str, Any]:
        """Gibt Threat-Statistiken zurück."""
        return {
            "threats_detected": self._stats["threats_detected"],
            "alerts_generated": self._stats["alerts_generated"],
            "responses_triggered": self._stats["responses_triggered"],
            "false_positives": self._stats["false_positives"],
            "active_alerts": len(self._active_alerts),
            "monitored_agents": len(self._baseline_metrics),
            "anomaly_detection_enabled": self.enable_anomaly_detection,
            "pattern_detection_enabled": self.enable_pattern_detection,
            "auto_response_enabled": self.enable_auto_response,
            "alert_threshold": self.alert_threshold
        }

    def get_active_alerts(self) -> list[SecurityAlert]:
        """Gibt aktive Alerts zurück."""
        return [
            alert for alert in self._active_alerts.values()
            if not alert.resolved
        ]

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Bestätigt Alert."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].acknowledged = True
            self._active_alerts[alert_id].assigned_to = user_id
            logger.info(f"Alert {alert_id} bestätigt von {user_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Löst Alert auf."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].resolved = True
            self._active_alerts[alert_id].assigned_to = user_id
            logger.info(f"Alert {alert_id} aufgelöst von {user_id}")
            return True
        return False
