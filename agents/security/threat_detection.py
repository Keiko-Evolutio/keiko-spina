# backend/kei_agents/security/threat_detection.py
"""Threat Detection und Anomalie-Erkennung für KEI-Agents.

Implementiert umfassende Bedrohungserkennung:
- ML-basierte Anomalie-Erkennung in Agent-Verhalten
- Intrusion Detection und Prevention
- Security Event Correlation
- Automated Threat Response
"""

import asyncio
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
from collections import defaultdict, deque

from .exceptions import AnomalyDetectionError, ThreatDetectionError

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat Level Enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        """Ermöglicht Vergleich von ThreatLevel-Werten."""
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        order = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other):
        """Ermöglicht <= Vergleich von ThreatLevel-Werten."""
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other):
        """Ermöglicht > Vergleich von ThreatLevel-Werten."""
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        order = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return order.index(self) > order.index(other)

    def __ge__(self, other):
        """Ermöglicht >= Vergleich von ThreatLevel-Werten."""
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self == other or self > other

    def __eq__(self, other):
        """Ermöglicht == Vergleich von ThreatLevel-Werten."""
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self.value == other.value

    def __hash__(self):
        """Ermöglicht Hashing von ThreatLevel-Werten."""
        return hash(self.value)


class SecurityEventType(Enum):
    """Security Event Types."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_PAYLOAD = "malicious_payload"

    @property
    def value(self) -> str:
        """Gibt den String-Wert des Enum zurück."""
        return self._value_


@dataclass
class SecurityEvent:
    """Security Event Data Structure."""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    source_ip: str | None = None
    target_agent: str | None = None
    user_id: str | None = None
    severity: str = "medium"
    details: dict[str, Any] | None = None


@dataclass
class AnomalyResult:
    """Anomalie-Erkennungsergebnis."""
    is_anomaly: bool
    confidence: float
    anomaly_type: str
    anomalous_features: list[str]
    baseline_deviation: float
    timestamp: datetime


@dataclass
class ThreatDetectionResult:
    """Threat Detection Ergebnis."""
    threat_detected: bool
    threat_level: ThreatLevel
    threat_type: str
    confidence: float
    indicators: list[str]
    recommended_actions: list[str]
    timestamp: datetime


class AnomalyDetector:
    """ML-basierte Anomalie-Erkennung für Agent-Verhalten."""

    def __init__(self, baseline_window: timedelta = timedelta(hours=24),
                 sensitivity_threshold: float = 2.0, learning_enabled: bool = True):
        """Initialisiert AnomalyDetector.

        Args:
            baseline_window: Zeitfenster für Baseline-Learning
            sensitivity_threshold: Sensitivitätsschwelle (Standard-Abweichungen)
            learning_enabled: Ob kontinuierliches Learning aktiviert ist
        """
        self.baseline_window = baseline_window
        self.sensitivity_threshold = sensitivity_threshold
        self.learning_enabled = learning_enabled

        # Baseline-Daten für verschiedene Agents
        self._baselines: dict[str, dict[str, Any]] = {}
        self._activity_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        logger.info(f"AnomalyDetector initialisiert mit Sensitivität {sensitivity_threshold}")

    async def learn_baseline(self, agent_id: str, activities: list[dict[str, Any]]) -> None:
        """Lernt Baseline-Verhalten für Agent.

        Args:
            agent_id: Agent-ID
            activities: Liste von Aktivitätsdaten
        """
        try:
            if not activities:
                raise AnomalyDetectionError("Keine Aktivitätsdaten für Baseline-Learning")

            # Extrahiere numerische Features
            features = {}
            for feature_name in ["duration", "cpu_usage", "memory_usage", "api_calls"]:
                values = [act.get(feature_name, 0) for act in activities if feature_name in act]
                if values:
                    features[feature_name] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }

            # Lerne Aktivitätsmuster
            action_counts = defaultdict(int)
            for activity in activities:
                action = activity.get("action", "unknown")
                action_counts[action] += 1

            # Speichere Baseline
            self._baselines[agent_id] = {
                "features": features,
                "action_patterns": dict(action_counts),
                "learned_at": datetime.now(UTC),
                "sample_count": len(activities)
            }

            logger.info(f"Baseline für Agent {agent_id} mit {len(activities)} Aktivitäten gelernt")

        except Exception as e:
            logger.error(f"Baseline-Learning für Agent {agent_id} fehlgeschlagen: {e}")
            raise AnomalyDetectionError(f"Baseline-Learning fehlgeschlagen: {e}")

    async def detect_anomaly(self, agent_id: str, activity: dict[str, Any]) -> AnomalyResult:
        """Erkennt Anomalien in Agent-Aktivität.

        Args:
            agent_id: Agent-ID
            activity: Aktuelle Aktivitätsdaten

        Returns:
            Anomalie-Erkennungsergebnis
        """
        try:
            if agent_id not in self._baselines:
                # Keine Baseline verfügbar - sammle Daten
                self._activity_history[agent_id].append(activity)
                return AnomalyResult(
                    is_anomaly=False,
                    confidence=0.0,
                    anomaly_type="no_baseline",
                    anomalous_features=[],
                    baseline_deviation=0.0,
                    timestamp=datetime.now(UTC)
                )

            baseline = self._baselines[agent_id]
            anomalous_features = []
            max_deviation = 0.0

            # Prüfe numerische Features
            for feature_name, baseline_stats in baseline["features"].items():
                if feature_name in activity:
                    current_value = activity[feature_name]
                    mean = baseline_stats["mean"]
                    std = baseline_stats["std"]

                    if std > 0:
                        deviation = abs(current_value - mean) / std
                        if deviation > self.sensitivity_threshold:
                            anomalous_features.append(feature_name)
                            max_deviation = max(max_deviation, deviation)

            # Prüfe Aktivitätsmuster
            current_action = activity.get("action", "unknown")
            if current_action not in baseline["action_patterns"]:
                anomalous_features.append("unknown_action")
                max_deviation = max(max_deviation, 3.0)  # Unbekannte Aktion ist verdächtig

            # Bestimme Anomalie
            is_anomaly = len(anomalous_features) > 0
            confidence = min(max_deviation / self.sensitivity_threshold, 1.0) if is_anomaly else 0.0

            # Update History für kontinuierliches Learning
            if self.learning_enabled:
                self._activity_history[agent_id].append(activity)

            return AnomalyResult(
                is_anomaly=is_anomaly,
                confidence=confidence,
                anomaly_type="behavioral" if is_anomaly else "normal",
                anomalous_features=anomalous_features,
                baseline_deviation=max_deviation,
                timestamp=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Anomalie-Erkennung für Agent {agent_id} fehlgeschlagen: {e}")
            raise AnomalyDetectionError(f"Anomalie-Erkennung fehlgeschlagen: {e}")

    async def learn_activity_pattern(self, agent_id: str, pattern: list[dict[str, Any]]) -> None:
        """Lernt Aktivitätsmuster für Agent.

        Args:
            agent_id: Agent-ID
            pattern: Aktivitätsmuster-Daten
        """
        if agent_id not in self._baselines:
            self._baselines[agent_id] = {"action_patterns": {}}

        for item in pattern:
            action = item.get("action")
            frequency = item.get("frequency", 1)
            if action:
                self._baselines[agent_id]["action_patterns"][action] = frequency

    async def detect_pattern_anomaly(self, agent_id: str,
                                   current_pattern: list[dict[str, Any]]) -> AnomalyResult:
        """Erkennt Anomalien in Aktivitätsmustern.

        Args:
            agent_id: Agent-ID
            current_pattern: Aktuelles Aktivitätsmuster

        Returns:
            Anomalie-Erkennungsergebnis
        """
        if agent_id not in self._baselines:
            return AnomalyResult(
                is_anomaly=False,
                confidence=0.0,
                anomaly_type="no_baseline",
                anomalous_features=[],
                baseline_deviation=0.0,
                timestamp=datetime.now(UTC)
            )

        baseline_patterns = self._baselines[agent_id].get("action_patterns", {})
        anomalous_features = []

        for item in current_pattern:
            action = item.get("action")
            frequency = item.get("frequency", 1)

            if action not in baseline_patterns:
                anomalous_features.append(action)
            elif frequency > baseline_patterns[action] * 2:  # Mehr als doppelt so häufig
                anomalous_features.append(f"{action}_high_frequency")

        is_anomaly = len(anomalous_features) > 0
        confidence = min(len(anomalous_features) / len(current_pattern), 1.0) if is_anomaly else 0.0

        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_type="pattern",
            anomalous_features=anomalous_features,
            baseline_deviation=confidence * 3.0,
            timestamp=datetime.now(UTC)
        )

    async def learn_temporal_pattern(self, agent_id: str, timestamps: list[datetime]) -> None:
        """Lernt zeitliche Aktivitätsmuster.

        Args:
            agent_id: Agent-ID
            timestamps: Liste von Aktivitätszeitpunkten
        """
        if agent_id not in self._baselines:
            self._baselines[agent_id] = {}

        # Extrahiere Stunden-Pattern
        hours = [ts.hour for ts in timestamps]
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1

        self._baselines[agent_id]["temporal_pattern"] = dict(hour_counts)

    async def detect_temporal_anomaly(self, agent_id: str, timestamp: datetime) -> AnomalyResult:
        """Erkennt zeitliche Anomalien.

        Args:
            agent_id: Agent-ID
            timestamp: Zeitpunkt der Aktivität

        Returns:
            Anomalie-Erkennungsergebnis
        """
        if agent_id not in self._baselines or "temporal_pattern" not in self._baselines[agent_id]:
            return AnomalyResult(
                is_anomaly=False,
                confidence=0.0,
                anomaly_type="no_temporal_baseline",
                anomalous_features=[],
                baseline_deviation=0.0,
                timestamp=datetime.now(UTC)
            )

        temporal_pattern = self._baselines[agent_id]["temporal_pattern"]
        current_hour = timestamp.hour

        # Prüfe ob Stunde in normalem Pattern
        if current_hour not in temporal_pattern or temporal_pattern[current_hour] < 2:
            return AnomalyResult(
                is_anomaly=True,
                confidence=0.8,
                anomaly_type="temporal",
                anomalous_features=["unusual_time"],
                baseline_deviation=3.0,
                timestamp=datetime.now(UTC)
            )

        return AnomalyResult(
            is_anomaly=False,
            confidence=0.0,
            anomaly_type="normal",
            anomalous_features=[],
            baseline_deviation=0.0,
            timestamp=datetime.now(UTC)
        )

    def get_baseline(self, agent_id: str) -> dict[str, Any] | None:
        """Gibt Baseline für Agent zurück.

        Args:
            agent_id: Agent-ID

        Returns:
            Baseline-Daten oder None
        """
        return self._baselines.get(agent_id)


class IntrusionDetector:
    """Intrusion Detection und Prevention System."""

    def __init__(self, enable_network_monitoring: bool = True,
                 enable_file_monitoring: bool = True,
                 enable_process_monitoring: bool = True):
        """Initialisiert IntrusionDetector.

        Args:
            enable_network_monitoring: Netzwerk-Monitoring aktivieren
            enable_file_monitoring: Datei-Monitoring aktivieren
            enable_process_monitoring: Prozess-Monitoring aktivieren
        """
        self.enable_network_monitoring = enable_network_monitoring
        self.enable_file_monitoring = enable_file_monitoring
        self.enable_process_monitoring = enable_process_monitoring

        # Tracking für Angriffsmuster
        self._failed_attempts: dict[str, list[datetime]] = defaultdict(list)
        self._suspicious_ips: set = set()

        logger.info("IntrusionDetector initialisiert")

    async def detect_unauthorized_access(self, access_event: dict[str, Any]) -> ThreatDetectionResult:
        """Erkennt unbefugte Zugriffe.

        Args:
            access_event: Zugriffs-Event-Daten

        Returns:
            Threat Detection Ergebnis
        """
        try:
            threat_indicators = []
            threat_level = ThreatLevel.LOW

            # Prüfe Credentials
            if access_event.get("credentials") == "invalid":
                threat_indicators.append("invalid_credentials")
                threat_level = ThreatLevel.HIGH

            # Prüfe User-Agent
            user_agent = access_event.get("user_agent", "")
            if any(suspicious in user_agent.lower() for suspicious in ["scanner", "bot", "crawler"]):
                threat_indicators.append("suspicious_user_agent")
                threat_level = max(threat_level, ThreatLevel.MEDIUM)

            # Prüfe Source IP
            source_ip = access_event.get("source_ip")
            if source_ip in self._suspicious_ips:
                threat_indicators.append("known_malicious_ip")
                threat_level = ThreatLevel.CRITICAL

            threat_detected = len(threat_indicators) > 0

            return ThreatDetectionResult(
                threat_detected=threat_detected,
                threat_level=threat_level,
                threat_type="unauthorized_access",
                confidence=0.8 if threat_detected else 0.0,
                indicators=threat_indicators,
                recommended_actions=["block_ip", "alert_security_team"] if threat_detected else [],
                timestamp=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Unauthorized access detection fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Unauthorized access detection fehlgeschlagen: {e}")

    async def detect_brute_force(self, failed_attempts: list[dict[str, Any]]) -> ThreatDetectionResult:
        """Erkennt Brute-Force-Angriffe.

        Args:
            failed_attempts: Liste fehlgeschlagener Login-Versuche

        Returns:
            Threat Detection Ergebnis
        """
        try:
            if not failed_attempts:
                return ThreatDetectionResult(
                    threat_detected=False,
                    threat_level=ThreatLevel.LOW,
                    threat_type="brute_force",
                    confidence=0.0,
                    indicators=[],
                    recommended_actions=[],
                    timestamp=datetime.now(UTC)
                )

            # Gruppiere nach Source IP
            ip_attempts = defaultdict(list)
            for attempt in failed_attempts:
                source_ip = attempt.get("source_ip")
                if source_ip:
                    ip_attempts[source_ip].append(attempt)

            # Prüfe auf Brute-Force-Pattern
            for source_ip, attempts in ip_attempts.items():
                if len(attempts) >= 5:  # 5+ fehlgeschlagene Versuche
                    # Prüfe Zeitfenster
                    timestamps = [datetime.fromisoformat(a["timestamp"]) if isinstance(a["timestamp"], str)
                                else a["timestamp"] for a in attempts]
                    time_span = max(timestamps) - min(timestamps)

                    if time_span <= timedelta(minutes=10):  # Innerhalb 10 Minuten
                        self._suspicious_ips.add(source_ip)

                        return ThreatDetectionResult(
                            threat_detected=True,
                            threat_level=ThreatLevel.CRITICAL,
                            threat_type="brute_force",
                            confidence=0.9,
                            indicators=["rapid_failed_attempts", "short_time_window", f"source_ip:{source_ip}", f"attempt_count:{len(attempts)}"],
                            recommended_actions=["block_ip", "alert_security_team", "increase_monitoring"],
                            timestamp=datetime.now(UTC)
                        )

            return ThreatDetectionResult(
                threat_detected=False,
                threat_level=ThreatLevel.LOW,
                threat_type="brute_force",
                confidence=0.0,
                indicators=[],
                recommended_actions=[],
                timestamp=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Brute-force detection fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Brute-force detection fehlgeschlagen: {e}")

    @staticmethod
    async def detect_malicious_payload(payload_event: dict[str, Any]) -> ThreatDetectionResult:
        """Erkennt schädliche Payloads.

        Args:
            payload_event: Payload-Event-Daten

        Returns:
            Threat Detection Ergebnis
        """
        try:
            threat_indicators = []
            attack_types = []

            payload = payload_event.get("payload", {})

            # Prüfe auf Command Injection
            if "command" in payload:
                command = str(payload["command"]).lower()
                dangerous_commands = ["rm -rf", "del /f", "format", "shutdown", "reboot"]
                if any(cmd in command for cmd in dangerous_commands):
                    threat_indicators.append("command_injection")
                    attack_types.append("command_injection")

            # Prüfe auf Script Injection
            if "script" in payload:
                script = str(payload["script"]).lower()
                if "eval(" in script or "exec(" in script or "base64.decode" in script:
                    threat_indicators.append("script_injection")
                    attack_types.append("script_injection")

            # Prüfe auf Path Traversal
            if "data" in payload:
                data = str(payload["data"])
                if "../" in data or "..\\" in data:
                    threat_indicators.append("path_traversal")
                    attack_types.append("path_traversal")

            threat_detected = len(threat_indicators) > 0
            threat_level = ThreatLevel.CRITICAL if threat_detected else ThreatLevel.LOW

            return ThreatDetectionResult(
                threat_detected=threat_detected,
                threat_level=threat_level,
                threat_type="malicious_payload",
                confidence=0.9 if threat_detected else 0.0,
                indicators=threat_indicators + [f"attack_types:{','.join(attack_types)}"] if attack_types else threat_indicators,
                recommended_actions=["block_request", "quarantine_agent", "alert_security_team"] if threat_detected else [],
                timestamp=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Malicious payload detection fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Malicious payload detection fehlgeschlagen: {e}")

    @staticmethod
    async def detect_privilege_escalation(escalation_event: dict[str, Any]) -> ThreatDetectionResult:
        """Erkennt Privilege-Escalation-Versuche.

        Args:
            escalation_event: Privilege-Escalation-Event

        Returns:
            Threat Detection Ergebnis
        """
        try:
            requested_permissions = escalation_event.get("requested_permissions", [])
            current_permissions = escalation_event.get("current_permissions", [])
            justification = escalation_event.get("justification", "")

            threat_indicators = []

            # Prüfe auf excessive Permissions
            high_privilege_permissions = ["admin_access", "system_modify", "agent_control", "root_access"]
            excessive_requests = [p for p in requested_permissions if p in high_privilege_permissions]

            if excessive_requests:
                threat_indicators.append("excessive_permissions")

            # Prüfe Justification
            if not justification or len(justification) < 10:
                threat_indicators.append("insufficient_justification")

            # Prüfe Permission-Sprung
            if len(requested_permissions) > len(current_permissions) * 2:
                threat_indicators.append("large_permission_increase")

            threat_detected = len(threat_indicators) > 0
            threat_level = ThreatLevel.HIGH if threat_detected else ThreatLevel.LOW

            return ThreatDetectionResult(
                threat_detected=threat_detected,
                threat_level=threat_level,
                threat_type="privilege_escalation",
                confidence=0.7 if threat_detected else 0.0,
                indicators=threat_indicators,
                recommended_actions=["deny_request", "require_approval", "audit_user"] if threat_detected else [],
                timestamp=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Privilege escalation detection fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Privilege escalation detection fehlgeschlagen: {e}")

    @staticmethod
    async def analyze_network_traffic(network_events: list[dict[str, Any]]) -> ThreatDetectionResult:
        """Analysiert Netzwerkverkehr auf verdächtige Aktivitäten.

        Args:
            network_events: Liste von Netzwerk-Events

        Returns:
            Threat Detection Ergebnis
        """
        try:
            if not network_events:
                return ThreatDetectionResult(
                    threat_detected=False,
                    threat_level=ThreatLevel.LOW,
                    threat_type="network_analysis",
                    confidence=0.0,
                    indicators=[],
                    recommended_actions=[],
                    timestamp=datetime.now(UTC)
                )

            threat_indicators = []
            attack_types = []

            # Gruppiere nach Source Agent
            agent_connections = defaultdict(list)
            for event in network_events:
                source_agent = event.get("source_agent")
                if source_agent:
                    agent_connections[source_agent].append(event)

            # Prüfe auf Port Scanning
            for source_agent, connections in agent_connections.items():
                unique_ports = set(conn.get("port") for conn in connections)
                if len(unique_ports) > 10:  # Viele verschiedene Ports
                    threat_indicators.append("port_scanning")
                    attack_types.append("port_scanning")

            # Prüfe auf Data Exfiltration
            total_data_size = sum(event.get("data_size", 0) for event in network_events)
            if total_data_size > 100 * 1024 * 1024:  # > 100MB
                threat_indicators.append("data_exfiltration")
                attack_types.append("data_exfiltration")

            threat_detected = len(threat_indicators) > 0
            threat_level = ThreatLevel.HIGH if threat_detected else ThreatLevel.LOW

            return ThreatDetectionResult(
                threat_detected=threat_detected,
                threat_level=threat_level,
                threat_type="network_analysis",
                confidence=0.8 if threat_detected else 0.0,
                indicators=threat_indicators + [f"attack_types:{','.join(attack_types)}"] if attack_types else threat_indicators,
                recommended_actions=["network_isolation", "traffic_analysis"] if threat_detected else [],
                timestamp=datetime.now(UTC)
            )

        except Exception as e:
            logger.error(f"Network traffic analysis fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Network traffic analysis fehlgeschlagen: {e}")


class SecurityEventCorrelator:
    """Korreliert Security Events für erweiterte Threat Detection."""

    def __init__(self, correlation_window: timedelta = timedelta(minutes=30)):
        """Initialisiert SecurityEventCorrelator.

        Args:
            correlation_window: Zeitfenster für Event-Korrelation
        """
        self.correlation_window = correlation_window
        self._event_history: list[SecurityEvent] = []

        logger.info(f"SecurityEventCorrelator initialisiert mit {correlation_window} Fenster")

    async def correlate_events(self, events: list[SecurityEvent]) -> list[dict[str, Any]]:
        """Korreliert Security Events.

        Args:
            events: Liste von Security Events

        Returns:
            Liste korrelierter Event-Gruppen
        """
        try:
            if not events:
                return []

            # Füge Events zur History hinzu
            self._event_history.extend(events)

            # Bereinige alte Events
            cutoff_time = datetime.now(UTC) - self.correlation_window
            self._event_history = [e for e in self._event_history if e.timestamp > cutoff_time]

            # Gruppiere Events nach verschiedenen Kriterien
            correlations = []

            # Korrelation nach Source IP
            ip_groups = self._correlate_by_source_ip()
            correlations.extend(ip_groups)

            # Korrelation nach Target Agent
            agent_groups = self._correlate_by_target_agent()
            correlations.extend(agent_groups)

            # Korrelation nach Event-Sequenz
            sequence_groups = self._correlate_by_sequence()
            correlations.extend(sequence_groups)

            return correlations

        except Exception as e:
            logger.error(f"Event-Korrelation fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Event-Korrelation fehlgeschlagen: {e}")

    def _correlate_by_source_ip(self) -> list[dict[str, Any]]:
        """Korreliert Events nach Source IP."""
        ip_events = defaultdict(list)

        for event in self._event_history:
            if event.source_ip:
                ip_events[event.source_ip].append(event)

        correlations = []
        for source_ip, events in ip_events.items():
            if len(events) >= 3:  # Mindestens 3 Events von derselben IP
                correlations.append({
                    "correlation_type": "source_ip",
                    "correlation_key": source_ip,
                    "events": events,
                    "event_count": len(events),
                    "time_span": max(e.timestamp for e in events) - min(e.timestamp for e in events),
                    "threat_level": "high" if len(events) >= 5 else "medium"
                })

        return correlations

    def _correlate_by_target_agent(self) -> list[dict[str, Any]]:
        """Korreliert Events nach Target Agent."""
        agent_events = defaultdict(list)

        for event in self._event_history:
            if event.target_agent:
                agent_events[event.target_agent].append(event)

        correlations = []
        for target_agent, events in agent_events.items():
            if len(events) >= 2:  # Mindestens 2 Events für denselben Agent
                correlations.append({
                    "correlation_type": "target_agent",
                    "correlation_key": target_agent,
                    "events": events,
                    "event_count": len(events),
                    "event_types": list(set(e.event_type.value for e in events)),
                    "threat_level": "high" if len(events) >= 4 else "medium"
                })

        return correlations

    def _correlate_by_sequence(self) -> list[dict[str, Any]]:
        """Korreliert Events nach verdächtigen Sequenzen."""
        correlations = []

        # Suche nach Attack-Sequenzen
        attack_sequences = [
            [SecurityEventType.AUTHENTICATION_FAILURE, SecurityEventType.AUTHORIZATION_VIOLATION],
            [SecurityEventType.INTRUSION_ATTEMPT, SecurityEventType.ANOMALOUS_BEHAVIOR],
            [SecurityEventType.AUTHORIZATION_VIOLATION, SecurityEventType.DATA_EXFILTRATION]
        ]

        for sequence in attack_sequences:
            matching_events = self._find_event_sequence(sequence)
            if matching_events:
                correlations.append({
                    "correlation_type": "attack_sequence",
                    "correlation_key": "_".join(s.value for s in sequence),
                    "events": matching_events,
                    "sequence": [s.value for s in sequence],
                    "threat_level": "critical"
                })

        return correlations

    def _find_event_sequence(self, sequence: list[SecurityEventType]) -> list[SecurityEvent]:
        """Findet Event-Sequenzen in der History."""
        if len(sequence) < 2:
            return []

        # Sortiere Events nach Timestamp
        sorted_events = sorted(self._event_history, key=lambda e: e.timestamp)

        for i in range(len(sorted_events) - len(sequence) + 1):
            # Prüfe ob Sequenz ab Position i passt
            match = True
            for j, expected_type in enumerate(sequence):
                if sorted_events[i + j].event_type != expected_type:
                    match = False
                    break

            if match:
                # Prüfe Zeitfenster
                time_span = sorted_events[i + len(sequence) - 1].timestamp - sorted_events[i].timestamp
                if time_span <= self.correlation_window:
                    return sorted_events[i:i + len(sequence)]

        return []


class ThreatDetector:
    """Zentraler Threat Detector für KEI-Agents."""

    def __init__(self, enable_ml_detection: bool = True,
                 enable_correlation: bool = True,
                 threat_threshold: float = 0.7):
        """Initialisiert ThreatDetector.

        Args:
            enable_ml_detection: ML-basierte Erkennung aktivieren
            enable_correlation: Event-Korrelation aktivieren
            threat_threshold: Schwellwert für Threat-Erkennung
        """
        self.enable_ml_detection = enable_ml_detection
        self.enable_correlation = enable_correlation
        self.threat_threshold = threat_threshold

        # Initialisiere Komponenten
        self.anomaly_detector = AnomalyDetector() if enable_ml_detection else None
        self.intrusion_detector = IntrusionDetector()
        self.event_correlator = SecurityEventCorrelator() if enable_correlation else None

        # Threat-Tracking
        self._active_threats: dict[str, ThreatDetectionResult] = {}
        self._threat_history: list[ThreatDetectionResult] = []

        logger.info(f"ThreatDetector initialisiert mit Threshold {threat_threshold}")

    async def detect_anomaly(self, agent_id: str, activity: dict[str, Any]) -> AnomalyResult:
        """Erkennt Anomalien in Agent-Aktivität.

        Args:
            agent_id: Agent-ID
            activity: Aktivitätsdaten

        Returns:
            Anomalie-Erkennungsergebnis
        """
        if not self.anomaly_detector:
            raise ThreatDetectionError("ML-basierte Anomalie-Erkennung nicht aktiviert")

        return await self.anomaly_detector.detect_anomaly(agent_id, activity)

    async def detect_intrusion(self, intrusion_event: dict[str, Any]) -> ThreatDetectionResult:
        """Erkennt Intrusion-Versuche.

        Args:
            intrusion_event: Intrusion-Event-Daten

        Returns:
            Threat Detection Ergebnis
        """
        event_type = intrusion_event.get("type", "unknown")

        if event_type == "unauthorized_access":
            return await self.intrusion_detector.detect_unauthorized_access(intrusion_event)
        if event_type == "brute_force":
            failed_attempts = intrusion_event.get("failed_attempts", [])
            return await self.intrusion_detector.detect_brute_force(failed_attempts)
        if event_type == "malicious_payload":
            return await self.intrusion_detector.detect_malicious_payload(intrusion_event)
        if event_type == "privilege_escalation":
            return await self.intrusion_detector.detect_privilege_escalation(intrusion_event)
        if event_type == "network_analysis":
            network_events = intrusion_event.get("network_events", [])
            return await self.intrusion_detector.analyze_network_traffic(network_events)
        raise ThreatDetectionError(f"Unbekannter Intrusion-Event-Typ: {event_type}")

    async def analyze_security_event(self, security_event: SecurityEvent) -> ThreatDetectionResult:
        """Analysiert Security Event auf Bedrohungen.

        Args:
            security_event: Security Event

        Returns:
            Threat Detection Ergebnis
        """
        try:
            threat_indicators = []
            threat_level = ThreatLevel.LOW

            # Analysiere Event-Typ
            if security_event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
                threat_indicators.append("auth_failure")
                threat_level = ThreatLevel.MEDIUM
            elif security_event.event_type == SecurityEventType.AUTHORIZATION_VIOLATION:
                threat_indicators.append("authz_violation")
                threat_level = ThreatLevel.HIGH
            elif security_event.event_type == SecurityEventType.INTRUSION_ATTEMPT:
                threat_indicators.append("intrusion_attempt")
                threat_level = ThreatLevel.CRITICAL

            # Analysiere Severity
            if security_event.severity in ["high", "critical"]:
                threat_level = max(threat_level, ThreatLevel.HIGH)

            # Event-Korrelation wenn aktiviert
            if self.event_correlator:
                correlations = await self.event_correlator.correlate_events([security_event])
                if correlations:
                    threat_indicators.append("correlated_events")
                    threat_level = ThreatLevel.CRITICAL

            threat_detected = len(threat_indicators) > 0
            confidence = 0.8 if threat_detected else 0.0

            result = ThreatDetectionResult(
                threat_detected=threat_detected,
                threat_level=threat_level,
                threat_type=security_event.event_type.value if hasattr(security_event.event_type, "value") else str(security_event.event_type),
                confidence=confidence,
                indicators=threat_indicators,
                recommended_actions=ThreatDetector.get_recommended_actions(threat_level),
                timestamp=datetime.now(UTC)
            )

            # Speichere aktive Threats
            if threat_detected and confidence >= self.threat_threshold:
                threat_id = f"{security_event.event_id}_{security_event.event_type.value}"
                self._active_threats[threat_id] = result
                self._threat_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Security Event Analyse fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Security Event Analyse fehlgeschlagen: {e}")

    @staticmethod
    def get_threat_level(threat_indicators: list[str]) -> str:
        """Bestimmt Threat Level basierend auf Indikatoren.

        Args:
            threat_indicators: Liste von Threat-Indikatoren

        Returns:
            Threat Level als String
        """
        if not threat_indicators:
            return ThreatLevel.LOW.value

        critical_indicators = ["intrusion_attempt", "data_exfiltration", "malicious_payload"]
        high_indicators = ["authz_violation", "privilege_escalation", "brute_force"]

        if any(indicator in critical_indicators for indicator in threat_indicators):
            return ThreatLevel.CRITICAL.value
        if any(indicator in high_indicators for indicator in threat_indicators):
            return ThreatLevel.HIGH.value
        if len(threat_indicators) >= 3:
            return ThreatLevel.MEDIUM.value
        return ThreatLevel.LOW.value

    @staticmethod
    def get_recommended_actions(threat_level: ThreatLevel) -> list[str]:
        """Gibt empfohlene Aktionen basierend auf Threat Level zurück."""
        if threat_level == ThreatLevel.CRITICAL:
            return ["immediate_isolation", "alert_security_team", "incident_response", "forensic_analysis"]
        if threat_level == ThreatLevel.HIGH:
            return ["increase_monitoring", "alert_security_team", "access_review"]
        if threat_level == ThreatLevel.MEDIUM:
            return ["log_event", "monitor_closely", "notify_admin"]
        return ["log_event"]

    async def get_active_threats(self) -> dict[str, ThreatDetectionResult]:
        """Gibt aktive Bedrohungen zurück.

        Returns:
            Dict mit aktiven Threats
        """
        return self._active_threats.copy()

    async def clear_threat(self, threat_id: str) -> bool:
        """Markiert Threat als behoben.

        Args:
            threat_id: Threat-ID

        Returns:
            True wenn erfolgreich, sonst False
        """
        if threat_id in self._active_threats:
            del self._active_threats[threat_id]
            logger.info(f"Threat {threat_id} als behoben markiert")
            return True
        return False


class ThreatResponse:
    """Threat Response System."""

    def __init__(self, response_timeout: timedelta = timedelta(minutes=5),
                 enable_automated_response: bool = True):
        """Initialisiert ThreatResponse.

        Args:
            response_timeout: Response-Timeout
            enable_automated_response: Automatische Response aktivieren
        """
        self.response_timeout = response_timeout
        self.enable_automated_response = enable_automated_response

        logger.info("ThreatResponse initialisiert")

    async def respond_to_threat(self, threat_result: ThreatDetectionResult) -> dict[str, Any]:
        """Reagiert auf erkannte Bedrohung.

        Args:
            threat_result: Threat Detection Ergebnis

        Returns:
            Response-Ergebnis
        """
        try:
            if not self.enable_automated_response:
                return {
                    "automated": False,
                    "manual_actions_required": threat_result.recommended_actions
                }

            # Führe empfohlene Aktionen aus
            executed_actions = []
            for action in threat_result.recommended_actions:
                result = await self._execute_response_action(action, threat_result)
                executed_actions.append(result)

            return {
                "automated": True,
                "threat_level": threat_result.threat_level.value,
                "actions_executed": len(executed_actions),
                "results": executed_actions
            }

        except Exception as e:
            logger.error(f"Threat Response fehlgeschlagen: {e}")
            raise ThreatDetectionError(f"Threat Response fehlgeschlagen: {e}")

    async def _execute_response_action(self, action: str,
                                     _threat_result: ThreatDetectionResult) -> dict[str, Any]:
        """Führt Response-Aktion aus."""
        try:
            if action == "immediate_isolation":
                return await self._isolate_service(_threat_result)
            if action == "alert_security_team":
                return await self._alert_security_team(_threat_result)
            if action == "block_ip":
                return await self._block_ip(_threat_result)
            if action == "increase_monitoring":
                return await self._increase_monitoring(_threat_result)
            return {"action": action, "status": "not_implemented"}

        except Exception as e:
            return {"action": action, "status": "failed", "error": str(e)}

    @staticmethod
    async def _isolate_service(_threat_result: ThreatDetectionResult) -> dict[str, Any]:
        """Isoliert Service (Mock)."""
        await asyncio.sleep(1)
        return {"action": "isolate_service", "status": "success"}

    @staticmethod
    async def _alert_security_team(_threat_result: ThreatDetectionResult) -> dict[str, Any]:
        """Alarmiert Security Team (Mock)."""
        await asyncio.sleep(0.5)
        return {"action": "alert_security_team", "status": "success"}

    @staticmethod
    async def _block_ip(_threat_result: ThreatDetectionResult) -> dict[str, Any]:
        """Blockiert IP-Adresse (Mock)."""
        await asyncio.sleep(0.2)
        return {"action": "block_ip", "status": "success"}

    @staticmethod
    async def _increase_monitoring(_threat_result: ThreatDetectionResult) -> dict[str, Any]:
        """Erhöht Monitoring (Mock)."""
        await asyncio.sleep(0.1)
        return {"action": "increase_monitoring", "status": "success"}


@dataclass
class DetectionRule:
    """Detection Rule Data Structure."""
    rule_id: str
    rule_name: str
    rule_type: str
    conditions: dict[str, Any]
    severity: str
    enabled: bool = True
    description: str | None = None
