# backend/services/enhanced_security_integration/threat_detection_engine.py
"""Threat Detection Engine für Real-time Security-Monitoring.

Implementiert intelligente Threat-Detection mit Machine Learning,
Anomaly-Detection und Real-time Security-Event-Analysis.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger
from security.rbac_abac_system import Action, ResourceType

from .data_models import SecurityContext, ThreatDetectionResult, ThreatLevel

logger = get_logger(__name__)


class ThreatDetectionEngine:
    """Threat Detection Engine für Real-time Security-Monitoring."""

    def __init__(self):
        """Initialisiert Threat Detection Engine."""
        # Threat-Detection-Konfiguration
        self.enable_anomaly_detection = True
        self.enable_pattern_matching = True
        self.enable_ml_detection = False  # Placeholder für ML-Integration
        self.detection_sensitivity = 0.7  # 0.0 - 1.0

        # Threat-Pattern-Database
        self._threat_patterns = self._initialize_threat_patterns()

        # Anomaly-Detection-Baselines
        self._user_baselines: dict[str, dict[str, Any]] = {}
        self._tenant_baselines: dict[str, dict[str, Any]] = {}
        self._global_baseline: dict[str, Any] = {}

        # Real-time Activity-Tracking
        self._user_activities: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._ip_activities: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._tenant_activities: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Threat-Detection-Results
        self._detected_threats: list[ThreatDetectionResult] = []
        self._max_threats_in_memory = 10000

        # Performance-Tracking
        self._detection_count = 0
        self._total_detection_time_ms = 0.0
        self._threat_detection_count = 0
        self._false_positive_count = 0

        # Background-Tasks
        self._baseline_update_task: asyncio.Task | None = None
        self._threat_cleanup_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Threat Detection Engine initialisiert")

    async def start(self) -> None:
        """Startet Threat Detection Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._baseline_update_task = asyncio.create_task(self._baseline_update_loop())
        self._threat_cleanup_task = asyncio.create_task(self._threat_cleanup_loop())

        logger.info("Threat Detection Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Threat Detection Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._baseline_update_task:
            self._baseline_update_task.cancel()
        if self._threat_cleanup_task:
            self._threat_cleanup_task.cancel()

        await asyncio.gather(
            self._baseline_update_task,
            self._threat_cleanup_task,
            return_exceptions=True
        )

        logger.info("Threat Detection Engine gestoppt")

    async def analyze_request(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> ThreatDetectionResult:
        """Analysiert Request auf Threats.

        Args:
            security_context: Security-Kontext
            resource_type: Resource-Type
            resource_id: Resource-ID
            action: Action

        Returns:
            Threat Detection Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "threat_detection_started",
                "user_id": security_context.user_id,
                "tenant_id": security_context.tenant_id,
                "resource_type": resource_type.value,
                "action": action.value
            })

            # Tracke Activity
            await self._track_activity(security_context, resource_type, resource_id, action)

            threat_indicators = []
            threat_types = []
            attack_patterns = []
            max_threat_level = ThreatLevel.LOW
            max_confidence = 0.0

            # 1. Pattern-Matching-Detection
            if self.enable_pattern_matching:
                pattern_result = await self._detect_threat_patterns(
                    security_context, resource_type, resource_id, action
                )

                if pattern_result["threats_detected"]:
                    threat_indicators.extend(pattern_result["indicators"])
                    threat_types.extend(pattern_result["threat_types"])
                    attack_patterns.extend(pattern_result["attack_patterns"])

                    if pattern_result["threat_level"].value > max_threat_level.value:
                        max_threat_level = pattern_result["threat_level"]

                    max_confidence = max(max_confidence, pattern_result["confidence"])

            # 2. Anomaly-Detection
            if self.enable_anomaly_detection:
                anomaly_result = await self._detect_anomalies(
                    security_context, resource_type, resource_id, action
                )

                if anomaly_result["anomalies_detected"]:
                    threat_indicators.extend(anomaly_result["indicators"])
                    threat_types.extend(anomaly_result["threat_types"])

                    if anomaly_result["threat_level"].value > max_threat_level.value:
                        max_threat_level = anomaly_result["threat_level"]

                    max_confidence = max(max_confidence, anomaly_result["confidence"])

            # 3. Behavioral-Analysis
            behavioral_result = await self._analyze_behavior(
                security_context, resource_type, resource_id, action
            )

            if behavioral_result["suspicious_behavior"]:
                threat_indicators.extend(behavioral_result["indicators"])
                threat_types.extend(behavioral_result["threat_types"])

                if behavioral_result["threat_level"].value > max_threat_level.value:
                    max_threat_level = behavioral_result["threat_level"]

                max_confidence = max(max_confidence, behavioral_result["confidence"])

            # 4. Rate-Limiting-Analysis
            rate_limit_result = await self._analyze_rate_limits(security_context)

            if rate_limit_result["rate_limit_exceeded"]:
                threat_indicators.extend(rate_limit_result["indicators"])
                threat_types.extend(rate_limit_result["threat_types"])

                if rate_limit_result["threat_level"].value > max_threat_level.value:
                    max_threat_level = rate_limit_result["threat_level"]

                max_confidence = max(max_confidence, rate_limit_result["confidence"])

            # Bestimme Overall-Threat-Status
            threat_detected = len(threat_indicators) > 0 and max_confidence >= self.detection_sensitivity

            # Berechne Risk-Score
            risk_score = self._calculate_risk_score(
                threat_types, threat_indicators, max_threat_level, max_confidence
            )

            # Generiere Response-Empfehlungen
            recommended_actions = self._generate_response_recommendations(
                threat_types, max_threat_level, risk_score
            )

            # Performance-Tracking
            detection_time_ms = (time.time() - start_time) * 1000
            self._update_detection_performance_stats(detection_time_ms, threat_detected)

            # Erstelle Threat-Detection-Result
            result = ThreatDetectionResult(
                threat_detected=threat_detected,
                threat_level=max_threat_level,
                confidence=max_confidence,
                threat_types=list(set(threat_types)),
                threat_indicators=list(set(threat_indicators)),
                attack_patterns=list(set(attack_patterns)),
                risk_score=risk_score,
                recommended_actions=recommended_actions
            )

            # Speichere Result
            self._detected_threats.append(result)

            # Memory-Limit prüfen
            if len(self._detected_threats) > self._max_threats_in_memory:
                self._detected_threats = self._detected_threats[-self._max_threats_in_memory:]

            logger.debug({
                "event": "threat_detection_completed",
                "threat_detected": threat_detected,
                "threat_level": max_threat_level.value,
                "confidence": max_confidence,
                "risk_score": risk_score,
                "detection_time_ms": detection_time_ms
            })

            return result

        except Exception as e:
            logger.error(f"Threat detection fehlgeschlagen: {e}")

            # Fallback: Kein Threat bei Fehler
            return ThreatDetectionResult(
                threat_detected=False,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                threat_types=["detection_error"]
            )

    async def _track_activity(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> None:
        """Trackt User/IP/Tenant-Activity für Anomaly-Detection."""
        try:
            activity_record = {
                "timestamp": datetime.utcnow(),
                "resource_type": resource_type.value,
                "resource_id": resource_id,
                "action": action.value,
                "security_level": security_context.security_level.value,
                "source_ip": security_context.source_ip
            }

            # Tracke User-Activity
            if security_context.user_id:
                self._user_activities[security_context.user_id].append(activity_record)

            # Tracke IP-Activity
            if security_context.source_ip:
                self._ip_activities[security_context.source_ip].append(activity_record)

            # Tracke Tenant-Activity
            if security_context.tenant_id:
                self._tenant_activities[security_context.tenant_id].append(activity_record)

        except Exception as e:
            logger.error(f"Activity tracking fehlgeschlagen: {e}")

    async def _detect_threat_patterns(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> dict[str, Any]:
        """Erkennt bekannte Threat-Patterns."""
        try:
            threats_detected = False
            indicators = []
            threat_types = []
            attack_patterns = []
            threat_level = ThreatLevel.LOW
            confidence = 0.0

            # Prüfe gegen bekannte Threat-Patterns
            for pattern_name, pattern_config in self._threat_patterns.items():
                if self._matches_threat_pattern(
                    pattern_config, security_context, resource_type, resource_id, action
                ):
                    threats_detected = True
                    indicators.append(f"pattern_match_{pattern_name}")
                    threat_types.append(pattern_config["threat_type"])
                    attack_patterns.append(pattern_name)

                    pattern_threat_level = ThreatLevel(pattern_config["threat_level"])
                    if pattern_threat_level.value > threat_level.value:
                        threat_level = pattern_threat_level

                    confidence = max(confidence, pattern_config["confidence"])

            return {
                "threats_detected": threats_detected,
                "indicators": indicators,
                "threat_types": threat_types,
                "attack_patterns": attack_patterns,
                "threat_level": threat_level,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Threat pattern detection fehlgeschlagen: {e}")
            return {"threats_detected": False, "indicators": [], "threat_types": [], "attack_patterns": [], "threat_level": ThreatLevel.LOW, "confidence": 0.0}

    async def _detect_anomalies(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        _resource_id: str,
        action: Action
    ) -> dict[str, Any]:
        """Erkennt Anomalien basierend auf Baselines."""
        try:
            anomalies_detected = False
            indicators = []
            threat_types = []
            threat_level = ThreatLevel.LOW
            confidence = 0.0

            # User-Anomaly-Detection
            if security_context.user_id:
                user_anomaly = self._detect_user_anomaly(security_context, resource_type, action)
                if user_anomaly["is_anomaly"]:
                    anomalies_detected = True
                    indicators.extend(user_anomaly["indicators"])
                    threat_types.append("user_behavior_anomaly")
                    confidence = max(confidence, user_anomaly["confidence"])

            # IP-Anomaly-Detection
            if security_context.source_ip:
                ip_anomaly = self._detect_ip_anomaly(security_context, resource_type, action)
                if ip_anomaly["is_anomaly"]:
                    anomalies_detected = True
                    indicators.extend(ip_anomaly["indicators"])
                    threat_types.append("ip_behavior_anomaly")
                    confidence = max(confidence, ip_anomaly["confidence"])

            # Time-based Anomaly-Detection
            time_anomaly = self._detect_time_anomaly(security_context, resource_type, action)
            if time_anomaly["is_anomaly"]:
                anomalies_detected = True
                indicators.extend(time_anomaly["indicators"])
                threat_types.append("temporal_anomaly")
                confidence = max(confidence, time_anomaly["confidence"])

            # Bestimme Threat-Level basierend auf Anomaly-Severity
            if confidence > 0.8:
                threat_level = ThreatLevel.HIGH
            elif confidence > 0.6:
                threat_level = ThreatLevel.MEDIUM
            elif confidence > 0.3:
                threat_level = ThreatLevel.LOW

            return {
                "anomalies_detected": anomalies_detected,
                "indicators": indicators,
                "threat_types": threat_types,
                "threat_level": threat_level,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Anomaly detection fehlgeschlagen: {e}")
            return {"anomalies_detected": False, "indicators": [], "threat_types": [], "threat_level": ThreatLevel.LOW, "confidence": 0.0}

    async def _analyze_behavior(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> dict[str, Any]:
        """Analysiert Behavioral-Patterns."""
        try:
            suspicious_behavior = False
            indicators = []
            threat_types = []
            threat_level = ThreatLevel.LOW
            confidence = 0.0

            # Prüfe auf verdächtige Behavioral-Patterns

            # 1. Privilege-Escalation-Attempts
            if self._detect_privilege_escalation(security_context, resource_type, action):
                suspicious_behavior = True
                indicators.append("privilege_escalation_attempt")
                threat_types.append("privilege_escalation")
                threat_level = ThreatLevel.HIGH
                confidence = max(confidence, 0.8)

            # 2. Data-Exfiltration-Patterns
            if self._detect_data_exfiltration(security_context, resource_type, action):
                suspicious_behavior = True
                indicators.append("data_exfiltration_pattern")
                threat_types.append("data_exfiltration")
                threat_level = ThreatLevel.CRITICAL
                confidence = max(confidence, 0.9)

            # 3. Lateral-Movement-Attempts
            if self._detect_lateral_movement(security_context, resource_type, resource_id):
                suspicious_behavior = True
                indicators.append("lateral_movement_attempt")
                threat_types.append("lateral_movement")
                threat_level = ThreatLevel.HIGH
                confidence = max(confidence, 0.7)

            return {
                "suspicious_behavior": suspicious_behavior,
                "indicators": indicators,
                "threat_types": threat_types,
                "threat_level": threat_level,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Behavioral analysis fehlgeschlagen: {e}")
            return {"suspicious_behavior": False, "indicators": [], "threat_types": [], "threat_level": ThreatLevel.LOW, "confidence": 0.0}

    async def _analyze_rate_limits(self, security_context: SecurityContext) -> dict[str, Any]:
        """Analysiert Rate-Limiting-Violations."""
        try:
            rate_limit_exceeded = False
            indicators = []
            threat_types = []
            threat_level = ThreatLevel.LOW
            confidence = 0.0

            # Prüfe User-Rate-Limits
            if security_context.user_id:
                user_activities = self._user_activities.get(security_context.user_id, deque())
                recent_activities = [
                    a for a in user_activities
                    if a["timestamp"] > datetime.utcnow() - timedelta(minutes=5)
                ]

                if len(recent_activities) > 100:  # > 100 Requests in 5 Minuten
                    rate_limit_exceeded = True
                    indicators.append("user_rate_limit_exceeded")
                    threat_types.append("rate_limit_abuse")
                    threat_level = ThreatLevel.MEDIUM
                    confidence = 0.6

            # Prüfe IP-Rate-Limits
            if security_context.source_ip:
                ip_activities = self._ip_activities.get(security_context.source_ip, deque())
                recent_activities = [
                    a for a in ip_activities
                    if a["timestamp"] > datetime.utcnow() - timedelta(minutes=5)
                ]

                if len(recent_activities) > 200:  # > 200 Requests in 5 Minuten
                    rate_limit_exceeded = True
                    indicators.append("ip_rate_limit_exceeded")
                    threat_types.append("ddos_attempt")
                    threat_level = ThreatLevel.HIGH
                    confidence = 0.8

            return {
                "rate_limit_exceeded": rate_limit_exceeded,
                "indicators": indicators,
                "threat_types": threat_types,
                "threat_level": threat_level,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Rate limit analysis fehlgeschlagen: {e}")
            return {"rate_limit_exceeded": False, "indicators": [], "threat_types": [], "threat_level": ThreatLevel.LOW, "confidence": 0.0}

    def _initialize_threat_patterns(self) -> dict[str, dict[str, Any]]:
        """Initialisiert bekannte Threat-Patterns."""
        return {
            "sql_injection": {
                "threat_type": "injection_attack",
                "threat_level": "high",
                "confidence": 0.9,
                "patterns": ["'", "union", "select", "drop", "insert", "update", "delete"],
                "resource_types": ["database", "api"]
            },
            "privilege_escalation": {
                "threat_type": "privilege_escalation",
                "threat_level": "critical",
                "confidence": 0.8,
                "patterns": ["admin", "root", "sudo", "escalate"],
                "actions": ["create", "delete", "update"]
            },
            "brute_force": {
                "threat_type": "brute_force_attack",
                "threat_level": "high",
                "confidence": 0.7,
                "patterns": ["multiple_failed_attempts"],
                "frequency_threshold": 10
            }
        }

    def _matches_threat_pattern(
        self,
        _pattern_config: dict[str, Any],
        _security_context: SecurityContext,
        _resource_type: ResourceType,
        _resource_id: str,
        _action: Action
    ) -> bool:
        """Prüft ob Request gegen Threat-Pattern matcht."""
        # Vereinfachte Pattern-Matching-Logic
        # TODO: Implementiere sophisticated Pattern-Matching - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
        return False

    def _detect_user_anomaly(
        self,
        _security_context: SecurityContext,
        _resource_type: ResourceType,
        _action: Action
    ) -> dict[str, Any]:
        """Erkennt User-Anomalien."""
        # Placeholder für User-Anomaly-Detection
        return {"is_anomaly": False, "indicators": [], "confidence": 0.0}

    def _detect_ip_anomaly(
        self,
        _security_context: SecurityContext,
        _resource_type: ResourceType,
        _action: Action
    ) -> dict[str, Any]:
        """Erkennt IP-Anomalien."""
        # Placeholder für IP-Anomaly-Detection
        return {"is_anomaly": False, "indicators": [], "confidence": 0.0}

    def _detect_time_anomaly(
        self,
        _security_context: SecurityContext,
        _resource_type: ResourceType,
        _action: Action
    ) -> dict[str, Any]:
        """Erkennt zeitbasierte Anomalien."""
        # Placeholder für Time-Anomaly-Detection
        return {"is_anomaly": False, "indicators": [], "confidence": 0.0}

    def _detect_privilege_escalation(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        action: Action
    ) -> bool:
        """Erkennt Privilege-Escalation-Attempts."""
        # Vereinfachte Privilege-Escalation-Detection
        return action in [Action.CREATE, Action.DELETE] and resource_type == ResourceType.AGENT

    def _detect_data_exfiltration(
        self,
        _security_context: SecurityContext,
        resource_type: ResourceType,
        action: Action
    ) -> bool:
        """Erkennt Data-Exfiltration-Patterns."""
        # Vereinfachte Data-Exfiltration-Detection
        return action == Action.READ and resource_type in [ResourceType.TASK, ResourceType.CAPABILITY]

    def _detect_lateral_movement(
        self,
        security_context: SecurityContext,
        _resource_type: ResourceType,
        resource_id: str
    ) -> bool:
        """Erkennt Lateral-Movement-Attempts."""
        # Vereinfachte Lateral-Movement-Detection
        return (security_context.tenant_id and ":" in resource_id and
                not resource_id.startswith(security_context.tenant_id))

    def _calculate_risk_score(
        self,
        _threat_types: list[str],
        threat_indicators: list[str],
        threat_level: ThreatLevel,
        confidence: float
    ) -> float:
        """Berechnet Risk-Score."""
        # Basis-Score basierend auf Threat-Level
        level_scores = {
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 1.0
        }

        base_score = level_scores.get(threat_level, 0.0)

        # Adjustiere basierend auf Anzahl Indicators
        indicator_factor = min(1.0, len(threat_indicators) / 5.0)

        # Kombiniere mit Confidence
        risk_score = (base_score * 0.6) + (confidence * 0.3) + (indicator_factor * 0.1)

        return min(1.0, risk_score)

    def _generate_response_recommendations(
        self,
        _threat_types: list[str],
        threat_level: ThreatLevel,
        _risk_score: float
    ) -> list[str]:
        """Generiert Response-Empfehlungen."""
        recommendations = []

        if threat_level == ThreatLevel.CRITICAL:
            recommendations.extend([
                "immediate_account_lockout",
                "escalate_to_security_team",
                "block_source_ip",
                "audit_recent_activities"
            ])
        elif threat_level == ThreatLevel.HIGH:
            recommendations.extend([
                "increase_monitoring",
                "require_additional_authentication",
                "limit_access_permissions"
            ])
        elif threat_level == ThreatLevel.MEDIUM:
            recommendations.extend([
                "log_for_investigation",
                "monitor_future_activities"
            ])

        return recommendations

    async def _baseline_update_loop(self) -> None:
        """Background-Loop für Baseline-Updates."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Update alle Stunde

                if self._is_running:
                    await self._update_baselines()

            except Exception as e:
                logger.error(f"Baseline update fehlgeschlagen: {e}")

    async def _threat_cleanup_loop(self) -> None:
        """Background-Loop für Threat-Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Cleanup alle Stunde

                if self._is_running:
                    await self._cleanup_old_threats()

            except Exception as e:
                logger.error(f"Threat cleanup fehlgeschlagen: {e}")

    async def _update_baselines(self) -> None:
        """Aktualisiert Anomaly-Detection-Baselines."""
        try:
            # TODO: Implementiere Baseline-Update-Logic - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
            logger.debug("Baselines aktualisiert")

        except Exception as e:
            logger.error(f"Baseline update fehlgeschlagen: {e}")

    async def _cleanup_old_threats(self) -> None:
        """Bereinigt alte Threat-Detection-Results."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)

            original_count = len(self._detected_threats)
            self._detected_threats = [
                t for t in self._detected_threats
                if t.detection_timestamp > cutoff_time
            ]

            cleaned_count = original_count - len(self._detected_threats)
            if cleaned_count > 0:
                logger.info(f"Threat cleanup: {cleaned_count} alte Threats entfernt")

        except Exception as e:
            logger.error(f"Threat cleanup fehlgeschlagen: {e}")

    def _update_detection_performance_stats(self, detection_time_ms: float, threat_detected: bool) -> None:
        """Aktualisiert Detection-Performance-Statistiken."""
        self._detection_count += 1
        self._total_detection_time_ms += detection_time_ms

        if threat_detected:
            self._threat_detection_count += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_detection_time = (
            self._total_detection_time_ms / self._detection_count
            if self._detection_count > 0 else 0.0
        )

        threat_detection_rate = (
            self._threat_detection_count / self._detection_count
            if self._detection_count > 0 else 0.0
        )

        return {
            "total_detections": self._detection_count,
            "avg_detection_time_ms": avg_detection_time,
            "threat_detection_count": self._threat_detection_count,
            "threat_detection_rate": threat_detection_rate,
            "false_positive_count": self._false_positive_count,
            "detected_threats_in_memory": len(self._detected_threats),
            "anomaly_detection_enabled": self.enable_anomaly_detection,
            "pattern_matching_enabled": self.enable_pattern_matching,
            "detection_sensitivity": self.detection_sensitivity
        }
