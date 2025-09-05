# backend/policy_engine/safety_guardrails.py
"""Safety Guardrails Engine für Keiko Personal Assistant

Implementiert Content-Safety-Checks, Toxicity-Detection, Bias-Detection,
Harmful-Content-Filtering und Real-time Safety-Monitoring.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class SafetyLevel(str, Enum):
    """Safety-Level für Content-Bewertung."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    UNSAFE = "unsafe"


class ViolationType(str, Enum):
    """Typen von Safety-Verletzungen."""
    TOXICITY = "toxicity"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    ILLEGAL_ACTIVITY = "illegal_activity"
    MISINFORMATION = "misinformation"
    BIAS = "bias"
    DISCRIMINATION = "discrimination"
    PRIVACY_VIOLATION = "privacy_violation"


@dataclass
class SafetyViolation:
    """Repräsentiert eine Safety-Verletzung."""
    violation_type: ViolationType
    severity: SafetyLevel
    confidence: float  # 0.0 - 1.0
    description: str
    detected_content: str
    suggested_action: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        """Prüft, ob Verletzung blockierend ist."""
        return self.severity in [SafetyLevel.HIGH_RISK, SafetyLevel.UNSAFE]


@dataclass
class ContentSafetyCheck:
    """Ergebnis einer Content-Safety-Prüfung."""
    content: str
    overall_safety_level: SafetyLevel
    violations: list[SafetyViolation] = field(default_factory=list)
    confidence: float = 0.0
    processing_time_ms: float = 0.0

    @property
    def is_safe(self) -> bool:
        """Prüft, ob Content sicher ist."""
        return self.overall_safety_level in [SafetyLevel.SAFE, SafetyLevel.LOW_RISK]

    @property
    def has_blocking_violations(self) -> bool:
        """Prüft, ob blockierende Verletzungen vorhanden sind."""
        return any(violation.is_blocking for violation in self.violations)


class SafetyDetector(ABC):
    """Basis-Klasse für Safety-Detektoren."""

    @abstractmethod
    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[SafetyViolation]:
        """Detektiert Safety-Verletzungen in Content."""


class ToxicityDetector(SafetyDetector):
    """Detektor für toxische Inhalte."""

    def __init__(self):
        """Initialisiert Toxicity Detector."""
        from .base_detectors import DetectionConfidence, KeywordBasedDetector, RiskLevel

        # Verwende KeywordBasedDetector als Basis
        self._base_detector = KeywordBasedDetector("ToxicityDetector")

        # Konfiguriere toxische Keywords nach Severity
        self._base_detector.add_keywords(
            "high_toxicity",
            ["hate", "kill", "die", "nazi", "terrorist", "bomb", "weapon"],
            RiskLevel.HIGH_RISK,
            DetectionConfidence.HIGH
        )

        self._base_detector.add_keywords(
            "medium_toxicity",
            ["stupid", "idiot", "moron", "racist"],
            RiskLevel.MEDIUM_RISK,
            DetectionConfidence.MEDIUM
        )

    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[SafetyViolation]:
        """Detektiert Toxizität in Content (konsolidiert)."""
        # Verwende Base-Detector für Detection
        base_results = await self._base_detector.detect(content, context)

        # Konvertiere zu SafetyViolation-Format
        violations = []
        for result in base_results:
            violation = SafetyViolation(
                violation_type=ViolationType.TOXICITY,
                severity=self._map_risk_to_safety_level(result.risk_level),
                confidence=result.confidence,
                description=result.description,
                detected_content=result.detected_content,
                suggested_action=result.suggested_action,
                metadata=result.metadata
            )
            violations.append(violation)

        return violations

    def _map_risk_to_safety_level(self, risk_level: str) -> SafetyLevel:
        """Mappt RiskLevel zu SafetyLevel."""
        mapping = {
            "safe": SafetyLevel.SAFE,
            "low_risk": SafetyLevel.LOW_RISK,
            "medium_risk": SafetyLevel.MEDIUM_RISK,
            "high_risk": SafetyLevel.HIGH_RISK,
            "critical": SafetyLevel.UNSAFE,
            "unsafe": SafetyLevel.UNSAFE
        }
        return mapping.get(risk_level, SafetyLevel.MEDIUM_RISK)


class BiasDetector(SafetyDetector):
    """Detektor für Bias und Diskriminierung."""

    def __init__(self):
        """Initialisiert Bias Detector."""
        # Bias-Kategorien und Indikatoren
        self._bias_indicators = {
            "gender": ["men are", "women are", "boys are", "girls are"],
            "race": ["black people", "white people", "asian people"],
            "age": ["old people", "young people", "millennials", "boomers"],
            "religion": ["muslims are", "christians are", "jews are"],
            "nationality": ["americans are", "germans are", "chinese are"]
        }

    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[SafetyViolation]:
        """Detektiert Bias in Content."""
        violations = []

        if not content:
            return violations

        content_lower = content.lower()

        for bias_category, indicators in self._bias_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    # Prüfe auf negative Generalisierungen
                    negative_words = ["stupid", "lazy", "dangerous", "inferior", "bad", "evil"]

                    for neg_word in negative_words:
                        if neg_word in content_lower:
                            violation = SafetyViolation(
                                violation_type=ViolationType.BIAS,
                                severity=SafetyLevel.MEDIUM_RISK,
                                confidence=0.7,
                                description=f"Potentielle {bias_category}-Bias erkannt",
                                detected_content=content[:100] + "..." if len(content) > 100 else content,
                                suggested_action="Content auf Bias überprüfen und neutralisieren",
                                metadata={
                                    "bias_category": bias_category,
                                    "indicator": indicator,
                                    "negative_word": neg_word
                                }
                            )
                            violations.append(violation)
                            break

        return violations


class HarmfulContentFilter(SafetyDetector):
    """Filter für schädliche Inhalte."""

    def __init__(self):
        """Initialisiert Harmful Content Filter."""
        # Kategorien schädlicher Inhalte
        self._harmful_categories = {
            ViolationType.VIOLENCE: [
                "kill", "murder", "assault", "attack", "fight", "violence", "weapon"
            ],
            ViolationType.SELF_HARM: [
                "suicide", "self-harm", "cut myself", "end my life", "kill myself"
            ],
            ViolationType.ILLEGAL_ACTIVITY: [
                "drugs", "cocaine", "heroin", "steal", "robbery", "fraud", "hack"
            ],
            ViolationType.SEXUAL_CONTENT: [
                "porn", "sex", "nude", "naked", "explicit"
            ]
        }

    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[SafetyViolation]:
        """Detektiert schädliche Inhalte."""
        violations = []

        if not content:
            return violations

        content_lower = content.lower()

        for violation_type, keywords in self._harmful_categories.items():
            detected_keywords = []

            for keyword in keywords:
                if keyword in content_lower:
                    detected_keywords.append(keyword)

            if detected_keywords:
                # Bestimme Severity basierend auf Anzahl und Typ
                severity = SafetyLevel.MEDIUM_RISK
                if violation_type in [ViolationType.VIOLENCE, ViolationType.SELF_HARM]:
                    severity = SafetyLevel.HIGH_RISK

                confidence = min(len(detected_keywords) * 0.3, 1.0)

                violation = SafetyViolation(
                    violation_type=violation_type,
                    severity=severity,
                    confidence=confidence,
                    description=f"Schädlicher Content erkannt: {violation_type.value}",
                    detected_content=content[:100] + "..." if len(content) > 100 else content,
                    suggested_action="Content blockieren oder moderieren",
                    metadata={
                        "detected_keywords": detected_keywords,
                        "category": violation_type.value
                    }
                )
                violations.append(violation)

        return violations


class SafetyGuardrailsEngine:
    """Engine für Safety Guardrails."""

    def __init__(self):
        """Initialisiert Safety Guardrails Engine."""
        self._detectors: list[SafetyDetector] = []
        self._safety_thresholds = {
            SafetyLevel.SAFE: 0.0,
            SafetyLevel.LOW_RISK: 0.3,
            SafetyLevel.MEDIUM_RISK: 0.6,
            SafetyLevel.HIGH_RISK: 0.8,
            SafetyLevel.UNSAFE: 0.9
        }

        # Statistiken
        self._checks_performed = 0
        self._violations_detected = 0
        self._content_blocked = 0

        # Standard-Detektoren registrieren
        self.register_detector(ToxicityDetector())
        self.register_detector(BiasDetector())
        self.register_detector(HarmfulContentFilter())

    def register_detector(self, detector: SafetyDetector) -> None:
        """Registriert Safety-Detektor."""
        self._detectors.append(detector)
        logger.info(f"Safety-Detektor registriert: {detector.__class__.__name__}")

    @trace_function("safety.check_content")
    async def check_content(
        self,
        content: str,
        context: dict[str, Any] | None = None
    ) -> ContentSafetyCheck:
        """Führt umfassende Content-Safety-Prüfung durch."""
        start_time = time.time()
        self._checks_performed += 1

        try:
            all_violations = []

            # Führe alle Detektoren parallel aus
            detection_tasks = [
                detector.detect(content, context) for detector in self._detectors
            ]

            detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)

            # Sammle alle Verletzungen
            for result in detection_results:
                if isinstance(result, Exception):
                    logger.error(f"Safety-Detection fehlgeschlagen: {result}")
                    continue

                if isinstance(result, list):
                    all_violations.extend(result)

            # Bestimme Overall Safety Level
            overall_safety_level = self._determine_overall_safety_level(all_violations)

            # Berechne Gesamt-Confidence
            overall_confidence = 0.0
            if all_violations:
                overall_confidence = sum(v.confidence for v in all_violations) / len(all_violations)

            processing_time = (time.time() - start_time) * 1000

            # Aktualisiere Statistiken
            if all_violations:
                self._violations_detected += len(all_violations)

            if overall_safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.UNSAFE]:
                self._content_blocked += 1

            return ContentSafetyCheck(
                content=content,
                overall_safety_level=overall_safety_level,
                violations=all_violations,
                confidence=overall_confidence,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.exception(f"Content-Safety-Check fehlgeschlagen: {e}")
            processing_time = (time.time() - start_time) * 1000

            return ContentSafetyCheck(
                content=content,
                overall_safety_level=SafetyLevel.MEDIUM_RISK,
                violations=[SafetyViolation(
                    violation_type=ViolationType.MISINFORMATION,
                    severity=SafetyLevel.MEDIUM_RISK,
                    confidence=0.5,
                    description=f"Safety-Check fehlgeschlagen: {e!s}",
                    detected_content=content[:100],
                    suggested_action="Manuell überprüfen"
                )],
                processing_time_ms=processing_time
            )

    def _determine_overall_safety_level(self, violations: list[SafetyViolation]) -> SafetyLevel:
        """Bestimmt Overall Safety Level basierend auf Verletzungen."""
        if not violations:
            return SafetyLevel.SAFE

        # Höchste Severity bestimmt Overall Level
        max_severity = SafetyLevel.SAFE
        severity_order = [
            SafetyLevel.SAFE,
            SafetyLevel.LOW_RISK,
            SafetyLevel.MEDIUM_RISK,
            SafetyLevel.HIGH_RISK,
            SafetyLevel.UNSAFE
        ]

        for violation in violations:
            violation_index = severity_order.index(violation.severity)
            max_index = severity_order.index(max_severity)

            if violation_index > max_index:
                max_severity = violation.severity

        return max_severity

    async def batch_check_content(
        self,
        content_list: list[str],
        context: dict[str, Any] | None = None
    ) -> list[ContentSafetyCheck]:
        """Führt Batch-Safety-Checks durch."""
        tasks = [
            self.check_content(content, context) for content in content_list
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        safety_checks = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch-Safety-Check fehlgeschlagen: {result}")
                # Fallback-Result
                safety_checks.append(ContentSafetyCheck(
                    content="",
                    overall_safety_level=SafetyLevel.MEDIUM_RISK
                ))
            else:
                safety_checks.append(result)

        return safety_checks

    def get_safety_statistics(self) -> dict[str, Any]:
        """Gibt Safety-Statistiken zurück."""
        return {
            "checks_performed": self._checks_performed,
            "violations_detected": self._violations_detected,
            "content_blocked": self._content_blocked,
            "block_rate": self._content_blocked / max(self._checks_performed, 1),
            "violation_rate": self._violations_detected / max(self._checks_performed, 1),
            "registered_detectors": len(self._detectors),
            "detector_types": [detector.__class__.__name__ for detector in self._detectors]
        }

    def configure_thresholds(self, thresholds: dict[SafetyLevel, float]) -> None:
        """Konfiguriert Safety-Schwellwerte."""
        self._safety_thresholds.update(thresholds)
        logger.info(f"Safety-Schwellwerte aktualisiert: {thresholds}")


# Globale Safety Guardrails Engine Instanz
safety_guardrails_engine = SafetyGuardrailsEngine()
