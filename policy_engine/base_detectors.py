# backend/policy_engine/base_detectors.py
"""Gemeinsame Base-Klassen für Detektoren.

Konsolidiert die gemeinsamen Patterns aus ToxicityDetector, BiasDetector,
PromptInjectionDetector und anderen Detektoren in wiederverwendbare Base-Klassen.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any

from kei_logging import get_logger

# Import konsolidierter Konstanten
from .constants import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_VERY_HIGH,
    RISK_ACTION_MAPPING,
)

logger = get_logger(__name__)


class DetectionConfidence(float, Enum):
    """Standard-Confidence-Level für Detektionen."""
    LOW = CONFIDENCE_LOW
    MEDIUM = CONFIDENCE_MEDIUM
    HIGH = CONFIDENCE_HIGH
    VERY_HIGH = CONFIDENCE_VERY_HIGH
    CERTAIN = CONFIDENCE_CERTAIN


class RiskLevel(str, Enum):
    """Konsolidierte Risk-Level."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    UNSAFE = "unsafe"


@dataclass
class DetectionResult:
    """Basis-Klasse für Detection-Ergebnisse."""
    detection_type: str
    risk_level: RiskLevel
    confidence: float
    description: str
    detected_content: str
    suggested_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        """Prüft, ob Detection blockierend ist."""
        return self.risk_level in [RiskLevel.HIGH_RISK, RiskLevel.CRITICAL, RiskLevel.UNSAFE]


@dataclass
class PatternMatch:
    """Repräsentiert einen Pattern-Match."""
    pattern: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Basis-Klasse für alle Detektoren."""

    def __init__(self, name: str | None = None) -> None:
        """Initialisiert Base-Detector."""
        self.name = name or self.__class__.__name__
        self._detections_performed = 0
        self._matches_found = 0
        self._patterns_compiled = False

    @abstractmethod
    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[DetectionResult]:
        """Detektiert Probleme in Content."""

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Detector-Statistiken zurück."""
        return {
            "detector_name": self.name,
            "detections_performed": self._detections_performed,
            "matches_found": self._matches_found,
            "match_rate": self._matches_found / max(self._detections_performed, 1)
        }


class PatternBasedDetector(BaseDetector):
    """Basis-Klasse für Pattern-basierte Detektoren."""

    def __init__(self, name: str | None = None) -> None:
        """Initialisiert Pattern-based Detector."""
        super().__init__(name)
        self._patterns: dict[str, list[str]] = {}
        self._compiled_patterns: dict[str, list[Pattern[str]]] = {}
        self._risk_mappings: dict[str, RiskLevel] = {}
        self._confidence_mappings: dict[str, float] = {}

    def add_patterns(
        self,
        category: str,
        patterns: list[str],
        risk_level: RiskLevel = RiskLevel.MEDIUM_RISK,
        confidence: float = DetectionConfidence.MEDIUM
    ) -> None:
        """Fügt Patterns für eine Kategorie hinzu."""
        self._patterns[category] = patterns
        self._risk_mappings[category] = risk_level
        self._confidence_mappings[category] = confidence
        self._patterns_compiled = False
        logger.debug(f"Patterns hinzugefügt für Kategorie: {category}")

    def _compile_patterns(self) -> None:
        """Kompiliert alle Patterns zu Regex-Objekten."""
        if self._patterns_compiled:
            return

        self._compiled_patterns = {}
        for category, patterns in self._patterns.items():
            self._compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        self._patterns_compiled = True
        logger.debug(f"Patterns kompiliert für {len(self._patterns)} Kategorien")

    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[DetectionResult]:
        """Detektiert Patterns in Content."""
        self._detections_performed += 1

        if not content:
            return []

        self._compile_patterns()
        results = []

        for category, compiled_patterns in self._compiled_patterns.items():
            matches = self._find_pattern_matches(content, compiled_patterns, category)

            for match in matches:
                result = self._create_detection_result(match, category, content)
                results.append(result)
                self._matches_found += 1

        return results

    def _find_pattern_matches(
        self,
        content: str,
        patterns: list[Pattern[str]],
        category: str
    ) -> list[PatternMatch]:
        """Findet alle Pattern-Matches in Content."""
        matches = []

        for pattern in patterns:
            for regex_match in pattern.finditer(content):
                match = PatternMatch(
                    pattern=pattern.pattern,
                    matched_text=regex_match.group(),
                    start_pos=regex_match.start(),
                    end_pos=regex_match.end(),
                    confidence=self._confidence_mappings.get(category, DetectionConfidence.MEDIUM),
                    metadata={
                        "category": category,
                        "full_match": regex_match.group(),
                        "groups": regex_match.groups()
                    }
                )
                matches.append(match)

        return matches

    def _create_detection_result(
        self,
        match: PatternMatch,
        category: str,
        original_content: str
    ) -> DetectionResult:
        """Erstellt DetectionResult aus PatternMatch."""
        risk_level = self._risk_mappings.get(category, RiskLevel.MEDIUM_RISK)

        return DetectionResult(
            detection_type=f"{self.name}_{category}",
            risk_level=risk_level,
            confidence=match.confidence,
            description=f"{category.replace('_', ' ').title()} erkannt",
            detected_content=match.matched_text,
            suggested_action=self._determine_action(risk_level),
            metadata={
                **match.metadata,
                "detector": self.name,
                "pattern": match.pattern,
                "position": {"start": match.start_pos, "end": match.end_pos},
                "content_length": len(original_content),
                "match_context": original_content[max(0, match.start_pos-20):match.end_pos+20]
            }
        )

    def _determine_action(self, risk_level: RiskLevel) -> str:
        """Bestimmt empfohlene Aktion basierend auf Risk-Level."""
        return RISK_ACTION_MAPPING.get(risk_level.value, "warn")


class KeywordBasedDetector(BaseDetector):
    """Basis-Klasse für Keyword-basierte Detektoren."""

    def __init__(self, name: str | None = None):
        """Initialisiert Keyword-based Detector."""
        super().__init__(name)
        self._keywords: dict[str, list[str]] = {}
        self._risk_mappings: dict[str, RiskLevel] = {}
        self._confidence_mappings: dict[str, float] = {}

    def add_keywords(
        self,
        category: str,
        keywords: list[str],
        risk_level: RiskLevel = RiskLevel.MEDIUM_RISK,
        confidence: float = DetectionConfidence.MEDIUM
    ) -> None:
        """Fügt Keywords für eine Kategorie hinzu."""
        self._keywords[category] = [kw.lower() for kw in keywords]
        self._risk_mappings[category] = risk_level
        self._confidence_mappings[category] = confidence
        logger.debug(f"Keywords hinzugefügt für Kategorie: {category}")

    async def detect(self, content: str, context: dict[str, Any] | None = None) -> list[DetectionResult]:
        """Detektiert Keywords in Content."""
        self._detections_performed += 1

        if not content:
            return []

        content_lower = content.lower()
        results = []

        for category, keywords in self._keywords.items():
            detected_keywords = []

            for keyword in keywords:
                if keyword in content_lower:
                    detected_keywords.append(keyword)

            if detected_keywords:
                result = self._create_keyword_result(category, detected_keywords, content)
                results.append(result)
                self._matches_found += 1

        return results

    def _create_keyword_result(
        self,
        category: str,
        detected_keywords: list[str],
        original_content: str
    ) -> DetectionResult:
        """Erstellt DetectionResult aus Keyword-Matches."""
        risk_level = self._risk_mappings.get(category, RiskLevel.MEDIUM_RISK)
        confidence = self._confidence_mappings.get(category, DetectionConfidence.MEDIUM)

        return DetectionResult(
            detection_type=f"{self.name}_{category}",
            risk_level=risk_level,
            confidence=confidence,
            description=f"{category.replace('_', ' ').title()} Keywords erkannt",
            detected_content=", ".join(detected_keywords),
            suggested_action=self._determine_action(risk_level),
            metadata={
                "category": category,
                "detected_keywords": detected_keywords,
                "keyword_count": len(detected_keywords),
                "detector": self.name,
                "content_length": len(original_content),
                "content_preview": original_content[:100] + "..." if len(original_content) > 100 else original_content
            }
        )

    def _determine_action(self, risk_level: RiskLevel) -> str:
        """Bestimmt empfohlene Aktion basierend auf Risk-Level."""
        return RISK_ACTION_MAPPING.get(risk_level.value, "warn")


class DetectorEngine:
    """Engine für Detector-Management."""

    def __init__(self, name: str = "DetectorEngine"):
        """Initialisiert Detector-Engine."""
        self.name = name
        self._detectors: list[BaseDetector] = []
        self._checks_performed = 0
        self._total_detections = 0

    def register_detector(self, detector: BaseDetector) -> None:
        """Registriert einen Detector."""
        self._detectors.append(detector)
        logger.info(f"Detector registriert: {detector.name}")

    async def detect_all(
        self,
        content: str,
        context: dict[str, Any] | None = None
    ) -> list[DetectionResult]:
        """Führt Detection mit allen registrierten Detektoren durch."""
        self._checks_performed += 1
        all_results = []

        for detector in self._detectors:
            try:
                results = await detector.detect(content, context)
                all_results.extend(results)
                self._total_detections += len(results)
            except Exception as e:
                logger.exception(f"Detection fehlgeschlagen für {detector.name}: {e}")

        return all_results

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Engine-Statistiken zurück."""
        detector_stats = [detector.get_statistics() for detector in self._detectors]

        return {
            "engine_name": self.name,
            "registered_detectors": len(self._detectors),
            "checks_performed": self._checks_performed,
            "total_detections": self._total_detections,
            "detection_rate": self._total_detections / max(self._checks_performed, 1),
            "detector_statistics": detector_stats
        }
