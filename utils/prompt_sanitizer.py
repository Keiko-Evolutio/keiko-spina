# backend/utils/prompt_sanitizer.py
"""Konsolidierte Prompt-Sanitization-Utility.

Vereint die Sanitization-Logik aus verschiedenen Modulen in eine
zentrale, konfigurierbare und testbare Implementierung.
Konsolidiert die Implementierungen aus utils und policy_engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from agents.constants import (
    CONTENT_SAFETY_BLACKLIST,
    MAX_PROMPT_LENGTH,
    MIN_PROMPT_LENGTH,
    QUALITY_ENHANCEMENT_TAGS,
    REDACTED_MARKER,
    SAFETY_SCORE_THRESHOLD,
    STYLE_OPTIMIZATION_MAP,
    ImageStyle,
)
from kei_logging import get_logger

from .sanitization_constants import (
    CONFIDENCE_SCORE_FACTORS,
    DANGEROUS_PATTERNS,
    REPLACEMENT_PATTERNS,
    SANITIZATION_MARKERS,
    THREAT_MARKER_MAP,
)

logger = get_logger(__name__)


class SanitizationStrategy(str, Enum):
    """Strategien für Prompt-Sanitization."""

    BLACKLIST_ONLY = "blacklist_only"
    PATTERN_BASED = "pattern_based"
    COMPREHENSIVE = "comprehensive"


class ThreatType(str, Enum):
    """Typen von Prompt-Bedrohungen."""

    DIRECT_INJECTION = "direct_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    INSTRUCTION_OVERRIDE = "instruction_override"
    EXPLICIT_CONTENT = "explicit_content"
    COMMAND_INJECTION = "command_injection"


@dataclass(slots=True)
class PromptThreat:
    """Erkannte Bedrohung in einem Prompt."""

    threat_type: ThreatType
    detected_pattern: str
    confidence: float
    suggested_action: str = "sanitize"


@dataclass(slots=True)
class SanitizationResult:
    """Ergebnis der Prompt-Sanitization."""

    is_safe: bool
    sanitized_prompt: str
    original_prompt: str
    violations_detected: list[str]
    confidence_score: float
    strategy_used: SanitizationStrategy
    threats_detected: list[PromptThreat] = None


class PromptSanitizer:
    """Zentrale Prompt-Sanitization-Utility.

    Konsolidiert verschiedene Sanitization-Ansätze und bietet
    konfigurierbare Strategien für unterschiedliche Anwendungsfälle.
    """

    def __init__(
        self,
        *,
        strategy: SanitizationStrategy = SanitizationStrategy.COMPREHENSIVE,
        custom_blacklist: set[str] | None = None,
        enable_pattern_detection: bool = True,
        enable_length_validation: bool = True,
    ) -> None:
        """Initialisiert den Prompt Sanitizer.

        Args:
            strategy: Sanitization-Strategie
            custom_blacklist: Zusätzliche Blacklist-Begriffe
            enable_pattern_detection: Pattern-basierte Erkennung aktivieren
            enable_length_validation: Längen-Validierung aktivieren
        """
        self.strategy = strategy
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_length_validation = enable_length_validation

        # Blacklist zusammenstellen
        self.blacklist = CONTENT_SAFETY_BLACKLIST.copy()
        if custom_blacklist:
            self.blacklist.update(custom_blacklist)

        logger.debug({
            "event": "prompt_sanitizer_init",
            "strategy": strategy.value,
            "blacklist_size": len(self.blacklist),
            "pattern_detection": enable_pattern_detection,
            "length_validation": enable_length_validation,
        })

    def sanitize(self, prompt: str) -> SanitizationResult:
        """Sanitisiert einen Prompt basierend auf der konfigurierten Strategie.

        Args:
            prompt: Zu sanitisierender Prompt

        Returns:
            Sanitization-Ergebnis mit Details
        """
        if not prompt:
            return self._create_empty_prompt_result(prompt)

        violations = []
        threats = []
        confidence_score = 1.0
        sanitized = prompt.strip()

        # Längen-Validierung
        if self.enable_length_validation:
            length_violations, sanitized = self._validate_length(sanitized)
            violations.extend(length_violations)
            confidence_score = self._adjust_confidence_score(
                confidence_score, length_violations, "LENGTH_VIOLATION"
            )

        # Strategie-spezifische Sanitization
        sanitized, strategy_violations, strategy_threats = self._apply_strategy_sanitization(
            sanitized, violations
        )
        violations.extend(strategy_violations)
        threats.extend(strategy_threats)

        # Confidence Score für Strategy-Violations anpassen
        blacklist_count = sum(1 for v in strategy_violations if "blacklist_keyword" in v)
        pattern_count = sum(1 for v in strategy_violations if "dangerous_pattern" in v)

        confidence_score -= blacklist_count * CONFIDENCE_SCORE_FACTORS["BLACKLIST_VIOLATION"]
        confidence_score -= pattern_count * CONFIDENCE_SCORE_FACTORS["PATTERN_VIOLATION"]

        # Finale Sanitization
        sanitized = self._apply_replacement_patterns(sanitized)

        # Safety-Assessment
        is_safe = self._assess_safety(violations, confidence_score)

        return SanitizationResult(
            is_safe=is_safe,
            sanitized_prompt=sanitized,
            original_prompt=prompt,
            violations_detected=violations,
            confidence_score=max(0.0, confidence_score),
            strategy_used=self.strategy,
            threats_detected=threats,
        )

    def optimize_for_style(self, prompt: str, style: ImageStyle) -> str:
        """Optimiert einen Prompt für einen spezifischen Bildstil.

        Args:
            prompt: Basis-Prompt
            style: Ziel-Bildstil

        Returns:
            Optimierter Prompt
        """
        if not prompt.strip():
            return prompt

        base = prompt.strip()
        style_tags = STYLE_OPTIMIZATION_MAP.get(style, "")

        # Optimierten Prompt zusammenstellen
        optimized = f"{base}. {style_tags}. {QUALITY_ENHANCEMENT_TAGS}."

        # Länge prüfen und ggf. kürzen
        if len(optimized) > MAX_PROMPT_LENGTH:
            # Basis-Prompt kürzen, aber Style-Tags beibehalten
            max_base_length = MAX_PROMPT_LENGTH - len(style_tags) - len(QUALITY_ENHANCEMENT_TAGS) - 4
            if max_base_length > MIN_PROMPT_LENGTH:
                base = base[:max_base_length].rstrip()
                optimized = f"{base}. {style_tags}. {QUALITY_ENHANCEMENT_TAGS}."

        return optimized

    def sanitize_with_threats(self, prompt: str, threats: list[PromptThreat]) -> str:
        """Sanitisiert Prompt basierend auf erkannten Bedrohungen.

        Kompatibilitäts-Methode für policy_engine Integration.

        Args:
            prompt: Zu sanitisierender Prompt
            threats: Liste erkannter Bedrohungen

        Returns:
            Sanitisierter Prompt
        """
        sanitized = prompt

        for threat in threats:
            if threat.suggested_action == "sanitize":
                sanitized = self._sanitize_threat(sanitized, threat)

        return sanitized

    def _create_empty_prompt_result(self, prompt: str) -> SanitizationResult:
        """Erstellt Ergebnis für leeren Prompt."""
        return SanitizationResult(
            is_safe=False,
            sanitized_prompt="",
            original_prompt=prompt,
            violations_detected=["empty_prompt"],
            confidence_score=0.0,
            strategy_used=self.strategy,
            threats_detected=[],
        )

    def _adjust_confidence_score(
        self,
        current_score: float,
        violations: list[str],
        violation_type: str
    ) -> float:
        """Passt Confidence Score basierend auf Verletzungen an."""
        if violations:
            factor = CONFIDENCE_SCORE_FACTORS.get(violation_type, 0.1)
            return current_score - factor
        return current_score

    def _apply_strategy_sanitization(
        self,
        sanitized: str,
        existing_violations: list[str]
    ) -> tuple[str, list[str], list[PromptThreat]]:
        """Wendet strategie-spezifische Sanitization an."""
        violations = []
        threats = []

        if self.strategy in (SanitizationStrategy.BLACKLIST_ONLY, SanitizationStrategy.COMPREHENSIVE):
            blacklist_violations, sanitized = self._apply_blacklist_filter(sanitized)
            violations.extend(blacklist_violations)

        if self.strategy in (SanitizationStrategy.PATTERN_BASED, SanitizationStrategy.COMPREHENSIVE):
            if self.enable_pattern_detection:
                pattern_violations, sanitized, pattern_threats = self._apply_pattern_filter_with_threats(sanitized)
                violations.extend(pattern_violations)
                threats.extend(pattern_threats)

        return sanitized, violations, threats



    def _assess_safety(self, violations: list[str], confidence_score: float) -> bool:
        """Bewertet die Sicherheit des Prompts."""
        return len(violations) == 0 and confidence_score >= SAFETY_SCORE_THRESHOLD

    def _sanitize_threat(self, prompt: str, threat: PromptThreat) -> str:
        """Sanitisiert spezifische Bedrohung."""
        pattern = threat.detected_pattern
        marker = THREAT_MARKER_MAP.get(threat.threat_type.value, SANITIZATION_MARKERS["GENERIC"])
        return prompt.replace(pattern, marker)

    def _validate_length(self, prompt: str) -> tuple[list[str], str]:
        """Validiert und korrigiert Prompt-Länge."""
        violations = []
        sanitized = prompt

        if len(prompt) < MIN_PROMPT_LENGTH:
            violations.append("prompt_too_short")
        elif len(prompt) > MAX_PROMPT_LENGTH:
            violations.append("prompt_too_long")
            sanitized = prompt[:MAX_PROMPT_LENGTH].rstrip()

        return violations, sanitized

    def _apply_blacklist_filter(self, prompt: str) -> tuple[list[str], str]:
        """Wendet Blacklist-Filter an."""
        violations = []
        sanitized = prompt
        prompt_lower = prompt.lower()

        for word in self.blacklist:
            word_lower = word.lower()
            if word_lower in prompt_lower:
                violations.append(f"blacklist_keyword: {word}")
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                sanitized = pattern.sub(REDACTED_MARKER, sanitized)

        return violations, sanitized

    def _apply_pattern_filter(self, prompt: str) -> tuple[list[str], str]:
        """Wendet Pattern-basierte Filter an."""
        violations = []
        sanitized = prompt

        for pattern_name, pattern in DANGEROUS_PATTERNS.items():
            if pattern.search(prompt.lower()):
                violations.append(f"dangerous_pattern: {pattern_name}")
                # Pattern durch Redacted-Marker ersetzen
                sanitized = pattern.sub(REDACTED_MARKER, sanitized)

        return violations, sanitized

    def _apply_pattern_filter_with_threats(self, prompt: str) -> tuple[list[str], str, list[PromptThreat]]:
        """Wendet Pattern-basierte Filter an und erkennt Threats."""
        violations = []
        threats = []
        sanitized = prompt

        for pattern_name, pattern in DANGEROUS_PATTERNS.items():
            match = pattern.search(prompt.lower())
            if match:
                violations.append(f"dangerous_pattern: {pattern_name}")

                # Threat-Objekt erstellen
                threat_type = self._map_pattern_to_threat_type(pattern_name)
                threat = PromptThreat(
                    threat_type=threat_type,
                    detected_pattern=match.group(),
                    confidence=0.8,
                    suggested_action="sanitize"
                )
                threats.append(threat)

                # Pattern durch entsprechenden Marker ersetzen
                marker = THREAT_MARKER_MAP.get(threat_type.value, REDACTED_MARKER)
                sanitized = pattern.sub(marker, sanitized)

        return violations, sanitized, threats

    def _map_pattern_to_threat_type(self, pattern_name: str) -> ThreatType:
        """Mappt Pattern-Namen zu Threat-Typen."""
        pattern_mapping = {
            "direct_injection": ThreatType.DIRECT_INJECTION,
            "jailbreak_attempt": ThreatType.JAILBREAK_ATTEMPT,
            "system_prompt_leak": ThreatType.SYSTEM_PROMPT_LEAK,
            "instruction_override": ThreatType.INSTRUCTION_OVERRIDE,
            "explicit_generation_en": ThreatType.EXPLICIT_CONTENT,
            "explicit_generation_de": ThreatType.EXPLICIT_CONTENT,
            "command_injection": ThreatType.COMMAND_INJECTION,
            "dangerous_commands": ThreatType.COMMAND_INJECTION,
        }
        return pattern_mapping.get(pattern_name, ThreatType.DIRECT_INJECTION)

    def _apply_replacement_patterns(self, prompt: str) -> str:
        """Wendet Replacement-Patterns für finale Sanitization an."""
        sanitized = prompt

        for pattern, replacement in REPLACEMENT_PATTERNS.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized


__all__ = [
    "PromptSanitizer",
    "PromptThreat",
    "SanitizationResult",
    "SanitizationStrategy",
    "ThreatType",
]
