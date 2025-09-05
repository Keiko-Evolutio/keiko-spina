# backend/policy_engine/prompt_guardrails.py
"""Prompt Guardrails Engine für Keiko Personal Assistant

Implementiert Input-Validation, Prompt-Injection-Detection,
Content-Filtering und Context-aware Prompt-Sanitization.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class PromptRiskLevel(str, Enum):
    """Risk-Level für Prompt-Bewertung."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class InjectionType(str, Enum):
    """Typen von Prompt-Injections."""
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    ROLE_PLAYING = "role_playing"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    INSTRUCTION_OVERRIDE = "instruction_override"
    CONTEXT_MANIPULATION = "context_manipulation"


class SanitizationAction(str, Enum):
    """Aktionen für Prompt-Sanitization."""
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    WARN = "warn"
    QUARANTINE = "quarantine"


@dataclass
class PromptThreat:
    """Repräsentiert eine erkannte Prompt-Bedrohung."""
    threat_type: InjectionType
    risk_level: PromptRiskLevel
    confidence: float  # 0.0 - 1.0
    description: str
    detected_pattern: str
    suggested_action: SanitizationAction
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptValidationResult:
    """Ergebnis einer Prompt-Validierung."""
    original_prompt: str
    is_safe: bool
    risk_level: PromptRiskLevel
    threats: list[PromptThreat] = field(default_factory=list)
    sanitized_prompt: str | None = None
    processing_time_ms: float = 0.0

    @property
    def requires_blocking(self) -> bool:
        """Prüft, ob Prompt blockiert werden sollte."""
        return self.risk_level in [PromptRiskLevel.HIGH_RISK, PromptRiskLevel.CRITICAL]

    @property
    def requires_sanitization(self) -> bool:
        """Prüft, ob Prompt sanitisiert werden sollte."""
        return any(threat.suggested_action == SanitizationAction.SANITIZE for threat in self.threats)


class PromptValidator(ABC):
    """Basis-Klasse für Prompt-Validatoren."""

    @abstractmethod
    def validate(self, prompt: str, context: dict[str, Any] | None = None) -> list[PromptThreat]:
        """Validiert Prompt und gibt erkannte Bedrohungen zurück."""


class PromptInjectionDetector(PromptValidator):
    """Detektor für Prompt-Injections."""

    def __init__(self):
        """Initialisiert Prompt Injection Detector."""
        # Injection-Patterns
        self._injection_patterns = {
            InjectionType.DIRECT_INJECTION: [
                r"ignore\s+(?:previous|all)\s+instructions?",
                r"forget\s+(?:everything|all)\s+(?:above|before)",
                r"new\s+instructions?:\s*",
                r"system\s*:\s*",
                r"override\s+(?:previous|system)\s+",
                r"disregard\s+(?:previous|all)\s+"
            ],
            InjectionType.JAILBREAK_ATTEMPT: [
                r"pretend\s+(?:you\s+are|to\s+be)",
                r"act\s+as\s+(?:if|a)",
                r"roleplay\s+as",
                r"imagine\s+(?:you\s+are|being)",
                r"let's\s+pretend",
                r"in\s+this\s+scenario"
            ],
            InjectionType.SYSTEM_PROMPT_LEAK: [
                r"show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
                r"what\s+(?:are\s+)?(?:your\s+)?(?:initial\s+)?instructions",
                r"reveal\s+(?:your\s+)?(?:system\s+)?prompt",
                r"print\s+(?:your\s+)?(?:system\s+)?prompt"
            ],
            InjectionType.INSTRUCTION_OVERRIDE: [
                r"instead\s+of\s+(?:following|doing)",
                r"don't\s+(?:follow|do)\s+(?:the\s+)?(?:above|previous)",
                r"replace\s+(?:the\s+)?(?:above|previous)\s+with",
                r"substitute\s+(?:the\s+)?instructions"
            ]
        }

        # Kompiliere Patterns
        self._compiled_patterns = {}
        for injection_type, patterns in self._injection_patterns.items():
            self._compiled_patterns[injection_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def validate(self, prompt: str, context: dict[str, Any] | None = None) -> list[PromptThreat]:
        """Detektiert Prompt-Injections."""
        threats = []

        for injection_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(prompt)

                for match in matches:
                    # Bestimme Risk-Level basierend auf Injection-Typ
                    risk_level = self._determine_risk_level(injection_type)

                    threat = PromptThreat(
                        threat_type=injection_type,
                        risk_level=risk_level,
                        confidence=0.8,  # Hohe Confidence für Pattern-Matches
                        description=f"{injection_type.value} erkannt",
                        detected_pattern=match.group(),
                        suggested_action=self._determine_action(risk_level),
                        metadata={
                            "match_start": match.start(),
                            "match_end": match.end(),
                            "pattern": pattern.pattern
                        }
                    )
                    threats.append(threat)

        return threats

    def _determine_risk_level(self, injection_type: InjectionType) -> PromptRiskLevel:
        """Bestimmt Risk-Level basierend auf Injection-Typ."""
        risk_mapping = {
            InjectionType.DIRECT_INJECTION: PromptRiskLevel.HIGH_RISK,
            InjectionType.JAILBREAK_ATTEMPT: PromptRiskLevel.MEDIUM_RISK,
            InjectionType.SYSTEM_PROMPT_LEAK: PromptRiskLevel.HIGH_RISK,
            InjectionType.INSTRUCTION_OVERRIDE: PromptRiskLevel.HIGH_RISK,
            InjectionType.INDIRECT_INJECTION: PromptRiskLevel.MEDIUM_RISK,
            InjectionType.ROLE_PLAYING: PromptRiskLevel.LOW_RISK,
            InjectionType.CONTEXT_MANIPULATION: PromptRiskLevel.MEDIUM_RISK
        }
        return risk_mapping.get(injection_type, PromptRiskLevel.MEDIUM_RISK)

    def _determine_action(self, risk_level: PromptRiskLevel) -> SanitizationAction:
        """Bestimmt empfohlene Aktion basierend auf Risk-Level."""
        action_mapping = {
            PromptRiskLevel.SAFE: SanitizationAction.ALLOW,
            PromptRiskLevel.LOW_RISK: SanitizationAction.WARN,
            PromptRiskLevel.MEDIUM_RISK: SanitizationAction.SANITIZE,
            PromptRiskLevel.HIGH_RISK: SanitizationAction.BLOCK,
            PromptRiskLevel.CRITICAL: SanitizationAction.QUARANTINE
        }
        return action_mapping.get(risk_level, SanitizationAction.SANITIZE)


class ContentFilter(PromptValidator):
    """Filter für schädliche Inhalte in Prompts."""

    def __init__(self):
        """Initialisiert Content Filter."""
        # Schädliche Content-Kategorien
        self._harmful_content = {
            "violence": [
                "kill", "murder", "assault", "attack", "violence", "weapon",
                "bomb", "explosive", "gun", "knife", "torture"
            ],
            "hate_speech": [
                "hate", "racist", "nazi", "supremacist", "bigot",
                "slur", "discrimination"
            ],
            "illegal": [
                "drugs", "cocaine", "heroin", "meth", "steal", "robbery",
                "fraud", "hack", "piracy", "counterfeit"
            ],
            "sexual": [
                "porn", "explicit", "sexual", "nude", "nsfw"
            ],
            "self_harm": [
                "suicide", "self-harm", "cut myself", "end my life"
            ]
        }

    def validate(self, prompt: str, context: dict[str, Any] | None = None) -> list[PromptThreat]:
        """Filtert schädliche Inhalte."""
        threats = []
        prompt_lower = prompt.lower()

        for category, keywords in self._harmful_content.items():
            detected_keywords = []

            for keyword in keywords:
                if keyword in prompt_lower:
                    detected_keywords.append(keyword)

            if detected_keywords:
                # Bestimme Risk-Level basierend auf Kategorie
                risk_level = self._get_category_risk_level(category)

                threat = PromptThreat(
                    threat_type=InjectionType.CONTEXT_MANIPULATION,  # Generischer Typ
                    risk_level=risk_level,
                    confidence=0.7,
                    description=f"Schädlicher Content erkannt: {category}",
                    detected_pattern=", ".join(detected_keywords),
                    suggested_action=self._determine_action(risk_level),
                    metadata={
                        "category": category,
                        "detected_keywords": detected_keywords
                    }
                )
                threats.append(threat)

        return threats

    def _get_category_risk_level(self, category: str) -> PromptRiskLevel:
        """Gibt Risk-Level für Content-Kategorie zurück."""
        risk_mapping = {
            "violence": PromptRiskLevel.HIGH_RISK,
            "hate_speech": PromptRiskLevel.HIGH_RISK,
            "illegal": PromptRiskLevel.HIGH_RISK,
            "sexual": PromptRiskLevel.MEDIUM_RISK,
            "self_harm": PromptRiskLevel.CRITICAL
        }
        return risk_mapping.get(category, PromptRiskLevel.MEDIUM_RISK)

    def _determine_action(self, risk_level: PromptRiskLevel) -> SanitizationAction:
        """Bestimmt empfohlene Aktion."""
        action_mapping = {
            PromptRiskLevel.SAFE: SanitizationAction.ALLOW,
            PromptRiskLevel.LOW_RISK: SanitizationAction.WARN,
            PromptRiskLevel.MEDIUM_RISK: SanitizationAction.SANITIZE,
            PromptRiskLevel.HIGH_RISK: SanitizationAction.BLOCK,
            PromptRiskLevel.CRITICAL: SanitizationAction.QUARANTINE
        }
        return action_mapping.get(risk_level, SanitizationAction.SANITIZE)


class InputValidator(PromptValidator):
    """Validator für Input-Format und -Struktur."""

    def __init__(self):
        """Initialisiert Input Validator."""
        self._max_length = 10000
        self._min_length = 1
        self._forbidden_chars = {"<", ">", "{", "}", "[", "]"}
        self._suspicious_patterns = [
            r"<script.*?>.*?</script>",  # Script-Tags
            r"javascript:",              # JavaScript-URLs
            r"data:.*base64",           # Base64-Data-URLs
            r"\\x[0-9a-fA-F]{2}",       # Hex-Escape-Sequenzen
        ]

        self._compiled_suspicious = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self._suspicious_patterns
        ]

    def validate(self, prompt: str, context: dict[str, Any] | None = None) -> list[PromptThreat]:
        """Validiert Input-Format."""
        threats = []

        # Längen-Validierung
        if len(prompt) > self._max_length:
            threat = PromptThreat(
                threat_type=InjectionType.CONTEXT_MANIPULATION,
                risk_level=PromptRiskLevel.MEDIUM_RISK,
                confidence=1.0,
                description=f"Prompt zu lang: {len(prompt)} > {self._max_length}",
                detected_pattern=f"Length: {len(prompt)}",
                suggested_action=SanitizationAction.SANITIZE,
                metadata={"actual_length": len(prompt), "max_length": self._max_length}
            )
            threats.append(threat)

        if len(prompt) < self._min_length:
            threat = PromptThreat(
                threat_type=InjectionType.CONTEXT_MANIPULATION,
                risk_level=PromptRiskLevel.LOW_RISK,
                confidence=1.0,
                description=f"Prompt zu kurz: {len(prompt)} < {self._min_length}",
                detected_pattern=f"Length: {len(prompt)}",
                suggested_action=SanitizationAction.WARN,
                metadata={"actual_length": len(prompt), "min_length": self._min_length}
            )
            threats.append(threat)

        # Verbotene Zeichen
        found_forbidden = set(prompt) & self._forbidden_chars
        if found_forbidden:
            threat = PromptThreat(
                threat_type=InjectionType.CONTEXT_MANIPULATION,
                risk_level=PromptRiskLevel.MEDIUM_RISK,
                confidence=0.9,
                description=f"Verbotene Zeichen gefunden: {found_forbidden}",
                detected_pattern=str(found_forbidden),
                suggested_action=SanitizationAction.SANITIZE,
                metadata={"forbidden_chars": list(found_forbidden)}
            )
            threats.append(threat)

        # Verdächtige Patterns
        for pattern in self._compiled_suspicious:
            matches = pattern.finditer(prompt)

            for match in matches:
                threat = PromptThreat(
                    threat_type=InjectionType.CONTEXT_MANIPULATION,
                    risk_level=PromptRiskLevel.HIGH_RISK,
                    confidence=0.8,
                    description="Verdächtiges Pattern erkannt",
                    detected_pattern=match.group(),
                    suggested_action=SanitizationAction.BLOCK,
                    metadata={
                        "pattern": pattern.pattern,
                        "match_start": match.start(),
                        "match_end": match.end()
                    }
                )
                threats.append(threat)

        return threats


class PromptSanitizer:
    """Sanitisiert Prompts basierend auf erkannten Bedrohungen."""

    def __init__(self):
        """Initialisiert Prompt Sanitizer."""
        self._sanitization_rules = {
            InjectionType.DIRECT_INJECTION: self._remove_injection_patterns,
            InjectionType.JAILBREAK_ATTEMPT: self._neutralize_jailbreak,
            InjectionType.SYSTEM_PROMPT_LEAK: self._block_system_queries,
            InjectionType.INSTRUCTION_OVERRIDE: self._remove_override_attempts,
        }

    def sanitize_prompt(
        self,
        prompt: str,
        threats: list[PromptThreat]
    ) -> str:
        """Sanitisiert Prompt basierend auf erkannten Bedrohungen."""
        sanitized = prompt

        for threat in threats:
            if threat.suggested_action == SanitizationAction.SANITIZE:
                sanitizer = self._sanitization_rules.get(threat.threat_type)
                if sanitizer:
                    sanitized = sanitizer(sanitized, threat)
                else:
                    # Generische Sanitization
                    sanitized = self._generic_sanitize(sanitized, threat)

        return sanitized

    def _remove_injection_patterns(self, prompt: str, threat: PromptThreat) -> str:
        """Entfernt Injection-Patterns."""
        pattern = threat.detected_pattern
        # Ersetze mit neutralem Text
        return prompt.replace(pattern, "[REMOVED_INJECTION]")

    def _neutralize_jailbreak(self, prompt: str, threat: PromptThreat) -> str:
        """Neutralisiert Jailbreak-Versuche."""
        pattern = threat.detected_pattern
        return prompt.replace(pattern, "[NEUTRALIZED_ROLEPLAY]")

    def _block_system_queries(self, prompt: str, threat: PromptThreat) -> str:
        """Blockiert System-Prompt-Queries."""
        pattern = threat.detected_pattern
        return prompt.replace(pattern, "[BLOCKED_SYSTEM_QUERY]")

    def _remove_override_attempts(self, prompt: str, threat: PromptThreat) -> str:
        """Entfernt Instruction-Override-Versuche."""
        pattern = threat.detected_pattern
        return prompt.replace(pattern, "[REMOVED_OVERRIDE]")

    def _generic_sanitize(self, prompt: str, threat: PromptThreat) -> str:
        """Generische Sanitization."""
        pattern = threat.detected_pattern
        return prompt.replace(pattern, "[SANITIZED]")


class PromptGuardrailsEngine:
    """Engine für Prompt Guardrails."""

    def __init__(self):
        """Initialisiert Prompt Guardrails Engine."""
        self._validators: list[PromptValidator] = []
        self._sanitizer = PromptSanitizer()

        # Statistiken
        self._validations_performed = 0
        self._threats_detected = 0
        self._prompts_blocked = 0
        self._prompts_sanitized = 0

        # Standard-Validatoren registrieren
        self.register_validator(PromptInjectionDetector())
        self.register_validator(ContentFilter())
        self.register_validator(InputValidator())

    def register_validator(self, validator: PromptValidator) -> None:
        """Registriert Prompt-Validator."""
        self._validators.append(validator)
        logger.info(f"Prompt-Validator registriert: {validator.__class__.__name__}")

    @trace_function("prompt_guardrails.validate")
    def validate_prompt(
        self,
        prompt: str,
        context: dict[str, Any] | None = None
    ) -> PromptValidationResult:
        """Validiert Prompt mit allen registrierten Validatoren."""
        start_time = time.time()
        self._validations_performed += 1

        try:
            all_threats = []

            # Führe alle Validatoren aus
            for validator in self._validators:
                try:
                    threats = validator.validate(prompt, context)
                    all_threats.extend(threats)
                except Exception as e:
                    logger.exception(f"Prompt-Validation fehlgeschlagen für {validator.__class__.__name__}: {e}")

            # Bestimme Overall Risk Level
            overall_risk_level = self._determine_overall_risk_level(all_threats)

            # Bestimme ob sicher
            is_safe = overall_risk_level in [PromptRiskLevel.SAFE, PromptRiskLevel.LOW_RISK]

            # Sanitisiere falls erforderlich
            sanitized_prompt = None
            if any(threat.suggested_action == SanitizationAction.SANITIZE for threat in all_threats):
                sanitized_prompt = self._sanitizer.sanitize_prompt(prompt, all_threats)
                self._prompts_sanitized += 1

            # Aktualisiere Statistiken
            if all_threats:
                self._threats_detected += len(all_threats)

            if overall_risk_level in [PromptRiskLevel.HIGH_RISK, PromptRiskLevel.CRITICAL]:
                self._prompts_blocked += 1

            processing_time = (time.time() - start_time) * 1000

            return PromptValidationResult(
                original_prompt=prompt,
                is_safe=is_safe,
                risk_level=overall_risk_level,
                threats=all_threats,
                sanitized_prompt=sanitized_prompt,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.exception(f"Prompt-Guardrails-Validation fehlgeschlagen: {e}")
            processing_time = (time.time() - start_time) * 1000

            return PromptValidationResult(
                original_prompt=prompt,
                is_safe=False,
                risk_level=PromptRiskLevel.MEDIUM_RISK,
                threats=[PromptThreat(
                    threat_type=InjectionType.CONTEXT_MANIPULATION,
                    risk_level=PromptRiskLevel.MEDIUM_RISK,
                    confidence=0.5,
                    description=f"Validation-Fehler: {e!s}",
                    detected_pattern="validation_error",
                    suggested_action=SanitizationAction.WARN
                )],
                processing_time_ms=processing_time
            )

    def _determine_overall_risk_level(self, threats: list[PromptThreat]) -> PromptRiskLevel:
        """Bestimmt Overall Risk Level basierend auf Bedrohungen."""
        if not threats:
            return PromptRiskLevel.SAFE

        # Höchste Risk-Level bestimmt Overall Level
        risk_levels = [threat.risk_level for threat in threats]
        level_order = [
            PromptRiskLevel.SAFE,
            PromptRiskLevel.LOW_RISK,
            PromptRiskLevel.MEDIUM_RISK,
            PromptRiskLevel.HIGH_RISK,
            PromptRiskLevel.CRITICAL
        ]

        max_level = PromptRiskLevel.SAFE
        for level in risk_levels:
            if level_order.index(level) > level_order.index(max_level):
                max_level = level

        return max_level

    def get_guardrails_statistics(self) -> dict[str, Any]:
        """Gibt Prompt-Guardrails-Statistiken zurück."""
        return {
            "validations_performed": self._validations_performed,
            "threats_detected": self._threats_detected,
            "prompts_blocked": self._prompts_blocked,
            "prompts_sanitized": self._prompts_sanitized,
            "block_rate": self._prompts_blocked / max(self._validations_performed, 1),
            "sanitization_rate": self._prompts_sanitized / max(self._validations_performed, 1),
            "threat_detection_rate": self._threats_detected / max(self._validations_performed, 1),
            "registered_validators": len(self._validators)
        }


# Globale Prompt Guardrails Engine Instanz
prompt_guardrails_engine = PromptGuardrailsEngine()
