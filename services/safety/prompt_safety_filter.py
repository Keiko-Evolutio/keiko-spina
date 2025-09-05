"""Konsolidierter Safety Filter für Content Policy Compliance.

Implementiert umfassende Prompt-Sanitization, Content-Filtering und Safety-Checks
für Azure OpenAI und andere AI-Services. Konsolidiert alle Safety-Funktionalitäten
aus der Keiko-Codebase in einem einheitlichen, wiederverwendbaren Modul.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# Konstanten für bessere Wartbarkeit
class SafetyLevel(Enum):
    """Safety-Level für Content-Bewertung."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    UNSAFE = "unsafe"


class ViolationType(Enum):
    """Typen von Safety-Verletzungen."""
    BLOCKED_KEYWORD = "blocked_keyword"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    PROMPT_TOO_LONG = "prompt_too_long"
    MULTIPLE_LANGUAGES = "multiple_languages"
    INJECTION_ATTEMPT = "injection_attempt"
    TOXICITY = "toxicity"


# Konfigurationskonstanten
class SafetyConfig:
    """Zentrale Konfiguration für Safety-Parameter."""

    # Längen-Limits
    MAX_PROMPT_LENGTH: int = 500
    MAX_SANITIZED_LENGTH: int = 400

    # Confidence-Score Schwellwerte
    SAFE_THRESHOLD: float = 0.5
    KEYWORD_PENALTY: float = 0.2
    PATTERN_PENALTY: float = 0.3
    LENGTH_PENALTY: float = 0.1
    LANGUAGE_PENALTY: float = 0.1

    # Suggestion-Limits
    MAX_SUGGESTIONS: int = 3


@dataclass
class SafetyViolation:
    """Einzelne Safety-Verletzung."""
    violation_type: ViolationType
    description: str
    confidence: float
    detected_content: str
    suggested_replacement: str | None = None


@dataclass
class SafetyFilterResult:
    """Konsolidiertes Ergebnis der Safety-Analyse."""
    is_safe: bool
    sanitized_prompt: str
    original_prompt: str
    violations: list[SafetyViolation]
    confidence_score: float
    safety_level: SafetyLevel
    processing_time_ms: float | None = None


# Abstrakte Basis-Klassen für Erweiterbarkeit
class SafetyDetector(ABC):
    """Basis-Klasse für Safety-Detektoren."""

    @abstractmethod
    def detect_violations(self, content: str) -> list[SafetyViolation]:
        """Detektiert Safety-Verletzungen in Content."""


class ContentSanitizer(ABC):
    """Basis-Klasse für Content-Sanitizer."""

    @abstractmethod
    def sanitize_content(self, content: str, violations: list[SafetyViolation]) -> str:
        """Sanitisiert Content basierend auf Verletzungen."""


# Konsolidierte Safety-Keyword-Registry
class SafetyKeywordRegistry:
    """Zentrale Registry für alle Safety-Keywords."""

    @staticmethod
    def get_blocked_keywords() -> set[str]:
        """Gibt konsolidierte Liste aller blockierten Keywords zurück."""
        return {
            # Gewalt und Waffen
            "weapon", "gun", "knife", "violence", "blood", "death", "kill",
            "waffe", "gewalt", "blut", "tod", "töten", "messer",

            # Explizite Inhalte
            "nude", "naked", "sexual", "explicit", "adult",
            "nackt", "sexuell", "explizit", "erwachsenen",

            # Hassrede und Diskriminierung
            "hate", "racist", "discrimination", "nazi", "terrorist",
            "hass", "rassistisch", "diskriminierung", "terror",

            # Drogen und illegale Substanzen
            "drug", "cocaine", "heroin", "marijuana", "illegal",
            "droge", "kokain", "marihuana",

            # Selbstverletzung
            "suicide", "self-harm", "cutting", "depression",
            "selbstmord", "selbstverletzung",

            # Toxische Begriffe (aus safety_guardrails.py konsolidiert)
            "stupid", "idiot", "moron", "bomb",

            # Politische Figuren (problematisch für AI)
            "hitler", "stalin", "putin", "trump", "biden",

            # Urheberrechtlich geschützte Charaktere
            "mickey mouse", "superman", "batman", "disney",
            "marvel", "dc comics", "pokemon", "nintendo"
        }

    @staticmethod
    def get_suspicious_patterns() -> list[str]:
        """Gibt konsolidierte Liste verdächtiger Patterns zurück."""
        return [
            # Content-Generation Patterns
            r"\b(make|create|generate|show)\s+(me\s+)?(a\s+)?(nude|naked|sexual)",
            r"\b(erstelle|mache|zeige)\s+(mir\s+)?(ein\s+)?(nackt|sexuell)",
            r"\b(realistic|photorealistic)\s+(person|human|woman|man)",
            r"\b(realistisch|fotorealistisch)\s+(person|mensch|frau|mann)",
            r"\b(famous|celebrity|politician)\s+person",
            r"\b(berühmt|prominente|politiker)\s+person",

            # Injection Patterns (aus input_sanitizer.py konsolidiert)
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"data:text/html",
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"[;&|`$(){}[\]\\]",
            r"\.\./|\.\.\\",
        ]


class ConsolidatedSafetyFilter:
    """Konsolidierter Safety Filter für alle Content-Typen."""

    def __init__(self, config: SafetyConfig | None = None):
        """Initialisiert konsolidierten Safety Filter."""
        self.config = config or SafetyConfig()
        self._blocked_keywords = SafetyKeywordRegistry.get_blocked_keywords()
        self._suspicious_patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self._suspicious_patterns
        ]
        self._replacement_patterns = self._load_replacement_patterns()
        self._safe_alternatives = self._load_safe_alternatives()

    def _load_replacement_patterns(self) -> dict[str, str]:
        """Lädt konsolidierte Ersetzungsmuster für problematische Begriffe."""
        return {
            # Gewalt → Friedlich
            r"\b(fight|fighting|battle|war)\b": "peaceful scene",
            r"\b(kämpf|kampf|krieg|schlacht)\b": "friedliche Szene",

            # Waffen → Werkzeuge
            r"\b(sword|gun|weapon)\b": "tool",
            r"\b(schwert|waffe|gewehr)\b": "Werkzeug",

            # Explizit → Künstlerisch
            r"\b(sexy|hot|attractive)\b": "elegant",
            r"\b(sexy|heiß|attraktiv)\b": "elegant",

            # Dunkel → Neutral
            r"\b(dark|evil|sinister)\b": "mysterious",
            r"\b(dunkel|böse|finster)\b": "mysteriös",

            # Extreme → Moderat
            r"\b(extreme|intense|hardcore)\b": "dynamic",
            r"\b(extrem|intensiv|hardcore)\b": "dynamisch"
        }

    def _load_safe_alternatives(self) -> dict[str, list[str]]:
        """Lädt sichere Alternativen für problematische Konzepte."""
        return {
            "person": [
                "a person", "someone", "a character", "an individual",
                "eine Person", "jemand", "ein Charakter", "eine Figur"
            ],
            "action": [
                "standing", "walking", "sitting", "smiling", "thinking",
                "stehend", "gehend", "sitzend", "lächelnd", "denkend"
            ],
            "setting": [
                "in a park", "in a garden", "in a peaceful place", "outdoors",
                "in einem Park", "in einem Garten", "an einem friedlichen Ort"
            ],
            "style": [
                "artistic", "beautiful", "elegant", "professional", "creative",
                "künstlerisch", "schön", "elegant", "professionell", "kreativ"
            ]
        }

    def analyze_prompt(self, prompt: str) -> SafetyFilterResult:
        """Analysiert Prompt auf Safety-Probleme mit konsolidierter Logik."""
        import time  # pylint: disable=import-outside-toplevel
        start_time = time.time()

        violations = []

        # Führe alle Safety-Checks durch
        violations.extend(self._check_blocked_keywords(prompt))
        violations.extend(self._check_suspicious_patterns(prompt))
        violations.extend(self._check_prompt_length(prompt))
        violations.extend(self._check_multiple_languages(prompt))

        # Berechne Confidence Score und Safety Level
        confidence_score = self._calculate_confidence_score(violations)
        safety_level = self._determine_safety_level(confidence_score, violations)
        is_safe = safety_level == SafetyLevel.SAFE and len(violations) == 0

        # Sanitisiere falls erforderlich
        sanitized_prompt = self._sanitize_prompt(prompt, violations) if not is_safe else prompt

        processing_time = (time.time() - start_time) * 1000

        return SafetyFilterResult(
            is_safe=is_safe,
            sanitized_prompt=sanitized_prompt,
            original_prompt=prompt,
            violations=violations,
            confidence_score=confidence_score,
            safety_level=safety_level,
            processing_time_ms=processing_time
        )
    def _check_blocked_keywords(self, prompt: str) -> list[SafetyViolation]:
        """Prüft auf blockierte Keywords."""
        violations = []
        prompt_lower = prompt.lower()

        for keyword in self._blocked_keywords:
            if keyword.lower() in prompt_lower:
                violation = SafetyViolation(
                    violation_type=ViolationType.BLOCKED_KEYWORD,
                    description=f"Blockiertes Keyword gefunden: {keyword}",
                    confidence=0.9,
                    detected_content=keyword,
                    suggested_replacement="[REDACTED]"
                )
                violations.append(violation)

        return violations

    def _check_suspicious_patterns(self, prompt: str) -> list[SafetyViolation]:
        """Prüft auf verdächtige Patterns."""
        violations = []

        for pattern in self._compiled_patterns:
            matches = pattern.finditer(prompt)
            for match in matches:
                violation = SafetyViolation(
                    violation_type=ViolationType.SUSPICIOUS_PATTERN,
                    description=f"Verdächtiges Pattern erkannt: {pattern.pattern}",
                    confidence=0.8,
                    detected_content=match.group(),
                    suggested_replacement="[SANITIZED]"
                )
                violations.append(violation)

        return violations

    def _check_prompt_length(self, prompt: str) -> list[SafetyViolation]:
        """Prüft Prompt-Länge."""
        violations = []

        if len(prompt) > self.config.MAX_PROMPT_LENGTH:
            violation = SafetyViolation(
                violation_type=ViolationType.PROMPT_TOO_LONG,
                description=f"Prompt zu lang: {len(prompt)} > {self.config.MAX_PROMPT_LENGTH}",
                confidence=1.0,
                detected_content=f"Length: {len(prompt)}",
                suggested_replacement=f"Truncated to {self.config.MAX_PROMPT_LENGTH} chars"
            )
            violations.append(violation)

        return violations

    def _check_multiple_languages(self, prompt: str) -> list[SafetyViolation]:
        """Prüft auf mehrere Sprachen."""
        violations = []

        if self._contains_multiple_languages(prompt):
            violation = SafetyViolation(
                violation_type=ViolationType.MULTIPLE_LANGUAGES,
                description="Mehrere Sprachen erkannt (kann problematisch sein)",
                confidence=0.6,
                detected_content="Mixed languages",
                suggested_replacement="Single language version"
            )
            violations.append(violation)

        return violations

    def _calculate_confidence_score(self, violations: list[SafetyViolation]) -> float:
        """Berechnet Confidence Score basierend auf Verletzungen."""
        if not violations:
            return 1.0

        score = 1.0
        for violation in violations:
            if violation.violation_type == ViolationType.BLOCKED_KEYWORD:
                score -= self.config.KEYWORD_PENALTY
            elif violation.violation_type == ViolationType.SUSPICIOUS_PATTERN:
                score -= self.config.PATTERN_PENALTY
            elif violation.violation_type == ViolationType.PROMPT_TOO_LONG:
                score -= self.config.LENGTH_PENALTY
            elif violation.violation_type == ViolationType.MULTIPLE_LANGUAGES:
                score -= self.config.LANGUAGE_PENALTY

        return max(0.0, score)

    def _determine_safety_level(self, confidence_score: float, violations: list[SafetyViolation]) -> SafetyLevel:
        """Bestimmt Safety Level basierend auf Score und Verletzungen."""
        if confidence_score >= 0.9 and not violations:
            return SafetyLevel.SAFE
        if confidence_score >= 0.7:
            return SafetyLevel.LOW_RISK
        if confidence_score >= 0.5:
            return SafetyLevel.MEDIUM_RISK
        if confidence_score >= 0.3:
            return SafetyLevel.HIGH_RISK
        return SafetyLevel.UNSAFE

    def _sanitize_prompt(self, prompt: str, violations: list[SafetyViolation]) -> str:
        """Sanitisiert problematischen Prompt basierend auf Verletzungen."""
        sanitized = prompt

        # 1. Replacement Patterns anwenden
        for pattern, replacement in self._replacement_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # 2. Spezifische Verletzungen behandeln
        for violation in violations:
            if violation.suggested_replacement and violation.detected_content in sanitized:
                sanitized = sanitized.replace(
                    violation.detected_content,
                    violation.suggested_replacement
                )

        # 3. Sicherheits-Präfix hinzufügen
        sanitized = self._add_safety_prefix(sanitized)

        # 4. Länge begrenzen
        if len(sanitized) > self.config.MAX_SANITIZED_LENGTH:
            sanitized = sanitized[:self.config.MAX_SANITIZED_LENGTH] + "..."

        return sanitized.strip()

    def _add_safety_prefix(self, content: str) -> str:
        """Fügt Sicherheits-Präfix hinzu falls erforderlich."""
        safety_prefixes = [
            "A safe, family-friendly image of",
            "An artistic representation of",
            "A creative illustration showing"
        ]

        # Prüfe ob bereits ein Präfix vorhanden ist
        if not any(prefix.lower() in content.lower() for prefix in safety_prefixes):
            return f"A safe, artistic image of {content}"

        return content

    def _contains_multiple_languages(self, text: str) -> bool:
        """Prüft ob Text mehrere Sprachen enthält (verbesserte Heuristik)."""
        # Deutsche Indikatoren
        german_indicators = {"der", "die", "das", "und", "mit", "von", "zu", "ein", "eine", "ist", "sind"}
        # Englische Indikatoren
        english_indicators = {"the", "and", "with", "of", "to", "a", "an", "in", "on", "is", "are"}

        words = set(text.lower().split())

        has_german = bool(words & german_indicators)
        has_english = bool(words & english_indicators)

        return has_german and has_english

    def get_safe_fallback_prompt(self, original_intent: str = "") -> str:
        """Gibt sicheren Fallback-Prompt basierend auf Intent zurück."""
        fallback_mapping = {
            "landscape": "A beautiful landscape with mountains and trees",
            "landschaft": "Eine schöne Landschaft mit Bergen und Bäumen",
            "garden": "A peaceful garden with colorful flowers",
            "garten": "Ein friedlicher Garten mit bunten Blumen",
            "abstract": "An abstract artistic composition with geometric shapes",
            "abstrakt": "Eine abstrakte künstlerische Komposition",
            "library": "A cozy library with books and warm lighting",
            "bibliothek": "Eine gemütliche Bibliothek mit Büchern"
        }

        intent_lower = original_intent.lower()

        # Suche nach passenden Keywords
        for keyword, fallback in fallback_mapping.items():
            if keyword in intent_lower:
                return fallback

        # Standard Fallback
        return "A beautiful landscape with mountains and trees"

    def create_safe_prompt_suggestions(self, unsafe_prompt: str) -> list[str]:
        """Erstellt sichere Prompt-Vorschläge basierend auf Konzept-Extraktion."""
        core_concepts = self._extract_core_concepts(unsafe_prompt)
        suggestions = []

        # Konzept-basierte Vorschläge
        concept_suggestions = {
            "nature": "A beautiful natural landscape with trees and flowers",
            "natur": "Eine schöne natürliche Landschaft mit Bäumen und Blumen",
            "person": "An artistic silhouette of a person in a peaceful setting",
            "mensch": "Eine künstlerische Silhouette einer Person in friedlicher Umgebung",
            "building": "A beautiful architectural structure in daylight",
            "gebäude": "Ein schönes architektonisches Bauwerk bei Tageslicht",
            "animal": "A cute, friendly animal in its natural habitat",
            "tier": "Ein niedliches, freundliches Tier in seinem natürlichen Lebensraum"
        }

        # Sammle Vorschläge basierend auf gefundenen Konzepten
        for concept in core_concepts:
            if concept in concept_suggestions:
                suggestions.append(concept_suggestions[concept])

        # Fallback-Vorschläge falls keine Konzepte gefunden
        if not suggestions:
            suggestions = [
                "A peaceful landscape with soft colors",
                "An artistic composition with geometric shapes",
                "A serene natural scene with water and trees"
            ]

        return suggestions[:self.config.MAX_SUGGESTIONS]

    def _extract_core_concepts(self, prompt: str) -> list[str]:
        """Extrahiert Kern-Konzepte aus Prompt mit optimierter Logik."""
        concept_keywords = {
            "nature": {"nature", "landscape", "tree", "flower", "mountain", "natur", "landschaft", "baum", "blume", "berg"},
            "person": {"person", "human", "people", "character", "mensch", "leute", "charakter"},
            "building": {"building", "house", "architecture", "structure", "gebäude", "haus", "architektur"},
            "animal": {"animal", "animals", "cat", "dog", "bird", "wildlife", "tier", "tiere", "katze", "hund", "vogel"},
            "object": {"object", "item", "thing", "tool", "objekt", "gegenstand", "ding", "werkzeug"}
        }

        words = set(prompt.lower().split())
        found_concepts = []

        for concept, keywords in concept_keywords.items():
            if words & keywords:  # Set intersection - mehr effizient
                found_concepts.append(concept)

        return found_concepts


# Factory-Funktion statt globaler Instanz
def create_safety_filter(config: SafetyConfig | None = None) -> ConsolidatedSafetyFilter:
    """Factory-Funktion für Safety Filter Instanzen."""
    return ConsolidatedSafetyFilter(config)


# Backward-Compatibility Alias
PromptSafetyFilter = ConsolidatedSafetyFilter

# Globale Instanz für einfache Verwendung
prompt_safety_filter = ConsolidatedSafetyFilter()

__all__ = [
    "ConsolidatedSafetyFilter",
    "PromptSafetyFilter",
    "SafetyConfig",
    "SafetyLevel",
    "ViolationType",
    "create_safety_filter",
    "prompt_safety_filter",
]
