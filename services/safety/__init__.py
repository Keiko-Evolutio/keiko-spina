"""Safety Services für Keiko Personal Assistant.

Konsolidierte Safety-Funktionalitäten für Content-Filtering, Prompt-Sanitization
und Security-Checks. Dieses Modul vereint alle Safety-bezogenen Implementierungen
aus der Keiko-Codebase in einem einheitlichen, wiederverwendbaren System.

Hauptkomponenten:
- ConsolidatedSafetyFilter: Hauptklasse für alle Safety-Checks
- SafetyFilterResult: Ergebnis-Datenstruktur
- SafetyConfig: Konfigurationsklasse
- SafetyKeywordRegistry: Zentrale Keyword-Verwaltung

Verwendung:
    from services.safety import create_safety_filter, SafetyConfig

    # Standard-Filter
    safety_filter = create_safety_filter()
    result = safety_filter.analyze_prompt("Test prompt")

    # Mit Custom-Config
    config = SafetyConfig()
    config.MAX_PROMPT_LENGTH = 1000
    safety_filter = create_safety_filter(config)
"""

from .prompt_safety_filter import (
    # Hauptklassen
    ConsolidatedSafetyFilter,
    ContentSanitizer,
    PromptSafetyFilter,  # Backward-Compatibility Alias
    # Konfiguration
    SafetyConfig,
    # Abstrakte Basis-Klassen
    SafetyDetector,
    # Datenstrukturen
    SafetyFilterResult,
    SafetyKeywordRegistry,
    # Enums
    SafetyLevel,
    SafetyViolation,
    ViolationType,
    # Factory-Funktion
    create_safety_filter,
)

# Öffentliche API
__all__ = [
    # Hauptklassen
    "ConsolidatedSafetyFilter",
    "ContentSanitizer",
    "PromptSafetyFilter",
    # Konfiguration
    "SafetyConfig",
    # Abstrakte Basis-Klassen
    "SafetyDetector",
    # Datenstrukturen
    "SafetyFilterResult",
    "SafetyKeywordRegistry",
    # Enums
    "SafetyLevel",
    "SafetyViolation",
    "ViolationType",
    # Factory-Funktion
    "create_safety_filter",
]

# Modul-Metadaten
__version__ = "1.0.0"
__author__ = "Keiko Development Team"
__description__ = "Konsolidierte Safety Services für Content-Filtering und Security-Checks"

# Convenience-Funktionen
def get_default_safety_filter() -> ConsolidatedSafetyFilter:
    """Gibt Standard Safety Filter Instanz zurück."""
    return create_safety_filter()


def quick_safety_check(prompt: str) -> bool:
    """Schnelle Safety-Prüfung für einfache Anwendungsfälle."""
    filter_instance = get_default_safety_filter()
    result = filter_instance.analyze_prompt(prompt)
    return result.is_safe


def get_safety_statistics() -> dict:
    """Gibt Safety-System Statistiken zurück."""
    return {
        "module": "backend.services.safety",
        "version": __version__,
        "components": {
            "consolidated_filter": True,
            "keyword_registry": True,
            "pattern_detection": True,
            "multi_language_support": True,
            "configurable_thresholds": True,
        },
        "supported_checks": [
            "blocked_keywords",
            "suspicious_patterns",
            "prompt_length",
            "multiple_languages",
            "injection_attempts",
            "toxicity_detection"
        ]
    }
