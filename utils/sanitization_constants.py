# backend/utils/sanitization_constants.py
"""Konstanten für Prompt-Sanitization.

Zentralisiert alle Magic Numbers und Konfigurationswerte für bessere Wartbarkeit.
"""

from __future__ import annotations

import re
from re import Pattern

# Confidence Score Faktoren
CONFIDENCE_SCORE_FACTORS = {
    "LENGTH_VIOLATION": 0.1,
    "BLACKLIST_VIOLATION": 0.2,
    "PATTERN_VIOLATION": 0.3,
}

# Dateiname-Generierung
UNIQUE_ID_LENGTH = 8
FILENAME_SEPARATOR_LENGTH = 4

# Performance-Metriken
MILLISECONDS_CONVERSION_FACTOR = 1000

# Sanitization-Marker
SANITIZATION_MARKERS = {
    "INJECTION": "[REMOVED_INJECTION]",
    "JAILBREAK": "[NEUTRALIZED_ROLEPLAY]",
    "SYSTEM_QUERY": "[BLOCKED_SYSTEM_QUERY]",
    "OVERRIDE": "[REMOVED_OVERRIDE]",
    "GENERIC": "[SANITIZED]",
}

# Gefährliche Patterns für bessere Performance vorkompiliert
DANGEROUS_PATTERNS: dict[str, Pattern[str]] = {
    # Script injection
    "script_injection": re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE),
    "javascript_protocol": re.compile(r"javascript:", re.IGNORECASE),
    "vbscript_protocol": re.compile(r"vbscript:", re.IGNORECASE),
    "data_html": re.compile(r"data:text/html", re.IGNORECASE),

    # Explicit content patterns
    "explicit_generation_en": re.compile(
        r"\b(make|create|generate|show)\s+(me\s+)?(a\s+)?(nude|naked|sexual)",
        re.IGNORECASE
    ),
    "explicit_generation_de": re.compile(
        r"\b(erstelle|mache|zeige)\s+(mir\s+)?(ein\s+)?(nackt|sexuell)",
        re.IGNORECASE
    ),

    # Realistic person generation (problematic)
    "realistic_person_en": re.compile(
        r"\b(realistic|photorealistic)\s+(person|human|woman|man)",
        re.IGNORECASE
    ),
    "realistic_person_de": re.compile(
        r"\b(realistisch|fotorealistisch)\s+(person|mensch|frau|mann)",
        re.IGNORECASE
    ),

    # Celebrity/public figure generation
    "celebrity_person_en": re.compile(
        r"\b(famous|celebrity|politician)\s+person",
        re.IGNORECASE
    ),
    "celebrity_person_de": re.compile(
        r"\b(berühmt|prominente|politiker)\s+person",
        re.IGNORECASE
    ),

    # Command injection
    "command_injection": re.compile(r"[;&|`$(){}[\]\\]"),
    "dangerous_commands": re.compile(r"\b(rm|del|format|shutdown|reboot|kill)\b", re.IGNORECASE),

    # Path traversal
    "path_traversal": re.compile(r"\.\./|\.\.\\"),

    # Prompt injection patterns
    "direct_injection": re.compile(
        r"\b(ignore|forget|disregard)\s+(all\s+)?(previous|above|all)\s+(instructions?|prompts?)",
        re.IGNORECASE
    ),
    "jailbreak_attempt": re.compile(
        r"\b(pretend|act|roleplay)\s+(as|like)\s+(a|an)?\s*(evil|harmful|malicious)",
        re.IGNORECASE
    ),
    "system_prompt_leak": re.compile(
        r"\b(show|tell|reveal)\s+(me\s+)?(your|the)\s+(system|initial)\s+(prompt|instructions?)",
        re.IGNORECASE
    ),
    "instruction_override": re.compile(
        r"\b(new|updated?)\s+(instructions?|rules?|guidelines?)\s*:",
        re.IGNORECASE
    ),
}

# Replacement-Patterns für finale Sanitization
REPLACEMENT_PATTERNS: dict[str, str] = {
    # Normalisiere Whitespace
    r"\s+": " ",
    # Entferne gefährliche Zeichen
    r"[<>\"'`]": "",
    # Normalisiere Interpunktion
    r"\.{2,}": ".",
    r"\!{2,}": "!",
    r"\?{2,}": "?",
}

# Threat-Type zu Marker Mapping
THREAT_MARKER_MAP = {
    "direct_injection": SANITIZATION_MARKERS["INJECTION"],
    "jailbreak_attempt": SANITIZATION_MARKERS["JAILBREAK"],
    "system_prompt_leak": SANITIZATION_MARKERS["SYSTEM_QUERY"],
    "instruction_override": SANITIZATION_MARKERS["OVERRIDE"],
    "explicit_content": SANITIZATION_MARKERS["GENERIC"],
    "command_injection": SANITIZATION_MARKERS["INJECTION"],
}

__all__ = [
    "CONFIDENCE_SCORE_FACTORS",
    "DANGEROUS_PATTERNS",
    "FILENAME_SEPARATOR_LENGTH",
    "MILLISECONDS_CONVERSION_FACTOR",
    "REPLACEMENT_PATTERNS",
    "SANITIZATION_MARKERS",
    "THREAT_MARKER_MAP",
    "UNIQUE_ID_LENGTH",
]
