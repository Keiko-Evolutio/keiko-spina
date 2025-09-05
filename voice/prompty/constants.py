"""Konstanten für das Prompty-Modul.

Zentrale Definition aller Konstanten für Template-Management,
Validation und Konfiguration mit Type-Safety und klarer Dokumentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

# =============================================================================
# Template-Konstanten
# =============================================================================

# Template-Dateien
DEFAULT_TEMPLATE_NAME: Final[str] = "voice"
TEMPLATE_FILE_EXTENSION: Final[str] = ".prompty"
TEMPLATE_DIRECTORY: Final[Path] = Path(__file__).parent

# Template-Versionen
SUPPORTED_TEMPLATE_VERSIONS: Final[tuple[str, ...]] = ("1.0.0", "1.1.0")
DEFAULT_TEMPLATE_VERSION: Final[str] = "1.0.0"

# Template-Größe
MAX_TEMPLATE_SIZE_BYTES: Final[int] = int(os.getenv("PROMPTY_MAX_SIZE", "51200"))   # 50KB
MIN_TEMPLATE_SIZE_BYTES: Final[int] = int(os.getenv("PROMPTY_MIN_SIZE", "10"))      # 10 Bytes

# =============================================================================
# Metadata-Konstanten
# =============================================================================

# Erforderliche Metadata-Felder
REQUIRED_METADATA_FIELDS: Final[frozenset[str]] = frozenset([
    "name",
])

# Optionale Metadata-Felder
OPTIONAL_METADATA_FIELDS: Final[frozenset[str]] = frozenset([
    "category",
    "version",
    "author",
    "description",
    "tools",
    "parameters",
    "examples",
])

# Alle unterstützten Metadata-Felder
ALL_METADATA_FIELDS: Final[frozenset[str]] = REQUIRED_METADATA_FIELDS | OPTIONAL_METADATA_FIELDS

# =============================================================================
# Kategorie-Konstanten
# =============================================================================

# Erlaubte Template-Kategorien
ALLOWED_TEMPLATE_CATEGORIES: Final[frozenset[str]] = frozenset([
    "assistant",
    "system",
    "user",
    "tool",
    "custom",
])

DEFAULT_TEMPLATE_CATEGORY: Final[str] = "assistant"

# =============================================================================
# Parsing-Konstanten
# =============================================================================

# YAML-Delimiter
YAML_DELIMITER: Final[str] = "---"
MIN_YAML_PARTS: Final[int] = 3  # Start delimiter, YAML content, End delimiter + Template

# Encoding
DEFAULT_ENCODING: Final[str] = "utf-8"

# =============================================================================
# Validation-Konstanten
# =============================================================================

# Template-Name Validation
MAX_TEMPLATE_NAME_LENGTH: Final[int] = 100
MIN_TEMPLATE_NAME_LENGTH: Final[int] = 1

# Template-Content Validation
MAX_TEMPLATE_CONTENT_LENGTH: Final[int] = 50000  # 50KB
MIN_TEMPLATE_CONTENT_LENGTH: Final[int] = 1

# Parameter Validation
MAX_PARAMETER_COUNT: Final[int] = 50
MAX_PARAMETER_NAME_LENGTH: Final[int] = 50

# =============================================================================
# Error-Messages
# =============================================================================

# Template-Errors
ERROR_TEMPLATE_NOT_FOUND: Final[str] = "Template '{template_name}' nicht gefunden in {directory}"
ERROR_TEMPLATE_TOO_LARGE: Final[str] = "Template zu groß: {size} Bytes (Max: {max_size})"
ERROR_TEMPLATE_TOO_SMALL: Final[str] = "Template zu klein: {size} Bytes (Min: {min_size})"
ERROR_INVALID_TEMPLATE_FORMAT: Final[str] = "Ungültiges Template-Format: {reason}"

# Parsing-Errors
ERROR_YAML_PARSING: Final[str] = "YAML-Parsing fehlgeschlagen: {error}"
ERROR_MISSING_DELIMITER: Final[str] = "Ungültiges Front Matter-Format"
ERROR_INVALID_YAML_STRUCTURE: Final[str] = "Ungültige YAML-Struktur: {reason}"

# Validation-Errors
ERROR_MISSING_REQUIRED_FIELD: Final[str] = "Erforderliches Feld fehlt: {field}"
ERROR_INVALID_FIELD_VALUE: Final[str] = "Ungültiger Wert für Feld '{field}': {value}"
ERROR_INVALID_CATEGORY: Final[str] = "Ungültige Kategorie '{category}'. Erlaubt: {allowed}"

# Rendering-Errors
ERROR_RENDERING_FAILED: Final[str] = "Template-Rendering fehlgeschlagen: {error}"
ERROR_MISSING_PARAMETER: Final[str] = "Erforderlicher Parameter fehlt: {parameter}"
ERROR_INVALID_PARAMETER_TYPE: Final[str] = "Ungültiger Parameter-Typ für '{parameter}': {type}"

# =============================================================================
# Default-Konfiguration
# =============================================================================

DEFAULT_TEMPLATE_CONFIG: Final[dict[str, Any]] = {
    "encoding": DEFAULT_ENCODING,
    "max_size": MAX_TEMPLATE_SIZE_BYTES,
    "min_size": MIN_TEMPLATE_SIZE_BYTES,
    "allowed_categories": list(ALLOWED_TEMPLATE_CATEGORIES),
    "default_category": DEFAULT_TEMPLATE_CATEGORY,
    "strict_validation": False,
    "cache_templates": True,
    "auto_reload": False,
}

# =============================================================================
# Environment-basierte Konfiguration
# =============================================================================

# Template-Verzeichnis (überschreibbar via Environment)
TEMPLATE_DIR_ENV: Final[str] = os.getenv("PROMPTY_TEMPLATE_DIR", str(TEMPLATE_DIRECTORY))
TEMPLATE_DIR: Final[Path] = Path(TEMPLATE_DIR_ENV)

# Debug-Modus
DEBUG_MODE: Final[bool] = os.getenv("PROMPTY_DEBUG", "false").lower() == "true"

# Cache-Konfiguration
CACHE_ENABLED: Final[bool] = os.getenv("PROMPTY_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS: Final[int] = int(os.getenv("PROMPTY_CACHE_TTL", "3600"))  # 1 Stunde

# =============================================================================
# Logging-Konstanten
# =============================================================================

# Logger-Namen
LOGGER_NAME: Final[str] = "voice.prompty"
PARSER_LOGGER_NAME: Final[str] = f"{LOGGER_NAME}.parser"
TEMPLATE_LOGGER_NAME: Final[str] = f"{LOGGER_NAME}.template"
VALIDATION_LOGGER_NAME: Final[str] = f"{LOGGER_NAME}.validation"

# Log-Messages
LOG_TEMPLATE_LOADED: Final[str] = "Template '{name}' erfolgreich geladen aus {path}"
LOG_TEMPLATE_CACHED: Final[str] = "Template '{name}' im Cache gespeichert"
LOG_TEMPLATE_RENDERED: Final[str] = "Template '{name}' erfolgreich gerendert"
LOG_VALIDATION_SUCCESS: Final[str] = "Template-Validation erfolgreich für '{name}'"
LOG_VALIDATION_FAILED: Final[str] = "Template-Validation fehlgeschlagen für '{name}': {reason}"
