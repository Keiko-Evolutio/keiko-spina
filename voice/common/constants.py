# backend/voice/common/constants.py
"""Voice-spezifische Konstanten und Konfigurationen.

Zentrale Definition aller Magic Numbers, Hard-coded Strings und
Konfigurationswerte für das Voice-Modul.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# =============================================================================
# File und Directory Konstanten
# =============================================================================

# Basis-Pfade
VOICE_MODULE_DIR: Final[Path] = Path(__file__).parent
PROMPTY_DIR: Final[Path] = VOICE_MODULE_DIR / "prompty"

# Dateinamen
DEFAULT_PROMPTY_FILENAME: Final[str] = "voice.prompty"
INLINE_PATH_NAME: Final[str] = "inline"

# File-Encoding
DEFAULT_FILE_ENCODING: Final[str] = "utf-8"

# =============================================================================
# YAML und Prompty Konstanten
# =============================================================================

# YAML-Delimiter
YAML_DELIMITER: Final[str] = "---"
YAML_DELIMITER_COUNT: Final[int] = 2  # Mindestanzahl für gültiges Front Matter

# Prompty Front Matter Keys
PROMPTY_NAME_KEY: Final[str] = "name"
PROMPTY_CATEGORY_KEY: Final[str] = "category"
PROMPTY_TOOLS_KEY: Final[str] = "tools"
PROMPTY_MODEL_KEY: Final[str] = "model"

# Default-Werte für Configuration
DEFAULT_CATEGORY: Final[str] = "assistant"
DEFAULT_TOOLS: Final[list[str]] = []

# =============================================================================
# Cosmos DB Konstanten
# =============================================================================

# Partition Key
COSMOS_PARTITION_KEY_PATH: Final[str] = "/category"

# SQL-Queries
COSMOS_QUERY_DEFAULT_CONFIG: Final[str] = "SELECT * FROM c WHERE c.default = true"

# Container-Konfiguration
COSMOS_ENABLE_CROSS_PARTITION: Final[bool] = True

# =============================================================================
# Configuration Keys
# =============================================================================

# Configuration-Felder
CONFIG_ID_KEY: Final[str] = "id"
CONFIG_NAME_KEY: Final[str] = "name"
CONFIG_CATEGORY_KEY: Final[str] = "category"
CONFIG_DEFAULT_KEY: Final[str] = "default"
CONFIG_CONTENT_KEY: Final[str] = "content"
CONFIG_TOOLS_KEY: Final[str] = "tools"

# =============================================================================
# Error Messages
# =============================================================================

# Cosmos DB Errors
ERROR_COSMOSDB_CONNECTION: Final[str] = "COSMOSDB_CONNECTION not set."
ERROR_COSMOSDB_CONNECT: Final[str] = "Failed to connect to CosmosDB: %s"
ERROR_COSMOSDB_QUERY: Final[str] = "Query-Fehler: %s"
ERROR_LOAD_DEFAULT_CONFIG: Final[str] = "Fehler beim Laden der Standard-Konfiguration: %s"

# File Errors
ERROR_PROMPTY_NOT_FOUND: Final[str] = "Prompty-Datei nicht gefunden: %s"
ERROR_PROMPTY_READ: Final[str] = "Fehler beim Lesen der Prompty-Datei %s: %s"
ERROR_PROMPTY_PARSE: Final[str] = "Fehler beim Parsen der Prompty-Datei %s: %s"
ERROR_FALLBACK_CONFIG: Final[str] = "Fallback-Konfiguration konnte nicht geladen werden: %s"

# =============================================================================
# Environment-basierte Konfiguration
# =============================================================================

class VoiceConfig:
    """Environment-Variable-basierte Voice-Konfiguration.

    Zentrale Konfigurationsklasse für alle Voice-spezifischen Einstellungen
    mit Fallback-Werten und Type-Safety.
    """

    # Prompty-Konfiguration
    PROMPTY_DIR: Final[Path] = Path(
        os.getenv("VOICE_PROMPTY_DIR", str(PROMPTY_DIR))
    )

    DEFAULT_PROMPTY_FILE: Final[str] = os.getenv(
        "VOICE_DEFAULT_PROMPTY_FILE",
        DEFAULT_PROMPTY_FILENAME
    )

    # File-Handling
    FILE_ENCODING: Final[str] = os.getenv(
        "VOICE_FILE_ENCODING",
        DEFAULT_FILE_ENCODING
    )

    # Configuration-Defaults
    DEFAULT_CATEGORY: Final[str] = os.getenv(
        "VOICE_DEFAULT_CATEGORY",
        DEFAULT_CATEGORY
    )

    # Cosmos DB-Konfiguration
    PARTITION_KEY_PATH: Final[str] = os.getenv(
        "VOICE_COSMOS_PARTITION_KEY",
        COSMOS_PARTITION_KEY_PATH
    )

    ENABLE_CROSS_PARTITION_QUERY: Final[bool] = (
        os.getenv("VOICE_COSMOS_CROSS_PARTITION", "true").lower() == "true"
    )

    # Logging-Konfiguration
    LOG_LEVEL: Final[str] = os.getenv("VOICE_LOG_LEVEL", "INFO")

    # Performance-Konfiguration
    MAX_FILE_SIZE_MB: Final[int] = int(
        os.getenv("VOICE_MAX_FILE_SIZE_MB", "10")
    )

    QUERY_TIMEOUT_SECONDS: Final[int] = int(
        os.getenv("VOICE_QUERY_TIMEOUT", "30")
    )

# =============================================================================
# Validation Patterns
# =============================================================================

# Prompty-Validation
MIN_PROMPTY_PARTS: Final[int] = 3  # Für gültiges Front Matter
REQUIRED_FRONT_MATTER_KEYS: Final[set[str]] = {
    PROMPTY_NAME_KEY,
    PROMPTY_CATEGORY_KEY
}

# Configuration-Validation
VALID_CATEGORIES: Final[set[str]] = {
    "assistant",
    "agent",
    "tool",
    "workflow",
    "custom"
}

# =============================================================================
# Performance Konstanten
# =============================================================================

# File-Handling
MAX_FILE_SIZE_BYTES: Final[int] = VoiceConfig.MAX_FILE_SIZE_MB * 1024 * 1024
READ_CHUNK_SIZE: Final[int] = 8192

# Cosmos DB
DEFAULT_QUERY_TIMEOUT: Final[int] = VoiceConfig.QUERY_TIMEOUT_SECONDS
MAX_QUERY_RESULTS: Final[int] = 1000

# =============================================================================
# Feature Flags
# =============================================================================

class VoiceFeatureFlags:
    """Feature Flags für Voice-Modul.

    Ermöglicht das Ein-/Ausschalten von Features zur Laufzeit
    über Environment-Variablen.
    """

    ENABLE_COSMOS_DB: Final[bool] = (
        os.getenv("VOICE_ENABLE_COSMOS_DB", "true").lower() == "true"
    )

    ENABLE_PROMPTY_CACHE: Final[bool] = (
        os.getenv("VOICE_ENABLE_PROMPTY_CACHE", "false").lower() == "true"
    )

    ENABLE_FALLBACK_CONFIG: Final[bool] = (
        os.getenv("VOICE_ENABLE_FALLBACK", "true").lower() == "true"
    )

    ENABLE_STRICT_VALIDATION: Final[bool] = (
        os.getenv("VOICE_STRICT_VALIDATION", "false").lower() == "true"
    )

    ENABLE_PERFORMANCE_METRICS: Final[bool] = (
        os.getenv("VOICE_ENABLE_METRICS", "false").lower() == "true"
    )

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CONFIG_CATEGORY_KEY",
    "CONFIG_CONTENT_KEY",
    "CONFIG_DEFAULT_KEY",
    # Configuration Keys
    "CONFIG_ID_KEY",
    "CONFIG_NAME_KEY",
    "CONFIG_TOOLS_KEY",
    "COSMOS_ENABLE_CROSS_PARTITION",
    # Cosmos DB Konstanten
    "COSMOS_PARTITION_KEY_PATH",
    "COSMOS_QUERY_DEFAULT_CONFIG",
    "DEFAULT_CATEGORY",
    "DEFAULT_FILE_ENCODING",
    "DEFAULT_PROMPTY_FILENAME",
    "DEFAULT_QUERY_TIMEOUT",
    "DEFAULT_TOOLS",
    "ERROR_COSMOSDB_CONNECT",
    # Error Messages
    "ERROR_COSMOSDB_CONNECTION",
    "ERROR_COSMOSDB_QUERY",
    "ERROR_FALLBACK_CONFIG",
    "ERROR_LOAD_DEFAULT_CONFIG",
    "ERROR_PROMPTY_NOT_FOUND",
    "ERROR_PROMPTY_PARSE",
    "ERROR_PROMPTY_READ",
    "INLINE_PATH_NAME",
    # Performance
    "MAX_FILE_SIZE_BYTES",
    "MAX_QUERY_RESULTS",
    # Validation
    "MIN_PROMPTY_PARTS",
    "PROMPTY_CATEGORY_KEY",
    "PROMPTY_DIR",
    "PROMPTY_MODEL_KEY",
    "PROMPTY_NAME_KEY",
    "PROMPTY_TOOLS_KEY",
    "READ_CHUNK_SIZE",
    "REQUIRED_FRONT_MATTER_KEYS",
    "VALID_CATEGORIES",
    # File Konstanten
    "VOICE_MODULE_DIR",
    # YAML Konstanten
    "YAML_DELIMITER",
    "YAML_DELIMITER_COUNT",
    # Konfiguration
    "VoiceConfig",
    "VoiceFeatureFlags",
]
