"""Konstanten für API v1 Module.

Zentrale Definition aller Konstanten, Magic Numbers und Hard-coded Strings
für das backend/api/v1/ Modul.
"""

from __future__ import annotations

from typing import Final

# =====================================================================
# ID-Generierung Konstanten
# =====================================================================

# Länge für Konfigurations-IDs (UUID-Hex-Teil)
CONFIG_ID_LENGTH: Final[int] = 8

# Präfix für Konfigurations-IDs
CONFIG_ID_PREFIX: Final[str] = "cfg"


# =====================================================================
# Error-Codes für HTTP Exceptions
# =====================================================================

# Resource Not Found Errors
ERROR_CODE_NOT_FOUND: Final[str] = "NOT_FOUND"
ERROR_MESSAGE_CONFIG_NOT_FOUND: Final[str] = "Konfiguration nicht gefunden"

# Conflict Errors
ERROR_CODE_NAME_EXISTS: Final[str] = "NAME_EXISTS"
ERROR_MESSAGE_NAME_EXISTS: Final[str] = "Name bereits vorhanden"

# Validation Errors
ERROR_CODE_VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
ERROR_CODE_INVALID_INPUT: Final[str] = "INVALID_INPUT"


# =====================================================================
# HTTP Status Messages
# =====================================================================

# Success Messages
SUCCESS_MESSAGE_CONFIG_CREATED: Final[str] = "Konfiguration erfolgreich erstellt"
SUCCESS_MESSAGE_CONFIG_UPDATED: Final[str] = "Konfiguration erfolgreich aktualisiert"
SUCCESS_MESSAGE_CONFIG_DELETED: Final[str] = "Konfiguration erfolgreich gelöscht"

# Info Messages
INFO_MESSAGE_CONFIG_RETRIEVED: Final[str] = "Konfiguration abgerufen"


# =====================================================================
# Logging Messages
# =====================================================================

# Configuration Operations
LOG_CONFIG_CREATED: Final[str] = "Konfiguration '{name}' erstellt mit ID {config_id}"
LOG_CONFIG_RETRIEVED: Final[str] = "Konfiguration '{name}' abgerufen (ID: {config_id})"
LOG_CONFIG_DELETED: Final[str] = "Konfiguration '{name}' gelöscht (ID: {config_id})"
LOG_CONFIG_NAME_CONFLICT: Final[str] = "Name-Konflikt bei Konfigurationserstellung: '{name}'"

# Error Logging
LOG_CONFIG_NOT_FOUND: Final[str] = "Konfiguration nicht gefunden: {config_id}"
LOG_VALIDATION_ERROR: Final[str] = "Validierungsfehler bei Konfiguration: {error}"


# =====================================================================
# Router Konfiguration
# =====================================================================

# Router Prefix und Tags
ROUTER_PREFIX: Final[str] = "/api/v1/configurations"
ROUTER_TAGS: Final[list[str]] = ["Agent-Konfigurationen"]

# API Endpoint Paths
ENDPOINT_CREATE_CONFIG: Final[str] = "/"
ENDPOINT_GET_CONFIG: Final[str] = "/{configuration_id}"
ENDPOINT_DELETE_CONFIG: Final[str] = "/{configuration_id}"


# =====================================================================
# Validierung Konstanten
# =====================================================================

# Minimale/Maximale Längen für Felder
MIN_NAME_LENGTH: Final[int] = 1
MAX_NAME_LENGTH: Final[int] = 100

MIN_SYSTEM_MESSAGE_LENGTH: Final[int] = 10
MAX_SYSTEM_MESSAGE_LENGTH: Final[int] = 5000

# Maximale Anzahl Tools pro Konfiguration
MAX_TOOLS_PER_CONFIG: Final[int] = 50


# =====================================================================
# Default-Werte
# =====================================================================

# Standard-Kategorie für neue Konfigurationen
DEFAULT_CATEGORY: Final[str] = "general"

# Standard-Werte für Voice Settings
DEFAULT_VOICE_SPEED: Final[float] = 1.0
DEFAULT_VOICE_PITCH: Final[float] = 0.0
DEFAULT_VOICE_VOLUME: Final[float] = 1.0


# =====================================================================
# Storage Konfiguration
# =====================================================================

# Name für globale Storage-Instanz
STORAGE_NAME: Final[str] = "configurations_v1"

# TTL für Cache-Einträge (in Sekunden)
CACHE_TTL_SECONDS: Final[int] = 3600  # 1 Stunde


# =====================================================================
# Metadaten Konstanten
# =====================================================================

# Felder für Zeitstempel
FIELD_CREATED_AT: Final[str] = "created_at"
FIELD_UPDATED_AT: Final[str] = "updated_at"

# Felder für Konfiguration
FIELD_ID: Final[str] = "id"
FIELD_NAME: Final[str] = "name"
FIELD_CATEGORY: Final[str] = "category"
FIELD_SYSTEM_MESSAGE: Final[str] = "system_message"
FIELD_TOOLS: Final[str] = "tools"
FIELD_VOICE_SETTINGS: Final[str] = "voice_settings"
FIELD_IS_DEFAULT: Final[str] = "is_default"


# =====================================================================
# Performance Konstanten
# =====================================================================

# Maximale Anzahl Konfigurationen im Memory Store
MAX_CONFIGURATIONS_IN_MEMORY: Final[int] = 1000

# Batch-Größe für Operationen
BATCH_SIZE: Final[int] = 100


# =====================================================================
# Security Konstanten
# =====================================================================

# Rate Limiting
MAX_REQUESTS_PER_MINUTE: Final[int] = 60
MAX_REQUESTS_PER_HOUR: Final[int] = 1000

# Input Sanitization
ALLOWED_NAME_CHARS: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "


# =====================================================================
# API Documentation Konstanten
# =====================================================================

# OpenAPI Tags und Beschreibungen
API_TAG_CONFIGURATIONS: Final[str] = "Agent-Konfigurationen"
API_DESCRIPTION_CONFIGURATIONS: Final[str] = "CRUD-Operationen für Agent-Konfigurationen"

# Response Descriptions
RESPONSE_DESC_CONFIG_CREATED: Final[str] = "Konfiguration erfolgreich erstellt"
RESPONSE_DESC_CONFIG_RETRIEVED: Final[str] = "Konfiguration erfolgreich abgerufen"
RESPONSE_DESC_CONFIG_DELETED: Final[str] = "Konfiguration erfolgreich gelöscht"
RESPONSE_DESC_CONFIG_NOT_FOUND: Final[str] = "Konfiguration nicht gefunden"
RESPONSE_DESC_NAME_CONFLICT: Final[str] = "Name bereits vorhanden"
RESPONSE_DESC_VALIDATION_ERROR: Final[str] = "Validierungsfehler in der Anfrage"


# =====================================================================
# Testing Konstanten
# =====================================================================

# Test-Daten
TEST_CONFIG_NAME: Final[str] = "Test Konfiguration"
TEST_SYSTEM_MESSAGE: Final[str] = "Du bist ein hilfreicher Test-Assistent"
TEST_CATEGORY: Final[str] = "test"

# Test-IDs
TEST_CONFIG_ID: Final[str] = "cfg_test123"
TEST_NONEXISTENT_ID: Final[str] = "cfg_nonexistent"


# =====================================================================
# Monitoring und Observability
# =====================================================================

# Metriken-Namen
METRIC_CONFIGS_CREATED: Final[str] = "configurations_created_total"
METRIC_CONFIGS_RETRIEVED: Final[str] = "configurations_retrieved_total"
METRIC_CONFIGS_DELETED: Final[str] = "configurations_deleted_total"
METRIC_CONFIG_ERRORS: Final[str] = "configuration_errors_total"

# Trace-Namen
TRACE_CREATE_CONFIG: Final[str] = "create_configuration"
TRACE_GET_CONFIG: Final[str] = "get_configuration"
TRACE_DELETE_CONFIG: Final[str] = "delete_configuration"


# =====================================================================
# Feature Flags
# =====================================================================

# Experimentelle Features
ENABLE_CONFIG_VERSIONING: Final[bool] = False
ENABLE_CONFIG_TEMPLATES: Final[bool] = False
ENABLE_CONFIG_VALIDATION_CACHE: Final[bool] = True
ENABLE_ASYNC_CONFIG_OPERATIONS: Final[bool] = False


# =====================================================================
# Utility Funktionen für Konstanten
# =====================================================================

def get_error_detail(error_code: str, message: str) -> dict[str, str]:
    """Erstellt standardisiertes Error-Detail-Dictionary.

    Args:
        error_code: Standardisierter Error-Code
        message: Benutzerfreundliche Fehlermeldung

    Returns:
        Dictionary mit error_code und message
    """
    return {
        "error_code": error_code,
        "message": message
    }


def get_success_detail(message: str, **additional_data: str) -> dict[str, str]:
    """Erstellt standardisiertes Success-Detail-Dictionary.

    Args:
        message: Erfolgsmeldung
        **additional_data: Zusätzliche Daten für die Response

    Returns:
        Dictionary mit message und zusätzlichen Daten
    """
    return {
        "message": message,
        **additional_data
    }
