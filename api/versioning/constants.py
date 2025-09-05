"""Konstanten für das API-Versionierungsmodul.

Zentrale Definition aller Magic Numbers, Hard-coded Strings und Konfigurationswerte
für das Versioning-System.
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# SERVICE IDENTIFIKATION
# ============================================================================

SERVICE_NAME: Final[str] = "keiko-api"
API_VERSION_V1: Final[str] = "v1"
API_VERSION_V2: Final[str] = "v2"
API_VERSION_NUMBER_V2: Final[str] = "2.0.0"

# ============================================================================
# HTTP STATUS CODES
# ============================================================================

HTTP_STATUS_OK: Final[int] = 200
HTTP_STATUS_ACCEPTED: Final[int] = 202
HTTP_STATUS_NOT_FOUND: Final[int] = 404

# ============================================================================
# HEALTH STATUS WERTE
# ============================================================================

HEALTH_STATUS_HEALTHY: Final[str] = "healthy"
HEALTH_STATUS_DEGRADED: Final[str] = "degraded"
HEALTH_STATUS_UNHEALTHY: Final[str] = "unhealthy"
HEALTH_STATUS_UNAVAILABLE: Final[str] = "unavailable"

# ============================================================================
# COMPONENT NAMEN
# ============================================================================

COMPONENT_AGENT_SYSTEM: Final[str] = "agent_system"
COMPONENT_SYSTEM: Final[str] = "system"

# ============================================================================
# WEBHOOK STATUS
# ============================================================================

WEBHOOK_STATUS_ACCEPTED: Final[str] = "accepted"

# ============================================================================
# EXECUTION STATUS
# ============================================================================

EXECUTION_STATUS_SUCCESS: Final[str] = "success"
EXECUTION_STATUS_ERROR: Final[str] = "error"

# ============================================================================
# ERROR CODES
# ============================================================================

ERROR_CODE_NOT_FOUND: Final[str] = "NOT_FOUND"
ERROR_CODE_VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
ERROR_CODE_HTTP_ERROR: Final[str] = "HTTP_ERROR"
ERROR_CODE_EXEC_ERROR: Final[str] = "EXEC_ERROR"
ERROR_CODE_UNEXPECTED_ERROR: Final[str] = "UNEXPECTED_ERROR"

# ============================================================================
# CONTENT TYPES
# ============================================================================

CONTENT_TYPE_JSON: Final[str] = "application/json"
CONTENT_TYPE_KEIKO_V2: Final[str] = "application/vnd.keiko.v2+json"

# ============================================================================
# HEADER NAMEN
# ============================================================================

HEADER_API_VERSION: Final[str] = "X-API-Version"
HEADER_API_VERSION_LOWER: Final[str] = "x-api-version"
HEADER_TENANT_ID: Final[str] = "X-Tenant-Id"
HEADER_ACCEPT: Final[str] = "Accept"
HEADER_WARNING: Final[str] = "Warning"

# ============================================================================
# WARNING MESSAGES
# ============================================================================

WARNING_V1_DEPRECATED: Final[str] = '299 - "API v1 is deprecated; migrate to v2"'

# ============================================================================
# URL PREFIXES
# ============================================================================

URL_PREFIX_V1: Final[str] = "/api/v1"
URL_PREFIX_V2: Final[str] = "/api/v2"
URL_PREFIX_V2_FUNCTIONS: Final[str] = "/api/v2/functions"

# ============================================================================
# PARAMETER TYPES
# ============================================================================

PARAM_TYPE_STRING: Final[str] = "string"
PARAM_TYPE_NUMBER: Final[str] = "number"
PARAM_TYPE_INTEGER: Final[str] = "integer"
PARAM_TYPE_BOOLEAN: Final[str] = "boolean"
PARAM_TYPE_OBJECT: Final[str] = "object"
PARAM_TYPE_ARRAY: Final[str] = "array"
PARAM_TYPE_ANY: Final[str] = "any"

# ============================================================================
# DEFAULT WERTE
# ============================================================================

DEFAULT_CATEGORY: Final[str] = "general"
DEFAULT_FUNCTION_NAME: Final[str] = "func"
DEFAULT_AGENT_NAME: Final[str] = "agent"
DEFAULT_PARAM_NAME: Final[str] = "param"

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

PROMETHEUS_COUNTER_NAME: Final[str] = "keiko_api_version_requests_total"
PROMETHEUS_COUNTER_DESCRIPTION: Final[str] = "Anzahl API Requests je Version"

# ============================================================================
# VALIDATION CONSTRAINTS
# ============================================================================

MAX_FUNCTION_NAME_LENGTH: Final[int] = 64
MAX_DESCRIPTION_LENGTH: Final[int] = 512
MIN_NAME_LENGTH: Final[int] = 1

# ============================================================================
# EXECUTION PREFIXES
# ============================================================================

EXECUTION_ID_PREFIX: Final[str] = "exec_"

__all__ = [
    "API_VERSION_NUMBER_V2",
    "API_VERSION_V1",
    "API_VERSION_V2",
    # Components
    "COMPONENT_AGENT_SYSTEM",
    "COMPONENT_SYSTEM",
    # Content Types
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_KEIKO_V2",
    "DEFAULT_AGENT_NAME",
    # Defaults
    "DEFAULT_CATEGORY",
    "DEFAULT_FUNCTION_NAME",
    "DEFAULT_PARAM_NAME",
    "ERROR_CODE_EXEC_ERROR",
    "ERROR_CODE_HTTP_ERROR",
    # Error Codes
    "ERROR_CODE_NOT_FOUND",
    "ERROR_CODE_UNEXPECTED_ERROR",
    "ERROR_CODE_VALIDATION_ERROR",
    # Execution
    "EXECUTION_ID_PREFIX",
    "EXECUTION_STATUS_ERROR",
    # Execution
    "EXECUTION_STATUS_SUCCESS",
    "HEADER_ACCEPT",
    # Headers
    "HEADER_API_VERSION",
    "HEADER_API_VERSION_LOWER",
    "HEADER_TENANT_ID",
    "HEADER_WARNING",
    "HEALTH_STATUS_DEGRADED",
    # Health Status
    "HEALTH_STATUS_HEALTHY",
    "HEALTH_STATUS_UNAVAILABLE",
    "HEALTH_STATUS_UNHEALTHY",
    "HTTP_STATUS_ACCEPTED",
    "HTTP_STATUS_NOT_FOUND",
    # HTTP Status
    "HTTP_STATUS_OK",
    "MAX_DESCRIPTION_LENGTH",
    # Validation
    "MAX_FUNCTION_NAME_LENGTH",
    "MIN_NAME_LENGTH",
    "PARAM_TYPE_ANY",
    "PARAM_TYPE_ARRAY",
    "PARAM_TYPE_BOOLEAN",
    "PARAM_TYPE_INTEGER",
    "PARAM_TYPE_NUMBER",
    "PARAM_TYPE_OBJECT",
    # Parameter Types
    "PARAM_TYPE_STRING",
    "PROMETHEUS_COUNTER_DESCRIPTION",
    # Prometheus
    "PROMETHEUS_COUNTER_NAME",
    # Service
    "SERVICE_NAME",
    # URLs
    "URL_PREFIX_V1",
    "URL_PREFIX_V2",
    "URL_PREFIX_V2_FUNCTIONS",
    # Warnings
    "WARNING_V1_DEPRECATED",
    # Webhook
    "WEBHOOK_STATUS_ACCEPTED",
]
