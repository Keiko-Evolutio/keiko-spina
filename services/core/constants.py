"""Zentrale Konstanten für Services.

Konsolidiert alle Magic Numbers und Hard-coded Strings aus dem Services-Modul.
"""

from __future__ import annotations

# =====================================================================
# Circuit Breaker Konstanten
# =====================================================================

# Standard Circuit Breaker Konfiguration
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_OPEN_TIMEOUT_SECONDS = 10.0
DEFAULT_HALF_OPEN_MAX_CONCURRENT = 1
DEFAULT_RECOVERY_BACKOFF_BASE = 1.5
DEFAULT_RECOVERY_BACKOFF_MAX_SECONDS = 30.0

# Service-spezifische Circuit Breaker Konfigurationen
KEI_RPC_CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": DEFAULT_FAILURE_THRESHOLD,
    "open_timeout_seconds": DEFAULT_OPEN_TIMEOUT_SECONDS,
}

KEI_GRPC_CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": DEFAULT_FAILURE_THRESHOLD,
    "open_timeout_seconds": DEFAULT_OPEN_TIMEOUT_SECONDS,
}

MCP_CLIENT_CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": DEFAULT_FAILURE_THRESHOLD,
    "open_timeout_seconds": 60.0,
    "half_open_max_concurrent": 3,
}

# =====================================================================
# HTTP Client Konstanten
# =====================================================================

# Standard Timeout-Werte
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_CONNECTION_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 30.0

# HTTP/2 Konfiguration
DEFAULT_HTTP2_MAX_CONNECTIONS = 100
DEFAULT_HTTP2_MAX_KEEPALIVE = 20

# Retry-Konfiguration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 2.0
DEFAULT_RETRY_MAX_DELAY = 60.0

# =====================================================================
# Service Status Konstanten
# =====================================================================

# Service-Status-Strings
SERVICE_STATUS_AVAILABLE = "available"
SERVICE_STATUS_UNAVAILABLE = "unavailable"
SERVICE_STATUS_UNREACHABLE = "unreachable"
SERVICE_STATUS_ERROR = "error"
SERVICE_STATUS_INITIALIZING = "initializing"
SERVICE_STATUS_SHUTTING_DOWN = "shutting_down"

# Health Check Konstanten
HEALTH_CHECK_INTERVAL_SECONDS = 30.0
HEALTH_CHECK_TIMEOUT_SECONDS = 5.0
MAX_HEALTH_CHECK_FAILURES = 3

# =====================================================================
# HTTP Status Code Konstanten
# =====================================================================

# Erfolgreiche Responses
HTTP_STATUS_OK = 200
HTTP_STATUS_CREATED = 201
HTTP_STATUS_ACCEPTED = 202
HTTP_STATUS_NO_CONTENT = 204

# Client Error Responses
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_CONFLICT = 409
HTTP_STATUS_UNPROCESSABLE_ENTITY = 422
HTTP_STATUS_TOO_MANY_REQUESTS = 429

# Server Error Responses
HTTP_STATUS_INTERNAL_SERVER_ERROR = 500
HTTP_STATUS_BAD_GATEWAY = 502
HTTP_STATUS_SERVICE_UNAVAILABLE = 503
HTTP_STATUS_GATEWAY_TIMEOUT = 504

# =====================================================================
# Heartbeat Service Konstanten
# =====================================================================

DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_HEARTBEAT_TIMEOUT = 5.0
DEFAULT_MAX_HEARTBEAT_FAILURES = 3

# =====================================================================
# Domain Revalidation Konstanten
# =====================================================================

DEFAULT_REVALIDATION_INTERVAL_HOURS = 24
FORCE_REVALIDATION_INTERVAL = 0  # Für sofortige Revalidierung

# =====================================================================
# A2A Service Konstanten
# =====================================================================

DEFAULT_A2A_VERSION = 1
DEFAULT_A2A_TIMEOUT_SECONDS = 10.0
A2A_REQUEST_TYPE = "a2a_request"
A2A_INVALID_REPLY_ERROR = "invalid_reply"
A2A_MESSAGE_VALIDATION_ERROR = "A2A Nachricht ungültig"

# =====================================================================
# User Agent Strings
# =====================================================================

KEIKO_WEBHOOK_USER_AGENT = "Keiko-Webhook/2"
KEI_MCP_CLIENT_USER_AGENT = "KEI-MCP-Client/1.0"
KEI_RPC_CLIENT_USER_AGENT = "KEI-RPC-Client/1.0"

# =====================================================================
# Content Types
# =====================================================================

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_FORM_URLENCODED = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTIPART = "multipart/form-data"

# =====================================================================
# Environment Konstanten
# =====================================================================

ENVIRONMENT_DEVELOPMENT = "development"
ENVIRONMENT_PRODUCTION = "production"
ENVIRONMENT_TESTING = "testing"
