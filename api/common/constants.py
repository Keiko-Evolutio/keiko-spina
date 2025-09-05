"""Konstanten für das API Common-Modul.

Zentralisiert alle Magic Numbers, Hard-coded Strings und Konfigurationswerte
um Code-Duplizierung zu vermeiden und Wartbarkeit zu verbessern.
"""

from __future__ import annotations

import os
from typing import Final

# ============================================================================
# PAGINATION CONSTANTS
# ============================================================================

# Standard-Pagination-Werte
DEFAULT_PAGE: Final[int] = 1
DEFAULT_PAGE_SIZE: Final[int] = int(os.getenv("API_DEFAULT_PAGE_SIZE", "20"))
MAX_PAGE_SIZE: Final[int] = int(os.getenv("API_MAX_PAGE_SIZE", "100"))
MIN_PAGE_SIZE: Final[int] = 1

# Pagination-Limits für verschiedene Endpoints
PAGINATION_LIMITS: Final[dict[str, int]] = {
    "default": MAX_PAGE_SIZE,
    "resources": 50,
    "logs": 200,
    "events": 100,
}


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# String-Längen-Limits
MIN_SERVER_NAME_LENGTH: Final[int] = 2
MAX_SERVER_NAME_LENGTH: Final[int] = int(os.getenv("API_MAX_SERVER_NAME_LENGTH", "64"))
MAX_INPUT_LENGTH: Final[int] = int(os.getenv("API_MAX_INPUT_LENGTH", "1000"))
MAX_MESSAGE_LENGTH: Final[int] = int(os.getenv("API_MAX_MESSAGE_LENGTH", "500"))

# Timeout-Werte (in Sekunden)
DEFAULT_REQUEST_TIMEOUT: Final[int] = int(os.getenv("API_REQUEST_TIMEOUT", "30"))
CACHE_EXPIRY_SECONDS: Final[int] = int(os.getenv("API_CACHE_EXPIRY", "300"))
NONCE_VALIDITY_SECONDS: Final[int] = 300  # 5 Minuten
IDEMPOTENCY_VALIDITY_SECONDS: Final[int] = 600  # 10 Minuten


# ============================================================================
# ERROR CODES
# ============================================================================

# Standard-Error-Codes
class ErrorCodes:
    """Zentrale Definition aller API-Error-Codes."""

    # Allgemeine Fehler
    INTERNAL_ERROR: Final[str] = "INTERNAL_ERROR"
    VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
    NOT_FOUND: Final[str] = "NOT_FOUND"
    CONFLICT: Final[str] = "CONFLICT"
    RATE_LIMIT_EXCEEDED: Final[str] = "RATE_LIMIT_EXCEEDED"

    # Authentifizierung/Autorisierung
    UNAUTHORIZED: Final[str] = "UNAUTHORIZED"
    FORBIDDEN: Final[str] = "FORBIDDEN"
    TOKEN_EXPIRED: Final[str] = "TOKEN_EXPIRED"

    # Request-spezifische Fehler
    BAD_REQUEST: Final[str] = "BAD_REQUEST"
    INVALID_FORMAT: Final[str] = "INVALID_FORMAT"
    MISSING_FIELD: Final[str] = "MISSING_FIELD"

    # Service-spezifische Fehler
    SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"
    TIMEOUT: Final[str] = "TIMEOUT"
    DEADLINE_EXCEEDED: Final[str] = "DEADLINE_EXCEEDED"


# ============================================================================
# HTTP STATUS CODES
# ============================================================================

# Häufig verwendete HTTP-Status-Codes als Konstanten
class HTTPStatusCodes:
    """HTTP-Status-Codes für konsistente Verwendung."""

    # Success
    OK: Final[int] = 200
    CREATED: Final[int] = 201
    ACCEPTED: Final[int] = 202
    NO_CONTENT: Final[int] = 204

    # Client Errors
    BAD_REQUEST: Final[int] = 400
    UNAUTHORIZED: Final[int] = 401
    FORBIDDEN: Final[int] = 403
    NOT_FOUND: Final[int] = 404
    CONFLICT: Final[int] = 409
    UNPROCESSABLE_ENTITY: Final[int] = 422
    TOO_MANY_REQUESTS: Final[int] = 429

    # Server Errors
    INTERNAL_SERVER_ERROR: Final[int] = 500
    BAD_GATEWAY: Final[int] = 502
    SERVICE_UNAVAILABLE: Final[int] = 503
    GATEWAY_TIMEOUT: Final[int] = 504


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Validierungs-Pattern als Konstanten
class ValidationPatterns:
    """Regex-Pattern für verschiedene Validierungen."""

    # UUID-Pattern (Version 4)
    UUID_V4: Final[str] = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

    # Correlation-ID-Pattern (timestamp-hex oder UUID)
    CORRELATION_ID: Final[str] = r"^(\d{13}-[0-9a-f]{12}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"

    # Alphanumerisch mit Bindestrichen und Unterstrichen
    ALPHANUMERIC: Final[str] = r"^[a-zA-Z0-9_-]+$"

    # Server-Name-Pattern (für MCP-Server)
    SERVER_NAME: Final[str] = r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$"

    # Email-Pattern (einfach)
    EMAIL: Final[str] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Middleware-Konfigurationswerte
class MiddlewareConfig:
    """Konfigurationskonstanten für Middleware-Komponenten."""

    # Performance-Monitoring
    ENABLE_PERFORMANCE_LOGGING: Final[bool] = os.getenv("API_ENABLE_PERF_LOGGING", "false").lower() == "true"
    PERFORMANCE_LOG_THRESHOLD_MS: Final[float] = float(os.getenv("API_PERF_THRESHOLD_MS", "100.0"))

    # Request-Limits
    MAX_REQUEST_SIZE_MB: Final[int] = int(os.getenv("API_MAX_REQUEST_SIZE_MB", "10"))
    MAX_CONCURRENT_REQUESTS: Final[int] = int(os.getenv("API_MAX_CONCURRENT_REQUESTS", "100"))

    # Retry-Konfiguration
    DEFAULT_RETRY_ATTEMPTS: Final[int] = int(os.getenv("API_DEFAULT_RETRY_ATTEMPTS", "3"))
    RETRY_BACKOFF_FACTOR: Final[float] = float(os.getenv("API_RETRY_BACKOFF_FACTOR", "2.0"))


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging-Konfiguration
class LoggingConfig:
    """Logging-Konfigurationskonstanten."""

    # Log-Level
    DEFAULT_LOG_LEVEL: Final[str] = os.getenv("API_LOG_LEVEL", "INFO")

    # Log-Format
    LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Structured Logging Fields
    CORRELATION_ID_FIELD: Final[str] = "correlation_id"
    TRACE_ID_FIELD: Final[str] = "trace_id"
    USER_ID_FIELD: Final[str] = "user_id"
    TENANT_ID_FIELD: Final[str] = "tenant_id"


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Cache-Konfiguration
class CacheConfig:
    """Cache-Konfigurationskonstanten."""

    # Cache-Keys-Prefixes
    WEBHOOK_NONCE_PREFIX: Final[str] = "kei:webhook:nonce:"
    WEBHOOK_IDEMPOTENCY_PREFIX: Final[str] = "kei:webhook:idem:"
    API_RATE_LIMIT_PREFIX: Final[str] = "kei:api:rate_limit:"

    # Cache-TTL-Werte (in Sekunden)
    DEFAULT_TTL: Final[int] = CACHE_EXPIRY_SECONDS
    SHORT_TTL: Final[int] = 60  # 1 Minute
    LONG_TTL: Final[int] = 3600  # 1 Stunde


# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Feature-Flags für experimentelle Features
class FeatureFlags:
    """Feature-Flags für bedingte Funktionalität."""

    ENABLE_DETAILED_ERROR_RESPONSES: Final[bool] = os.getenv("API_DETAILED_ERRORS", "false").lower() == "true"
    ENABLE_REQUEST_TRACING: Final[bool] = os.getenv("API_REQUEST_TRACING", "false").lower() == "true"
    ENABLE_METRICS_COLLECTION: Final[bool] = os.getenv("API_METRICS_COLLECTION", "true").lower() == "true"
    ENABLE_SECURITY_HEADERS: Final[bool] = os.getenv("API_SECURITY_HEADERS", "true").lower() == "true"


# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

# Umgebungs-Detection
ENVIRONMENT: Final[str] = os.getenv("ENVIRONMENT", "development").lower()
IS_DEVELOPMENT: Final[bool] = ENVIRONMENT == "development"
IS_PRODUCTION: Final[bool] = ENVIRONMENT == "production"
IS_TESTING: Final[bool] = ENVIRONMENT == "testing"

# Debug-Modus
DEBUG_MODE: Final[bool] = (os.getenv("DEBUG", "false").lower() == "true") or IS_DEVELOPMENT
