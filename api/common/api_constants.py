"""Minimal API Constants

Nur die notwendigen Konstanten für Router-Factory und andere Komponenten.
Ersetzt die vorherige umfangreiche api_constants.py.
"""

from typing import Final


class APIPaths:
    """API-Pfad-Konstanten."""

    # API Prefixes
    API_V1_PREFIX: Final[str] = "/api/v1"

    # Core Paths
    HEALTH: Final[str] = "/health"
    CHAT: Final[str] = "/chat"
    FUNCTIONS: Final[str] = "/functions"
    AGENTS: Final[str] = "/agents"
    CAPABILITIES: Final[str] = "/capabilities"


class HTTPStatus:
    """HTTP-Status-Code-Konstanten."""

    # Success
    OK: Final[int] = 200
    CREATED: Final[int] = 201
    ACCEPTED: Final[int] = 202

    # Client Errors
    BAD_REQUEST: Final[int] = 400
    UNAUTHORIZED: Final[int] = 401
    FORBIDDEN: Final[int] = 403
    NOT_FOUND: Final[int] = 404
    UNPROCESSABLE_ENTITY: Final[int] = 422
    TOO_MANY_REQUESTS: Final[int] = 429

    # Server Errors
    INTERNAL_SERVER_ERROR: Final[int] = 500
    SERVICE_UNAVAILABLE: Final[int] = 503


class ResponseConstants:
    """Response-Format-Konstanten."""

    JSON_CONTENT_TYPE: Final[str] = "application/json"


class ConfigDefaults:
    """Konfiguration-Standard-Werte."""

    DEFAULT_TIMEOUT: Final[int] = 30
    DEFAULT_RETRIES: Final[int] = 3
    DEFAULT_PAGE_SIZE: Final[int] = 50
    SERVICE_NAME: Final[str] = "keiko-backend"


class HeaderNames:
    """HTTP-Header-Namen."""

    AUTHORIZATION: Final[str] = "Authorization"
    CONTENT_TYPE: Final[str] = "Content-Type"
    USER_AGENT: Final[str] = "User-Agent"
    X_REQUEST_ID: Final[str] = "X-Request-ID"
    X_CORRELATION_ID: Final[str] = "X-Correlation-ID"
    X_TRACE_ID: Final[str] = "X-Trace-ID"


# Export für Kompatibilität
__all__ = [
    "APIPaths",
    "ConfigDefaults",
    "HTTPStatus",
    "HeaderNames",
    "ResponseConstants"
]
