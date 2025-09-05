"""Zentrale Konstanten für das Core-Modul.

Dieses Modul definiert alle Magic Numbers, HTTP-Status-Codes,
Severity-Level und andere Konstanten um Code-Duplikation zu vermeiden
und Wartbarkeit zu verbessern.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


# HTTP-Status-Code-Konstanten
class HTTPStatus:
    """HTTP-Status-Code-Konstanten für Error-Handler."""

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
    METHOD_NOT_ALLOWED: Final[int] = 405
    CONFLICT: Final[int] = 409
    UNPROCESSABLE_ENTITY: Final[int] = 422
    TOO_MANY_REQUESTS: Final[int] = 429

    # Server Errors
    INTERNAL_SERVER_ERROR: Final[int] = 500
    NOT_IMPLEMENTED: Final[int] = 501
    BAD_GATEWAY: Final[int] = 502
    SERVICE_UNAVAILABLE: Final[int] = 503
    GATEWAY_TIMEOUT: Final[int] = 504


class SeverityLevel(Enum):
    """Severity-Level für Exceptions und Logging."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCode:
    """Zentrale Error-Code-Konstanten."""

    # Base Errors
    INTERNAL_ERROR: Final[str] = "INTERNAL_ERROR"
    UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"

    # Domain-specific Errors
    AGENT_ERROR: Final[str] = "AGENT_ERROR"
    AZURE_ERROR: Final[str] = "AZURE_ERROR"
    NETWORK_ERROR: Final[str] = "NETWORK_ERROR"
    AUTH_ERROR: Final[str] = "AUTH_ERROR"
    VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
    NOT_FOUND: Final[str] = "NOT_FOUND"
    BAD_REQUEST: Final[str] = "BAD_REQUEST"

    # Operational Errors
    RATE_LIMIT_EXCEEDED: Final[str] = "RATE_LIMIT_EXCEEDED"
    TIMEOUT: Final[str] = "TIMEOUT"
    DEADLINE_EXCEEDED: Final[str] = "DEADLINE_EXCEEDED"
    CONFLICT: Final[str] = "CONFLICT"

    # Policy and Business Logic Errors
    POLICY_VIOLATION: Final[str] = "POLICY_VIOLATION"
    BUDGET_EXCEEDED: Final[str] = "BUDGET_EXCEEDED"

    # Service and Dependency Errors
    SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"
    DEPENDENCY_ERROR: Final[str] = "DEPENDENCY_ERROR"


class RetryConfig:
    """Konfiguration für Retry-Strategien."""

    # Default Retry-Timeouts (in Sekunden)
    DEFAULT_RETRY_AFTER: Final[int] = 5
    RATE_LIMIT_RETRY_AFTER: Final[int] = 60
    SERVICE_UNAVAILABLE_RETRY_AFTER: Final[int] = 30
    TIMEOUT_RETRY_AFTER: Final[int] = 10

    # Maximale Retry-Versuche
    MAX_RETRIES: Final[int] = 3

    # Exponential Backoff-Faktoren
    BACKOFF_FACTOR: Final[float] = 2.0
    MAX_BACKOFF: Final[int] = 300  # 5 Minuten


class LoggingConfig:
    """Konfiguration für Logging-System."""

    # Logger-Namen
    AUDIT_LOGGER_NAME: Final[str] = "kei.audit"
    PERFORMANCE_LOGGER_NAME: Final[str] = "kei.perf"
    SECURITY_LOGGER_NAME: Final[str] = "kei.sec"
    ERROR_LOGGER_NAME: Final[str] = "kei.error"

    # JSON-Serialisierung
    JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
    JSON_ENSURE_ASCII: Final[bool] = False


# HTTP-Status-Code-Mapping für Error-Codes
ERROR_CODE_TO_HTTP_STATUS: Final[dict[str, int]] = {
    # Validation and Input Errors
    ErrorCode.VALIDATION_ERROR: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.BAD_REQUEST: HTTPStatus.BAD_REQUEST,

    # Authentication and Authorization
    ErrorCode.AUTH_ERROR: HTTPStatus.UNAUTHORIZED,

    # Resource Errors
    ErrorCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorCode.CONFLICT: HTTPStatus.CONFLICT,

    # Rate Limiting
    ErrorCode.RATE_LIMIT_EXCEEDED: HTTPStatus.TOO_MANY_REQUESTS,

    # Timeout and Deadline Errors
    ErrorCode.TIMEOUT: HTTPStatus.GATEWAY_TIMEOUT,
    ErrorCode.DEADLINE_EXCEEDED: HTTPStatus.GATEWAY_TIMEOUT,

    # Service and Dependency Errors
    ErrorCode.SERVICE_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.DEPENDENCY_ERROR: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.AZURE_ERROR: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.NETWORK_ERROR: HTTPStatus.SERVICE_UNAVAILABLE,

    # Policy and Business Logic
    ErrorCode.POLICY_VIOLATION: HTTPStatus.FORBIDDEN,
    ErrorCode.BUDGET_EXCEEDED: HTTPStatus.FORBIDDEN,

    # Agent-specific Errors
    ErrorCode.AGENT_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,

    # Fallback
    ErrorCode.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.UNKNOWN_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}


# Retryable Error-Codes
RETRYABLE_ERROR_CODES: Final[set[str]] = {
    ErrorCode.RATE_LIMIT_EXCEEDED,
    ErrorCode.TIMEOUT,
    ErrorCode.DEADLINE_EXCEEDED,
    ErrorCode.SERVICE_UNAVAILABLE,
    ErrorCode.DEPENDENCY_ERROR,
    ErrorCode.NETWORK_ERROR,
    ErrorCode.AZURE_ERROR,
}


# Severity-Level-Mapping für Logging
SEVERITY_TO_LOG_LEVEL: Final[dict[str, str]] = {
    SeverityLevel.LOW.value: "INFO",
    SeverityLevel.MEDIUM.value: "WARNING",
    SeverityLevel.HIGH.value: "ERROR",
    SeverityLevel.CRITICAL.value: "CRITICAL",
}


# Default-Nachrichten für Error-Codes
DEFAULT_ERROR_MESSAGES: Final[dict[str, str]] = {
    ErrorCode.AGENT_ERROR: "Agent-Fehler",
    ErrorCode.AZURE_ERROR: "Azure-Integrationsfehler",
    ErrorCode.NETWORK_ERROR: "Netzwerkfehler",
    ErrorCode.AUTH_ERROR: "Authentifizierungsfehler",
    ErrorCode.VALIDATION_ERROR: "Validierung fehlgeschlagen",
    ErrorCode.NOT_FOUND: "Ressource nicht gefunden",
    ErrorCode.BAD_REQUEST: "Ungültige Anfrage",
    ErrorCode.RATE_LIMIT_EXCEEDED: "Ratenlimit überschritten",
    ErrorCode.TIMEOUT: "Timeout überschritten",
    ErrorCode.DEADLINE_EXCEEDED: "Deadline überschritten",
    ErrorCode.CONFLICT: "Konflikt",
    ErrorCode.POLICY_VIOLATION: "Policy-Verstoß",
    ErrorCode.BUDGET_EXCEEDED: "Budget überschritten",
    ErrorCode.SERVICE_UNAVAILABLE: "Service nicht verfügbar",
    ErrorCode.DEPENDENCY_ERROR: "Abhängigkeit fehlgeschlagen",
    ErrorCode.INTERNAL_ERROR: "Interner Serverfehler",
    ErrorCode.UNKNOWN_ERROR: "Unbekannter Fehler",
}


# Retry-Timeout-Mapping für Error-Codes
ERROR_CODE_TO_RETRY_TIMEOUT: Final[dict[str, int]] = {
    ErrorCode.RATE_LIMIT_EXCEEDED: RetryConfig.RATE_LIMIT_RETRY_AFTER,
    ErrorCode.SERVICE_UNAVAILABLE: RetryConfig.SERVICE_UNAVAILABLE_RETRY_AFTER,
    ErrorCode.TIMEOUT: RetryConfig.TIMEOUT_RETRY_AFTER,
    ErrorCode.DEADLINE_EXCEEDED: RetryConfig.TIMEOUT_RETRY_AFTER,
    ErrorCode.DEPENDENCY_ERROR: RetryConfig.DEFAULT_RETRY_AFTER,
    ErrorCode.NETWORK_ERROR: RetryConfig.DEFAULT_RETRY_AFTER,
    ErrorCode.AZURE_ERROR: RetryConfig.DEFAULT_RETRY_AFTER,
}


# OpenTelemetry-Attribute-Namen
class OTelAttributes:
    """OpenTelemetry-Attribute-Namen für Tracing."""

    ERROR_CODE: Final[str] = "kei.error.code"
    ERROR_SEVERITY: Final[str] = "kei.error.severity"
    HTTP_STATUS_CODE: Final[str] = "kei.http.status_code"
    TRACE_ID: Final[str] = "kei.trace.id"
    COMPONENT: Final[str] = "kei.component"
    OPERATION: Final[str] = "kei.operation"


__all__ = [
    "DEFAULT_ERROR_MESSAGES",
    # Mappings
    "ERROR_CODE_TO_HTTP_STATUS",
    "ERROR_CODE_TO_RETRY_TIMEOUT",
    "RETRYABLE_ERROR_CODES",
    "SEVERITY_TO_LOG_LEVEL",
    "ErrorCode",
    # Classes
    "HTTPStatus",
    "LoggingConfig",
    "OTelAttributes",
    "RetryConfig",
    "SeverityLevel",
]
