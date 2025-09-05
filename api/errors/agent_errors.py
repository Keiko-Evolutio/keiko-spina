"""Konsolidierte Error-Codes und HTTP-Status-Definitionen für die gesamte Keiko-Codebase.

Eliminiert Code-Duplikation durch zentrale Definition aller Error-Codes,
HTTP-Status-Codes und Error-Messages mit einheitlicher Hierarchie.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import Final

# ============================================================================
# HTTP-STATUS-CODES (Konsolidiert aus constants.py)
# ============================================================================

class HTTPStatusCodes:
    """Zentrale HTTP-Status-Code-Definitionen."""

    # 2xx Success
    OK: Final[int] = HTTPStatus.OK.value
    CREATED: Final[int] = HTTPStatus.CREATED.value
    ACCEPTED: Final[int] = HTTPStatus.ACCEPTED.value
    NO_CONTENT: Final[int] = HTTPStatus.NO_CONTENT.value

    # 4xx Client Errors
    BAD_REQUEST: Final[int] = HTTPStatus.BAD_REQUEST.value
    UNAUTHORIZED: Final[int] = HTTPStatus.UNAUTHORIZED.value
    FORBIDDEN: Final[int] = HTTPStatus.FORBIDDEN.value
    NOT_FOUND: Final[int] = HTTPStatus.NOT_FOUND.value
    CONFLICT: Final[int] = HTTPStatus.CONFLICT.value
    GONE: Final[int] = HTTPStatus.GONE.value
    UNPROCESSABLE_ENTITY: Final[int] = HTTPStatus.UNPROCESSABLE_ENTITY.value
    TOO_MANY_REQUESTS: Final[int] = HTTPStatus.TOO_MANY_REQUESTS.value

    # 5xx Server Errors
    INTERNAL_SERVER_ERROR: Final[int] = HTTPStatus.INTERNAL_SERVER_ERROR.value
    BAD_GATEWAY: Final[int] = HTTPStatus.BAD_GATEWAY.value
    SERVICE_UNAVAILABLE: Final[int] = HTTPStatus.SERVICE_UNAVAILABLE.value
    GATEWAY_TIMEOUT: Final[int] = HTTPStatus.GATEWAY_TIMEOUT.value


# ============================================================================
# ERROR-CODES (Konsolidiert aus allen Implementierungen)
# ============================================================================

class ErrorCodes:
    """Zentrale Error-Code-Definitionen für die gesamte Keiko-Codebase."""

    # Allgemeine System-Fehler
    INTERNAL_ERROR: Final[str] = "INTERNAL_ERROR"
    UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"

    # Request-Validierung
    VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
    BAD_REQUEST: Final[str] = "BAD_REQUEST"
    INVALID_FORMAT: Final[str] = "INVALID_FORMAT"
    MISSING_FIELD: Final[str] = "MISSING_FIELD"

    # Ressourcen-Fehler
    NOT_FOUND: Final[str] = "NOT_FOUND"
    CONFLICT: Final[str] = "CONFLICT"
    GONE: Final[str] = "GONE"

    # Authentifizierung/Autorisierung
    UNAUTHORIZED: Final[str] = "UNAUTHORIZED"
    FORBIDDEN: Final[str] = "FORBIDDEN"
    TOKEN_EXPIRED: Final[str] = "TOKEN_EXPIRED"

    # Rate-Limiting und Kapazität
    RATE_LIMIT_EXCEEDED: Final[str] = "RATE_LIMIT_EXCEEDED"
    BUDGET_EXCEEDED: Final[str] = "BUDGET_EXCEEDED"

    # Timeouts und Deadlines
    TIMEOUT: Final[str] = "TIMEOUT"
    DEADLINE_EXCEEDED: Final[str] = "DEADLINE_EXCEEDED"

    # Service-Verfügbarkeit
    SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"
    DEPENDENCY_ERROR: Final[str] = "DEPENDENCY_ERROR"

    # Agent-spezifische Fehler
    AGENT_NOT_FOUND: Final[str] = "AGENT_NOT_FOUND"
    CAPABILITY_NOT_AVAILABLE: Final[str] = "CAPABILITY_NOT_AVAILABLE"
    TASK_IDEMPOTENCY_CONFLICT: Final[str] = "TASK_IDEMPOTENCY_CONFLICT"
    A2A_DELIVERY_FAILED: Final[str] = "A2A_DELIVERY_FAILED"
    POLICY_BLOCKED: Final[str] = "POLICY_BLOCKED"
    VERSION_DEPRECATED: Final[str] = "VERSION_DEPRECATED"
    TOOL_VALIDATION_FAILED: Final[str] = "TOOL_VALIDATION_FAILED"
    TENANT_FORBIDDEN: Final[str] = "TENANT_FORBIDDEN"

    # Netzwerk und Azure-spezifische Fehler
    NETWORK_ERROR: Final[str] = "NETWORK_ERROR"
    AZURE_ERROR: Final[str] = "AZURE_ERROR"
    AUTH_ERROR: Final[str] = "AUTH_ERROR"


# ============================================================================
# ERROR-KATEGORIEN
# ============================================================================

class ErrorCategory(Enum):
    """Kategorisierung von Fehlern für bessere Strukturierung."""

    SYSTEM = "system_error"
    VALIDATION = "validation_error"
    RESOURCE = "resource_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    RATE_LIMIT = "rate_limit_error"
    TIMEOUT = "timeout_error"
    SERVICE = "service_error"
    AGENT = "agent_error"
    NETWORK = "network_error"
    POLICY = "policy_error"
    CAPABILITY = "capability_error"
    TRANSPORT = "transport_error"
    DEPRECATION = "deprecation_error"
    CONFLICT = "conflict_error"


# ============================================================================
# STANDARDISIERTE ERROR-DEFINITION
# ============================================================================

@dataclass(frozen=True)
class StandardError:
    """Standardisierte Error-Definition für einheitliche Fehlerbehandlung.

    Attributes:
        code: Maschinell lesbarer Fehlercode (SCREAMING_SNAKE_CASE)
        http_status: Entsprechender HTTP-Statuscode
        message: Kurze, menschlich lesbare Beschreibung (Deutsch)
        category: Error-Kategorie für Gruppierung
        retryable: Ob der Fehler wiederholbar ist
    """

    code: str
    http_status: int
    message: str
    category: ErrorCategory
    retryable: bool = False


# ============================================================================
# KONSOLIDIERTER ERROR-KATALOG
# ============================================================================

# Konsolidiert alle Error-Definitionen aus der gesamten Codebase
STANDARD_ERRORS: dict[str, StandardError] = {
    # System-Fehler
    ErrorCodes.INTERNAL_ERROR: StandardError(
        ErrorCodes.INTERNAL_ERROR,
        HTTPStatusCodes.INTERNAL_SERVER_ERROR,
        "Interner Serverfehler",
        ErrorCategory.SYSTEM,
        retryable=True
    ),
    ErrorCodes.UNKNOWN_ERROR: StandardError(
        ErrorCodes.UNKNOWN_ERROR,
        HTTPStatusCodes.INTERNAL_SERVER_ERROR,
        "Unbekannter Fehler",
        ErrorCategory.SYSTEM,
        retryable=False
    ),

    # Validierungs-Fehler
    ErrorCodes.VALIDATION_ERROR: StandardError(
        ErrorCodes.VALIDATION_ERROR,
        HTTPStatusCodes.UNPROCESSABLE_ENTITY,
        "Validierungsfehler",
        ErrorCategory.VALIDATION,
        retryable=False
    ),
    ErrorCodes.BAD_REQUEST: StandardError(
        ErrorCodes.BAD_REQUEST,
        HTTPStatusCodes.BAD_REQUEST,
        "Ungültige Anfrage",
        ErrorCategory.VALIDATION,
        retryable=False
    ),
    ErrorCodes.INVALID_FORMAT: StandardError(
        ErrorCodes.INVALID_FORMAT,
        HTTPStatusCodes.BAD_REQUEST,
        "Ungültiges Format",
        ErrorCategory.VALIDATION,
        retryable=False
    ),
    ErrorCodes.MISSING_FIELD: StandardError(
        ErrorCodes.MISSING_FIELD,
        HTTPStatusCodes.BAD_REQUEST,
        "Pflichtfeld fehlt",
        ErrorCategory.VALIDATION,
        retryable=False
    ),

    # Ressourcen-Fehler
    ErrorCodes.NOT_FOUND: StandardError(
        ErrorCodes.NOT_FOUND,
        HTTPStatusCodes.NOT_FOUND,
        "Ressource nicht gefunden",
        ErrorCategory.RESOURCE,
        retryable=False
    ),
    ErrorCodes.CONFLICT: StandardError(
        ErrorCodes.CONFLICT,
        HTTPStatusCodes.CONFLICT,
        "Ressourcenkonflikt",
        ErrorCategory.CONFLICT,
        retryable=False
    ),
    ErrorCodes.GONE: StandardError(
        ErrorCodes.GONE,
        HTTPStatusCodes.GONE,
        "Ressource nicht mehr verfügbar",
        ErrorCategory.RESOURCE,
        retryable=False
    ),

    # Authentifizierung/Autorisierung
    ErrorCodes.UNAUTHORIZED: StandardError(
        ErrorCodes.UNAUTHORIZED,
        HTTPStatusCodes.UNAUTHORIZED,
        "Nicht authentifiziert",
        ErrorCategory.AUTHENTICATION,
        retryable=False
    ),
    ErrorCodes.FORBIDDEN: StandardError(
        ErrorCodes.FORBIDDEN,
        HTTPStatusCodes.FORBIDDEN,
        "Zugriff verweigert",
        ErrorCategory.AUTHORIZATION,
        retryable=False
    ),
    ErrorCodes.TOKEN_EXPIRED: StandardError(
        ErrorCodes.TOKEN_EXPIRED,
        HTTPStatusCodes.UNAUTHORIZED,
        "Token abgelaufen",
        ErrorCategory.AUTHENTICATION,
        retryable=False
    ),

    # Rate-Limiting
    ErrorCodes.RATE_LIMIT_EXCEEDED: StandardError(
        ErrorCodes.RATE_LIMIT_EXCEEDED,
        HTTPStatusCodes.TOO_MANY_REQUESTS,
        "Rate-Limit überschritten",
        ErrorCategory.RATE_LIMIT,
        retryable=True
    ),
    ErrorCodes.BUDGET_EXCEEDED: StandardError(
        ErrorCodes.BUDGET_EXCEEDED,
        HTTPStatusCodes.TOO_MANY_REQUESTS,
        "Budget überschritten",
        ErrorCategory.RATE_LIMIT,
        retryable=True
    ),

    # Timeouts
    ErrorCodes.TIMEOUT: StandardError(
        ErrorCodes.TIMEOUT,
        HTTPStatusCodes.GATEWAY_TIMEOUT,
        "Zeitüberschreitung",
        ErrorCategory.TIMEOUT,
        retryable=True
    ),
    ErrorCodes.DEADLINE_EXCEEDED: StandardError(
        ErrorCodes.DEADLINE_EXCEEDED,
        HTTPStatusCodes.GATEWAY_TIMEOUT,
        "Deadline überschritten",
        ErrorCategory.TIMEOUT,
        retryable=True
    ),

    # Service-Verfügbarkeit
    ErrorCodes.SERVICE_UNAVAILABLE: StandardError(
        ErrorCodes.SERVICE_UNAVAILABLE,
        HTTPStatusCodes.SERVICE_UNAVAILABLE,
        "Service nicht verfügbar",
        ErrorCategory.SERVICE,
        retryable=True
    ),
    ErrorCodes.DEPENDENCY_ERROR: StandardError(
        ErrorCodes.DEPENDENCY_ERROR,
        HTTPStatusCodes.BAD_GATEWAY,
        "Abhängigkeitsfehler",
        ErrorCategory.SERVICE,
        retryable=True
    ),

    # Agent-spezifische Fehler
    ErrorCodes.AGENT_NOT_FOUND: StandardError(
        ErrorCodes.AGENT_NOT_FOUND,
        HTTPStatusCodes.NOT_FOUND,
        "Agent nicht gefunden",
        ErrorCategory.AGENT,
        retryable=False
    ),
    ErrorCodes.CAPABILITY_NOT_AVAILABLE: StandardError(
        ErrorCodes.CAPABILITY_NOT_AVAILABLE,
        HTTPStatusCodes.CONFLICT,
        "Capability nicht verfügbar",
        ErrorCategory.CAPABILITY,
        retryable=True
    ),
    ErrorCodes.TASK_IDEMPOTENCY_CONFLICT: StandardError(
        ErrorCodes.TASK_IDEMPOTENCY_CONFLICT,
        HTTPStatusCodes.CONFLICT,
        "Idempotenzkonflikt für Task",
        ErrorCategory.CONFLICT,
        retryable=False
    ),
    ErrorCodes.A2A_DELIVERY_FAILED: StandardError(
        ErrorCodes.A2A_DELIVERY_FAILED,
        HTTPStatusCodes.BAD_GATEWAY,
        "A2A-Zustellung fehlgeschlagen",
        ErrorCategory.TRANSPORT,
        retryable=True
    ),
    ErrorCodes.POLICY_BLOCKED: StandardError(
        ErrorCodes.POLICY_BLOCKED,
        HTTPStatusCodes.UNPROCESSABLE_ENTITY,
        "Policy-Verstoß: Anfrage blockiert",
        ErrorCategory.POLICY,
        retryable=False
    ),
    ErrorCodes.VERSION_DEPRECATED: StandardError(
        ErrorCodes.VERSION_DEPRECATED,
        HTTPStatusCodes.GONE,
        "Agent-Version veraltet",
        ErrorCategory.DEPRECATION,
        retryable=False
    ),
    ErrorCodes.TOOL_VALIDATION_FAILED: StandardError(
        ErrorCodes.TOOL_VALIDATION_FAILED,
        HTTPStatusCodes.UNPROCESSABLE_ENTITY,
        "Tool-Parameter Validierung fehlgeschlagen",
        ErrorCategory.VALIDATION,
        retryable=False
    ),
    ErrorCodes.TENANT_FORBIDDEN: StandardError(
        ErrorCodes.TENANT_FORBIDDEN,
        HTTPStatusCodes.FORBIDDEN,
        "Tenant nicht berechtigt",
        ErrorCategory.AUTHORIZATION,
        retryable=False
    ),

    # Netzwerk-Fehler
    ErrorCodes.NETWORK_ERROR: StandardError(
        ErrorCodes.NETWORK_ERROR,
        HTTPStatusCodes.BAD_GATEWAY,
        "Netzwerkfehler",
        ErrorCategory.NETWORK,
        retryable=True
    ),
    ErrorCodes.AZURE_ERROR: StandardError(
        ErrorCodes.AZURE_ERROR,
        HTTPStatusCodes.BAD_GATEWAY,
        "Azure-Service-Fehler",
        ErrorCategory.SERVICE,
        retryable=True
    ),
    ErrorCodes.AUTH_ERROR: StandardError(
        ErrorCodes.AUTH_ERROR,
        HTTPStatusCodes.UNAUTHORIZED,
        "Authentifizierungsfehler",
        ErrorCategory.AUTHENTICATION,
        retryable=False
    ),
}


# ============================================================================
# UTILITY-FUNKTIONEN (Konsolidiert und vereinfacht)
# ============================================================================

def get_error_definition(code: str) -> StandardError | None:
    """Ruft Error-Definition für gegebenen Code ab.

    Args:
        code: Error-Code

    Returns:
        StandardError-Definition oder None falls nicht gefunden
    """
    return STANDARD_ERRORS.get(code)


def is_retryable_error(code: str) -> bool:
    """Prüft ob ein Error wiederholbar ist.

    Args:
        code: Error-Code

    Returns:
        True wenn wiederholbar, False sonst
    """
    error_def = get_error_definition(code)
    return error_def.retryable if error_def else False


def get_http_status_for_error(code: str) -> int:
    """Ruft HTTP-Status-Code für Error-Code ab.

    Args:
        code: Error-Code

    Returns:
        HTTP-Status-Code oder 500 als Fallback
    """
    error_def = get_error_definition(code)
    return error_def.http_status if error_def else HTTPStatusCodes.INTERNAL_SERVER_ERROR


def build_error_detail(
    code: str,
    *,
    reason: str | None = None,
    **extra: object
) -> dict[str, object]:
    """Erzeugt konsistentes Fehler-Detail für HTTPException.

    Konsolidiert die ursprüngliche build_error_detail Funktion mit
    verbesserter Typisierung und Standardisierung.

    Args:
        code: Fehlercode aus STANDARD_ERRORS
        reason: Optionale spezifische Begründung
        **extra: Zusätzliche Felder

    Returns:
        Detail-Dictionary mit standardisierten Feldern
    """
    error_def = get_error_definition(code)

    if not error_def:
        # Fallback auf generischen Fehler
        base_detail = {
            "code": code,
            "message": "Unbekannter Fehler",
            "category": ErrorCategory.SYSTEM.value,
            "retryable": False
        }
    else:
        base_detail = {
            "code": error_def.code,
            "message": error_def.message,
            "category": error_def.category.value,
            "retryable": error_def.retryable
        }

    # Optionale Felder hinzufügen
    if reason:
        base_detail["reason"] = reason

    if extra:
        base_detail.update(extra)

    return base_detail


def get_errors_by_category(category: ErrorCategory) -> dict[str, StandardError]:
    """Ruft alle Errors einer bestimmten Kategorie ab.

    Args:
        category: Error-Kategorie

    Returns:
        Dictionary mit Error-Code -> StandardError Mapping
    """
    return {
        code: error_def
        for code, error_def in STANDARD_ERRORS.items()
        if error_def.category == category
    }


def get_retryable_errors() -> dict[str, StandardError]:
    """Ruft alle wiederholbaren Errors ab.

    Returns:
        Dictionary mit wiederholbaren Error-Definitionen
    """
    return {
        code: error_def
        for code, error_def in STANDARD_ERRORS.items()
        if error_def.retryable
    }


# ============================================================================
# BACKWARD-COMPATIBILITY (Deprecated)
# ============================================================================

# Legacy AgentError für Backward-Compatibility
@dataclass(frozen=True)
class AgentError:
    """Legacy AgentError-Klasse für Backward-Compatibility.

    DEPRECATED: Verwende StandardError stattdessen.
    """

    code: str
    http_status: int
    message: str
    type: str


# Legacy AGENT_ERRORS für Backward-Compatibility
AGENT_ERRORS: dict[str, AgentError] = {
    code: AgentError(
        error_def.code,
        error_def.http_status,
        error_def.message,
        error_def.category.value
    )
    for code, error_def in STANDARD_ERRORS.items()
    if error_def.category in {
        ErrorCategory.AGENT,
        ErrorCategory.CAPABILITY,
        ErrorCategory.TRANSPORT,
        ErrorCategory.POLICY,
        ErrorCategory.DEPRECATION
    }
}


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AGENT_ERRORS",
    "STANDARD_ERRORS",
    # Legacy API (Deprecated)
    "AgentError",
    "ErrorCategory",
    "ErrorCodes",
    # Neue standardisierte API
    "HTTPStatusCodes",
    "StandardError",
    "build_error_detail",
    "get_error_definition",
    "get_errors_by_category",
    "get_http_status_for_error",
    "get_retryable_errors",
    "is_retryable_error",
]
