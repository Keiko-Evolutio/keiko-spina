# backend/services/clients/common/error_handling.py
"""Error Handling Utilities für Client Services.

Bietet konsistente Fehlerbehandlung und Logging für alle Client Services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .constants import (
    CONTENT_POLICY_VIOLATION_ERROR,
    SAFETY_SYSTEM_ERROR,
    SERVICE_NOT_CONFIGURED_ERROR,
    SERVICE_UNAVAILABLE_ERROR,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class ServiceError:
    """Strukturierte Fehlerinformation für Services."""

    error_type: str
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = False
    retry_after: float | None = None


class ClientServiceException(Exception):
    """Base Exception für Client Service Fehler."""

    def __init__(
        self,
        message: str,
        error_type: str = "client_error",
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.recoverable = recoverable


class ServiceNotConfiguredException(ClientServiceException):
    """Exception für nicht konfigurierte Services."""

    def __init__(self, service_name: str, missing_config: str | None = None) -> None:
        message = f"{service_name}: {SERVICE_NOT_CONFIGURED_ERROR}"
        if missing_config:
            message += f" (fehlend: {missing_config})"

        super().__init__(
            message=message,
            error_type="configuration_error",
            details={"service": service_name, "missing_config": missing_config},
            recoverable=False,
        )


class ServiceUnavailableException(ClientServiceException):
    """Exception für nicht verfügbare Services."""

    def __init__(
        self,
        service_name: str,
        reason: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        message = f"{service_name}: {SERVICE_UNAVAILABLE_ERROR}"
        if reason:
            message += f" ({reason})"

        super().__init__(
            message=message,
            error_type="availability_error",
            details={"service": service_name, "reason": reason},
            recoverable=True,
        )
        self.retry_after = retry_after


class ContentPolicyViolationException(ClientServiceException):
    """Exception für Content Policy Violations."""

    def __init__(
        self,
        original_prompt: str,
        sanitized_prompt: str | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        message = "Content Policy Violation erkannt"

        super().__init__(
            message=message,
            error_type="content_policy_violation",
            details={
                "original_prompt": original_prompt,
                "sanitized_prompt": sanitized_prompt,
                "suggestions": suggestions or [],
            },
            recoverable=True,
        )


def handle_http_error(
    response_status: int,
    response_data: dict[str, Any] | None = None,
    service_name: str = "unknown",
) -> ServiceError:
    """Behandelt HTTP-Fehler und erstellt strukturierte Fehlerinformation.

    Args:
        response_status: HTTP Status Code
        response_data: Response-Daten (optional)
        service_name: Name des Services

    Returns:
        ServiceError mit strukturierten Fehlerinformationen
    """
    error_message = f"HTTP {response_status}"
    error_type = "http_error"
    recoverable = False
    retry_after = None

    # Spezifische HTTP-Status-Codes behandeln
    if response_status == 400:
        error_type = "bad_request"
        error_message = "Ungültige Anfrage"
    elif response_status == 401:
        error_type = "authentication_error"
        error_message = "Authentifizierung fehlgeschlagen"
    elif response_status == 403:
        error_type = "authorization_error"
        error_message = "Zugriff verweigert"
    elif response_status == 404:
        error_type = "not_found"
        error_message = "Ressource nicht gefunden"
    elif response_status == 429:
        error_type = "rate_limit_exceeded"
        error_message = "Rate Limit überschritten"
        recoverable = True
        # Retry-After Header aus response_data extrahieren falls vorhanden
        if response_data and "retry_after" in response_data:
            retry_after = float(response_data["retry_after"])
        else:
            retry_after = 60.0  # Default: 1 Minute
    elif 500 <= response_status < 600:
        error_type = "server_error"
        error_message = "Server-Fehler"
        recoverable = True
        retry_after = 30.0  # Default: 30 Sekunden

    # Response-Daten in Fehlermeldung einbeziehen
    if response_data:
        if "error" in response_data:
            error_message += f": {response_data['error']}"
        elif "message" in response_data:
            error_message += f": {response_data['message']}"

    logger.warning({
        "event": "http_error_handled",
        "service": service_name,
        "status": response_status,
        "error_type": error_type,
        "recoverable": recoverable,
        "retry_after": retry_after,
    })

    return ServiceError(
        error_type=error_type,
        message=error_message,
        details={
            "status_code": response_status,
            "response_data": response_data,
            "service": service_name,
        },
        recoverable=recoverable,
        retry_after=retry_after,
    )


def is_content_policy_violation(error_message: str) -> bool:
    """Prüft ob ein Fehler eine Content Policy Violation ist.

    Args:
        error_message: Fehlermeldung

    Returns:
        True wenn es eine Content Policy Violation ist
    """
    error_lower = error_message.lower()
    return (
        CONTENT_POLICY_VIOLATION_ERROR in error_lower or
        SAFETY_SYSTEM_ERROR in error_lower or
        "content filter" in error_lower or
        "policy violation" in error_lower
    )


def log_service_error(
    error: Exception,
    service_name: str,
    operation: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Loggt Service-Fehler mit strukturierten Informationen.

    Args:
        error: Die aufgetretene Exception
        service_name: Name des Services
        operation: Name der Operation
        context: Zusätzlicher Kontext (optional)
    """
    log_data = {
        "event": "service_error",
        "service": service_name,
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    # Zusätzliche Informationen für ClientServiceException
    if isinstance(error, ClientServiceException):
        log_data.update({
            "client_error_type": error.error_type,
            "recoverable": error.recoverable,
            "details": error.details,
        })

    # Kontext hinzufügen
    if context:
        log_data["context"] = context

    logger.error(log_data)


def create_fallback_result(
    service_name: str,
    operation: str,
    reason: str,
    default_value: Any = None,
) -> dict[str, Any]:
    """Erstellt ein strukturiertes Fallback-Ergebnis.

    Args:
        service_name: Name des Services
        operation: Name der Operation
        reason: Grund für den Fallback
        default_value: Standard-Wert (optional)

    Returns:
        Strukturiertes Fallback-Ergebnis
    """
    from datetime import datetime

    result = {
        "success": False,
        "fallback": True,
        "service": service_name,
        "operation": operation,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if default_value is not None:
        result["default_value"] = default_value

    logger.info({
        "event": "fallback_result_created",
        "service": service_name,
        "operation": operation,
        "reason": reason,
    })

    return result


class ErrorHandler:
    """Utility-Klasse für konsistente Fehlerbehandlung."""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name

    def handle_exception(
        self,
        error: Exception,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> ServiceError:
        """Behandelt eine Exception und erstellt ServiceError.

        Args:
            error: Die aufgetretene Exception
            operation: Name der Operation
            context: Zusätzlicher Kontext (optional)

        Returns:
            ServiceError mit strukturierten Informationen
        """
        log_service_error(error, self.service_name, operation, context)

        if isinstance(error, ClientServiceException):
            return ServiceError(
                error_type=error.error_type,
                message=str(error),
                details=error.details,
                recoverable=error.recoverable,
            )

        # Allgemeine Exception-Behandlung
        return ServiceError(
            error_type="unexpected_error",
            message=str(error),
            details={"exception_type": type(error).__name__},
            recoverable=False,
        )
