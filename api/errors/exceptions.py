"""Konsolidierte Exception-Hierarchie für die Keiko-API."""

from __future__ import annotations

import uuid
from typing import Any

from .agent_errors import (
    ErrorCategory,
    ErrorCodes,
    get_error_definition,
)

# ============================================================================
# BASIS-EXCEPTION-KLASSE
# ============================================================================

class KeikoAPIException(Exception):
    """Basis-Exception für alle Keiko-API-Fehler."""

    def __init__(
        self,
        error_code: str,
        message: str | None = None,
        *,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        **extra: Any
    ) -> None:
        """Initialisiert Keiko-API-Exception.

        Args:
            error_code: Error-Code aus ErrorCodes
            message: Optionale Fehlermeldung (überschreibt Standard)
            http_status: Optionaler HTTP-Status (überschreibt Standard)
            details: Zusätzliche Error-Details
            correlation_id: Korrelations-ID
            **extra: Zusätzliche Felder für Details
        """
        # Error-Definition aus Katalog laden
        error_def = get_error_definition(error_code)

        if error_def:
            self.error_code = error_def.code
            self.message = message or error_def.message
            self.http_status = http_status or error_def.http_status
            self.category = error_def.category
            self.retryable = error_def.retryable
        else:
            # Fallback für unbekannte Error-Codes
            self.error_code = error_code
            self.message = message or "Unbekannter Fehler"
            self.http_status = http_status or 500
            self.category = ErrorCategory.SYSTEM
            self.retryable = False

        # Details zusammenführen
        self.details = details or {}
        if extra:
            self.details.update(extra)

        # Korrelations-ID generieren falls nicht vorhanden
        self.correlation_id = correlation_id or str(uuid.uuid4())

        # Exception-Message setzen
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu Dictionary.

        Returns:
            Dictionary mit allen Exception-Daten
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "http_status": self.http_status,
            "category": self.category.value,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "retryable": self.retryable,
        }

    def with_details(self, **details: Any) -> KeikoAPIException:
        """Erstellt neue Exception mit zusätzlichen Details.

        Args:
            **details: Zusätzliche Detail-Felder

        Returns:
            Neue Exception-Instanz mit erweiterten Details
        """
        new_details = {**self.details, **details}
        return self.__class__(
            self.error_code,
            self.message,
            http_status=self.http_status,
            details=new_details,
            correlation_id=self.correlation_id
        )

    def with_correlation_id(self, correlation_id: str) -> KeikoAPIException:
        """Erstellt neue Exception mit spezifischer Korrelations-ID.

        Args:
            correlation_id: Neue Korrelations-ID

        Returns:
            Neue Exception-Instanz mit neuer Korrelations-ID
        """
        return self.__class__(
            self.error_code,
            self.message,
            http_status=self.http_status,
            details=self.details,
            correlation_id=correlation_id
        )

    def __str__(self) -> str:
        """String-Repräsentation der Exception."""
        return f"{self.error_code}: {self.message}"

    def __repr__(self) -> str:
        """Debug-Repräsentation der Exception."""
        return (
            f"{self.__class__.__name__}("
            f"error_code='{self.error_code}', "
            f"message='{self.message}', "
            f"http_status={self.http_status}, "
            f"correlation_id='{self.correlation_id}'"
            f")"
        )


# ============================================================================
# SPEZIFISCHE EXCEPTION-KLASSEN
# ============================================================================

class ValidationException(KeikoAPIException):
    """Exception für Validierungsfehler."""

    def __init__(
        self,
        message: str = "Validierungsfehler",
        *,
        field: str | None = None,
        value: Any | None = None,
        **kwargs: Any
    ) -> None:
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        super().__init__(
            ErrorCodes.VALIDATION_ERROR,
            message,
            details=details,
            **kwargs
        )


class ResourceNotFoundException(KeikoAPIException):
    """Exception für nicht gefundene Ressourcen."""

    def __init__(
        self,
        resource_type: str,
        identifier: str | None = None,
        **kwargs: Any
    ) -> None:
        message = f"{resource_type} nicht gefunden"
        if identifier:
            message += f": {identifier}"

        details = {"resource_type": resource_type}
        if identifier:
            details["identifier"] = identifier

        super().__init__(
            ErrorCodes.NOT_FOUND,
            message,
            details=details,
            **kwargs
        )


class ConflictException(KeikoAPIException):
    """Exception für Ressourcen-Konflikte."""

    def __init__(
        self,
        message: str = "Ressourcenkonflikt",
        *,
        resource_type: str | None = None,
        **kwargs: Any
    ) -> None:
        details = {}
        if resource_type:
            details["resource_type"] = resource_type

        super().__init__(
            ErrorCodes.CONFLICT,
            message,
            details=details,
            **kwargs
        )


class AuthenticationException(KeikoAPIException):
    """Exception für Authentifizierungsfehler."""

    def __init__(
        self,
        message: str = "Authentifizierung fehlgeschlagen",
        **kwargs: Any
    ) -> None:
        super().__init__(ErrorCodes.UNAUTHORIZED, message, **kwargs)


class AuthorizationException(KeikoAPIException):
    """Exception für Autorisierungsfehler."""

    def __init__(
        self,
        message: str = "Zugriff verweigert",
        *,
        required_permission: str | None = None,
        **kwargs: Any
    ) -> None:
        details = {}
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(
            ErrorCodes.FORBIDDEN,
            message,
            details=details,
            **kwargs
        )


class RateLimitException(KeikoAPIException):
    """Exception für Rate-Limit-Überschreitungen."""

    def __init__(
        self,
        message: str = "Rate-Limit überschritten",
        *,
        retry_after: int | None = None,
        limit: int | None = None,
        **kwargs: Any
    ) -> None:
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        if limit:
            details["limit"] = limit

        super().__init__(
            ErrorCodes.RATE_LIMIT_EXCEEDED,
            message,
            details=details,
            **kwargs
        )


class TimeoutException(KeikoAPIException):
    """Exception für Timeout-Fehler."""

    def __init__(
        self,
        message: str = "Zeitüberschreitung",
        *,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        **kwargs: Any
    ) -> None:
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation

        super().__init__(
            ErrorCodes.TIMEOUT,
            message,
            details=details,
            **kwargs
        )


class ServiceUnavailableException(KeikoAPIException):
    """Exception für nicht verfügbare Services."""

    def __init__(
        self,
        message: str = "Service nicht verfügbar",
        *,
        service_name: str | None = None,
        **kwargs: Any
    ) -> None:
        details = {}
        if service_name:
            details["service_name"] = service_name

        super().__init__(
            ErrorCodes.SERVICE_UNAVAILABLE,
            message,
            details=details,
            **kwargs
        )


# ============================================================================
# AGENT-SPEZIFISCHE EXCEPTIONS
# ============================================================================

class AgentNotFoundException(ResourceNotFoundException):
    """Exception für nicht gefundene Agents."""

    def __init__(
        self,
        agent_id: str,
        **kwargs: Any
    ) -> None:
        super().__init__(
            "Agent",
            agent_id,
            error_code=ErrorCodes.AGENT_NOT_FOUND,
            **kwargs
        )


class CapabilityNotAvailableException(KeikoAPIException):
    """Exception für nicht verfügbare Capabilities."""

    def __init__(
        self,
        capability: str,
        *,
        agent_id: str | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Capability '{capability}' nicht verfügbar"
        if agent_id:
            message += f" für Agent {agent_id}"

        details = {"capability": capability}
        if agent_id:
            details["agent_id"] = agent_id

        super().__init__(
            ErrorCodes.CAPABILITY_NOT_AVAILABLE,
            message,
            details=details,
            **kwargs
        )


class PolicyViolationException(KeikoAPIException):
    """Exception für Policy-Verstöße."""

    def __init__(
        self,
        policy: str,
        message: str = "Policy-Verstoß",
        **kwargs: Any
    ) -> None:
        details = {"policy": policy}

        super().__init__(
            ErrorCodes.POLICY_BLOCKED,
            f"{message}: {policy}",
            details=details,
            **kwargs
        )


# ============================================================================
# CONVENIENCE-FUNKTIONEN
# ============================================================================

def create_exception(
    error_code: str,
    message: str | None = None,
    **kwargs: Any
) -> KeikoAPIException:
    """Erstellt Exception basierend auf Error-Code.

    Args:
        error_code: Error-Code
        message: Optionale Fehlermeldung
        **kwargs: Zusätzliche Parameter

    Returns:
        Passende Exception-Instanz
    """
    # Spezifische Exception-Klassen für häufige Error-Codes
    exception_mapping = {
        ErrorCodes.VALIDATION_ERROR: ValidationException,
        ErrorCodes.NOT_FOUND: ResourceNotFoundException,
        ErrorCodes.CONFLICT: ConflictException,
        ErrorCodes.UNAUTHORIZED: AuthenticationException,
        ErrorCodes.FORBIDDEN: AuthorizationException,
        ErrorCodes.RATE_LIMIT_EXCEEDED: RateLimitException,
        ErrorCodes.TIMEOUT: TimeoutException,
        ErrorCodes.SERVICE_UNAVAILABLE: ServiceUnavailableException,
        ErrorCodes.AGENT_NOT_FOUND: AgentNotFoundException,
        ErrorCodes.CAPABILITY_NOT_AVAILABLE: CapabilityNotAvailableException,
        ErrorCodes.POLICY_BLOCKED: PolicyViolationException,
    }

    exception_class = exception_mapping.get(error_code, KeikoAPIException)

    if exception_class == KeikoAPIException:
        return exception_class(error_code, message, **kwargs)
    # Für spezifische Exception-Klassen message als ersten Parameter
    if message:
        return exception_class(message, **kwargs)
    return exception_class(**kwargs)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Agent-spezifische Exceptions
    "AgentNotFoundException",
    "AuthenticationException",
    "AuthorizationException",
    "CapabilityNotAvailableException",
    "ConflictException",
    # Basis-Exception
    "KeikoAPIException",
    "PolicyViolationException",
    "RateLimitException",
    "ResourceNotFoundException",
    "ServiceUnavailableException",
    "TimeoutException",
    # Allgemeine Exceptions
    "ValidationException",
    # Utility-Funktionen
    "create_exception",
]
