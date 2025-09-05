# backend/kei_agents/exceptions.py
"""Standardisierte Exception-Klassen für Keiko Personal Assistant

Enterprise-Grade Exception-Handling mit:
- Hierarchische Exception-Struktur
- Strukturierte Error-Metadaten
- Logging-Integration
- Internationalisierung-Support
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Schweregrade für Fehler."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Kategorien für Fehler."""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"


@dataclass
class ErrorContext:
    """Kontext-Informationen für Fehler."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    component: str | None = None
    operation: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class KEIAgentError(Exception):
    """Basis-Exception für alle KEI-Agent-Framework Fehler."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        context: ErrorContext | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        """Initialisiert KEI-Agent-Error.

        Args:
            message: Fehlermeldung
            error_code: Eindeutiger Error-Code
            severity: Schweregrad des Fehlers
            category: Kategorie des Fehlers
            context: Error-Kontext
            details: Zusätzliche Details
            cause: Ursprüngliche Exception
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.details = details or {}
        self.cause = cause

    def _generate_error_code(self) -> str:
        """Generiert automatischen Error-Code."""
        class_name = self.__class__.__name__
        return f"KEI_{class_name.upper()}_{int(time.time())}"

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu Dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": {
                "error_id": self.context.error_id,
                "timestamp": self.context.timestamp,
                "component": self.context.component,
                "operation": self.context.operation,
                "agent_id": self.context.agent_id,
                "user_id": self.context.user_id,
                "correlation_id": self.context.correlation_id,
                "trace_id": self.context.trace_id,
                "metadata": self.context.metadata,
            },
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String-Repräsentation der Exception."""
        return f"[{self.error_code}] {self.message}"


class KEIValidationError(KEIAgentError):
    """Fehler bei Input-Validierung."""

    def __init__(
        self,
        message: str,
        validation_field: str | None = None,
        value: Any = None,
        **kwargs
    ):
        """Initialisiert Validation-Error.

        Args:
            message: Fehlermeldung
            field: Feld das validiert wurde
            value: Ungültiger Wert
            **kwargs: Weitere KEIAgentError-Parameter
        """
        details = kwargs.pop("details", {})
        if validation_field:
            details["field"] = validation_field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            details=details,
            **kwargs
        )


class KEIAuthenticationError(KEIAgentError):
    """Fehler bei Authentifizierung."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class KEIAuthorizationError(KEIAgentError):
    """Fehler bei Autorisierung."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class KEIConfigurationError(KEIAgentError):
    """Fehler bei Konfiguration."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class KEINetworkError(KEIAgentError):
    """Fehler bei Netzwerk-Operationen."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class KEITimeoutError(KEIAgentError):
    """Fehler bei Timeout."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class KEIResourceError(KEIAgentError):
    """Fehler bei Ressourcen-Management."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type

        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class KEIBusinessLogicError(KEIAgentError):
    """Fehler in Business-Logik."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class KEIExternalServiceError(KEIAgentError):
    """Fehler bei externen Services."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        status_code: int | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code

        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class KEIAgentNotFoundError(KEIAgentError):
    """Fehler wenn Agent nicht gefunden wird."""

    def __init__(
        self,
        agent_id: str,
        **kwargs
    ):
        message = f"Agent nicht gefunden: {agent_id}"
        details = kwargs.pop("details", {})
        details["agent_id"] = agent_id

        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class KEICapabilityError(KEIAgentError):
    """Fehler bei Capability-Operationen."""

    def __init__(
        self,
        message: str,
        capability_name: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if capability_name:
            details["capability_name"] = capability_name

        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class KEISecurityError(KEIAgentError):
    """Fehler bei Security-Operationen."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


# Convenience-Funktionen für häufige Error-Patterns

def validation_error(
    message: str,
    validation_field: str | None = None,
    value: Any = None
) -> KEIValidationError:
    """Erstellt Validation-Error."""
    return KEIValidationError(message, validation_field=validation_field, value=value)


def authentication_error(message: str) -> KEIAuthenticationError:
    """Erstellt Authentication-Error."""
    return KEIAuthenticationError(message)


def authorization_error(message: str) -> KEIAuthorizationError:
    """Erstellt Authorization-Error."""
    return KEIAuthorizationError(message)


def timeout_error(
    message: str,
    timeout_seconds: float | None = None
) -> KEITimeoutError:
    """Erstellt Timeout-Error."""
    return KEITimeoutError(message, timeout_seconds=timeout_seconds)


def agent_not_found_error(agent_id: str) -> KEIAgentNotFoundError:
    """Erstellt Agent-Not-Found-Error."""
    return KEIAgentNotFoundError(agent_id)


def external_service_error(
    message: str,
    service_name: str | None = None,
    status_code: int | None = None
) -> KEIExternalServiceError:
    """Erstellt External-Service-Error."""
    return KEIExternalServiceError(
        message,
        service_name=service_name,
        status_code=status_code
    )
