"""Zentrale Keiko Exception-Hierarchie.

Alle Ausnahmen erben von ``KeikoException`` und tragen konsistente Felder
für Fehlercode, Nachricht, optionale Details und Schweregrad. Die
Fehlermeldungen sind deutsch, die Klassennamen bleiben englisch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .constants import (
    DEFAULT_ERROR_MESSAGES,
    ErrorCode,
    SeverityLevel,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class KeikoErrorPayload:
    """Strukturierte Fehlerdaten für API/Logging.

    Attributes:
        error_code: Stabiler, maschinenlesbarer Fehlercode
        message: Deutsche, menschlich lesbare Beschreibung
        severity: Schweregrad (z. B. LOW, MEDIUM, HIGH, CRITICAL)
        details: Optionale Zusatzinformationen (PII-bereinigt)
    """

    error_code: str
    message: str
    severity: str  # SeverityLevel.value
    details: Mapping[str, Any] | None


class KeikoException(Exception):
    """Basisklasse für alle domänenspezifischen Keiko-Ausnahmen.

    Args:
        error_code: Maschineller Fehlercode in SCREAMING_SNAKE_CASE
        message: Deutsche Fehlermeldung
        severity: Schweregrad, Standard ``HIGH``
        details: Optionale strukturierte Zusatzinfos
        cause: Optionale ursprüngliche Ausnahme (Verkettung)

    Returns:
        None
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        severity: str = SeverityLevel.HIGH.value,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code: str = error_code
        self.message: str = message
        self.severity: str = severity
        self.details: Mapping[str, Any] | None = details
        self.__cause__ = cause

    def to_payload(self) -> KeikoErrorPayload:
        """Serialisiert die Ausnahme in ein strukturiertes Payload-Objekt."""
        return KeikoErrorPayload(
            error_code=self.error_code,
            message=self.message,
            severity=self.severity,
            details=self.details,
        )

    def __str__(self) -> str:
        """Gibt die deutsche Fehlermeldung zurück."""
        return self.message


# ---------------------------------------------------------------------------
# Exception Factory für konsolidierte Exception-Erstellung
# ---------------------------------------------------------------------------


class ExceptionFactory:
    """Factory-Klasse für die Erstellung von domänenspezifischen Exceptions.

    Konsolidiert die Erstellung von Exception-Instanzen und eliminiert
    Code-Duplikation durch einheitliche Factory-Methoden.
    """

    @staticmethod
    def create_exception(
        error_code: str,
        message: str | None = None,
        **kwargs: Any
    ) -> KeikoException:
        """Erstellt eine KeikoException mit dem gegebenen Error-Code.

        Args:
            error_code: Error-Code aus ErrorCode-Konstanten
            message: Optionale benutzerdefinierte Nachricht
            **kwargs: Zusätzliche Parameter für KeikoException

        Returns:
            KeikoException-Instanz
        """
        if message is None:
            message = DEFAULT_ERROR_MESSAGES.get(error_code, "Unbekannter Fehler")

        return KeikoException(error_code, message, **kwargs)


# ---------------------------------------------------------------------------
# Domänenspezifische Ausnahmen (konsolidiert über Factory)
# ---------------------------------------------------------------------------


class AgentError(KeikoException):
    """Agent-bezogener Fehler (z. B. Capability, Routing, Policy)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.AGENT_ERROR]
        super().__init__(ErrorCode.AGENT_ERROR, message, **kwargs)


class AzureError(KeikoException):
    """Azure-spezifischer Integrationsfehler (z. B. SDK, Quoten)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.AZURE_ERROR]
        super().__init__(ErrorCode.AZURE_ERROR, message, **kwargs)


class NetworkError(KeikoException):
    """Netzwerk-/Transportfehler (Timeouts, DNS, TLS)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.NETWORK_ERROR]
        super().__init__(ErrorCode.NETWORK_ERROR, message, **kwargs)


class AuthError(KeikoException):
    """Authentifizierungs-/Autorisierungsfehler."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.AUTH_ERROR]
        super().__init__(ErrorCode.AUTH_ERROR, message, **kwargs)


class ValidationError(KeikoException):
    """Validierungsfehler für Eingaben oder Konfigurationen."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.VALIDATION_ERROR]
        super().__init__(ErrorCode.VALIDATION_ERROR, message, **kwargs)


class NotFoundError(KeikoException):
    """Ressource wurde nicht gefunden."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.NOT_FOUND]
        super().__init__(ErrorCode.NOT_FOUND, message, **kwargs)


class RateLimitExceeded(KeikoException):
    """Ratenlimit überschritten."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.RATE_LIMIT_EXCEEDED]
        super().__init__(ErrorCode.RATE_LIMIT_EXCEEDED, message, **kwargs)


class OperationTimeout(KeikoException):
    """Zeitlimit für eine Operation überschritten."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.TIMEOUT]
        super().__init__(ErrorCode.TIMEOUT, message, **kwargs)


class ConflictError(KeikoException):
    """Zustandskonflikt (z. B. Idempotenzkonflikt)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.CONFLICT]
        super().__init__(ErrorCode.CONFLICT, message, **kwargs)


class PolicyViolationError(KeikoException):
    """Verstoß gegen Richtlinien/Policies."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.POLICY_VIOLATION]
        super().__init__(ErrorCode.POLICY_VIOLATION, message, **kwargs)


class BudgetExceededError(KeikoException):
    """Budgetgrenzen überschritten (Kosten/Tokens/Zeit)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.BUDGET_EXCEEDED]
        super().__init__(ErrorCode.BUDGET_EXCEEDED, message, **kwargs)


class DeadlineExceededError(KeikoException):
    """Deadline überschritten (Ende-zu-Ende-Timeout)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.DEADLINE_EXCEEDED]
        super().__init__(ErrorCode.DEADLINE_EXCEEDED, message, **kwargs)


class ServiceUnavailableError(KeikoException):
    """Abhängigkeit/Service nicht erreichbar oder bereit."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.SERVICE_UNAVAILABLE]
        super().__init__(ErrorCode.SERVICE_UNAVAILABLE, message, **kwargs)


class DependencyError(KeikoException):
    """Fehler in abhängigen Komponenten (externe Systeme)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.DEPENDENCY_ERROR]
        super().__init__(ErrorCode.DEPENDENCY_ERROR, message, **kwargs)


class BadRequestError(KeikoException):
    """Ungültige Anfrage (HTTP 400)."""

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        if message is None:
            message = DEFAULT_ERROR_MESSAGES[ErrorCode.BAD_REQUEST]
        super().__init__(ErrorCode.BAD_REQUEST, message, **kwargs)


__all__ = [
    # Domain Exceptions
    "AgentError",
    "AuthError",
    "AzureError",
    "BadRequestError",
    "BudgetExceededError",
    "ConflictError",
    "DeadlineExceededError",
    "DependencyError",
    "ExceptionFactory",
    "KeikoErrorPayload",
    # Base Classes
    "KeikoException",
    "NetworkError",
    "NotFoundError",
    "OperationTimeout",
    "PolicyViolationError",
    "RateLimitExceeded",
    "ServiceUnavailableError",
    "ValidationError",
]

# ---------------------------------------------------------------------------
# Backward Compatibility Aliases (Deprecated - Use ExceptionFactory instead)
# ---------------------------------------------------------------------------


# Kompatibilitäts-Aliases für bestehenden Code
# Diese sollten schrittweise durch ExceptionFactory.create_exception() ersetzt werden
KeikoValidationError = ValidationError
KeikoServiceError = ServiceUnavailableError
KeikoAzureError = AzureError
KeikoTimeoutError = OperationTimeout
KeikoAuthenticationError = AuthError
KeikoRateLimitError = RateLimitExceeded
KeikoNotFoundError = NotFoundError
KeikoBadRequestError = BadRequestError


# Erweiterte __all__ Liste mit Backward-Compatibility
__all__ += [
    "KeikoAuthenticationError",
    "KeikoAzureError",
    "KeikoBadRequestError",
    "KeikoNotFoundError",
    "KeikoRateLimitError",
    "KeikoServiceError",
    "KeikoTimeoutError",
    # Deprecated Aliases (for backward compatibility)
    "KeikoValidationError",
]
