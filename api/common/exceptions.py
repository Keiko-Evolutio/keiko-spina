"""Gemeinsame Exception-Utilities für API-Module.

Dieses Modul stellt wiederverwendbare HTTP Exception-Utilities bereit,
die konsistente Fehlerbehandlung in allen API-Modulen ermöglichen.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from kei_logging import get_logger

logger = get_logger(__name__)


# Standard Error-Codes für konsistente API-Responses
class ErrorCodes:
    """Zentrale Definition aller API Error-Codes."""

    # Resource-bezogene Errors
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    NAME_EXISTS = "NAME_EXISTS"
    CONFLICT = "CONFLICT"

    # Validierungs-Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"

    # Authentifizierung/Autorisierung
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Server-Errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"

    # Business Logic Errors
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"


class APIExceptionBuilder:
    """Builder-Klasse für konsistente HTTP Exception-Erstellung.

    Ermöglicht fluent API für Exception-Konfiguration mit standardisierten
    Error-Codes, Nachrichten und zusätzlichen Metadaten.
    """

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        """Initialisiert Exception Builder.

        Args:
            status_code: HTTP Status-Code
            error_code: Standardisierter Error-Code
            message: Benutzerfreundliche Fehlermeldung
        """
        self._status_code = status_code
        self._error_code = error_code
        self._message = message
        self._details: dict[str, Any] = {}
        self._headers: dict[str, str] | None = None

    def with_details(self, **details: Any) -> APIExceptionBuilder:
        """Fügt zusätzliche Details zur Exception hinzu.

        Args:
            **details: Zusätzliche Metadaten für die Exception

        Returns:
            Self für Method-Chaining
        """
        self._details.update(details)
        return self

    def with_headers(self, headers: dict[str, str]) -> APIExceptionBuilder:
        """Fügt HTTP-Headers zur Exception hinzu.

        Args:
            headers: HTTP-Headers für die Response

        Returns:
            Self für Method-Chaining
        """
        self._headers = headers
        return self

    def with_field_error(self, field: str, error: str) -> APIExceptionBuilder:
        """Fügt Feld-spezifischen Validierungsfehler hinzu.

        Args:
            field: Name des fehlerhaften Feldes
            error: Fehlerbeschreibung

        Returns:
            Self für Method-Chaining
        """
        if "field_errors" not in self._details:
            self._details["field_errors"] = {}
        self._details["field_errors"][field] = error
        return self

    def with_suggestion(self, suggestion: str) -> APIExceptionBuilder:
        """Fügt Lösungsvorschlag zur Exception hinzu.

        Args:
            suggestion: Vorschlag zur Fehlerbehebung

        Returns:
            Self für Method-Chaining
        """
        self._details["suggestion"] = suggestion
        return self

    def build(self) -> HTTPException:
        """Erstellt die finale HTTPException.

        Returns:
            Konfigurierte HTTPException
        """
        detail = {
            "error_code": self._error_code,
            "message": self._message,
            **self._details
        }

        exception = HTTPException(
            status_code=self._status_code,
            detail=detail,
            headers=self._headers
        )

        # Logging für Debugging
        logger.debug(
            f"HTTP Exception erstellt: {self._status_code} - {self._error_code} - {self._message}",
            extra={"details": self._details}
        )

        return exception


def not_found_error(
    message: str = "Ressource nicht gefunden",
    resource_type: str | None = None,
    resource_id: str | None = None
) -> HTTPException:
    """Erstellt standardisierte 404 Not Found Exception.

    Args:
        message: Benutzerfreundliche Fehlermeldung
        resource_type: Typ der nicht gefundenen Ressource
        resource_id: ID der nicht gefundenen Ressource

    Returns:
        HTTPException mit 404 Status
    """
    builder = APIExceptionBuilder(
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=ErrorCodes.NOT_FOUND,
        message=message
    )

    if resource_type:
        builder.with_details(resource_type=resource_type)
    if resource_id:
        builder.with_details(resource_id=resource_id)

    return builder.build()


def conflict_error(
    message: str = "Ressource existiert bereits",
    conflicting_field: str | None = None,
    conflicting_value: str | None = None
) -> HTTPException:
    """Erstellt standardisierte 409 Conflict Exception.

    Args:
        message: Benutzerfreundliche Fehlermeldung
        conflicting_field: Name des konfliktverursachenden Feldes
        conflicting_value: Wert des konfliktverursachenden Feldes

    Returns:
        HTTPException mit 409 Status
    """
    builder = APIExceptionBuilder(
        status_code=status.HTTP_409_CONFLICT,
        error_code=ErrorCodes.ALREADY_EXISTS,
        message=message
    )

    if conflicting_field:
        builder.with_details(conflicting_field=conflicting_field)
    if conflicting_value:
        builder.with_details(conflicting_value=conflicting_value)

    return builder.build()


def name_exists_error(
    name: str,
    resource_type: str = "Ressource"
) -> HTTPException:
    """Erstellt spezifische Exception für bereits existierende Namen.

    Args:
        name: Der bereits existierende Name
        resource_type: Typ der Ressource

    Returns:
        HTTPException mit 409 Status
    """
    return APIExceptionBuilder(
        status_code=status.HTTP_409_CONFLICT,
        error_code=ErrorCodes.NAME_EXISTS,
        message=f"{resource_type} mit diesem Namen existiert bereits"
    ).with_details(
        conflicting_field="name",
        conflicting_value=name
    ).with_suggestion(
        f"Wählen Sie einen anderen Namen für die {resource_type.lower()}"
    ).build()


def validation_error(
    message: str = "Validierungsfehler",
    field_errors: dict[str, str] | None = None
) -> HTTPException:
    """Erstellt standardisierte 422 Validation Exception.

    Args:
        message: Benutzerfreundliche Fehlermeldung
        field_errors: Dictionary mit Feld-spezifischen Fehlern

    Returns:
        HTTPException mit 422 Status
    """
    builder = APIExceptionBuilder(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code=ErrorCodes.VALIDATION_ERROR,
        message=message
    )

    if field_errors:
        builder.with_details(field_errors=field_errors)

    return builder.build()


def internal_server_error(
    message: str = "Interner Serverfehler",
    error_id: str | None = None
) -> HTTPException:
    """Erstellt standardisierte 500 Internal Server Error Exception.

    Args:
        message: Benutzerfreundliche Fehlermeldung
        error_id: Eindeutige Error-ID für Debugging

    Returns:
        HTTPException mit 500 Status
    """
    builder = APIExceptionBuilder(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code=ErrorCodes.INTERNAL_ERROR,
        message=message
    )

    if error_id:
        builder.with_details(error_id=error_id)

    return builder.build()


def rate_limit_error(
    message: str = "Rate Limit überschritten",
    retry_after: int | None = None,
    limit: int | None = None,
    window: str | None = None
) -> HTTPException:
    """Erstellt standardisierte 429 Rate Limit Exception.

    Args:
        message: Benutzerfreundliche Fehlermeldung
        retry_after: Sekunden bis zum nächsten Versuch
        limit: Anzahl erlaubter Requests
        window: Zeitfenster für das Limit

    Returns:
        HTTPException mit 429 Status
    """
    builder = APIExceptionBuilder(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error_code=ErrorCodes.RATE_LIMIT_EXCEEDED,
        message=message
    )

    headers = {}
    if retry_after:
        headers["Retry-After"] = str(retry_after)
        builder.with_details(retry_after_seconds=retry_after)

    if limit:
        builder.with_details(limit=limit)
    if window:
        builder.with_details(window=window)

    if headers:
        builder.with_headers(headers)

    return builder.build()


def business_logic_error(
    message: str,
    rule: str | None = None,
    context: dict[str, Any] | None = None
) -> HTTPException:
    """Erstellt Exception für Business Logic Verletzungen.

    Args:
        message: Benutzerfreundliche Fehlermeldung
        rule: Name der verletzten Business Rule
        context: Zusätzlicher Kontext für den Fehler

    Returns:
        HTTPException mit 422 Status
    """
    builder = APIExceptionBuilder(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code=ErrorCodes.BUSINESS_RULE_VIOLATION,
        message=message
    )

    if rule:
        builder.with_details(violated_rule=rule)
    if context:
        builder.with_details(context=context)

    return builder.build()


# Convenience-Funktionen für häufige Exception-Patterns
def resource_not_found(resource_type: str, resource_id: str | int) -> HTTPException:
    """Convenience-Funktion für Resource Not Found Errors."""
    return not_found_error(
        message=f"{resource_type} nicht gefunden",
        resource_type=resource_type,
        resource_id=str(resource_id)
    )


def duplicate_name(name: str, resource_type: str = "Ressource") -> HTTPException:
    """Convenience-Funktion für Duplicate Name Errors."""
    return name_exists_error(name, resource_type)


def invalid_field(field: str, value: Any, reason: str) -> HTTPException:
    """Convenience-Funktion für Invalid Field Errors."""
    return validation_error(
        message=f"Ungültiger Wert für Feld '{field}'",
        field_errors={field: f"{reason}. Erhaltener Wert: {value}"}
    )
