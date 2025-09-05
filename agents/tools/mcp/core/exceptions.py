"""KEI MCP Exception-Hierarchie.

Einheitliche Exception-Klassen für das KEI MCP System mit
strukturierter Fehlerbehandlung und detailliertem Error-Reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .constants import ERROR_CODES, ERROR_MESSAGES


@dataclass
class ErrorContext:
    """Kontext-Informationen für Fehler."""

    server_name: str | None = None
    tool_name: str | None = None
    resource_name: str | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    additional_data: dict[str, Any] = field(default_factory=dict)


class KEIMCPError(Exception):
    """Basis-Exception für alle KEI MCP Fehler."""

    def __init__(
        self,
        message: str,
        error_code: int | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        """Initialisiert KEI MCP Error.

        Args:
            message: Fehlermeldung
            error_code: Numerischer Error-Code
            context: Fehler-Kontext
            cause: Ursprüngliche Exception (falls vorhanden)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext()
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu Dictionary für Serialisierung."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": {
                "server_name": self.context.server_name,
                "tool_name": self.context.tool_name,
                "resource_name": self.context.resource_name,
                "request_id": self.context.request_id,
                "correlation_id": self.context.correlation_id,
                "timestamp": self.context.timestamp.isoformat(),
                "additional_data": self.context.additional_data,
            },
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String-Repräsentation der Exception."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"(Code: {self.error_code})")

        if self.context.server_name:
            parts.append(f"[Server: {self.context.server_name}]")

        if self.context.tool_name:
            parts.append(f"[Tool: {self.context.tool_name}]")

        if self.context.request_id:
            parts.append(f"[Request: {self.context.request_id}]")

        return " ".join(parts)


class KEIMCPConnectionError(KEIMCPError):
    """Fehler bei der Verbindung zu MCP Servern."""

    def __init__(
        self,
        message: str | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        super().__init__(
            message or ERROR_MESSAGES["CONNECTION_FAILED"],
            ERROR_CODES["CONNECTION_FAILED"],
            context,
            cause
        )


class KEIMCPTimeoutError(KEIMCPError):
    """Timeout-Fehler bei MCP Requests."""

    def __init__(
        self,
        message: str | None = None,
        timeout_seconds: float | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        msg = message or ERROR_MESSAGES["TIMEOUT"]
        if timeout_seconds:
            msg += f" (Timeout: {timeout_seconds}s)"

        super().__init__(
            msg,
            ERROR_CODES["TIMEOUT"],
            context,
            cause
        )
        self.timeout_seconds = timeout_seconds


class KEIMCPAuthenticationError(KEIMCPError):
    """Authentifizierungs-Fehler bei MCP Servern."""

    def __init__(
        self,
        message: str | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        super().__init__(
            message or ERROR_MESSAGES["AUTHENTICATION_FAILED"],
            ERROR_CODES["AUTHENTICATION_FAILED"],
            context,
            cause
        )


class KEIMCPValidationError(KEIMCPError):
    """Schema-Validierungs-Fehler."""

    def __init__(
        self,
        message: str | None = None,
        validation_errors: list[str] | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        msg = message or ERROR_MESSAGES["VALIDATION_FAILED"]
        if validation_errors:
            msg += f": {', '.join(validation_errors)}"

        super().__init__(
            msg,
            ERROR_CODES["VALIDATION_FAILED"],
            context,
            cause
        )
        self.validation_errors = validation_errors or []


class KEIMCPServerError(KEIMCPError):
    """Server-seitige Fehler."""

    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        msg = message or ERROR_MESSAGES["SERVER_ERROR"]
        if status_code:
            msg += f" (HTTP {status_code})"

        super().__init__(
            msg,
            ERROR_CODES["SERVER_ERROR"],
            context,
            cause
        )
        self.status_code = status_code
        self.response_body = response_body


class KEIMCPCircuitBreakerError(KEIMCPError):
    """Circuit Breaker ist geöffnet."""

    def __init__(
        self,
        message: str | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        super().__init__(
            message or ERROR_MESSAGES["CIRCUIT_BREAKER_OPEN"],
            ERROR_CODES["CIRCUIT_BREAKER_OPEN"],
            context,
            cause
        )


class KEIMCPRateLimitError(KEIMCPError):
    """Rate Limit überschritten."""

    def __init__(
        self,
        message: str | None = None,
        retry_after_seconds: float | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None
    ):
        msg = message or ERROR_MESSAGES["RATE_LIMIT_EXCEEDED"]
        if retry_after_seconds:
            msg += f" (Retry after: {retry_after_seconds}s)"

        super().__init__(
            msg,
            ERROR_CODES["RATE_LIMIT_EXCEEDED"],
            context,
            cause
        )
        self.retry_after_seconds = retry_after_seconds


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_error_context(
    server_name: str | None = None,
    tool_name: str | None = None,
    resource_name: str | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
    **additional_data: Any
) -> ErrorContext:
    """Erstellt Error-Context mit gegebenen Parametern."""
    return ErrorContext(
        server_name=server_name,
        tool_name=tool_name,
        resource_name=resource_name,
        request_id=request_id,
        correlation_id=correlation_id,
        additional_data=additional_data
    )


def handle_http_error(
    response_status: int,
    response_body: str | None = None,
    context: ErrorContext | None = None
) -> KEIMCPError:
    """Konvertiert HTTP-Status-Code zu entsprechender KEI MCP Exception."""
    if response_status == 401:
        return KEIMCPAuthenticationError(context=context)
    if response_status == 408:
        return KEIMCPTimeoutError(context=context)
    if response_status == 429:
        return KEIMCPRateLimitError(context=context)
    if 400 <= response_status < 500:
        return KEIMCPValidationError(
            f"Client-Fehler: HTTP {response_status}",
            context=context
        )
    if 500 <= response_status < 600:
        return KEIMCPServerError(
            status_code=response_status,
            response_body=response_body,
            context=context
        )
    return KEIMCPError(
        f"Unbekannter HTTP-Status: {response_status}",
        context=context
    )
