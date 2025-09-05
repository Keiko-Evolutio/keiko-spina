"""Standardisierte Error-Handler und Exception-Klassen für das API-Modul.

Eliminiert Code-Duplikation bei Error-Handling und stellt einheitliche
Error-Response-Formate für alle API-Endpoints bereit.
"""

from __future__ import annotations

import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kei_logging import get_logger

from .constants import ErrorCodes, HTTPStatusCodes

logger = get_logger(__name__)


# ============================================================================
# CUSTOM EXCEPTION CLASSES
# ============================================================================

class APIError(Exception):
    """Basis-Exception für alle API-Fehler."""

    def __init__(
        self,
        message: str,
        status_code: int = HTTPStatusCodes.INTERNAL_SERVER_ERROR,
        error_code: str = ErrorCodes.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None
    ) -> None:
        """Initialisiert API-Fehler.

        Args:
            message: Fehlermeldung
            status_code: HTTP-Status-Code
            error_code: Interner Error-Code
            details: Zusätzliche Fehlerdetails
            correlation_id: Korrelations-ID für Request-Tracking
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())


class ValidationError(APIError):
    """Fehler bei Eingabe-Validierung."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: str | None = None,
        value: Any | None = None,
        correlation_id: str | None = None
    ) -> None:
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
            correlation_id=correlation_id
        )


class NotFoundError(APIError):
    """Fehler wenn Ressource nicht gefunden wurde."""

    def __init__(
        self,
        resource: str,
        identifier: str | None = None,
        correlation_id: str | None = None
    ) -> None:
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"

        details = {"resource": resource}
        if identifier:
            details["identifier"] = identifier

        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details=details,
            correlation_id=correlation_id
        )


class ConflictError(APIError):
    """Fehler bei Ressourcen-Konflikten."""

    def __init__(
        self,
        message: str = "Resource conflict",
        resource: str | None = None,
        correlation_id: str | None = None
    ) -> None:
        details = {}
        if resource:
            details["resource"] = resource

        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            details=details,
            correlation_id=correlation_id
        )


class RateLimitError(APIError):
    """Fehler bei Rate-Limit-Überschreitung."""

    def __init__(
        self,
        retry_after: int | None = None,
        correlation_id: str | None = None
    ) -> None:
        details = {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message="Rate limit exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
            correlation_id=correlation_id
        )


# ============================================================================
# ERROR RESPONSE MODELS
# ============================================================================

class ErrorDetail(BaseModel):
    """Detaillierte Fehlerinformation."""

    field: str | None = Field(None, description="Betroffenes Feld")
    message: str = Field(..., description="Fehlermeldung")
    code: str | None = Field(None, description="Fehlercode")


class ErrorResponse(BaseModel):
    """Standardisierte Error-Response."""

    success: bool = Field(False, description="Erfolgs-Status")
    error: str = Field(..., description="Fehlertyp")
    message: str = Field(..., description="Fehlermeldung")
    details: dict[str, Any] | None = Field(None, description="Zusätzliche Details")
    correlation_id: str = Field(..., description="Korrelations-ID")
    timestamp: str = Field(..., description="Zeitstempel")
    path: str | None = Field(None, description="Request-Pfad")


# ============================================================================
# ERROR HANDLER FUNCTIONS
# ============================================================================

def create_error_response(
    error: APIError | HTTPException | Exception,
    request: Request | None = None,
    include_traceback: bool = False
) -> ErrorResponse:
    """Erstellt standardisierte Error-Response.

    Args:
        error: Exception-Objekt
        request: HTTP-Request (optional)
        include_traceback: Traceback in Response einschließen

    Returns:
        Standardisierte Error-Response
    """
    correlation_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).isoformat()
    path = request.url.path if request else None

    if isinstance(error, APIError):
        return ErrorResponse(
            success=False,
            error=error.error_code,
            message=error.message,
            details=error.details,
            correlation_id=error.correlation_id,
            timestamp=timestamp,
            path=path
        )
    if isinstance(error, HTTPException):
        return ErrorResponse(
            success=False,
            error=f"HTTP_{error.status_code}",
            message=str(error.detail),
            correlation_id=correlation_id,
            timestamp=timestamp,
            path=path
        )
    details = {}
    if include_traceback:
        details["traceback"] = traceback.format_exc()

    return ErrorResponse(
        success=False,
        error="INTERNAL_ERROR",
        message="Internal server error",
        details=details,
        correlation_id=correlation_id,
        timestamp=timestamp,
        path=path
    )


def handle_standard_exceptions(
    error: Exception,
    request: Request | None = None,
    include_details: bool = False
) -> JSONResponse:
    """Behandelt Standard-Exceptions und gibt JSON-Response zurück.

    Args:
        error: Exception-Objekt
        request: HTTP-Request
        include_details: Detaillierte Fehlerinformationen einschließen

    Returns:
        JSON-Error-Response
    """
    error_response = create_error_response(
        error,
        request,
        include_traceback=include_details
    )

    # Status-Code bestimmen
    if isinstance(error, APIError | HTTPException):
        status_code = error.status_code
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # Logging
    logger.error(
        f"API Error: {error_response.error} - {error_response.message}",
        extra={
            "correlation_id": error_response.correlation_id,
            "path": error_response.path,
            "status_code": status_code
        }
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


class StandardErrorHandler:
    """Klasse für standardisierte Error-Behandlung in API-Endpoints."""

    def __init__(self, include_details: bool = False) -> None:
        """Initialisiert Error-Handler.

        Args:
            include_details: Detaillierte Fehlerinformationen einschließen
        """
        self.include_details = include_details
        self.logger = get_logger(self.__class__.__name__)

    def handle_error(
        self,
        error: Exception,
        operation: str,
        correlation_id: str | None = None,
        request: Request | None = None
    ) -> JSONResponse:
        """Behandelt Fehler für einen spezifischen Operation.

        Args:
            error: Exception-Objekt
            operation: Name der Operation
            correlation_id: Korrelations-ID
            request: HTTP-Request

        Returns:
            JSON-Error-Response
        """
        # Korrelations-ID setzen falls nicht vorhanden
        if isinstance(error, APIError) and not error.correlation_id:
            error.correlation_id = correlation_id or str(uuid.uuid4())

        self.logger.error(
            f"Operation '{operation}' fehlgeschlagen: {error}",
            extra={"correlation_id": correlation_id}
        )

        return handle_standard_exceptions(error, request, self.include_details)

    def wrap_endpoint(self, operation: str):
        """Decorator für automatische Error-Behandlung in Endpoints.

        Args:
            operation: Name der Operation

        Returns:
            Decorator-Funktion
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    # Request aus kwargs extrahieren falls vorhanden
                    request = kwargs.get("request")
                    correlation_id = kwargs.get("correlation_id")

                    return self.handle_error(exc, operation, correlation_id, request)
            return wrapper
        return decorator
