"""Unified Error-Response-Builder für konsistente Fehlerbehandlung.

Konsolidiert alle duplizierte Response-Building-Logik aus:
- backend/api/errors/agent_errors.py::build_error_detail()
- backend/api/common/error_handlers.py::create_error_response()
- backend/core/error_handler.py::build_response()

Eliminiert Code-Duplikation und stellt einheitliche API bereit.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent_errors import (
    ErrorCodes,
    HTTPStatusCodes,
    StandardError,
    get_error_definition,
    get_http_status_for_error,
)

# ============================================================================
# RESPONSE-MODELS (Konsolidiert aus error_handlers.py)
# ============================================================================

@dataclass(frozen=True)
class ErrorContext:
    """Kontext-Informationen für Error-Responses."""

    trace_id: str | None = None
    correlation_id: str | None = None
    request_id: str | None = None
    route: str | None = None
    method: str | None = None
    tenant: str | None = None
    user: str | None = None
    timestamp: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class RecoveryStrategy:
    """Recovery-Strategie für Error-Responses."""

    retryable: bool = False
    retry_after_seconds: int | None = None
    max_retries: int | None = None
    backoff_strategy: str | None = None


class ErrorDetail(BaseModel):
    """Detaillierte Fehlerinformation für Pydantic-Responses."""

    field: str | None = Field(None, description="Betroffenes Feld")
    message: str = Field(..., description="Fehlermeldung")
    code: str | None = Field(None, description="Fehlercode")


class StandardErrorResponse(BaseModel):
    """Standardisierte Error-Response für alle API-Endpoints."""

    success: bool = Field(False, description="Erfolgs-Status")
    error: dict[str, Any] = Field(..., description="Error-Details")
    context: dict[str, Any] | None = Field(None, description="Request-Kontext")
    recovery: dict[str, Any] | None = Field(None, description="Recovery-Strategie")
    timestamp: str = Field(..., description="Zeitstempel")


# ============================================================================
# UNIFIED ERROR-RESPONSE-BUILDER
# ============================================================================

class ErrorResponseBuilder:
    """Zentrale Klasse für einheitliche Error-Response-Erstellung.

    Konsolidiert alle Error-Response-Building-Logik und eliminiert
    Code-Duplikation zwischen verschiedenen Error-Handlern.
    """

    def __init__(self, include_details: bool = False) -> None:
        """Initialisiert Error-Response-Builder.

        Args:
            include_details: Detaillierte Fehlerinformationen einschließen
        """
        self.include_details = include_details

    def _generate_correlation_id(self) -> str:
        """Generiert eindeutige Korrelations-ID."""
        return str(uuid.uuid4())

    def _get_current_timestamp(self) -> str:
        """Ruft aktuellen Zeitstempel ab."""
        return datetime.now(UTC).isoformat()

    def _extract_context_from_request(self, request: Request | None) -> ErrorContext:
        """Extrahiert Kontext-Informationen aus Request.

        Args:
            request: HTTP-Request-Objekt

        Returns:
            ErrorContext mit extrahierten Informationen
        """
        if not request:
            return ErrorContext(
                correlation_id=self._generate_correlation_id(),
                timestamp=self._get_current_timestamp()
            )

        # Trace-ID aus Request-State extrahieren
        trace_id = getattr(request.state, "trace_id", None)

        # Auth-Informationen extrahieren
        auth = getattr(request.state, "auth", None)
        tenant = getattr(auth, "tenant", None) if auth else None
        user = getattr(auth, "subject", None) if auth else None

        return ErrorContext(
            trace_id=trace_id,
            correlation_id=self._generate_correlation_id(),
            route=str(request.url.path) if hasattr(request, "url") else None,
            method=str(request.method) if hasattr(request, "method") else None,
            tenant=tenant,
            user=user,
            timestamp=self._get_current_timestamp()
        )

    def _build_error_details(
        self,
        error: Exception | StandardError | str,
        reason: str | None = None,
        **extra: Any
    ) -> dict[str, Any]:
        """Erstellt Error-Details-Dictionary.

        Args:
            error: Error-Objekt, StandardError oder Error-Code
            reason: Optionale spezifische Begründung
            **extra: Zusätzliche Felder

        Returns:
            Error-Details-Dictionary
        """
        if isinstance(error, str):
            # Error-Code als String
            error_def = get_error_definition(error)
            if error_def:
                details = {
                    "code": error_def.code,
                    "message": error_def.message,
                    "category": error_def.category.value,
                    "retryable": error_def.retryable
                }
            else:
                details = {
                    "code": error,
                    "message": "Unbekannter Fehler",
                    "category": "system_error",
                    "retryable": False
                }
        elif isinstance(error, StandardError):
            # StandardError-Objekt
            details = {
                "code": error.code,
                "message": error.message,
                "category": error.category.value,
                "retryable": error.retryable
            }
        elif isinstance(error, HTTPException):
            # FastAPI HTTPException
            details = {
                "code": f"HTTP_{error.status_code}",
                "message": str(error.detail),
                "category": "http_error",
                "retryable": error.status_code >= 500
            }
        else:
            # Generische Exception
            details = {
                "code": ErrorCodes.INTERNAL_ERROR,
                "message": str(error) if str(error) else "Interner Serverfehler",
                "category": "system_error",
                "retryable": True
            }

        # Optionale Felder hinzufügen
        if reason:
            details["reason"] = reason

        if self.include_details and isinstance(error, Exception):
            details["exception_type"] = type(error).__name__
            if hasattr(error, "__traceback__"):
                import traceback
                details["traceback"] = traceback.format_exc()

        if extra:
            details.update(extra)

        return details

    def _build_recovery_strategy(self, error_details: dict[str, Any]) -> RecoveryStrategy:
        """Erstellt Recovery-Strategie basierend auf Error-Details.

        Args:
            error_details: Error-Details-Dictionary

        Returns:
            RecoveryStrategy-Objekt
        """
        retryable = error_details.get("retryable", False)

        if not retryable:
            return RecoveryStrategy(retryable=False)

        # Retry-Parameter basierend auf Error-Code bestimmen
        code = error_details.get("code", "")

        if code in {ErrorCodes.RATE_LIMIT_EXCEEDED, ErrorCodes.BUDGET_EXCEEDED}:
            return RecoveryStrategy(
                retryable=True,
                retry_after_seconds=60,
                max_retries=3,
                backoff_strategy="exponential"
            )
        if code in {ErrorCodes.TIMEOUT, ErrorCodes.DEADLINE_EXCEEDED}:
            return RecoveryStrategy(
                retryable=True,
                retry_after_seconds=5,
                max_retries=5,
                backoff_strategy="linear"
            )
        if code in {ErrorCodes.SERVICE_UNAVAILABLE, ErrorCodes.DEPENDENCY_ERROR}:
            return RecoveryStrategy(
                retryable=True,
                retry_after_seconds=30,
                max_retries=3,
                backoff_strategy="exponential"
            )
        return RecoveryStrategy(
            retryable=True,
            retry_after_seconds=5,
            max_retries=3,
            backoff_strategy="linear"
        )

    def build_error_detail(
        self,
        code: str,
        *,
        reason: str | None = None,
        **extra: Any
    ) -> dict[str, Any]:
        """Erstellt Error-Detail-Dictionary (Legacy-Kompatibilität).

        Args:
            code: Error-Code
            reason: Optionale Begründung
            **extra: Zusätzliche Felder

        Returns:
            Error-Detail-Dictionary
        """
        return self._build_error_details(code, reason=reason, **extra)

    def create_error_response(
        self,
        error: Exception | StandardError | str,
        request: Request | None = None,
        *,
        reason: str | None = None,
        **extra: Any
    ) -> StandardErrorResponse:
        """Erstellt standardisierte Error-Response.

        Konsolidiert create_error_response() aus error_handlers.py
        mit verbesserter Typisierung und Funktionalität.

        Args:
            error: Error-Objekt, StandardError oder Error-Code
            request: HTTP-Request-Objekt
            reason: Optionale spezifische Begründung
            **extra: Zusätzliche Error-Details

        Returns:
            Standardisierte Error-Response
        """
        context = self._extract_context_from_request(request)
        error_details = self._build_error_details(error, reason=reason, **extra)
        recovery = self._build_recovery_strategy(error_details)

        return StandardErrorResponse(
            success=False,
            error=error_details,
            context=asdict(context) if context else None,
            recovery=asdict(recovery) if recovery else None,
            timestamp=context.timestamp or self._get_current_timestamp()
        )

    def build_json_response(
        self,
        error: Exception | StandardError | str,
        request: Request | None = None,
        *,
        status_code: int | None = None,
        reason: str | None = None,
        **extra: Any
    ) -> JSONResponse:
        """Erstellt JSON-Error-Response für FastAPI.

        Konsolidiert build_response() aus error_handler.py und
        handle_standard_exceptions() aus error_handlers.py.

        Args:
            error: Error-Objekt, StandardError oder Error-Code
            request: HTTP-Request-Objekt
            status_code: Optionaler HTTP-Status-Code (überschreibt Standard)
            reason: Optionale spezifische Begründung
            **extra: Zusätzliche Error-Details

        Returns:
            JSONResponse mit Error-Details
        """
        error_response = self.create_error_response(
            error, request, reason=reason, **extra
        )

        # HTTP-Status-Code bestimmen
        if status_code is None:
            if isinstance(error, str):
                status_code = get_http_status_for_error(error)
            elif isinstance(error, StandardError):
                status_code = error.http_status
            elif isinstance(error, HTTPException):
                status_code = error.status_code
            else:
                status_code = HTTPStatusCodes.INTERNAL_SERVER_ERROR

        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )

    def handle_exception(
        self,
        exception: Exception,
        request: Request | None = None,
        operation: str | None = None
    ) -> JSONResponse:
        """Behandelt Exception und erstellt JSON-Response.

        Vereinfachte High-Level-API für Exception-Handling.

        Args:
            exception: Aufgetretene Exception
            request: HTTP-Request-Objekt
            operation: Name der Operation (für Logging)

        Returns:
            JSONResponse mit Error-Details
        """
        # Logging (falls Logger verfügbar)
        try:
            from kei_logging import get_logger
            logger = get_logger(__name__)

            context = self._extract_context_from_request(request)
            logger.error(
                f"Exception in operation '{operation or 'unknown'}': {exception}",
                extra={
                    "correlation_id": context.correlation_id,
                    "trace_id": context.trace_id,
                    "operation": operation,
                    "exception_type": type(exception).__name__
                }
            )
        except ImportError:
            get_logger = None  # type: ignore

        return self.build_json_response(exception, request)


# ============================================================================
# CONVENIENCE-FUNKTIONEN
# ============================================================================

# Globale Instanz für einfache Verwendung
_default_builder = ErrorResponseBuilder()
_detailed_builder = ErrorResponseBuilder(include_details=True)


def build_error_detail(
    code: str,
    *,
    reason: str | None = None,
    **extra: Any
) -> dict[str, Any]:
    """Convenience-Funktion für Error-Detail-Erstellung.

    Ersetzt die ursprüngliche build_error_detail() Funktion.
    """
    return _default_builder.build_error_detail(code, reason=reason, **extra)


def create_error_response(
    error: Exception | StandardError | str,
    request: Request | None = None,
    *,
    include_details: bool = False,
    reason: str | None = None,
    **extra: Any
) -> StandardErrorResponse:
    """Convenience-Funktion für Error-Response-Erstellung.

    Ersetzt die ursprüngliche create_error_response() Funktion.
    """
    builder = _detailed_builder if include_details else _default_builder
    return builder.create_error_response(error, request, reason=reason, **extra)


def handle_standard_exceptions(
    error: Exception,
    request: Request | None = None,
    include_details: bool = False
) -> JSONResponse:
    """Convenience-Funktion für Standard-Exception-Handling.

    Ersetzt die ursprüngliche handle_standard_exceptions() Funktion.
    """
    builder = _detailed_builder if include_details else _default_builder
    return builder.handle_exception(error, request)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Klassen
    "ErrorContext",
    "ErrorDetail",
    "ErrorResponseBuilder",
    "RecoveryStrategy",
    "StandardErrorResponse",
    # Convenience-Funktionen
    "build_error_detail",
    "create_error_response",
    "handle_standard_exceptions",
]
