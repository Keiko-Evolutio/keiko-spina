"""Strukturiertes Error Handling Middleware für einheitliche Fehlerbehandlung.
Implementiert standardisierte Error Responses mit strukturierten JSON-Schemas.
"""

import traceback
import uuid
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from api.common.error_schemas import (
    ErrorCategory,
    ErrorDetail,
    ErrorSeverity,
    StructuredErrorResponse,
)
from api.common.structured_exceptions import (
    KeikoBaseException,
    KeikoExternalServiceError,
    KeikoValidationError,
    KeikoVoiceError,
    create_error_context_from_request,
    map_http_exception_to_keiko,
)
from config.settings import get_settings
from kei_logging import get_logger

logger = get_logger(__name__)


class StructuredErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware für strukturierte Fehlerbehandlung mit einheitlichen Error Responses.

    Features:
    - Strukturierte JSON Error Responses
    - Environment-spezifische Detail-Level
    - Automatische Exception-Mapping
    - Logging Integration mit Trace-IDs
    - Backward Compatibility
    """

    def __init__(self, app, include_debug_details: bool = None):
        super().__init__(app)
        self.settings = get_settings()

        # Environment-spezifische Debug-Details
        if include_debug_details is None:
            self.include_debug_details = self.settings.environment == "development"
        else:
            self.include_debug_details = include_debug_details

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Hauptverarbeitungslogik für Error Handling."""
        # Trace-ID für Request-Tracking
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        try:
            # Normale Request-Verarbeitung
            response = await call_next(request)
            return response

        except Exception as exc:
            # Strukturierte Fehlerbehandlung
            return await self._handle_exception(request, exc, trace_id)

    async def _handle_exception(self, request: Request, exc: Exception, trace_id: str) -> JSONResponse:
        """Behandelt Exceptions und erstellt strukturierte Error Responses."""
        # Error Context aus Request erstellen
        error_context = create_error_context_from_request(request)
        error_context.correlation_id = trace_id

        # Exception zu strukturierter Keiko Exception konvertieren
        if isinstance(exc, KeikoBaseException):
            keiko_exc = exc
        elif isinstance(exc, HTTPException):
            keiko_exc = map_http_exception_to_keiko(exc)
        elif isinstance(exc, PydanticValidationError):
            keiko_exc = self._handle_pydantic_validation_error(exc)
        else:
            keiko_exc = self._handle_unexpected_exception(exc)

        # Context zur Exception hinzufügen
        keiko_exc.context = error_context

        # Strukturierte Error Response erstellen
        error_response = keiko_exc.to_structured_response(include_debug=self.include_debug_details)

        # Logging mit strukturierten Metadaten
        await self._log_error(request, keiko_exc, error_response, trace_id)

        # JSON Response erstellen
        return self._create_json_response(error_response)

    def _handle_pydantic_validation_error(self, exc: PydanticValidationError) -> KeikoValidationError:
        """Behandelt Pydantic Validierungsfehler."""
        field_errors = []
        for error in exc.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            field_errors.append(ErrorDetail(
                field=field_path,
                code=error["type"],
                message=error["msg"],
                value=error.get("input") if self.include_debug_details else None
            ))

        return KeikoValidationError(
            message="Validierungsfehler in den Eingabedaten",
            field_errors=field_errors,
            error_code="PYDANTIC_VALIDATION_FAILED"
        )

    def _handle_unexpected_exception(self, exc: Exception) -> KeikoBaseException:
        """Behandelt unerwartete Exceptions."""
        # Spezifische Exception-Typen erkennen
        exc_type = type(exc).__name__
        exc_message = str(exc)

        # Voice-spezifische Fehler erkennen
        if any(keyword in exc_message.lower() for keyword in ["websocket", "azure", "openai", "voice", "audio"]):
            return KeikoVoiceError(
                message=f"Voice-System Fehler: {exc_message}",
                voice_component=exc_type,
                error_code="VOICE_SYSTEM_ERROR"
            )

        # Externe Service-Fehler erkennen
        if any(keyword in exc_message.lower() for keyword in ["connection", "timeout", "service", "api"]):
            return KeikoExternalServiceError(
                message=f"Externer Service Fehler: {exc_message}",
                service_name="unknown",
                error_code="EXTERNAL_SERVICE_ERROR"
            )

        # Allgemeiner Systemfehler
        return KeikoBaseException(
            message="Ein unerwarteter Fehler ist aufgetreten",
            error_code="UNEXPECTED_ERROR",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            status_code=500
        )

    async def _log_error(
        self,
        request: Request,
        keiko_exc: KeikoBaseException,
        _: StructuredErrorResponse,
        trace_id: str
    ) -> None:
        """Loggt Fehler mit strukturierten Metadaten."""
        # Basis-Log-Daten
        log_data = {
            "error_code": keiko_exc.error_code,
            "category": keiko_exc.category.value,
            "severity": keiko_exc.severity.value,
            "status_code": keiko_exc.status_code,
            "trace_id": trace_id,
            "request_path": str(request.url.path),
            "request_method": request.method,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        # User-Context hinzufügen (falls verfügbar)
        if hasattr(request.state, "user_id"):
            log_data["user_id"] = request.state.user_id
        if hasattr(request.state, "tenant_id"):
            log_data["tenant_id"] = request.state.tenant_id

        # Log-Level basierend auf Schweregrad
        if keiko_exc.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🔴 CRITICAL ERROR: {keiko_exc.message}", extra=log_data)
        elif keiko_exc.severity == ErrorSeverity.HIGH:
            logger.error(f"🟠 HIGH SEVERITY ERROR: {keiko_exc.message}", extra=log_data)
        elif keiko_exc.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"🟡 MEDIUM SEVERITY ERROR: {keiko_exc.message}", extra=log_data)
        else:
            logger.info(f"🔵 LOW SEVERITY ERROR: {keiko_exc.message}", extra=log_data)

        # Stack-Trace für kritische Fehler (nur in Development)
        if self.include_debug_details and keiko_exc.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"Stack trace for {trace_id}:\n{traceback.format_exc()}")

    def _create_json_response(self, error_response: StructuredErrorResponse) -> JSONResponse:
        """Erstellt JSON Response aus strukturierter Error Response."""
        # Response-Daten serialisieren
        response_data = error_response.model_dump(exclude_none=True)

        # Zusätzliche Headers für Error Responses
        headers = {
            "X-Error-Code": error_response.error_code,
            "X-Trace-ID": error_response.trace_id,
            "X-Error-Category": error_response.category.value,
        }

        # Rate Limiting Header hinzufügen
        if error_response.retry_after:
            headers["Retry-After"] = str(error_response.retry_after)

        return JSONResponse(
            status_code=error_response.status_code,
            content=response_data,
            headers=headers
        )


class LegacyErrorCompatibilityMiddleware(BaseHTTPMiddleware):
    """Middleware für Backward Compatibility mit bestehenden Error Responses.
    Kann parallel zur strukturierten Error Middleware verwendet werden.
    """

    def __init__(self, app, enable_legacy_format: bool = True):
        super().__init__(app)
        self.enable_legacy_format = enable_legacy_format

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet Requests und fügt Legacy-Kompatibilität hinzu."""
        # Prüfe ob Client Legacy-Format erwartet
        accept_header = request.headers.get("accept", "")
        user_agent = request.headers.get("user-agent", "")

        # Legacy-Clients erkennen (z.B. alte Mobile Apps, spezifische User-Agents)
        is_legacy_client = (
            "legacy" in accept_header.lower() or
            "application/vnd.keiko.v1+json" in accept_header or
            any(legacy_ua in user_agent.lower() for legacy_ua in ["keiko-mobile/1.", "keiko-cli/1."])
        )

        if is_legacy_client and self.enable_legacy_format:
            request.state.use_legacy_error_format = True

        return await call_next(request)


# Factory-Funktionen für Middleware-Setup
def create_structured_error_middleware(include_debug: bool = None):
    """Factory-Funktion für strukturiertes Error Middleware."""
    def middleware_factory(app):
        return StructuredErrorHandlingMiddleware(app, include_debug_details=include_debug)
    return middleware_factory


def create_legacy_compatibility_middleware(enable_legacy: bool = True):
    """Factory-Funktion für Legacy Compatibility Middleware."""
    def middleware_factory(app):
        return LegacyErrorCompatibilityMiddleware(app, enable_legacy_format=enable_legacy)
    return middleware_factory
