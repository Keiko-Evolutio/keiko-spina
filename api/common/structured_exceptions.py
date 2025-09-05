"""Strukturierte Exception-Hierarchie für einheitliche Fehlerbehandlung.
Definiert spezifische Exception-Typen die automatisch zu HTTP-Status-Codes gemappt werden.
"""

from typing import Any

from fastapi import HTTPException, status

from .error_schemas import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    BusinessLogicErrorResponse,
    ErrorCategory,
    ErrorContext,
    ErrorDetail,
    ErrorSeverity,
    ExternalServiceErrorResponse,
    RateLimitErrorResponse,
    StructuredErrorResponse,
    SystemErrorResponse,
    ValidationErrorResponse,
    VoiceErrorResponse,
)


class KeikoBaseException(Exception):
    """Basis-Exception für alle Keiko-spezifischen Fehler."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        status_code: int = 500,
        details: list[ErrorDetail] | None = None,
        context: ErrorContext | None = None,
        help_url: str | None = None,
        retry_after: int | None = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.category = category
        self.severity = severity
        self.status_code = status_code
        self.details = details or []
        self.context = context
        self.help_url = help_url
        self.retry_after = retry_after
        self.extra_data = kwargs

    def to_structured_response(self, include_debug: bool = False) -> StructuredErrorResponse:
        """Konvertiert die Exception zu einer strukturierten Error Response."""
        response_data = {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category,
            "severity": self.severity,
            "status_code": self.status_code,
            "details": self.details,
            "context": self.context,
            "help_url": self.help_url,
            "retry_after": self.retry_after,
            **self.extra_data
        }

        if include_debug:
            import traceback
            response_data["stack_trace"] = traceback.format_exc()
            response_data["debug_info"] = {
                "exception_type": self.__class__.__name__,
                "module": self.__class__.__module__,
            }

        return StructuredErrorResponse(**response_data)

    @staticmethod
    def _get_stack_trace() -> str:
        """Hilfsmethode für Stack-Trace-Extraktion."""
        import traceback
        return traceback.format_exc()


class KeikoValidationError(KeikoBaseException):
    """Validierungsfehler für Eingabedaten."""

    def __init__(self, message: str, field_errors: list[ErrorDetail] | None = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_FAILED",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=field_errors or [],
            **kwargs
        )

    def to_structured_response(self, include_debug: bool = False) -> ValidationErrorResponse:
        """Konvertiert zu einer spezialisierten Validierungsfehler-Response."""
        response_data = {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "context": self.context,
            "help_url": self.help_url,
        }

        if include_debug:
            response_data["stack_trace"] = self._get_stack_trace()

        return ValidationErrorResponse(**response_data)


class KeikoAuthenticationError(KeikoBaseException):
    """Authentifizierungsfehler."""

    def __init__(self, message: str = "Authentifizierung fehlgeschlagen", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            status_code=status.HTTP_401_UNAUTHORIZED,
            **kwargs
        )

    def to_structured_response(self, include_debug: bool = False) -> AuthenticationErrorResponse:
        return AuthenticationErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            context=self.context,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


class KeikoAuthorizationError(KeikoBaseException):
    """Autorisierungsfehler."""

    def __init__(self, message: str = "Unzureichende Berechtigung", required_scope: str | None = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_FAILED",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            status_code=status.HTTP_403_FORBIDDEN,
            **kwargs
        )
        self.required_scope = required_scope

    def to_structured_response(self, include_debug: bool = False) -> AuthorizationErrorResponse:
        debug_info = {"required_scope": self.required_scope} if include_debug and self.required_scope else None
        return AuthorizationErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            context=self.context,
            debug_info=debug_info,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


class KeikoBusinessLogicError(KeikoBaseException):
    """Business Logic Fehler."""

    def __init__(self, message: str, error_code: str = "BUSINESS_LOGIC_FAILED", **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            **kwargs
        )

    def to_structured_response(self, include_debug: bool = False) -> BusinessLogicErrorResponse:
        return BusinessLogicErrorResponse(
            error_code=self.error_code,
            message=self.message,
            status_code=self.status_code,
            details=self.details,
            context=self.context,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


class KeikoExternalServiceError(KeikoBaseException):
    """Fehler bei externen Services (Azure, OpenAI, etc.)."""

    def __init__(self, message: str, service_name: str, error_code: str = "EXTERNAL_SERVICE_FAILED", **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            status_code=status.HTTP_502_BAD_GATEWAY,
            **kwargs
        )
        self.service_name = service_name

    def to_structured_response(self, include_debug: bool = False) -> ExternalServiceErrorResponse:
        debug_info = {"service_name": self.service_name} if include_debug else None
        return ExternalServiceErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            context=self.context,
            debug_info=debug_info,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


class KeikoVoiceError(KeikoBaseException):
    """Voice-spezifische Fehler (Azure OpenAI Realtime API, WebSocket, etc.)."""

    def __init__(self, message: str, voice_component: str | None = None, error_code: str = "VOICE_FAILED", **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VOICE,
            severity=ErrorSeverity.MEDIUM,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            **kwargs
        )
        self.voice_component = voice_component

    def to_structured_response(self, include_debug: bool = False) -> VoiceErrorResponse:
        debug_info = {"voice_component": self.voice_component} if include_debug and self.voice_component else None
        return VoiceErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            context=self.context,
            debug_info=debug_info,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


class KeikoRateLimitError(KeikoBaseException):
    """Rate Limiting Fehler."""

    def __init__(self, message: str = "Rate Limit überschritten", retry_after: int = 60, **kwargs):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            retry_after=retry_after,
            **kwargs
        )

    def to_structured_response(self, include_debug: bool = False) -> RateLimitErrorResponse:
        return RateLimitErrorResponse(
            error_code=self.error_code,
            message=self.message,
            retry_after=self.retry_after,
            details=self.details,
            context=self.context,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


class KeikoResourceNotFoundError(KeikoBaseException):
    """Resource nicht gefunden Fehler."""

    def __init__(self, resource_type: str, resource_id: str | None = None, **kwargs):
        message = f"{resource_type} nicht gefunden"
        if resource_id:
            message += f": {resource_id}"

        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.LOW,
            status_code=status.HTTP_404_NOT_FOUND,
            **kwargs
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class KeikoSystemError(KeikoBaseException):
    """Systemfehler für unerwartete interne Fehler."""

    def __init__(self, message: str = "Ein interner Systemfehler ist aufgetreten", **kwargs):
        super().__init__(
            message=message,
            error_code="INTERNAL_SYSTEM_ERROR",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            **kwargs
        )

    def to_structured_response(self, include_debug: bool = False) -> SystemErrorResponse:
        return SystemErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            context=self.context,
            stack_trace=self._get_stack_trace() if include_debug else None
        )


# Utility-Funktionen für Exception-Mapping
def map_http_exception_to_keiko(exc: HTTPException) -> KeikoBaseException:
    """Mappt FastAPI HTTPException zu Keiko Exception."""
    status_code = exc.status_code
    message = str(exc.detail) if exc.detail else "HTTP Fehler"

    if status_code == 400:
        return KeikoValidationError(message)
    if status_code == 401:
        return KeikoAuthenticationError(message)
    if status_code == 403:
        return KeikoAuthorizationError(message)
    if status_code == 404:
        return KeikoResourceNotFoundError("Resource", message)
    if status_code == 422:
        return KeikoBusinessLogicError(message)
    if status_code == 429:
        return KeikoRateLimitError(message)
    if status_code >= 500:
        return KeikoSystemError(message)
    return KeikoBaseException(
        message=message,
        status_code=status_code,
        error_code=f"HTTP_{status_code}"
    )


def create_error_context_from_request(request: Any) -> ErrorContext:
    """Erstellt ErrorContext aus FastAPI Request."""
    return ErrorContext(
        request_path=str(request.url.path) if hasattr(request, "url") else None,
        request_method=request.method if hasattr(request, "method") else None,
        client_ip=request.client.host if hasattr(request, "client") and request.client else None,
        user_agent=request.headers.get("user-agent") if hasattr(request, "headers") else None,
        correlation_id=getattr(request.state, "correlation_id", None) if hasattr(request, "state") else None,
        user_id=getattr(request.state, "user_id", None) if hasattr(request, "state") else None,
        tenant_id=getattr(request.state, "tenant_id", None) if hasattr(request, "state") else None,
        session_id=getattr(request.state, "session_id", None) if hasattr(request, "state") else None,
    )
