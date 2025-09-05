"""Strukturierte Error Response Schemas für einheitliche API-Fehlerbehandlung.
Definiert standardisierte JSON-Schemas für alle API-Fehlerantworten.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorSeverity(str, Enum):
    """Schweregrad von Fehlern."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Kategorien von Fehlern für bessere Klassifizierung."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    NETWORK = "network"
    VOICE = "voice"
    WEBHOOK = "webhook"
    RATE_LIMIT = "rate_limit"


class ErrorDetail(BaseModel):
    """Detaillierte Fehlerinformation für spezifische Felder oder Kontexte."""
    field: str | None = Field(None, description="Betroffenes Feld (bei Validierungsfehlern)")
    code: str = Field(..., description="Spezifischer Fehlercode")
    message: str = Field(..., description="Detaillierte Fehlermeldung")
    value: Any | None = Field(None, description="Problematischer Wert (nur in Development)")


class ErrorContext(BaseModel):
    """Kontextuelle Informationen zum Fehler."""
    user_id: str | None = Field(None, description="Benutzer-ID (falls verfügbar)")
    tenant_id: str | None = Field(None, description="Tenant-ID")
    request_path: str | None = Field(None, description="API-Pfad der fehlgeschlagenen Anfrage")
    request_method: str | None = Field(None, description="HTTP-Methode")
    client_ip: str | None = Field(None, description="Client IP-Adresse")
    user_agent: str | None = Field(None, description="User-Agent Header")
    session_id: str | None = Field(None, description="Session-ID (falls verfügbar)")
    correlation_id: str | None = Field(None, description="Request Correlation-ID")


class StructuredErrorResponse(BaseModel):
    """Standardisierte Error Response Struktur für alle API-Fehler."""

    # Basis-Fehlerinformationen
    error_code: str = Field(..., description="Eindeutiger Fehlercode (z.B. VALIDATION_FAILED)")
    message: str = Field(..., description="Benutzerfreundliche Fehlermeldung")

    # Kategorisierung und Schweregrad
    category: ErrorCategory = Field(..., description="Fehlerkategorie")
    severity: ErrorSeverity = Field(ErrorSeverity.MEDIUM, description="Schweregrad des Fehlers")

    # HTTP-spezifische Informationen
    status_code: int = Field(..., description="HTTP-Status-Code")

    # Zeitstempel und Tracking
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC-Zeitstempel des Fehlers")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Eindeutige Trace-ID für Debugging")

    # Detaillierte Informationen
    details: list[ErrorDetail] = Field(default_factory=list, description="Detaillierte Fehlerinformationen")
    context: ErrorContext | None = Field(None, description="Kontextuelle Informationen")

    # Development-spezifische Informationen (nur in Development-Umgebung)
    stack_trace: str | None = Field(None, description="Stack-Trace (nur Development)")
    debug_info: dict[str, Any] | None = Field(None, description="Debug-Informationen (nur Development)")

    # Hilfreiche Informationen für Clients
    help_url: str | None = Field(None, description="URL zur Dokumentation oder Hilfe")
    retry_after: int | None = Field(None, description="Sekunden bis Retry möglich (bei Rate Limiting)")

    # Backward Compatibility
    error: str | None = Field(None, description="Legacy error field für Backward Compatibility")
    detail: str | None = Field(None, description="Legacy detail field für Backward Compatibility")

    def model_post_init(self, __context: Any) -> None:
        """Post-Initialisierung für Backward Compatibility."""
        # Legacy-Felder für Backward Compatibility setzen
        if not self.error:
            self.error = self.error_code
        if not self.detail:
            self.detail = self.message

    class Config:
        """Pydantic Konfiguration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }
        use_enum_values = True


class ValidationErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Validierungsfehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **data
        )


class AuthenticationErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Authentifizierungsfehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            status_code=401,
            **data
        )


class AuthorizationErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Autorisierungsfehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            status_code=403,
            **data
        )


class BusinessLogicErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Business Logic Fehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            **data
        )


class ExternalServiceErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für externe Service-Fehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            **data
        )


class VoiceErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Voice-spezifische Fehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.VOICE,
            severity=ErrorSeverity.MEDIUM,
            **data
        )


class RateLimitErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Rate Limiting Fehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            status_code=429,
            **data
        )


class SystemErrorResponse(StructuredErrorResponse):
    """Spezialisierte Error Response für Systemfehler."""

    def __init__(self, **data):
        super().__init__(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            **data
        )


# Utility-Funktionen für schnelle Error Response Erstellung
def create_validation_error(
    message: str,
    details: list[ErrorDetail] = None,
    **kwargs
) -> ValidationErrorResponse:
    """Erstellt eine standardisierte Validierungsfehler-Response."""
    return ValidationErrorResponse(
        error_code="VALIDATION_FAILED",
        message=message,
        status_code=400,
        details=details or [],
        **kwargs
    )


def create_not_found_error(
    resource: str,
    resource_id: str = None,
    **kwargs
) -> StructuredErrorResponse:
    """Erstellt eine standardisierte 404-Fehler-Response."""
    message = f"{resource} nicht gefunden"
    if resource_id:
        message += f": {resource_id}"

    return StructuredErrorResponse(
        error_code="RESOURCE_NOT_FOUND",
        message=message,
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.LOW,
        status_code=404,
        **kwargs
    )


def create_internal_server_error(
    message: str = "Ein interner Serverfehler ist aufgetreten",
    **kwargs
) -> SystemErrorResponse:
    """Erstellt eine standardisierte 500-Fehler-Response."""
    return SystemErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        message=message,
        **kwargs
    )
