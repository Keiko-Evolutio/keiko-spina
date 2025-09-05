# backend/data_models/api.py
"""API Models für Request/Response-Strukturen und Error Handling."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    CATEGORY_BUSINESS,
    CATEGORY_CREATIVE,
    # Category Constants
    CATEGORY_GENERAL,
    CATEGORY_TECHNICAL,
    # Pagination Constants
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    DEFAULT_SORT_FIELD,
    DEFAULT_SORT_ORDER,
    DEFAULT_VOICE_PITCH,
    # Voice Settings Constants
    DEFAULT_VOICE_SPEED,
    DEFAULT_VOICE_VOLUME,
    ERROR_CODE_CONFLICT,
    ERROR_CODE_INTERNAL_SERVER_ERROR,
    ERROR_CODE_NOT_FOUND,
    ERROR_CODE_RATE_LIMIT_EXCEEDED,
    # Error Code Constants
    ERROR_CODE_VALIDATION_ERROR,
    MAX_NAME_LENGTH,
    MAX_PAGE_SIZE,
    MAX_VOICE_PITCH,
    MAX_VOICE_SPEED,
    MAX_VOICE_VOLUME,
    # Field Length Constants
    MIN_NAME_LENGTH,
    MIN_PAGE_SIZE,
    MIN_SYSTEM_MESSAGE_LENGTH,
    MIN_VOICE_PITCH,
    MIN_VOICE_SPEED,
    MIN_VOICE_VOLUME,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    # Error Severity Constants
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SORT_ORDER_PATTERN,
    # Success Messages
    SUCCESS_MSG_OPERATION_SUCCESSFUL,
)
from .utils import utc_now

# Enums

class ErrorSeverity(str, Enum):
    """Fehler-Schweregrade."""
    LOW = SEVERITY_LOW
    MEDIUM = SEVERITY_MEDIUM
    HIGH = SEVERITY_HIGH
    CRITICAL = SEVERITY_CRITICAL


class AgentConfigurationCategory(str, Enum):
    """Kategorien für Agent-Konfigurationen."""
    GENERAL = CATEGORY_GENERAL
    TECHNICAL = CATEGORY_TECHNICAL
    CREATIVE = CATEGORY_CREATIVE
    BUSINESS = CATEGORY_BUSINESS

# Request Models

class VoiceSettings(BaseModel):
    """Voice-Konfiguration für TTS."""
    voice_id: str = Field(..., description="Voice-ID")
    speed: float = Field(
        default=DEFAULT_VOICE_SPEED,
        ge=MIN_VOICE_SPEED,
        le=MAX_VOICE_SPEED,
        description="Sprechgeschwindigkeit"
    )
    pitch: float = Field(
        default=DEFAULT_VOICE_PITCH,
        ge=MIN_VOICE_PITCH,
        le=MAX_VOICE_PITCH,
        description="Tonhöhe"
    )
    volume: float = Field(
        default=DEFAULT_VOICE_VOLUME,
        ge=MIN_VOICE_VOLUME,
        le=MAX_VOICE_VOLUME,
        description="Lautstärke"
    )


class ToolConfiguration(BaseModel):
    """Tool-Konfiguration."""
    name: str = Field(..., description="Tool-Name")
    enabled: bool = Field(default=True, description="Tool aktiviert")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool-Parameter")


class AgentConfigurationCreateRequest(BaseModel):
    """Request zum Erstellen einer Agent-Konfiguration."""
    name: str = Field(
        ...,
        min_length=MIN_NAME_LENGTH,
        max_length=MAX_NAME_LENGTH,
        description="Eindeutiger Name"
    )
    category: AgentConfigurationCategory = Field(..., description="Kategorie")
    system_message: str = Field(
        ...,
        min_length=MIN_SYSTEM_MESSAGE_LENGTH,
        description="System-Prompt"
    )
    is_default: bool = Field(default=False, description="Als Standard setzen")
    voice_settings: VoiceSettings | None = Field(None, description="Voice-Einstellungen")
    tools: list[ToolConfiguration] = Field(default_factory=list, description="Tool-Konfigurationen")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Keiko Assistant",
                "category": "general",
                "system_message": "Du bist Keiko, ein hilfsbereit persönlicher Assistent.",
                "is_default": True,
                "voice_settings": {
                    "voice_id": "nova",
                    "speed": 1.0,
                    "pitch": 0.0,
                    "volume": 1.0
                },
                "tools": [
                    {
                        "name": "web_search",
                        "enabled": True,
                        "parameters": {}
                    }
                ]
            }
        }
    )


class AgentConfigurationUpdateRequest(BaseModel):
    """Request zum Aktualisieren einer Agent-Konfiguration."""
    name: str | None = Field(
        None,
        min_length=MIN_NAME_LENGTH,
        max_length=MAX_NAME_LENGTH,
        description="Neuer Name"
    )
    category: AgentConfigurationCategory | None = Field(None, description="Neue Kategorie")
    system_message: str | None = Field(
        None,
        min_length=MIN_SYSTEM_MESSAGE_LENGTH,
        description="Neuer System-Prompt"
    )
    is_default: bool | None = Field(None, description="Standard-Status ändern")
    voice_settings: VoiceSettings | None = Field(None, description="Neue Voice-Einstellungen")
    tools: list[ToolConfiguration] | None = Field(None, description="Neue Tool-Konfigurationen")


class PaginationRequest(BaseModel):
    """Request für Paginierung."""
    page: int = Field(default=DEFAULT_PAGE, ge=MIN_PAGE_SIZE, description="Seiten-Nummer")
    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=MIN_PAGE_SIZE,
        le=MAX_PAGE_SIZE,
        description="Einträge pro Seite"
    )
    sort_by: str | None = Field(default=DEFAULT_SORT_FIELD, description="Sortierfeld")
    sort_order: str = Field(
        default=DEFAULT_SORT_ORDER,
        pattern=SORT_ORDER_PATTERN,
        description="Sortier-Reihenfolge"
    )


# Response Models

class BaseResponse(BaseModel):
    """Basis-Response für alle API-Antworten."""
    success: bool = Field(default=True, description="Erfolgsindikator")
    timestamp: datetime = Field(default_factory=utc_now, description="Zeitstempel")
    request_id: str | None = Field(None, description="Request-ID für Tracing")


class PaginationMeta(BaseModel):
    """Paginierungs-Metadaten."""
    page: int = Field(..., description="Aktuelle Seite")
    page_size: int = Field(..., description="Einträge pro Seite")
    total_items: int = Field(..., description="Gesamtanzahl Einträge")
    total_pages: int = Field(..., description="Gesamtanzahl Seiten")
    has_next: bool = Field(..., description="Weitere Seiten verfügbar")
    has_previous: bool = Field(..., description="Vorherige Seiten verfügbar")


class SuccessResponse(BaseResponse):
    """Erfolgreiche Operation ohne spezifische Daten."""
    message: str = Field(default=SUCCESS_MSG_OPERATION_SUCCESSFUL, description="Erfolgs-Nachricht")


class AgentConfigurationResponse(BaseResponse):
    """Response für Agent-Konfiguration."""
    configuration: dict[str, Any] = Field(..., description="Konfigurationsdaten")


class PaginatedResponse(BaseResponse):
    """Basis für paginierte Responses."""
    meta: PaginationMeta = Field(..., description="Paginierungs-Metadaten")


class AgentConfigurationListResponse(PaginatedResponse):
    """Response für Konfigurationsliste."""
    configurations: list[dict[str, Any]] = Field(..., description="Konfigurationsliste")


class ConfigurationStatsResponse(BaseResponse):
    """Response für Konfigurationsstatistiken."""
    stats: dict[str, Any] = Field(..., description="Statistiken")


# Error Models

class ValidationDetail(BaseModel):
    """Validierungsfehler-Details."""
    field: str = Field(..., description="Feldname mit Fehler")
    message: str = Field(..., description="Fehlerbeschreibung")
    invalid_value: Any = Field(None, description="Ungültiger Wert")


class APIError(BaseModel):
    """Basis-Error-Klasse mit allen wichtigen Feldern."""
    error_code: str = Field(..., description="Maschinen-lesbarer Fehlercode")
    message: str = Field(..., description="Menschenlesbare Fehlermeldung")
    severity: ErrorSeverity = Field(default=ErrorSeverity.MEDIUM, description="Schweregrad")
    timestamp: datetime = Field(default_factory=utc_now, description="Zeitstempel")
    request_id: str | None = Field(None, description="Request-ID für Tracing")
    details: list[ValidationDetail] | None = Field(None, description="Detaillierte Fehler")


class ValidationError(APIError):
    """Validierungsfehler - spezialisierte Error-Klasse."""
    error_code: str = Field(default=ERROR_CODE_VALIDATION_ERROR, description="Validierungsfehlercode")
    details: list[ValidationDetail] = Field(default_factory=list, description="Validierungsdetails")


class NotFoundError(APIError):
    """Ressource nicht gefunden."""
    error_code: str = Field(default=ERROR_CODE_NOT_FOUND, description="Not-Found-Fehlercode")
    resource_type: str | None = Field(None, description="Typ der nicht gefundenen Ressource")
    resource_id: str | None = Field(None, description="ID der nicht gefundenen Ressource")


class ConflictError(APIError):
    """Ressourcenkonflikt."""
    error_code: str = Field(default=ERROR_CODE_CONFLICT, description="Konfliktfehlercode")
    conflicting_field: str | None = Field(None, description="Konfliktverursachendes Feld")


class RateLimitError(APIError):
    """Rate-Limit-Fehler."""
    error_code: str = Field(default=ERROR_CODE_RATE_LIMIT_EXCEEDED, description="Rate-Limit-Fehlercode")
    limit: int | None = Field(None, description="Request-Limit")
    remaining: int | None = Field(None, description="Verbleibende Requests")
    reset_time: datetime | None = Field(None, description="Reset-Zeit")
    retry_after: int | None = Field(None, description="Retry-After in Sekunden")


class InternalServerError(APIError):
    """Interner Serverfehler."""
    error_code: str = Field(default=ERROR_CODE_INTERNAL_SERVER_ERROR, description="Serverfehlercode")
    severity: ErrorSeverity = Field(default=ErrorSeverity.HIGH, description="Hoher Schweregrad")
    incident_id: str | None = Field(None, description="Incident-ID für Support")


class ErrorResponse(BaseModel):
    """Einheitliche Error-Response für alle API-Fehler."""
    success: bool = Field(default=False, description="Erfolgsindikator")
    error: APIError = Field(..., description="Fehler-Details")
    timestamp: datetime = Field(default_factory=utc_now, description="Zeitstempel")
    request_id: str | None = Field(None, description="Request-ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": {
                    "error_code": "VALIDATION_ERROR",
                    "message": "Die übermittelten Daten sind ungültig",
                    "severity": "medium",
                    "timestamp": "2025-08-05T19:00:00Z",
                    "request_id": "req_12345"
                },
                "timestamp": "2025-08-05T19:00:00Z",
                "request_id": "req_12345"
            }
        }
    )
