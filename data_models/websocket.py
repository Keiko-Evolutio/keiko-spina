# backend/data_models/websocket.py
"""WebSocket Event Models für Real-time Kommunikation."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    # Connection Status Constants
    CONNECTION_STATUS_CONNECTED,
    CONNECTION_STATUS_DISCONNECTED,
    CONNECTION_STATUS_ERROR,
    CONNECTION_STATUS_RECONNECTING,
    # Audio Constants
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_LANGUAGE,
    DEFAULT_SAMPLE_RATE,
    # Event Type Constants
    EVENT_TYPE_AGENT_RESPONSE,
    EVENT_TYPE_CONNECTION_STATUS,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_FUNCTION_CALL,
    EVENT_TYPE_FUNCTION_RESULT,
    EVENT_TYPE_STATUS_UPDATE,
    EVENT_TYPE_VOICE_INPUT,
    EVENT_TYPE_VOICE_OUTPUT,
    MAX_PROGRESS,
    # Message Type Constants
    MESSAGE_TYPE_FUNCTION_CONFIRMATION,
    # Progress Constants
    MIN_PROGRESS,
)
from .utils import generate_short_id, utc_now


class EventType(str, Enum):
    """WebSocket Event-Typen."""
    AGENT_RESPONSE = EVENT_TYPE_AGENT_RESPONSE
    STATUS_UPDATE = EVENT_TYPE_STATUS_UPDATE
    ERROR = EVENT_TYPE_ERROR
    FUNCTION_CALL = EVENT_TYPE_FUNCTION_CALL
    FUNCTION_RESULT = EVENT_TYPE_FUNCTION_RESULT
    VOICE_INPUT = EVENT_TYPE_VOICE_INPUT
    VOICE_OUTPUT = EVENT_TYPE_VOICE_OUTPUT
    CONNECTION_STATUS = EVENT_TYPE_CONNECTION_STATUS
    SYSTEM_HEARTBEAT = "system_heartbeat"

# Base Event

class BaseWebSocketEvent(BaseModel):
    """Basis-Event mit minimalen required Feldern."""
    event_id: str = Field(default_factory=lambda: generate_short_id("evt_"))
    event_type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    correlation_id: str | None = Field(None, description="Korrelations-ID für Event-Tracking")

    # Pydantic v2 Konfiguration
    model_config = ConfigDict(use_enum_values=True)


# Event Classes

class AgentResponseEvent(BaseWebSocketEvent):
    """Agent-Antwort Event."""
    event_type: Literal[EventType.AGENT_RESPONSE] = EventType.AGENT_RESPONSE
    content: str = Field(..., description="Agent-Antwort-Inhalt")
    is_final: bool = Field(default=True, description="Finale Antwort oder Stream-Teil")
    metadata: dict[str, Any] | None = Field(None, description="Zusätzliche Metadaten")


class StatusUpdateEvent(BaseWebSocketEvent):
    """Status-Update Event."""
    event_type: Literal[EventType.STATUS_UPDATE] = EventType.STATUS_UPDATE
    status: str = Field(..., description="Aktueller Status")
    progress: float | None = Field(
        None,
        ge=MIN_PROGRESS,
        le=MAX_PROGRESS,
        description="Fortschritt 0.0-1.0"
    )
    details: str | None = Field(None, description="Status-Details")


class ErrorEvent(BaseWebSocketEvent):
    """Fehler Event."""
    event_type: Literal[EventType.ERROR] = EventType.ERROR
    error_code: str = Field(..., description="Fehlercode")
    message: str = Field(..., description="Fehlermeldung")
    recoverable: bool = Field(default=True, description="Wiederherstellbarer Fehler")
    details: dict[str, Any] | None = Field(None, description="Fehler-Details")


class FunctionCallEvent(BaseWebSocketEvent):
    """Funktionsaufruf Event."""
    event_type: Literal[EventType.FUNCTION_CALL] = EventType.FUNCTION_CALL
    function_name: str = Field(..., description="Name der aufgerufenen Funktion")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Funktions-Argumente")
    requires_confirmation: bool = Field(default=False,
                                        description="Benutzerbestätigung erforderlich")
    description: str | None = Field(None, description="Funktions-Beschreibung")


class FunctionResultEvent(BaseWebSocketEvent):
    """Funktions-Ergebnis Event."""
    event_type: Literal[EventType.FUNCTION_RESULT] = EventType.FUNCTION_RESULT
    function_call_id: str = Field(..., description="ID des ursprünglichen Funktionsaufrufs")
    result: Any = Field(..., description="Funktions-Ergebnis")
    success: bool = Field(..., description="Erfolgreiche Ausführung")
    error_message: str | None = Field(None, description="Fehlermeldung bei Misserfolg")
    execution_time_ms: int | None = Field(None, description="Ausführungszeit in Millisekunden")


class VoiceInputEvent(BaseWebSocketEvent):
    """Sprach-Eingabe Event."""
    event_type: Literal[EventType.VOICE_INPUT] = EventType.VOICE_INPUT
    audio_data: str = Field(..., description="Base64-kodierte Audio-Daten")
    format: str = Field(default=DEFAULT_AUDIO_FORMAT, description="Audio-Format")
    language: str = Field(default=DEFAULT_LANGUAGE, description="Sprache")
    duration_ms: int | None = Field(None, description="Audio-Länge in Millisekunden")
    sample_rate: int = Field(default=DEFAULT_SAMPLE_RATE, description="Sample-Rate")


class VoiceOutputEvent(BaseWebSocketEvent):
    """Sprach-Ausgabe Event."""
    event_type: Literal[EventType.VOICE_OUTPUT] = EventType.VOICE_OUTPUT
    text: str = Field(..., description="Zu sprechender Text")
    voice_settings: dict[str, Any] | None = Field(None, description="Voice-Einstellungen")
    audio_data: str | None = Field(None, description="Base64-kodierte Audio-Daten")


class ConnectionStatusEvent(BaseWebSocketEvent):
    """Verbindungsstatus Event."""
    event_type: Literal[EventType.CONNECTION_STATUS] = EventType.CONNECTION_STATUS
    status: str = Field(
        ...,
        description=f"Verbindungsstatus: {CONNECTION_STATUS_CONNECTED}|{CONNECTION_STATUS_DISCONNECTED}|{CONNECTION_STATUS_RECONNECTING}|{CONNECTION_STATUS_ERROR}"
    )
    connection_info: dict[str, Any] | None = Field(None, description="Verbindungs-Details")
    client_count: int | None = Field(None, description="Anzahl verbundener Clients")


class SystemHeartbeatEvent(BaseWebSocketEvent):
    """System Heartbeat Event für Live-Service-Status."""
    event_type: Literal[EventType.SYSTEM_HEARTBEAT] = EventType.SYSTEM_HEARTBEAT
    timestamp: float = Field(..., description="Heartbeat-Timestamp")
    phase: str = Field(..., description="System-Phase (ready, starting, etc.)")
    overall_status: str = Field(..., description="Gesamtstatus (healthy, degraded, unhealthy)")
    services: dict[str, Any] = Field(..., description="Service-Details")
    summary: dict[str, int] = Field(..., description="Service-Zusammenfassung")
    uptime_seconds: float = Field(..., description="System-Uptime in Sekunden")
    message: str = Field(..., description="Status-Nachricht")


# Union Type für alle Events

WebSocketEvent = Union[
    AgentResponseEvent,
    StatusUpdateEvent,
    ErrorEvent,
    FunctionCallEvent,
    SystemHeartbeatEvent,
    FunctionResultEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
    ConnectionStatusEvent
]


# Client-seitige Nachrichten

class FunctionConfirmationMessage(BaseModel):
    """Funktions-Bestätigung vom Client."""
    message_type: Literal["function_confirmation"] = MESSAGE_TYPE_FUNCTION_CONFIRMATION
    function_call_id: str = Field(..., description="ID des zu bestätigenden Funktionsaufrufs")
    confirmed: bool = Field(..., description="Bestätigung oder Ablehnung")
    user_input: dict[str, Any] | None = Field(None, description="Zusätzliche Benutzereingaben")
    timestamp: datetime = Field(default_factory=utc_now)


# Event Factory Funktionen

def create_agent_response(
    content: str,
    is_final: bool = True,
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None
) -> AgentResponseEvent:
    """Factory-Funktion für AgentResponseEvent."""
    return AgentResponseEvent(
        content=content,
        is_final=is_final,
        correlation_id=correlation_id,
        metadata=metadata
    )


def create_status_update(
    status: str,
    progress: float | None = None,
    details: str | None = None,
    correlation_id: str | None = None
) -> StatusUpdateEvent:
    """Factory-Funktion für StatusUpdateEvent."""
    return StatusUpdateEvent(
        status=status,
        progress=progress,
        details=details,
        correlation_id=correlation_id
    )


