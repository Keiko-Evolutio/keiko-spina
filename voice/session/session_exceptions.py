"""Exception-Hierarchie für Voice Session Management.

Spezifische Exception-Types für strukturierte Fehlerbehandlung
im Voice Session System.
"""

from __future__ import annotations

from typing import Any


class VoiceSessionError(Exception):
    """Basis-Exception für alle Voice Session Fehler."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu Dictionary für Logging/API."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String-Representation mit Error-Code."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# =============================================================================
# Session-Management Errors
# =============================================================================

class SessionNotAvailableError(VoiceSessionError):
    """Realtime-Session ist nicht verfügbar oder nicht initialisiert."""

    def __init__(self, message: str = "Realtime-Session nicht verfügbar") -> None:
        super().__init__(
            message=message,
            error_code="SESSION_NOT_AVAILABLE",
            details={"category": "session_management"}
        )


class SessionConfigurationError(VoiceSessionError):
    """Fehler bei Session-Konfiguration."""

    def __init__(
        self,
        message: str,
        config_field: str | None = None,
        config_value: Any = None,
    ) -> None:
        details = {"category": "session_configuration"}
        if config_field:
            details["config_field"] = config_field
        if config_value is not None:
            details["config_value"] = str(config_value)

        super().__init__(
            message=message,
            error_code="SESSION_CONFIG_ERROR",
            details=details
        )


class SessionUpdateError(VoiceSessionError):
    """Fehler beim Aktualisieren der Session-Konfiguration."""

    def __init__(
        self,
        message: str = "Fehler beim Session-Update",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="SESSION_UPDATE_ERROR",
            details={"category": "session_management"},
            cause=cause
        )


# =============================================================================
# Audio-Processing Errors
# =============================================================================

class AudioProcessingError(VoiceSessionError):
    """Basis-Exception für Audio-Processing-Fehler."""

    def __init__(
        self,
        message: str,
        audio_operation: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"category": "audio_processing"}
        if audio_operation:
            details["audio_operation"] = audio_operation

        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            details=details,
            cause=cause
        )


class AudioSendingError(AudioProcessingError):
    """Fehler beim Senden von Audio-Daten."""

    def __init__(
        self,
        message: str = "Fehler beim Audio-Sending",
        audio_size: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"audio_operation": "send_audio"}
        if audio_size is not None:
            details["audio_size_bytes"] = audio_size

        super().__init__(
            message=message,
            audio_operation="send_audio",
            cause=cause
        )
        self.details.update(details)
        self.error_code = "AUDIO_SENDING_ERROR"


class ResponseStartError(AudioProcessingError):
    """Fehler beim Starten einer Response."""

    def __init__(
        self,
        message: str = "Fehler beim Response-Start",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            audio_operation="start_response",
            cause=cause
        )
        self.error_code = "RESPONSE_START_ERROR"


# =============================================================================
# Event-Handling Errors
# =============================================================================

class EventHandlingError(VoiceSessionError):
    """Basis-Exception für Event-Handling-Fehler."""

    def __init__(
        self,
        message: str,
        event_type: str | None = None,
        event_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"category": "event_handling"}
        if event_type:
            details["event_type"] = event_type
        if event_id:
            details["event_id"] = event_id

        super().__init__(
            message=message,
            error_code="EVENT_HANDLING_ERROR",
            details=details,
            cause=cause
        )


class EventProcessingError(EventHandlingError):
    """Fehler bei der Verarbeitung von Events."""

    def __init__(
        self,
        message: str,
        event_type: str | None = None,
        processing_stage: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            event_type=event_type,
            cause=cause
        )
        if processing_stage:
            self.details["processing_stage"] = processing_stage
        self.error_code = "EVENT_PROCESSING_ERROR"


# =============================================================================
# Update-Sending Errors
# =============================================================================

class UpdateSendingError(VoiceSessionError):
    """Fehler beim Senden von Updates an Client."""

    def __init__(
        self,
        message: str = "Fehler beim Senden des Updates",
        update_type: str | None = None,
        update_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"category": "update_sending"}
        if update_type:
            details["update_type"] = update_type
        if update_id:
            details["update_id"] = update_id

        super().__init__(
            message=message,
            error_code="UPDATE_SENDING_ERROR",
            details=details,
            cause=cause
        )


class ThreadForwardingError(VoiceSessionError):
    """Fehler beim Weiterleiten von Nachrichten an Thread."""

    def __init__(
        self,
        message: str = "Fehler bei Thread-Weiterleitung",
        thread_id: str | None = None,
        role: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"category": "thread_forwarding"}
        if thread_id:
            details["thread_id"] = thread_id
        if role:
            details["role"] = role

        super().__init__(
            message=message,
            error_code="THREAD_FORWARDING_ERROR",
            details=details,
            cause=cause
        )


class SessionImportError(VoiceSessionError):
    """Fehler beim Importieren von Modulen."""

    def __init__(
        self,
        message: str,
        module_name: str | None = None,
        function_name: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"category": "import_error"}
        if module_name:
            details["module_name"] = module_name
        if function_name:
            details["function_name"] = function_name

        super().__init__(
            message=message,
            error_code="IMPORT_ERROR",
            details=details,
            cause=cause
        )


# =============================================================================
# Content-Processing Errors
# =============================================================================

class ContentProcessingError(VoiceSessionError):
    """Fehler bei Content-Verarbeitung."""

    def __init__(
        self,
        message: str,
        content_type: str | None = None,
        item_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = {"category": "content_processing"}
        if content_type:
            details["content_type"] = content_type
        if item_id:
            details["item_id"] = item_id

        super().__init__(
            message=message,
            error_code="CONTENT_PROCESSING_ERROR",
            details=details,
            cause=cause
        )


# =============================================================================
# Utility-Funktionen
# =============================================================================

def create_session_error(
    error_type: str,
    message: str,
    **kwargs: Any
) -> VoiceSessionError:
    """Factory-Funktion für Session-Errors basierend auf Type.

    Args:
        error_type: Type der Exception
        message: Error-Message
        **kwargs: Zusätzliche Parameter

    Returns:
        Entsprechende Exception-Instanz
    """
    error_classes = {
        "session_not_available": SessionNotAvailableError,
        "session_config": SessionConfigurationError,
        "session_update": SessionUpdateError,
        "audio_processing": AudioProcessingError,
        "audio_sending": AudioSendingError,
        "response_start": ResponseStartError,
        "event_handling": EventHandlingError,
        "event_processing": EventProcessingError,
        "update_sending": UpdateSendingError,
        "thread_forwarding": ThreadForwardingError,
        "import_error": SessionImportError,
        "content_processing": ContentProcessingError,
    }

    error_class = error_classes.get(error_type, VoiceSessionError)
    return error_class(message, **kwargs)


def is_recoverable_error(error: Exception) -> bool:
    """Prüft ob Fehler recoverable ist.

    Args:
        error: Zu prüfende Exception

    Returns:
        True wenn recoverable, False sonst
    """
    # Network/Connection-Errors sind meist recoverable
    recoverable_types = (
        UpdateSendingError,
        AudioSendingError,
        ThreadForwardingError,
    )

    # Configuration-Errors sind meist nicht recoverable
    non_recoverable_types = (
        SessionConfigurationError,
        SessionImportError,
    )

    if isinstance(error, non_recoverable_types):
        return False

    if isinstance(error, recoverable_types):
        return True

    # Default: nicht recoverable für Sicherheit
    return False
