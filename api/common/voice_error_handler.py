"""Voice-spezifische Error Handler für Azure OpenAI Realtime API und WebSocket-Verbindungen.
Integriert Voice-Fehler in das strukturierte Error-Handling-System.
"""

from datetime import datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed, InvalidMessage

from api.common.error_schemas import ErrorContext, ErrorDetail, ErrorSeverity
from api.common.structured_exceptions import KeikoExternalServiceError, KeikoVoiceError
from kei_logging import get_logger

logger = get_logger(__name__)


class VoiceErrorHandler:
    """Spezialisierter Error Handler für Voice-spezifische Fehler."""

    def __init__(self, user_id: str = None, session_id: str = None):
        self.user_id = user_id
        self.session_id = session_id
        self.error_count = 0
        self.last_error_time = None

    async def handle_websocket_error(
        self,
        websocket: WebSocket,
        error: Exception,
        context: str = "websocket_communication"
    ) -> dict[str, Any] | None:
        """Behandelt WebSocket-spezifische Fehler und erstellt strukturierte Responses.

        Args:
            websocket: WebSocket-Verbindung
            error: Aufgetretener Fehler
            context: Kontext des Fehlers (z.B. "connection", "message_handling")

        Returns:
            Strukturierte Error Response für WebSocket-Client
        """
        self.error_count += 1
        self.last_error_time = datetime.utcnow()

        # Error Context erstellen
        error_context = ErrorContext(
            user_id=self.user_id,
            tenant_id=getattr(self, "tenant_id", None),
            session_id=self.session_id,
            request_path="/voice/websocket",
            request_method="WEBSOCKET",
            client_ip=websocket.client.host if websocket.client else None,
            user_agent=getattr(self, "user_agent", None),
            correlation_id=getattr(self, "correlation_id", None)
        )

        # Spezifische WebSocket-Fehler behandeln
        if isinstance(error, WebSocketDisconnect):
            voice_error = KeikoVoiceError(
                message="WebSocket-Verbindung wurde unterbrochen",
                voice_component="websocket_connection",
                error_code="WEBSOCKET_DISCONNECT",
                context=error_context,
                severity=ErrorSeverity.LOW
            )

        elif isinstance(error, ConnectionClosed):
            voice_error = KeikoVoiceError(
                message="WebSocket-Verbindung wurde geschlossen",
                voice_component="websocket_connection",
                error_code="WEBSOCKET_CONNECTION_CLOSED",
                context=error_context,
                severity=ErrorSeverity.MEDIUM
            )

        elif isinstance(error, InvalidMessage):
            voice_error = KeikoVoiceError(
                message="Ungültige WebSocket-Nachricht empfangen",
                voice_component="websocket_message",
                error_code="WEBSOCKET_INVALID_MESSAGE",
                context=error_context,
                details=[ErrorDetail(
                    field="message",
                    code="INVALID_FORMAT",
                    message=str(error)
                )]
            )

        else:
            voice_error = KeikoVoiceError(
                message=f"WebSocket-Fehler: {error!s}",
                voice_component=context,
                error_code="WEBSOCKET_ERROR",
                context=error_context
            )

        # Logging
        await self._log_voice_error(voice_error, context)

        # Strukturierte Response für WebSocket-Client
        error_response = voice_error.to_structured_response(include_debug=True)

        return {
            "type": "error",
            "error": error_response.model_dump(exclude_none=True),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def handle_azure_realtime_error(
        self,
        error: Exception,
        operation: str = "unknown",
        azure_response: dict[str, Any] = None
    ) -> KeikoExternalServiceError:
        """Behandelt Azure OpenAI Realtime API Fehler.

        Args:
            error: Aufgetretener Fehler
            operation: Azure-Operation die fehlgeschlagen ist
            azure_response: Azure API Response (falls verfügbar)

        Returns:
            Strukturierte Azure Service Error
        """
        # Error Details aus Azure Response extrahieren
        details = []
        if azure_response:
            if "error" in azure_response:
                azure_error = azure_response["error"]
                details.append(ErrorDetail(
                    field="azure_error_code",
                    code=azure_error.get("code", "UNKNOWN"),
                    message=azure_error.get("message", str(error))
                ))

                if "details" in azure_error:
                    for detail in azure_error["details"]:
                        details.append(ErrorDetail(
                            field=detail.get("target", "unknown"),
                            code=detail.get("code", "AZURE_ERROR"),
                            message=detail.get("message", "Azure service error")
                        ))

        # Error Context
        error_context = ErrorContext(
            user_id=self.user_id,
            tenant_id=getattr(self, "tenant_id", None),
            session_id=self.session_id,
            request_path="/voice/azure-realtime",
            request_method="POST",
            client_ip=getattr(self, "client_ip", None),
            user_agent=getattr(self, "user_agent", None),
            correlation_id=getattr(self, "correlation_id", None)
        )

        # Azure Service Error erstellen
        azure_error = KeikoExternalServiceError(
            message=f"Azure OpenAI Realtime API Fehler bei {operation}: {error!s}",
            service_name="azure_openai_realtime",
            error_code="AZURE_REALTIME_ERROR",
            context=error_context,
            details=details,
            severity=ErrorSeverity.HIGH
        )

        # Logging
        await self._log_voice_error(azure_error, f"azure_realtime_{operation}")

        return azure_error

    async def handle_voice_configuration_error(
        self,
        config_field: str,
        invalid_value: Any,
        validation_message: str
    ) -> KeikoVoiceError:
        """Behandelt Voice Configuration Validierungsfehler.

        Args:
            config_field: Fehlerhaftes Konfigurationsfeld
            invalid_value: Ungültiger Wert
            validation_message: Validierungsfehlermeldung

        Returns:
            Voice Configuration Error
        """
        details = [ErrorDetail(
            field=config_field,
            code="INVALID_VOICE_CONFIG",
            message=validation_message,
            value=invalid_value
        )]

        error_context = ErrorContext(
            user_id=self.user_id,
            tenant_id=getattr(self, "tenant_id", None),
            session_id=self.session_id,
            request_path="/voice/settings",
            request_method="POST",
            client_ip=getattr(self, "client_ip", None),
            user_agent=getattr(self, "user_agent", None),
            correlation_id=getattr(self, "correlation_id", None)
        )

        config_error = KeikoVoiceError(
            message=f"Ungültige Voice-Konfiguration: {validation_message}",
            voice_component="voice_configuration",
            error_code="VOICE_CONFIG_VALIDATION_ERROR",
            context=error_context,
            details=details,
            severity=ErrorSeverity.LOW
        )

        await self._log_voice_error(config_error, "voice_configuration")

        return config_error

    async def handle_audio_processing_error(
        self,
        error: Exception,
        audio_format: str = None,
        processing_stage: str = "unknown"
    ) -> KeikoVoiceError:
        """Behandelt Audio-Verarbeitungsfehler.

        Args:
            error: Aufgetretener Fehler
            audio_format: Audio-Format (falls relevant)
            processing_stage: Verarbeitungsstufe (encoding, decoding, etc.)

        Returns:
            Audio Processing Error
        """
        details = []
        if audio_format:
            details.append(ErrorDetail(
                field="audio_format",
                code="AUDIO_FORMAT_ERROR",
                message=f"Fehler bei Audio-Format: {audio_format}"
            ))

        details.append(ErrorDetail(
            field="processing_stage",
            code="AUDIO_PROCESSING_ERROR",
            message=f"Fehler in Verarbeitungsstufe: {processing_stage}"
        ))

        error_context = ErrorContext(
            user_id=self.user_id,
            tenant_id=getattr(self, "tenant_id", None),
            session_id=self.session_id,
            request_path="/voice/audio",
            request_method="POST",
            client_ip=getattr(self, "client_ip", None),
            user_agent=getattr(self, "user_agent", None),
            correlation_id=getattr(self, "correlation_id", None)
        )

        audio_error = KeikoVoiceError(
            message=f"Audio-Verarbeitungsfehler in {processing_stage}: {error!s}",
            voice_component="audio_processing",
            error_code="AUDIO_PROCESSING_ERROR",
            context=error_context,
            details=details,
            severity=ErrorSeverity.MEDIUM
        )

        await self._log_voice_error(audio_error, f"audio_{processing_stage}")

        return audio_error

    async def _log_voice_error(self, error: Exception, context: str) -> None:
        """Loggt Voice-Fehler mit strukturierten Metadaten."""
        log_data = {
            "voice_context": context,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "error_count": self.error_count,
            "error_type": type(error).__name__
        }

        if isinstance(error, (KeikoVoiceError, KeikoExternalServiceError)):
            log_data.update({
                "error_code": error.error_code,
                "category": error.category.value,
                "severity": error.severity.value
            })

            if error.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"🔴 CRITICAL VOICE ERROR: {error.message}", extra=log_data)
            elif error.severity == ErrorSeverity.HIGH:
                logger.error(f"🟠 HIGH VOICE ERROR: {error.message}", extra=log_data)
            elif error.severity == ErrorSeverity.MEDIUM:
                logger.warning(f"🟡 MEDIUM VOICE ERROR: {error.message}", extra=log_data)
            else:
                logger.info(f"🔵 LOW VOICE ERROR: {error.message}", extra=log_data)
        else:
            logger.error(f"🔴 VOICE ERROR: {error!s}", extra=log_data)

    def get_error_statistics(self) -> dict[str, Any]:
        """Gibt Error-Statistiken für die aktuelle Session zurück."""
        return {
            "error_count": self.error_count,
            "last_error_time": self.last_error_time.isoformat() + "Z" if self.last_error_time else None,
            "session_id": self.session_id,
            "user_id": self.user_id
        }


# Utility-Funktionen für Voice Error Handling
async def create_voice_websocket_error_response(
    error: Exception,
    user_id: str = None,
    session_id: str = None
) -> dict[str, Any]:
    """Erstellt eine WebSocket Error Response für Voice-Fehler."""
    from unittest.mock import Mock

    # Mock WebSocket für Type-Safety in Tests
    mock_websocket = Mock()
    mock_websocket.client = Mock()
    mock_websocket.client.host = "test-host"

    handler = VoiceErrorHandler(user_id=user_id, session_id=session_id)
    return await handler.handle_websocket_error(mock_websocket, error)


def is_voice_related_error(error: Exception) -> bool:
    """Prüft ob ein Fehler Voice-bezogen ist."""
    error_message = str(error).lower()
    voice_keywords = [
        "websocket", "azure", "openai", "voice", "audio",
        "realtime", "speech", "transcription", "synthesis",
        "microphone", "speaker", "vad", "detection"
    ]

    return any(keyword in error_message for keyword in voice_keywords)
