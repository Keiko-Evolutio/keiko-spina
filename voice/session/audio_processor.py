"""Audio-Processing für Voice Session Management.

Audio-Processing-Service mit verbesserter Error-Handling,
State-Management und Type-Safety.
"""

from __future__ import annotations

from typing import Any, Protocol

from kei_logging import get_logger

from .session_constants import (
    AUDIO_SENDING_ERROR_MSG,
    MAX_AUDIO_CHUNK_SIZE_BYTES,
    RESPONSE_START_ERROR_MSG,
    SESSION_NOT_AVAILABLE_MSG,
)
from .session_exceptions import (
    AudioSendingError,
    ResponseStartError,
    SessionNotAvailableError,
)

logger = get_logger(__name__)


class RealtimeClientProtocol(Protocol):
    """Protocol für Realtime-Client-Interface."""

    async def send_audio(self, audio_data: bytes) -> None:
        """Sendet Audio-Daten an Realtime-Service."""
        ...

    async def create_response(self) -> None:
        """Startet neue Response im Realtime-Service."""
        ...

    async def send_session_update(self, config_dict: dict) -> None:
        """Sendet Session-Update an Realtime-Service."""
        ...


class AudioProcessor:
    """Service für Audio-Processing in Realtime-Sessions.

    Extrahierte und verbesserte Audio-Processing-Logik mit besserer
    Error-Handling, State-Management und Type-Safety.
    """

    def __init__(self, realtime_client: RealtimeClientProtocol | None = None) -> None:
        """Initialisiert Audio-Processor.

        Args:
            realtime_client: Realtime-Client für Audio-Operations
        """
        self.realtime_client = realtime_client
        self._active_response = False
        self._audio_stats = {
            "chunks_sent": 0,
            "total_bytes_sent": 0,
            "responses_started": 0,
            "errors_count": 0,
        }

    # -------------------------------------------------------------------------
    # Audio-Sending
    # -------------------------------------------------------------------------

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """Sendet Audio-Chunk an Realtime-Session.

        Args:
            audio_data: Audio-Daten als Bytes

        Returns:
            True wenn erfolgreich gesendet, False sonst

        Raises:
            SessionNotAvailableError: Wenn Realtime-Client nicht verfügbar
            AudioSendingError: Bei kritischen Audio-Sending-Fehlern
        """
        if not self.realtime_client:
            logger.warning(SESSION_NOT_AVAILABLE_MSG)
            raise SessionNotAvailableError

        if not audio_data:
            logger.debug("Audio-Daten sind leer")
            return False

        if not isinstance(audio_data, bytes):
            logger.error("Audio-Daten müssen bytes sein, erhalten: %s", type(audio_data).__name__)
            return False

        try:
            await self.realtime_client.send_audio(audio_data)

            # Update Statistics
            self._audio_stats["chunks_sent"] += 1
            self._audio_stats["total_bytes_sent"] += len(audio_data)

            logger.debug("Audio-Chunk gesendet: %d bytes", len(audio_data))
            return True

        except Exception as e:
            self._audio_stats["errors_count"] += 1
            logger.exception("%s: %s", AUDIO_SENDING_ERROR_MSG, e)

            # Für kritische Fehler Exception werfen
            if self._is_critical_audio_error(e):
                raise AudioSendingError(
                    message=AUDIO_SENDING_ERROR_MSG,
                    audio_size=len(audio_data),
                    cause=e
                ) from e

            return False

    async def send_audio_chunk_safe(self, audio_data: bytes) -> bool:
        """Sendet Audio-Chunk mit vollständiger Error-Suppression.

        Args:
            audio_data: Audio-Daten als Bytes

        Returns:
            True wenn erfolgreich gesendet, False sonst
        """
        try:
            return await self.send_audio_chunk(audio_data)
        except Exception as e:
            logger.exception("Fehler beim sicheren Audio-Chunk-Sending: %s", e)
            return False

    # -------------------------------------------------------------------------
    # Response-Management
    # -------------------------------------------------------------------------

    async def start_response(self) -> bool:
        """Startet neue Response in Realtime-Session.

        Returns:
            True wenn erfolgreich gestartet, False sonst

        Raises:
            SessionNotAvailableError: Wenn Realtime-Client nicht verfügbar
            ResponseStartError: Bei kritischen Response-Start-Fehlern
        """
        if not self.realtime_client:
            logger.warning(SESSION_NOT_AVAILABLE_MSG)
            raise SessionNotAvailableError

        try:
            await self.realtime_client.create_response()
            self._active_response = True

            # Update Statistics
            self._audio_stats["responses_started"] += 1

            logger.debug("Response erfolgreich gestartet")
            return True

        except Exception as e:
            self._audio_stats["errors_count"] += 1
            logger.exception("%s: %s", RESPONSE_START_ERROR_MSG, e)

            # Für kritische Fehler Exception werfen
            if self._is_critical_response_error(e):
                raise ResponseStartError(
                    message=RESPONSE_START_ERROR_MSG,
                    cause=e
                ) from e

            return False

    async def start_response_safe(self) -> bool:
        """Startet Response mit vollständiger Error-Suppression.

        Returns:
            True wenn erfolgreich gestartet, False sonst
        """
        try:
            return await self.start_response()
        except Exception as e:
            logger.exception("Fehler beim sicheren Response-Start: %s", e)
            return False

    def stop_response(self) -> None:
        """Stoppt aktive Response."""
        self._active_response = False
        logger.debug("Response gestoppt")

    # -------------------------------------------------------------------------
    # State-Management
    # -------------------------------------------------------------------------

    def is_response_active(self) -> bool:
        """Prüft ob Response aktiv ist."""
        return self._active_response

    def set_response_active(self, active: bool) -> None:
        """Setzt Response-Status.

        Args:
            active: Neuer Response-Status
        """
        self._active_response = active
        logger.debug("Response-Status gesetzt: %s", active)

    def has_realtime_client(self) -> bool:
        """Prüft ob Realtime-Client verfügbar ist."""
        return self.realtime_client is not None

    def set_realtime_client(self, client: RealtimeClientProtocol | None) -> None:
        """Setzt neuen Realtime-Client.

        Args:
            client: Neuer Realtime-Client
        """
        self.realtime_client = client
        logger.debug("Realtime-Client gesetzt: %s", client is not None)

    # -------------------------------------------------------------------------
    # Statistics und Monitoring
    # -------------------------------------------------------------------------

    def get_audio_stats(self) -> dict[str, int]:
        """Liefert Audio-Processing-Statistiken.

        Returns:
            Dictionary mit Audio-Statistiken
        """
        return self._audio_stats.copy()

    def reset_audio_stats(self) -> None:
        """Setzt Audio-Statistiken zurück."""
        self._audio_stats = {
            "chunks_sent": 0,
            "total_bytes_sent": 0,
            "responses_started": 0,
            "errors_count": 0,
        }
        logger.debug("Audio-Statistiken zurückgesetzt")

    def get_average_chunk_size(self) -> float:
        """Berechnet durchschnittliche Chunk-Größe.

        Returns:
            Durchschnittliche Chunk-Größe in Bytes
        """
        chunks_sent = self._audio_stats["chunks_sent"]
        if chunks_sent == 0:
            return 0.0

        return self._audio_stats["total_bytes_sent"] / chunks_sent

    def get_error_rate(self) -> float:
        """Berechnet Error-Rate.

        Returns:
            Error-Rate als Prozentsatz (0.0 - 1.0)
        """
        total_operations = (
            self._audio_stats["chunks_sent"] +
            self._audio_stats["responses_started"] +
            self._audio_stats["errors_count"]
        )

        if total_operations == 0:
            return 0.0

        return self._audio_stats["errors_count"] / total_operations

    # -------------------------------------------------------------------------
    # Private Helper-Methoden
    # -------------------------------------------------------------------------

    def _is_critical_audio_error(self, error: Exception) -> bool:
        """Prüft ob Audio-Fehler kritisch ist.

        Args:
            error: Zu prüfender Fehler

        Returns:
            True wenn kritisch, False sonst
        """
        # Connection-Errors sind meist nicht kritisch
        critical_error_types = (
            TypeError,
            AttributeError,
            ValueError,
        )

        return isinstance(error, critical_error_types)

    def _is_critical_response_error(self, error: Exception) -> bool:
        """Prüft ob Response-Fehler kritisch ist.

        Args:
            error: Zu prüfender Fehler

        Returns:
            True wenn kritisch, False sonst
        """
        # Connection-Errors sind meist nicht kritisch
        critical_error_types = (
            TypeError,
            AttributeError,
            ValueError,
        )

        return isinstance(error, critical_error_types)


# =============================================================================
# Audio-Validation Utilities
# =============================================================================

def validate_audio_data(audio_data: bytes) -> bool:
    """Validiert Audio-Daten.

    Args:
        audio_data: Zu validierende Audio-Daten

    Returns:
        True wenn gültig, False sonst
    """
    if not isinstance(audio_data, bytes):
        return False

    if len(audio_data) == 0:
        return False

    # Prüfe auf minimale Chunk-Größe (z.B. 1 byte)
    if len(audio_data) < 1:
        return False

    # Prüfe auf maximale Chunk-Größe (z.B. 1MB)
    return not len(audio_data) > MAX_AUDIO_CHUNK_SIZE_BYTES


def get_audio_format_info(audio_data: bytes) -> dict[str, Any]:
    """Extrahiert Format-Informationen aus Audio-Daten.

    Args:
        audio_data: Audio-Daten

    Returns:
        Dictionary mit Format-Informationen
    """
    return {
        "size_bytes": len(audio_data),
        "is_valid": validate_audio_data(audio_data),
        "type": "bytes",
    }
