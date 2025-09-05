# backend/voice/session/session_refactored.py
"""Realtime Voice Session Management für Azure OpenAI Integration.

Session-Klasse mit Separation of Concerns, Error-Handling
und reduzierter Komplexität durch Delegation an spezialisierte Services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .audio_processor import AudioProcessor, RealtimeClientProtocol
from .core_operations import ClientProtocol, SessionOperations
from .event_handler import (
    SessionEventHandler,
)
from .session_exceptions import (
    SessionNotAvailableError,
    SessionUpdateError,
)

if TYPE_CHECKING:
    from data_models import Update

    from .session_config import SessionConfig

logger = get_logger(__name__)

class RealtimeSession:
    """Realtime Voice Session für Azure OpenAI Integration.

    Session-Klasse mit klarer Separation of Concerns:
    - AudioProcessor: Audio-Processing und Response-Management
    - SessionOperations: Update-Sending und Thread-Forwarding
    - SessionEventHandler: Event-Handling und Processing
    - SessionConfig: Strukturierte Konfiguration
    """

    def __init__(
        self,
        realtime: RealtimeClientProtocol | None = None,
        client: ClientProtocol | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Initialisiert Realtime-Session.

        Args:
            realtime: Realtime-Client für Audio-Operations
            client: Client für Update-Sending
            thread_id: Thread-ID für Message-Forwarding
        """
        # Core-Services initialisieren
        self.audio_processor = AudioProcessor(realtime)
        self.session_operations = SessionOperations(client, thread_id)

        # Event-Handler mit Dependency-Injection
        self.event_handler = SessionEventHandler(
            update_sender=self.session_operations.update_sender,
            thread_forwarder=self.session_operations.thread_forwarder,
        )

        # Session-State
        self.realtime = realtime
        self.client = client
        self.thread_id = thread_id

        logger.debug("RealtimeSession initialisiert")

    # -------------------------------------------------------------------------
    # Session Configuration (vereinfacht)
    # -------------------------------------------------------------------------

    async def update_realtime_session(self, config: SessionConfig) -> None:
        """Aktualisiert Realtime-Session-Konfiguration mit strukturierter Config.

        Args:
            config: Strukturierte Session-Konfiguration

        Raises:
            SessionNotAvailableError: Wenn Realtime-Client nicht verfügbar
            SessionUpdateError: Bei Session-Update-Fehlern
        """
        if not self.realtime:
            logger.warning("Realtime-Session nicht verfügbar")
            raise SessionNotAvailableError

        try:
            config_dict = config.to_dict()
            await self.realtime.send_session_update(config_dict)
            logger.info("Session-Konfiguration aktualisiert")

        except Exception as e:
            logger.exception("Fehler Session-Update: %s", e)
            raise SessionUpdateError(cause=e) from e



    # -------------------------------------------------------------------------
    # Audio Processing (delegiert)
    # -------------------------------------------------------------------------

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """Sendet Audio-Chunk an Session.

        Args:
            audio_data: Audio-Daten als Bytes

        Returns:
            True wenn erfolgreich gesendet, False sonst
        """
        return await self.audio_processor.send_audio_chunk_safe(audio_data)

    async def start_response(self) -> bool:
        """Startet neue Response.

        Returns:
            True wenn erfolgreich gestartet, False sonst
        """
        success = await self.audio_processor.start_response_safe()
        if success:
            # Synchronisiere Event-Handler-State
            self.event_handler.set_response_active(True)
        return success

    def is_response_active(self) -> bool:
        """Prüft ob Response aktiv ist."""
        return self.audio_processor.is_response_active()

    # -------------------------------------------------------------------------
    # Event Handlers (delegiert)
    # -------------------------------------------------------------------------

    async def handle_response_done(self, event: Any) -> None:
        """Verarbeitet Response-Done-Event.

        Args:
            event: Response-Done-Event-Daten
        """
        await self.event_handler.handle_response_done(event)
        # Synchronisiere Audio-Processor-State
        self.audio_processor.set_response_active(False)

    async def handle_output_item_done(self, event: Any) -> None:
        """Verarbeitet Output-Item-Done-Event.

        Args:
            event: Output-Item-Done-Event-Daten
        """
        await self.event_handler.handle_output_item_done(event)

    async def handle_audio_delta(self, event: Any) -> None:
        """Verarbeitet Audio-Delta-Event.

        Args:
            event: Audio-Delta-Event-Daten
        """
        await self.event_handler.handle_audio_delta(event)

    async def handle_speech_started(self, event: Any) -> None:
        """Verarbeitet Speech-Started-Event.

        Args:
            event: Speech-Started-Event-Daten
        """
        await self.event_handler.handle_speech_started(event)

    # -------------------------------------------------------------------------
    # Core Operations (delegiert)
    # -------------------------------------------------------------------------

    async def send_update(self, update: Update) -> bool:
        """Sendet Update an Client.

        Args:
            update: Zu sendendes Update

        Returns:
            True wenn erfolgreich gesendet, False sonst
        """
        return await self.session_operations.send_update(update)

    async def forward_to_thread(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Leitet Nachricht an Thread weiter.

        Args:
            role: Role der Nachricht
            content: Content der Nachricht
            metadata: Optionale Metadaten

        Returns:
            Message-ID wenn erfolgreich, None sonst
        """
        return await self.session_operations.forward_to_thread(role, content, metadata)

    # -------------------------------------------------------------------------
    # Configuration und State-Management
    # -------------------------------------------------------------------------

    def set_client(self, client: ClientProtocol | None) -> None:
        """Setzt neuen Client für Update-Sending.

        Args:
            client: Neuer Client
        """
        self.client = client
        self.session_operations.set_client(client)
        # Safe call to set_client method if it exists
        if hasattr(self.event_handler.update_sender, "set_client"):
            self.event_handler.update_sender.set_client(client)

    def set_thread_id(self, thread_id: str | None) -> None:
        """Setzt neue Thread-ID für Message-Forwarding.

        Args:
            thread_id: Neue Thread-ID
        """
        self.thread_id = thread_id
        self.session_operations.set_thread_id(thread_id)
        # Safe call to set_thread_id method if it exists
        if hasattr(self.event_handler.thread_forwarder, "set_thread_id"):
            self.event_handler.thread_forwarder.set_thread_id(thread_id)

    def set_realtime_client(self, realtime: RealtimeClientProtocol | None) -> None:
        """Setzt neuen Realtime-Client für Audio-Operations.

        Args:
            realtime: Neuer Realtime-Client
        """
        self.realtime = realtime
        self.audio_processor.set_realtime_client(realtime)

    # -------------------------------------------------------------------------
    # Statistics und Monitoring
    # -------------------------------------------------------------------------

    def get_session_stats(self) -> dict[str, Any]:
        """Liefert Session-Statistiken.

        Returns:
            Dictionary mit Session-Statistiken
        """
        return {
            "audio_stats": self.audio_processor.get_audio_stats(),
            "event_handler_stats": {
                "response_active": self.event_handler.is_response_active(),
                "queue_size": self.event_handler.get_queue_size(),
            },
            "session_state": {
                "has_client": self.session_operations.has_client(),
                "has_thread_id": self.session_operations.has_thread_id(),
                "has_realtime_client": self.audio_processor.has_realtime_client(),
            },
        }

    def reset_stats(self) -> None:
        """Setzt alle Statistiken zurück."""
        self.audio_processor.reset_audio_stats()
        logger.debug("Session-Statistiken zurückgesetzt")


