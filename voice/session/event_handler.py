"""Event-Handler für Voice Session Management.

Separate Event-Handler-Klasse für bessere Separation of Concerns
und reduzierte Komplexität der RealtimeSession-Klasse.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from data_models import (
    AudioUpdate,
    ConsoleUpdate,
    MessageUpdate,
    Update,
    UpdateType,
)
from kei_logging import get_logger

from .content_utils import extract_and_validate_content
from .role_utils import convert_legacy_role_mapping
from .session_constants import (
    FUNCTION_CALL_TYPE,
    INFO_LEVEL,
    MESSAGE_TYPE,
    OUTPUT_ITEM_DONE_MSG,
    RESPONSE_COMPLETED_MSG,
)
from .session_exceptions import (
    ContentProcessingError,
    EventHandlingError,
    EventProcessingError,
)

logger = get_logger(__name__)


class EventData(Protocol):
    """Protocol für Event-Daten."""

    event_id: str | None


class ResponseDoneEventData(EventData):
    """Protocol für Response-Done-Event-Daten."""

    response: Any | None


class OutputItemDoneEventData(EventData):
    """Protocol für Output-Item-Done-Event-Daten."""

    item: Any | None


class AudioDeltaEventData(EventData):
    """Protocol für Audio-Delta-Event-Daten."""

    delta: bytes | None


class UpdateSender(Protocol):
    """Protocol für Update-Sender (Client)."""

    async def send_update(self, update: Update) -> None:
        """Sendet Update an Client."""
        ...


class ThreadForwarder(Protocol):
    """Protocol für Thread-Forwarding."""

    async def forward_to_thread(self, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        """Leitet Nachricht an Thread weiter."""
        ...


class SessionEventHandler:
    """Event-Handler für Realtime-Session-Events.

    Extrahierte Event-Handler-Logik aus RealtimeSession für bessere
    Separation of Concerns und reduzierte Klassenkomplexität.
    """

    def __init__(
        self,
        update_sender: UpdateSender | None = None,
        thread_forwarder: ThreadForwarder | None = None,
    ) -> None:
        """Initialisiert Event-Handler.

        Args:
            update_sender: Client für Update-Sending
            thread_forwarder: Service für Thread-Forwarding
        """
        self.update_sender = update_sender
        self.thread_forwarder = thread_forwarder
        self._response_queue: list[Any] = []
        self._response_lock = asyncio.Lock()
        self._active_response = False

    # -------------------------------------------------------------------------
    # Core Event-Handler
    # -------------------------------------------------------------------------

    async def handle_response_done(self, event: ResponseDoneEventData) -> None:
        """Verarbeitet Response-Done-Event.

        Args:
            event: Response-Done-Event-Daten

        Raises:
            EventHandlingError: Bei kritischen Event-Handling-Fehlern
        """
        try:
            self._active_response = False
            logger.info(RESPONSE_COMPLETED_MSG)

            if not event.response or not hasattr(event.response, "output") or not event.response.output:
                logger.debug("Response hat keine Output-Items")
                return

            # Verarbeite alle Output-Items
            for output in event.response.output:
                await self._process_output_item(output)

            # Verarbeite Response-Queue
            await self._process_response_queue()

        except Exception as e:
            logger.exception("Fehler bei Response-Done-Handling: %s", e)
            raise EventHandlingError(
                message="Fehler bei Response-Done-Event-Verarbeitung",
                event_type="response.done",
                event_id=getattr(event, "event_id", None),
                cause=e
            ) from e

    async def handle_output_item_done(self, event: OutputItemDoneEventData) -> None:
        """Verarbeitet Output-Item-Done-Event.

        Args:
            event: Output-Item-Done-Event-Daten

        Raises:
            EventHandlingError: Bei kritischen Event-Handling-Fehlern
        """
        try:
            if not event.item:
                logger.debug("Event hat kein Item")
                return

            item = event.item
            logger.info(OUTPUT_ITEM_DONE_MSG, item.type, item.id)

            # Verarbeite Message-Items
            if item.type == MESSAGE_TYPE and hasattr(item, "content") and item.content:
                await self._handle_message_item(item)

            # Verarbeite Function-Call-Items
            elif item.type == FUNCTION_CALL_TYPE:
                await self._handle_function_call_item(item)

            else:
                logger.debug("Unbekannter Item-Type: %s", item.type)

        except Exception as e:
            logger.exception("Fehler bei Output-Item-Done-Handling: %s", e)
            raise EventHandlingError(
                message="Fehler bei Output-Item-Done-Event-Verarbeitung",
                event_type="response.output_item.done",
                event_id=getattr(event, "event_id", None),
                cause=e
            ) from e

    async def handle_audio_delta(self, event: AudioDeltaEventData) -> None:
        """Verarbeitet Audio-Delta-Event.

        Args:
            event: Audio-Delta-Event-Daten

        Raises:
            EventHandlingError: Bei kritischen Event-Handling-Fehlern
        """
        try:
            if not event.delta or not self.update_sender:
                return

            audio_update = AudioUpdate(
                update_id=str(getattr(event, "event_id", "unknown")),
                type=UpdateType.AUDIO,
                audio_data=event.delta,
            )

            await self.update_sender.send_update(audio_update)

        except Exception as e:
            logger.exception("Fehler bei Audio-Delta-Handling: %s", e)
            raise EventHandlingError(
                message="Fehler bei Audio-Delta-Event-Verarbeitung",
                event_type="response.audio.delta",
                event_id=getattr(event, "event_id", None),
                cause=e
            ) from e

    async def handle_speech_started(self, event: EventData) -> None:
        """Verarbeitet Speech-Started-Event.

        Args:
            event: Speech-Started-Event-Daten

        Raises:
            EventHandlingError: Bei kritischen Event-Handling-Fehlern
        """
        try:
            if not self.update_sender:
                return

            interrupt_update = Update(
                update_id=str(getattr(event, "event_id", "unknown")),
                type=UpdateType.INTERRUPT
            )

            await self.update_sender.send_update(interrupt_update)

        except Exception as e:
            logger.exception("Fehler bei Speech-Started-Handling: %s", e)
            raise EventHandlingError(
                message="Fehler bei Speech-Started-Event-Verarbeitung",
                event_type="input_audio_buffer.speech_started",
                event_id=getattr(event, "event_id", None),
                cause=e
            ) from e

    # -------------------------------------------------------------------------
    # Private Helper-Methoden
    # -------------------------------------------------------------------------

    async def _process_output_item(self, output: Any) -> None:
        """Verarbeitet einzelnes Output-Item.

        Args:
            output: Output-Item

        Raises:
            EventProcessingError: Bei Verarbeitungsfehlern
        """
        try:
            if output.type == MESSAGE_TYPE and hasattr(output, "content") and output.content:
                await self._handle_message_item(output)

        except Exception as e:
            logger.exception("Fehler bei Output-Item-Verarbeitung: %s", e)
            raise EventProcessingError(
                message="Fehler bei Output-Item-Verarbeitung",
                event_type="output_item",
                processing_stage="process_output_item",
                cause=e
            ) from e

    async def _handle_message_item(self, item: Any) -> None:
        """Verarbeitet Message-Item.

        Args:
            item: Message-Item

        Raises:
            ContentProcessingError: Bei Content-Verarbeitungsfehlern
        """
        try:
            # Extrahiere und validiere Content
            content = extract_and_validate_content(item)
            if not content:
                logger.debug("Kein gültiger Content in Message-Item")
                return

            # Mappe Role zu Enum
            role_enum = convert_legacy_role_mapping(getattr(item, "role", None))

            # Sende Update an Client
            if self.update_sender:
                message_update = MessageUpdate(
                    update_id=str(getattr(item, "id", "unknown")),
                    type=UpdateType.MESSAGE,
                    role=role_enum,
                    content=content,
                )
                await self.update_sender.send_update(message_update)

            # Leite an Thread weiter
            if self.thread_forwarder:
                role_str = getattr(item, "role", None) or "assistant"
                await self.thread_forwarder.forward_to_thread(role_str, content)

        except Exception as e:
            logger.exception("Fehler bei Message-Item-Handling: %s", e)
            raise ContentProcessingError(
                message="Fehler bei Message-Item-Verarbeitung",
                content_type=MESSAGE_TYPE,
                item_id=str(getattr(item, "id", "unknown")),
                cause=e
            ) from e

    async def _handle_function_call_item(self, item: Any) -> None:
        """Verarbeitet Function-Call-Item.

        Args:
            item: Function-Call-Item

        Raises:
            ContentProcessingError: Bei Content-Verarbeitungsfehlern
        """
        try:
            if not self.update_sender:
                return

            # Erstelle Console-Update für Function-Call
            console_update = ConsoleUpdate(
                update_id=str(getattr(item, "call_id", getattr(item, "id", "unknown"))),
                type=UpdateType.CONSOLE,
                payload=item.model_dump(exclude={"id", "call_id"}) if hasattr(item, "model_dump") else {},
                level=INFO_LEVEL,
            )

            await self.update_sender.send_update(console_update)

        except Exception as e:
            logger.exception("Fehler bei Function-Call-Item-Handling: %s", e)
            raise ContentProcessingError(
                message="Fehler bei Function-Call-Item-Verarbeitung",
                content_type=FUNCTION_CALL_TYPE,
                item_id=str(getattr(item, "id", "unknown")),
                cause=e
            ) from e

    async def _process_response_queue(self) -> None:
        """Verarbeitet Response-Queue.

        Raises:
            EventProcessingError: Bei Queue-Verarbeitungsfehlern
        """
        try:
            async with self._response_lock:
                while self._response_queue:
                    queued_item = self._response_queue.pop(0)
                    await self._process_output_item(queued_item)

        except Exception as e:
            logger.exception("Fehler bei Response-Queue-Verarbeitung: %s", e)
            raise EventProcessingError(
                message="Fehler bei Response-Queue-Verarbeitung",
                processing_stage="process_response_queue",
                cause=e
            ) from e

    # -------------------------------------------------------------------------
    # State-Management
    # -------------------------------------------------------------------------

    def is_response_active(self) -> bool:
        """Prüft ob Response aktiv ist."""
        return self._active_response

    def set_response_active(self, active: bool) -> None:
        """Setzt Response-Status."""
        self._active_response = active

    async def add_to_response_queue(self, item: Any) -> None:
        """Fügt Item zur Response-Queue hinzu."""
        async with self._response_lock:
            self._response_queue.append(item)

    def get_queue_size(self) -> int:
        """Liefert aktuelle Queue-Größe."""
        return len(self._response_queue)
