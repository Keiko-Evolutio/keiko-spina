"""StreamService Interface-Definition."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import CoreService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ._types import ChannelName, EventData, MessageHandler, OptionalTimeout


class StreamService(CoreService):
    """Definiert den Vertrag für Streaming-Funktionalität.

    Kern-Service für bidirektionale Stream-Kommunikation und Event-Streaming.
    """

    @abstractmethod
    async def send_event(self, channel: ChannelName, event: EventData) -> None:
        """Sendet ein Event auf einen Kanal.

        Args:
            channel: Name des Ziel-Kanals.
            event: Event-Daten als Dictionary.

        Raises:
            ValueError: Bei ungültigem Kanal-Namen oder Event-Daten.
            RuntimeError: Bei Verbindungsfehlern.
        """

    @abstractmethod
    async def subscribe_channel(
        self,
        channel: ChannelName,
        handler: MessageHandler,
        timeout: OptionalTimeout = None
    ) -> None:
        """Abonniert einen Kanal für Event-Empfang.

        Args:
            channel: Name des zu abonnierenden Kanals.
            handler: Callback-Funktion für eingehende Events.
            timeout: Optionales Timeout für die Subscription.

        Raises:
            ValueError: Bei ungültigen Parametern.
            RuntimeError: Bei Verbindungsfehlern.
        """

    @abstractmethod
    async def create_stream(self, channel: ChannelName) -> AsyncIterator[EventData]:
        """Erstellt einen Event-Stream für einen Kanal.

        Args:
            channel: Name des Kanals für den Stream.

        Yields:
            Event-Daten vom Kanal.

        Raises:
            ValueError: Bei ungültigem Kanal-Namen.
            RuntimeError: Bei Stream-Fehlern.
        """
