"""BusService Interface-Definition."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import CoreService

if TYPE_CHECKING:
    from ._types import EventHandler, OptionalQueue, SubjectName


class BusService(CoreService):
    """Definiert den Vertrag für den Nachrichtenbus-Service.

    Kern-Service für Publish/Subscribe-Messaging und Event-Distribution.
    """

    @abstractmethod
    async def publish(self, subject: SubjectName, data: bytes) -> None:
        """Veröffentlicht Daten auf einem Subject.

        Args:
            subject: Name des Subjects für die Nachricht.
            data: Zu veröffentlichende Daten als Bytes.

        Raises:
            ValueError: Bei ungültigem Subject-Namen.
            RuntimeError: Bei Verbindungsfehlern.
        """

    @abstractmethod
    async def subscribe(
        self,
        subject: SubjectName,
        queue: OptionalQueue,
        handler: EventHandler,
        *,
        durable: OptionalQueue = None,
    ) -> None:
        """Abonniert ein Subject und ruft einen Handler auf.

        Args:
            subject: Name des zu abonnierenden Subjects.
            queue: Optionaler Queue-Name für Load-Balancing.
            handler: Callback-Funktion für eingehende Nachrichten.
            durable: Optionaler Name für durable Subscriptions.

        Raises:
            ValueError: Bei ungültigen Parametern.
            RuntimeError: Bei Verbindungsfehlern.
        """

    @abstractmethod
    async def unsubscribe(self, subject: SubjectName, queue: OptionalQueue = None) -> None:
        """Beendet Subscription für ein Subject.

        Args:
            subject: Name des Subjects.
            queue: Optionaler Queue-Name.

        Raises:
            ValueError: Bei ungültigen Parametern.
        """
