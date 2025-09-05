"""Interface für Webhook Manager."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import FeatureService

if TYPE_CHECKING:
    from ._types import EventData, OperationResult, ServiceConfig

# Alias für Backward Compatibility wird unten definiert


class WebhookManagerService(FeatureService):
    """Definiert den Vertrag für den Webhook Manager.

    Feature-Service für ausgehende Webhook-Verarbeitung und Worker-Pool-Management.
    """

    @abstractmethod
    async def start_worker_pool(self) -> None:
        """Startet den Worker-Pool für ausgehende Webhooks.

        Raises:
            RuntimeError: Wenn Worker-Pool bereits läuft oder Konfigurationsfehler.
        """

    @abstractmethod
    async def stop_worker_pool(self) -> None:
        """Stoppt den Worker-Pool und bereinigt Ressourcen.

        Raises:
            RuntimeError: Bei Fehlern beim Stoppen der Worker.
        """

    @abstractmethod
    async def send_webhook(
        self,
        url: str,
        payload: EventData,
        config: ServiceConfig
    ) -> OperationResult:
        """Sendet einen Webhook an die angegebene URL.

        Args:
            url: Ziel-URL für den Webhook.
            payload: Zu sendende Daten.
            config: Webhook-Konfiguration (Retry, Timeout, etc.).

        Returns:
            True bei erfolgreichem Versand.

        Raises:
            ValueError: Bei ungültiger URL oder Payload.
            RuntimeError: Bei Versand-Fehlern.
        """

    @abstractmethod
    async def register_webhook(self, event_type: str, url: str, config: ServiceConfig) -> str:
        """Registriert einen Webhook für einen Event-Typ.

        Args:
            event_type: Typ des Events für den Webhook.
            url: Ziel-URL für den Webhook.
            config: Webhook-Konfiguration.

        Returns:
            Eindeutige Webhook-ID.

        Raises:
            ValueError: Bei ungültigen Parametern.
        """

    @abstractmethod
    async def unregister_webhook(self, webhook_id: str) -> OperationResult:
        """Entfernt einen registrierten Webhook.

        Args:
            webhook_id: ID des zu entfernenden Webhooks.

        Returns:
            True bei erfolgreichem Entfernen.

        Raises:
            ValueError: Bei ungültiger Webhook-ID.
        """

    @abstractmethod
    async def get_registered_webhooks(self) -> list[str]:
        """Liefert Liste aller registrierten Webhook-IDs.

        Returns:
            Liste der Webhook-IDs.
        """


# Backward Compatibility Alias
WebhookManagerInterface = WebhookManagerService
