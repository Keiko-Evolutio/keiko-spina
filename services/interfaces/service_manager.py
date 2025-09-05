"""Interface für zentralen Service Manager."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import InfrastructureService, ServiceStatus

if TYPE_CHECKING:
    from ._types import HealthStatus, ServiceId

# Alias für Backward Compatibility wird unten definiert


class ServiceManagerService(InfrastructureService):
    """Definiert den Vertrag für den zentralen Service Manager.

    Infrastructure-Service für Service-Lifecycle-Management und Health-Monitoring.
    """

    @abstractmethod
    async def register_service(self, service_id: ServiceId, service: InfrastructureService) -> None:
        """Registriert einen Service beim Manager.

        Args:
            service_id: Eindeutige ID des Services.
            service: Service-Instanz zur Registrierung.

        Raises:
            ValueError: Bei ungültiger Service-ID oder bereits registriertem Service.
        """

    @abstractmethod
    async def unregister_service(self, service_id: ServiceId) -> None:
        """Entfernt einen Service vom Manager.

        Args:
            service_id: ID des zu entfernenden Services.

        Raises:
            ValueError: Bei ungültiger Service-ID.
            RuntimeError: Wenn Service nicht gefunden wird.
        """

    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Liefert Health-Status für alle Services zurück.

        Returns:
            Aggregierter Health-Status aller registrierten Services.
        """

    @abstractmethod
    def is_healthy(self) -> bool:
        """Prüft grundlegenden Gesundheitsstatus synchron.

        Returns:
            True wenn alle kritischen Services gesund sind.
        """

    @abstractmethod
    async def get_service_list(self) -> list[ServiceId]:
        """Liefert Liste aller registrierten Services.

        Returns:
            Liste der Service-IDs.
        """

    @abstractmethod
    async def get_service_status(self, service_id: ServiceId) -> ServiceStatus:
        """Liefert Status eines spezifischen Services.

        Args:
            service_id: ID des Services.

        Returns:
            Aktueller Status des Services.

        Raises:
            ValueError: Bei ungültiger Service-ID.
            RuntimeError: Wenn Service nicht gefunden wird.
        """


# Backward Compatibility Alias
ServiceManagerInterface = ServiceManagerService
