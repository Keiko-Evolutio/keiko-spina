"""Interface für DomainRevalidationService."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import InfrastructureService

if TYPE_CHECKING:
    from ._types import OperationResult, ServiceResult

# Alias für Backward Compatibility wird unten definiert


class DomainRevalidationService(InfrastructureService):
    """Vertrag für den Domain-Revalidierungs-Service.

    Infrastructure-Service für automatische Domain-Validierung und SSL-Zertifikat-Management.
    """

    @abstractmethod
    async def start_revalidation(self) -> None:
        """Startet periodische Revalidierung und optionalen Config-Reload.

        Raises:
            RuntimeError: Wenn Service bereits läuft oder Konfigurationsfehler.
        """

    @abstractmethod
    async def stop_revalidation(self) -> None:
        """Stoppt den Service und alle Hintergrundaufgaben.

        Raises:
            RuntimeError: Bei Fehlern beim Stoppen der Tasks.
        """

    @abstractmethod
    async def force_revalidation(self) -> ServiceResult:
        """Erzwingt sofortige Revalidierung für alle bekannten Server.

        Returns:
            Dictionary mit Revalidierungs-Ergebnissen pro Domain.

        Raises:
            RuntimeError: Bei Revalidierungs-Fehlern.
        """

    @abstractmethod
    async def add_domain(self, domain: str) -> OperationResult:
        """Fügt eine Domain zur Überwachung hinzu.

        Args:
            domain: Domain-Name zur Überwachung.

        Returns:
            True bei erfolgreichem Hinzufügen.

        Raises:
            ValueError: Bei ungültigem Domain-Namen.
        """

    @abstractmethod
    async def remove_domain(self, domain: str) -> OperationResult:
        """Entfernt eine Domain aus der Überwachung.

        Args:
            domain: Domain-Name zum Entfernen.

        Returns:
            True bei erfolgreichem Entfernen.

        Raises:
            ValueError: Bei ungültigem Domain-Namen.
        """

    @abstractmethod
    async def get_monitored_domains(self) -> list[str]:
        """Liefert Liste aller überwachten Domains.

        Returns:
            Liste der überwachten Domain-Namen.
        """


# Backward Compatibility Alias
DomainRevalidationServiceInterface = DomainRevalidationService
