"""AgentService Interface-Definition."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import CoreService

if TYPE_CHECKING:
    from ._types import AgentId, CapabilityConfig, ServiceResult, TaskPayload


class AgentService(CoreService):
    """Definiert den Vertrag für den Agenten-Service.

    Kern-Service für Agent-Management und Task-Execution.
    """

    @abstractmethod
    async def run_task(self, agent_id: AgentId, payload: TaskPayload) -> ServiceResult:
        """Führt eine Aufgabe für einen Agenten aus.

        Args:
            agent_id: Eindeutige ID des Agenten.
            payload: Eingabedaten für die Aufgabe.

        Returns:
            Ergebnisdaten der Ausführung.

        Raises:
            ValueError: Bei ungültiger Agent-ID oder Payload.
            RuntimeError: Bei Ausführungsfehlern.
        """

    @abstractmethod
    async def register_capabilities(self, agent_id: AgentId, capabilities: CapabilityConfig) -> None:
        """Registriert Fähigkeiten eines Agenten.

        Args:
            agent_id: Eindeutige ID des Agenten.
            capabilities: Konfiguration der Agent-Fähigkeiten.

        Raises:
            ValueError: Bei ungültiger Agent-ID oder Capabilities.
        """

    @abstractmethod
    async def get_agent_status(self, agent_id: AgentId) -> ServiceResult:
        """Liefert Status-Informationen für einen Agenten.

        Args:
            agent_id: Eindeutige ID des Agenten.

        Returns:
            Status-Informationen des Agenten.

        Raises:
            ValueError: Bei ungültiger Agent-ID.
            RuntimeError: Wenn Agent nicht gefunden wird.
        """
