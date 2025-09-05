"""Registry-Kern-Funktionalität.

Zentrale Funktionen für Registry-Operationen.
"""

from datetime import datetime
from typing import Any

from kei_logging import get_logger

from ..utils.types import AgentCollection, AgentID, AgentInstance

logger = get_logger(__name__)


class RegistryCore:
    """Kern-Funktionalität für Registry-Operationen."""

    def __init__(self):
        """Initialisiert den Registry-Kern."""
        self._agents: AgentCollection = {}
        self._metadata: dict[AgentID, dict[str, Any]] = {}
        self._last_update = datetime.now()

    def add_agent(self, agent_id: AgentID, agent: AgentInstance) -> bool:
        """Fügt einen Agent hinzu.

        Args:
            agent_id: Agent-ID
            agent: Agent-Instanz

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        self._agents[agent_id] = agent
        self._metadata[agent_id] = {
            "added_at": datetime.now(),
            "last_accessed": None,
            "access_count": 0,
        }
        self._last_update = datetime.now()
        return True

    def remove_agent(self, agent_id: AgentID) -> bool:
        """Entfernt einen Agent.

        Args:
            agent_id: Agent-ID

        Returns:
            True wenn erfolgreich entfernt
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._metadata[agent_id]
            self._last_update = datetime.now()
            return True
        return False

    def get_agent(self, agent_id: AgentID) -> AgentInstance | None:
        """Holt einen Agent.

        Args:
            agent_id: Agent-ID

        Returns:
            Agent-Instanz oder None
        """
        if agent_id in self._agents:
            # Update Zugriffs-Metadaten
            self._metadata[agent_id]["last_accessed"] = datetime.now()
            self._metadata[agent_id]["access_count"] += 1
            return self._agents[agent_id]
        return None

    def list_agents(self) -> list[AgentID]:
        """Listet alle Agent-IDs auf.

        Returns:
            Liste der Agent-IDs
        """
        return list(self._agents.keys())

    def get_agent_count(self) -> int:
        """Gibt Anzahl der Agents zurück.

        Returns:
            Anzahl der Agents
        """
        return len(self._agents)

    def clear_agents(self) -> None:
        """Löscht alle Agents."""
        self._agents.clear()
        self._metadata.clear()
        self._last_update = datetime.now()

    def get_agent_metadata(self, agent_id: AgentID) -> dict[str, Any] | None:
        """Holt Agent-Metadaten.

        Args:
            agent_id: Agent-ID

        Returns:
            Metadaten-Dictionary oder None
        """
        return self._metadata.get(agent_id)

    def get_all_agents(self) -> AgentCollection:
        """Holt alle Agents.

        Returns:
            Agent-Collection
        """
        return self._agents.copy()

    def has_agent(self, agent_id: AgentID) -> bool:
        """Prüft ob Agent existiert.

        Args:
            agent_id: Agent-ID

        Returns:
            True wenn Agent existiert
        """
        return agent_id in self._agents

    def get_last_update(self) -> datetime:
        """Gibt Zeitpunkt der letzten Aktualisierung zurück.

        Returns:
            Zeitpunkt der letzten Aktualisierung
        """
        return self._last_update
