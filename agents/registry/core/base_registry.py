"""Abstrakte Basis-Klasse für alle Registry-Implementierungen.

Definiert die gemeinsame Schnittstelle zwischen verschiedenen Registry-Typen.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from ..exceptions import AgentNotFoundError, RegistryError
from ..utils.constants import CacheConstants, ErrorConstants
from ..utils.types import (
    AgentCollection,
    AgentID,
    AgentInstance,
    CapabilityList,
    Timestamp,
)

logger = get_logger(__name__)


class BaseRegistry(ABC):
    """Abstrakte Basis-Klasse für Registry-Implementierungen.

    Definiert die gemeinsame Schnittstelle und Basis-Funktionalität
    für alle Registry-Typen.
    """

    def __init__(self, max_cache_age: timedelta | None = None):
        """Initialisiert die Basis-Registry.

        Args:
            max_cache_age: Maximales Cache-Alter
        """
        self._agents: AgentCollection = {}
        self._max_cache_age = max_cache_age or CacheConstants.DEFAULT_CACHE_AGE
        self._last_refresh: Timestamp | None = None
        self._last_error: str | None = None
        self._initialized = False

        # Performance-Metriken
        self._metrics = {
            "search_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "refresh_count": 0,
        }

    @property
    def agents(self) -> AgentCollection:
        """Gibt die Agent-Collection zurück."""
        return self._agents

    @property
    def is_initialized(self) -> bool:
        """Prüft ob Registry initialisiert ist."""
        return self._initialized

    @property
    def last_refresh(self) -> Timestamp | None:
        """Zeitpunkt der letzten Aktualisierung."""
        return self._last_refresh

    @property
    def last_error(self) -> str | None:
        """Letzter aufgetretener Fehler."""
        return self._last_error

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialisiert die Registry.

        Returns:
            True wenn erfolgreich initialisiert
        """

    @abstractmethod
    async def refresh_agents(self) -> None:
        """Lädt alle verfügbaren Agents neu."""

    @abstractmethod
    async def search_agents(
        self,
        query: str | None = None,
        capabilities: CapabilityList | None = None,
        limit: int = 10,
        **kwargs,
    ) -> list[Any]:
        """Sucht Agents basierend auf Kriterien.

        Args:
            query: Suchtext
            capabilities: Erforderliche Capabilities
            limit: Maximale Anzahl Ergebnisse
            **kwargs: Zusätzliche Suchparameter

        Returns:
            Liste von Agent-Matches
        """

    async def get_agent_by_id(self, agent_id: AgentID) -> AgentInstance | None:
        """Holt einen Agent anhand seiner ID.

        Args:
            agent_id: Agent-ID

        Returns:
            Agent-Instanz oder None
        """
        if self._is_cache_expired():
            await self.refresh_agents()

        agent = self._agents.get(agent_id)
        if agent:
            self._metrics["cache_hits"] += 1
        else:
            self._metrics["cache_misses"] += 1

        return agent

    async def register_agent(
        self,
        agent_id: AgentID,
        agent: AgentInstance,
        overwrite: bool = False,
    ) -> bool:
        """Registriert einen Agent.

        Args:
            agent_id: Agent-ID
            agent: Agent-Instanz
            overwrite: Überschreiben erlauben

        Returns:
            True wenn erfolgreich registriert

        Raises:
            RegistryError: Bei Registrierungsfehlern
        """
        if not overwrite and agent_id in self._agents:
            raise RegistryError(
                f"Agent {agent_id} bereits registriert",
                error_code=ErrorConstants.DUPLICATE_REGISTRATION,
            )

        self._agents[agent_id] = agent
        logger.debug(f"Agent {agent_id} registriert")
        return True

    async def unregister_agent(self, agent_id: AgentID) -> bool:
        """Entfernt einen Agent aus der Registry.

        Args:
            agent_id: Agent-ID

        Returns:
            True wenn erfolgreich entfernt

        Raises:
            AgentNotFoundError: Wenn Agent nicht gefunden
        """
        if agent_id not in self._agents:
            raise AgentNotFoundError(
                f"Agent {agent_id} nicht gefunden",
                error_code=ErrorConstants.AGENT_NOT_FOUND,
            )

        del self._agents[agent_id]
        logger.debug(f"Agent {agent_id} entfernt")
        return True

    def get_status(self) -> dict[str, Any]:
        """Gibt Registry-Status zurück.

        Returns:
            Status-Dictionary
        """
        return {
            "total_agents": len(self._agents),
            "initialized": self._initialized,
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "cache_expired": self._is_cache_expired(),
            "last_error": self._last_error,
            "metrics": self._metrics.copy(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Performance-Metriken zurück.

        Returns:
            Metriken-Dictionary
        """
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Setzt Performance-Metriken zurück."""
        self._metrics = {
            "search_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "refresh_count": 0,
        }

    async def cleanup(self) -> None:
        """Bereinigt Registry-Ressourcen."""
        self._agents.clear()
        self._last_refresh = None
        self._initialized = False
        self.reset_metrics()
        logger.debug("Registry bereinigt")

    def _is_cache_expired(self) -> bool:
        """Prüft ob Cache abgelaufen ist.

        Returns:
            True wenn Cache abgelaufen ist
        """
        if not self._last_refresh:
            return True

        return datetime.now() - self._last_refresh > self._max_cache_age

    def _update_metrics(self, operation: str) -> None:
        """Aktualisiert Performance-Metriken.

        Args:
            operation: Name der Operation
        """
        if operation in self._metrics:
            self._metrics[operation] += 1

    def _set_error(self, error: str) -> None:
        """Setzt den letzten Fehler.

        Args:
            error: Fehlermeldung
        """
        self._last_error = error
        logger.error(f"Registry-Fehler: {error}")

    def _clear_error(self) -> None:
        """Löscht den letzten Fehler."""
        self._last_error = None
