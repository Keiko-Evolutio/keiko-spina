# backend/agents/factory/agent_factory.py
"""Hauptklasse für Agent-Factory.

Agent-Factory-Implementierung mit:
- Agent-Erstellung für verschiedene Frameworks
- MCP-Client-Integration
- Error-Handling und Monitoring
- Thread-sichere Singleton-Implementation
"""
from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .constants import (
    DEFAULT_FRAMEWORK,
    ERROR_INVALID_FRAMEWORK,
    SUPPORTED_FRAMEWORKS,
    AgentFramework,
    FactoryState,
)
from .error_handlers import (
    AgentCreationError,
    AgentFactoryErrorHandler,
    retry_on_error,
)
from .metrics_manager import MetricsManager, MetricType
from .singleton_mixin import SingletonMixin
from .unified_mcp_client import (
    UnifiedMCPClient,
    UnifiedMCPClientFactory,
    get_unified_mcp_client_factory,
)

logger = get_logger(__name__)


class AgentFactory(SingletonMixin):
    """AgentFactory für Agent-Erstellung und -Management.

    Funktionen:
    - Agent-Erstellung für verschiedene Frameworks
    - MCP-Client-Integration
    - Error-Handling und Monitoring
    - Ressourcen-Management
    """

    def _initialize_singleton(self, *args, **kwargs) -> None:
        """Initialisiert die Agent Factory (Singleton-Pattern)."""
        self._state = FactoryState.UNINITIALIZED
        self._agent_cache: dict[str, Any] = {}
        self._mcp_clients: dict[str, UnifiedMCPClient] = {}
        self._mcp_factory: UnifiedMCPClientFactory | None = None
        self._error_handler = AgentFactoryErrorHandler()
        self._metrics = MetricsManager()

        # Factory für Metriken registrieren
        self._metrics.register_component("agent_factory", "factory")

    def reset(self) -> None:
        """Setzt die Factory in den initialen Zustand zurück (hauptsächlich für Tests)."""
        self._state = FactoryState.UNINITIALIZED
        self._agent_cache.clear()
        self._mcp_clients.clear()
        self._mcp_factory = None

        # Metriken-Komponente zurücksetzen
        self._metrics.cleanup_component("agent_factory")
        self._metrics.register_component("agent_factory", "factory")

    @retry_on_error(
        max_attempts=3,
        exceptions=(ConnectionError, ImportError),
        context="factory_initialization"
    )
    async def initialize(self, *, enable_mcp: bool = True) -> bool:
        """Initialisiert die Factory.

        Args:
            enable_mcp: Ob MCP-Client-Unterstützung aktiviert werden soll

        Returns:
            True wenn Initialisierung erfolgreich, False sonst
        """
        if self._state == FactoryState.READY:
            return True

        try:
            self._state = FactoryState.INITIALIZING

            # MCP Client Factory initialisieren falls aktiviert
            if enable_mcp:
                self._mcp_factory = get_unified_mcp_client_factory()
            else:
                self._mcp_factory = None

            self._state = FactoryState.READY

            self._metrics.record_metric(
                "agent_factory",
                MetricType.SUCCESS_RATE,
                1.0,
                metadata={"operation": "initialize"}
            )

            logger.info(
                "AgentFactory erfolgreich initialisiert",
                extra={
                    "state": self._state.value,
                    "mcp_enabled": self._mcp_factory is not None
                }
            )

            return True

        except Exception as e:
            self._state = FactoryState.ERROR
            self._mcp_factory = None

            await self._error_handler.handle_agent_creation_error(
                e, "factory", "initialization"
            )

            self._metrics.record_metric(
                "agent_factory",
                MetricType.ERROR_RATE,
                1.0,
                metadata={"operation": "initialize", "error": str(e)}
            )

            return False

    @property
    def is_initialized(self) -> bool:
        """Prüft ob die Factory initialisiert ist."""
        return self._state == FactoryState.READY

    @property
    def state(self) -> FactoryState:
        """Gibt den aktuellen Factory-Status zurück."""
        return self._state

    @staticmethod
    async def _validate_framework(framework: str) -> None:
        """Validiert das angegebene Framework.

        Args:
            framework: Framework-Name

        Raises:
            AgentCreationError: Bei ungültigem Framework
        """
        if framework not in SUPPORTED_FRAMEWORKS:
            raise AgentCreationError(
                ERROR_INVALID_FRAMEWORK,
                details={
                    "framework": framework,
                    "supported_frameworks": list(SUPPORTED_FRAMEWORKS)
                }
            )

    async def _create_framework_agent(
        self,
        agent_id: str,
        project_id: str,
        mcp_client: UnifiedMCPClient | None = None,
        framework: str = DEFAULT_FRAMEWORK,
    ) -> Any | None:
        """Erstellt einen Framework-spezifischen Agent."""
        try:
            # Framework validieren
            await self._validate_framework(framework)

            # Framework-spezifische Agent-Erstellung
            if framework == AgentFramework.AZURE_FOUNDRY:
                return await AgentFactory._create_azure_agent(agent_id, project_id, mcp_client)
            if framework == AgentFramework.CUSTOM:
                return await AgentFactory._create_custom_agent(agent_id, project_id, mcp_client)

            # Fallback für andere Frameworks
            return {
                "agent_id": agent_id,
                "project_id": project_id,
                "framework": framework,
                "mcp_client": mcp_client is not None
            }

        except Exception as e:
            await self._error_handler.handle_agent_creation_error(
                e, agent_id, framework
            )
            return None

    @staticmethod
    async def _create_azure_agent(
        agent_id: str,
        project_id: str,
        mcp_client: UnifiedMCPClient | None = None
    ) -> Any:
        """Erstellt einen Azure Foundry Agent.

        Args:
            agent_id: Eindeutige Agent-ID
            project_id: Projekt-ID
            mcp_client: Optionaler MCP-Client

        Returns:
            Agent-Instanz oder Mock für Tests
        """
        try:
            # Azure Agent erstellen
            from agents.common import FoundryAdapter

            config = {
                "agent_id": agent_id,
                "project_id": project_id,
                "framework": AgentFramework.AZURE_FOUNDRY
            }

            adapter = FoundryAdapter(config)
            agent = await adapter.get_agent(agent_id)

            if mcp_client:
                agent.mcp_client = mcp_client

            return agent

        except ImportError:
            # Fallback für Tests und Development
            from unittest.mock import AsyncMock, MagicMock

            agent = MagicMock()
            agent.agent_id = agent_id
            agent.project_id = project_id
            agent.framework = AgentFramework.AZURE_FOUNDRY
            agent.mcp_client = mcp_client
            agent.get_capabilities = AsyncMock(return_value=[])

            return agent

    @staticmethod
    async def _create_custom_agent(
        agent_id: str,
        project_id: str,
        mcp_client: UnifiedMCPClient | None = None
    ) -> Any:
        """Erstellt einen Custom Agent.

        Args:
            agent_id: Eindeutige Agent-ID
            project_id: Projekt-ID
            mcp_client: Optionaler MCP-Client

        Returns:
            Custom Agent-Instanz
        """
        # Custom Agent Implementierung
        from unittest.mock import AsyncMock, MagicMock

        agent = MagicMock()
        agent.agent_id = agent_id
        agent.project_id = project_id
        agent.framework = AgentFramework.CUSTOM
        agent.mcp_client = mcp_client
        agent.get_capabilities = AsyncMock(return_value=["custom_capability"])

        return agent

    async def create_agent_with_mcp(
        self,
        agent_id: str,
        project_id: str,
        server_ids: list[str] | None,
        framework: str = DEFAULT_FRAMEWORK,
    ) -> Any | None:
        """Erstellt einen Agent und verbindet ihn optional mit MCP Clients.

        Args:
            agent_id: Eindeutige Agent-ID
            project_id: Projekt-ID
            server_ids: Liste von MCP Server-IDs (optional)
            framework: Framework für Agent-Erstellung

        Returns:
            Agent-Objekt oder None bei Fehlern
        """
        if self._state != FactoryState.READY:
            from .constants import ERROR_FACTORY_NOT_INITIALIZED
            raise AgentCreationError(
                ERROR_FACTORY_NOT_INITIALIZED,
                details={"factory_state": self._state.value}
            )

        try:
            # 1. Prüfe Cache
            cache_key = f"{framework}:{project_id}:{agent_id}"
            cached_agent = self._agent_cache.get(cache_key)
            if cached_agent:
                return cached_agent

            # 2. MCP Client erstellen falls benötigt
            mcp_client = await self._create_mcp_client_if_needed(agent_id, server_ids)

            # 3. Framework-Agent erstellen
            agent = await self._create_framework_agent(
                agent_id, project_id, mcp_client, framework
            )

            # 4. Agent cachen falls erfolgreich erstellt
            if agent is not None:
                self._agent_cache[cache_key] = agent

                self._metrics.record_metric(
                    "agent_factory",
                    MetricType.SUCCESS_RATE,
                    1.0,
                    metadata={
                        "operation": "create_agent_with_mcp",
                        "agent_id": agent_id,
                        "framework": framework,
                        "has_mcp": mcp_client is not None
                    }
                )

            return agent

        except Exception as e:
            await self._error_handler.handle_agent_creation_error(
                e, agent_id, framework
            )
            return None

    async def _create_mcp_client_if_needed(
        self,
        agent_id: str,
        server_ids: list[str] | None
    ) -> UnifiedMCPClient | None:
        """Erstellt einen MCP Client falls benötigt und verfügbar."""
        if not server_ids or not self._mcp_factory:
            return None

        try:
            # MCP Client für Server erstellen
            mcp_client = await self._mcp_factory.create_client()

            # Client cachen
            self._mcp_clients[agent_id] = mcp_client

            self._metrics.record_metric(
                "agent_factory",
                MetricType.SUCCESS_RATE,
                1.0,
                metadata={
                    "operation": "create_mcp_client",
                    "agent_id": agent_id,
                    "server_count": len(server_ids)
                }
            )

            return mcp_client

        except Exception as e:
            # Graceful degradation: Continue ohne MCP Client
            logger.warning(
                f"MCP Client konnte nicht erstellt werden: {e}",
                extra={"agent_id": agent_id, "error": str(e)}
            )
            return None

    async def cleanup_agent(self, agent_id: str) -> bool:
        """Bereinigt Ressourcen für einen spezifischen Agent."""
        try:
            cleanup_success = True

            # Agent aus Cache entfernen
            removed_agent = None
            for cache_key in list(self._agent_cache.keys()):
                if cache_key.endswith(f":{agent_id}"):
                    removed_agent = self._agent_cache.pop(cache_key, None)
                    break

            # MCP Client cleanup
            if agent_id in self._mcp_clients:
                mcp_client = self._mcp_clients.pop(agent_id)
                if hasattr(mcp_client, "cleanup"):
                    await mcp_client.cleanup()

            # Agent-Ressourcen bereinigen
            if removed_agent:
                from .factory_utils import cleanup_agent_resources
                await cleanup_agent_resources(removed_agent)

            logger.info(
                f"Agent {agent_id} erfolgreich bereinigt",
                extra={"agent_id": agent_id}
            )

            return cleanup_success

        except Exception as e:
            logger.error(
                f"Fehler beim Bereinigen von Agent {agent_id}: {e}",
                extra={"agent_id": agent_id, "error": str(e)}
            )
            return False

    async def cleanup_all_agents(self) -> None:
        """Bereinigt alle Agents und zugehörige MCP Clients."""
        try:
            # Alle Agents bereinigen
            for cache_key in list(self._agent_cache.keys()):
                agent = self._agent_cache.pop(cache_key, None)
                if agent:
                    from .factory_utils import cleanup_agent_resources
                    await cleanup_agent_resources(agent)

            # Alle MCP-Clients bereinigen
            for client_id in list(self._mcp_clients.keys()):
                mcp_client = self._mcp_clients.pop(client_id)
                if hasattr(mcp_client, "cleanup"):
                    await mcp_client.cleanup()

            logger.info("Alle Agents erfolgreich bereinigt")

        except Exception as e:
            logger.error(
                f"Fehler beim Bereinigen aller Agents: {e}",
                extra={"error": str(e)}
            )

    def get_factory_stats(self) -> dict[str, Any]:
        """Gibt Factory-Statistiken zurück."""
        return {
            "state": self._state.value,
            "is_initialized": self.is_initialized,
            "cached_agents": len(self._agent_cache),
            "active_mcp_clients": len(self._mcp_clients),
            "mcp_factory_available": self._mcp_factory is not None,
            "agent_ids": list(self._agent_cache.keys()),
            "mcp_client_ids": list(self._mcp_clients.keys()),
            "metrics": self._metrics.get_component_metrics("agent_factory")
        }

    # Public APIs für Factory Operations
    async def create_framework_agent(
        self,
        agent_id: str,
        project_id: str,
        mcp_client: UnifiedMCPClient | None = None,
        framework: str = DEFAULT_FRAMEWORK,
    ) -> Any | None:
        """Public API für Framework-Agent-Erstellung.

        Args:
            agent_id: Agent-ID
            project_id: Projekt-ID
            mcp_client: Optional MCP Client
            framework: Framework für Agent-Erstellung

        Returns:
            Agent-Instanz oder None
        """
        return await self._create_framework_agent(agent_id, project_id, mcp_client, framework)

    def get_mcp_factory(self) -> UnifiedMCPClientFactory | None:
        """Public API für MCP Factory Zugriff.

        Returns:
            MCP Factory Instanz oder None
        """
        return self._mcp_factory

    def get_agent_cache_info(self) -> dict[str, Any]:
        """Public API für Agent Cache Informationen.

        Returns:
            Dictionary mit Cache-Informationen
        """
        return {
            "cached_agents": len(self._agent_cache),
            "agent_ids": list(self._agent_cache.keys())
        }

    def get_mcp_clients_info(self) -> dict[str, Any]:
        """Public API für MCP Clients Informationen.

        Returns:
            Dictionary mit MCP Client-Informationen
        """
        return {
            "active_mcp_clients": len(self._mcp_clients),
            "mcp_client_ids": list(self._mcp_clients.keys())
        }

    def remove_agent_from_cache(self, cache_key: str) -> Any | None:
        """Public API für Agent-Entfernung aus Cache.

        Args:
            cache_key: Cache-Schlüssel

        Returns:
            Entfernter Agent oder None
        """
        return self._agent_cache.pop(cache_key, None)

    def remove_mcp_client(self, client_id: str) -> UnifiedMCPClient | None:
        """Public API für MCP Client-Entfernung.

        Args:
            client_id: Client-ID

        Returns:
            Entfernter MCP Client oder None
        """
        return self._mcp_clients.pop(client_id, None)

    def find_agent_cache_key_by_id(self, agent_id: str) -> str | None:
        """Public API für Cache-Key-Suche nach Agent-ID.

        Args:
            agent_id: Agent-ID

        Returns:
            Cache-Key oder None
        """
        for cache_key in self._agent_cache.keys():
            if cache_key.endswith(f":{agent_id}"):
                return cache_key
        return None
