# backend/agents/factory/factory_api.py
"""Öffentliche API für das Factory-Modul.

API-Funktionen für:
- Agent-Erstellung und -Management
- Factory-Status und -Kontrolle
- Cleanup-Operationen
"""
from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .agent_factory import AgentFactory
from .constants import AgentFramework, FactoryState
from .factory_utils import validate_agent_id, validate_project_id

logger = get_logger(__name__)


async def get_factory() -> AgentFactory:
    """Gibt die Singleton-Factory-Instanz zurück.

    Initialisiert die Factory automatisch falls nötig.

    Returns:
        AgentFactory-Instanz

    Raises:
        RuntimeError: Wenn Factory nicht initialisiert werden kann
    """
    factory = AgentFactory()

    if not factory.is_initialized:
        success = await factory.initialize()
        if not success:
            raise RuntimeError("Factory konnte nicht initialisiert werden")

    return factory


async def create_agent(
    agent_id: str,
    project_id: str,
    framework: str = AgentFramework.AZURE_FOUNDRY
) -> Any | None:
    """Erstellt einen neuen Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        project_id: Projekt-ID
        framework: Framework für Agent-Erstellung

    Returns:
        Agent-Instanz oder None bei Fehler

    Raises:
        ValueError: Bei ungültigen Parametern
    """
    # Validierung
    if not validate_agent_id(agent_id):
        raise ValueError(f"Ungültige Agent-ID: {agent_id}")

    if not validate_project_id(project_id):
        raise ValueError(f"Ungültige Projekt-ID: {project_id}")

    factory = await get_factory()
    return await factory.create_framework_agent(agent_id, project_id, framework=framework)


async def create_agent_with_mcp(
    agent_id: str,
    project_id: str,
    mcp_servers: list[str] | None = None,
    framework: str = AgentFramework.AZURE_FOUNDRY
) -> Any | None:
    """Erstellt einen Agent mit MCP-Integration.

    Args:
        agent_id: Eindeutige Agent-ID
        project_id: Projekt-ID
        mcp_servers: Liste der MCP-Server-Endpunkte
        framework: Framework für Agent-Erstellung

    Returns:
        Agent-Instanz mit MCP-Client oder None bei Fehler
    """
    # Validierung
    if not validate_agent_id(agent_id):
        raise ValueError(f"Ungültige Agent-ID: {agent_id}")

    if not validate_project_id(project_id):
        raise ValueError(f"Ungültige Projekt-ID: {project_id}")

    factory = await get_factory()

    # MCP-Client erstellen falls Server angegeben
    mcp_client = None
    mcp_factory = factory.get_mcp_factory()
    if mcp_servers and mcp_factory:
        try:
            mcp_client = await mcp_factory.create_client()
            # MCP-Server-Konfiguration
            logger.debug(
                f"MCP-Client für Agent {agent_id} erstellt",
                extra={"mcp_servers": mcp_servers}
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                f"MCP-Client konnte nicht erstellt werden - Verbindungsproblem: {e}",
                extra={"agent_id": agent_id, "error": str(e)}
            )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"MCP-Client konnte nicht erstellt werden - Konfigurationsfehler: {e}",
                extra={"agent_id": agent_id, "error": str(e)}
            )
        except Exception as e:
            logger.warning(
                f"MCP-Client konnte nicht erstellt werden - Unerwarteter Fehler: {e}",
                extra={"agent_id": agent_id, "error": str(e)}
            )

    return await factory.create_framework_agent(
        agent_id, project_id, mcp_client, framework
    )


async def cleanup_agent(agent_id: str) -> bool:
    """Bereinigt einen spezifischen Agent.

    Args:
        agent_id: ID des zu bereinigenden Agents

    Returns:
        True wenn erfolgreich, False sonst
    """
    try:
        factory = AgentFactory()

        # Agent aus Cache entfernen
        removed_agent = None
        cache_key = factory.find_agent_cache_key_by_id(agent_id)
        if cache_key:
            removed_agent = factory.remove_agent_from_cache(cache_key)

        # MCP-Client bereinigen
        mcp_clients_info = factory.get_mcp_clients_info()
        if agent_id in mcp_clients_info["mcp_client_ids"]:
            mcp_client = factory.remove_mcp_client(agent_id)
            if mcp_client and hasattr(mcp_client, "cleanup"):
                await mcp_client.cleanup()

        # Agent-Ressourcen bereinigen
        if removed_agent:
            from .factory_utils import cleanup_agent_resources
            await cleanup_agent_resources(removed_agent)

        logger.info(
            f"Agent {agent_id} erfolgreich bereinigt",
            extra={"agent_id": agent_id}
        )

        return True

    except Exception as e:
        logger.error(
            f"Fehler beim Bereinigen von Agent {agent_id}: {e}",
            extra={"agent_id": agent_id, "error": str(e)}
        )
        return False


async def cleanup_all_agents() -> None:
    """Bereinigt alle Agents und zugehörige MCP Clients der Singleton Factory."""
    try:
        factory = AgentFactory()

        # Alle Agents bereinigen
        cache_info = factory.get_agent_cache_info()
        for cache_key in list(cache_info["agent_ids"]):
            agent = factory.remove_agent_from_cache(cache_key)
            if agent:
                from .factory_utils import cleanup_agent_resources
                await cleanup_agent_resources(agent)

        # Alle MCP-Clients bereinigen
        mcp_clients_info = factory.get_mcp_clients_info()
        for client_id in list(mcp_clients_info["mcp_client_ids"]):
            mcp_client = factory.remove_mcp_client(client_id)
            if mcp_client and hasattr(mcp_client, "cleanup"):
                await mcp_client.cleanup()

        # Factory-Status zurücksetzen
        factory.reset()

        logger.info("Alle Agents erfolgreich bereinigt")

    except Exception as e:
        logger.error(
            f"Fehler beim Bereinigen aller Agents: {e}",
            extra={"error": str(e)}
        )


def is_factory_available() -> bool:
    """Prüft ob die Factory verfügbar und initialisiert ist."""
    try:
        factory = AgentFactory()
        return factory.is_initialized
    except Exception:
        return False


def get_factory_status() -> dict[str, Any]:
    """Gibt den aktuellen Status der Factory zurück."""
    try:
        factory = AgentFactory()
        return factory.get_factory_stats()
    except Exception:
        return {
            "available": False,
            "state": FactoryState.ERROR.value,
            "is_initialized": False,
            "cached_agents": 0,
            "active_mcp_clients": 0,
            "error": "Factory nicht verfügbar"
        }


async def create_azure_foundry_agent_with_mcp(
    agent_id: str,
    project_id: str,
    mcp_servers: list[str] | None = None,
    framework: str = AgentFramework.AZURE_FOUNDRY
) -> Any | None:
    """Azure Foundry Agent-Erstellung mit MCP Integration.

    Args:
        agent_id: Eindeutige Agent-ID
        project_id: Projekt-ID
        mcp_servers: Liste der MCP-Server-Endpunkte
        framework: Framework (default: AZURE_FOUNDRY)

    Returns:
        Agent-Instanz oder None bei Fehler
    """
    return await create_agent_with_mcp(agent_id, project_id, mcp_servers, framework)




__all__ = [
    "cleanup_agent",
    "cleanup_all_agents",
    "create_agent",
    "create_agent_with_mcp",
    "create_azure_foundry_agent_with_mcp",
    "get_factory",
    "get_factory_status",
    "is_factory_available",
]
