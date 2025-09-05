# backend/agents/factory/factory_utils.py
"""Utility-Funktionen für das Factory-Modul.

Utility-Funktionen für:
- Adapter-Erstellung
- Validierungslogik
- Cleanup-Funktionen
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from kei_logging import get_logger

logger = get_logger(__name__)


def create_foundry_adapter(config: dict[str, Any]) -> Any:
    """Erstellt einen Foundry-Adapter für Agent-Erstellung.

    Implementiert robuste Fehlerbehandlung und Logging.
    In Tests wird dies durch Mock ersetzt.

    Args:
        config: Konfiguration für den Adapter

    Returns:
        FoundryAdapter-Instanz oder Mock für Tests
    """
    try:
        # Echten Adapter importieren
        from agents.common import FoundryAdapter

        logger.debug(
            "Erstelle echten FoundryAdapter",
            extra={"config_keys": list(config.keys())}
        )

        return FoundryAdapter(config)

    except ImportError:
        # Fallback für Tests
        logger.debug(
            "Erstelle Mock FoundryAdapter für Tests",
            extra={"config_keys": list(config.keys())}
        )

        adapter = MagicMock()
        adapter.get_agent = AsyncMock()
        adapter.config = config
        return adapter


def validate_agent_id(agent_id: str) -> bool:
    """Validiert eine Agent-ID.

    Args:
        agent_id: Zu validierende Agent-ID

    Returns:
        True wenn gültig, False sonst
    """
    if not agent_id or not isinstance(agent_id, str):
        return False

    # Agent-ID sollte nicht leer sein und keine Sonderzeichen enthalten
    if len(agent_id.strip()) == 0:
        return False

    # Validierung: alphanumerische Zeichen und Bindestriche
    import re
    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, agent_id))


def validate_project_id(project_id: str) -> bool:
    """Validiert eine Projekt-ID.

    Args:
        project_id: Zu validierende Projekt-ID

    Returns:
        True wenn gültig, False sonst
    """
    if not project_id or not isinstance(project_id, str):
        return False

    # Projekt-ID sollte nicht leer sein
    if len(project_id.strip()) == 0:
        return False

    # Validierung: alphanumerische Zeichen und Bindestriche
    import re
    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, project_id))


def sanitize_agent_id(agent_id: str) -> str:
    """Bereinigt eine Agent-ID von ungültigen Zeichen.

    Args:
        agent_id: Zu bereinigende Agent-ID

    Returns:
        Bereinigte Agent-ID
    """
    if not agent_id:
        return ""

    # Entferne Leerzeichen und ersetze ungültige Zeichen
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_id.strip())

    # Entferne mehrfache Unterstriche
    sanitized = re.sub(r"_+", "_", sanitized)

    # Entferne führende/nachfolgende Unterstriche
    sanitized = sanitized.strip("_")

    return sanitized


def create_agent_cache_key(agent_id: str, project_id: str, framework: str) -> str:
    """Erstellt einen Cache-Schlüssel für einen Agent.

    Args:
        agent_id: Agent-ID
        project_id: Projekt-ID
        framework: Framework-Name

    Returns:
        Cache-Schlüssel
    """
    return f"{framework}:{project_id}:{agent_id}"


def parse_agent_cache_key(cache_key: str) -> dict[str, str] | None:
    """Parst einen Agent-Cache-Schlüssel.

    Args:
        cache_key: Cache-Schlüssel

    Returns:
        Dictionary mit framework, project_id, agent_id oder None bei Fehler
    """
    try:
        parts = cache_key.split(":", 2)
        if len(parts) != 3:
            return None

        return {
            "framework": parts[0],
            "project_id": parts[1],
            "agent_id": parts[2]
        }
    except (ValueError, IndexError) as e:
        logger.debug(f"Cache-Key-Parsing fehlgeschlagen - Format-/Index-Fehler: {e}")
        return None
    except Exception as e:
        logger.warning(f"Cache-Key-Parsing fehlgeschlagen - Unerwarteter Fehler: {e}")
        return None


async def cleanup_agent_resources(agent: Any) -> bool:
    """Bereinigt Ressourcen eines Agents.

    Args:
        agent: Agent-Instanz

    Returns:
        True wenn erfolgreich, False sonst
    """
    import asyncio

    try:
        # Agent-spezifische Cleanup-Methoden
        if hasattr(agent, "cleanup"):
            if asyncio.iscoroutinefunction(agent.cleanup):
                await agent.cleanup()
            else:
                agent.cleanup()

        # Bereinige MCP-Client falls vorhanden
        if hasattr(agent, "mcp_client") and agent.mcp_client:
            if hasattr(agent.mcp_client, "cleanup"):
                if asyncio.iscoroutinefunction(agent.mcp_client.cleanup):
                    await agent.mcp_client.cleanup()
                else:
                    agent.mcp_client.cleanup()

        logger.debug(
            "Agent-Ressourcen erfolgreich bereinigt",
            extra={
                "agent_id": getattr(agent, "agent_id", "unknown"),
                "has_mcp_client": hasattr(agent, "mcp_client")
            }
        )

        return True

    except Exception as e:
        logger.warning(
            f"Fehler beim Bereinigen der Agent-Ressourcen: {e}",
            extra={
                "agent_id": getattr(agent, "agent_id", "unknown"),
                "error": str(e)
            }
        )
        return False


def is_test_environment() -> bool:
    """Prüft ob wir in einer Test-Umgebung laufen.

    Returns:
        True wenn Test-Umgebung, False sonst
    """
    import os
    import sys

    # Prüfe auf pytest
    if "pytest" in sys.modules:
        return True

    # Prüfe auf Test-Environment-Variable
    if os.getenv("TESTING", "").lower() in ("true", "1", "yes"):
        return True

    # Prüfe auf unittest
    if "unittest" in sys.modules:
        return True

    return False


def get_default_agent_config(
    agent_id: str,
    project_id: str,
    framework: str
) -> dict[str, Any]:
    """Erstellt eine Standard-Konfiguration für einen Agent.

    Args:
        agent_id: Agent-ID
        project_id: Projekt-ID
        framework: Framework-Name

    Returns:
        Standard-Konfiguration
    """
    return {
        "agent_id": agent_id,
        "project_id": project_id,
        "framework": framework,
        "created_at": None,  # Wird bei Erstellung gesetzt
        "capabilities": [],
        "metadata": {},
        "settings": {
            "timeout": 30,
            "retry_attempts": 3,
            "enable_logging": True
        }
    }




__all__ = [
    "cleanup_agent_resources",
    "create_agent_cache_key",
    "create_foundry_adapter",
    "get_default_agent_config",
    "is_test_environment",
    "parse_agent_cache_key",
    "sanitize_agent_id",
    "validate_agent_id",
    "validate_project_id",
]
