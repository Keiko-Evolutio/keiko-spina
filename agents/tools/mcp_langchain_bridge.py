"""MCP↔LangChain Tool-Bridge.

Ermöglicht die Integration von MCP-Tools in LangChain-Workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

logger = get_logger(__name__)


# LangChain ist optional - für Tests und generische Nutzung nicht erforderlich
LANGCHAIN_AVAILABLE = False


@dataclass
class BridgeConfig:
    """Konfiguration für die MCP↔LC-Bridge."""

    tool_name_prefix: str = "mcp_"
    auto_register: bool = True


def create_langchain_tool_from_mcp(
    *,
    tool_id: str,
    mcp_invoke: Callable[[str, dict[str, Any]], Awaitable[Any]],
    bridge_config: BridgeConfig | None = None,
):
    """Erstellt ein LangChain-Tool, das einen MCP-Toolaufruf kapselt.

    Args:
        tool_id: Eindeutige Tool-ID auf MCP-Seite.
        mcp_invoke: Aufruf-Funktion: parameters -> Ergebnis.
        bridge_config: Optionale Bridge-Konfiguration.

    Returns:
        LangChain-Tool-Funktion oder Fallback-Funktion.
    """
    bridge_config or BridgeConfig()

    async def _fallback_tool(**kwargs: Any) -> Any:
        """Fallback, falls LangChain nicht verfügbar ist."""
        return await mcp_invoke(tool_id, kwargs)

    # Fallback-Funktion für Tests und generische Nutzung
    return _fallback_tool


async def register_mcp_tools_as_langchain(
    *,
    discover: Callable[[], Awaitable[Iterable[dict[str, Any]]]],
    create_tool: Callable[
        [str, Callable[[str, dict[str, Any]], Awaitable[Any]], BridgeConfig | None], Any
    ],
    mcp_invoke: Callable[[str, dict[str, Any]], Awaitable[Any]],
    bridge_config: BridgeConfig | None = None,
) -> list[Any]:
    """Registriert alle entdeckten MCP-Tools als LangChain-Tools.

    Args:
        discover: Asynchrone Funktion, die eine Iterable von Tool-Deskriptoren liefert.
        create_tool: Fabrikfunktion zur Erstellung einzelner LC-Tools.
        mcp_invoke: Aufruf-Funktion in Richtung MCP.
        bridge_config: Optionale Bridge-Konfiguration.

    Returns:
        Liste erstellter LangChain-Tool-Objekte/Funktionen.
    """
    tools: list[Any] = []
    try:
        descriptors = await discover()
        for desc in descriptors:
            tool_id = str(desc.get("id") or desc.get("name") or "tool")
            lc_tool = create_tool(tool_id, mcp_invoke, bridge_config)
            tools.append(lc_tool)
    except Exception as exc:
        logger.warning(f"MCP Tool-Discovery fehlgeschlagen: {exc}")
    return tools


class LangChainToolInvoker:
    """Erlaubt MCP-Seite, LangChain-Tools aufzurufen (Richtung MCP → LC).

    Diese Klasse hält eine Registry beliebiger async-Tools (callables) und
    stellt eine einheitliche `invoke`-Schnittstelle bereit.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {}

    def register(self, name: str, func: Callable[[dict[str, Any]], Awaitable[Any]]) -> None:
        """Registriert ein Tool in der lokalen Registry."""
        self._registry[name] = func

    async def invoke(self, name: str, params: dict[str, Any]) -> Any:
        """Ruft ein registriertes Tool auf oder wirft Fehler, wenn unbekannt."""
        func = self._registry.get(name)
        if not func:
            from core.exceptions import KeikoValidationError

            raise KeikoValidationError("Unbekanntes LC-Tool", details={"name": name})
        return await func(params)


__all__ = [
    "BridgeConfig",
    "LangChainToolInvoker",
    "create_langchain_tool_from_mcp",
    "register_mcp_tools_as_langchain",
]
