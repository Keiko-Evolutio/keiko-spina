"""Bridge zwischen Agents und externen MCP Servern.

Stellt eine einheitliche Schnittstelle für die Integration externer MCP Server
in Agent-Workflows bereit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agents.tools.mcp.kei_mcp_registry import kei_mcp_registry as mcp_registry
from kei_logging import get_logger
from observability import trace_function

from ..constants import (
    CACHE_SETTINGS,
    MCP_BRIDGE_SETTINGS,
    VALIDATION_PATTERNS,
    get_error_message,
)

if TYPE_CHECKING:
    from agents.tools.mcp.kei_mcp_client import MCPToolResult

logger = get_logger(__name__)


@dataclass(slots=True)
class AgentToolDefinition:
    """Agent-kompatible Tool-Definition.

    Attributes:
        id: Eindeutige Tool-ID im Format "server:tool"
        name: Name des Tools
        description: Beschreibung der Tool-Funktionalität
        parameters: JSON Schema für Parameter
        required: Liste erforderlicher Parameter
        server: Name des MCP Servers
        capabilities: Liste von Capabilities die das Tool unterstützt
        examples: Beispiele für Tool-Verwendung
        metadata: Zusätzliche Metadaten
    """

    id: str
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    server: str
    capabilities: list[str]
    examples: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolExecutionContext:
    """Kontext für Tool-Ausführung.

    Attributes:
        agent_id: ID des ausführenden Agents
        session_id: Session-ID für Kontext
        user_id: Benutzer-ID
        request_id: Eindeutige Request-ID
        metadata: Zusätzliche Kontext-Metadaten
    """

    agent_id: str
    session_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] | None = None


class ExternalMCPBridge:
    """Bridge für Integration externer MCP Server in Agent-Workflows.

    Diese Klasse stellt eine einheitliche Schnittstelle zwischen Keiko Agents
    und externen MCP Servern bereit. Sie übersetzt zwischen den verschiedenen
    Datenformaten und bietet erweiterte Funktionen wie Capability-Mapping
    und intelligente Tool-Auswahl.
    """

    def __init__(self):
        """Initialisiert die MCP Bridge."""
        # Capability-Mappings aus Constants
        self._capability_mappings = {
            key: set(values) for key, values in MCP_BRIDGE_SETTINGS["capability_mappings"].items()
        }

        self._tool_cache: dict[str, list[AgentToolDefinition]] = {}
        self._cache_ttl = CACHE_SETTINGS["ttl"]
        self._last_cache_update = 0

    @trace_function("agents.external_mcp.discover_tools")
    async def discover_available_tools(
        self, refresh_cache: bool = False
    ) -> list[AgentToolDefinition]:
        """Entdeckt alle verfügbaren Tools von externen MCP Servern.

        Aufgeteilte Implementierung für bessere Wartbarkeit und Testbarkeit.

        Args:
            refresh_cache: Ob der Tool-Cache aktualisiert werden soll

        Returns:
            Liste aller verfügbaren Agent-kompatiblen Tools
        """
        # Cache-Prüfung
        if not refresh_cache and self._is_cache_valid():
            return self._get_cached_tools()

        # Tools von allen Servern abrufen
        tools_by_server = mcp_registry.get_all_tools()

        # Tools konvertieren und cachen
        agent_tools = self._convert_tools_to_agent_format(tools_by_server)
        self._update_cache(agent_tools)

        logger.info(
            f"Entdeckte {len(agent_tools)} Tools von {len(tools_by_server)} externen MCP Servern"
        )
        return agent_tools

    def _is_cache_valid(self) -> bool:
        """Prüft ob der Cache noch gültig ist."""
        current_time = time.time()
        return (
            bool(self._tool_cache)
            and current_time - self._last_cache_update < self._cache_ttl
        )

    def _get_cached_tools(self) -> list[AgentToolDefinition]:
        """Gibt gecachte Tools zurück."""
        return next(iter(self._tool_cache.values())) if self._tool_cache else []

    def _convert_tools_to_agent_format(
        self,
        tools_by_server: dict[str, list[dict[str, Any]]]
    ) -> list[AgentToolDefinition]:
        """Konvertiert MCP-Tools zu Agent-kompatiblem Format.

        Args:
            tools_by_server: Tools gruppiert nach Server

        Returns:
            Liste von Agent-Tool-Definitionen
        """
        agent_tools = []
        current_time = time.time()

        for server_name, tools in tools_by_server.items():
            for tool in tools:
                agent_tool = self._create_agent_tool_definition(
                    tool, server_name, current_time
                )
                agent_tools.append(agent_tool)

        return agent_tools

    def _create_agent_tool_definition(
        self,
        tool: dict[str, Any],
        server_name: str,
        discovery_time: float
    ) -> AgentToolDefinition:
        """Erstellt eine Agent-Tool-Definition aus einem MCP-Tool.

        Args:
            tool: MCP-Tool-Definition
            server_name: Name des MCP-Servers
            discovery_time: Zeitpunkt der Entdeckung

        Returns:
            Agent-kompatible Tool-Definition
        """
        capabilities = self._extract_capabilities(tool)

        return AgentToolDefinition(
            id=f"{server_name}:{tool.get('name', 'unknown')}",
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            parameters=tool.get("parameters", {}),
            required=tool.get("required", []),
            server=server_name,
            capabilities=capabilities,
            examples=tool.get("examples"),
            metadata={
                "server_type": "external_mcp",
                "discovery_time": discovery_time
            },
        )

    def _update_cache(self, agent_tools: list[AgentToolDefinition]) -> None:
        """Aktualisiert den Tool-Cache."""
        self._tool_cache = {"all": agent_tools}
        self._last_cache_update = time.time()

    @trace_function("agents.external_mcp.invoke_tool")
    async def invoke_external_tool(
        self,
        tool_id: str,
        parameters: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Ruft externes MCP Tool auf.

        Aufgeteilte Implementierung für bessere Wartbarkeit und Error-Handling.

        Args:
            tool_id: Tool-ID im Format "server:tool"
            parameters: Parameter für den Tool-Aufruf
            context: Ausführungskontext

        Returns:
            Standardisiertes Ergebnis-Dictionary
        """
        # Tool-ID validieren
        validation_result = self._validate_tool_id(tool_id)
        if not validation_result["valid"]:
            return validation_result["error_response"]

        server_name, tool_name = validation_result["parsed"]

        # Parameter mit Kontext anreichern
        enhanced_parameters = self._enhance_parameters_with_context(parameters, context)

        # Tool ausführen
        try:
            result = await mcp_registry.invoke_tool(
                server_name=server_name,
                tool_name=tool_name,
                parameters=enhanced_parameters
            )
            return self._convert_mcp_result_to_agent_format(result, tool_id, context)

        except Exception as exc:
            logger.exception(f"Tool-Ausführung fehlgeschlagen für {tool_id}: {exc}")
            return {
                "success": False,
                "error": get_error_message("tool_execution_error"),
                "error_type": "execution_error",
                "details": str(exc),
            }

    def _validate_tool_id(self, tool_id: str) -> dict[str, Any]:
        """Validiert Tool-ID Format.

        Args:
            tool_id: Zu validierende Tool-ID

        Returns:
            Validierungsergebnis mit parsed Werten oder Error-Response
        """
        import re

        if not re.match(VALIDATION_PATTERNS["tool_id"], tool_id):
            error_msg = f"{get_error_message('invalid_tool_id')}: {tool_id}"
            return {
                "valid": False,
                "error_response": {
                    "success": False,
                    "error": error_msg,
                    "error_type": "invalid_tool_id",
                }
            }

        server_name, tool_name = tool_id.split(":", 1)
        return {
            "valid": True,
            "parsed": (server_name, tool_name)
        }

    def _enhance_parameters_with_context(
        self,
        parameters: dict[str, Any],
        context: ToolExecutionContext | None
    ) -> dict[str, Any]:
        """Reichert Parameter mit Kontext-Informationen an.

        Args:
            parameters: Original-Parameter
            context: Ausführungskontext

        Returns:
            Angereicherte Parameter
        """
        enhanced_parameters = parameters.copy()

        if context:
            enhanced_parameters["_context"] = {
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "user_id": context.user_id,
                "request_id": context.request_id,
                "metadata": context.metadata,
            }

        return enhanced_parameters

    async def get_tools_for_capability(self, capability: str) -> list[AgentToolDefinition]:
        """Filtert externe Tools nach Capability.

        Args:
            capability: Gewünschte Capability

        Returns:
            Liste passender Tools
        """
        all_tools = await self.discover_available_tools()

        # Direkte Capability-Übereinstimmung
        matching_tools = [
            tool
            for tool in all_tools
            if capability.lower() in [cap.lower() for cap in tool.capabilities]
        ]

        # Erweiterte Suche über Capability-Mappings
        if not matching_tools and capability.lower() in self._capability_mappings:
            keywords = self._capability_mappings[capability.lower()]
            for tool in all_tools:
                description = tool.description.lower()
                name = tool.name.lower()

                if any(keyword in description or keyword in name for keyword in keywords):
                    matching_tools.append(tool)

        logger.debug(f"Gefunden {len(matching_tools)} Tools für Capability '{capability}'")
        return matching_tools

    async def get_tools_by_server(self, server_name: str) -> list[AgentToolDefinition]:
        """Gibt alle Tools eines spezifischen Servers zurück.

        Args:
            server_name: Name des MCP Servers

        Returns:
            Liste der Tools des Servers
        """
        all_tools = await self.discover_available_tools()
        return [tool for tool in all_tools if tool.server == server_name]

    async def find_best_tool_for_task(self, task_description: str) -> AgentToolDefinition | None:
        """Findet das beste Tool für eine gegebene Aufgabe.

        Aufgeteilte Implementierung mit konfigurierbaren Scoring-Gewichtungen.

        Args:
            task_description: Beschreibung der Aufgabe

        Returns:
            Bestes passendes Tool oder None
        """
        all_tools = await self.discover_available_tools()

        if not all_tools:
            return None

        # Tools bewerten
        scored_tools = self._score_tools_for_task(all_tools, task_description)

        if scored_tools:
            # Bestes Tool auswählen
            best_tool = scored_tools[0][0]
            logger.info(
                f"Bestes Tool für '{task_description}': {best_tool.name} (Server: {best_tool.server})"
            )
            return best_tool

        return None

    def _score_tools_for_task(
        self,
        tools: list[AgentToolDefinition],
        task_description: str
    ) -> list[tuple[AgentToolDefinition, int]]:
        """Bewertet Tools für eine gegebene Aufgabe.

        Args:
            tools: Liste verfügbarer Tools
            task_description: Aufgabenbeschreibung

        Returns:
            Nach Score sortierte Liste von (Tool, Score) Tupeln
        """
        task_lower = task_description.lower()
        task_words = set(task_lower.split())
        scored_tools = []

        # Scoring-Gewichtungen aus Constants
        weights = MCP_BRIDGE_SETTINGS["scoring_weights"]

        for tool in tools:
            score = self._calculate_tool_score(tool, task_words, task_lower, weights)

            if score > 0:
                scored_tools.append((tool, score))

        # Nach Score sortieren (höchster zuerst)
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools

    def _calculate_tool_score(
        self,
        tool: AgentToolDefinition,
        task_words: set[str],
        task_lower: str,
        weights: dict[str, int]
    ) -> int:
        """Berechnet Score für ein einzelnes Tool.

        Args:
            tool: Tool-Definition
            task_words: Wörter aus der Aufgabenbeschreibung
            task_lower: Aufgabenbeschreibung in Kleinbuchstaben
            weights: Scoring-Gewichtungen

        Returns:
            Berechneter Score
        """
        score = 0

        # Name-Match Score
        tool_name_words = set(tool.name.lower().split())
        name_matches = len(task_words & tool_name_words)
        score += name_matches * weights["name_match"]

        # Beschreibung-Match Score
        description_words = set(tool.description.lower().split())
        description_matches = len(task_words & description_words)
        score += description_matches * weights["description_match"]

        # Capability-Match Score
        for capability in tool.capabilities:
            if capability.lower() in task_lower:
                score += weights["capability_match"]

        return score

    def _extract_capabilities(self, tool: dict[str, Any]) -> list[str]:
        """Extrahiert Capabilities aus Tool-Definition.

        Args:
            tool: Tool-Definition vom MCP Server

        Returns:
            Liste abgeleiteter Capabilities
        """
        capabilities = set()

        # Explizite Capabilities aus Metadaten
        if "capabilities" in tool:
            capabilities.update(tool["capabilities"])

        # Capabilities aus Name ableiten
        name = tool.get("name", "").lower()
        description = tool.get("description", "").lower()

        for capability, keywords in self._capability_mappings.items():
            if any(keyword in name or keyword in description for keyword in keywords):
                capabilities.add(capability)

        return list(capabilities)

    def _convert_mcp_result_to_agent_format(
        self, mcp_result: MCPToolResult, tool_id: str, context: ToolExecutionContext | None
    ) -> dict[str, Any]:
        """Konvertiert MCP-Ergebnis in Agent-kompatibles Format.

        Args:
            mcp_result: Ergebnis vom MCP Server
            tool_id: ID des aufgerufenen Tools
            context: Ausführungskontext

        Returns:
            Standardisiertes Agent-Ergebnis
        """
        result = {
            "success": mcp_result.success,
            "tool_id": tool_id,
            "server": mcp_result.server,
            "execution_time_ms": mcp_result.execution_time_ms,
        }

        if mcp_result.success:
            result["result"] = mcp_result.result
            result["metadata"] = mcp_result.metadata or {}
        else:
            result["error"] = mcp_result.error
            result["error_type"] = "tool_execution_error"

        # Kontext-Informationen hinzufügen
        if context:
            result["context"] = {
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "request_id": context.request_id,
            }

        return result

    async def get_server_health_status(self) -> dict[str, Any]:
        """Gibt Gesundheitsstatus aller Server zurück.

        Returns:
            Dictionary mit Server-Status-Informationen
        """
        all_servers = mcp_registry.get_all_servers()
        available_servers = mcp_registry.get_available_servers()

        server_stats = {}
        for server_name in all_servers:
            stats = mcp_registry.get_server_stats(server_name)
            if stats:
                server_stats[server_name] = stats

        return {
            "total_servers": len(all_servers),
            "available_servers": len(available_servers),
            "unavailable_servers": len(all_servers) - len(available_servers),
            "server_details": server_stats,
        }


# Globale Bridge-Instanz
external_mcp_bridge = ExternalMCPBridge()


__all__ = [
    "AgentToolDefinition",
    "ExternalMCPBridge",
    "ToolExecutionContext",
    "external_mcp_bridge",
]
