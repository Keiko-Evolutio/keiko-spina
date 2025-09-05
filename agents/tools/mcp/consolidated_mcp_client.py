"""Konsolidierter MCP Client - Vereint alle MCP-Client-Implementierungen.

Dieser Client konsolidiert die Funktionalität aus:
- kei_mcp_client.py (KEIMCPClient)
- unified_mcp_client.py (UnifiedKEIMCPClient)
- factory/unified_mcp_client.py (UnifiedMCPClient)

Basiert auf BaseHTTPClient für einheitliche HTTP-Kommunikation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from ..tools_constants import HTTP_STATUS_CODES, HTTP_TIMEOUTS
from .core.base_client import BaseHTTPClient, HTTPClientConfig
from .core.constants import ENDPOINTS
from .kei_mcp_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = get_logger(__name__)


@dataclass(slots=True)
class ExternalMCPConfig:
    """Konfiguration für externe MCP Server.

    Konsolidiert Konfiguration aus verschiedenen MCP-Client-Implementierungen.
    """

    server_name: str
    base_url: str
    timeout_seconds: float = 30.0
    api_key: str | None = None
    custom_headers: dict[str, str] | None = None
    verify_ssl: bool = True
    max_concurrent_requests: int = 10


@dataclass(slots=True)
class MCPToolDefinition:
    """MCP Tool Definition."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class MCPToolResult:
    """MCP Tool Execution Result."""

    success: bool
    content: Any
    error: str | None = None
    metadata: dict[str, Any] | None = None


class ConsolidatedMCPClient:
    """Konsolidierter MCP Client mit Enterprise-Features.

    Vereint die Funktionalität aller MCP-Client-Implementierungen:
    - HTTP-Kommunikation über BaseHTTPClient
    - Circuit Breaker für Fehlerbehandlung
    - Einheitliche API für alle MCP-Operationen
    - Umfassende Metriken und Monitoring
    """

    def __init__(self, config: ExternalMCPConfig):
        """Initialisiert den konsolidierten MCP Client.

        Args:
            config: Konfiguration für den externen MCP Server
        """
        self.config = config

        # HTTP-Client-Konfiguration erstellen
        http_config = HTTPClientConfig(
            server_name=config.server_name,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            verify_ssl=config.verify_ssl,
            connection_pool_size=config.max_concurrent_requests,
            custom_headers=config.custom_headers or {}
        )

        # BaseHTTPClient initialisieren
        self._http_client = BaseHTTPClient(http_config)

        # Circuit Breaker für Fehlerbehandlung
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            timeout_seconds=config.timeout_seconds
        )
        self._circuit_breaker = CircuitBreaker(
            f"mcp_client_{config.server_name}",
            circuit_config
        )

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._http_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self._http_client.__aexit__(exc_type, exc_val, exc_tb)

    @trace_function("mcp.client.list_tools")
    async def list_tools(self) -> list[MCPToolDefinition]:
        """Listet verfügbare Tools auf dem MCP Server auf.

        Returns:
            Liste der verfügbaren Tools
        """
        try:
            response = await self._http_client.make_request(
                method="GET",
                endpoint=ENDPOINTS["TOOLS"],
                timeout=HTTP_TIMEOUTS["default"]
            )

            tools_data = response.json()
            return [
                MCPToolDefinition(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {})
                )
                for tool in tools_data.get("tools", [])
            ]

        except Exception as exc:
            logger.error(f"Tool-Listing fehlgeschlagen für {self.config.server_name}: {exc}")
            return []

    @trace_function("mcp.client.invoke_tool")
    async def invoke_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPToolResult:
        """Ruft Tool auf externem MCP Server auf.

        Args:
            tool_name: Name des aufzurufenden Tools
            parameters: Parameter für den Tool-Aufruf

        Returns:
            Ergebnis des Tool-Aufrufs
        """
        payload = {
            "tool": tool_name,
            "parameters": parameters,
            "server_info": {
                "name": self.config.server_name,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        try:
            response = await self._http_client.make_request(
                method="POST",
                endpoint=ENDPOINTS["TOOLS_INVOKE"],
                data=payload,
                timeout=self.config.timeout_seconds
            )

            result_data = response.json()

            return MCPToolResult(
                success=True,
                content=result_data.get("content"),
                metadata=result_data.get("metadata", {})
            )

        except Exception as exc:
            logger.error(f"Tool-Aufruf fehlgeschlagen für {tool_name}: {exc}")
            return MCPToolResult(
                success=False,
                content=None,
                error=str(exc)
            )

    @trace_function("mcp.client.health_check")
    async def health_check(self) -> bool:
        """Führt Health-Check für den MCP Server durch.

        Returns:
            True wenn Server erreichbar und gesund
        """
        try:
            response = await self._http_client.make_request(
                method="GET",
                endpoint=ENDPOINTS["HEALTH"],
                timeout=HTTP_TIMEOUTS["health_check"]
            )

            return response.status_code == HTTP_STATUS_CODES["ok"]

        except Exception as exc:
            logger.debug(f"Health-Check fehlgeschlagen für {self.config.server_name}: {exc}")
            return False

    async def close(self) -> None:
        """Schließt den MCP Client und gibt Ressourcen frei."""
        await self._http_client.close()
        logger.debug(f"Konsolidierter MCP Client für {self.config.server_name} geschlossen")


# Backward-Compatibility Aliases
KEIMCPClient = ConsolidatedMCPClient
UnifiedKEIMCPClient = ConsolidatedMCPClient
UnifiedMCPClient = ConsolidatedMCPClient


__all__ = [
    "ConsolidatedMCPClient",
    "ExternalMCPConfig",
    "KEIMCPClient",  # Backward compatibility
    "MCPToolDefinition",
    "MCPToolResult",
    "UnifiedKEIMCPClient",  # Backward compatibility
    "UnifiedMCPClient",  # Backward compatibility
]
