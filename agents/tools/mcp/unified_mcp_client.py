"""Unified MCP Client basierend auf UnifiedHTTPClient.

Migriert KEIMCPClient zur neuen UnifiedHTTPClient-Architektur
während die bestehende API beibehalten wird.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger
from services.core.unified_client import UnifiedClientConfig, UnifiedHTTPClient

# Optional imports für erweiterte Features
try:
    from observability import trace_function
except ImportError:
    def trace_function(_name):
        def decorator(func):
            return func
        return decorator

try:
    from config.kei_mcp_config import KEI_MCP_SETTINGS
except ImportError:
    class MockSettings:
        max_concurrent_requests = 10
    KEI_MCP_SETTINGS = MockSettings()

try:
    from observability.kei_mcp_metrics import categorize_error, kei_mcp_metrics
except ImportError:
    class MockMetrics:
        def record_tool_call(self, **kwargs):
            pass
    kei_mcp_metrics = MockMetrics()
    def categorize_error(_exc):
        return "unknown"

from .kei_mcp_client import (
    ExternalMCPConfig,
    MCPResource,
    MCPResourceResult,
    MCPServerInfo,
    MCPTool,
    MCPToolResult,
)

logger = get_logger(__name__)


class UnifiedKEIMCPClient:
    """Unified MCP Client basierend auf UnifiedHTTPClient.

    Drop-in Replacement für KEIMCPClient mit verbesserter Architektur:
    - Verwendet UnifiedHTTPClient für HTTP-Kommunikation
    - Behält vollständige API-Kompatibilität bei
    - Verbesserte Error-Handling und Circuit Breaker Integration
    - Konsolidierte Konfiguration und Monitoring
    """

    def __init__(self, config: ExternalMCPConfig):
        """Initialisiert den Unified MCP Client.

        Args:
            config: Konfiguration für den externen MCP Server
        """
        self.config = config
        self._unified_client: UnifiedHTTPClient | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Stellt sicher, dass der Unified HTTP Client initialisiert ist."""
        if self._unified_client is not None:
            return

        # Erstelle UnifiedClientConfig basierend auf ExternalMCPConfig
        unified_config = self._create_unified_config()
        self._unified_client = UnifiedHTTPClient(unified_config)

        # Initialisiere den Client
        await self._unified_client._ensure_client()

        # Concurrency Control Semaphore
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(KEI_MCP_SETTINGS.max_concurrent_requests)

        logger.debug(f"Unified MCP Client für {self.config.server_name} initialisiert")

    def _create_unified_config(self) -> UnifiedClientConfig:
        """Erstellt UnifiedClientConfig aus ExternalMCPConfig."""
        # Custom Headers für MCP
        custom_headers = {}
        if self.config.api_key:
            custom_headers["X-API-Key"] = self.config.api_key
        if self.config.custom_headers:
            custom_headers.update(self.config.custom_headers)

        # SSL-Konfiguration
        verify_ssl = getattr(self.config, "verify_ssl", True)
        _ssl_context = None

        return UnifiedClientConfig.for_mcp_client(
            base_url=self.config.base_url,
            server_name=self.config.server_name,
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            custom_headers=custom_headers,
            verify_ssl=verify_ssl,
            ssl_context=_ssl_context,
            http2_enabled=True,  # HTTP/2 Support
        )

    @trace_function("mcp_client_call_tool")
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout_override: float | None = None
    ) -> MCPToolResult:
        """Ruft ein Tool auf dem MCP Server auf.

        Args:
            tool_name: Name des aufzurufenden Tools
            arguments: Argumente für das Tool
            timeout_override: Optionales Timeout-Override

        Returns:
            MCPToolResult mit dem Ergebnis des Tool-Aufrufs
        """
        await self._ensure_client()

        async with self._semaphore:
            start_time = datetime.now(UTC)

            try:
                # Request-Payload erstellen
                payload = {
                    "tool": tool_name,
                    "arguments": arguments or {}
                }

                # Timeout-Konfiguration
                timeout = timeout_override or self.config.timeout_seconds

                # HTTP-Request über UnifiedHTTPClient
                response = await self._unified_client.post_json(
                    "/tools/call",
                    json_data=payload,
                    timeout=timeout
                )

                # Execution Time berechnen
                execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

                # Erfolgreiche Response verarbeiten
                result = MCPToolResult(
                    success=True,
                    result=response,
                    server=self.config.server_name,
                    execution_time_ms=execution_time_ms
                )

                # Metrics erfassen
                kei_mcp_metrics.record_tool_call(
                    server_name=self.config.server_name,
                    tool_name=tool_name,
                    success=True,
                    execution_time_ms=execution_time_ms
                )

                return result

            except Exception as exc:
                execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
                error_category = categorize_error(exc)

                # Error Metrics erfassen
                kei_mcp_metrics.record_tool_call(
                    server_name=self.config.server_name,
                    tool_name=tool_name,
                    success=False,
                    execution_time_ms=execution_time_ms,
                    error_category=error_category
                )

                # Error Result erstellen
                return MCPToolResult(
                    success=False,
                    error=str(exc),
                    server=self.config.server_name,
                    execution_time_ms=execution_time_ms
                )

    @trace_function("mcp_client_list_tools")
    async def list_tools(self) -> list[MCPTool]:
        """Listet verfügbare Tools auf dem MCP Server auf.

        Returns:
            Liste der verfügbaren MCPTool-Objekte
        """
        await self._ensure_client()

        try:
            response = await self._unified_client.get_json("/tools")

            # Response zu MCPTool-Objekten konvertieren
            tools = []
            for tool_data in response.get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {})
                )
                tools.append(tool)

            return tools

        except Exception as exc:
            logger.exception(f"Fehler beim Auflisten der Tools für {self.config.server_name}: {exc}")
            return []

    @trace_function("mcp_client_get_resource")
    async def get_resource(self, resource_uri: str) -> MCPResourceResult:
        """Ruft eine Ressource vom MCP Server ab.

        Args:
            resource_uri: URI der anzufordernden Ressource

        Returns:
            MCPResourceResult mit der Ressource
        """
        await self._ensure_client()

        start_time = datetime.now(UTC)

        try:
            response = await self._unified_client.get_json(
                "/resources/read",
                params={"uri": resource_uri}
            )

            execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return MCPResourceResult(
                success=True,
                resource=response,
                server=self.config.server_name,
                execution_time_ms=execution_time_ms
            )

        except Exception as exc:
            execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return MCPResourceResult(
                success=False,
                error=str(exc),
                server=self.config.server_name,
                execution_time_ms=execution_time_ms
            )

    @trace_function("mcp_client_list_resources")
    async def list_resources(self) -> list[MCPResource]:
        """Listet verfügbare Ressourcen auf dem MCP Server auf.

        Returns:
            Liste der verfügbaren MCPResource-Objekte
        """
        await self._ensure_client()

        try:
            response = await self._unified_client.get_json("/resources")

            resources = []
            for resource_data in response.get("resources", []):
                resource = MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data.get("name", ""),
                    description=resource_data.get("description", ""),
                    mime_type=resource_data.get("mimeType")
                )
                resources.append(resource)

            return resources

        except Exception as exc:
            logger.exception(f"Fehler beim Auflisten der Ressourcen für {self.config.server_name}: {exc}")
            return []

    async def get_server_info(self) -> MCPServerInfo | None:
        """Ruft Server-Informationen vom MCP Server ab.

        Returns:
            MCPServerInfo oder None bei Fehlern
        """
        await self._ensure_client()

        try:
            response = await self._unified_client.get_json("/info")

            return MCPServerInfo(
                name=response.get("name", self.config.server_name),
                version=response.get("version", "unknown"),
                capabilities=response.get("capabilities", [])
            )

        except Exception as exc:
            logger.exception(f"Fehler beim Abrufen der Server-Info für {self.config.server_name}: {exc}")
            return None

    async def health_check(self) -> bool:
        """Führt Health-Check für den MCP Server durch.

        Returns:
            True wenn Server erreichbar und gesund
        """
        await self._ensure_client()

        try:
            health_result = await self._unified_client.health_check("/health")
            return health_result["status"] == "healthy"

        except Exception as exc:
            logger.debug(f"Health-Check fehlgeschlagen für {self.config.server_name}: {exc}")
            return False

    async def close(self) -> None:
        """Schließt den MCP Client und gibt Ressourcen frei."""
        if self._unified_client:
            await self._unified_client.close()
            self._unified_client = None

        logger.debug(f"Unified MCP Client für {self.config.server_name} geschlossen")


# Backward-Compatibility Alias
KEIMCPClient = UnifiedKEIMCPClient
