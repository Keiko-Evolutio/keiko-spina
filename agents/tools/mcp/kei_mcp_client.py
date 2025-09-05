"""HTTP-basierter MCP Client für externe MCP Server.

Dieser Client kommuniziert über REST API statt direkter MCP-Protokoll-Verbindung
und ermöglicht die Integration externer MCP Server über standardisierte HTTP-Endpoints.
"""

from __future__ import annotations

import asyncio
import ssl
from dataclasses import dataclass
from typing import Any

import httpx

from config.kei_mcp_config import KEI_MCP_SETTINGS, KEIMCPConfig
from config.mtls_config import MTLS_SETTINGS, MTLSCertificateConfig
from kei_logging import get_logger
from observability import trace_function
from observability.kei_mcp_metrics import categorize_error, kei_mcp_metrics

from ..tools_constants import HTTP_STATUS_CODES, HTTP_TIMEOUTS
from .core.constants import (
    ENDPOINTS,
)
from .kei_mcp_circuit_breaker import CircuitBreakerConfig, circuit_breaker_registry

logger = get_logger(__name__)


@dataclass(slots=True)
class ExternalMCPConfig:
    """Konfiguration für externen MCP Server.

    Attributes:
        base_url: Basis-URL des externen MCP Servers
        api_key: Optionaler API-Key für Authentifizierung
        timeout_seconds: Timeout für HTTP-Requests
        max_retries: Maximale Anzahl von Wiederholungsversuchen
        server_name: Eindeutiger Name des Servers
        custom_headers: Zusätzliche HTTP-Headers
        mtls_cert_config: mTLS-Zertifikat-Konfiguration für diesen Server
        verify_ssl: SSL-Zertifikat-Validierung aktivieren
    """

    base_url: str
    server_name: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    custom_headers: dict[str, str] | None = None
    mtls_cert_config: MTLSCertificateConfig | None = None
    verify_ssl: bool = True


@dataclass(slots=True)
class MCPToolDefinition:
    """Definition eines MCP Tools.

    Attributes:
        name: Name des Tools
        description: Beschreibung der Tool-Funktionalität
        parameters: JSON Schema für Tool-Parameter
        required: Liste der erforderlichen Parameter
        examples: Beispiele für Tool-Verwendung
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    examples: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class MCPToolResult:
    """Ergebnis eines MCP Tool-Aufrufs.

    Attributes:
        success: Ob der Aufruf erfolgreich war
        result: Ergebnis-Daten bei Erfolg
        error: Fehlermeldung bei Misserfolg
        server: Name des ausführenden Servers
        execution_time_ms: Ausführungszeit in Millisekunden
        metadata: Zusätzliche Metadaten
    """

    success: bool
    result: Any | None = None
    error: str | None = None
    server: str | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class MCPResourceResult:
    """Ergebnis eines MCP Resource-Aufrufs.

    Attributes:
        success: Ob der Aufruf erfolgreich war
        resource: Resource-Daten bei Erfolg
        error: Fehlermeldung bei Misserfolg
        server: Name des ausführenden Servers
        execution_time_ms: Ausführungszeit in Millisekunden
    """

    success: bool
    resource: Any | None = None
    error: str | None = None
    server: str | None = None
    execution_time_ms: float | None = None


@dataclass(slots=True)
class MCPPromptResult:
    """Ergebnis eines MCP Prompt-Aufrufs.

    Attributes:
        success: Ob der Aufruf erfolgreich war
        prompt: Prompt-Daten bei Erfolg
        error: Fehlermeldung bei Misserfolg
        server: Name des ausführenden Servers
        execution_time_ms: Ausführungszeit in Millisekunden
    """

    success: bool
    prompt: Any | None = None
    error: str | None = None
    server: str | None = None
    execution_time_ms: float | None = None


@dataclass(slots=True)
class MCPServerInfo:
    """Informationen über einen MCP Server.

    Attributes:
        name: Name des Servers
        version: Version des Servers
        capabilities: Liste der unterstützten Capabilities
    """

    name: str
    version: str
    capabilities: list[str]


@dataclass(slots=True)
class MCPCapability:
    """MCP Capability Definition.

    Attributes:
        name: Name der Capability
        description: Beschreibung der Capability
    """

    name: str
    description: str


@dataclass(slots=True)
class MCPTool:
    """MCP Tool Definition.

    Attributes:
        name: Name des Tools
        description: Beschreibung des Tools
        input_schema: JSON Schema für Input-Parameter
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class MCPResource:
    """MCP Resource Definition.

    Attributes:
        uri: URI der Ressource
        name: Name der Ressource
        description: Beschreibung der Ressource
        mime_type: MIME-Type der Ressource
    """

    uri: str
    name: str
    description: str
    mime_type: str | None = None


@dataclass(slots=True)
class MCPPrompt:
    """MCP Prompt Definition.

    Attributes:
        name: Name des Prompts
        description: Beschreibung des Prompts
        arguments: Argument-Schema
    """

    name: str
    description: str
    arguments: dict[str, Any]


class KEIMCPClient:
    """HTTP-Client für externe MCP Server über REST API.

    Dieser Client implementiert eine standardisierte HTTP-Schnittstelle für die
    Kommunikation mit externen MCP Servern. Er unterstützt automatische Retries,
    Timeout-Handling, HTTP/2, Connection Pooling und umfassende Fehlerbehandlung.
    """

    def __init__(self, config: ExternalMCPConfig):
        """Initialisiert den MCP Client.

        Args:
            config: Konfiguration für den externen MCP Server
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None

        # Circuit Breaker für Fehlerbehandlung
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            timeout_seconds=config.timeout_seconds
        )
        self._circuit_breaker = circuit_breaker_registry.get_or_create(
            f"mcp_client_{config.server_name}",
            circuit_config
        )

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self.aclose()

    async def _ensure_client(self):
        """Stellt sicher, dass HTTP Client verfügbar ist."""
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"MCP-Client/1.0 ({self.config.server_name})"
            }

            # API-Key Authentifizierung
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            # Zusätzliche Custom Headers
            if self.config.custom_headers:
                headers.update(self.config.custom_headers)

            # mTLS-Konfiguration
            cert = None
            verify = True

            if MTLS_SETTINGS.outbound.enabled:
                # Server-spezifische oder Default-Zertifikat-Konfiguration
                cert_config = (self.config.mtls_cert_config or
                             MTLS_SETTINGS.outbound.get_cert_config(self.config.server_name))

                if cert_config:
                    cert = cert_config.to_httpx_cert()
                    verify = cert_config.get_verify_param()

                    if MTLS_SETTINGS.enable_mtls_logging:
                        logger.info(f"mTLS aktiviert für Server {self.config.server_name}: "
                                   f"Cert={cert_config.cert_path}, Verify={verify}")
                else:
                    logger.warning(f"mTLS aktiviert aber keine Zertifikate für Server {self.config.server_name}")

            # SSL-Verifikation überschreiben falls konfiguriert
            if not self.config.verify_ssl:
                verify = False
                logger.warning(f"SSL-Verifikation deaktiviert für Server {self.config.server_name}")

            # HTTP/2 Support und Connection Pooling Konfiguration (httpx 0.25+)
            limits = httpx.Limits(
                max_keepalive_connections=KEI_MCP_SETTINGS.connection_pool_size,
                max_connections=KEI_MCP_SETTINGS.connection_pool_size * 2,
                keepalive_expiry=30.0  # Keep-Alive für 30 Sekunden
                # pool_connections und pool_maxsize entfernt (nicht in httpx verfügbar)
            )

            # Timeout-Konfiguration
            timeout = httpx.Timeout(
                connect=10.0,  # Connection Timeout
                read=self.config.timeout_seconds,  # Read Timeout
                write=10.0,  # Write Timeout
                pool=5.0  # Pool Timeout
            )

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=timeout,
                limits=limits,
                http2=True,  # HTTP/2 Support aktivieren
                cert=cert,  # mTLS Client-Zertifikat
                verify=verify,  # SSL-Verifikation mit CA-Bundle
                follow_redirects=True,  # Redirects folgen
                max_redirects=3  # Maximale Redirects
            )

            logger.debug(f"HTTP Client für {self.config.server_name} erstellt: "
                        f"HTTP/2={True}, Pool={KEI_MCP_SETTINGS.connection_pool_size}")

        # Concurrency Control Semaphore pro Server
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(KEI_MCP_SETTINGS.max_concurrent_requests)
            logger.debug(f"Concurrency Semaphore für {self.config.server_name} erstellt: "
                        f"max_concurrent={KEI_MCP_SETTINGS.max_concurrent_requests}")

    async def aclose(self):
        """Schließt HTTP Client und gibt Ressourcen frei."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @trace_function("external_mcp.discover_tools")
    async def discover_tools(self) -> list[MCPToolDefinition]:
        """Entdeckt verfügbare Tools vom externen MCP Server.

        Returns:
            Liste der verfügbaren Tool-Definitionen

        Raises:
            httpx.HTTPError: Bei HTTP-Kommunikationsfehlern
        """
        await self._ensure_client()

        # Circuit Breaker für Tool-Discovery
        async def _discover_tools_internal():
            # Concurrency Control anwenden
            async with self._semaphore:
                response = await self._client.get(ENDPOINTS["TOOLS"])
                response.raise_for_status()

                tools_data = response.json()
                discovered_tools = []

                for tool_data in tools_data.get("tools", []):
                    tool = MCPToolDefinition(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        parameters=tool_data.get("parameters", {}),
                        required=tool_data.get("required", []),
                        examples=tool_data.get("examples")
                    )
                    discovered_tools.append(tool)

                return discovered_tools

        start_time = asyncio.get_event_loop().time()
        try:
            # Circuit Breaker verwenden
            discovered_tools = await self._circuit_breaker.call(_discover_tools_internal)

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Metriken erfassen
            kei_mcp_metrics.record_request(
                server_name=self.config.server_name,
                operation="discovery",
                response_time_ms=execution_time,
                success=True
            )

            logger.info(f"Entdeckte {len(discovered_tools)} Tools auf Server {self.config.server_name}")
            return discovered_tools

        except Exception as exc:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Metriken erfassen
            kei_mcp_metrics.record_request(
                server_name=self.config.server_name,
                operation="discovery",
                response_time_ms=execution_time,
                success=False,
                error_category=categorize_error(exc)
            )

            logger.exception(f"Tool-Discovery fehlgeschlagen für {self.config.server_name}: {exc}")
            raise

    @trace_function("external_mcp.invoke_tool")
    async def invoke_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPToolResult:
        """Ruft Tool auf externem MCP Server auf.

        Args:
            tool_name: Name des aufzurufenden Tools
            parameters: Parameter für den Tool-Aufruf

        Returns:
            Ergebnis des Tool-Aufrufs
        """
        await self._ensure_client()

        payload = {
            "tool": tool_name,
            "parameters": parameters,
            "server_info": {
                "name": self.config.server_name,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        # Circuit Breaker für Tool-Invocation
        async def _invoke_tool_internal():
            # Concurrency Control anwenden
            async with self._semaphore:
                for attempt in range(self.config.max_retries + 1):
                    try:
                        response = await self._client.post(ENDPOINTS["TOOLS_INVOKE"], json=payload)
                        response.raise_for_status()

                        result_data = response.json()

                        return MCPToolResult(
                            success=True,
                            result=result_data.get("result"),
                            server=self.config.server_name,
                            execution_time_ms=0,  # Wird später gesetzt
                            metadata=result_data.get("metadata", {})
                        )

                    except httpx.HTTPStatusError as http_error:
                        if http_error.response.status_code < 500 or attempt == self.config.max_retries:
                            return MCPToolResult(
                                success=False,
                                error=f"HTTP {http_error.response.status_code}: {http_error.response.text}",
                                server=self.config.server_name,
                                execution_time_ms=0  # Wird später gesetzt
                            )

                        # Retry bei 5xx Fehlern mit exponential backoff
                        await asyncio.sleep(2 ** attempt)

                    except ssl.SSLError as ssl_error:
                        # mTLS/SSL-spezifische Fehlerbehandlung
                        ssl_error_msg = self._categorize_ssl_error(ssl_error)
                        logger.exception(f"SSL-Fehler bei {self.config.server_name}: {ssl_error_msg}")

                        return MCPToolResult(
                            success=False,
                            error=f"SSL/mTLS-Fehler: {ssl_error_msg}",
                            server=self.config.server_name,
                            execution_time_ms=0
                        )

                    except httpx.ConnectError as connect_error:
                        # Verbindungsfehler (kann mTLS-related sein)
                        if "certificate" in str(connect_error).lower() or "ssl" in str(connect_error).lower():
                            logger.exception(f"mTLS-Verbindungsfehler bei {self.config.server_name}: {connect_error}")
                            return MCPToolResult(
                                success=False,
                                error=f"mTLS-Verbindungsfehler: {connect_error!s}",
                                server=self.config.server_name,
                                execution_time_ms=0
                            )

                        if attempt == self.config.max_retries:
                            return MCPToolResult(
                                success=False,
                                error=f"Verbindungsfehler: {connect_error!s}",
                                server=self.config.server_name,
                                execution_time_ms=0
                            )

                        await asyncio.sleep(2 ** attempt)

                    except Exception as general_error:
                        if attempt == self.config.max_retries:
                            return MCPToolResult(
                                success=False,
                                error=str(general_error),
                                server=self.config.server_name,
                                execution_time_ms=0  # Wird später gesetzt
                            )

                        await asyncio.sleep(2 ** attempt)
            return None

        start_time = asyncio.get_event_loop().time()
        try:
            # Circuit Breaker verwenden
            result = await self._circuit_breaker.call(_invoke_tool_internal)

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            result.execution_time_ms = execution_time

            # Metriken erfassen
            kei_mcp_metrics.record_request(
                server_name=self.config.server_name,
                operation="invoke",
                response_time_ms=execution_time,
                success=result.success,
                error_category=categorize_error(Exception(result.error)) if not result.success else None
            )

            return result

        except Exception as exc:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Metriken erfassen
            kei_mcp_metrics.record_request(
                server_name=self.config.server_name,
                operation="invoke",
                response_time_ms=execution_time,
                success=False,
                error_category=categorize_error(exc)
            )

            return MCPToolResult(
                success=False,
                error=str(exc),
                server=self.config.server_name,
                execution_time_ms=execution_time
            )

    @trace_function("external_mcp.health_check")
    async def health_check(self) -> bool:
        """Prüft Gesundheit des externen MCP Servers.

        Returns:
            True wenn Server erreichbar und gesund, False sonst
        """
        await self._ensure_client()

        # Health Check ohne Circuit Breaker (um Circuit Breaker Status zu ermitteln)
        # Concurrency Control anwenden (aber mit kürzerem Timeout für Health Checks)
        async with self._semaphore:
            start_time = asyncio.get_event_loop().time()
            try:
                # Kürzerer Timeout für Health Checks
                response = await self._client.get(ENDPOINTS["HEALTH"], timeout=HTTP_TIMEOUTS["health_check"])
                is_healthy = response.status_code == HTTP_STATUS_CODES["ok"]

                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

                # Metriken erfassen
                kei_mcp_metrics.record_request(
                    server_name=self.config.server_name,
                    operation="health",
                    response_time_ms=execution_time,
                    success=is_healthy
                )

                return is_healthy

            except Exception as exc:
                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

                # Metriken erfassen
                kei_mcp_metrics.record_request(
                    server_name=self.config.server_name,
                    operation="health",
                    response_time_ms=execution_time,
                    success=False,
                    error_category=categorize_error(exc)
                )

                logger.debug(f"Health Check fehlgeschlagen für {self.config.server_name}: {exc}")
                return False

    @trace_function("external_mcp.get_server_info")
    async def get_server_info(self) -> dict[str, Any]:
        """Ruft Server-Informationen ab.

        Returns:
            Dictionary mit Server-Informationen
        """
        await self._ensure_client()

        try:
            response = await self._client.get("/mcp/info")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning(f"Server-Info nicht verfügbar für {self.config.server_name}: {exc}")
            return {
                "name": self.config.server_name,
                "status": "unknown",
                "error": str(exc)
            }

    @trace_function("external_mcp.discover_resources")
    async def discover_resources(self) -> list[dict[str, Any]]:
        """Entdeckt verfügbare Resources auf dem externen MCP Server.

        Returns:
            Liste der verfügbaren Resources
        """
        await self._ensure_client()

        try:
            response = await self._client.get("/mcp/resources")
            response.raise_for_status()

            resources_data = response.json()
            resources = resources_data.get("resources", [])

            logger.debug(f"Entdeckte {len(resources)} Resources auf {self.config.server_name}")
            return resources

        except Exception as exc:
            logger.warning(f"Resource-Discovery fehlgeschlagen für {self.config.server_name}: {exc}")
            return []

    @trace_function("external_mcp.get_resource")
    async def get_resource(
        self,
        resource_id: str,
        if_none_match: str | None = None,
        range_header: str | None = None
    ) -> MCPResourceResult:
        """Ruft eine spezifische Resource ab.

        Args:
            resource_id: ID der Resource
            if_none_match: ETag für Conditional Requests
            range_header: Range-Header für partielle Inhalte

        Returns:
            Resource-Inhalt und Metadaten
        """
        await self._ensure_client()

        headers = {}
        if if_none_match:
            headers["If-None-Match"] = if_none_match
        if range_header:
            headers["Range"] = range_header

        try:
            response = await self._client.get(f"/mcp/resources/{resource_id}", headers=headers)

            # Import hier um zirkuläre Imports zu vermeiden
            from .kei_mcp_registry import MCPResourceResult

            if response.status_code == 304:
                return MCPResourceResult(
                    success=True,
                    status_code=304
                )

            if response.status_code == 206:
                return MCPResourceResult(
                    success=True,
                    content=response.content,
                    content_type=response.headers.get("Content-Type"),
                    content_length=len(response.content),
                    status_code=206,
                    headers=dict(response.headers)
                )

            response.raise_for_status()

            return MCPResourceResult(
                success=True,
                content=response.content,
                content_type=response.headers.get("Content-Type"),
                content_length=len(response.content),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                headers=dict(response.headers)
            )

        except Exception as exc:
            logger.exception(f"Resource-Abruf fehlgeschlagen: {resource_id} - {exc}")
            from .kei_mcp_registry import MCPResourceResult
            return MCPResourceResult(
                success=False,
                error=str(exc)
            )

    @trace_function("external_mcp.discover_prompts")
    async def discover_prompts(self) -> list[dict[str, Any]]:
        """Entdeckt verfügbare Prompts auf dem externen MCP Server.

        Returns:
            Liste der verfügbaren Prompts
        """
        await self._ensure_client()

        try:
            response = await self._client.get("/mcp/prompts")
            response.raise_for_status()

            prompts_data = response.json()
            prompts = prompts_data.get("prompts", [])

            logger.debug(f"Entdeckte {len(prompts)} Prompts auf {self.config.server_name}")
            return prompts

        except Exception as exc:
            logger.warning(f"Prompt-Discovery fehlgeschlagen für {self.config.server_name}: {exc}")
            return []

    @trace_function("external_mcp.get_prompt")
    async def get_prompt(self, prompt_name: str, version: str | None = None) -> dict[str, Any]:
        """Ruft ein spezifisches Prompt-Template ab.

        Args:
            prompt_name: Name des Prompts
            version: Spezifische Version (optional)

        Returns:
            Prompt-Template und Metadaten
        """
        await self._ensure_client()

        params = {}
        if version:
            params["version"] = version

        try:
            response = await self._client.get(f"/mcp/prompts/{prompt_name}", params=params)
            response.raise_for_status()

            prompt_data = response.json()
            logger.debug(f"Prompt {prompt_name} erfolgreich abgerufen von {self.config.server_name}")

            return {
                "success": True,
                **prompt_data
            }

        except Exception as exc:
            logger.exception(f"Prompt-Abruf fehlgeschlagen: {prompt_name} - {exc}")
            return {
                "success": False,
                "error": str(exc)
            }

    def _categorize_ssl_error(self, exc: ssl.SSLError) -> str:
        """Kategorisiert SSL-Fehler für bessere Fehlerbehandlung.

        Args:
            exc: SSL-Exception

        Returns:
            Kategorisierte Fehlermeldung
        """
        error_str = str(exc).lower()

        if "certificate verify failed" in error_str:
            return "Server-Zertifikat-Validierung fehlgeschlagen"
        if "certificate required" in error_str:
            return "Client-Zertifikat erforderlich aber nicht bereitgestellt"
        if "bad certificate" in error_str:
            return "Ungültiges Client-Zertifikat"
        if "unknown ca" in error_str:
            return "Unbekannte Certificate Authority"
        if "certificate expired" in error_str:
            return "Zertifikat abgelaufen"
        if "certificate not yet valid" in error_str:
            return "Zertifikat noch nicht gültig"
        if "hostname mismatch" in error_str:
            return "Hostname stimmt nicht mit Zertifikat überein"
        if "handshake failure" in error_str:
            return "SSL-Handshake fehlgeschlagen"
        if "protocol version" in error_str:
            return "Inkompatible SSL/TLS-Protokollversion"
        return f"SSL-Fehler: {exc!s}"

    async def close(self):
        """Schließt den HTTP Client."""
        if self._client:
            await self._client.aclose()
            self._client = None


__all__ = [
    "ExternalMCPConfig",
    "KEIMCPClient",
    # Rückwärtskompatibilität: einige Tests importieren KEIMCPConfig hier
    "KEIMCPConfig",
    "MCPToolDefinition",
    "MCPToolResult",
]
