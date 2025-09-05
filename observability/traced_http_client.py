"""Traced HTTP Client für automatische OpenTelemetry Trace-Propagation.

Dieser Client stellt sicher, dass alle ausgehenden HTTP-Requests automatisch
mit Trace-Context-Headern versehen werden für End-to-End-Tracing.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from opentelemetry import trace

from kei_logging import get_logger
from observability.tracing import OPENTELEMETRY_AVAILABLE, create_mcp_span, inject_trace_context

from .base_metrics import MetricsConstants

logger = get_logger(__name__)


class TracedHTTPXClient:
    """HTTPX Client mit automatischer OpenTelemetry Trace-Propagation.

    Dieser Client erweitert httpx.AsyncClient um automatische Trace-Context-
    Propagation für alle ausgehenden HTTP-Requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = MetricsConstants.DEFAULT_TIMEOUT_SECONDS,
        headers: dict[str, str] | None = None,
        **kwargs
    ):
        """Initialisiert den Traced HTTP Client.

        Args:
            base_url: Basis-URL für Requests
            timeout: Request-Timeout in Sekunden
            headers: Standard-Headers für alle Requests
            **kwargs: Zusätzliche httpx.AsyncClient-Parameter
        """
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = headers or {}
        self.client_kwargs = kwargs
        self._client: httpx.AsyncClient | None = None

        logger.debug(f"TracedHTTPXClient initialisiert für {base_url}")

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self):
        """Stellt sicher, dass der HTTP-Client initialisiert ist."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self.default_headers,
                **self.client_kwargs
            )

    def _prepare_headers(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        """Bereitet Headers mit Trace-Context vor.

        Args:
            headers: Request-spezifische Headers

        Returns:
            Headers mit injiziertem Trace-Context
        """
        # Standard-Headers mit Request-Headers kombinieren
        combined_headers = self.default_headers.copy()
        if headers:
            combined_headers.update(headers)

        # Trace-Context injizieren
        if OPENTELEMETRY_AVAILABLE:
            combined_headers = inject_trace_context(combined_headers)

        return combined_headers

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        span_name: str | None = None,
        span_attributes: dict[str, Any] | None = None,
        **kwargs
    ) -> httpx.Response:
        """Führt HTTP-Request mit automatischer Trace-Propagation aus.

        Args:
            method: HTTP-Methode
            url: Request-URL
            headers: Request-Headers
            span_name: Name für den Trace-Span
            span_attributes: Zusätzliche Span-Attribute
            **kwargs: Zusätzliche httpx-Parameter

        Returns:
            HTTP-Response
        """
        await self._ensure_client()

        # Headers mit Trace-Context vorbereiten
        traced_headers = self._prepare_headers(headers)

        # Span-Name generieren
        if not span_name:
            span_name = f"HTTP {method.upper()}"

        # Span-Attribute vorbereiten
        attributes = {
            "http.method": method.upper(),
            "http.url": str(url),
            "http.client": "traced_httpx",
            **(span_attributes or {})
        }

        # Request mit Tracing ausführen
        if OPENTELEMETRY_AVAILABLE:
            with create_mcp_span(
                operation_name=span_name,
                operation_type="http_request",
                attributes=attributes
            ) as span:
                start_time = time.time()

                try:
                    response = await self._client.request(
                        method=method,
                        url=url,
                        headers=traced_headers,
                        **kwargs
                    )

                    # Response-Attribute setzen
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.response_size", len(response.content))
                    span.set_attribute("http.duration_ms", duration_ms)

                    # Status basierend auf HTTP-Code setzen
                    if response.status_code >= 400:
                        span.set_status(
                            trace.Status(
                                trace.StatusCode.ERROR,
                                f"HTTP {response.status_code}"
                            )
                        )
                    else:
                        span.set_status(trace.Status(trace.StatusCode.OK))

                    return response

                except Exception as e:
                    # Fehler-Attribute setzen
                    span.set_attribute("http.error.type", type(e).__name__)
                    span.set_attribute("http.error.message", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        else:
            # Fallback ohne Tracing
            return await self._client.request(
                method=method,
                url=url,
                headers=traced_headers,
                **kwargs
            )

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET-Request mit Trace-Propagation."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST-Request mit Trace-Propagation."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        """PUT-Request mit Trace-Propagation."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """DELETE-Request mit Trace-Propagation."""
        return await self.request("DELETE", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> httpx.Response:
        """PATCH-Request mit Trace-Propagation."""
        return await self.request("PATCH", url, **kwargs)


class MCPServerHTTPClient(TracedHTTPXClient):
    """Spezialisierter HTTP-Client für MCP-Server-Kommunikation.

    Erweitert TracedHTTPXClient um MCP-spezifische Tracing-Funktionalität.
    """

    def __init__(
        self,
        server_name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = 30.0,
        **kwargs
    ):
        """Initialisiert MCP-Server HTTP-Client.

        Args:
            server_name: Name des MCP-Servers
            base_url: Basis-URL des MCP-Servers
            api_key: API-Key für Authentifizierung
            timeout: Request-Timeout
            **kwargs: Zusätzliche Client-Parameter
        """
        self.server_name = server_name

        # Standard-Headers für MCP-Server
        headers = {
            "User-Agent": "KEI-MCP-Client/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        super().__init__(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            **kwargs
        )

        logger.debug(f"MCP-Server HTTP-Client für '{server_name}' initialisiert")

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> httpx.Response:
        """Führt Tool-Invocation mit MCP-spezifischem Tracing aus.

        Args:
            tool_name: Name des Tools
            parameters: Tool-Parameter
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        span_attributes = {
            "mcp.server.name": self.server_name,
            "mcp.tool.name": tool_name,
            "mcp.tool.parameter_count": len(parameters),
            "mcp.operation.type": "tool_invocation"
        }

        payload = {
            "tool": tool_name,
            "parameters": parameters
        }

        return await self.post(
            "/tools/invoke",
            json=payload,
            span_name=f"mcp.tool.invoke[{self.server_name}:{tool_name}]",
            span_attributes=span_attributes,
            **kwargs
        )

    async def list_tools(self, **kwargs) -> httpx.Response:
        """Listet verfügbare Tools mit MCP-spezifischem Tracing.

        Args:
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        span_attributes = {
            "mcp.server.name": self.server_name,
            "mcp.operation.type": "tool_discovery"
        }

        return await self.get(
            "/tools",
            span_name=f"mcp.tools.list[{self.server_name}]",
            span_attributes=span_attributes,
            **kwargs
        )

    async def get_resource(
        self,
        resource_uri: str,
        **kwargs
    ) -> httpx.Response:
        """Ruft Resource mit MCP-spezifischem Tracing ab.

        Args:
            resource_uri: URI der Resource
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        span_attributes = {
            "mcp.server.name": self.server_name,
            "mcp.resource.uri": resource_uri,
            "mcp.operation.type": "resource_access"
        }

        return await self.get(
            f"/resources/{resource_uri}",
            span_name=f"mcp.resource.get[{self.server_name}]",
            span_attributes=span_attributes,
            **kwargs
        )

    async def list_resources(self, **kwargs) -> httpx.Response:
        """Listet verfügbare Resources mit MCP-spezifischem Tracing.

        Args:
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        span_attributes = {
            "mcp.server.name": self.server_name,
            "mcp.operation.type": "resource_discovery"
        }

        return await self.get(
            "/resources",
            span_name=f"mcp.resources.list[{self.server_name}]",
            span_attributes=span_attributes,
            **kwargs
        )


@asynccontextmanager
async def traced_http_client(
    base_url: str | None = None,
    **kwargs
) -> TracedHTTPXClient:
    """Async Context Manager für Traced HTTP Client.

    Args:
        base_url: Basis-URL für Requests
        **kwargs: Zusätzliche Client-Parameter

    Yields:
        TracedHTTPXClient-Instanz
    """
    async with TracedHTTPXClient(base_url=base_url, **kwargs) as client:
        yield client


@asynccontextmanager
async def mcp_server_client(
    server_name: str,
    base_url: str,
    **kwargs
) -> MCPServerHTTPClient:
    """Async Context Manager für MCP-Server HTTP Client.

    Args:
        server_name: Name des MCP-Servers
        base_url: Basis-URL des MCP-Servers
        **kwargs: Zusätzliche Client-Parameter

    Yields:
        MCPServerHTTPClient-Instanz
    """
    async with MCPServerHTTPClient(
        server_name=server_name,
        base_url=base_url,
        **kwargs
    ) as client:
        yield client
