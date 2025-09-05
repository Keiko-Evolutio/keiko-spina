"""Unified HTTP Client für die gesamte Keiko-Codebase.

Konsolidiert alle HTTP-Client-Implementierungen in eine einheitliche Lösung.
Ersetzt duplizierte Patterns in kei_mcp, services/webhooks, services/clients, etc.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx

from kei_logging import get_logger
from services.core.circuit_breaker import CircuitBreaker, CircuitPolicy
from services.core.constants import (
    CONTENT_TYPE_JSON,
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    KEI_MCP_CLIENT_USER_AGENT,
    KEI_RPC_CIRCUIT_BREAKER_CONFIG,
    KEI_RPC_CLIENT_USER_AGENT,
    KEIKO_WEBHOOK_USER_AGENT,
    MCP_CLIENT_CIRCUIT_BREAKER_CONFIG,
)

logger = get_logger(__name__)


@dataclass
class UnifiedClientConfig:
    """Einheitliche Konfiguration für alle HTTP-Clients."""

    base_url: str
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR

    # Authentication
    api_key: str | None = None
    bearer_token: str | None = None
    tenant_id: str | None = None

    # Headers
    user_agent: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_name: str | None = None
    circuit_breaker_config: dict[str, Any] | None = None

    # SSL/TLS
    verify_ssl: bool = True
    ssl_context: Any | None = None

    # HTTP/2
    http2_enabled: bool = True

    @classmethod
    def for_kei_rpc(
        cls,
        base_url: str,
        api_token: str,
        tenant_id: str,
        **kwargs: Any
    ) -> UnifiedClientConfig:
        """Erstellt Konfiguration für KEI-RPC Client."""
        return cls(
            base_url=base_url,
            bearer_token=api_token,
            tenant_id=tenant_id,
            user_agent=KEI_RPC_CLIENT_USER_AGENT,
            circuit_breaker_name="api-grpc-upstream",
            circuit_breaker_config=KEI_RPC_CIRCUIT_BREAKER_CONFIG,
            **kwargs
        )

    @classmethod
    def for_mcp_client(
        cls,
        base_url: str,
        server_name: str,
        **kwargs: Any
    ) -> UnifiedClientConfig:
        """Erstellt Konfiguration für MCP Client."""
        return cls(
            base_url=base_url,
            user_agent=KEI_MCP_CLIENT_USER_AGENT,
            circuit_breaker_name=f"mcp_client_{server_name}",
            circuit_breaker_config=MCP_CLIENT_CIRCUIT_BREAKER_CONFIG,
            **kwargs
        )

    @classmethod
    def for_webhook(
        cls,
        base_url: str,
        **kwargs: Any
    ) -> UnifiedClientConfig:
        """Erstellt Konfiguration für Webhook Client."""
        return cls(
            base_url=base_url,
            user_agent=KEIKO_WEBHOOK_USER_AGENT,
            circuit_breaker_enabled=False,  # Webhooks haben eigene Circuit Breaker
            **kwargs
        )


class UnifiedHTTPClient:
    """Einheitlicher HTTP-Client für die gesamte Keiko-Codebase.

    Konsolidiert alle HTTP-Client-Patterns:
    - KEI-RPC Client
    - MCP Client
    - Webhook Client
    - Alerting Client
    - Monitoring Client
    """

    def __init__(self, config: UnifiedClientConfig) -> None:
        """Initialisiert den Unified HTTP Client.

        Args:
            config: Client-Konfiguration
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._circuit_breaker: CircuitBreaker | None = None

        # Circuit Breaker Setup
        if config.circuit_breaker_enabled:
            cb_name = config.circuit_breaker_name or f"unified_client_{id(self)}"
            cb_config = config.circuit_breaker_config or {}
            self._circuit_breaker = CircuitBreaker(
                name=cb_name,
                policy=CircuitPolicy(**cb_config)
            )

    async def __aenter__(self) -> UnifiedHTTPClient:
        """Async Context Manager Entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async Context Manager Exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Stellt sicher, dass HTTP-Client initialisiert ist."""
        if self._client is not None:
            return

        # Headers zusammenstellen
        headers = {
            "Accept": CONTENT_TYPE_JSON,
            "Content-Type": CONTENT_TYPE_JSON,
        }

        if self.config.user_agent:
            headers["User-Agent"] = self.config.user_agent

        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        elif self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        if self.config.tenant_id:
            headers["X-Tenant-ID"] = self.config.tenant_id

        headers.update(self.config.custom_headers)

        # Timeout-Konfiguration
        timeout = httpx.Timeout(
            connect=self.config.connection_timeout,
            read=self.config.timeout_seconds,
            write=self.config.timeout_seconds,
            pool=self.config.timeout_seconds
        )

        # Client erstellen
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout,
            verify=self.config.verify_ssl,
            http2=self.config.http2_enabled,
            follow_redirects=True,
        )

        logger.debug(f"Unified HTTP Client initialisiert für {self.config.base_url}")

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> httpx.Response:
        """Führt HTTP-Request mit Circuit Breaker und Retry aus.

        Args:
            method: HTTP-Methode
            url: Request-URL (relativ zur base_url)
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        await self._ensure_client()

        # Vollständige URL erstellen
        full_url = urljoin(self.config.base_url, url) if not url.startswith("http") else url

        async def _request() -> httpx.Response:
            response = await self._client.request(method, full_url, **kwargs)
            response.raise_for_status()
            return response

        # Mit Circuit Breaker ausführen
        if self._circuit_breaker:
            return await self._circuit_breaker.call(_request)
        return await _request()

    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> httpx.Response:
        """Führt HTTP-Request mit Retry-Logik aus.

        Args:
            method: HTTP-Methode
            url: Request-URL
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._make_request(method, url, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = (self.config.retry_backoff_factor ** attempt)
                    logger.debug(f"Request fehlgeschlagen (Versuch {attempt + 1}/{self.config.max_retries + 1}), "
                               f"Wiederholung in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.exception(f"Request nach {self.config.max_retries + 1} Versuchen fehlgeschlagen: {e}")

        raise last_exception

    # Standard HTTP-Methoden
    async def get(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """GET-Request."""
        return await self._make_request("GET", url, **kwargs)

    async def post(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """POST-Request."""
        return await self._make_request("POST", url, **kwargs)

    async def put(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """PUT-Request."""
        return await self._make_request("PUT", url, **kwargs)

    async def patch(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """PATCH-Request."""
        return await self._make_request("PATCH", url, **kwargs)

    async def delete(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """DELETE-Request."""
        return await self._make_request("DELETE", url, **kwargs)

    # Retry-Varianten
    async def get_with_retry(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """GET-Request mit Retry."""
        return await self._make_request_with_retry("GET", url, **kwargs)

    async def post_with_retry(self, url: str = "", **kwargs: Any) -> httpx.Response:
        """POST-Request mit Retry."""
        return await self._make_request_with_retry("POST", url, **kwargs)

    # JSON-Convenience-Methoden
    async def get_json(self, url: str = "", **kwargs: Any) -> Any:
        """GET-Request mit JSON-Response."""
        response = await self.get(url, **kwargs)
        return response.json()

    async def post_json(self, url: str = "", json_data: Any = None, **kwargs: Any) -> Any:
        """POST-Request mit JSON-Payload und JSON-Response."""
        if json_data is not None:
            kwargs["json"] = json_data
        response = await self.post(url, **kwargs)
        return response.json()

    async def put_json(self, url: str = "", json_data: Any = None, **kwargs: Any) -> Any:
        """PUT-Request mit JSON-Payload und JSON-Response."""
        if json_data is not None:
            kwargs["json"] = json_data
        response = await self.put(url, **kwargs)
        return response.json()

    async def health_check(self, health_endpoint: str = "/health") -> dict[str, Any]:
        """Führt Health-Check durch.

        Args:
            health_endpoint: Health-Check-Endpoint

        Returns:
            Health-Status-Dictionary
        """
        try:
            start_time = asyncio.get_event_loop().time()
            response = await self.get(health_endpoint)
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "status_code": response.status_code,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def close(self) -> None:
        """Schließt HTTP-Client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug(f"Unified HTTP Client für {self.config.base_url} geschlossen")
