"""Client Factory für einheitliche HTTP-Client-Erstellung.

Konsolidiert alle Client-Erstellungslogik und bietet Backward-Compatibility.
"""

from __future__ import annotations

from typing import Any

from services.core.constants import (
    KEI_RPC_CLIENT_USER_AGENT,
)
from services.core.unified_client import UnifiedClientConfig, UnifiedHTTPClient


class ClientFactory:
    """Factory für einheitliche HTTP-Client-Erstellung.

    Konsolidiert alle Client-Patterns und bietet Backward-Compatibility
    für bestehende Implementierungen.
    """

    @staticmethod
    def create_kei_rpc_client(
        base_url: str,
        api_token: str,
        tenant_id: str,
        timeout_seconds: float = 30.0,
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt KEI-RPC Client.

        Ersetzt: services.clients.kei_rpc_client.KEIRPCClient

        Args:
            base_url: Basis-URL des KEI-RPC Services
            api_token: API-Token für Authentifizierung
            tenant_id: Tenant-ID
            timeout_seconds: Request-Timeout
            **kwargs: Zusätzliche Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        config = UnifiedClientConfig.for_kei_rpc(
            base_url=base_url,
            api_token=api_token,
            tenant_id=tenant_id,
            timeout_seconds=timeout_seconds,
            **kwargs
        )
        return UnifiedHTTPClient(config)

    @staticmethod
    def create_kei_grpc_client_http_fallback(
        base_url: str,
        api_token: str,
        tenant_id: str,
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt HTTP-Fallback für KEI-gRPC Client.

        Für Fälle wo gRPC nicht verfügbar ist.

        Args:
            base_url: Basis-URL des Services
            api_token: API-Token
            tenant_id: Tenant-ID
            **kwargs: Zusätzliche Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        config = UnifiedClientConfig(
            base_url=base_url,
            bearer_token=api_token,
            tenant_id=tenant_id,
            user_agent=KEI_RPC_CLIENT_USER_AGENT,
            circuit_breaker_name="kei-grpc-http-fallback",
            **kwargs
        )
        return UnifiedHTTPClient(config)

    @staticmethod
    def create_mcp_client(
        base_url: str,
        server_name: str,
        timeout_seconds: float = 30.0,
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt MCP Client.

        Ersetzt: agents.tools.mcp.kei_mcp_client.KEIMCPClient

        Args:
            base_url: Basis-URL des MCP Servers
            server_name: Name des MCP Servers
            timeout_seconds: Request-Timeout
            **kwargs: Zusätzliche Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        config = UnifiedClientConfig.for_mcp_client(
            base_url=base_url,
            server_name=server_name,
            timeout_seconds=timeout_seconds,
            **kwargs
        )
        return UnifiedHTTPClient(config)

    @staticmethod
    def create_webhook_client(
        webhook_url: str,
        timeout_seconds: float = 5.0,
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt Webhook Client.

        Ersetzt: services.webhooks.alerting HTTP-Clients

        Args:
            webhook_url: Webhook-URL
            timeout_seconds: Request-Timeout
            **kwargs: Zusätzliche Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        config = UnifiedClientConfig.for_webhook(
            base_url=webhook_url,
            timeout_seconds=timeout_seconds,
            **kwargs
        )
        return UnifiedHTTPClient(config)

    @staticmethod
    def create_alerting_client(
        webhook_url: str,
        adapter_type: str = "slack",
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt Alerting Client.

        Ersetzt: services.webhooks.alerting.SlackAdapter, TeamsAdapter

        Args:
            webhook_url: Webhook-URL
            adapter_type: Typ des Adapters (slack, teams)
            **kwargs: Zusätzliche Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        user_agent = f"Keiko-Alerting-{adapter_type.title()}/1.0"

        config = UnifiedClientConfig(
            base_url=webhook_url,
            timeout_seconds=5.0,
            user_agent=user_agent,
            circuit_breaker_enabled=False,  # Alerting hat eigene Retry-Logik
            **kwargs
        )
        return UnifiedHTTPClient(config)

    @staticmethod
    def create_monitoring_client(
        base_url: str,
        service_name: str,
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt Monitoring Client.

        Für Health-Checks, Metrics, etc.

        Args:
            base_url: Basis-URL des Services
            service_name: Name des Services
            **kwargs: Zusätzliche Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        config = UnifiedClientConfig(
            base_url=base_url,
            timeout_seconds=5.0,
            user_agent=f"Keiko-Monitor/{service_name}",
            circuit_breaker_name=f"monitor_{service_name}",
            max_retries=1,  # Monitoring sollte schnell fehlschlagen
            **kwargs
        )
        return UnifiedHTTPClient(config)

    @staticmethod
    def create_generic_client(
        base_url: str,
        **kwargs: Any
    ) -> UnifiedHTTPClient:
        """Erstellt generischen HTTP Client.

        Für allgemeine HTTP-Anfragen ohne spezielle Konfiguration.

        Args:
            base_url: Basis-URL
            **kwargs: Konfiguration

        Returns:
            Konfigurierter UnifiedHTTPClient
        """
        config = UnifiedClientConfig(
            base_url=base_url,
            **kwargs
        )
        return UnifiedHTTPClient(config)


# Convenience-Funktionen für häufige Use Cases
async def quick_get(url: str, **kwargs: Any) -> Any:
    """Schneller GET-Request mit JSON-Response.

    Args:
        url: Vollständige URL
        **kwargs: Request-Parameter

    Returns:
        JSON-Response
    """
    async with ClientFactory.create_generic_client(url) as client:
        return await client.get_json("", **kwargs)


async def quick_post(url: str, json_data: Any = None, **kwargs: Any) -> Any:
    """Schneller POST-Request mit JSON-Payload und JSON-Response.

    Args:
        url: Vollständige URL
        json_data: JSON-Payload
        **kwargs: Request-Parameter

    Returns:
        JSON-Response
    """
    async with ClientFactory.create_generic_client(url) as client:
        return await client.post_json("", json_data=json_data, **kwargs)


async def health_check(url: str, endpoint: str = "/health") -> dict[str, Any]:
    """Schneller Health-Check.

    Args:
        url: Basis-URL des Services
        endpoint: Health-Check-Endpoint

    Returns:
        Health-Status
    """
    async with ClientFactory.create_monitoring_client(url, "health_check") as client:
        return await client.health_check(endpoint)


# Backward-Compatibility Aliases
create_kei_rpc_client = ClientFactory.create_kei_rpc_client
create_mcp_client = ClientFactory.create_mcp_client
create_webhook_client = ClientFactory.create_webhook_client
