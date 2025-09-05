"""Unit-Tests für services.core.unified_client.

Testet den konsolidierten HTTP-Client und Client Factory.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from services.core.client_factory import ClientFactory
from services.core.constants import (
    KEI_MCP_CLIENT_USER_AGENT,
    KEI_RPC_CLIENT_USER_AGENT,
    KEIKO_WEBHOOK_USER_AGENT,
)
from services.core.unified_client import UnifiedClientConfig, UnifiedHTTPClient


class TestUnifiedClientConfig:
    """Tests für UnifiedClientConfig."""

    def test_default_config(self):
        """Testet Standard-Konfiguration."""
        config = UnifiedClientConfig(base_url="https://api.example.com")

        assert config.base_url == "https://api.example.com"
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.circuit_breaker_enabled is True
        assert config.http2_enabled is True
        assert config.verify_ssl is True

    def test_kei_rpc_config(self):
        """Testet KEI-RPC Konfiguration."""
        config = UnifiedClientConfig.for_kei_rpc(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        assert config.base_url == "https://api-grpc.example.com"
        assert config.bearer_token == "test-token"
        assert config.tenant_id == "test-tenant"
        assert config.user_agent == KEI_RPC_CLIENT_USER_AGENT
        assert config.circuit_breaker_name == "api-grpc-upstream"

    def test_mcp_client_config(self):
        """Testet MCP Client Konfiguration."""
        config = UnifiedClientConfig.for_mcp_client(
            base_url="https://mcp.example.com",
            server_name="test-server"
        )

        assert config.base_url == "https://mcp.example.com"
        assert config.user_agent == KEI_MCP_CLIENT_USER_AGENT
        assert config.circuit_breaker_name == "mcp_client_test-server"

    def test_webhook_config(self):
        """Testet Webhook Konfiguration."""
        config = UnifiedClientConfig.for_webhook(
            base_url="https://hooks.slack.com/webhook"
        )

        assert config.base_url == "https://hooks.slack.com/webhook"
        assert config.user_agent == KEIKO_WEBHOOK_USER_AGENT
        assert config.circuit_breaker_enabled is False


class TestUnifiedHTTPClient:
    """Tests für UnifiedHTTPClient."""

    def test_client_initialization(self):
        """Testet Client-Initialisierung."""
        config = UnifiedClientConfig(base_url="https://api.example.com")
        client = UnifiedHTTPClient(config)

        assert client.config == config
        assert client._client is None
        assert client._circuit_breaker is not None  # Circuit Breaker ist standardmäßig aktiviert

    def test_client_initialization_without_circuit_breaker(self):
        """Testet Client-Initialisierung ohne Circuit Breaker."""
        config = UnifiedClientConfig(
            base_url="https://api.example.com",
            circuit_breaker_enabled=False
        )
        client = UnifiedHTTPClient(config)

        assert client._circuit_breaker is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Testet Async Context Manager."""
        config = UnifiedClientConfig(base_url="https://api.example.com")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with UnifiedHTTPClient(config) as client:
                assert client._client is not None

            # Client sollte geschlossen worden sein
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_creates_httpx_client(self):
        """Testet _ensure_client erstellt httpx.AsyncClient."""
        config = UnifiedClientConfig(
            base_url="https://api.example.com",
            bearer_token="test-token",
            tenant_id="test-tenant",
            user_agent="Test-Agent"
        )
        client = UnifiedHTTPClient(config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            await client._ensure_client()

            # httpx.AsyncClient sollte mit korrekten Parametern erstellt worden sein
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs

            assert call_kwargs["base_url"] == "https://api.example.com"
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"
            assert call_kwargs["headers"]["X-Tenant-ID"] == "test-tenant"
            assert call_kwargs["headers"]["User-Agent"] == "Test-Agent"

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Testet erfolgreichen HTTP-Request."""
        config = UnifiedClientConfig(base_url="https://api.example.com")
        client = UnifiedHTTPClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await client._make_request("GET", "/test")

            assert response == mock_response
            mock_client.request.assert_called_once_with("GET", "https://api.example.com/test")

    @pytest.mark.asyncio
    async def test_make_request_with_circuit_breaker(self):
        """Testet HTTP-Request mit Circuit Breaker."""
        config = UnifiedClientConfig(
            base_url="https://api.example.com",
            circuit_breaker_name="test-cb"
        )
        client = UnifiedHTTPClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            with patch.object(client._circuit_breaker, "call") as mock_cb_call:
                mock_cb_call.return_value = mock_response

                response = await client._make_request("GET", "/test")

                assert response == mock_response
                mock_cb_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_methods(self):
        """Testet alle HTTP-Methoden."""
        config = UnifiedClientConfig(base_url="https://api.example.com")
        client = UnifiedHTTPClient(config)

        mock_response = MagicMock()

        with patch.object(client, "_make_request") as mock_make_request:
            mock_make_request.return_value = mock_response

            # GET
            response = await client.get("/test")
            assert response == mock_response
            mock_make_request.assert_called_with("GET", "/test")

            # POST
            response = await client.post("/test", json={"key": "value"})
            assert response == mock_response
            mock_make_request.assert_called_with("POST", "/test", json={"key": "value"})

            # PUT
            response = await client.put("/test")
            assert response == mock_response
            mock_make_request.assert_called_with("PUT", "/test")

            # DELETE
            response = await client.delete("/test")
            assert response == mock_response
            mock_make_request.assert_called_with("DELETE", "/test")

    @pytest.mark.asyncio
    async def test_json_convenience_methods(self):
        """Testet JSON-Convenience-Methoden."""
        config = UnifiedClientConfig(base_url="https://api.example.com")
        client = UnifiedHTTPClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}

        with patch.object(client, "get") as mock_get:
            mock_get.return_value = mock_response

            result = await client.get_json("/test")
            assert result == {"result": "success"}
            mock_get.assert_called_once_with("/test")

        with patch.object(client, "post") as mock_post:
            mock_post.return_value = mock_response

            result = await client.post_json("/test", json_data={"input": "data"})
            assert result == {"result": "success"}
            mock_post.assert_called_once_with("/test", json={"input": "data"})

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Testet erfolgreichen Health-Check."""
        config = UnifiedClientConfig(base_url="https://api.example.com")
        client = UnifiedHTTPClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "get") as mock_get:
            mock_get.return_value = mock_response

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert result["status_code"] == 200
            assert "response_time_ms" in result
            mock_get.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Testet fehlgeschlagenen Health-Check."""
        config = UnifiedClientConfig(base_url="https://api.example.com")
        client = UnifiedHTTPClient(config)

        with patch.object(client, "get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")

            result = await client.health_check()

            assert result["status"] == "unhealthy"
            assert "Connection failed" in result["error"]


class TestClientFactory:
    """Tests für ClientFactory."""

    def test_create_kei_rpc_client(self):
        """Testet KEI-RPC Client-Erstellung."""
        client = ClientFactory.create_kei_rpc_client(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        assert isinstance(client, UnifiedHTTPClient)
        assert client.config.base_url == "https://api-grpc.example.com"
        assert client.config.bearer_token == "test-token"
        assert client.config.tenant_id == "test-tenant"

    def test_create_mcp_client(self):
        """Testet MCP Client-Erstellung."""
        client = ClientFactory.create_mcp_client(
            base_url="https://mcp.example.com",
            server_name="test-server"
        )

        assert isinstance(client, UnifiedHTTPClient)
        assert client.config.base_url == "https://mcp.example.com"
        assert client.config.circuit_breaker_name == "mcp_client_test-server"

    def test_create_webhook_client(self):
        """Testet Webhook Client-Erstellung."""
        client = ClientFactory.create_webhook_client(
            webhook_url="https://hooks.slack.com/webhook"
        )

        assert isinstance(client, UnifiedHTTPClient)
        assert client.config.base_url == "https://hooks.slack.com/webhook"
        assert client.config.circuit_breaker_enabled is False

    def test_create_alerting_client(self):
        """Testet Alerting Client-Erstellung."""
        client = ClientFactory.create_alerting_client(
            webhook_url="https://hooks.slack.com/webhook",
            adapter_type="slack"
        )

        assert isinstance(client, UnifiedHTTPClient)
        assert client.config.base_url == "https://hooks.slack.com/webhook"
        assert "Slack" in client.config.user_agent

    def test_create_monitoring_client(self):
        """Testet Monitoring Client-Erstellung."""
        client = ClientFactory.create_monitoring_client(
            base_url="https://service.example.com",
            service_name="test-service"
        )

        assert isinstance(client, UnifiedHTTPClient)
        assert client.config.base_url == "https://service.example.com"
        assert client.config.timeout_seconds == 5.0
        assert client.config.max_retries == 1

    def test_create_generic_client(self):
        """Testet generischen Client-Erstellung."""
        client = ClientFactory.create_generic_client(
            base_url="https://api.example.com"
        )

        assert isinstance(client, UnifiedHTTPClient)
        assert client.config.base_url == "https://api.example.com"
