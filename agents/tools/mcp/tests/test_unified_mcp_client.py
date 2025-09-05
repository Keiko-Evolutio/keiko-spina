"""Unit-Tests für kei_mcp.unified_mcp_client.

Testet die Migration von KEIMCPClient zu UnifiedKEIMCPClient.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agents.tools.mcp.kei_mcp_client import ExternalMCPConfig, MCPResource, MCPTool, MCPToolResult
from agents.tools.mcp.unified_mcp_client import UnifiedKEIMCPClient


class TestUnifiedKEIMCPClient:
    """Tests für UnifiedKEIMCPClient."""

    def test_client_initialization(self):
        """Testet Client-Initialisierung."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com",
            timeout_seconds=30.0
        )

        client = UnifiedKEIMCPClient(config)

        assert client.config == config
        assert client._unified_client is None
        assert client._semaphore is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Testet Async Context Manager."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        with patch.object(UnifiedKEIMCPClient, "_ensure_client") as mock_ensure:
            with patch.object(UnifiedKEIMCPClient, "close") as mock_close:
                async with UnifiedKEIMCPClient(config) as client:
                    assert isinstance(client, UnifiedKEIMCPClient)

                mock_ensure.assert_called_once()
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_creates_unified_client(self):
        """Testet _ensure_client erstellt UnifiedHTTPClient."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com",
            api_key="test-key"
        )

        client = UnifiedKEIMCPClient(config)

        with patch("agents.tools.mcp.unified_mcp_client.UnifiedHTTPClient") as mock_client_class:
            mock_unified_client = AsyncMock()
            mock_client_class.return_value = mock_unified_client

            await client._ensure_client()

            # UnifiedHTTPClient sollte erstellt worden sein
            mock_client_class.assert_called_once()
            assert client._unified_client == mock_unified_client
            mock_unified_client._ensure_client.assert_called_once()

    def test_create_unified_config(self):
        """Testet _create_unified_config."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com",
            api_key="test-key",
            timeout_seconds=45.0,
            max_retries=5,
            custom_headers={"X-Custom": "value"}
        )

        client = UnifiedKEIMCPClient(config)
        unified_config = client._create_unified_config()

        assert unified_config.base_url == "https://mcp.example.com"
        assert unified_config.timeout_seconds == 45.0
        assert unified_config.max_retries == 5
        assert unified_config.custom_headers["X-API-Key"] == "test-key"
        assert unified_config.custom_headers["X-Custom"] == "value"
        assert unified_config.circuit_breaker_name == "mcp_client_test-server"

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Testet erfolgreichen Tool-Aufruf."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.post_json.return_value = {"result": "success", "data": "test"}
        client._unified_client = mock_unified_client

        # Mock Semaphore
        client._semaphore = AsyncMock()

        # Mock Metrics
        with patch("agents.tools.mcp.unified_mcp_client.kei_mcp_metrics") as mock_metrics:
            with patch("asyncio.Semaphore") as mock_semaphore_class:
                mock_semaphore = AsyncMock()
                mock_semaphore_class.return_value = mock_semaphore
                client._semaphore = mock_semaphore

                result = await client.call_tool("test_tool", {"arg1": "value1"})

                assert isinstance(result, MCPToolResult)
                assert result.success is True
                assert result.result == {"result": "success", "data": "test"}
                assert result.server == "test-server"
                assert result.execution_time_ms > 0

                # Verify HTTP call
                mock_unified_client.post_json.assert_called_once_with(
                    "/tools/call",
                    json_data={"tool": "test_tool", "arguments": {"arg1": "value1"}},
                    timeout=30.0
                )

                # Verify metrics call
                mock_metrics.record_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_failure(self):
        """Testet fehlgeschlagenen Tool-Aufruf."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient mit Fehler
        mock_unified_client = AsyncMock()
        mock_unified_client.post_json.side_effect = Exception("Connection failed")
        client._unified_client = mock_unified_client

        # Mock Semaphore
        client._semaphore = AsyncMock()

        # Mock Metrics
        with patch("agents.tools.mcp.unified_mcp_client.kei_mcp_metrics") as mock_metrics:
            result = await client.call_tool("test_tool")

            assert isinstance(result, MCPToolResult)
            assert result.success is False
            assert "Connection failed" in result.error
            assert result.server == "test-server"

            # Verify metrics call for failure
            mock_metrics.record_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_success(self):
        """Testet erfolgreiches Auflisten von Tools."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.return_value = {
            "tools": [
                {
                    "name": "tool1",
                    "description": "Test tool 1",
                    "inputSchema": {"type": "object"}
                },
                {
                    "name": "tool2",
                    "description": "Test tool 2"
                }
            ]
        }
        client._unified_client = mock_unified_client

        tools = await client.list_tools()

        assert len(tools) == 2
        assert isinstance(tools[0], MCPTool)
        assert tools[0].name == "tool1"
        assert tools[0].description == "Test tool 1"
        assert tools[1].name == "tool2"

        mock_unified_client.get_json.assert_called_once_with("/tools")

    @pytest.mark.asyncio
    async def test_list_tools_failure(self):
        """Testet fehlgeschlagenes Auflisten von Tools."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient mit Fehler
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.side_effect = Exception("Server error")
        client._unified_client = mock_unified_client

        tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_get_resource_success(self):
        """Testet erfolgreiches Abrufen einer Ressource."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.return_value = {
            "content": "resource content",
            "mimeType": "text/plain"
        }
        client._unified_client = mock_unified_client

        result = await client.get_resource("file://test.txt")

        assert result.success is True
        assert result.resource["content"] == "resource content"
        assert result.server == "test-server"

        mock_unified_client.get_json.assert_called_once_with(
            "/resources/read",
            params={"uri": "file://test.txt"}
        )

    @pytest.mark.asyncio
    async def test_list_resources_success(self):
        """Testet erfolgreiches Auflisten von Ressourcen."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.return_value = {
            "resources": [
                {
                    "uri": "file://test1.txt",
                    "name": "Test File 1",
                    "description": "First test file",
                    "mimeType": "text/plain"
                },
                {
                    "uri": "file://test2.txt",
                    "name": "Test File 2"
                }
            ]
        }
        client._unified_client = mock_unified_client

        resources = await client.list_resources()

        assert len(resources) == 2
        assert isinstance(resources[0], MCPResource)
        assert resources[0].uri == "file://test1.txt"
        assert resources[0].name == "Test File 1"
        assert resources[1].uri == "file://test2.txt"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Testet erfolgreichen Health-Check."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.health_check.return_value = {"status": "healthy"}
        client._unified_client = mock_unified_client

        is_healthy = await client.health_check()

        assert is_healthy is True
        mock_unified_client.health_check.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Testet fehlgeschlagenen Health-Check."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient mit Fehler
        mock_unified_client = AsyncMock()
        mock_unified_client.health_check.side_effect = Exception("Connection failed")
        client._unified_client = mock_unified_client

        is_healthy = await client.health_check()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_close(self):
        """Testet Client-Schließung."""
        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = UnifiedKEIMCPClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        client._unified_client = mock_unified_client

        await client.close()

        mock_unified_client.close.assert_called_once()
        assert client._unified_client is None


class TestBackwardCompatibility:
    """Tests für Backward-Compatibility."""

    def test_kei_mcp_client_alias(self):
        """Testet KEIMCPClient Alias."""
        from agents.tools.mcp.unified_mcp_client import KEIMCPClient

        config = ExternalMCPConfig(
            server_name="test-server",
            base_url="https://mcp.example.com"
        )

        client = KEIMCPClient(config)
        assert isinstance(client, UnifiedKEIMCPClient)
