"""Unit-Tests für services.clients.unified_kei_rpc_client.

Testet die Migration von KEIRPCClient zu UnifiedKEIRPCClient.
"""

from unittest.mock import AsyncMock, patch

import pytest

from services.clients.kei_rpc_client import KEIRPCClientConfig
from services.clients.unified_kei_rpc_client import UnifiedKEIRPCClient


class TestUnifiedKEIRPCClient:
    """Tests für UnifiedKEIRPCClient."""

    def test_client_initialization(self):
        """Testet Client-Initialisierung."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant",
            timeout_seconds=45.0
        )

        client = UnifiedKEIRPCClient(config)

        assert client.config == config
        assert client._unified_client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Testet Async Context Manager."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        with patch.object(UnifiedKEIRPCClient, "_ensure_client") as mock_ensure:
            with patch.object(UnifiedKEIRPCClient, "close") as mock_close:
                async with UnifiedKEIRPCClient(config) as client:
                    assert isinstance(client, UnifiedKEIRPCClient)

                mock_ensure.assert_called_once()
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_creates_unified_client(self):
        """Testet _ensure_client erstellt UnifiedHTTPClient."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant",
            timeout_seconds=45.0  # Expliziter Timeout
        )

        client = UnifiedKEIRPCClient(config)

        with patch("services.clients.unified_kei_rpc_client.ClientFactory") as mock_factory:
            mock_unified_client = AsyncMock()
            mock_factory.create_kei_rpc_client.return_value = mock_unified_client

            await client._ensure_client()

            # ClientFactory sollte aufgerufen worden sein
            mock_factory.create_kei_rpc_client.assert_called_once_with(
                base_url="https://api-grpc.example.com",
                api_token="test-token",
                tenant_id="test-tenant",
                timeout_seconds=45.0  # Expliziter Timeout aus config
            )
            assert client._unified_client == mock_unified_client
            mock_unified_client._ensure_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_resource_success(self):
        """Testet erfolgreiches Abrufen einer Ressource."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.return_value = {
            "id": "resource-123",
            "name": "Test Resource",
            "type": "document"
        }
        client._unified_client = mock_unified_client

        result = await client.get_resource("resource-123")

        assert result["id"] == "resource-123"
        assert result["name"] == "Test Resource"
        mock_unified_client.get_json.assert_called_once_with("/resources/resource-123")

    @pytest.mark.asyncio
    async def test_list_resources_success(self):
        """Testet erfolgreiches Auflisten von Ressourcen."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.return_value = {
            "resources": [
                {"id": "resource-1", "name": "Resource 1"},
                {"id": "resource-2", "name": "Resource 2"}
            ],
            "total": 2,
            "limit": 10,
            "offset": 0
        }
        client._unified_client = mock_unified_client

        result = await client.list_resources(limit=10, offset=0, filters={"type": "document"})

        assert len(result["resources"]) == 2
        assert result["total"] == 2
        mock_unified_client.get_json.assert_called_once_with(
            "/resources",
            params={"limit": 10, "offset": 0, "type": "document"}
        )

    @pytest.mark.asyncio
    async def test_create_resource_success(self):
        """Testet erfolgreiches Erstellen einer Ressource."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.post_json.return_value = {
            "id": "resource-new",
            "name": "New Resource",
            "type": "document",
            "created_at": "2024-01-01T00:00:00Z"
        }
        client._unified_client = mock_unified_client

        resource_data = {
            "name": "New Resource",
            "type": "document",
            "content": "Test content"
        }

        result = await client.create_resource(resource_data)

        assert result["id"] == "resource-new"
        assert result["name"] == "New Resource"
        mock_unified_client.post_json.assert_called_once_with(
            "/resources",
            json_data=resource_data
        )

    @pytest.mark.asyncio
    async def test_update_resource_success(self):
        """Testet erfolgreiches Aktualisieren einer Ressource."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.put_json.return_value = {
            "id": "resource-123",
            "name": "Updated Resource",
            "type": "document",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        client._unified_client = mock_unified_client

        resource_data = {
            "name": "Updated Resource",
            "content": "Updated content"
        }

        result = await client.update_resource("resource-123", resource_data)

        assert result["id"] == "resource-123"
        assert result["name"] == "Updated Resource"
        mock_unified_client.put_json.assert_called_once_with(
            "/resources/resource-123",
            json_data=resource_data
        )

    @pytest.mark.asyncio
    async def test_delete_resource_success(self):
        """Testet erfolgreiches Löschen einer Ressource."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        client._unified_client = mock_unified_client

        await client.delete_resource("resource-123")

        mock_unified_client.delete.assert_called_once_with("/resources/resource-123")

    @pytest.mark.asyncio
    async def test_execute_action_success(self):
        """Testet erfolgreiches Ausführen einer Aktion."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.post_json.return_value = {
            "action": "process_document",
            "status": "completed",
            "result": {"processed_pages": 5}
        }
        client._unified_client = mock_unified_client

        action_data = {"document_id": "doc-123", "options": {"ocr": True}}

        result = await client.execute_action("process_document", action_data)

        assert result["action"] == "process_document"
        assert result["status"] == "completed"
        mock_unified_client.post_json.assert_called_once_with(
            "/actions",
            json_data={
                "action": "process_document",
                "data": action_data
            }
        )

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """Testet erfolgreiches Abrufen des Status."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        mock_unified_client.get_json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600
        }
        client._unified_client = mock_unified_client

        result = await client.get_status()

        assert result["status"] == "healthy"
        assert result["version"] == "1.0.0"
        mock_unified_client.get_json.assert_called_once_with("/status")

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Testet erfolgreichen Health-Check."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

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
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient mit Fehler
        mock_unified_client = AsyncMock()
        mock_unified_client.health_check.side_effect = Exception("Connection failed")
        client._unified_client = mock_unified_client

        is_healthy = await client.health_check()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_close(self):
        """Testet Client-Schließung."""
        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = UnifiedKEIRPCClient(config)

        # Mock UnifiedHTTPClient
        mock_unified_client = AsyncMock()
        client._unified_client = mock_unified_client

        await client.close()

        mock_unified_client.close.assert_called_once()
        assert client._unified_client is None


class TestBackwardCompatibility:
    """Tests für Backward-Compatibility."""

    def test_kei_rpc_client_alias(self):
        """Testet KEIRPCClient Alias."""
        from services.clients.unified_kei_rpc_client import KEIRPCClient

        config = KEIRPCClientConfig(
            base_url="https://api-grpc.example.com",
            api_token="test-token",
            tenant_id="test-tenant"
        )

        client = KEIRPCClient(config)
        assert isinstance(client, UnifiedKEIRPCClient)
