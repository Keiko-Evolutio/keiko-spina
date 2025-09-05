# backend/services/core/tests/test_manager.py
"""Tests für Service Manager.

Testet Service-Lifecycle, Health-Monitoring und Cleanup-Funktionalität.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.core.manager import (
    CleanupTask,
    ServiceManager,
    ServiceSetupResult,
)


class TestServiceSetupResult:
    """Tests für ServiceSetupResult."""

    def test_successful_result(self):
        """Testet erfolgreiches Setup-Ergebnis."""
        service = MagicMock()
        result = ServiceSetupResult("test_service", True, service)

        assert result.name == "test_service"
        assert result.success is True
        assert result.service == service
        assert result.error is None

    def test_failed_result(self):
        """Testet fehlgeschlagenes Setup-Ergebnis."""
        result = ServiceSetupResult("test_service", False, error="Setup failed")

        assert result.name == "test_service"
        assert result.success is False
        assert result.service is None
        assert result.error == "Setup failed"


class TestCleanupTask:
    """Tests für CleanupTask."""

    def test_cleanup_task_creation(self):
        """Testet CleanupTask-Erstellung."""
        mock_coro = AsyncMock()
        task = CleanupTask("close", "test_service", mock_coro)

        assert task.task_type == "close"
        assert task.service_name == "test_service"
        assert task.coroutine == mock_coro


class TestServiceManager:
    """Tests für ServiceManager."""

    @pytest.fixture
    def service_manager(self):
        """Service Manager Instanz für Tests."""
        return ServiceManager()

    def test_initialization(self, service_manager):
        """Testet initiale Werte."""
        assert service_manager._services == {}
        assert service_manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, service_manager):
        """Testet dass bereits initialisierter Manager übersprungen wird."""
        service_manager._initialized = True

        with patch.object(service_manager, "_setup_available_services") as mock_setup:
            await service_manager.initialize()
            mock_setup.assert_not_called()

    @pytest.mark.asyncio
    @patch("services.core.manager.features")
    async def test_initialize_with_services(self, mock_features, service_manager):
        """Testet Initialisierung mit verfügbaren Services."""
        # Mock verfügbare Features
        mock_features.is_available.side_effect = lambda feature: feature == "http_clients"

        # Mock Setup-Methode
        setup_result = ServiceSetupResult("http_client", True, MagicMock())
        with patch.object(service_manager, "_setup_http_client", return_value=setup_result):
            await service_manager.initialize()

        assert service_manager._initialized is True

    @pytest.mark.asyncio
    @patch("services.core.manager.features")
    async def test_setup_available_services(self, mock_features, service_manager):
        """Testet Setup aller verfügbaren Services."""
        # Mock alle Features als verfügbar
        mock_features.is_available.return_value = True

        # Mock Setup-Methoden
        http_result = ServiceSetupResult("http_client", True, MagicMock())
        azure_result = ServiceSetupResult("azure_services", True, MagicMock())
        pools_result = ServiceSetupResult("pools", True, True)

        with patch.object(service_manager, "_setup_http_client", return_value=http_result), \
             patch.object(service_manager, "_setup_azure_services", return_value=azure_result), \
             patch.object(service_manager, "_setup_pools", return_value=pools_result):

            results = await service_manager._setup_available_services()

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    @patch("services.clients.clients.HTTPClient")
    async def test_setup_http_client_success(self, mock_http_client, service_manager):
        """Testet erfolgreiches HTTP-Client Setup."""
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        result = await service_manager._setup_http_client()

        assert result.success is True
        assert result.name == "http_client"
        assert result.service == mock_client
        assert service_manager._services["http_client"] == mock_client

    @pytest.mark.asyncio
    async def test_setup_http_client_failure(self, service_manager):
        """Testet fehlgeschlagenes HTTP-Client Setup."""
        with patch("services.clients.clients.HTTPClient", side_effect=ImportError("Module not found")):
            result = await service_manager._setup_http_client()

        assert result.success is False
        assert result.name == "http_client"
        assert result.service is None
        assert "HTTP-Client Setup fehlgeschlagen" in result.error

    @pytest.mark.asyncio
    @patch("services.clients.clients.Services")
    async def test_setup_azure_services_success(self, mock_services, service_manager):
        """Testet erfolgreiches Azure Services Setup."""
        mock_service = MagicMock()
        mock_services.return_value = mock_service

        with patch.object(service_manager, "_setup_deep_research_service"):
            result = await service_manager._setup_azure_services()

        assert result.success is True
        assert result.name == "azure_services"
        assert service_manager._services["azure_services"] == mock_service

    @pytest.mark.asyncio
    async def test_setup_azure_services_failure(self, service_manager):
        """Testet fehlgeschlagenes Azure Services Setup."""
        with patch("services.clients.clients.Services", side_effect=ImportError("Module not found")):
            result = await service_manager._setup_azure_services()

        assert result.success is False
        assert result.name == "azure_services"

    @pytest.mark.asyncio
    @patch("services.clients.deep_research.create_deep_research_service")
    async def test_setup_deep_research_service_success(self, mock_create_dr, service_manager):
        """Testet erfolgreiches Deep Research Service Setup."""
        mock_dr_service = MagicMock()
        mock_create_dr.return_value = mock_dr_service

        await service_manager._setup_deep_research_service()

        assert service_manager._services["deep_research"] == mock_dr_service

    @pytest.mark.asyncio
    @patch("services.clients.deep_research.create_deep_research_service")
    async def test_setup_deep_research_service_none(self, mock_create_dr, service_manager):
        """Testet Deep Research Service Setup mit None-Rückgabe."""
        mock_create_dr.return_value = None

        await service_manager._setup_deep_research_service()

        assert "deep_research" not in service_manager._services

    @pytest.mark.asyncio
    async def test_setup_deep_research_service_failure(self, service_manager):
        """Testet fehlgeschlagenes Deep Research Service Setup."""
        with patch("services.clients.deep_research.create_deep_research_service",
                   side_effect=ImportError("Module not found")):
            await service_manager._setup_deep_research_service()

        # Sollte ohne Fehler durchlaufen, aber Service nicht registrieren
        assert "deep_research" not in service_manager._services

    @pytest.mark.asyncio
    @patch("services.pools.azure_pools.initialize_pools")
    async def test_setup_pools_success(self, mock_initialize, service_manager):
        """Testet erfolgreiches Pool Setup."""
        mock_initialize.return_value = None

        result = await service_manager._setup_pools()

        assert result.success is True
        assert result.name == "pools"
        assert service_manager._services["pools"] is True

    @pytest.mark.asyncio
    async def test_setup_pools_failure(self, service_manager):
        """Testet fehlgeschlagenes Pool Setup."""
        with patch("services.pools.azure_pools.initialize_pools",
                   side_effect=ImportError("Module not found")):
            result = await service_manager._setup_pools()

        assert result.success is False
        assert result.name == "pools"

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self, service_manager):
        """Testet Cleanup bei nicht initialisiertem Manager."""
        await service_manager.cleanup()

        # Sollte ohne Fehler durchlaufen
        assert service_manager._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_with_services(self, service_manager):
        """Testet Cleanup mit Services."""
        # Setup Services mit Mock-Cleanup-Methoden
        mock_service1 = MagicMock()

        # Erstelle echte Coroutine für close
        async def mock_close():
            pass
        mock_service1.close.return_value = mock_close()

        mock_service2 = MagicMock()

        # Erstelle echte Coroutine für cleanup
        async def mock_cleanup():
            pass
        mock_service2.cleanup.return_value = mock_cleanup()
        # Service2 hat keine close-Methode
        del mock_service2.close

        service_manager._services = {
            "service1": mock_service1,
            "service2": mock_service2
        }
        service_manager._initialized = True

        # Mock inspect.signature für service1.close
        with patch("inspect.signature") as mock_signature:
            mock_sig = MagicMock()
            mock_sig.parameters = {}  # Keine Parameter
            mock_signature.return_value = mock_sig

            await service_manager.cleanup()

        mock_service1.close.assert_called_once()
        mock_service2.cleanup.assert_called_once()
        assert service_manager._services == {}
        assert service_manager._initialized is False

    def test_create_service_cleanup_task_with_close(self, service_manager):
        """Testet Cleanup-Task-Erstellung für Service mit close-Methode."""
        mock_service = MagicMock()
        mock_coro = AsyncMock()
        mock_service.close.return_value = mock_coro

        # Mock inspect.signature um Parameter-Check zu umgehen
        with patch("inspect.signature") as mock_signature:
            mock_sig = MagicMock()
            mock_sig.parameters = {}  # Keine Parameter
            mock_signature.return_value = mock_sig

            task = service_manager._create_service_cleanup_task("test", mock_service)

        assert task is not None
        assert task.task_type == "close"
        assert task.service_name == "test"

    def test_create_service_cleanup_task_with_cleanup(self, service_manager):
        """Testet Cleanup-Task-Erstellung für Service mit cleanup-Methode."""
        mock_service = MagicMock()
        mock_coro = AsyncMock()
        mock_service.cleanup.return_value = mock_coro
        # Service hat keine close-Methode
        del mock_service.close

        task = service_manager._create_service_cleanup_task("test", mock_service)

        assert task is not None
        assert task.task_type == "cleanup"
        assert task.service_name == "test"

    def test_create_service_cleanup_task_no_methods(self, service_manager):
        """Testet Cleanup-Task-Erstellung für Service ohne Cleanup-Methoden."""
        mock_service = MagicMock(spec=[])  # Keine Methoden

        task = service_manager._create_service_cleanup_task("test", mock_service)

        assert task is None

    @pytest.mark.asyncio
    async def test_get_health_status(self, service_manager):
        """Testet Health-Status-Abfrage."""
        mock_service = MagicMock()
        mock_service.is_healthy = AsyncMock(return_value=True)

        service_manager._services = {"test_service": mock_service}
        service_manager._initialized = True

        status = await service_manager.get_health_status()

        assert status["initialized"] is True
        assert status["service_count"] == 1
        assert status["services"]["test_service"] is True

    def test_get_service(self, service_manager):
        """Testet Service-Abruf."""
        mock_service = MagicMock()
        service_manager._services = {"test_service": mock_service}

        result = service_manager.get_service("test_service")
        assert result == mock_service

        result = service_manager.get_service("nonexistent")
        assert result is None

    def test_is_healthy(self, service_manager):
        """Testet grundlegende Health-Prüfung."""
        # Nicht initialisiert
        assert service_manager.is_healthy() is False

        # Initialisiert aber keine Services
        service_manager._initialized = True
        assert service_manager.is_healthy() is False

        # Mit Services und HTTP-Clients verfügbar
        service_manager._services = {"test": MagicMock()}
        with patch("services.core.manager.features") as mock_features:
            mock_features.is_available.return_value = True
            assert service_manager.is_healthy() is True
