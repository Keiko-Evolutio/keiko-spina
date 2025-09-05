"""Tests für den ReportingService.

Testet die Service-Interface-Integration und Lifecycle-Management.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.interfaces import ServiceStatus
from services.reporting import (
    ReportingService,
    ReportingServiceConfig,
    ReportingServiceError,
    SchedulerError,
)


class TestReportingService:
    """Tests für den ReportingService."""

    @pytest.fixture
    def mock_grafana_client(self):
        """Mock für GrafanaClient."""
        client = AsyncMock()
        client.base_url = "https://test.grafana.com"
        client.api_token = "test-token"
        client.config = MagicMock()
        client.config.panel_timeout_seconds = 30.0
        client.config.dashboard_timeout_seconds = 60.0
        return client

    @pytest.fixture
    def mock_scheduler(self):
        """Mock für ReportingScheduler."""
        scheduler = AsyncMock()
        scheduler._running = False
        scheduler._task = None
        scheduler.interval_minutes = 60
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()
        scheduler._generate_and_distribute_reports = AsyncMock()
        return scheduler

    @pytest.fixture
    def reporting_config(self):
        """Test-Konfiguration für ReportingService."""
        return ReportingServiceConfig()

    def test_service_initialization(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Service-Initialisierung."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        assert service.config == reporting_config
        assert service.grafana_client == mock_grafana_client
        assert service.scheduler == mock_scheduler
        assert service._status == ServiceStatus.UNINITIALIZED
        assert not service._initialized

    def test_service_initialization_with_defaults(self):
        """Testet Service-Initialisierung mit Standard-Komponenten."""
        with patch("services.reporting.service.GrafanaClient") as mock_grafana_class, \
             patch("services.reporting.service.ReportingScheduler") as mock_scheduler_class:

            mock_grafana = AsyncMock()
            mock_scheduler = AsyncMock()
            mock_grafana_class.return_value = mock_grafana
            mock_scheduler_class.return_value = mock_scheduler

            service = ReportingService()

            assert service.grafana_client == mock_grafana
            assert service.scheduler == mock_scheduler
            mock_grafana_class.assert_called_once()
            mock_scheduler_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_initialize_with_reporting_enabled(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Service-Initialisierung mit aktiviertem Reporting."""
        with patch("services.reporting.service.settings") as mock_settings:
            mock_settings.reporting_enabled = True

            service = ReportingService(
                config=reporting_config,
                grafana_client=mock_grafana_client,
                scheduler=mock_scheduler
            )

            await service.initialize()

            assert service._initialized
            assert service._status == ServiceStatus.RUNNING
            mock_scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_initialize_with_reporting_disabled(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Service-Initialisierung mit deaktiviertem Reporting."""
        with patch("services.reporting.service.settings") as mock_settings:
            mock_settings.reporting_enabled = False

            service = ReportingService(
                config=reporting_config,
                grafana_client=mock_grafana_client,
                scheduler=mock_scheduler
            )

            await service.initialize()

            assert service._initialized
            assert service._status == ServiceStatus.RUNNING
            mock_scheduler.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_initialize_already_initialized(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet doppelte Initialisierung."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._initialized = True

        await service.initialize()

        # Sollte keine Aktionen durchführen
        mock_scheduler.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_initialize_error(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Initialisierungsfehler."""
        mock_scheduler.start.side_effect = Exception("Scheduler error")

        with patch("services.reporting.service.settings") as mock_settings:
            mock_settings.reporting_enabled = True

            service = ReportingService(
                config=reporting_config,
                grafana_client=mock_grafana_client,
                scheduler=mock_scheduler
            )

            with pytest.raises(ReportingServiceError):
                await service.initialize()

            assert service._status == ServiceStatus.ERROR
            assert not service._initialized

    @pytest.mark.asyncio
    async def test_service_shutdown(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Service-Shutdown."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._initialized = True
        service._status = ServiceStatus.RUNNING

        await service.shutdown()

        assert not service._initialized
        assert service._status == ServiceStatus.STOPPED
        mock_scheduler.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_shutdown_not_initialized(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Shutdown ohne Initialisierung."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        await service.shutdown()

        # Sollte keine Aktionen durchführen
        mock_scheduler.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Health-Check bei gesundem Service."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._status = ServiceStatus.RUNNING
        service._initialized = True

        with patch("services.reporting.service.settings") as mock_settings:
            mock_settings.reporting_enabled = True

            health = await service.health_check()

            assert health["status"] == "healthy"
            assert health["service_name"] == reporting_config.service_name
            assert health["initialized"] is True
            assert "scheduler" in health
            assert "grafana" in health

    @pytest.mark.asyncio
    async def test_health_check_error(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Health-Check bei Service-Fehler."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._status = ServiceStatus.ERROR

        with patch("services.reporting.service.settings") as mock_settings:
            mock_settings.reporting_enabled = True

            health = await service.health_check()

            assert health["status"] == "unhealthy"

    def test_get_status(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Status-Abfrage."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._status = ServiceStatus.RUNNING

        assert service.get_status() == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_generate_manual_report_success(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet erfolgreiche manuelle Report-Generierung."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._initialized = True

        result = await service.generate_manual_report()

        assert result["success"] is True
        assert "timestamp" in result
        mock_scheduler._generate_and_distribute_reports.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_manual_report_not_initialized(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet manuelle Report-Generierung ohne Initialisierung."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        with pytest.raises(ReportingServiceError):
            await service.generate_manual_report()

    @pytest.mark.asyncio
    async def test_start_scheduler_success(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet erfolgreichen Scheduler-Start."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        result = await service.start_scheduler()

        assert result["success"] is True
        mock_scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_scheduler_error(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Scheduler-Start-Fehler."""
        mock_scheduler.start.side_effect = Exception("Start error")

        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        with pytest.raises(SchedulerError):
            await service.start_scheduler()

    @pytest.mark.asyncio
    async def test_stop_scheduler_success(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet erfolgreichen Scheduler-Stop."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        result = await service.stop_scheduler()

        assert result["success"] is True
        mock_scheduler.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_scheduler_error(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Scheduler-Stop-Fehler."""
        mock_scheduler.stop.side_effect = Exception("Stop error")

        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        result = await service.stop_scheduler()

        assert result["success"] is False
        assert "Stop error" in result["message"]

    def test_get_service_info(self, reporting_config, mock_grafana_client, mock_scheduler):
        """Testet Service-Info-Abfrage."""
        service = ReportingService(
            config=reporting_config,
            grafana_client=mock_grafana_client,
            scheduler=mock_scheduler
        )

        service._status = ServiceStatus.RUNNING
        service._initialized = True

        with patch("services.reporting.service.settings") as mock_settings:
            mock_settings.reporting_enabled = True

            info = service.get_service_info()

            assert info["name"] == reporting_config.service_name
            assert info["version"] == reporting_config.version
            assert info["type"] == "FeatureService"
            assert info["status"] == ServiceStatus.RUNNING.value
            assert "capabilities" in info
            assert "configuration" in info
