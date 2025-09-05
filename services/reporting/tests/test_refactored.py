"""Tests für das refactored Reporting-Modul.

Testet die verbesserte Funktionalität nach dem Clean Code Refactoring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.reporting import (
    GrafanaClient,
    GrafanaClientError,
    GrafanaConfig,
    ReportGenerationError,
    ReportingScheduler,
    ReportingServiceConfig,
)


class TestRefactoredGrafanaClient:
    """Tests für den refactored GrafanaClient."""

    @pytest.fixture
    def grafana_config(self):
        """Test-Konfiguration für GrafanaClient."""
        return GrafanaConfig(
            panel_timeout_seconds=10.0,
            dashboard_timeout_seconds=20.0,
            default_panel_width=800,
            default_panel_height=600,
            max_retries=2,
            retry_delay_seconds=0.1
        )

    @pytest.mark.asyncio
    async def test_grafana_client_with_config(self, grafana_config):
        """Testet GrafanaClient mit benutzerdefinierter Konfiguration."""
        client = GrafanaClient(
            base_url="https://test.grafana.com",
            api_token="test-token",
            config=grafana_config
        )

        assert client.base_url == "https://test.grafana.com"
        assert client.api_token == "test-token"
        assert client.config.max_retries == 2
        assert client.config.default_panel_width == 800

    def test_get_headers(self, grafana_config):
        """Testet Header-Generierung."""
        client = GrafanaClient(
            base_url="https://test.grafana.com",
            api_token="test-token",
            config=grafana_config
        )

        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_export_panel_png_with_retry_success(self, grafana_config):
        """Testet erfolgreichen PNG-Export mit Retry-Logic."""
        client = GrafanaClient(config=grafana_config)

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            # Mock HTTP-Client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"test-png-data"
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.export_panel_png("test-dashboard", 1)

            assert result == b"test-png-data"
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_panel_png_with_retry_failure(self, grafana_config):
        """Testet PNG-Export mit Retry-Fehlern."""
        client = GrafanaClient(config=grafana_config)

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            # Mock HTTP-Client mit Timeout-Fehler
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection error")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GrafanaClientError):
                await client.export_panel_png("test-dashboard", 1)

            # Sollte max_retries + 1 Versuche machen
            assert mock_client.get.call_count == grafana_config.max_retries + 1

    @pytest.mark.asyncio
    async def test_export_panel_png_authentication_error(self, grafana_config):
        """Testet PNG-Export mit Authentifizierungsfehler."""
        client = GrafanaClient(config=grafana_config)

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            # Mock HTTP-Client mit 401-Fehler
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(Exception):
                await client.export_panel_png("test-dashboard", 1)

            # Sollte nur einen Versuch machen (keine Retries bei Auth-Fehlern)
            assert mock_client.get.call_count == 1


class TestRefactoredReportingScheduler:
    """Tests für den refactored ReportingScheduler."""

    @pytest.fixture
    def reporting_config(self):
        """Test-Konfiguration für ReportingScheduler."""
        return ReportingServiceConfig()

    @pytest.fixture
    def mock_grafana_client(self):
        """Mock für GrafanaClient."""
        client = AsyncMock()
        client.export_panel_png.return_value = b"test-png-data"
        return client

    def test_scheduler_initialization_with_config(self, reporting_config, mock_grafana_client):
        """Testet Scheduler-Initialisierung mit Konfiguration."""
        scheduler = ReportingScheduler(
            interval_minutes=30,
            config=reporting_config,
            grafana_client=mock_grafana_client
        )

        assert scheduler.interval_minutes == 30
        assert scheduler.config == reporting_config
        assert scheduler.grafana_client == mock_grafana_client

    def test_parse_recipients(self, reporting_config, mock_grafana_client):
        """Testet Empfänger-Parsing."""
        scheduler = ReportingScheduler(config=reporting_config, grafana_client=mock_grafana_client)

        # Test mit gültigen Empfängern
        recipients = scheduler._parse_recipients("test@example.com, admin@example.com, ")
        assert recipients == ["test@example.com", "admin@example.com"]

        # Test mit leerem String
        recipients = scheduler._parse_recipients("")
        assert recipients == []

        # Test mit None
        recipients = scheduler._parse_recipients(None)
        assert recipients == []

    @pytest.mark.asyncio
    async def test_generate_report_success(self, reporting_config, mock_grafana_client):
        """Testet erfolgreiche Report-Generierung."""
        scheduler = ReportingScheduler(config=reporting_config, grafana_client=mock_grafana_client)

        result = await scheduler._generate_report()

        assert result == b"test-png-data"
        mock_grafana_client.export_panel_png.assert_called_once_with(
            dashboard_uid=reporting_config.report.default_dashboard_uid,
            panel_id=reporting_config.report.default_panel_id,
            params=reporting_config.report.default_export_params
        )

    @pytest.mark.asyncio
    async def test_generate_report_failure(self, reporting_config, mock_grafana_client):
        """Testet Report-Generierung mit Fehler."""
        mock_grafana_client.export_panel_png.side_effect = Exception("Grafana error")
        scheduler = ReportingScheduler(config=reporting_config, grafana_client=mock_grafana_client)

        with pytest.raises(ReportGenerationError) as exc_info:
            await scheduler._generate_report()

        assert "Report-Generierung fehlgeschlagen" in str(exc_info.value)
        assert exc_info.value.report_type == "png"

    @pytest.mark.asyncio
    async def test_distribute_report_success(self, reporting_config, mock_grafana_client):
        """Testet erfolgreiche Report-Verteilung."""
        with patch("services.reporting.scheduler.get_alert_dispatcher") as mock_dispatcher_func:
            mock_dispatcher = AsyncMock()
            mock_dispatcher_func.return_value = mock_dispatcher

            scheduler = ReportingScheduler(config=reporting_config, grafana_client=mock_grafana_client)

            await scheduler._distribute_report(["test@example.com"])

            mock_dispatcher.send_email.assert_called_once_with(
                subject=reporting_config.report.default_subject,
                body=reporting_config.report.default_body,
                severity=reporting_config.report.default_severity
            )

    @pytest.mark.asyncio
    async def test_distribute_report_no_recipients(self, reporting_config, mock_grafana_client):
        """Testet Report-Verteilung ohne Empfänger."""
        with patch("services.reporting.scheduler.get_alert_dispatcher") as mock_dispatcher_func:
            scheduler = ReportingScheduler(config=reporting_config, grafana_client=mock_grafana_client)

            await scheduler._distribute_report([])

            # Dispatcher sollte nicht aufgerufen werden
            mock_dispatcher_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_report_workflow(self, reporting_config, mock_grafana_client):
        """Testet den kompletten Report-Workflow."""
        with patch("services.reporting.scheduler.settings") as mock_settings, \
             patch("services.reporting.scheduler.get_alert_dispatcher") as mock_dispatcher_func:

            mock_settings.reporting_default_recipients = "test@example.com"
            mock_dispatcher = AsyncMock()
            mock_dispatcher_func.return_value = mock_dispatcher

            scheduler = ReportingScheduler(config=reporting_config, grafana_client=mock_grafana_client)

            await scheduler._generate_and_distribute_reports()

            # Verify Grafana call
            mock_grafana_client.export_panel_png.assert_called_once()

            # Verify email sending
            mock_dispatcher.send_email.assert_called_once()


class TestConfigurationIntegration:
    """Tests für Konfiguration-Integration."""

    def test_default_configuration(self):
        """Testet Standard-Konfiguration."""
        from services.reporting.config import get_reporting_config

        config = get_reporting_config()

        assert config.grafana.panel_timeout_seconds == 30.0
        assert config.scheduler.default_interval_minutes == 60
        assert config.report.default_dashboard_uid == "keiko-overview"

    def test_custom_configuration(self):
        """Testet benutzerdefinierte Konfiguration."""
        from services.reporting.config import get_reporting_config, set_reporting_config

        custom_config = ReportingServiceConfig()
        custom_config.grafana.panel_timeout_seconds = 45.0

        set_reporting_config(custom_config)
        retrieved_config = get_reporting_config()

        assert retrieved_config.grafana.panel_timeout_seconds == 45.0
