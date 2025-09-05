"""Baseline-Tests für das Reporting-Modul.

Testet die aktuelle Funktionalität um Regressionen zu vermeiden.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.reporting.grafana_client import GrafanaClient
from services.reporting.scheduler import ReportingScheduler


class TestGrafanaClientBaseline:
    """Baseline-Tests für GrafanaClient."""

    @pytest.mark.asyncio
    async def test_grafana_client_initialization(self, mock_settings):
        """Testet die Initialisierung des GrafanaClient."""
        with patch("services.reporting.grafana_client.settings", mock_settings):
            client = GrafanaClient()

            assert client.base_url == "https://grafana.example.com"
            assert client.api_token == "test-token"

    @pytest.mark.asyncio
    async def test_grafana_client_custom_params(self):
        """Testet GrafanaClient mit benutzerdefinierten Parametern."""
        client = GrafanaClient(
            base_url="https://custom.grafana.com",
            api_token="custom-token"
        )

        assert client.base_url == "https://custom.grafana.com"
        assert client.api_token == "custom-token"

    def test_headers_generation(self, mock_settings):
        """Testet die Header-Generierung."""
        with patch("services.reporting.grafana_client.settings", mock_settings):
            client = GrafanaClient()
            headers = client._get_headers()

            assert headers["Content-Type"] == "application/json"
            assert headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_export_panel_png_success(self, mock_settings, mock_httpx_client):
        """Testet erfolgreichen PNG-Export."""
        with patch("services.reporting.grafana_client.settings", mock_settings), \
             patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:

            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            client = GrafanaClient()
            result = await client.export_panel_png("test-dashboard", 1)

            assert result == b"fake-png-content"
            mock_httpx_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_dashboard_pdf_success(self, mock_settings, mock_httpx_client):
        """Testet erfolgreichen PDF-Export."""
        with patch("services.reporting.grafana_client.settings", mock_settings), \
             patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:

            mock_client_class.return_value.__aenter__.return_value = mock_httpx_client

            client = GrafanaClient()
            result = await client.export_dashboard_pdf("test-dashboard")

            assert result == b"fake-png-content"
            mock_httpx_client.get.assert_called_once()


class TestReportingSchedulerBaseline:
    """Baseline-Tests für ReportingScheduler."""

    def test_scheduler_initialization_default(self, mock_settings):
        """Testet Standard-Initialisierung des Schedulers."""
        with patch("services.reporting.scheduler.settings", mock_settings):
            scheduler = ReportingScheduler()

            assert scheduler.interval_minutes == 60
            assert scheduler._task is None
            assert scheduler._running is False

    def test_scheduler_initialization_custom(self):
        """Testet Initialisierung mit benutzerdefinierten Parametern."""
        scheduler = ReportingScheduler(interval_minutes=30)

        assert scheduler.interval_minutes == 30
        assert scheduler._task is None
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scheduler_start_stop_lifecycle(self, mock_settings):
        """Testet Start/Stop-Lifecycle des Schedulers."""
        with patch("services.reporting.scheduler.settings", mock_settings):
            scheduler = ReportingScheduler()

            # Test Start
            await scheduler.start()
            assert scheduler._running is True
            assert scheduler._task is not None

            # Test Stop - CancelledError ist erwartet
            try:
                await scheduler.stop()
            except asyncio.CancelledError:
                pass  # Erwartet bei Task-Cancellation

            assert scheduler._running is False
            # Task wird auf None gesetzt nach dem Stop
            assert scheduler._task is None or scheduler._task.cancelled()

    @pytest.mark.asyncio
    async def test_scheduler_double_start(self, mock_settings):
        """Testet dass doppelter Start ignoriert wird."""
        with patch("services.reporting.scheduler.settings", mock_settings):
            scheduler = ReportingScheduler()

            await scheduler.start()
            first_task = scheduler._task

            await scheduler.start()  # Sollte ignoriert werden
            assert scheduler._task is first_task

            # Test Stop - CancelledError ist erwartet
            try:
                await scheduler.stop()
            except asyncio.CancelledError:
                pass  # Erwartet bei Task-Cancellation

    @pytest.mark.asyncio
    async def test_generate_and_distribute_reports_success(self, mock_settings, mock_alert_dispatcher):
        """Testet erfolgreiche Report-Generierung und -Versand."""
        with patch("services.reporting.scheduler.settings", mock_settings), \
             patch("services.reporting.scheduler.GrafanaClient") as mock_grafana_class, \
             patch("services.reporting.scheduler.get_alert_dispatcher", return_value=mock_alert_dispatcher):

            # Mock GrafanaClient
            mock_grafana = AsyncMock()
            mock_grafana.export_panel_png.return_value = b"fake-png"
            mock_grafana_class.return_value = mock_grafana

            scheduler = ReportingScheduler()
            await scheduler._generate_and_distribute_reports()

            # Verify Grafana call
            mock_grafana.export_panel_png.assert_called_once_with(
                dashboard_uid="keiko-overview",
                panel_id=1,
                params={"width": 1600, "height": 900}
            )

            # Verify email sending
            mock_alert_dispatcher.send_email.assert_called_once_with(
                subject="Keiko KPI Report",
                body="Automatischer KPI-Report. Siehe Grafana-Dashboard: Keiko Overview.",
                severity="info"
            )

    @pytest.mark.asyncio
    async def test_generate_and_distribute_reports_grafana_error(self, mock_settings, mock_alert_dispatcher):
        """Testet Report-Generierung bei Grafana-Fehler."""
        with patch("services.reporting.scheduler.settings", mock_settings), \
             patch("services.reporting.scheduler.get_alert_dispatcher", return_value=mock_alert_dispatcher):

            # Mock GrafanaClient mit Fehler
            mock_grafana = AsyncMock()
            mock_grafana.export_panel_png.side_effect = Exception("Grafana error")

            scheduler = ReportingScheduler(grafana_client=mock_grafana)

            # Sollte ReportGenerationError werfen
            with pytest.raises(Exception):  # ReportGenerationError oder andere Exception
                await scheduler._generate_and_distribute_reports()

    @pytest.mark.asyncio
    async def test_generate_and_distribute_reports_no_recipients(self, mock_settings):
        """Testet Report-Generierung ohne Empfänger."""
        mock_settings.reporting_default_recipients = ""

        with patch("services.reporting.scheduler.settings", mock_settings), \
             patch("services.reporting.scheduler.GrafanaClient") as mock_grafana_class, \
             patch("services.reporting.scheduler.get_alert_dispatcher") as mock_dispatcher:

            mock_grafana = AsyncMock()
            mock_grafana.export_panel_png.return_value = b"fake-png"
            mock_grafana_class.return_value = mock_grafana

            scheduler = ReportingScheduler()
            await scheduler._generate_and_distribute_reports()

            # Kein E-Mail-Versand bei fehlenden Empfängern
            mock_dispatcher.assert_not_called()


class TestReportingIntegrationBaseline:
    """Baseline-Integration-Tests."""

    @pytest.mark.asyncio
    async def test_full_reporting_workflow(self, mock_settings, mock_alert_dispatcher):
        """Testet den kompletten Reporting-Workflow."""
        with patch("services.reporting.scheduler.settings", mock_settings), \
             patch("services.reporting.grafana_client.settings", mock_settings), \
             patch("services.reporting.scheduler.get_alert_dispatcher", return_value=mock_alert_dispatcher), \
             patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:

            # Mock HTTP-Client
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"test-png-data"
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Test kompletter Workflow
            scheduler = ReportingScheduler()
            await scheduler._generate_and_distribute_reports()

            # Verify HTTP-Request
            mock_client.get.assert_called_once()

            # Verify E-Mail-Versand
            mock_alert_dispatcher.send_email.assert_called_once()

            call_args = mock_alert_dispatcher.send_email.call_args
            assert call_args[1]["subject"] == "Keiko KPI Report"
            assert call_args[1]["severity"] == "info"
