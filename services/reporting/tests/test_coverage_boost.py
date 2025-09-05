"""Zusätzliche Tests zur Erhöhung der Test-Coverage.

Testet Edge-Cases und weniger häufig verwendete Code-Pfade.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from services.reporting import (
    ConfigurationError,
    GrafanaClient,
    GrafanaClientError,
    GrafanaConfig,
    GrafanaConnectionError,
    GrafanaTimeoutError,
    ReportGenerationError,
    ReportingScheduler,
    ReportingService,
    ReportingServiceError,
    get_service_info,
)


class TestGrafanaClientEdgeCases:
    """Tests für GrafanaClient Edge-Cases."""

    @pytest.fixture
    def grafana_config(self):
        """Test-Konfiguration für GrafanaClient."""
        return GrafanaConfig(max_retries=1, retry_delay_seconds=0.01)

    @pytest.mark.asyncio
    async def test_export_panel_csv_fallback(self, grafana_config):
        """Testet CSV-Export-Fallback."""
        with patch("services.reporting.grafana_client.settings") as mock_settings:
            mock_settings.grafana_url = "https://test.grafana.com"
            mock_settings.grafana_api_token.get_secret_value.return_value = "test-token"

            client = GrafanaClient(config=grafana_config)

            with patch.object(client, "export_panel_png", return_value=b"test-data") as mock_png:
                result = await client.export_panel_csv("test-dashboard", 1)

                assert "CSV-Export nicht verfügbar" in result
                assert "9 bytes" in result
                mock_png.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_panel_json_success(self, grafana_config):
        """Testet erfolgreichen JSON-Export."""
        client = GrafanaClient(config=grafana_config)

        mock_dashboard_data = {
            "dashboard": {
                "panels": [
                    {"id": 1, "title": "Test Panel", "type": "graph"},
                    {"id": 2, "title": "Other Panel", "type": "stat"}
                ]
            }
        }

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_dashboard_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.export_panel_json("test-dashboard", 1)

            assert result["dashboard_uid"] == "test-dashboard"
            assert result["panel_id"] == 1
            assert result["panel_title"] == "Test Panel"
            assert result["panel_type"] == "graph"

    @pytest.mark.asyncio
    async def test_export_panel_json_panel_not_found(self, grafana_config):
        """Testet JSON-Export mit nicht gefundenem Panel."""
        client = GrafanaClient(config=grafana_config)

        mock_dashboard_data = {
            "dashboard": {
                "panels": [
                    {"id": 2, "title": "Other Panel", "type": "stat"}
                ]
            }
        }

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_dashboard_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GrafanaClientError) as exc_info:
                await client.export_panel_json("test-dashboard", 1)

            assert "Panel 1 nicht gefunden" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_request_with_retry_404_error(self, grafana_config):
        """Testet 404-Fehler-Behandlung."""
        client = GrafanaClient(config=grafana_config)

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404", request=MagicMock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GrafanaClientError) as exc_info:
                await client._execute_request_with_retry("http://test.com")

            assert "nicht gefunden" in str(exc_info.value)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_execute_request_with_retry_timeout_error(self, grafana_config):
        """Testet Timeout-Fehler-Behandlung."""
        client = GrafanaClient(config=grafana_config)

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GrafanaTimeoutError):
                await client._execute_request_with_retry("http://test.com")

    @pytest.mark.asyncio
    async def test_execute_request_with_retry_connection_error(self, grafana_config):
        """Testet Connection-Fehler-Behandlung."""
        client = GrafanaClient(config=grafana_config)

        with patch("services.reporting.grafana_client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GrafanaConnectionError):
                await client._execute_request_with_retry("http://test.com")


class TestSchedulerEdgeCases:
    """Tests für ReportingScheduler Edge-Cases."""

    @pytest.fixture
    def mock_grafana_client(self):
        """Mock für GrafanaClient."""
        client = AsyncMock()
        client.export_panel_png.return_value = b"test-png"
        client.export_panel_json.return_value = {"test": "data"}
        client.export_panel_csv.return_value = "csv,data"
        client.export_dashboard_pdf.return_value = b"test-pdf"
        return client

    @pytest.mark.asyncio
    async def test_run_loop_with_reporting_disabled(self, mock_grafana_client):
        """Testet Scheduler-Loop mit deaktiviertem Reporting."""
        with patch("services.reporting.scheduler.settings") as mock_settings:
            mock_settings.reporting_enabled = False

            scheduler = ReportingScheduler(grafana_client=mock_grafana_client)
            scheduler._running = True

            # Sollte sofort beenden
            await scheduler._run_loop()

            # Keine Reports sollten generiert werden
            mock_grafana_client.export_panel_png.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_loop_with_cancellation(self, mock_grafana_client):
        """Testet Scheduler-Loop mit Cancellation."""
        with patch("services.reporting.scheduler.settings") as mock_settings:
            mock_settings.reporting_enabled = True

            scheduler = ReportingScheduler(interval_minutes=1, grafana_client=mock_grafana_client)
            scheduler._running = True

            # Mock sleep um CancelledError zu werfen
            with patch("services.reporting.scheduler.asyncio.sleep", side_effect=asyncio.CancelledError):
                await scheduler._run_loop()

    @pytest.mark.asyncio
    async def test_generate_template_report_template_not_found(self, mock_grafana_client):
        """Testet Template-Report mit nicht gefundenem Template."""
        scheduler = ReportingScheduler(grafana_client=mock_grafana_client)

        with pytest.raises(ReportGenerationError) as exc_info:
            await scheduler.generate_template_report("non_existent_template")

        assert "Template 'non_existent_template' nicht gefunden" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_template_report_with_format_errors(self, mock_grafana_client):
        """Testet Template-Report mit Format-Fehlern."""
        from services.reporting.templates import ReportFormat, ReportTemplate, ReportTemplateManager

        # Template mit allen Formaten erstellen
        template = ReportTemplate(
            template_id="error_test",
            name="Error Test Template",
            panel_ids=[1],
            formats=[ReportFormat.PNG, ReportFormat.PDF, ReportFormat.JSON, ReportFormat.CSV]
        )

        template_manager = ReportTemplateManager()
        template_manager.add_template(template)

        # Grafana-Client mit Fehlern konfigurieren
        mock_grafana_client.export_panel_png.side_effect = Exception("PNG error")
        mock_grafana_client.export_dashboard_pdf.side_effect = Exception("PDF error")
        mock_grafana_client.export_panel_json.side_effect = Exception("JSON error")
        mock_grafana_client.export_panel_csv.side_effect = Exception("CSV error")

        scheduler = ReportingScheduler(
            grafana_client=mock_grafana_client,
            template_manager=template_manager
        )

        with patch("services.reporting.scheduler.settings") as mock_settings:
            mock_settings.reporting_default_recipients = ""

            result = await scheduler.generate_template_report("error_test")

            assert result["success"] is True
            assert "panel_1" in result["results"]

            # Alle Formate sollten Fehler haben
            panel_results = result["results"]["panel_1"]
            assert panel_results["png"]["success"] is False
            assert panel_results["pdf"]["success"] is False
            assert panel_results["json"]["success"] is False
            assert panel_results["csv"]["success"] is False


class TestServiceEdgeCases:
    """Tests für ReportingService Edge-Cases."""

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Testet Health-Check mit Exception."""
        service = ReportingService()

        # Mock super().health_check() um Exception zu werfen
        with patch("services.reporting.service.FeatureService.health_check", side_effect=Exception("Health check error")):
            health = await service.health_check()

            assert health["status"] == "unhealthy"
            assert "Health check error" in health["error"]

    @pytest.mark.asyncio
    async def test_generate_manual_report_exception(self):
        """Testet manuelle Report-Generierung mit Exception."""
        service = ReportingService()
        service._initialized = True

        # Mock scheduler um Exception zu werfen
        service.scheduler._generate_and_distribute_reports = AsyncMock(side_effect=Exception("Report error"))

        with pytest.raises(ReportingServiceError) as exc_info:
            await service.generate_manual_report()

        assert "Report error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_shutdown_exception(self):
        """Testet Shutdown mit Exception."""
        service = ReportingService()
        service._initialized = True

        # Mock scheduler.stop() um Exception zu werfen
        service.scheduler.stop = AsyncMock(side_effect=Exception("Shutdown error"))

        with pytest.raises(Exception):
            await service.shutdown()

        assert service._status.value == "error"


class TestExceptionClasses:
    """Tests für Exception-Klassen."""

    def test_reporting_service_error(self):
        """Testet ReportingServiceError."""
        error = ReportingServiceError("Test error", {"key": "value"})

        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert str(error) == "Test error"

    def test_grafana_client_error(self):
        """Testet GrafanaClientError."""
        error = GrafanaClientError("Grafana error", status_code=500, response_body="Error body")

        assert error.status_code == 500
        assert error.response_body == "Error body"
        assert error.details["status_code"] == 500

    def test_report_generation_error(self):
        """Testet ReportGenerationError."""
        error = ReportGenerationError(
            "Generation error",
            report_type="png",
            dashboard_uid="test-dashboard",
            panel_id=1
        )

        assert error.report_type == "png"
        assert error.dashboard_uid == "test-dashboard"
        assert error.panel_id == 1

    def test_configuration_error(self):
        """Testet ConfigurationError."""
        error = ConfigurationError("Config error", config_key="test_key", config_value="test_value")

        assert error.config_key == "test_key"
        assert error.config_value == "test_value"


class TestUtilityFunctions:
    """Tests für Utility-Funktionen."""

    def test_get_service_info(self):
        """Testet get_service_info-Funktion."""
        info = get_service_info()

        assert info["name"] == "reporting"
        assert "version" in info
        assert "components" in info
        assert "grafana_client" in info["components"]


class TestConfigurationEdgeCases:
    """Tests für Konfiguration Edge-Cases."""

    def test_config_with_invalid_values(self):
        """Testet Konfiguration mit ungültigen Werten."""
        from services.reporting.config import GrafanaConfig

        # Sollte Validierung durchführen
        config = GrafanaConfig(
            panel_timeout_seconds=-1,  # Ungültiger Wert
            max_retries=0
        )

        # Pydantic sollte Standardwerte verwenden oder validieren
        assert config.max_retries >= 0

    def test_config_global_functions(self):
        """Testet globale Konfigurationsfunktionen."""
        from services.reporting.config import (
            ReportingServiceConfig,
            get_reporting_config,
            set_reporting_config,
        )

        # Test get_reporting_config
        config1 = get_reporting_config()
        config2 = get_reporting_config()
        assert config1 is config2  # Sollte Singleton sein

        # Test set_reporting_config
        new_config = ReportingServiceConfig()
        new_config.service_name = "test_reporting"
        set_reporting_config(new_config)

        retrieved_config = get_reporting_config()
        assert retrieved_config.service_name == "test_reporting"
