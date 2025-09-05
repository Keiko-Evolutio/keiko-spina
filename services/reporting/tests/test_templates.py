"""Tests für das Template-System.

Testet Template-Erstellung, -Verwaltung und -basierte Report-Generierung.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from services.reporting.exceptions import ReportingServiceError
from services.reporting.templates import (
    ReportFormat,
    ReportTemplate,
    ReportTemplateManager,
    ReportType,
)


class TestReportTemplate:
    """Tests für ReportTemplate."""

    def test_template_creation(self):
        """Testet Template-Erstellung."""
        template = ReportTemplate(
            template_id="test_template",
            name="Test Template",
            description="Test-Beschreibung",
            dashboard_uid="test-dashboard",
            panel_ids=[1, 2, 3],
            formats=[ReportFormat.PNG, ReportFormat.JSON]
        )

        assert template.template_id == "test_template"
        assert template.name == "Test Template"
        assert template.dashboard_uid == "test-dashboard"
        assert template.panel_ids == [1, 2, 3]
        assert ReportFormat.PNG in template.formats
        assert ReportFormat.JSON in template.formats

    def test_template_to_dict(self):
        """Testet Template-zu-Dictionary-Konvertierung."""
        template = ReportTemplate(
            template_id="test_template",
            name="Test Template",
            report_type=ReportType.KPI_OVERVIEW,
            formats=[ReportFormat.PNG]
        )

        data = template.to_dict()

        assert data["template_id"] == "test_template"
        assert data["name"] == "Test Template"
        assert data["report_type"] == "kpi_overview"
        assert data["formats"] == ["png"]

    def test_template_from_dict(self):
        """Testet Template-aus-Dictionary-Erstellung."""
        data = {
            "template_id": "test_template",
            "name": "Test Template",
            "report_type": "performance_dashboard",
            "formats": ["png", "pdf"],
            "panel_ids": [1, 2]
        }

        template = ReportTemplate.from_dict(data)

        assert template.template_id == "test_template"
        assert template.name == "Test Template"
        assert template.report_type == ReportType.PERFORMANCE_DASHBOARD
        assert ReportFormat.PNG in template.formats
        assert ReportFormat.PDF in template.formats
        assert template.panel_ids == [1, 2]


class TestReportTemplateManager:
    """Tests für ReportTemplateManager."""

    @pytest.fixture
    def template_manager(self):
        """Template-Manager für Tests."""
        return ReportTemplateManager()

    @pytest.fixture
    def sample_template(self):
        """Beispiel-Template für Tests."""
        return ReportTemplate(
            template_id="sample_template",
            name="Sample Template",
            description="Test-Template",
            formats=[ReportFormat.PNG, ReportFormat.JSON]
        )

    def test_manager_initialization(self, template_manager):
        """Testet Manager-Initialisierung."""
        # Sollte Standard-Templates laden
        templates = template_manager.list_templates()
        assert len(templates) >= 3  # Mindestens die 3 Standard-Templates

        # Prüfe Standard-Templates
        template_ids = [t.template_id for t in templates]
        assert "kpi_overview" in template_ids
        assert "performance_dashboard" in template_ids
        assert "system_health" in template_ids

    def test_get_template(self, template_manager):
        """Testet Template-Abfrage."""
        # Existierendes Template
        template = template_manager.get_template("kpi_overview")
        assert template is not None
        assert template.template_id == "kpi_overview"

        # Nicht-existierendes Template
        template = template_manager.get_template("non_existent")
        assert template is None

    def test_add_template(self, template_manager, sample_template):
        """Testet Template-Hinzufügung."""
        initial_count = len(template_manager.list_templates())

        template_manager.add_template(sample_template)

        assert len(template_manager.list_templates()) == initial_count + 1
        retrieved = template_manager.get_template("sample_template")
        assert retrieved is not None
        assert retrieved.name == "Sample Template"

    def test_add_duplicate_template(self, template_manager, sample_template):
        """Testet Hinzufügung doppelter Template-ID."""
        template_manager.add_template(sample_template)

        with pytest.raises(ReportingServiceError):
            template_manager.add_template(sample_template)

    def test_update_template(self, template_manager, sample_template):
        """Testet Template-Aktualisierung."""
        template_manager.add_template(sample_template)

        # Template modifizieren
        sample_template.name = "Updated Template"
        template_manager.update_template(sample_template)

        retrieved = template_manager.get_template("sample_template")
        assert retrieved.name == "Updated Template"

    def test_update_nonexistent_template(self, template_manager, sample_template):
        """Testet Aktualisierung nicht-existierender Template."""
        with pytest.raises(ReportingServiceError):
            template_manager.update_template(sample_template)

    def test_remove_template(self, template_manager, sample_template):
        """Testet Template-Entfernung."""
        template_manager.add_template(sample_template)
        initial_count = len(template_manager.list_templates())

        result = template_manager.remove_template("sample_template")

        assert result is True
        assert len(template_manager.list_templates()) == initial_count - 1
        assert template_manager.get_template("sample_template") is None

    def test_remove_nonexistent_template(self, template_manager):
        """Testet Entfernung nicht-existierender Template."""
        result = template_manager.remove_template("non_existent")
        assert result is False

    def test_get_templates_by_type(self, template_manager):
        """Testet Template-Abfrage nach Typ."""
        kpi_templates = template_manager.get_templates_by_type(ReportType.KPI_OVERVIEW)

        assert len(kpi_templates) >= 1
        for template in kpi_templates:
            assert template.report_type == ReportType.KPI_OVERVIEW

    def test_export_templates(self, template_manager, sample_template):
        """Testet Template-Export."""
        template_manager.add_template(sample_template)

        json_data = template_manager.export_templates()

        # Sollte gültiges JSON sein
        parsed_data = json.loads(json_data)
        assert isinstance(parsed_data, dict)
        assert "sample_template" in parsed_data

    def test_import_templates(self, template_manager):
        """Testet Template-Import."""
        # Test-Template-Daten
        test_data = {
            "imported_template": {
                "template_id": "imported_template",
                "name": "Imported Template",
                "description": "Importiertes Test-Template",
                "report_type": "custom",
                "formats": ["png"]
            }
        }

        json_data = json.dumps(test_data)
        imported_count = template_manager.import_templates(json_data)

        assert imported_count == 1
        imported_template = template_manager.get_template("imported_template")
        assert imported_template is not None
        assert imported_template.name == "Imported Template"

    def test_import_invalid_json(self, template_manager):
        """Testet Import mit ungültigem JSON."""
        with pytest.raises(ReportingServiceError):
            template_manager.import_templates("invalid json")


class TestTemplateIntegration:
    """Integration-Tests für Template-System."""

    @pytest.fixture
    def mock_grafana_client(self):
        """Mock für GrafanaClient."""
        client = AsyncMock()
        client.export_panel_png.return_value = b"fake-png-data"
        client.export_panel_json.return_value = {"panel": "data"}
        client.export_panel_csv.return_value = "csv,data"
        client.export_dashboard_pdf.return_value = b"fake-pdf-data"
        return client

    @pytest.fixture
    def template_manager_with_test_template(self):
        """Template-Manager mit Test-Template."""
        manager = ReportTemplateManager()
        test_template = ReportTemplate(
            template_id="test_integration",
            name="Integration Test Template",
            dashboard_uid="test-dashboard",
            panel_ids=[1, 2],
            formats=[ReportFormat.PNG, ReportFormat.JSON],
            email_subject="Test Report",
            email_body="Test report body"
        )
        manager.add_template(test_template)
        return manager

    @pytest.mark.asyncio
    async def test_template_based_report_generation(self, mock_grafana_client, template_manager_with_test_template):
        """Testet Template-basierte Report-Generierung."""
        from services.reporting.scheduler import ReportingScheduler

        with patch("services.reporting.scheduler.settings") as mock_settings, \
             patch("services.reporting.scheduler.get_alert_dispatcher") as mock_dispatcher_func:

            mock_settings.reporting_default_recipients = "test@example.com"
            mock_dispatcher = AsyncMock()
            mock_dispatcher_func.return_value = mock_dispatcher

            scheduler = ReportingScheduler(
                grafana_client=mock_grafana_client,
                template_manager=template_manager_with_test_template
            )

            result = await scheduler.generate_template_report("test_integration")

            assert result["success"] is True
            assert result["template_id"] == "test_integration"
            assert result["panels_processed"] == 2
            assert result["formats_processed"] == 2

            # Verify Grafana calls
            assert mock_grafana_client.export_panel_png.call_count == 2  # 2 panels
            assert mock_grafana_client.export_panel_json.call_count == 2  # 2 panels

            # Verify email sending
            mock_dispatcher.send_email.assert_called_once()

    def test_available_templates_listing(self, template_manager_with_test_template):
        """Testet Auflistung verfügbarer Templates."""
        from services.reporting.scheduler import ReportingScheduler

        scheduler = ReportingScheduler(template_manager=template_manager_with_test_template)
        templates = scheduler.get_available_templates()

        # Sollte Standard-Templates + Test-Template enthalten
        assert len(templates) >= 4

        # Prüfe Test-Template
        test_template = next((t for t in templates if t["template_id"] == "test_integration"), None)
        assert test_template is not None
        assert test_template["name"] == "Integration Test Template"
        assert test_template["panel_count"] == 2
        assert "png" in test_template["formats"]
        assert "json" in test_template["formats"]
