"""Template-System für Report-Generierung.

Konfigurierbare Templates für verschiedene Report-Typen mit
flexiblen Parametern und Formatierungsoptionen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger

from .exceptions import ReportingServiceError

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Unterstützte Report-Formate."""
    PNG = "png"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class ReportType(Enum):
    """Vordefinierte Report-Typen."""
    KPI_OVERVIEW = "kpi_overview"
    PERFORMANCE_DASHBOARD = "performance_dashboard"
    SYSTEM_HEALTH = "system_health"
    CUSTOM = "custom"


@dataclass
class ReportTemplate:
    """Template für Report-Konfiguration.

    Definiert alle Parameter für die Report-Generierung
    einschließlich Dashboard-Konfiguration, Export-Parameter
    und Verteilungseinstellungen.
    """

    # Template-Identifikation
    template_id: str
    name: str
    description: str = ""
    report_type: ReportType = ReportType.CUSTOM

    # Dashboard-Konfiguration
    dashboard_uid: str = "keiko-overview"
    panel_ids: list[int] = field(default_factory=lambda: [1])

    # Export-Konfiguration
    formats: list[ReportFormat] = field(default_factory=lambda: [ReportFormat.PNG])
    export_params: dict[str, Any] = field(default_factory=dict)

    # E-Mail-Konfiguration
    email_subject: str = "Keiko Report"
    email_body: str = "Automatisch generierter Report."
    email_severity: str = "info"

    # Zeitplan-Konfiguration
    schedule_enabled: bool = True
    interval_minutes: int | None = None

    # Metadaten
    created_by: str = "system"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Template zu Dictionary.

        Returns:
            Template als Dictionary
        """
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "report_type": self.report_type.value,
            "dashboard_uid": self.dashboard_uid,
            "panel_ids": self.panel_ids,
            "formats": [f.value for f in self.formats],
            "export_params": self.export_params,
            "email_subject": self.email_subject,
            "email_body": self.email_body,
            "email_severity": self.email_severity,
            "schedule_enabled": self.schedule_enabled,
            "interval_minutes": self.interval_minutes,
            "created_by": self.created_by,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportTemplate:
        """Erstellt Template aus Dictionary.

        Args:
            data: Template-Daten als Dictionary

        Returns:
            ReportTemplate-Instanz
        """
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data.get("description", ""),
            report_type=ReportType(data.get("report_type", "custom")),
            dashboard_uid=data.get("dashboard_uid", "keiko-overview"),
            panel_ids=data.get("panel_ids", [1]),
            formats=[ReportFormat(f) for f in data.get("formats", ["png"])],
            export_params=data.get("export_params", {}),
            email_subject=data.get("email_subject", "Keiko Report"),
            email_body=data.get("email_body", "Automatisch generierter Report."),
            email_severity=data.get("email_severity", "info"),
            schedule_enabled=data.get("schedule_enabled", True),
            interval_minutes=data.get("interval_minutes"),
            created_by=data.get("created_by", "system"),
            tags=data.get("tags", [])
        )


class ReportTemplateManager:
    """Manager für Report-Templates.

    Verwaltet vordefinierte und benutzerdefinierte Templates
    für verschiedene Report-Typen.
    """

    def __init__(self) -> None:
        """Initialisiert den Template-Manager."""
        self._templates: dict[str, ReportTemplate] = {}
        self._load_default_templates()

        logger.info(f"ReportTemplateManager initialisiert mit {len(self._templates)} Templates")

    def _load_default_templates(self) -> None:
        """Lädt vordefinierte Standard-Templates."""
        # KPI-Overview Template
        kpi_template = ReportTemplate(
            template_id="kpi_overview",
            name="KPI Overview Report",
            description="Umfassender KPI-Überblick mit wichtigsten Metriken",
            report_type=ReportType.KPI_OVERVIEW,
            dashboard_uid="keiko-overview",
            panel_ids=[1, 2, 3],
            formats=[ReportFormat.PNG, ReportFormat.JSON],
            export_params={"width": 1600, "height": 900, "theme": "light"},
            email_subject="Keiko KPI Overview Report",
            email_body="Wöchentlicher KPI-Überblick mit wichtigsten Systemmetriken.",
            tags=["kpi", "overview", "weekly"]
        )

        # Performance Dashboard Template
        performance_template = ReportTemplate(
            template_id="performance_dashboard",
            name="Performance Dashboard",
            description="Detaillierte Performance-Metriken und Trends",
            report_type=ReportType.PERFORMANCE_DASHBOARD,
            dashboard_uid="performance-metrics",
            panel_ids=[1, 4, 5, 6],
            formats=[ReportFormat.PNG, ReportFormat.PDF],
            export_params={"width": 1920, "height": 1080, "theme": "dark"},
            email_subject="Keiko Performance Report",
            email_body="Täglicher Performance-Report mit Trend-Analysen.",
            interval_minutes=1440,  # Täglich
            tags=["performance", "metrics", "daily"]
        )

        # System Health Template
        health_template = ReportTemplate(
            template_id="system_health",
            name="System Health Check",
            description="Systemgesundheit und Verfügbarkeits-Metriken",
            report_type=ReportType.SYSTEM_HEALTH,
            dashboard_uid="system-health",
            panel_ids=[1, 2],
            formats=[ReportFormat.PNG, ReportFormat.JSON],
            export_params={"width": 1200, "height": 800},
            email_subject="Keiko System Health Report",
            email_body="Stündlicher System-Health-Check mit Verfügbarkeits-Metriken.",
            interval_minutes=60,  # Stündlich
            tags=["health", "system", "hourly"]
        )

        # Templates registrieren
        self._templates[kpi_template.template_id] = kpi_template
        self._templates[performance_template.template_id] = performance_template
        self._templates[health_template.template_id] = health_template

    def get_template(self, template_id: str) -> ReportTemplate | None:
        """Gibt Template nach ID zurück.

        Args:
            template_id: Template-ID

        Returns:
            ReportTemplate oder None falls nicht gefunden
        """
        return self._templates.get(template_id)

    def list_templates(self) -> list[ReportTemplate]:
        """Gibt alle verfügbaren Templates zurück.

        Returns:
            Liste aller Templates
        """
        return list(self._templates.values())

    def add_template(self, template: ReportTemplate) -> None:
        """Fügt neues Template hinzu.

        Args:
            template: Hinzuzufügendes Template

        Raises:
            ReportingServiceError: Falls Template-ID bereits existiert
        """
        if template.template_id in self._templates:
            raise ReportingServiceError(f"Template '{template.template_id}' existiert bereits")

        self._templates[template.template_id] = template
        logger.info(f"Template '{template.template_id}' hinzugefügt")

    def update_template(self, template: ReportTemplate) -> None:
        """Aktualisiert existierendes Template.

        Args:
            template: Zu aktualisierendes Template

        Raises:
            ReportingServiceError: Falls Template nicht existiert
        """
        if template.template_id not in self._templates:
            raise ReportingServiceError(f"Template '{template.template_id}' nicht gefunden")

        self._templates[template.template_id] = template
        logger.info(f"Template '{template.template_id}' aktualisiert")

    def remove_template(self, template_id: str) -> bool:
        """Entfernt Template.

        Args:
            template_id: ID des zu entfernenden Templates

        Returns:
            True falls entfernt, False falls nicht gefunden
        """
        if template_id in self._templates:
            del self._templates[template_id]
            logger.info(f"Template '{template_id}' entfernt")
            return True
        return False

    def get_templates_by_type(self, report_type: ReportType) -> list[ReportTemplate]:
        """Gibt Templates nach Typ zurück.

        Args:
            report_type: Report-Typ

        Returns:
            Liste der Templates des angegebenen Typs
        """
        return [t for t in self._templates.values() if t.report_type == report_type]

    def export_templates(self) -> str:
        """Exportiert alle Templates als JSON.

        Returns:
            JSON-String mit allen Templates
        """
        templates_data = {tid: template.to_dict() for tid, template in self._templates.items()}
        return json.dumps(templates_data, indent=2, ensure_ascii=False)

    def import_templates(self, json_data: str) -> int:
        """Importiert Templates aus JSON.

        Args:
            json_data: JSON-String mit Template-Daten

        Returns:
            Anzahl der importierten Templates
        """
        try:
            templates_data = json.loads(json_data)
            imported_count = 0

            for template_id, template_data in templates_data.items():
                template = ReportTemplate.from_dict(template_data)
                self._templates[template_id] = template
                imported_count += 1

            logger.info(f"{imported_count} Templates importiert")
            return imported_count

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ReportingServiceError(f"Template-Import fehlgeschlagen: {e!s}") from e


__all__ = [
    "ReportFormat",
    "ReportTemplate",
    "ReportTemplateManager",
    "ReportType"
]
