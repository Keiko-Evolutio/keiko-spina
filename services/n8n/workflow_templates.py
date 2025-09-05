"""Vorgefertigte n8n-Workflow-Templates und dynamische Instanziierung."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from kei_logging import get_logger
from observability import trace_function

if TYPE_CHECKING:
    import builtins

logger = get_logger(__name__)


@dataclass(slots=True)
class TemplateParameter:
    """Definiert einen Template-Parameter mit Metadaten."""

    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Any | None = None


@dataclass(slots=True)
class WorkflowTemplate:
    """Repräsentiert ein n8n-Workflow-Template."""

    template_id: str
    display_name: str
    category: str
    description: str
    parameters: list[TemplateParameter]
    workflow_spec: dict[str, Any]


class TemplateRenderer(Protocol):
    """Interface für Template-Rendering."""

    def render(self, template: WorkflowTemplate, values: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - Protokoll
        ...


class DefaultTemplateRenderer:
    """Einfache Renderer-Implementierung, die Parameter substituiert."""

    def render(self, template: WorkflowTemplate, values: dict[str, Any]) -> dict[str, Any]:
        # Einfache Validierung und Substitution
        rendered = {**template.workflow_spec}
        context = {}
        for param in template.parameters:
            if param.required and param.name not in values and param.default is None:
                raise ValueError(f"Missing required parameter: {param.name}")
            context[param.name] = values.get(param.name, param.default)

        # Platzhalterersetzung in bekannten Feldern (vereinfachte Variante)
        def substitute(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [substitute(x) for x in obj]
            if isinstance(obj, str):
                try:
                    return obj.format(**context)
                except Exception:
                    return obj
            return obj

        return substitute(rendered)


class WorkflowTemplateRegistry:
    """In-Memory Registry für vordefinierte Business-Templates."""

    def __init__(self) -> None:
        self._templates: dict[str, WorkflowTemplate] = {}
        self._renderer: TemplateRenderer = DefaultTemplateRenderer()
        self._register_defaults()
        # Erweiterungen dynamisch laden (optional)
        try:  # pragma: no cover - optionales Modul
            from .workflow_templates_ext import register_extended_templates

            register_extended_templates(self)
        except Exception:  # pragma: no cover - bei Fehlen/Fehler still
            logger.debug("Keine erweiterten n8n-Templates geladen")

    def list(self) -> builtins.list[WorkflowTemplate]:
        """Listet alle Templates."""
        return list(self._templates.values())

    def get(self, template_id: str) -> WorkflowTemplate | None:
        """Liefert Template oder None."""
        return self._templates.get(template_id)

    @trace_function("n8n.templates.render")
    def render(self, template_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Rendert ein Template zu einer n8n-Workflow-Spezifikation."""
        tpl = self.get(template_id)
        if not tpl:
            raise KeyError(f"Unknown template: {template_id}")
        return self._renderer.render(tpl, values)

    # ------------------------------------------------------------------
    # Default Templates
    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        """Registriert vorgefertigte Templates für Geschäftsprozesse."""
        # 1) E-Mail-Benachrichtigungen / Newsletter
        email_tpl = WorkflowTemplate(
            template_id="email_notification",
            display_name="E-Mail Benachrichtigung",
            category="communication",
            description="Sendet eine E-Mail via SMTP/Provider",
            parameters=[
                TemplateParameter("to", "string", True, "Empfänger E-Mail"),
                TemplateParameter("subject", "string", True, "Betreff"),
                TemplateParameter("body", "string", True, "Nachrichtentext"),
            ],
            workflow_spec={
                "name": "Email Notification",
                "nodes": [
                    {
                        "type": "emailSend",
                        "parameters": {
                            "to": "{to}",
                            "subject": "{subject}",
                            "text": "{body}",
                        },
                    }
                ],
            },
        )

        # 2) CRM-Integration (Kontakt-Sync / Lead-Management)
        crm_tpl = WorkflowTemplate(
            template_id="crm_contact_sync",
            display_name="CRM Kontakt Sync",
            category="crm",
            description="Synchronisiert Kontaktinformationen in ein CRM-System",
            parameters=[
                TemplateParameter("contact_email", "string", True, "Kontakt E-Mail"),
                TemplateParameter("first_name", "string", True, "Vorname"),
                TemplateParameter("last_name", "string", True, "Nachname"),
            ],
            workflow_spec={
                "name": "CRM Contact Sync",
                "nodes": [
                    {
                        "type": "crmUpsert",
                        "parameters": {
                            "email": "{contact_email}",
                            "firstName": "{first_name}",
                            "lastName": "{last_name}",
                        },
                    }
                ],
            },
        )

        # 3) Dokumenten-Verarbeitung (PDF-Generation, File-Upload)
        doc_tpl = WorkflowTemplate(
            template_id="document_processing",
            display_name="Dokumenten Verarbeitung",
            category="documents",
            description="Generiert PDF und lädt Datei in Storage hoch",
            parameters=[
                TemplateParameter("title", "string", True, "Dokumenttitel"),
                TemplateParameter("content", "string", True, "Dokumentinhalt"),
                TemplateParameter("target_path", "string", True, "Zielpfad im Storage"),
            ],
            workflow_spec={
                "name": "Document Processing",
                "nodes": [
                    {"type": "pdfGenerate", "parameters": {"title": "{title}", "content": "{content}"}},
                    {"type": "fileUpload", "parameters": {"path": "{target_path}"}},
                ],
            },
        )

        # 4) Social Media Automation
        social_tpl = WorkflowTemplate(
            template_id="social_post",
            display_name="Social Media Post",
            category="social",
            description="Postet Text auf LinkedIn/Twitter",
            parameters=[
                TemplateParameter("platform", "string", True, "Zielplattform (linkedin|twitter)", default="linkedin"),
                TemplateParameter("text", "string", True, "Beitragstext"),
            ],
            workflow_spec={
                "name": "Social Post",
                "nodes": [
                    {"type": "socialPost", "parameters": {"platform": "{platform}", "text": "{text}"}},
                ],
            },
        )

        # 5) Kalender-Integration (Termin/Reminder)
        calendar_tpl = WorkflowTemplate(
            template_id="calendar_schedule",
            display_name="Kalender Termin",
            category="calendar",
            description="Erstellt Termin und Reminder",
            parameters=[
                TemplateParameter("title", "string", True, "Termin Titel"),
                TemplateParameter("start_iso", "string", True, "Startzeit ISO"),
                TemplateParameter("duration_min", "integer", True, "Dauer in Minuten"),
            ],
            workflow_spec={
                "name": "Calendar Schedule",
                "nodes": [
                    {
                        "type": "calendarCreate",
                        "parameters": {"title": "{title}", "start": "{start_iso}", "duration": "{duration_min}"},
                    },
                    {"type": "calendarReminder", "parameters": {"offset": 15}},
                ],
            },
        )

        for tpl in (email_tpl, crm_tpl, doc_tpl, social_tpl, calendar_tpl):
            self._templates[tpl.template_id] = tpl


# Singleton-Registry
workflow_template_registry = WorkflowTemplateRegistry()


__all__ = [
    "TemplateParameter",
    "WorkflowTemplate",
    "WorkflowTemplateRegistry",
    "workflow_template_registry",
]
