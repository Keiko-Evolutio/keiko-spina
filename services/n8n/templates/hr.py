"""HR-Templates: Onboarding, Performance-Review, Offboarding."""

from __future__ import annotations

from services.n8n.workflow_templates import (
    TemplateParameter,
    WorkflowTemplate,
    WorkflowTemplateRegistry,
)

from . import register_templates


def register(registry: WorkflowTemplateRegistry) -> None:
    """Registriert HR-bezogene Templates."""
    hr_onboarding_tpl = WorkflowTemplate(
        template_id="hr_onboarding",
        display_name="HR Onboarding",
        category="hr",
        description="Erstellt Benutzer, Systeme-Zugänge und Begrüßungs-Mail",
        parameters=[
            TemplateParameter("employee_email", "string", True, "E-Mail"),
            TemplateParameter("employee_name", "string", True, "Name"),
            TemplateParameter("systems", "array", False, "Systeme", default=["git", "jira"]),
        ],
        workflow_spec={
            "name": "HR Onboarding",
            "nodes": [
                {"type": "directoryCreateUser", "parameters": {"email": "{employee_email}", "name": "{employee_name}"}},
                {"type": "provisionSystems", "parameters": {"systems": "{systems}"}},
                {"type": "emailSend", "parameters": {"to": "{employee_email}", "subject": "Welcome", "text": "Willkommen!"}},
            ],
        },
    )

    hr_performance_tpl = WorkflowTemplate(
        template_id="hr_performance_review",
        display_name="Performance Review",
        category="hr",
        description="Sammelt Feedback und generiert Review-Dokument",
        parameters=[
            TemplateParameter("employee_email", "string", True, "E-Mail"),
            TemplateParameter("manager_email", "string", True, "Manager E-Mail"),
            TemplateParameter("period", "string", True, "Zeitraum"),
        ],
        workflow_spec={
            "name": "Performance Review",
            "nodes": [
                {"type": "collectFeedback", "parameters": {"employee": "{employee_email}"}},
                {"type": "pdfGenerate", "parameters": {"title": "Review {period}", "content": "Feedback"}},
                {"type": "emailSend", "parameters": {"to": "{manager_email}", "subject": "Review {period}", "text": "Anbei"}},
            ],
        },
    )

    hr_offboarding_tpl = WorkflowTemplate(
        template_id="hr_offboarding",
        display_name="HR Offboarding",
        category="hr",
        description="Entzieht Zugänge und informiert Stakeholder",
        parameters=[
            TemplateParameter("employee_email", "string", True, "E-Mail"),
            TemplateParameter("manager_email", "string", True, "Manager E-Mail"),
        ],
        workflow_spec={
            "name": "HR Offboarding",
            "nodes": [
                {"type": "directoryDisableUser", "parameters": {"email": "{employee_email}"}},
                {"type": "emailSend", "parameters": {"to": "{manager_email}", "subject": "Offboarding", "text": "Abgeschlossen"}},
            ],
        },
    )

    register_templates(registry, hr_onboarding_tpl, hr_performance_tpl, hr_offboarding_tpl)


__all__ = ["register"]

