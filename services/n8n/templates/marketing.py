"""Marketing-Templates: Lead Nurturing, Campaign Management, Newsletter."""

from __future__ import annotations

from services.n8n.workflow_templates import (
    TemplateParameter,
    WorkflowTemplate,
    WorkflowTemplateRegistry,
)

from . import register_templates


def register(registry: WorkflowTemplateRegistry) -> None:
    """Registriert Marketing-Templates."""
    marketing_lead_tpl = WorkflowTemplate(
        template_id="mkt_lead_nurturing",
        display_name="Lead Nurturing",
        category="marketing",
        description="Segmentiert Leads und sendet Sequenz-Mails",
        parameters=[
            TemplateParameter("lead_email", "string", True, "Lead E-Mail"),
            TemplateParameter("segment", "string", False, "Segment", default="general"),
        ],
        workflow_spec={
            "name": "Lead Nurturing",
            "nodes": [
                {"type": "segmentAssign", "parameters": {"segment": "{segment}"}},
                {"type": "emailSend", "parameters": {"to": "{lead_email}", "subject": "Welcome", "text": "Phase 1"}},
                {"type": "wait", "parameters": {"days": 3}},
                {"type": "emailSend", "parameters": {"to": "{lead_email}", "subject": "Follow Up", "text": "Phase 2"}},
            ],
        },
    )

    marketing_campaign_tpl = WorkflowTemplate(
        template_id="mkt_campaign_management",
        display_name="Campaign Management",
        category="marketing",
        description="Plant Kampagne, verteilt Inhalte und sammelt KPIs",
        parameters=[
            TemplateParameter("name", "string", True, "Kampagnenname"),
            TemplateParameter("channels", "array", False, "Kanäle", default=["email", "social"]),
        ],
        workflow_spec={
            "name": "Campaign Management",
            "nodes": [
                {"type": "campaignPlan", "parameters": {"name": "{name}", "channels": "{channels}"}},
                {"type": "contentDistribute", "parameters": {"channels": "{channels}"}},
                {"type": "collectKPIs", "parameters": {}},
            ],
        },
    )

    marketing_newsletter_tpl = WorkflowTemplate(
        template_id="mkt_newsletter",
        display_name="Newsletter Versand",
        category="marketing",
        description="Versendet Newsletter an Segment",
        parameters=[
            TemplateParameter("segment", "string", True, "Segment"),
            TemplateParameter("subject", "string", True, "Betreff"),
            TemplateParameter("body", "string", True, "Inhalt"),
        ],
        workflow_spec={
            "name": "Newsletter",
            "nodes": [
                {"type": "fetchSegment", "parameters": {"segment": "{segment}"}},
                {"type": "emailBulkSend", "parameters": {"subject": "{subject}", "text": "{body}"}},
            ],
        },
    )

    register_templates(registry, marketing_lead_tpl, marketing_campaign_tpl, marketing_newsletter_tpl)


__all__ = ["register"]

