"""CRM-Templates: Salesforce und HubSpot."""

from __future__ import annotations

from services.n8n.workflow_templates import (
    TemplateParameter,
    WorkflowTemplate,
    WorkflowTemplateRegistry,
)

from . import register_templates


def register(registry: WorkflowTemplateRegistry) -> None:
    """Registriert CRM-bezogene Templates."""
    salesforce_lead_tpl = WorkflowTemplate(
        template_id="crm_salesforce_lead",
        display_name="Salesforce Lead Erstellen",
        category="crm_integration",
        description="Erstellt oder aktualisiert einen Lead in Salesforce",
        parameters=[
            TemplateParameter("sf_instance_url", "string", True, "Salesforce Instance URL"),
            TemplateParameter("sf_access_token", "string", True, "Salesforce Access Token"),
            TemplateParameter("company", "string", True, "Firma"),
            TemplateParameter("last_name", "string", True, "Nachname"),
            TemplateParameter("email", "string", True, "E-Mail"),
        ],
        workflow_spec={
            "name": "Salesforce Lead Upsert",
            "nodes": [
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "{sf_instance_url}/services/data/v58.0/sobjects/Lead",
                        "headers": {"Authorization": "Bearer {sf_access_token}"},
                        "body": {"Company": "{company}", "LastName": "{last_name}", "Email": "{email}"},
                    },
                }
            ],
        },
    )

    hubspot_contact_tpl = WorkflowTemplate(
        template_id="crm_hubspot_contact",
        display_name="HubSpot Kontakt Upsert",
        category="crm_integration",
        description="Erstellt oder aktualisiert einen Kontakt in HubSpot",
        parameters=[
            TemplateParameter("hs_api_key", "string", True, "HubSpot API Key"),
            TemplateParameter("email", "string", True, "E-Mail"),
            TemplateParameter("first_name", "string", False, "Vorname", default=""),
            TemplateParameter("last_name", "string", False, "Nachname", default=""),
        ],
        workflow_spec={
            "name": "HubSpot Contact Upsert",
            "nodes": [
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "https://api.hubapi.com/crm/v3/objects/contacts",
                        "headers": {"Authorization": "Bearer {hs_api_key}"},
                        "body": {
                            "properties": {
                                "email": "{email}",
                                "firstname": "{first_name}",
                                "lastname": "{last_name}"
                            }
                        },
                    },
                }
            ],
        },
    )

    salesforce_account_tpl = WorkflowTemplate(
        template_id="crm_salesforce_account",
        display_name="Salesforce Account Upsert",
        category="crm_integration",
        description="Erstellt/aktualisiert Account in Salesforce",
        parameters=[
            TemplateParameter("sf_instance_url", "string", True, "Salesforce Instance URL"),
            TemplateParameter("sf_access_token", "string", True, "Salesforce Access Token"),
            TemplateParameter("name", "string", True, "Accountname"),
        ],
        workflow_spec={
            "name": "Salesforce Account Upsert",
            "nodes": [
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "{sf_instance_url}/services/data/v58.0/sobjects/Account",
                        "headers": {"Authorization": "Bearer {sf_access_token}"},
                        "body": {"Name": "{name}"},
                    },
                }
            ],
        },
    )

    hubspot_deal_tpl = WorkflowTemplate(
        template_id="crm_hubspot_deal",
        display_name="HubSpot Deal Erstellen",
        category="crm_integration",
        description="Erstellt Deal in HubSpot",
        parameters=[
            TemplateParameter("hs_api_key", "string", True, "HubSpot API Key"),
            TemplateParameter("amount", "number", True, "Dealbetrag"),
            TemplateParameter("dealname", "string", True, "Dealname"),
        ],
        workflow_spec={
            "name": "HubSpot Deal Create",
            "nodes": [
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "https://api.hubapi.com/crm/v3/objects/deals",
                        "headers": {"Authorization": "Bearer {hs_api_key}"},
                        "body": {"properties": {"amount": "{amount}", "dealname": "{dealname}"}},
                    },
                }
            ],
        },
    )

    register_templates(registry, salesforce_lead_tpl, hubspot_contact_tpl, salesforce_account_tpl, hubspot_deal_tpl)


__all__ = ["register"]

