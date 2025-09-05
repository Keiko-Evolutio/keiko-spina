"""Finance-Templates: Invoice Processing, Expense Approval, Payment Reconciliation."""

from __future__ import annotations

from services.n8n.workflow_templates import (
    TemplateParameter,
    WorkflowTemplate,
    WorkflowTemplateRegistry,
)

from . import register_templates


def register(registry: WorkflowTemplateRegistry) -> None:
    """Registriert Finance-Templates."""
    finance_invoice_tpl = WorkflowTemplate(
        template_id="finance_invoice_processing",
        display_name="Invoice Processing",
        category="finance",
        description="Extrahiert Rechnungsdaten und bucht in ERP",
        parameters=[
            TemplateParameter("invoice_url", "string", True, "Rechnungs-URL"),
            TemplateParameter("erp_endpoint", "string", True, "ERP Endpoint"),
            TemplateParameter("erp_token", "string", True, "ERP Token"),
        ],
        workflow_spec={
            "name": "Invoice Processing",
            "nodes": [
                {"type": "httpRequest", "parameters": {"method": "GET", "url": "{invoice_url}"}},
                {"type": "ocrExtract", "parameters": {}},
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "{erp_endpoint}/invoices",
                        "headers": {"Authorization": "Bearer {erp_token}"},
                        "body": {"items": "{{ $json.extracted }}"},
                    },
                },
            ],
        },
    )

    finance_expense_tpl = WorkflowTemplate(
        template_id="finance_expense_approval",
        display_name="Expense Approval",
        category="finance",
        description="Prüft Spesen und führt Freigabe-Workflow",
        parameters=[
            TemplateParameter("employee", "string", True, "Mitarbeiter-ID/E-Mail"),
            TemplateParameter("amount", "number", True, "Betrag"),
            TemplateParameter("manager_email", "string", True, "Manager E-Mail"),
        ],
        workflow_spec={
            "name": "Expense Approval",
            "nodes": [
                {"type": "rulesCheck", "parameters": {"amount": "{amount}"}},
                {"type": "emailSend", "parameters": {"to": "{manager_email}", "subject": "Expense Approval", "text": "Bitte freigeben"}},
                {"type": "approvalWait", "parameters": {}},
                {"type": "directoryNotify", "parameters": {"employee": "{employee}"}},
            ],
        },
    )

    finance_payment_reconciliation_tpl = WorkflowTemplate(
        template_id="finance_payment_reconciliation",
        display_name="Payment Reconciliation",
        category="finance",
        description="Abgleich von Zahlungen mit offenen Posten",
        parameters=[
            TemplateParameter("erp_endpoint", "string", True, "ERP Endpoint"),
            TemplateParameter("erp_token", "string", True, "ERP Token"),
        ],
        workflow_spec={
            "name": "Payment Reconciliation",
            "nodes": [
                {"type": "httpRequest", "parameters": {"method": "GET", "url": "{erp_endpoint}/open_items", "headers": {"Authorization": "Bearer {erp_token}"}}},
                {"type": "matchPayments", "parameters": {}},
            ],
        },
    )

    register_templates(registry, finance_invoice_tpl, finance_expense_tpl, finance_payment_reconciliation_tpl)


__all__ = ["register"]

