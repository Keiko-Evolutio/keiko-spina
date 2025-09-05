"""E-Commerce-Templates."""

from __future__ import annotations

from services.n8n.workflow_templates import (
    TemplateParameter,
    WorkflowTemplate,
    WorkflowTemplateRegistry,
)

from . import register_templates


def register(registry: WorkflowTemplateRegistry) -> None:
    """Registriert E-Commerce Templates."""
    order_processing_tpl = WorkflowTemplate(
        template_id="ecom_order_processing",
        display_name="E-Commerce Bestellverarbeitung",
        category="ecommerce",
        description="Validiert Bestellung, verbucht Zahlung und sendet Bestätigung",
        parameters=[
            TemplateParameter("order_id", "string", True, "Bestell-ID"),
            TemplateParameter("payment_api", "string", True, "Payment API Endpoint"),
            TemplateParameter("payment_token", "string", True, "Payment API Token"),
            TemplateParameter("email", "string", True, "Kunden E-Mail"),
        ],
        workflow_spec={
            "name": "Order Processing",
            "nodes": [
                {"type": "orderValidate", "parameters": {"orderId": "{order_id}"}},
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "{payment_api}/charge",
                        "headers": {"Authorization": "Bearer {payment_token}"},
                        "body": {"orderId": "{order_id}"},
                    },
                },
                {"type": "emailSend", "parameters": {"to": "{email}", "subject": "Order {order_id}", "text": "Danke!"}},
            ],
        },
    )

    inventory_management_tpl = WorkflowTemplate(
        template_id="ecom_inventory_management",
        display_name="Inventory Management",
        category="ecommerce",
        description="Aktualisiert Lagerbestand und triggert Nachbestellung",
        parameters=[
            TemplateParameter("sku", "string", True, "Artikelnummer"),
            TemplateParameter("delta", "integer", True, "Bestandsänderung"),
            TemplateParameter("reorder_threshold", "integer", True, "Schwellenwert"),
        ],
        workflow_spec={
            "name": "Inventory Management",
            "nodes": [
                {"type": "inventoryAdjust", "parameters": {"sku": "{sku}", "delta": "{delta}"}},
                {"type": "inventoryCheck", "parameters": {"sku": "{sku}", "threshold": "{reorder_threshold}"}},
                {"type": "reorderIfNeeded", "parameters": {"sku": "{sku}"}},
            ],
        },
    )

    ecom_refund_tpl = WorkflowTemplate(
        template_id="ecom_refund",
        display_name="E-Commerce Refund",
        category="ecommerce",
        description="Erstattet Bestellung und informiert Kunden",
        parameters=[
            TemplateParameter("order_id", "string", True, "Bestell-ID"),
            TemplateParameter("payment_api", "string", True, "Payment API Endpoint"),
            TemplateParameter("payment_token", "string", True, "Payment API Token"),
            TemplateParameter("email", "string", True, "Kunden E-Mail"),
        ],
        workflow_spec={
            "name": "Order Refund",
            "nodes": [
                {
                    "type": "httpRequest",
                    "parameters": {
                        "method": "POST",
                        "url": "{payment_api}/refund",
                        "headers": {"Authorization": "Bearer {payment_token}"},
                        "body": {"orderId": "{order_id}"},
                    },
                },
                {"type": "emailSend", "parameters": {"to": "{email}", "subject": "Refund {order_id}", "text": "Erstattung ausgelöst"}},
            ],
        },
    )

    ecom_abandoned_cart_tpl = WorkflowTemplate(
        template_id="ecom_abandoned_cart",
        display_name="Abandoned Cart Reminder",
        category="ecommerce",
        description="Erinnert Kunden an abgebrochene Warenkörbe",
        parameters=[
            TemplateParameter("email", "string", True, "Kunden E-Mail"),
            TemplateParameter("cart_items", "array", False, "Warenkorb-Items"),
        ],
        workflow_spec={
            "name": "Abandoned Cart",
            "nodes": [
                {"type": "emailSend", "parameters": {"to": "{email}", "subject": "Ihr Warenkorb wartet", "text": "Kauf fortsetzen"}},
            ],
        },
    )

    register_templates(registry, order_processing_tpl, inventory_management_tpl, ecom_refund_tpl, ecom_abandoned_cart_tpl)


__all__ = ["register"]

