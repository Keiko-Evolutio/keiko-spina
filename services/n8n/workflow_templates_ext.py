"""Erweiterte Business-Process-Templates für n8n – delegiert an Sub-Module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kei_logging import get_logger

from .templates.crm import register as register_crm
from .templates.ecommerce import register as register_ecommerce
from .templates.finance import register as register_finance
from .templates.hr import register as register_hr
from .templates.marketing import register as register_marketing

if TYPE_CHECKING:
    from .workflow_templates import WorkflowTemplateRegistry

logger = get_logger(__name__)


def register_extended_templates(registry: WorkflowTemplateRegistry) -> None:
    """Registriert erweiterte Templates über Sub-Registrare.

    Args:
        registry: Ziel-Registry
    """
    try:
        register_crm(registry)
        register_ecommerce(registry)
        register_hr(registry)
        register_finance(registry)
        register_marketing(registry)
        logger.debug("Erweiterte n8n-Templates registriert")
    except Exception as exc:  # pragma: no cover - defensiv
        logger.warning(f"Erweiterte Templates teilweise fehlgeschlagen: {exc}")


__all__ = ["register_extended_templates"]
