"""Kategorie-Registrare für erweiterte n8n-Templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..workflow_templates import WorkflowTemplate, WorkflowTemplateRegistry


class TemplateRegistrar(Protocol):
    """Protokoll für Template-Registrare."""

    def __call__(self, registry: WorkflowTemplateRegistry) -> None:  # pragma: no cover - Protokoll
        ...


def register_templates(registry: WorkflowTemplateRegistry, *templates: WorkflowTemplate) -> None:
    """Registriert mehrere Templates in einer Registry.

    Diese Utility-Funktion eliminiert duplizierte Registration-Patterns
    und bietet eine einheitliche API für Template-Registrierung.

    Args:
        registry: Die Ziel-Registry
        *templates: Variable Anzahl von WorkflowTemplate-Objekten
    """
    for template in templates:
        registry._templates[template.template_id] = template  # noqa: SLF001


__all__ = ["TemplateRegistrar", "register_templates"]

