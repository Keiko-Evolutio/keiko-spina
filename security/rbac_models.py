"""RBAC Modelle für das Kei‑Webhook System.

Enthält Pydantic‑Modelle für Rollen, Scope‑Definitionen und Zuweisungen.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RoleDefinition(BaseModel):
    """Definition einer Rolle mit zugehörigen Scopes.

    Attribute:
        name: Eindeutiger Rollenname
        scopes: Liste von Scope‑Ausdrücken (Wildcards erlaubt, z. B. "webhook:outbound:send:*")
    """

    name: str = Field(..., description="Rollenname")
    scopes: list[str] = Field(default_factory=list, description="Scope‑Ausdrücke")


class RoleAssignment(BaseModel):
    """Zuweisung von Rollen zu Subjekten (Nutzer/Service Accounts).

    Attribute:
        subject: Subjekt‑Identität (z. B. JWT 'sub')
        roles: Zugeordnete Rollen (Namen)
        tenant_id: Optionaler Tenant‑Bezug für diese Zuweisung
    """

    subject: str = Field(..., description="Subjekt‑ID (JWT sub)")
    roles: list[str] = Field(default_factory=list, description="Rollenliste")
    tenant_id: str | None = Field(default=None, description="Eingeschränkter Tenant")


class RBACConfig(BaseModel):
    """Gesamtkonfiguration für RBAC.

    Attribute:
        roles: Liste verfügbarer Rollen
        assignments: Statische Zuweisungen (optional)
    """

    roles: list[RoleDefinition] = Field(default_factory=list)
    assignments: list[RoleAssignment] = Field(default_factory=list)


__all__ = [
    "RBACConfig",
    "RoleAssignment",
    "RoleDefinition",
]
