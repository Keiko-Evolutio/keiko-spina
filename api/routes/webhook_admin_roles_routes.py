"""Admin‑APIs für RBAC Rollenzuweisungen im Webhook‑System.

Stellt Endpunkte zum Auflisten, Zuweisen und Entfernen von Rollen für Subjekte
bereit. Nutzt die bestehende RBAC‑Konfiguration und erweitert diese dynamisch
im Speicher (prozesslokal).
"""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel, Field

from api.middleware.scope_middleware import require_scopes
from kei_logging import get_logger
from security.rbac_config_loader import load_rbac_config
from security.rbac_models import RBACConfig, RoleAssignment, RoleDefinition

from .base import create_router

logger = get_logger(__name__)

router = create_router("/api/v1/webhooks", ["webhooks-admin-roles"])


class RoleUpsertRequest(BaseModel):
    """Payload zur Erstellung/Aktualisierung einer Rolle."""

    name: str = Field(...)
    scopes: list[str] = Field(default_factory=list)


class RoleAssignRequest(BaseModel):
    """Payload zur Zuweisung einer Rolle an ein Subjekt."""

    subject: str = Field(...)
    roles: list[str] = Field(default_factory=list)
    tenant_id: str | None = Field(default=None)


class RBACStateResponse(BaseModel):
    """Aktueller RBAC‑Zustand (Rollen und Zuweisungen)."""

    roles: list[RoleDefinition]
    assignments: list[RoleAssignment]


_RBAC_STATE: RBACConfig | None = None


def _get_state() -> RBACConfig:
    """Gibt veränderbare RBAC‑Konfiguration zurück (prozesslokal)."""
    global _RBAC_STATE
    if _RBAC_STATE is None:
        _RBAC_STATE = load_rbac_config()
    return _RBAC_STATE


@router.get("/admin/roles", response_model=RBACStateResponse)
async def list_roles(request: Request) -> RBACStateResponse:
    """Listet alle Rollen und statischen Zuweisungen auf."""
    require_scopes(request, ["webhook:admin:*"])
    state = _get_state()
    return RBACStateResponse(roles=state.roles, assignments=state.assignments)


@router.post("/admin/roles", response_model=RBACStateResponse)
async def upsert_role(payload: RoleUpsertRequest, request: Request) -> RBACStateResponse:
    """Erstellt oder aktualisiert eine Rolle mit Scopes."""
    require_scopes(request, ["webhook:admin:*"])
    state = _get_state()
    # Ersetzen oder hinzufügen
    for i, r in enumerate(state.roles):
        if r.name == payload.name:
            state.roles[i] = RoleDefinition(name=payload.name, scopes=list(payload.scopes or []))
            break
    else:
        state.roles.append(RoleDefinition(name=payload.name, scopes=list(payload.scopes or [])))
    return RBACStateResponse(roles=state.roles, assignments=state.assignments)


@router.post("/admin/roles/assign", response_model=RBACStateResponse)
async def assign_roles(payload: RoleAssignRequest, request: Request) -> RBACStateResponse:
    """Weist Rollen einem Subjekt zu (optional tenant‑spezifisch)."""
    require_scopes(request, ["webhook:admin:*"])
    state = _get_state()
    state.assignments.append(
        RoleAssignment(subject=payload.subject, roles=list(payload.roles or []), tenant_id=payload.tenant_id)
    )
    return RBACStateResponse(roles=state.roles, assignments=state.assignments)


@router.delete("/admin/roles/assign", response_model=RBACStateResponse)
async def remove_assignment(subject: str, tenant_id: str | None = None, request: Request = None) -> RBACStateResponse:  # type: ignore[assignment]
    """Entfernt alle Zuweisungen für ein Subjekt (optional nur für Tenant)."""
    require_scopes(request, ["webhook:admin:*"])
    state = _get_state()
    state.assignments = [a for a in state.assignments if not (a.subject == subject and (tenant_id is None or a.tenant_id == tenant_id))]
    return RBACStateResponse(roles=state.roles, assignments=state.assignments)


__all__ = ["router"]
