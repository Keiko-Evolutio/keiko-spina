"""Zentrale Autorisierungslogik für Kei‑Webhook RBAC.

Stellt Scope‑Matching, Rollenauflösung und Tenant‑Validierung bereit.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from kei_logging import get_logger

if TYPE_CHECKING:
    from .rbac_models import RBACConfig

logger = get_logger(__name__)


class Principal(BaseModel):
    """Abgeleitete Identität aus JWT.

    Attribute:
        subject: Subjekt‑ID (JWT sub)
        tenant_id: Aktueller Tenant‑Kontext (Header)
        scopes: Direkte Scopes aus Token (optional)
        roles: Rollen aus Token (optional)
    """

    subject: str = Field(...)
    tenant_id: str | None = Field(default=None)
    scopes: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)


class AuthorizationDecision(BaseModel):
    """Ergebnis einer Autorisierungsprüfung."""

    allowed: bool
    reason: str | None = None


class WebhookAuthorizationService:
    """Führt zentrale Autorisierungsprüfungen durch.

    Unterstützt Scope‑Ausdrücke und Wildcards in Rollen.
    """

    def __init__(self, config: RBACConfig) -> None:
        self._config = config
        # Vorabkarte: Rollenname → Scopes
        self._role_to_scopes: dict[str, list[str]] = {
            r.name: list(r.scopes or []) for r in (self._config.roles or [])
        }
        # Statische Assignments nach Subject gruppieren
        self._static_assignments: dict[str, list[str]] = {}
        for a in self._config.assignments or []:
            if not a.subject:
                continue
            self._static_assignments.setdefault(a.subject, []).extend(a.roles or [])

    def _expand_roles_to_scopes(self, principal: Principal) -> list[str]:
        """Leitet effektive Scopes aus Rollen und Token‑Scopes ab."""
        scopes: list[str] = list(principal.scopes or [])
        role_names: list[str] = list(principal.roles or [])
        # Statische Role‑Assignments ergänzen
        role_names.extend(self._static_assignments.get(principal.subject, []))
        # Rollen auflösen
        for role in role_names:
            for s in self._role_to_scopes.get(role, []):
                scopes.append(s)
        # Deduplizieren bei Erhalt der Reihenfolge
        seen: set[str] = set()
        unique_scopes: list[str] = []
        for s in scopes:
            if s in seen:
                continue
            seen.add(s)
            unique_scopes.append(s)
        return unique_scopes

    @staticmethod
    def _scope_matches(required: str, owned: str) -> bool:
        """Vergleicht einen erforderlichen Scope mit einem besessenen Scope.

        Unterstützt einfache Wildcards via fnmatch ("*"), z. B. webhook:admin:*.
        """
        return fnmatch.fnmatchcase(owned, required) or fnmatch.fnmatchcase(required, owned)

    def authorize(self, principal: Principal, required_scopes: list[str], tenant_id: str | None) -> AuthorizationDecision:
        """Prüft, ob der Principal die benötigten Scopes für den Tenant besitzt.

        Regeln:
        - webhook:admin:* gewährt Vollzugriff unabhängig vom konkreten Scope
        - Tenant muss vorhanden und identisch mit Request‑Tenant sein (Tenant‑Isolation)
        """
        # Tenant‑Validierung
        if tenant_id is None:
            return AuthorizationDecision(allowed=False, reason="tenant_required")
        if principal.tenant_id and principal.tenant_id != tenant_id:
            return AuthorizationDecision(allowed=False, reason="tenant_mismatch")

        effective_scopes = self._expand_roles_to_scopes(principal)
        # Admin Shortcut
        if any(self._scope_matches("webhook:admin:*", s) for s in effective_scopes):
            return AuthorizationDecision(allowed=True)

        for required in required_scopes:
            if any(self._scope_matches(required, s) for s in effective_scopes):
                return AuthorizationDecision(allowed=True)
        return AuthorizationDecision(allowed=False, reason="insufficient_scope")


__all__ = [
    "AuthorizationDecision",
    "Principal",
    "WebhookAuthorizationService",
]
