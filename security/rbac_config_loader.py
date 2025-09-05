"""Lädt RBAC‑Konfiguration aus Umgebungsvariablen oder Dateien.

Unterstützt einfache JSON‑Definitionen für Rollen und Assignments.
"""

from __future__ import annotations

import json
import os

from kei_logging import get_logger

from .rbac_models import RBACConfig, RoleAssignment, RoleDefinition

logger = get_logger(__name__)


def load_rbac_config() -> RBACConfig:
    """Lädt RBAC‑Konfiguration aus ENV.

    Erwartete Variablen:
        RBAC_ROLES_JSON: JSON‑Array von RoleDefinition
        RBAC_ASSIGNMENTS_JSON: JSON‑Array von RoleAssignment
    """
    roles_json = os.environ.get("RBAC_ROLES_JSON", "").strip()
    assignments_json = os.environ.get("RBAC_ASSIGNMENTS_JSON", "").strip()

    roles = []
    assignments = []
    try:
        if roles_json:
            roles = [RoleDefinition(**it) for it in json.loads(roles_json)]
    except Exception as exc:
        logger.warning(f"RBAC Rollen Parse‑Fehler: {exc}")
    try:
        if assignments_json:
            assignments = [RoleAssignment(**it) for it in json.loads(assignments_json)]
    except Exception as exc:
        logger.warning(f"RBAC Assignments Parse‑Fehler: {exc}")

    if not roles:
        roles = [
            RoleDefinition(name="webhook_admin", scopes=["webhook:admin:*"])
        ]
    return RBACConfig(roles=roles, assignments=assignments)


__all__ = ["load_rbac_config"]
