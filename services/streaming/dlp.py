"""Einfache DLP-/Redaction-Policies für KEI-Stream.

Ermöglicht feldweise Redaction basierend auf konfigurierbaren Pfaden.
Die Implementierung ist leichtgewichtig und arbeitet rein auf JSON-
Strukturen (dict/list/primitive).
"""

from __future__ import annotations

from typing import Any

from .config_utils import get_env_str
from .constants import (
    DEFAULT_REDACTION_MASK,
    ENV_KEI_STREAM_DLP_RULES,
    ENV_KEI_STREAM_REDACTION_MASK,
)


def _parse_redaction_rules(env_value: str) -> dict[str, list[str]]:
    """Parst Redaktionsregeln aus ENV.

    Format: "tenant1=payload.ssn,payload.email;tenant2=payload.card".
    """
    rules: dict[str, list[str]] = {}
    entries = [e.strip() for e in env_value.split(";") if e.strip()]
    for entry in entries:
        if "=" not in entry:
            continue
        tenant, paths = entry.split("=", 1)
        rules[tenant.strip()] = [p.strip() for p in paths.split(",") if p.strip()]
    return rules


def _apply_redaction_to_path(obj: Any, path: str, redaction_mask: str) -> None:
    """Setzt Wert auf redaction_mask für gegebenen Pfad (z. B. payload.ssn).

    Args:
        obj: Objekt in dem redaktiert werden soll
        path: Punkt-separierter Pfad zum Feld (z.B. "payload.ssn")
        redaction_mask: Wert der als Ersatz gesetzt wird
    """
    parts = [p for p in path.split(".") if p]
    if not parts:
        return

    current = obj
    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        if isinstance(current, dict) and part in current:
            if is_last:
                current[part] = redaction_mask
            else:
                current = current[part]
        else:
            # Pfad existiert nicht, keine Redaction möglich
            return


def redact_payload_for_tenant(tenant_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Reduziert sensible Felder im Payload anhand Tenant-spezifischer Regeln.

    Regeln werden über ENV KEI_STREAM_DLP_RULES konfiguriert.
    Format: "tenant1=payload.ssn,payload.email;tenant2=payload.card".

    Args:
        tenant_id: ID des Tenants für den redaktiert werden soll
        payload: Payload-Dictionary das redaktiert werden soll

    Returns:
        Kopie des Payloads mit redaktierten Feldern
    """
    try:
        # Lade Redaction-Regeln und -Mask aus ENV
        rules_env = get_env_str(ENV_KEI_STREAM_DLP_RULES, "")
        redaction_mask = get_env_str(ENV_KEI_STREAM_REDACTION_MASK, DEFAULT_REDACTION_MASK)

        if not rules_env:
            return payload

        rules = _parse_redaction_rules(rules_env)
        paths = rules.get(tenant_id, [])

        if not paths:
            return payload

        # Defensive Kopie erstellen
        redacted_payload = dict(payload)

        # Redaction auf alle konfigurierten Pfade anwenden
        for path in paths:
            _apply_redaction_to_path(redacted_payload, path, redaction_mask)

        return redacted_payload

    except Exception:
        # Bei Fehlern Original-Payload zurückgeben (fail-safe)
        return payload


__all__ = ["redact_payload_for_tenant"]
