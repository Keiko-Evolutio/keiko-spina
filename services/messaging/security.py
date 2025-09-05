"""Bus-Sicherheitslayer: JWT/OIDC-Validierung und ACL-Erzwingung."""

from __future__ import annotations

import fnmatch
from typing import Any

from kei_logging import get_logger

from .config import bus_settings

try:  # pragma: no cover - optional OIDC/JWT
    import jwt
    from jwt import PyJWKClient  # type: ignore
    _JWT_AVAILABLE = True
except Exception:  # pragma: no cover
    _JWT_AVAILABLE = False

logger = get_logger(__name__)


class JWTClaims:
    """Einfache Repräsentation von JWT Claims (bereits validiert)."""

    def __init__(self, subject: str | None, scopes: list[str], tenant: str | None):
        self.subject = subject
        self.scopes = scopes
        self.tenant = tenant


def _extract_claims_from_headers(headers: dict[str, Any]) -> JWTClaims | None:
    """Extrahiert Claims aus Bus-Headern (z. B. `authorization`, `x-scopes`, `x-tenant`)."""
    try:
        raw_scopes = headers.get("x-scopes") or headers.get("scopes")
        scopes: list[str] = []
        if isinstance(raw_scopes, str):
            scopes = [s.strip() for s in raw_scopes.split(" ") if s.strip()]
        elif isinstance(raw_scopes, list):
            scopes = [str(s) for s in raw_scopes]

        tenant = headers.get("x-tenant") or headers.get("tenant")
        subject = headers.get("x-sub") or headers.get("sub")
        return JWTClaims(subject=subject, scopes=scopes, tenant=tenant)
    except Exception:
        return None


def _subject_matches(pattern: str, subject: str) -> bool:
    """Prüft Pattern-Match mit `>` Wildcards (einfach via fnmatch)."""
    # NATS nutzt `>` als Multi-Wildcard. fnmatch nutzt `*`. Konvertieren.
    converted = pattern.replace(">", "*")
    return fnmatch.fnmatch(subject, converted)


def _validate_jwt_if_required(headers: dict[str, Any]) -> JWTClaims | None:
    """Validiert JWT aus `authorization` Header gegen OIDC‑Settings.

    Returns:
        Extrahierte Claims bei Erfolg oder None
    Raises:
        PermissionError: bei ungültigem oder fehlendem Token, wenn OIDC aktiv ist
    """
    sec = bus_settings.security
    if not sec.enable_oidc:
        # Keine Validierung – nur Claims extrahieren
        return _extract_claims_from_headers(headers)
    # Token aus Header lesen
    auth = headers.get("authorization") or headers.get("Authorization")
    if not isinstance(auth, str) or not auth.lower().startswith("bearer "):
        raise PermissionError("Authorization Bearer Token fehlt")
    token = auth.split(" ", 1)[1]
    if not _JWT_AVAILABLE:
        raise PermissionError("JWT Validierung nicht verfügbar")
    # Vereinfachte Validierung: Issuer/Audience per ENV aus App übernehmen
    import os
    issuer = os.getenv("KEI_BUS_OIDC_ISSUER")
    audience = os.getenv("KEI_BUS_OIDC_AUDIENCE", "kei-bus")
    jwks_uri = os.getenv("KEI_BUS_OIDC_JWKS_URI")
    if not issuer or not jwks_uri:
        raise PermissionError("OIDC Konfiguration unvollständig")
    try:
        jwk_client = PyJWKClient(jwks_uri)
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256", "PS256"],
            audience=audience,
            issuer=issuer,
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_iss": True,
                "verify_exp": True,
            },
        )
        scopes: list[str] = []
        if isinstance(payload.get("scope"), str):
            scopes.extend(payload["scope"].split())
        if isinstance(payload.get("scp"), list):
            scopes.extend([str(x) for x in payload["scp"]])
        subject = str(payload.get("sub")) if payload.get("sub") else None
        tenant = headers.get("x-tenant") or headers.get("tenant")
        return JWTClaims(subject=subject, scopes=scopes, tenant=tenant)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - robust
        raise PermissionError(f"JWT ungültig: {exc}")


def _has_required_scopes(claims: JWTClaims, subject: str, action: str) -> bool:
    """Prüft, ob Claims erforderliche Scopes für Aktion/Subject besitzen."""
    for pattern, actions in bus_settings.required_scopes_by_subject.items():
        if _subject_matches(pattern, subject):
            required = actions.get(action, [])
            if not required:
                return True
            return all(req in claims.scopes for req in required)
    # Kein Match: konservativ ablehnen
    return False


def _check_jwt_requirement(action: str, security_config) -> bool:
    """Prüft, ob JWT für die Aktion erforderlich ist."""
    if action == "publish":
        return security_config.require_jwt_for_publish
    if action == "consume":
        return security_config.require_jwt_for_consume
    return False


def _validate_bus_base_scopes(claims, action: str, security_config) -> None:
    """Validiert Bus-Basis-Scopes."""
    if not security_config.require_bus_base_scopes:
        return

    required_scope = (
        security_config.publish_scope_name if action == "publish"
        else security_config.consume_scope_name
    )

    if required_scope and required_scope not in claims.scopes:
        raise PermissionError("Fehlender Bus-Basis-Scope")


def _validate_tenant_scope(claims, security_config) -> None:
    """Validiert Tenant-spezifische Scopes."""
    if not security_config.require_tenant_scope or not claims.tenant:
        return

    tenant_scope = f"{security_config.tenant_scope_prefix}{claims.tenant}"
    if tenant_scope not in claims.scopes:
        raise PermissionError("Fehlender Tenant-Scope")


def authorize_message(headers: dict[str, Any], subject: str, action: str) -> None:
    """Erzwingt JWT-basierte ACLs für Publish/Consume.

    Args:
        headers: Message-Headers mit JWT-Token
        subject: Subject/Topic der Nachricht
        action: Aktion ("publish" oder "consume")

    Raises:
        PermissionError: bei fehlenden/ungültigen Claims/Scopes
    """
    security_config = bus_settings.security

    # Früher Exit wenn OIDC deaktiviert
    if not security_config.enable_oidc:
        return

    # Früher Exit wenn JWT für diese Aktion nicht erforderlich
    if not _check_jwt_requirement(action, security_config):
        return

    # JWT-Claims validieren
    claims = _validate_jwt_if_required(headers or {})
    if claims is None:
        raise PermissionError("JWT Claims fehlen in Message-Headers")

    # Verschiedene Scope-Validierungen
    _validate_bus_base_scopes(claims, action, security_config)
    _validate_tenant_scope(claims, security_config)

    # Subject-spezifische Scopes prüfen
    if not _has_required_scopes(claims, subject, action):
        raise PermissionError("Fehlende Scopes für Subject")
