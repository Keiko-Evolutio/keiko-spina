"""KEI-Stream Management-Endpoints.

Stellt Verwaltungsendpunkte bereit zum Auflisten/Aufräumen von Sessions,
Statistiken, sowie zum Abbrechen von Streams und Ausstellen von Reconnect-
Tokens (vereinfachte Demo-Variante).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Any

from fastapi import APIRouter

from config.mtls_config import MTLS_SETTINGS
from kei_logging import get_logger

from .config_utils import get_env_int, get_env_str
from .constants import (
    DEFAULT_RECONNECT_SECRET,
    DEFAULT_RECONNECT_TTL_SECONDS,
    ENV_KEI_STREAM_RECONNECT_SECRET,
    ENV_KEI_STREAM_RECONNECT_TTL_SECS,
)
from .session import session_manager

logger = get_logger(__name__)


router = APIRouter(prefix="/stream/manage", tags=["kei-stream-management"])


@router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    """Listet aktuelle Sessions und einfache Statistiken."""
    # Interner Zugriff: Direkter Zugriff auf Manager-Struktur wird vermieden,
    # hier lediglich eine schlanke Sicht zurückgeben.
    sessions: list[dict[str, Any]] = []
    # Vorsicht: Kein Lock hier; nur best-effort Snapshot via get()
    for sid in list(getattr(session_manager, "_sessions", {}).keys()):
        ctx = await session_manager.get(sid)
        if not ctx:
            continue
        sessions.append(
            {
                "session_id": sid,
                "created_at": ctx.created_at.isoformat(),
                "last_seen_at": ctx.last_seen_at.isoformat(),
                "stream_count": len(ctx.streams),
            }
        )
    return {"sessions": sessions}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """Schließt eine Session."""
    await session_manager.close(session_id)
    return {"status": "closed", "session_id": session_id}


@router.post("/cleanup")
async def cleanup_idle() -> dict[str, Any]:
    """Bereinigt inaktive Sessions."""
    removed = await session_manager.cleanup_idle()
    return {"removed": removed}


@router.delete("/sessions/{session_id}/streams/{stream_id}")
async def cancel_stream(session_id: str, stream_id: str) -> dict[str, Any]:
    """Bricht einen Stream innerhalb einer Session ab."""
    ok = await session_manager.cancel_stream(session_id, stream_id)
    if not ok:
        from core.exceptions import KeikoNotFoundError
        raise KeikoNotFoundError("Stream nicht gefunden", details={"session_id": session_id, "stream_id": stream_id})
    return {"status": "cancelled", "session_id": session_id, "stream_id": stream_id}


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Einfache Statistiken über Sessions/Streams."""
    return await session_manager.stats()


@router.get("/infra/ws-mtls-check")
async def ws_mtls_infra_check() -> dict[str, Any]:
    """Gibt Hinweise zur WS-mTLS Infrastruktur-Konfiguration.

    Dieser Endpoint dient als Checkliste, ob die Umgebung korrekt
    für WebSocket-mTLS via TLS-Termination/Proxy vorbereitet ist.

    Returns:
        Diagnose-Informationen und empfohlene Proxy-Header
    """
    enabled = MTLS_SETTINGS.inbound.enabled
    mode = MTLS_SETTINGS.inbound.mode.value if MTLS_SETTINGS.inbound else "disabled"
    header_name = MTLS_SETTINGS.inbound.cert_header_name
    recommendations = [
        {
            "component": "TLS-Termination/Proxy",
            "checks": [
                "Client Certificate Authentication aktivieren (mTLS)",
                f"Client-Zertifikat als Base64-PEM in Header '{header_name}' weiterleiten",
                "Header auch bei WebSocket Upgrade (101 Switching Protocols) setzen",
                "Optional: SSL_CLIENT_CERT oder X-SSL-Client-Cert als Fallback setzen",
            ],
        },
        {
            "component": "KEI-Stream Server",
            "checks": [
                "KEI_MCP_INBOUND_MTLS_ENABLED=true setzen",
                f"KEI_MCP_CLIENT_CERT_HEADER={header_name} konsistent konfigurieren",
                "KEI_MCP_CLIENT_CA_PATH setzen, wenn CA-Validierung gewünscht",
                "KEI_MCP_INBOUND_MTLS_MODE=required|optional passend zum Sicherheitsprofil wählen",
            ],
        },
    ]

    return {
        "ws_mtls": {
            "enabled": enabled,
            "mode": mode,
            "cert_header": header_name,
        },
        "recommendations": recommendations,
    }


# ---------------- Reconnect Token Ausstellung/Validierung ----------------

def _sign_token(data: str, secret: str) -> str:
    """Signiert Daten mit HMAC-SHA256 und liefert Base64-URL-Signatur."""
    mac = hmac.new(secret.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("utf-8").rstrip("=")


@router.post("/sessions/{session_id}/reconnect")
async def issue_reconnect_token(session_id: str, tenant_id: str | None = None) -> dict[str, Any]:
    """Stellt ein kurzlebiges Reconnect-Token für eine Session aus.

    Token-Format (Base64URL): base64url(session_id|tenant_id|exp_ts|sig)
    sig = HMAC_SHA256(secret, session_id|tenant_id|exp_ts)
    """
    # Lade Konfiguration aus ENV mit Defaults
    secret = get_env_str(ENV_KEI_STREAM_RECONNECT_SECRET, DEFAULT_RECONNECT_SECRET)
    ttl_secs = get_env_int(ENV_KEI_STREAM_RECONNECT_TTL_SECS, DEFAULT_RECONNECT_TTL_SECONDS)

    # Token-Erstellung
    exp_ts = str(int(time.time()) + max(1, ttl_secs))
    tenant_val = tenant_id or ""
    payload = f"{session_id}|{tenant_val}|{exp_ts}"
    sig = _sign_token(payload, secret)
    token = base64.urlsafe_b64encode((payload + "|" + sig).encode("utf-8")).decode("utf-8").rstrip("=")

    return {"session_id": session_id, "token": token, "expires_at": int(exp_ts)}


@router.post("/sessions/{session_id}/reconnect/validate")
async def validate_reconnect_token(session_id: str, token: str) -> dict[str, Any]:
    """Validiert ein Reconnect-Token und gibt Status zurück."""
    secret = os.getenv("KEI_STREAM_RECONNECT_SECRET", "change-me")
    try:
        padded = token + "=" * ((4 - len(token) % 4) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
        parts = raw.split("|")
        if len(parts) != 4:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Ungültiges Tokenformat")
        tok_sid, tok_tenant, exp_ts, sig = parts
        payload = f"{tok_sid}|{tok_tenant}|{exp_ts}"
        expected = _sign_token(payload, secret)
        if not hmac.compare_digest(expected, sig):
            from core.exceptions import KeikoAuthenticationError
            raise KeikoAuthenticationError("Ungültige Signatur")
        if tok_sid != session_id:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Abweichende session_id")
        if int(exp_ts) < int(__import__("time").time()):
            from core.exceptions import KeikoAuthenticationError
            raise KeikoAuthenticationError("Token abgelaufen")
        return {"valid": True, "session_id": tok_sid, "tenant_id": tok_tenant or None, "exp": int(exp_ts)}
    except Exception as exc:
        from core.exceptions import KeikoBadRequestError
        raise KeikoBadRequestError("Ungültige Anfrage", details={"valid": False, "error": str(exc)})
