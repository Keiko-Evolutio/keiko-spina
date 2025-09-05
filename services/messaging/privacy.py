"""Privacy-Funktionen: Redaction und Feld-basierte Verschlüsselung (KMS-Adapter)."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger
from kei_logging.pii_redaction import redact_structure

from .config import bus_settings

logger = get_logger(__name__)


# =============================================================================
# Redaction
# =============================================================================

def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduziert PII aus Payload basierend auf globalen Redaction-Regeln.

    Nutzt die bestehende Redaction-Logik aus `kei_logging.pii_redaction`.
    """
    try:
        return redact_structure(payload)
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"Redaction Fehler: {exc}")
        return payload


# =============================================================================
# Einfache KMS-Schnittstelle (Platzhalter):
# - LocalKMS: AES-GCM über abgeleiteten Key (nur Demo, kein echter KMS)
# - Adapter für Azure/Cloud KMS kann später gehookt werden
# =============================================================================

try:  # pragma: no cover - optional, wird in pyproject als dep geführt
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None  # type: ignore


def _derive_key_from_env(key_id: str) -> bytes:
    """Leitet 256-bit Key deterministisch von ENV ab (nur Demo)."""
    seed = (os.getenv("KEI_BUS_LOCAL_KMS_SEED", "keiko-dev-seed") + key_id).encode("utf-8")
    # Naiv: pad/truncate auf 32 Bytes; produktiv: KDF/HKDF mit Salt
    return (seed * 32)[:32]


def _aesgcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct = aes.encrypt(nonce, plaintext, aad)
    return nonce + ct


def _aesgcm_decrypt(key: bytes, data: bytes, aad: bytes) -> bytes:
    nonce, ct = data[:12], data[12:]
    aes = AESGCM(key)
    return aes.decrypt(nonce, ct, aad)


def encrypt_fields(payload: dict[str, Any], fields: list[str], key_id: str) -> dict[str, Any]:
    """Verschlüsselt ausgewählte Felder in Payload (flach/nested via Punkt-Pfad)."""
    if AESGCM is None:
        logger.warning("AESGCM nicht verfügbar - Verschlüsselung übersprungen")
        return payload

    key = _derive_key_from_env(key_id)
    aad = key_id.encode("utf-8")

    def set_by_path(obj: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        cur: Any = obj
        for p in parts[:-1]:
            if not isinstance(cur, dict):
                return
            cur = cur.get(p)
            if cur is None:
                return
        if isinstance(cur, dict):
            cur[parts[-1]] = value

    def get_by_path(obj: dict[str, Any], path: str) -> Any | None:
        parts = path.split(".")
        cur: Any = obj
        for p in parts:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
            if cur is None:
                return None
        return cur

    out = json.loads(json.dumps(payload))
    for path in fields:
        val = get_by_path(out, path)
        if val is None:
            continue
        try:
            raw = json.dumps(val).encode("utf-8")
            ct = _aesgcm_encrypt(key, raw, aad)
            enc = {
                "__enc__": True,
                "alg": "AESGCM",
                "kid": key_id,
                "data": base64.b64encode(ct).decode("utf-8"),
            }
            set_by_path(out, path, enc)
        except Exception as exc:
            logger.warning(f"Feldverschlüsselung fehlgeschlagen für {path}: {exc}")
    return out


def decrypt_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Entschlüsselt Felder, die mit encrypt_fields markiert wurden."""
    if AESGCM is None:
        return payload

    def visit(obj: Any) -> Any:
        if isinstance(obj, dict) and obj.get("__enc__") is True and "kid" in obj and "data" in obj:
            try:
                key_id = str(obj["kid"])
                key = _derive_key_from_env(key_id)
                aad = key_id.encode("utf-8")
                data = base64.b64decode(str(obj["data"]))
                raw = _aesgcm_decrypt(key, data, aad)
                return json.loads(raw.decode("utf-8"))
            except Exception:  # pragma: no cover
                return obj
        if isinstance(obj, dict):
            return {k: visit(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [visit(v) for v in obj]
        return obj

    return visit(payload)


def get_active_kms_key_id() -> str:
    """Ermittelt aktiven KMS Key nach Rotations-Policy.

    - Nutzt `bus_settings.kms_key_id` als Basis
    - Wendet (optionale) periodische Rotation nach `kms_rotation_days` an
    """
    base = bus_settings.kms_key_id
    days = max(0, int(bus_settings.kms_rotation_days))
    if days <= 0:
        return base
    # Berechne Periodenindex seit Epochenbeginn
    today = datetime.now(UTC).date()
    epoch = datetime(1970, 1, 1, tzinfo=UTC).date()
    delta_days = (today - epoch).days
    period = delta_days // days
    return f"{base}-{period}"
