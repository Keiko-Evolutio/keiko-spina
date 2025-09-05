"""KEI-Stream Admin/Stats Routen.

Zusätzliche HTTP-Endpunkte für Statistiken und Limits gemäß KEI-Stream
Spezifikation (z. B. max_streams, max_messages_per_sec, max_frame_bytes).
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter

from kei_logging import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/api/v1/stream", tags=["kei-stream"])


def _int_env(name: str, default: int) -> int:
    """Liest Integer aus ENV mit Fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@router.get("/limits")
async def get_stream_limits() -> dict[str, Any]:
    """Gibt konfigurierbare Limits für KEI-Stream zurück."""
    return {
        "max_streams": _int_env("KEI_STREAM_MAX_STREAMS", 64),
        "max_messages_per_sec": _int_env("KEI_STREAM_MAX_MPS", 200),
        "max_frame_bytes": _int_env("KEI_STREAM_MAX_FRAME_BYTES", 1_048_576),
        "reconnect_token_ttl_secs": _int_env("KEI_STREAM_RECONNECT_TTL_SECS", 300),
    }


@router.get("/limits/tenant/{tenant_id}")
async def get_tenant_limits(tenant_id: str) -> dict[str, Any]:
    """Gibt Limits/Quotas für einen Tenant zurück (ENV-/JSON-basiert)."""
    # Einfacher Lookup über ENV Overrides
    def _tenant_env_int(suffix: str, default: int) -> int:
        name = f"KEI_STREAM_TENANT_{tenant_id.upper()}_{suffix}"
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    return {
        "tenant_id": tenant_id,
        "max_streams": _tenant_env_int("MAX_STREAMS", _int_env("KEI_STREAM_MAX_STREAMS", 64)),
        "max_messages_per_sec": _tenant_env_int("MAX_MPS", _int_env("KEI_STREAM_MAX_MPS", 200)),
        "max_frame_bytes": _tenant_env_int("MAX_FRAME_BYTES", _int_env("KEI_STREAM_MAX_FRAME_BYTES", 1_048_576)),
    }


@router.get("/capabilities")
async def get_stream_capabilities() -> dict[str, Any]:
    """Capabilities-Endpoint für Feature-Flags/Versionierung."""
    return {
        "version": "1.0.0",
        "features": [
            "resume",
            "ack",
            "backpressure",
            "chunking",
            "traceparent",
            "sse-readonly",
            "ws-permessage-deflate",
            "grpc-compression",
        ],
    }
