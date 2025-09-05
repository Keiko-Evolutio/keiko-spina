"""Kompressions-Policies für KEI-Stream und gRPC.

Dieses Modul liefert eine einfache, ENV-/JSON-gesteuerte Auflösung
von Kompressionsprofilen pro Tenant oder API-Key.
"""

from __future__ import annotations

from dataclasses import dataclass

import grpc

from .config_utils import get_env_bool, get_env_json, get_env_str, resolve_hierarchical_config
from .constants import (
    DEFAULT_GRPC_COMPRESSION,
    DEFAULT_WS_PERMESSAGE_DEFLATE,
    ENV_KEI_RPC_DEFAULT_COMPRESSION,
    ENV_KEI_STREAM_COMPRESSION,
    ENV_KEI_STREAM_WS_PERMESSAGE_DEFLATE,
    SUPPORTED_GRPC_COMPRESSIONS,
)


@dataclass
class CompressionProfile:
    """Kompressionsprofil für Transportebene.

    Attributes:
        ws_permessage_deflate: Ob WebSocket permessage-deflate gewünscht ist
        grpc_compression: gRPC Kompressionsalgorithmus: "gzip", "deflate", "none"
    """

    ws_permessage_deflate: bool
    grpc_compression: str





def get_grpc_compression_from_str(value: str | None) -> grpc.Compression | None:
    """Mappt String auf gRPC-Kompressionsenum.

    Args:
        value: "gzip", "deflate", "none" oder None

    Returns:
        grpc.Compression oder None
    """
    if not value:
        return None
    val = value.lower()
    if val == "gzip":
        return grpc.Compression.Gzip
    if val == "deflate":
        return grpc.Compression.Deflate
    if val in SUPPORTED_GRPC_COMPRESSIONS:
        return grpc.Compression.NoCompression
    return None


def resolve_compression(tenant_id: str | None, api_key: str | None) -> CompressionProfile:
    """Ermittelt Kompressionsprofil anhand Tenant/API-Key/Defaults.

    Auflösungsreihenfolge:
      1. JSON `KEI_STREAM_COMPRESSION`:
         - `api_keys[<api_key>]`
         - `tenants[<tenant_id>]`
         - `default`
      2. Einzelne ENV-Fallbacks

    JSON-Schema-Beispiel:
    {
      "default": {"ws_permessage_deflate": true, "grpc_compression": "gzip"},
      "tenants": {"tenantA": {"ws_permessage_deflate": false}},
      "api_keys": {"K123": {"grpc_compression": "none"}}
    }
    """
    # Lade JSON-Konfiguration
    cfg = get_env_json(ENV_KEI_STREAM_COMPRESSION)

    # Default-Werte aus ENV oder Constants
    default_ws = get_env_bool(ENV_KEI_STREAM_WS_PERMESSAGE_DEFLATE, DEFAULT_WS_PERMESSAGE_DEFLATE)
    default_grpc = get_env_str(ENV_KEI_RPC_DEFAULT_COMPRESSION, DEFAULT_GRPC_COMPRESSION)

    # Hierarchische Konfiguration auflösen
    effective_config = resolve_hierarchical_config(
        base_config=cfg,
        tenant_id=tenant_id,
        api_key=api_key,
        config_key="default"
    )

    # Fallback auf ENV-Defaults wenn nicht in JSON konfiguriert
    ws_deflate = effective_config.get("ws_permessage_deflate", default_ws)
    grpc_comp = effective_config.get("grpc_compression", default_grpc)

    return CompressionProfile(
        ws_permessage_deflate=bool(ws_deflate),
        grpc_compression=str(grpc_comp),
    )


__all__ = [
    "CompressionProfile",
    "get_grpc_compression_from_str",
    "resolve_compression",
]
