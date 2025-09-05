"""Quota-Profile und Limitauflösung für KEI-Stream.

Dieses Modul liefert eine einfache, ENV-gesteuerte Quota-Auflösung für
`max_streams` pro Tenant oder API-Key. Fallback auf globale Defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config_utils import (
    get_env_int,
    get_env_json,
    get_tenant_specific_env,
    resolve_hierarchical_config,
)
from .constants import (
    DEFAULT_MAX_STREAMS,
    ENV_KEI_STREAM_MAX_STREAMS,
    ENV_KEI_STREAM_QUOTAS,
    MIN_QUOTA_VALUE,
)


@dataclass
class QuotaProfile:
    """Quota-Profil für einen Client-Kontext.

    Attributes:
        max_streams: Maximale Anzahl paralleler Streams je Session
    """

    max_streams: int


def _resolve_from_json_config(tenant_id: str | None, api_key: str | None) -> QuotaProfile | None:
    """Liest Quotas aus `KEI_STREAM_QUOTAS` JSON, falls vorhanden.

    Erwartete Struktur:
    {
      "default": {"max_streams": 64},
      "tenants": {"tenantA": {"max_streams": 128}},
      "api_keys": {"mykey": {"max_streams": 32}}
    }
    """
    cfg = get_env_json(ENV_KEI_STREAM_QUOTAS)
    if not cfg:
        return None

    # Hierarchische Konfiguration auflösen
    effective_config = resolve_hierarchical_config(
        base_config=cfg,
        tenant_id=tenant_id,
        api_key=api_key,
        config_key="default"
    )

    max_streams = effective_config.get("max_streams")
    if max_streams is not None:
        # Validierung des Werts
        max_streams = max(int(max_streams), MIN_QUOTA_VALUE)
        return QuotaProfile(max_streams=max_streams)

    return None


def _resolve_from_env_vars(tenant_id: str | None, api_key: str | None) -> QuotaProfile | None:
    """Liest Quotas aus ENV-Variablen-Präfixen, falls vorhanden.

    Unterstützte Variablen:
    - KEI_STREAM_MAX_STREAMS (globaler Default)
    - KEI_STREAM_TENANT_<TENANT>_MAX_STREAMS
    - KEI_STREAM_APIKEY_<APIKEY>_MAX_STREAMS
    """
    # Verwende tenant-spezifische ENV-Variable-Auflösung
    max_streams = get_tenant_specific_env(
        base_name="KEI_STREAM_MAX_STREAMS",
        tenant_id=tenant_id,
        api_key=api_key,
        parser_func=get_env_int,
        default=None
    )

    if max_streams is not None:
        # Validierung des Werts
        max_streams = max(max_streams, MIN_QUOTA_VALUE)
        return QuotaProfile(max_streams=max_streams)

    return None


def resolve_quota(tenant_id: str | None, api_key: str | None) -> QuotaProfile:
    """Ermittelt das wirksame Quota-Profil für den Client-Kontext."""
    # Versuche JSON-Konfiguration
    profile = _resolve_from_json_config(tenant_id, api_key)
    if profile:
        return profile

    # Versuche ENV-Variablen
    profile = _resolve_from_env_vars(tenant_id, api_key)
    if profile:
        return profile

    # Fallback auf globale Defaults
    max_streams = get_env_int(ENV_KEI_STREAM_MAX_STREAMS, DEFAULT_MAX_STREAMS)
    max_streams = max(max_streams, MIN_QUOTA_VALUE)  # Validierung

    return QuotaProfile(max_streams=max_streams)


__all__ = ["QuotaProfile", "resolve_quota"]
