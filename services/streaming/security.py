"""Sicherheits- und ACL-Helfer für KEI-Stream.

Erzwingt OIDC-Scopes pro Tenant und Topic (Stream-ID) sowie einfache
Topic-ACLs. Unterstützt Wildcards über "kei.stream.*" und differenziert
zwischen Lese- und Schreibrechten (r, w, rw).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class StreamAclConfig:
    """Konfiguration der Scope- und ACL-Präfixe."""

    tenant_scope_prefix: str = "kei.tenant:"
    stream_scope_prefix: str = "kei.stream.topic:"
    wildcard_scope: str = "kei.stream.*"

    @classmethod
    def from_env(cls) -> StreamAclConfig:
        """Erstellt Konfiguration aus Umgebungsvariablen."""
        return cls(
            tenant_scope_prefix=os.getenv("KEI_STREAM_TENANT_SCOPE_PREFIX", "kei.tenant:"),
            stream_scope_prefix=os.getenv("KEI_STREAM_TOPIC_SCOPE_PREFIX", "kei.stream.topic:"),
            wildcard_scope=os.getenv("KEI_STREAM_WILDCARD_SCOPE", "kei.stream.*"),
        )


def _normalize_scopes(scopes: Iterable[str] | None) -> list[str]:
    """Normalisiert Scopes zu sortierter Liste."""
    if not scopes:
        return []
    unique = {s.strip() for s in scopes if isinstance(s, str) and s.strip()}
    return sorted(unique)


def has_tenant_access(scopes: Iterable[str] | None, tenant_id: str, *, cfg: StreamAclConfig | None = None) -> bool:
    """Prüft, ob Scopes Zugriff auf gegebenen Tenant erlauben.

    Erlaubt wird Zugriff, wenn ein Scope exakt "<tenant_prefix><tenant_id>"
    vorhanden ist. Beispiel: "kei.tenant:acme".
    """
    if not tenant_id:
        return False
    config = cfg or StreamAclConfig.from_env()
    scope_list = _normalize_scopes(scopes)
    required = f"{config.tenant_scope_prefix}{tenant_id}"
    return required in scope_list


def has_topic_access(
    scopes: Iterable[str] | None,
    topic: str,
    *,
    write: bool,
    cfg: StreamAclConfig | None = None,
) -> bool:
    """Prüft Topic-ACLs anhand der Scopes.

    Zugelassen, wenn einer der folgenden Bedingungen erfüllt ist:
    - Wildcard-Scope (z. B. "kei.stream.*")
    - Exakter Topic-Scope (z. B. "kei.stream.topic:orders:rw"). Bei Schreibzugriff
      wird "rw" oder "w" akzeptiert. Bei Lesezugriff "rw" oder "r".
    """
    if not topic:
        return False
    config = cfg or StreamAclConfig.from_env()
    scope_list = _normalize_scopes(scopes)

    if config.wildcard_scope in scope_list:
        return True

    # Formate: "<prefix><topic>:rw|r|w" oder ohne Suffix (implizit rw)
    base = f"{config.stream_scope_prefix}{topic}"
    if base in scope_list:
        return True

    required = {f"{base}:rw"}
    if write:
        required.add(f"{base}:w")
    else:
        required.add(f"{base}:r")

    return any(r in scope_list for r in required)


__all__ = [
    "StreamAclConfig",
    "has_tenant_access",
    "has_topic_access",
]
