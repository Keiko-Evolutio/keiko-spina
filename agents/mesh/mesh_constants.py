"""Zentrale Konstanten für das Mesh-Subsystem.

Definiert Konstanten für:
- Encoding und Hash-Algorithmen
- Policy-Entscheidungen
- EventBus-Konfiguration
- Cache-Einstellungen
"""

from typing import Final

# Hash und Encoding-Konfiguration
HASH_ALGORITHM: Final[str] = "sha256"
"""Standard-Hash-Algorithmus für Idempotenz und Cache-Keys."""

DEFAULT_ENCODING: Final[str] = "utf-8"
"""Standard-Encoding für String-zu-Bytes-Konvertierung."""

# Policy-Engine-Entscheidungen
POLICY_DECISION_ALLOW: Final[str] = "allow"
"""Policy-Entscheidung: Zugriff erlaubt."""

POLICY_DECISION_DENY: Final[str] = "deny"
"""Policy-Entscheidung: Zugriff verweigert."""

POLICY_DECISION_UNKNOWN: Final[str] = "unknown"
"""Policy-Entscheidung: Unbekannt oder Fehler."""

# EventBus-Konfiguration
MAX_EVENT_HISTORY_SIZE: Final[int] = 10_000
"""Maximale Anzahl Events in der Event-Historie."""

DEFAULT_CACHE_TTL_SECONDS: Final[int] = 300
"""Standard-TTL für Cache-Einträge in Sekunden (5 Minuten)."""

__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_ENCODING",
    "HASH_ALGORITHM",
    "MAX_EVENT_HISTORY_SIZE",
    "POLICY_DECISION_ALLOW",
    "POLICY_DECISION_DENY",
    "POLICY_DECISION_UNKNOWN",
]
