"""Konfiguration für API-Versionierung, Deprecations und Sunset-Policy.

Diese Datei definiert optionale Deprecation-/Sunset-Header für bestimmte
Endpoint-Pfade. Die Middleware liest diese Konfiguration und ergänzt
Responses automatisch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DeprecationRule:
    """Definition einer Deprecation-Regel für Endpoint-Pfade."""

    path_prefix: str
    deprecation: str = "true"  # "true" oder RFC-1123 Datum
    sunset: str | None = None  # RFC-1123 Datum (z. B. "Wed, 31 Dec 2025 23:59:59 GMT")
    link: str | None = None  # URL auf Dokumentation


# Beispiel: Keine aktiven Deprecations standardmäßig, nur Schema bereitstellen.
DEPRECATION_RULES: list[DeprecationRule] = [
    # Globale Deprecation für v1 – mit Sunset in 6 Monaten ab jetzt (Beispiel)
    DeprecationRule(
        path_prefix="/api/v1",
        deprecation="true",
        sunset="Wed, 31 Dec 2025 23:59:59 GMT",
        link="https://docs.keiko.dev/migration/v2"
    ),
]


__all__ = ["DEPRECATION_RULES", "DeprecationRule"]
