"""SemVer Versionierung, Deprecation-Policy und Migrationshinweise.

DEPRECATED: Dieses Modul ist deprecated. Verwende backend.core.versioning.

Legacy-Kompatibilität für bestehenden Code.
Alle Funktionen sind Wrapper um das konsolidierte core.versioning Modul.
"""

from __future__ import annotations

import warnings

# Import aus konsolidiertem Modul
from core.versioning import (
    DeprecationInfo as _DeprecationInfo,
)
from core.versioning import (
    capability_flag_for_version as _capability_flag_for_version,
)
from core.versioning import (
    evaluate_deprecation as _evaluate_deprecation,
)
from core.versioning import (
    is_deprecation_due as _is_deprecation_due,
)
from core.versioning import (
    parse_semver as _parse_semver,
)

# Deprecation-Warning
warnings.warn(
    "backend.agents.metadata.versioning ist deprecated. "
    "Verwende backend.core.versioning für neue Implementierungen.",
    DeprecationWarning,
    stacklevel=2
)


# Legacy-Wrapper
def parse_semver(version: str) -> tuple[int, int, int]:
    """Parst eine SemVer‑Zeichenkette (major.minor.patch) robust.

    DEPRECATED: Verwende backend.core.versioning.SemanticVersion.parse()
    """
    return _parse_semver(version)


def is_deprecation_due(current: str, target: str, min_minor_gap: int = 2) -> bool:
    """Prüft, ob `target` im Verhältnis zu `current` (SemVer) mindestens `min_minor_gap` Minor entfernt ist.

    DEPRECATED: Verwende backend.core.versioning.is_deprecation_due()
    """
    return _is_deprecation_due(current, target, min_minor_gap)


# Re-export
DeprecationInfo = _DeprecationInfo


def evaluate_deprecation(
    *,
    current_version: str,
    target_version: str,
    migration_link: str | None = None,
    changelog_link: str | None = None,
) -> DeprecationInfo:
    """Erzeugt Deprecation‑Information nach SemVer‑Regel (≥ 2 Minor).

    DEPRECATED: Verwende backend.core.versioning.evaluate_deprecation()
    """
    return _evaluate_deprecation(
        current_version=current_version,
        target_version=target_version,
        migration_link=migration_link,
        changelog_link=changelog_link,
    )


def capability_flag_for_version(current_version: str, feature_since: str) -> bool:
    """Gibt `True` zurück, wenn Capability ab `feature_since` verfügbar ist (SemVer‑Vergleich).

    DEPRECATED: Verwende backend.core.versioning.capability_flag_for_version()
    """
    return _capability_flag_for_version(current_version, feature_since)


__all__ = [
    "DeprecationInfo",
    "capability_flag_for_version",
    "evaluate_deprecation",
    "is_deprecation_due",
    "parse_semver",
]
