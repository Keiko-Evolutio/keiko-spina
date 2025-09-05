# backend/core/versioning.py
"""Konsolidiertes Versioning-Modul für Keiko Personal Assistant.

Vereint SemVer-Parsing, Deprecation-Management und Version-Vergleiche
aus verschiedenen Modulen in einer einheitlichen, enterprise-grade Implementierung.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


class VersionConstraintType(str, Enum):
    """Typ von Versions-Constraints."""

    EXACT = "exact"
    MINIMUM = "minimum"
    COMPATIBLE = "compatible"
    RANGE = "range"


@dataclass(frozen=True)
class SemanticVersion:
    """Enterprise-grade semantische Versionierung (SemVer) Implementation.

    Kombiniert die beste Funktionalität aus metadata/versioning.py und
    registry/enhanced_models.py in einer konsolidierten Implementierung.
    """

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    def __post_init__(self) -> None:
        """Validiert Versions-Komponenten."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Versions-Komponenten müssen nicht-negativ sein")

    @classmethod
    def parse(cls, version_string: str) -> SemanticVersion:
        """Parst Versions-String zu SemanticVersion mit robuster Fehlerbehandlung.

        Kombiniert die robuste Parsing-Logik aus beiden ursprünglichen Implementierungen.

        Args:
            version_string: Version im Format "major.minor.patch[-prerelease][+build]"

        Returns:
            SemanticVersion-Instanz

        Raises:
            ValueError: Bei ungültigem Versions-Format
        """
        if not version_string or not isinstance(version_string, str):
            raise ValueError(f"Ungültiger Versions-String: {version_string}")

        # Robuste Parsing-Logik (aus metadata/versioning.py)
        try:
            parts = version_string.split(".")
            major = int("".join(ch for ch in parts[0] if ch.isdigit())) if parts else 0
            minor = int("".join(ch for ch in parts[1] if ch.isdigit())) if len(parts) > 1 else 0
            patch = int("".join(ch for ch in parts[2] if ch.isdigit())) if len(parts) > 2 else 0

            # Erweiterte SemVer-Parsing (aus registry/enhanced_models.py)
            semver_pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
            match = re.match(semver_pattern, version_string)

            if match:
                major, minor, patch, prerelease, build = match.groups()
                return cls(
                    major=int(major),
                    minor=int(minor),
                    patch=int(patch),
                    prerelease=prerelease,
                    build=build
                )

            # Fallback für einfache Versionen
            return cls(major=major, minor=minor, patch=patch)

        except Exception as e:
            logger.warning(f"Fallback zu 0.0.0 für ungültige Version: {version_string}, Fehler: {e}")
            return cls(major=0, minor=0, patch=0)

    def __str__(self) -> str:
        """String-Repräsentation der Version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __eq__(self, other: Any) -> bool:
        """Gleichheits-Vergleich."""
        if not isinstance(other, SemanticVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: SemanticVersion) -> bool:
        """Kleiner-als-Vergleich nach SemVer-Regeln."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        # Vergleiche Major.Minor.Patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        # Prerelease-Vergleich
        if self.prerelease is None and other.prerelease is None:
            return False
        if self.prerelease is None:
            return False  # Release > Prerelease
        if other.prerelease is None:
            return True   # Prerelease < Release

        return self.prerelease < other.prerelease

    def __le__(self, other: SemanticVersion) -> bool:
        """Kleiner-gleich-Vergleich."""
        return self < other or self == other

    def __gt__(self, other: SemanticVersion) -> bool:
        """Größer-als-Vergleich."""
        return not self <= other

    def __ge__(self, other: SemanticVersion) -> bool:
        """Größer-gleich-Vergleich."""
        return not self < other

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Prüft Kompatibilität nach SemVer-Regeln.

        Args:
            other: Andere Version

        Returns:
            True wenn kompatibel (gleiche Major-Version, neuere Minor/Patch)
        """
        return self.major == other.major and (
            self.minor > other.minor or (self.minor == other.minor and self.patch >= other.patch)
        )


@dataclass
class VersionConstraint:
    """Versions-Constraint für Dependencies und Kompatibilitätsprüfungen."""

    constraint_type: VersionConstraintType
    version: SemanticVersion
    upper_version: SemanticVersion | None = None  # Für Range-Constraints

    def satisfies(self, version: SemanticVersion) -> bool:
        """Prüft ob Version den Constraint erfüllt.

        Args:
            version: Zu prüfende Version

        Returns:
            True wenn Version den Constraint erfüllt
        """
        if self.constraint_type == VersionConstraintType.EXACT:
            return version == self.version
        if self.constraint_type == VersionConstraintType.MINIMUM:
            return version >= self.version
        if self.constraint_type == VersionConstraintType.COMPATIBLE:
            return version.is_compatible_with(self.version)
        if self.constraint_type == VersionConstraintType.RANGE:
            if self.upper_version is None:
                return version >= self.version
            return self.version <= version <= self.upper_version

        # Fallback für unbekannte Constraint-Typen
        return False


@dataclass
class DeprecationInfo:
    """Ergebnis einer Deprecation-Prüfung."""

    deprecated: bool
    warning: str | None = None
    migration_link: str | None = None
    changelog_link: str | None = None


def is_deprecation_due(current: str, target: str, min_minor_gap: int = 2) -> bool:
    """Prüft ob Deprecation aufgrund von Versions-Abstand fällig ist.

    Args:
        current: Aktuelle Version
        target: Ziel-Version
        min_minor_gap: Minimaler Minor-Versions-Abstand für Deprecation

    Returns:
        True wenn Deprecation fällig ist
    """
    current_ver = SemanticVersion.parse(current)
    target_ver = SemanticVersion.parse(target)

    if target_ver.major > current_ver.major:
        return True
    if target_ver.major == current_ver.major and (target_ver.minor - current_ver.minor) >= min_minor_gap:
        return True
    return False


def evaluate_deprecation(
    *,
    current_version: str,
    target_version: str,
    migration_link: str | None = None,
    changelog_link: str | None = None,
    min_minor_gap: int = 2
) -> DeprecationInfo:
    """Erzeugt Deprecation-Information nach SemVer-Regel.

    Args:
        current_version: Aktuelle Version
        target_version: Ziel-Version
        migration_link: Link zu Migrationshinweisen
        changelog_link: Link zum Changelog
        min_minor_gap: Minimaler Minor-Versions-Abstand

    Returns:
        DeprecationInfo mit Deprecation-Status und Hinweisen
    """
    if is_deprecation_due(current_version, target_version, min_minor_gap):
        warning = (
            f"Feature deprecating: current={current_version}, target={target_version}. "
            f"Bitte Migrationshinweise beachten."
        )
        return DeprecationInfo(
            deprecated=True,
            warning=warning,
            migration_link=migration_link,
            changelog_link=changelog_link,
        )
    return DeprecationInfo(deprecated=False)


def capability_flag_for_version(current_version: str, feature_since: str) -> bool:
    """Prüft ob Capability ab bestimmter Version verfügbar ist.

    Args:
        current_version: Aktuelle Version
        feature_since: Version seit der das Feature verfügbar ist

    Returns:
        True wenn Capability verfügbar ist
    """
    current = SemanticVersion.parse(current_version)
    since = SemanticVersion.parse(feature_since)
    return current >= since


# Legacy-Kompatibilität für bestehenden Code
def parse_semver(version: str) -> tuple[int, int, int]:
    """Legacy-Funktion für Rückwärtskompatibilität.

    Args:
        version: Versions-String

    Returns:
        Tuple von (major, minor, patch)

    Deprecated:
        Verwende SemanticVersion.parse() für neue Implementierungen
    """
    logger.warning("parse_semver() ist deprecated. Verwende SemanticVersion.parse()")
    parsed = SemanticVersion.parse(version)
    return parsed.major, parsed.minor, parsed.patch


__all__ = [
    "DeprecationInfo",
    "SemanticVersion",
    "VersionConstraint",
    "VersionConstraintType",
    "capability_flag_for_version",
    "evaluate_deprecation",
    "is_deprecation_due",
    "parse_semver",  # Legacy
]
