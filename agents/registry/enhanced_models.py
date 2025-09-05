# backend/kei_agents/registry/enhanced_models.py
"""Erweiterte Datenmodelle für Enterprise Agent Registry.

Implementiert Versionierung, Multi-Tenancy, Rollout-Strategien und erweiterte
Agent-Metadaten für das Keiko Personal Assistant
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


class VersionConstraintType(str, Enum):
    """Typen von Versions-Constraints."""

    EXACT = "exact"  # =1.2.3
    CARET = "caret"  # ^1.2.3 (kompatible Änderungen)
    TILDE = "tilde"  # ~1.2.3 (Patch-Level-Änderungen)
    GREATER = "greater"  # >1.2.3
    GREATER_EQUAL = "gte"  # >=1.2.3
    LESS = "less"  # <1.2.3
    LESS_EQUAL = "lte"  # <=1.2.3
    RANGE = "range"  # 1.2.0 - 1.3.0


class AgentStatus(str, Enum):
    """Erweiterte Agent-Status-Enum."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    ROLLOUT = "rollout"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    FAILED = "failed"


class RolloutStrategy(str, Enum):
    """Rollout-Strategien für Agent-Deployments."""

    IMMEDIATE = "immediate"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    FEATURE_FLAG = "feature_flag"


class TenantAccessLevel(str, Enum):
    """Zugriffslevel für Tenant-übergreifende Operationen."""

    PRIVATE = "private"  # Nur eigener Tenant
    SHARED = "shared"  # Explizit geteilte Agenten
    PUBLIC = "public"  # Öffentlich verfügbar
    RESTRICTED = "restricted"  # Eingeschränkter Zugriff


@dataclass
class SemanticVersion:
    """Semantische Versionierung (SemVer) Implementation."""

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    def __post_init__(self):
        """Validiert Versions-Komponenten."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Versions-Komponenten müssen nicht-negativ sein")

    @classmethod
    def parse(cls, version_string: str) -> SemanticVersion:
        """Parst Versions-String zu SemanticVersion.

        Args:
            version_string: Version im Format "major.minor.patch[-prerelease][+build]"

        Returns:
            SemanticVersion-Instanz

        Raises:
            ValueError: Bei ungültigem Versions-Format
        """
        # SemVer Regex Pattern
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
        match = re.match(pattern, version_string)

        if not match:
            raise ValueError(f"Ungültiges Versions-Format: {version_string}")

        major, minor, patch, prerelease, build = match.groups()

        return cls(
            major=int(major), minor=int(minor), patch=int(patch), prerelease=prerelease, build=build
        )

    def __str__(self) -> str:
        """String-Repräsentation der Version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __eq__(self, other: object) -> bool:
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
        """Kleiner-als-Vergleich."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch

        # Prerelease-Vergleich
        if self.prerelease is None and other.prerelease is None:
            return False
        if self.prerelease is None:
            return False  # Release > Prerelease
        if other.prerelease is None:
            return True  # Prerelease < Release

        return self.prerelease < other.prerelease

    def __le__(self, other: SemanticVersion) -> bool:
        """Kleiner-gleich-Vergleich."""
        return self == other or self < other

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
    """Versions-Constraint für Agent-Dependencies."""

    constraint_type: VersionConstraintType
    version: SemanticVersion
    upper_version: SemanticVersion | None = None  # Für Range-Constraints

    @classmethod
    def parse(cls, constraint_string: str) -> VersionConstraint:
        """Parst Constraint-String zu VersionConstraint.

        Args:
            constraint_string: Constraint im Format "^1.2.3", "~1.2.3", "=1.2.3", etc.

        Returns:
            VersionConstraint-Instanz
        """
        constraint_string = constraint_string.strip()

        # Exact constraint (=1.2.3 oder 1.2.3)
        if constraint_string.startswith("="):
            version_str = constraint_string[1:]
            return cls(VersionConstraintType.EXACT, SemanticVersion.parse(version_str))

        # Caret constraint (^1.2.3)
        if constraint_string.startswith("^"):
            version_str = constraint_string[1:]
            return cls(VersionConstraintType.CARET, SemanticVersion.parse(version_str))

        # Tilde constraint (~1.2.3)
        if constraint_string.startswith("~"):
            version_str = constraint_string[1:]
            return cls(VersionConstraintType.TILDE, SemanticVersion.parse(version_str))

        # Greater than (>1.2.3)
        if constraint_string.startswith(">="):
            version_str = constraint_string[2:]
            return cls(VersionConstraintType.GREATER_EQUAL, SemanticVersion.parse(version_str))

        if constraint_string.startswith(">"):
            version_str = constraint_string[1:]
            return cls(VersionConstraintType.GREATER, SemanticVersion.parse(version_str))

        # Less than (<1.2.3)
        if constraint_string.startswith("<="):
            version_str = constraint_string[2:]
            return cls(VersionConstraintType.LESS_EQUAL, SemanticVersion.parse(version_str))

        if constraint_string.startswith("<"):
            version_str = constraint_string[1:]
            return cls(VersionConstraintType.LESS, SemanticVersion.parse(version_str))

        # Range constraint (1.2.0 - 1.3.0)
        if " - " in constraint_string:
            lower_str, upper_str = constraint_string.split(" - ", 1)
            return cls(
                VersionConstraintType.RANGE,
                SemanticVersion.parse(lower_str.strip()),
                SemanticVersion.parse(upper_str.strip()),
            )

        # Default: exact constraint
        return cls(VersionConstraintType.EXACT, SemanticVersion.parse(constraint_string))

    def satisfies(self, version: SemanticVersion) -> bool:
        """Prüft ob Version das Constraint erfüllt.

        Args:
            version: Zu prüfende Version

        Returns:
            True wenn Version das Constraint erfüllt
        """
        if self.constraint_type == VersionConstraintType.EXACT:
            return version == self.version

        if self.constraint_type == VersionConstraintType.CARET:
            # ^1.2.3 erlaubt >=1.2.3 <2.0.0
            return version >= self.version and version.major == self.version.major

        if self.constraint_type == VersionConstraintType.TILDE:
            # ~1.2.3 erlaubt >=1.2.3 <1.3.0
            return (
                version >= self.version
                and version.major == self.version.major
                and version.minor == self.version.minor
            )

        if self.constraint_type == VersionConstraintType.GREATER:
            return version > self.version

        if self.constraint_type == VersionConstraintType.GREATER_EQUAL:
            return version >= self.version

        if self.constraint_type == VersionConstraintType.LESS:
            return version < self.version

        if self.constraint_type == VersionConstraintType.LESS_EQUAL:
            return version <= self.version

        if self.constraint_type == VersionConstraintType.RANGE:
            return self.upper_version is not None and self.version <= version <= self.upper_version

        return False


@dataclass
class TenantMetadata:
    """Metadaten für Tenant-Management."""

    tenant_id: str
    tenant_name: str
    organization: str | None = None
    contact_email: str | None = None
    access_level: TenantAccessLevel = TenantAccessLevel.PRIVATE
    resource_quotas: dict[str, Any] = field(default_factory=dict)
    billing_info: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validiert Tenant-Metadaten."""
        if not self.tenant_id:
            raise ValueError("tenant_id ist erforderlich")
        if not self.tenant_name:
            raise ValueError("tenant_name ist erforderlich")


@dataclass
class RolloutConfiguration:
    """Konfiguration für Agent-Rollouts."""

    rollout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: RolloutStrategy = RolloutStrategy.IMMEDIATE

    # Canary-spezifische Konfiguration
    canary_percentage: float = 10.0  # Prozent für Canary-Deployment
    canary_duration_minutes: int = 60

    # Blue-Green-spezifische Konfiguration
    blue_green_switch_delay_minutes: int = 5

    # Rolling-Update-Konfiguration
    rolling_batch_size: int = 1
    rolling_delay_seconds: int = 30

    # Feature-Flag-Konfiguration
    feature_flags: dict[str, bool] = field(default_factory=dict)

    # Allgemeine Konfiguration
    auto_rollback_on_error: bool = True
    health_check_interval_seconds: int = 30
    success_threshold_percentage: float = 95.0
    max_rollout_duration_minutes: int = 120

    # Traffic-Migration (gestufte Anteile in Prozent)
    traffic_split_steps: list[int] = field(
        default_factory=lambda: [5, 10, 25, 50, 100]
    )

    # Monitoring und Alerts
    monitoring_enabled: bool = True
    alert_on_failure: bool = True
    notification_channels: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validiert Rollout-Konfiguration."""
        if not 0 <= self.canary_percentage <= 100:
            raise ValueError("canary_percentage muss zwischen 0 und 100 liegen")
        if not 0 <= self.success_threshold_percentage <= 100:
            raise ValueError("success_threshold_percentage muss zwischen 0 und 100 liegen")
        # Validierung der Traffic-Split-Schritte
        if not self.traffic_split_steps:
            raise ValueError("traffic_split_steps darf nicht leer sein")
        if self.traffic_split_steps[-1] != 100:
            raise ValueError("traffic_split_steps muss mit 100 enden")
        prev = -1
        for step in self.traffic_split_steps:
            if step <= prev:
                raise ValueError("traffic_split_steps müssen streng ansteigend sein")
            if step <= 0 or step > 100:
                raise ValueError("traffic_split_steps müssen zwischen 1 und 100 liegen")
            prev = step


@dataclass
class AgentVersionMetadata:
    """Erweiterte Metadaten für Agent-Versionen."""

    agent_id: str
    version: SemanticVersion
    tenant_id: str

    # Basis-Metadaten
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Versions-spezifische Metadaten
    changelog: str = ""
    breaking_changes: list[str] = field(default_factory=list)
    migration_guide_url: str | None = None

    # Framework-Kompatibilität
    framework_version_constraint: VersionConstraint | None = None
    dependencies: dict[str, VersionConstraint] = field(default_factory=dict)

    # Deployment-Metadaten
    status: AgentStatus = AgentStatus.AVAILABLE
    rollout_config: RolloutConfiguration | None = None
    deployment_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Tenant und Zugriff
    access_level: TenantAccessLevel = TenantAccessLevel.PRIVATE
    shared_with_tenants: set[str] = field(default_factory=set)

    # Performance und Monitoring
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    health_check_config: dict[str, Any] = field(default_factory=dict)

    # Lifecycle-Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deprecated_at: datetime | None = None
    end_of_life_at: datetime | None = None

    # Ownership und Governance
    owner: str | None = None
    maintainers: list[str] = field(default_factory=list)
    approval_required: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    def __post_init__(self):
        """Validiert Agent-Versions-Metadaten."""
        if not self.agent_id:
            raise ValueError("agent_id ist erforderlich")
        if not self.tenant_id:
            raise ValueError("tenant_id ist erforderlich")
        if not self.name:
            raise ValueError("name ist erforderlich")

    def get_full_agent_id(self) -> str:
        """Gibt vollständige Agent-ID mit Version zurück."""
        return f"{self.agent_id}@{self.version}"

    def is_compatible_with_framework(self, framework_version: SemanticVersion) -> bool:
        """Prüft Kompatibilität mit Framework-Version.

        Args:
            framework_version: Framework-Version

        Returns:
            True wenn kompatibel
        """
        if not self.framework_version_constraint:
            return True  # Keine Constraint = kompatibel

        return self.framework_version_constraint.satisfies(framework_version)

    def is_accessible_by_tenant(self, requesting_tenant_id: str) -> bool:
        """Prüft ob Agent für Tenant zugänglich ist.

        Args:
            requesting_tenant_id: Anfragender Tenant

        Returns:
            True wenn zugänglich
        """
        # Eigener Tenant hat immer Zugriff
        if requesting_tenant_id == self.tenant_id:
            return True

        # Public Agents sind für alle zugänglich
        if self.access_level == TenantAccessLevel.PUBLIC:
            return True

        # Shared Agents sind für explizit geteilte Tenants zugänglich
        if (
            self.access_level == TenantAccessLevel.SHARED
            and requesting_tenant_id in self.shared_with_tenants
        ):
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung."""
        return {
            "agent_id": self.agent_id,
            "version": str(self.version),
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "status": self.status.value,
            "access_level": self.access_level.value,
            "shared_with_tenants": list(self.shared_with_tenants),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "end_of_life_at": self.end_of_life_at.isoformat() if self.end_of_life_at else None,
            "owner": self.owner,
            "maintainers": self.maintainers,
            "framework_version_constraint": (
                str(self.framework_version_constraint.version)
                if self.framework_version_constraint
                else None
            ),
            "performance_metrics": self.performance_metrics,
            "rollout_config": self.rollout_config.__dict__ if self.rollout_config else None,
        }
