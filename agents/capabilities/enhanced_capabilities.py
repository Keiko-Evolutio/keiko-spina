# backend/agents/capabilities/enhanced_capabilities.py
"""Enhanced Capabilities System.

Implementiert erweiterte Capabilities mit besserer Code-Qualität.
Verwendet konsolidierte Module für Health-Checks und Utilities.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from kei_logging import get_logger

from ..metadata.agent_metadata import (
    CapabilityCategory,
    CapabilityStatus,
)
from .capability_utils import (
    CapabilityMetrics,
    VersionCompatibility,
    VersionInfo,
)
from .health_interfaces import (
    BaseHealthChecker as HealthChecker,
)
from .health_interfaces import (
    BaseReadinessChecker as ReadinessChecker,
)
from .health_interfaces import (
    DefaultHealthChecker,
    DefaultReadinessChecker,
    HealthCheckResult,
    ReadinessCheckResult,
)

logger = get_logger(__name__)





class EnhancedCapability(BaseModel):
    """Erweiterte Capability-Definition mit Health/Readiness und Versionierung.

    Implementiert erweiterte Funktionalität mit verbesserter Code-Qualität.
    """

    id: str = Field(..., description="Eindeutige Capability-ID")
    name: str = Field(..., description="Name der Capability")
    description: str = Field(..., description="Beschreibung der Capability")
    category: CapabilityCategory = Field(..., description="Kategorie der Capability")

    version_info: VersionInfo = Field(..., description="Versionsinformationen")
    supported_versions: list[str] = Field(
        default_factory=list, description="Unterstützte Versionen"
    )

    status: CapabilityStatus = Field(default=CapabilityStatus.AVAILABLE)

    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Capability-Parameter"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Abhängigkeiten"
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Zugehörige Endpunkte"
    )

    tags: set[str] = Field(default_factory=set, description="Tags für Kategorisierung")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Zusätzliche Metadaten"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_health_check: datetime | None = None
    last_readiness_check: datetime | None = None

    class Config:
        """Pydantic-Konfiguration."""
        use_enum_values = True
        arbitrary_types_allowed = True

    @field_validator("version_info")
    @classmethod
    def validate_version_info(cls, v: VersionInfo) -> VersionInfo:
        """Validiert Versionsinformationen."""
        return v

    @field_validator("id")
    @classmethod
    def validate_capability_id(cls, v: str) -> str:
        """Validiert Capability-ID."""
        from .capability_utils import CapabilityValidator

        if not CapabilityValidator.is_valid_capability_id(v):
            raise ValueError(f"Ungültige Capability-ID: {v}")
        return v

    @field_validator("name")
    @classmethod
    def validate_capability_name(cls, v: str) -> str:
        """Validiert Capability-Namen."""
        from .capability_utils import CapabilityValidator

        if not CapabilityValidator.validate_capability_name(v):
            raise ValueError(f"Ungültiger Capability-Name: {v}")
        return v

    @field_validator("description")
    @classmethod
    def validate_capability_description(cls, v: str) -> str:
        """Validiert Capability-Beschreibung."""
        from .capability_utils import CapabilityValidator

        if not CapabilityValidator.validate_capability_description(v):
            raise ValueError("Ungültige Capability-Beschreibung")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_capability_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validiert Capability-Parameter."""
        from .capability_utils import CapabilityValidator

        if not CapabilityValidator.validate_capability_parameters(v):
            raise ValueError("Ungültige Capability-Parameter")
        return v

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, v: list[str]) -> list[str]:
        """Validiert Abhängigkeitsliste."""
        from .capability_utils import CapabilityValidator

        if not CapabilityValidator.validate_dependencies(v):
            raise ValueError("Ungültige Abhängigkeitsliste")
        return v

    def is_deprecated(self) -> bool:
        """Prüft, ob Capability veraltet ist."""
        from .capability_utils import CapabilityHelper

        return CapabilityHelper.is_capability_deprecated(self.status, self.version_info)

    def is_compatible_with_version(self, version: str) -> bool:
        """Prüft Kompatibilität mit einer bestimmten Version."""
        if version in self.supported_versions:
            return True

        try:
            from packaging import version as pkg_version
            current = pkg_version.parse(self.version_info.version)
            target = pkg_version.parse(version)

            return current.major == target.major
        except Exception:
            return False

    def update_timestamp(self) -> None:
        """Aktualisiert den updated_at Zeitstempel."""
        self.updated_at = datetime.now(UTC)

    def record_health_check(self) -> None:
        """Zeichnet Health-Check-Zeitstempel auf."""
        self.last_health_check = datetime.now(UTC)

    def record_readiness_check(self) -> None:
        """Zeichnet Readiness-Check-Zeitstempel auf."""
        self.last_readiness_check = datetime.now(UTC)

    def get_summary(self) -> str:
        """Gibt eine formatierte Zusammenfassung der Capability zurück."""
        from .capability_utils import CapabilityHelper

        mock_metrics = CapabilityMetrics()

        return CapabilityHelper.format_capability_summary(
            self.id, self.name, self.category, self.status, mock_metrics
        )

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Capability zu Dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version_info.version,
            "status": self.status.value,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "endpoints": self.endpoints,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_deprecated": self.is_deprecated(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnhancedCapability:
        """Erstellt Capability aus Dictionary."""
        from .capability_utils import CapabilityFactory


        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        if "version_info" not in data and "version" in data:
            data["version_info"] = CapabilityFactory.create_version_info(data["version"])

        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = set(data["tags"])

        return cls(**data)



__all__ = [
    "CapabilityMetrics",
    "DefaultHealthChecker",
    "DefaultReadinessChecker",
    "EnhancedCapability",
    "HealthCheckResult",
    "HealthChecker",
    "ReadinessCheckResult",
    "ReadinessChecker",
    "VersionCompatibility",
    "VersionInfo",
]
