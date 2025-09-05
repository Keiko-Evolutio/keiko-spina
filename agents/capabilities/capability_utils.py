# backend/agents/capabilities/capability_utils.py
"""Utility-Funktionen für das Capabilities-System.

Konsolidiert gemeinsame Funktionalitäten und eliminiert Code-Duplikation
im gesamten Capabilities-System.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger

from ..metadata.agent_metadata import CapabilityCategory, CapabilityStatus

logger = get_logger(__name__)


class VersionCompatibility(str, Enum):
    """Versionkompatibilität für Capabilities."""

    COMPATIBLE = "compatible"
    DEPRECATED = "deprecated"
    BREAKING = "breaking"


@dataclass
class VersionInfo:
    """Versionsinformationen für Capabilities."""

    version: str
    introduced_at: datetime
    deprecated_at: datetime | None = None
    removal_planned_at: datetime | None = None
    compatibility: VersionCompatibility = VersionCompatibility.COMPATIBLE
    migration_notes: str | None = None

    def __post_init__(self) -> None:
        """Validiert SemVer-Format."""
        if not CapabilityValidator.is_valid_semver(self.version):
            raise ValueError(f"Ungültiges SemVer-Format: {self.version}")


@dataclass
class CapabilityMetrics:
    """Metriken für Capability-Performance und -Nutzung."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_response_time_ms: float = 0.0
    last_call_timestamp: datetime | None = None
    last_success_timestamp: datetime | None = None
    last_failure_timestamp: datetime | None = None
    last_invocation_at: datetime | None = None

    # Aliase für Backward-Compatibility
    @property
    def total_invocations(self) -> int:
        """Alias für total_calls."""
        return self.total_calls

    @total_invocations.setter
    def total_invocations(self, value: int) -> None:
        """Setter für total_invocations."""
        self.total_calls = value

    @property
    def successful_invocations(self) -> int:
        """Alias für successful_calls."""
        return self.successful_calls

    @successful_invocations.setter
    def successful_invocations(self, value: int) -> None:
        """Setter für successful_invocations."""
        self.successful_calls = value

    @property
    def failed_invocations(self) -> int:
        """Alias für failed_calls."""
        return self.failed_calls

    @failed_invocations.setter
    def failed_invocations(self, value: int) -> None:
        """Setter für failed_invocations."""
        self.failed_calls = value

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Berechnet Fehlerrate."""
        return 1.0 - self.success_rate

    @property
    def error_rate(self) -> float:
        """Alias für failure_rate."""
        return self.failure_rate

    def record_call(self, success: bool, response_time_ms: float) -> None:
        """Zeichnet einen Capability-Aufruf auf."""
        now = datetime.now(UTC)

        self.total_calls += 1
        self.last_call_timestamp = now

        if success:
            self.successful_calls += 1
            self.last_success_timestamp = now
        else:
            self.failed_calls += 1
            self.last_failure_timestamp = now

        if self.total_calls == 1:
            self.average_response_time_ms = response_time_ms
        else:
            alpha = 0.1
            self.average_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.average_response_time_ms
            )


class CapabilityValidator:
    """Validator für Capability-Definitionen und -Parameter."""

    SEMVER_PATTERN = re.compile(
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
        r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
        r"?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )

    CAPABILITY_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    @classmethod
    def is_valid_semver(cls, version_str: str) -> bool:
        """Prüft, ob Version SemVer-konform ist."""
        return bool(cls.SEMVER_PATTERN.match(version_str))

    @classmethod
    def is_valid_capability_id(cls, capability_id: str) -> bool:
        """Prüft, ob Capability-ID gültig ist."""
        if not capability_id or len(capability_id) > 100:
            return False
        return bool(cls.CAPABILITY_ID_PATTERN.match(capability_id))

    @classmethod
    def validate_capability_name(cls, name: str) -> bool:
        """Validiert Capability-Namen."""
        if not name or len(name.strip()) == 0:
            return False
        if len(name) > 200:
            return False
        return True

    @classmethod
    def validate_capability_description(cls, description: str) -> bool:
        """Validiert Capability-Beschreibung."""
        if not description or len(description.strip()) == 0:
            return False
        if len(description) > 1000:
            return False
        return True

    @classmethod
    def validate_capability_parameters(cls, parameters: dict[str, Any]) -> bool:
        """Validiert Capability-Parameter."""
        if not isinstance(parameters, dict):
            return False

        for key in parameters:
            if not isinstance(key, str) or len(key) == 0:
                return False
            if len(key) > 100:
                return False

        return True

    @classmethod
    def validate_dependencies(cls, dependencies: list[str]) -> bool:
        """Validiert Abhängigkeitsliste."""
        if not isinstance(dependencies, list):
            return False

        for dep in dependencies:
            if not isinstance(dep, str) or not cls.is_valid_capability_id(dep):
                return False

        return True


class CapabilityFactory:
    """Factory für die Erstellung von Capability-Objekten."""

    @staticmethod
    def create_version_info(
        version: str,
        compatibility: VersionCompatibility = VersionCompatibility.COMPATIBLE,
        migration_notes: str | None = None
    ) -> VersionInfo:
        """Erstellt VersionInfo-Objekt mit aktueller Zeit."""
        return VersionInfo(
            version=version,
            introduced_at=datetime.now(UTC),
            compatibility=compatibility,
            migration_notes=migration_notes
        )

    @staticmethod
    def create_capability_metrics() -> CapabilityMetrics:
        """Erstellt neues CapabilityMetrics-Objekt."""
        return CapabilityMetrics()


class CapabilityHelper:
    """Helper-Funktionen für Capability-Operationen."""

    @staticmethod
    def generate_capability_id(name: str, category: CapabilityCategory) -> str:
        """Generiert standardisierte Capability-ID."""
        normalized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        normalized_name = re.sub(r"_+", "_", normalized_name).strip("_")

        return f"{category.value}_{normalized_name}"

    @staticmethod
    def is_capability_deprecated(
        status: CapabilityStatus,
        version_info: VersionInfo
    ) -> bool:
        """Prüft, ob Capability veraltet ist."""
        return (
            status == CapabilityStatus.DEPRECATED or
            version_info.deprecated_at is not None or
            version_info.compatibility == VersionCompatibility.DEPRECATED
        )

    @staticmethod
    def calculate_capability_health_score(metrics: CapabilityMetrics) -> float:
        """Berechnet Health-Score basierend auf Metriken."""
        if metrics.total_calls == 0:
            return 1.0

        success_weight = 0.6
        response_time_weight = 0.4

        success_score = metrics.success_rate

        max_acceptable_time = 1000.0
        response_score = max(0.0, 1.0 - (metrics.average_response_time_ms / max_acceptable_time))
        health_score = (
            success_weight * success_score +
            response_time_weight * response_score
        )

        return min(1.0, max(0.0, health_score))

    @staticmethod
    def format_capability_summary(
        capability_id: str,
        name: str,
        category: CapabilityCategory,
        status: CapabilityStatus,
        metrics: CapabilityMetrics
    ) -> str:
        """Formatiert Capability-Zusammenfassung für Logging."""
        health_score = CapabilityHelper.calculate_capability_health_score(metrics)

        return (
            f"Capability[{capability_id}]: {name} "
            f"({category.value}, {status.value}) - "
            f"Health: {health_score:.2f}, "
            f"Calls: {metrics.total_calls}, "
            f"Success Rate: {metrics.success_rate:.2%}"
        )


class CapabilityConstants:
    """Konstanten für das Capability-System."""

    DEFAULT_EXECUTION_TIMEOUT = 30
    MAX_EXECUTION_TIMEOUT = 300

    MAX_CONCURRENT_EXECUTIONS = 10
    MAX_DEPENDENCIES = 20
    MAX_PARAMETERS = 50

    DEFAULT_VERSION = "1.0.0"

    HEALTH_CHECK_INTERVAL = 30
    READINESS_CHECK_INTERVAL = 10
    MIN_SUCCESS_RATE = 0.8
    MAX_RESPONSE_TIME_MS = 1000.0
__all__ = [
    "CapabilityConstants",
    "CapabilityFactory",
    "CapabilityHelper",
    "CapabilityMetrics",
    "CapabilityValidator",
    "VersionCompatibility",
    "VersionInfo",
]
