# backend/kei_agents/registry/version_manager.py
"""Version Manager für Agent Registry.

Implementiert semantische Versionierung, Version-Resolution und
Kompatibilitätsprüfungen für das Keiko Personal Assistant
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from kei_logging import (
    BusinessLogicError,
    LogLinkedError,
    ValidationError,
    get_logger,
    with_log_links,
)

from .enhanced_models import (
    AgentStatus,
    AgentVersionMetadata,
    SemanticVersion,
    VersionConstraint,
)

logger = get_logger(__name__)


class VersionConflictError(LogLinkedError):
    """Fehler bei Versions-Konflikten."""

    def __init__(self, message: str, conflicting_versions: list[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.conflicting_versions = conflicting_versions or []


class VersionResolutionError(LogLinkedError):
    """Fehler bei Version-Resolution."""

    def __init__(
        self,
        message: str,
        requested_constraint: str = None,
        available_versions: list[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.requested_constraint = requested_constraint
        self.available_versions = available_versions or []


class AgentVersionManager:
    """Manager für Agent-Versionierung und Version-Resolution."""

    def __init__(self):
        """Initialisiert Version Manager."""
        self._agent_versions: dict[str, dict[str, AgentVersionMetadata]] = defaultdict(dict)
        self._version_aliases: dict[str, dict[str, str]] = defaultdict(
            dict
        )  # agent_id -> {alias -> version}
        self._framework_version = SemanticVersion.parse("1.0.0")  # Default Framework-Version

    def set_framework_version(self, version: str) -> None:
        """Setzt Framework-Version.

        Args:
            version: Framework-Version als String
        """
        self._framework_version = SemanticVersion.parse(version)
        logger.info(f"Framework-Version gesetzt: {self._framework_version}")

    @with_log_links(component="version_manager", operation="register_version")
    def register_agent_version(self, metadata: AgentVersionMetadata) -> None:
        """Registriert neue Agent-Version.

        Args:
            metadata: Agent-Versions-Metadaten

        Raises:
            ValidationError: Bei ungültigen Metadaten
            BusinessLogicError: Bei Versions-Konflikten
        """
        agent_id = metadata.agent_id
        version_str = str(metadata.version)

        # Prüfe Framework-Kompatibilität
        if not metadata.is_compatible_with_framework(self._framework_version):
            raise ValidationError(
                message=f"Agent {agent_id}@{version_str} ist nicht kompatibel mit Framework-Version {self._framework_version}",
                field="framework_version_constraint",
                value=(
                    str(metadata.framework_version_constraint.version)
                    if metadata.framework_version_constraint
                    else None
                ),
                agent_id=agent_id,
                version=version_str,
            )

        # Prüfe ob Version bereits existiert
        if version_str in self._agent_versions[agent_id]:
            existing = self._agent_versions[agent_id][version_str]
            if existing.status != AgentStatus.DEPRECATED:
                raise BusinessLogicError(
                    message=f"Agent-Version {agent_id}@{version_str} ist bereits registriert",
                    agent_id=agent_id,
                    version=version_str,
                    existing_status=existing.status.value,
                )

        # Registriere Version
        self._agent_versions[agent_id][version_str] = metadata

        # Erstelle Standard-Aliases
        self._update_version_aliases(agent_id)

        logger.info(
            f"Agent-Version registriert: {agent_id}@{version_str}",
            extra={
                "agent_id": agent_id,
                "version": version_str,
                "tenant_id": metadata.tenant_id,
                "status": metadata.status.value,
                "capabilities": metadata.capabilities,
            },
        )

    def _update_version_aliases(self, agent_id: str) -> None:
        """Aktualisiert Version-Aliases für Agent.

        Args:
            agent_id: Agent-ID
        """
        versions = self.get_available_versions(agent_id)
        if not versions:
            return

        # Sortiere Versionen
        sorted_versions = sorted(versions, reverse=True)

        # Latest-Alias
        self._version_aliases[agent_id]["latest"] = str(sorted_versions[0])

        # Major.Minor-Aliases (z.B. "1.2" -> "1.2.3")
        major_minor_map = {}
        for version in sorted_versions:
            major_minor = f"{version.major}.{version.minor}"
            if major_minor not in major_minor_map:
                major_minor_map[major_minor] = version

        for major_minor, version in major_minor_map.items():
            self._version_aliases[agent_id][major_minor] = str(version)

        # Major-Aliases (z.B. "1" -> "1.2.3")
        major_map = {}
        for version in sorted_versions:
            major = str(version.major)
            if major not in major_map:
                major_map[major] = version

        for major, version in major_map.items():
            self._version_aliases[agent_id][major] = str(version)

    @with_log_links(component="version_manager", operation="resolve_version")
    def resolve_version(
        self, agent_id: str, version_constraint: str, tenant_id: str | None = None
    ) -> AgentVersionMetadata:
        """Löst Version-Constraint zu konkreter Agent-Version auf.

        Args:
            agent_id: Agent-ID
            version_constraint: Versions-Constraint (z.B. "^1.2.0", "latest")
            tenant_id: Tenant-ID für Zugriffsprüfung

        Returns:
            Agent-Versions-Metadaten

        Raises:
            VersionResolutionError: Wenn keine passende Version gefunden wird
        """
        # Prüfe Alias
        if version_constraint in self._version_aliases.get(agent_id, {}):
            version_constraint = self._version_aliases[agent_id][version_constraint]

        # Hole verfügbare Versionen
        available_versions = self.get_available_versions(agent_id, tenant_id)

        if not available_versions:
            raise VersionResolutionError(
                message=f"Keine verfügbaren Versionen für Agent {agent_id}",
                agent_id=agent_id,
                requested_constraint=version_constraint,
                tenant_id=tenant_id,
            )

        try:
            # Parse Constraint
            constraint = VersionConstraint.parse(version_constraint)

            # Finde passende Versionen
            matching_versions = []
            for version in available_versions:
                if constraint.satisfies(version):
                    metadata = self._agent_versions[agent_id][str(version)]

                    # Prüfe Tenant-Zugriff
                    if tenant_id and not metadata.is_accessible_by_tenant(tenant_id):
                        continue

                    matching_versions.append((version, metadata))

            if not matching_versions:
                raise VersionResolutionError(
                    message=f"Keine Version von Agent {agent_id} erfüllt Constraint '{version_constraint}'",
                    agent_id=agent_id,
                    requested_constraint=version_constraint,
                    available_versions=[str(v) for v in available_versions],
                    tenant_id=tenant_id,
                )

            # Wähle beste Version (höchste)
            best_version, best_metadata = max(matching_versions, key=lambda x: x[0])

            logger.info(
                f"Version aufgelöst: {agent_id}@{version_constraint} -> {best_version}",
                extra={
                    "agent_id": agent_id,
                    "requested_constraint": version_constraint,
                    "resolved_version": str(best_version),
                    "tenant_id": tenant_id,
                },
            )

            return best_metadata

        except ValueError as e:
            raise VersionResolutionError(
                message=f"Ungültiges Versions-Constraint '{version_constraint}': {e}",
                agent_id=agent_id,
                requested_constraint=version_constraint,
                cause=e,
            )

    def get_available_versions(
        self, agent_id: str, tenant_id: str | None = None
    ) -> list[SemanticVersion]:
        """Holt verfügbare Versionen für Agent.

        Args:
            agent_id: Agent-ID
            tenant_id: Tenant-ID für Zugriffsprüfung

        Returns:
            Liste verfügbarer Versionen
        """
        if agent_id not in self._agent_versions:
            return []

        available_versions = []
        for version_str, metadata in self._agent_versions[agent_id].items():
            # Prüfe Status
            if metadata.status in [AgentStatus.DEPRECATED, AgentStatus.FAILED]:
                continue

            # Prüfe Tenant-Zugriff
            if tenant_id and not metadata.is_accessible_by_tenant(tenant_id):
                continue

            available_versions.append(metadata.version)

        return sorted(available_versions, reverse=True)

    def get_agent_metadata(
        self, agent_id: str, version: str | None = None, tenant_id: str | None = None
    ) -> AgentVersionMetadata | None:
        """Holt Agent-Metadaten für spezifische Version.

        Args:
            agent_id: Agent-ID
            version: Version (optional, default: latest)
            tenant_id: Tenant-ID für Zugriffsprüfung

        Returns:
            Agent-Metadaten oder None
        """
        if version is None:
            version = "latest"

        try:
            return self.resolve_version(agent_id, version, tenant_id)
        except VersionResolutionError:
            return None

    @with_log_links(component="version_manager", operation="deprecate_version")
    def deprecate_version(
        self, agent_id: str, version: str, reason: str, replacement_version: str | None = None
    ) -> None:
        """Markiert Agent-Version als deprecated.

        Args:
            agent_id: Agent-ID
            version: Version
            reason: Grund für Deprecation
            replacement_version: Ersatz-Version

        Raises:
            ValidationError: Wenn Version nicht existiert
        """
        if agent_id not in self._agent_versions or version not in self._agent_versions[agent_id]:
            raise ValidationError(
                message=f"Agent-Version {agent_id}@{version} nicht gefunden",
                field="version",
                value=version,
                agent_id=agent_id,
            )

        metadata = self._agent_versions[agent_id][version]
        metadata.status = AgentStatus.DEPRECATED
        metadata.deprecated_at = datetime.now(UTC)
        metadata.updated_at = datetime.now(UTC)

        # Aktualisiere Aliases
        self._update_version_aliases(agent_id)

        logger.warning(
            f"Agent-Version deprecated: {agent_id}@{version}",
            extra={
                "agent_id": agent_id,
                "version": version,
                "reason": reason,
                "replacement_version": replacement_version,
            },
        )

    def check_dependency_compatibility(
        self, agent_id: str, version: str, dependencies: dict[str, str]
    ) -> tuple[bool, list[str]]:
        """Prüft Kompatibilität von Agent-Dependencies.

        Args:
            agent_id: Agent-ID
            version: Agent-Version
            dependencies: Dependencies als {agent_id: version_constraint}

        Returns:
            Tuple von (kompatibel, Konflikt-Liste)
        """
        if agent_id not in self._agent_versions or version not in self._agent_versions[agent_id]:
            return False, [f"Agent {agent_id}@{version} nicht gefunden"]

        metadata = self._agent_versions[agent_id][version]
        conflicts = []

        # Prüfe Framework-Kompatibilität
        if not metadata.is_compatible_with_framework(self._framework_version):
            conflicts.append(f"Inkompatibel mit Framework-Version {self._framework_version}")

        # Prüfe Agent-Dependencies
        for dep_agent_id, dep_constraint in dependencies.items():
            try:
                self.resolve_version(dep_agent_id, dep_constraint)
            except VersionResolutionError as e:
                conflicts.append(f"Dependency {dep_agent_id}@{dep_constraint}: {e!s}")

        # Prüfe eigene Dependencies
        for dep_agent_id, dep_constraint in metadata.dependencies.items():
            try:
                self.resolve_version(dep_agent_id, str(dep_constraint.version))
            except VersionResolutionError as e:
                conflicts.append(f"Eigene Dependency {dep_agent_id}@{dep_constraint}: {e!s}")

        return len(conflicts) == 0, conflicts

    def get_version_statistics(self) -> dict[str, Any]:
        """Holt Statistiken über registrierte Versionen.

        Returns:
            Statistiken-Dictionary
        """
        total_agents = len(self._agent_versions)
        total_versions = sum(len(versions) for versions in self._agent_versions.values())

        status_counts = defaultdict(int)
        tenant_counts = defaultdict(int)

        for agent_versions in self._agent_versions.values():
            for metadata in agent_versions.values():
                status_counts[metadata.status.value] += 1
                tenant_counts[metadata.tenant_id] += 1

        return {
            "total_agents": total_agents,
            "total_versions": total_versions,
            "status_distribution": dict(status_counts),
            "tenant_distribution": dict(tenant_counts),
            "framework_version": str(self._framework_version),
            "aliases_count": sum(len(aliases) for aliases in self._version_aliases.values()),
        }

    def cleanup_deprecated_versions(self, max_age_days: int = 90) -> int:
        """Entfernt alte deprecated Versionen.

        Args:
            max_age_days: Maximales Alter in Tagen

        Returns:
            Anzahl entfernter Versionen
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=max_age_days)
        removed_count = 0

        for agent_id in list(self._agent_versions.keys()):
            versions_to_remove = []

            for version_str, metadata in self._agent_versions[agent_id].items():
                if (
                    metadata.status == AgentStatus.DEPRECATED
                    and metadata.deprecated_at
                    and metadata.deprecated_at < cutoff_date
                ):
                    versions_to_remove.append(version_str)

            for version_str in versions_to_remove:
                del self._agent_versions[agent_id][version_str]
                removed_count += 1

            # Entferne Agent komplett wenn keine Versionen mehr vorhanden
            if not self._agent_versions[agent_id]:
                del self._agent_versions[agent_id]
                if agent_id in self._version_aliases:
                    del self._version_aliases[agent_id]
            else:
                # Aktualisiere Aliases
                self._update_version_aliases(agent_id)

        if removed_count > 0:
            logger.info(f"Cleanup: {removed_count} deprecated Versionen entfernt")

        return removed_count

    def get_agent_ids(self) -> list[str]:
        """Public API für alle Agent-IDs mit Versionen.

        Returns:
            Liste aller Agent-IDs
        """
        return list(self._agent_versions.keys())


# Globale Version Manager Instanz
version_manager = AgentVersionManager()
