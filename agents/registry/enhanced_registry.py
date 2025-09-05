# backend/kei_agents/registry/enhanced_registry.py
"""Erweiterte Agent Registry für Keiko Personal Assistant

Integriert Version Pinning, Multi-Tenancy, Advanced Discovery und Rollout-Strategien
mit Backward-Compatibility zur bestehenden Registry.
"""

from __future__ import annotations

import uuid
from typing import Any

from kei_logging import (
    BusinessLogicError,
    ValidationError,
    get_logger,
    with_log_links,
)

from ..metadata.agent_metadata import AgentMetadata
from .discovery_engine import DiscoveryQuery, DiscoveryStrategy, discovery_engine

# Import bestehender Registry-Komponenten
from .dynamic_registry import DynamicAgentRegistry, dynamic_registry

# Import neuer Enhanced-Komponenten
from .enhanced_models import (
    AgentStatus,
    AgentVersionMetadata,
    RolloutConfiguration,
    SemanticVersion,
    TenantAccessLevel,
    TenantMetadata,
)
from .rollout_manager import rollout_manager
from .tenant_manager import tenant_manager
from .version_manager import VersionResolutionError, version_manager

logger = get_logger(__name__)


class EnhancedAgentRegistry:
    """Erweiterte Agent Registry mit Enterprise-Features."""

    def __init__(self, legacy_registry: DynamicAgentRegistry | None = None):
        """Initialisiert Enhanced Registry.

        Args:
            legacy_registry: Bestehende Registry für Backward-Compatibility
        """
        self._legacy_registry = legacy_registry or dynamic_registry
        self._initialized = False

        # Integration-Flags
        self._version_management_enabled = True
        self._multi_tenancy_enabled = True
        self._advanced_discovery_enabled = True
        self._rollout_management_enabled = True

    async def initialize(self) -> None:
        """Initialisiert Enhanced Registry."""
        if self._initialized:
            return

        try:
            # Starte Rollout-Monitoring
            if self._rollout_management_enabled:
                rollout_manager.start_monitoring()

            # Migriere bestehende Agents falls vorhanden
            await self._migrate_legacy_agents()

            self._initialized = True

            logger.info(
                "Enhanced Agent Registry initialisiert",
                extra={
                    "version_management": self._version_management_enabled,
                    "multi_tenancy": self._multi_tenancy_enabled,
                    "advanced_discovery": self._advanced_discovery_enabled,
                    "rollout_management": self._rollout_management_enabled,
                },
            )

        except Exception as e:
            logger.error(f"Fehler bei Registry-Initialisierung: {e}")
            raise BusinessLogicError(
                message=f"Enhanced Registry Initialisierung fehlgeschlagen: {e}",
                component="enhanced_registry",
                operation="initialize",
                cause=e,
            )

    async def _migrate_legacy_agents(self) -> None:
        """Migriert bestehende Agents zur Enhanced Registry."""
        try:
            # Hole bestehende Agents aus Legacy Registry
            legacy_agents = await self._legacy_registry.list_agents()

            if not legacy_agents:
                logger.info("Keine Legacy-Agents für Migration gefunden")
                return

            migrated_count = 0

            # Handle both dict and list return types from legacy registry
            if isinstance(legacy_agents, dict):
                agent_items = legacy_agents.items()
            else:
                # If it's a list, create dict-like items
                agent_items = [(getattr(agent, "agent_id", str(i)), agent) for i, agent in enumerate(legacy_agents)]

            for agent_id, agent_info in agent_items:
                try:
                    # Erstelle Standard-Tenant falls nicht vorhanden
                    default_tenant_id = "default"
                    if not tenant_manager.has_tenant(default_tenant_id):
                        default_tenant = TenantMetadata(
                            tenant_id=default_tenant_id,
                            tenant_name="Default Tenant",
                            organization="Legacy Migration",
                            access_level=TenantAccessLevel.PRIVATE,
                        )
                        tenant_manager.register_tenant(default_tenant)

                    # Konvertiere zu Enhanced Metadata
                    enhanced_metadata = await self._convert_legacy_metadata(
                        agent_info, default_tenant_id
                    )

                    # Registriere in Enhanced Registry
                    await self._register_agent_version_internal(enhanced_metadata)

                    migrated_count += 1

                except Exception as e:
                    logger.warning(f"Migration von Agent {agent_id} fehlgeschlagen: {e}")

            logger.info(f"Legacy-Migration abgeschlossen: {migrated_count} Agents migriert")

        except Exception as e:
            logger.warning(f"Legacy-Migration fehlgeschlagen: {e}")

    async def _convert_legacy_metadata(
        self, legacy_metadata: AgentMetadata, tenant_id: str
    ) -> AgentVersionMetadata:
        """Konvertiert Legacy-Metadata zu Enhanced-Metadata.

        Args:
            legacy_metadata: Legacy Agent-Metadata
            tenant_id: Tenant-ID für Migration

        Returns:
            Enhanced Agent-Metadata
        """
        # Bestimme Version (Default: 1.0.0)
        version = SemanticVersion(1, 0, 0)

        # Konvertiere Capabilities
        capabilities = list(legacy_metadata.available_capabilities.keys())

        return AgentVersionMetadata(
            agent_id=legacy_metadata.agent_id,
            version=version,
            tenant_id=tenant_id,
            name=legacy_metadata.agent_name,
            description="Migriert von Legacy Registry",
            capabilities=capabilities,
            tags=legacy_metadata.tags,
            owner=legacy_metadata.owner,
            status=AgentStatus.AVAILABLE,
            access_level=TenantAccessLevel.PRIVATE,
        )

    @with_log_links(component="enhanced_registry", operation="register_agent")
    async def register_agent(
        self,
        agent_metadata: AgentMetadata | AgentVersionMetadata,
        tenant_id: str | None = None,
        version: str | None = None,
        rollout_config: RolloutConfiguration | None = None,
    ) -> str:
        """Registriert Agent in Enhanced Registry.

        Args:
            agent_metadata: Agent-Metadaten (Legacy oder Enhanced)
            tenant_id: Tenant-ID (erforderlich für Enhanced Mode)
            version: Version (falls Legacy Metadata)
            rollout_config: Rollout-Konfiguration

        Returns:
            Agent-ID

        Raises:
            ValidationError: Bei ungültigen Metadaten
            TenantNotFoundError: Bei unbekanntem Tenant
        """
        await self.initialize()

        # Konvertiere Legacy zu Enhanced falls nötig
        if isinstance(agent_metadata, AgentMetadata):
            if not tenant_id:
                raise ValidationError(
                    message="tenant_id ist erforderlich für Agent-Registrierung",
                    field="tenant_id",
                    value=None,
                )

            # Konvertiere zu Enhanced Metadata
            enhanced_metadata = await self._convert_legacy_to_enhanced(
                agent_metadata, tenant_id, version
            )
        else:
            enhanced_metadata = agent_metadata

        # Setze Rollout-Konfiguration
        if rollout_config:
            enhanced_metadata.rollout_config = rollout_config

        # Registriere Agent-Version
        await self._register_agent_version_internal(enhanced_metadata)

        # Backward-Compatibility: Registriere auch in Legacy Registry
        if isinstance(agent_metadata, AgentMetadata):
            await self._legacy_registry.register_agent(agent_metadata)

        logger.info(
            f"Agent registriert: {enhanced_metadata.agent_id}@{enhanced_metadata.version}",
            extra={
                "agent_id": enhanced_metadata.agent_id,
                "version": str(enhanced_metadata.version),
                "tenant_id": enhanced_metadata.tenant_id,
                "capabilities": enhanced_metadata.capabilities,
            },
        )

        return enhanced_metadata.agent_id

    async def _convert_legacy_to_enhanced(
        self, legacy_metadata: AgentMetadata, tenant_id: str, version: str | None = None
    ) -> AgentVersionMetadata:
        """Konvertiert Legacy zu Enhanced Metadata."""
        # Parse Version
        if version:
            sem_version = SemanticVersion.parse(version)
        else:
            sem_version = SemanticVersion(1, 0, 0)

        # Konvertiere Capabilities
        capabilities = list(legacy_metadata.available_capabilities.keys())

        return AgentVersionMetadata(
            agent_id=legacy_metadata.agent_id,
            version=sem_version,
            tenant_id=tenant_id,
            name=legacy_metadata.agent_name,
            description="Konvertiert von Legacy Registry",
            capabilities=capabilities,
            tags=legacy_metadata.tags,
            owner=legacy_metadata.owner,
            status=AgentStatus.AVAILABLE,
            access_level=TenantAccessLevel.PRIVATE,
        )

    async def _register_agent_version_internal(self, metadata: AgentVersionMetadata) -> None:
        """Interne Agent-Version-Registrierung."""
        # Prüfe Tenant-Existenz
        if self._multi_tenancy_enabled:
            tenant_manager.get_tenant(metadata.tenant_id)  # Wirft TenantNotFoundError

        # Registriere Version
        if self._version_management_enabled:
            version_manager.register_agent_version(metadata)

        # Registriere für Tenant
        if self._multi_tenancy_enabled:
            tenant_manager.register_agent_for_tenant(
                metadata.tenant_id, metadata.agent_id, metadata
            )

        # Registriere Discovery-Instanz
        if self._advanced_discovery_enabled:
            from .discovery_engine import AgentInstance

            instance = AgentInstance(
                instance_id=str(uuid.uuid4()),
                agent_metadata=metadata,
                status=metadata.status,
                health_score=1.0,
                load_factor=0.0,
            )

            discovery_engine.register_agent_instance(instance)

    @with_log_links(component="enhanced_registry", operation="get_agent")
    async def get_agent(
        self, agent_id: str, version_constraint: str = "latest", tenant_id: str | None = None
    ) -> AgentVersionMetadata | None:
        """Holt Agent basierend auf Version-Constraint.

        Args:
            agent_id: Agent-ID
            version_constraint: Versions-Constraint
            tenant_id: Tenant-ID für Zugriffsprüfung

        Returns:
            Agent-Metadaten oder None
        """
        await self.initialize()

        if not self._version_management_enabled:
            # Fallback zu Legacy Registry
            legacy_agent = await self._legacy_registry.get_agent_by_id(agent_id)
            if legacy_agent and isinstance(legacy_agent, AgentMetadata):
                return await self._convert_legacy_to_enhanced(legacy_agent, tenant_id or "default")
            return None

        try:
            return version_manager.resolve_version(agent_id, version_constraint, tenant_id)
        except VersionResolutionError:
            return None

    @with_log_links(component="enhanced_registry", operation="discover_agents")
    async def discover_agents(
        self,
        capabilities: list[str] | None = None,
        tenant_id: str | None = None,
        strategy: DiscoveryStrategy = DiscoveryStrategy.HYBRID,
        max_results: int = 10,
        **kwargs,
    ) -> list[AgentVersionMetadata]:
        """Erweiterte Agent-Discovery.

        Args:
            capabilities: Erforderliche Capabilities
            tenant_id: Tenant-ID für Zugriffsprüfung
            strategy: Discovery-Strategie
            max_results: Maximale Ergebnisse
            **kwargs: Zusätzliche Discovery-Parameter

        Returns:
            Liste von Agent-Metadaten
        """
        await self.initialize()

        if not self._advanced_discovery_enabled:
            # Fallback zu Legacy Registry
            legacy_agents = await self._legacy_registry.list_agents()
            results = []

            # Handle both dict and list return types from legacy registry
            if isinstance(legacy_agents, dict):
                agent_items = legacy_agents.items()
            else:
                # If it's a list, create dict-like items
                agent_items = [(getattr(agent, "agent_id", str(i)), agent) for i, agent in enumerate(legacy_agents)]

            for agent_id, agent_metadata in agent_items:
                if capabilities:
                    agent_caps = set(
                        cap.lower() for cap in agent_metadata.available_capabilities.keys()
                    )
                    required_caps = set(cap.lower() for cap in capabilities)
                    if not required_caps.issubset(agent_caps):
                        continue

                enhanced_metadata = await self._convert_legacy_to_enhanced(
                    agent_metadata, tenant_id or "default"
                )
                results.append(enhanced_metadata)

                if len(results) >= max_results:
                    break

            return results

        # Erstelle Discovery-Query
        query = DiscoveryQuery(
            capabilities=capabilities or [],
            tenant_id=tenant_id,
            strategy=strategy,
            max_results=max_results,
            **kwargs,
        )

        # Führe Discovery durch
        discovery_results = await discovery_engine.discover_agents(query)

        # Extrahiere Agent-Metadaten
        return [result.instance.agent_metadata for result in discovery_results]

    @with_log_links(component="enhanced_registry", operation="start_rollout")
    async def start_agent_rollout(
        self,
        agent_id: str,
        source_version: str,
        target_version: str,
        tenant_id: str,
        rollout_config: RolloutConfiguration | None = None,
    ) -> str:
        """Startet Agent-Rollout.

        Args:
            agent_id: Agent-ID
            source_version: Quell-Version
            target_version: Ziel-Version
            tenant_id: Tenant-ID
            rollout_config: Rollout-Konfiguration

        Returns:
            Rollout-ID
        """
        await self.initialize()

        if not self._rollout_management_enabled:
            raise BusinessLogicError(
                message="Rollout-Management ist nicht aktiviert",
                component="enhanced_registry",
                operation="start_rollout",
            )

        return await rollout_manager.start_rollout(
            agent_id=agent_id,
            source_version=source_version,
            target_version=target_version,
            tenant_id=tenant_id,
            config=rollout_config,
        )

    @with_log_links(component="enhanced_registry", operation="register_tenant")
    async def register_tenant(self, tenant_metadata: TenantMetadata) -> None:
        """Registriert neuen Tenant.

        Args:
            tenant_metadata: Tenant-Metadaten
        """
        await self.initialize()

        if not self._multi_tenancy_enabled:
            raise BusinessLogicError(
                message="Multi-Tenancy ist nicht aktiviert",
                component="enhanced_registry",
                operation="register_tenant",
            )

        tenant_manager.register_tenant(tenant_metadata)

    async def list_agents(
        self, tenant_id: str | None = None, include_versions: bool = False
    ) -> dict[str, Any]:
        """Listet Agents auf.

        Args:
            tenant_id: Filter nach Tenant-ID
            include_versions: Alle Versionen einschließen

        Returns:
            Dictionary mit Agent-Informationen
        """
        await self.initialize()

        if not self._version_management_enabled:
            # Fallback zu Legacy Registry
            return await self._legacy_registry.list_agents()

        agents = {}

        # Hole Agents vom Version Manager
        for agent_id in version_manager.get_agent_ids():
            available_versions = version_manager.get_available_versions(agent_id, tenant_id)

            if not available_versions:
                continue

            if include_versions:
                agents[agent_id] = {
                    "versions": [str(v) for v in available_versions],
                    "latest": str(available_versions[0]) if available_versions else None,
                }
            else:
                # Nur latest Version
                latest_metadata = version_manager.get_agent_metadata(agent_id, "latest", tenant_id)
                if latest_metadata:
                    agents[agent_id] = latest_metadata.to_dict()

        return agents

    async def get_registry_statistics(self) -> dict[str, Any]:
        """Holt Registry-Statistiken.

        Returns:
            Umfassende Statistiken
        """
        await self.initialize()

        stats = {
            "enhanced_registry": {
                "initialized": self._initialized,
                "version_management_enabled": self._version_management_enabled,
                "multi_tenancy_enabled": self._multi_tenancy_enabled,
                "advanced_discovery_enabled": self._advanced_discovery_enabled,
                "rollout_management_enabled": self._rollout_management_enabled,
            }
        }

        # Version Manager Statistiken
        if self._version_management_enabled:
            stats["version_management"] = version_manager.get_version_statistics()

        # Tenant Manager Statistiken
        if self._multi_tenancy_enabled:
            tenant_stats = {}
            for tenant_id in tenant_manager.get_tenant_ids():
                tenant_stats[tenant_id] = tenant_manager.get_tenant_statistics(tenant_id)
            stats["tenant_management"] = tenant_stats

        # Discovery Engine Statistiken
        if self._advanced_discovery_enabled:
            stats["discovery_engine"] = discovery_engine.get_discovery_statistics()

        # Rollout Manager Statistiken
        if self._rollout_management_enabled:
            stats["rollout_management"] = rollout_manager.get_rollout_statistics()

        # Legacy Registry Statistiken
        try:
            legacy_agents = await self._legacy_registry.list_agents()
            stats["legacy_registry"] = {
                "total_agents": len(legacy_agents),
                "agent_ids": list(legacy_agents.keys()) if isinstance(legacy_agents, dict) else [getattr(agent, "agent_id", str(i)) for i, agent in enumerate(legacy_agents)],
            }
        except Exception as e:
            stats["legacy_registry"] = {"error": str(e)}

        return stats

    async def cleanup_registry(self, max_age_days: int = 90) -> dict[str, int]:
        """Führt Registry-Cleanup durch.

        Args:
            max_age_days: Maximales Alter für Cleanup

        Returns:
            Cleanup-Statistiken
        """
        await self.initialize()

        cleanup_stats = {}

        # Version Manager Cleanup
        if self._version_management_enabled:
            cleanup_stats["deprecated_versions_removed"] = (
                version_manager.cleanup_deprecated_versions(max_age_days)
            )

        # Tenant Manager Cleanup
        if self._multi_tenancy_enabled:
            cleanup_stats["inactive_tenants_removed"] = tenant_manager.cleanup_inactive_tenants(
                max_age_days
            )

        logger.info(f"Registry-Cleanup abgeschlossen: {cleanup_stats}")

        return cleanup_stats

    async def shutdown(self) -> None:
        """Fährt Enhanced Registry herunter."""
        if self._rollout_management_enabled:
            rollout_manager.stop_monitoring()

        logger.info("Enhanced Agent Registry heruntergefahren")


# Globale Enhanced Registry Instanz
enhanced_registry = EnhancedAgentRegistry()
