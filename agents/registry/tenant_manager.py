"""Tenant Manager für Multi-Tenancy Support in Agent Registry.

Implementiert Tenant-Isolation, Cross-Tenant-Policies und
Ressourcen-Quotas.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from kei_logging import (
    AuthorizationError,
    BusinessLogicError,
    LogLinkedError,
    get_logger,
    with_log_links,
)

from .enhanced_models import AgentVersionMetadata, TenantAccessLevel, TenantMetadata

logger = get_logger(__name__)


class TenantNotFoundError(LogLinkedError):
    """Fehler wenn Tenant nicht gefunden wird."""

    def __init__(self, tenant_id: str, **kwargs):
        super().__init__(f"Tenant '{tenant_id}' nicht gefunden", **kwargs)
        self.tenant_id = tenant_id


class TenantQuotaExceededError(LogLinkedError):
    """Fehler bei Überschreitung von Tenant-Quotas."""

    def __init__(
        self, tenant_id: str, quota_type: str, current_usage: int, quota_limit: int, **kwargs
    ):
        super().__init__(
            f"Tenant '{tenant_id}' hat Quota für '{quota_type}' überschritten: {current_usage}/{quota_limit}",
            **kwargs,
        )
        self.tenant_id = tenant_id
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit


class TenantAccessDeniedError(AuthorizationError):
    """Fehler bei verweigerten Tenant-Zugriff."""

    def __init__(self, requesting_tenant: str, target_tenant: str, resource: str, **kwargs):
        super().__init__(
            f"Tenant '{requesting_tenant}' hat keinen Zugriff auf Ressource '{resource}' von Tenant '{target_tenant}'",
            **kwargs,
        )
        self.requesting_tenant = requesting_tenant
        self.target_tenant = target_tenant
        self.resource = resource


class TenantManager:
    """Manager für Multi-Tenant-Funktionalitäten."""

    def __init__(self):
        """Initialisiert Tenant Manager."""
        self._tenants: dict[str, TenantMetadata] = {}
        self._tenant_agents: dict[str, set[str]] = defaultdict(set)  # tenant_id -> agent_ids
        self._cross_tenant_policies: dict[str, dict[str, Any]] = {}
        self._default_quotas = {
            "max_agents": 100,
            "max_versions_per_agent": 10,
            "max_storage_mb": 1024,
            "max_requests_per_hour": 10000,
            "max_concurrent_deployments": 5,
        }

    @with_log_links(component="tenant_manager", operation="register_tenant")
    def register_tenant(self, metadata: TenantMetadata) -> None:
        """Registriert neuen Tenant.

        Args:
            metadata: Tenant-Metadaten

        Raises:
            ValidationError: Bei ungültigen Metadaten
            BusinessLogicError: Wenn Tenant bereits existiert
        """
        tenant_id = metadata.tenant_id

        if tenant_id in self._tenants:
            raise BusinessLogicError(
                message=f"Tenant '{tenant_id}' ist bereits registriert", tenant_id=tenant_id
            )

        # Setze Standard-Quotas falls nicht definiert
        if not metadata.resource_quotas:
            metadata.resource_quotas = self._default_quotas.copy()

        self._tenants[tenant_id] = metadata

        logger.info(
            f"Tenant registriert: {tenant_id}",
            extra={
                "tenant_id": tenant_id,
                "tenant_name": metadata.tenant_name,
                "organization": metadata.organization,
                "access_level": metadata.access_level.value,
            },
        )

    def get_tenant(self, tenant_id: str) -> TenantMetadata:
        """Holt Tenant-Metadaten.

        Args:
            tenant_id: Tenant-ID

        Returns:
            Tenant-Metadaten

        Raises:
            TenantNotFoundError: Wenn Tenant nicht existiert
        """
        if tenant_id not in self._tenants:
            raise TenantNotFoundError(tenant_id)

        return self._tenants[tenant_id]

    def list_tenants(
        self,
        requesting_tenant_id: str | None = None,
        access_level: TenantAccessLevel | None = None,
    ) -> list[TenantMetadata]:
        """Listet Tenants basierend auf Zugriffsberechtigung.

        Args:
            requesting_tenant_id: Anfragender Tenant
            access_level: Filter nach Access-Level

        Returns:
            Liste von Tenant-Metadaten
        """
        tenants = []

        for tenant in self._tenants.values():
            # Access-Level-Filter
            if access_level and tenant.access_level != access_level:
                continue

            # Zugriffsprüfung
            if requesting_tenant_id:
                if (
                    tenant.tenant_id != requesting_tenant_id
                    and tenant.access_level == TenantAccessLevel.PRIVATE
                ):
                    continue

            tenants.append(tenant)

        return tenants

    @with_log_links(component="tenant_manager", operation="register_agent")
    def register_agent_for_tenant(
        self, tenant_id: str, agent_id: str, metadata: AgentVersionMetadata
    ) -> None:
        """Registriert Agent für Tenant.

        Args:
            tenant_id: Tenant-ID
            agent_id: Agent-ID
            metadata: Agent-Metadaten

        Raises:
            TenantNotFoundError: Wenn Tenant nicht existiert
            TenantQuotaExceededError: Bei Quota-Überschreitung
        """
        # Prüfe Tenant-Existenz
        tenant = self.get_tenant(tenant_id)

        # Prüfe Agent-Quota
        current_agents = len(self._tenant_agents[tenant_id])
        max_agents = tenant.resource_quotas.get("max_agents", self._default_quotas["max_agents"])

        if agent_id not in self._tenant_agents[tenant_id] and current_agents >= max_agents:
            raise TenantQuotaExceededError(
                tenant_id=tenant_id,
                quota_type="max_agents",
                current_usage=current_agents,
                quota_limit=max_agents,
            )

        # Registriere Agent
        self._tenant_agents[tenant_id].add(agent_id)

        logger.info(
            f"Agent für Tenant registriert: {agent_id} -> {tenant_id}",
            extra={
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "agent_version": str(metadata.version),
                "total_agents": len(self._tenant_agents[tenant_id]),
            },
        )

    def unregister_agent_from_tenant(self, tenant_id: str, agent_id: str) -> None:
        """Entfernt Agent von Tenant.

        Args:
            tenant_id: Tenant-ID
            agent_id: Agent-ID
        """
        if tenant_id in self._tenant_agents:
            self._tenant_agents[tenant_id].discard(agent_id)

            logger.info(
                f"Agent von Tenant entfernt: {agent_id} <- {tenant_id}",
                extra={
                    "tenant_id": tenant_id,
                    "agent_id": agent_id,
                    "remaining_agents": len(self._tenant_agents[tenant_id]),
                },
            )

    def get_tenant_agents(self, tenant_id: str) -> set[str]:
        """Holt Agent-IDs für Tenant.

        Args:
            tenant_id: Tenant-ID

        Returns:
            Set von Agent-IDs
        """
        return self._tenant_agents.get(tenant_id, set())

    def check_agent_access(
        self, requesting_tenant_id: str, agent_metadata: AgentVersionMetadata
    ) -> bool:
        """Prüft ob Tenant Zugriff auf Agent hat.

        Args:
            requesting_tenant_id: Anfragender Tenant
            agent_metadata: Agent-Metadaten

        Returns:
            True wenn Zugriff erlaubt
        """
        return agent_metadata.is_accessible_by_tenant(requesting_tenant_id)

    @with_log_links(component="tenant_manager", operation="share_agent")
    def share_agent_with_tenant(
        self,
        owner_tenant_id: str,
        agent_id: str,
        target_tenant_id: str,
        metadata: AgentVersionMetadata,
    ) -> None:
        """Teilt Agent mit anderem Tenant.

        Args:
            owner_tenant_id: Besitzer-Tenant
            agent_id: Agent-ID
            target_tenant_id: Ziel-Tenant
            metadata: Agent-Metadaten

        Raises:
            TenantNotFoundError: Wenn Tenant nicht existiert
            AuthorizationError: Wenn keine Berechtigung
        """
        # Prüfe Tenant-Existenz
        self.get_tenant(owner_tenant_id)
        self.get_tenant(target_tenant_id)

        # Prüfe Berechtigung
        if metadata.tenant_id != owner_tenant_id:
            raise AuthorizationError(
                message=f"Tenant '{owner_tenant_id}' ist nicht Besitzer von Agent '{agent_id}'",
                tenant_id=owner_tenant_id,
                agent_id=agent_id,
            )

        # Teile Agent
        metadata.shared_with_tenants.add(target_tenant_id)
        metadata.access_level = TenantAccessLevel.SHARED
        metadata.updated_at = datetime.now(UTC)

        logger.info(
            f"Agent geteilt: {agent_id} von {owner_tenant_id} mit {target_tenant_id}",
            extra={
                "owner_tenant_id": owner_tenant_id,
                "target_tenant_id": target_tenant_id,
                "agent_id": agent_id,
                "shared_with_count": len(metadata.shared_with_tenants),
            },
        )

    def revoke_agent_sharing(
        self,
        owner_tenant_id: str,
        agent_id: str,
        target_tenant_id: str,
        metadata: AgentVersionMetadata,
    ) -> None:
        """Widerruft Agent-Sharing.

        Args:
            owner_tenant_id: Besitzer-Tenant
            agent_id: Agent-ID
            target_tenant_id: Ziel-Tenant
            metadata: Agent-Metadaten
        """
        metadata.shared_with_tenants.discard(target_tenant_id)

        # Wenn keine Tenants mehr geteilt, setze auf PRIVATE
        if not metadata.shared_with_tenants:
            metadata.access_level = TenantAccessLevel.PRIVATE

        metadata.updated_at = datetime.now(UTC)

        logger.info(
            f"Agent-Sharing widerrufen: {agent_id} von {owner_tenant_id} für {target_tenant_id}",
            extra={
                "owner_tenant_id": owner_tenant_id,
                "target_tenant_id": target_tenant_id,
                "agent_id": agent_id,
                "remaining_shared_count": len(metadata.shared_with_tenants),
            },
        )

    def check_quota_usage(self, tenant_id: str, quota_type: str) -> dict[str, Any]:
        """Prüft Quota-Nutzung für Tenant.

        Args:
            tenant_id: Tenant-ID
            quota_type: Quota-Typ

        Returns:
            Quota-Nutzungs-Informationen
        """
        tenant = self.get_tenant(tenant_id)
        quota_limit = tenant.resource_quotas.get(quota_type, 0)

        current_usage = 0

        if quota_type == "max_agents":
            current_usage = len(self._tenant_agents[tenant_id])
        elif quota_type == "max_versions_per_agent":
            # Placeholder - würde von Version Manager kommen
            current_usage = 0
        elif quota_type == "max_storage_mb":
            # Placeholder - würde von Storage Manager kommen
            current_usage = 0
        elif quota_type == "max_requests_per_hour":
            # Placeholder - würde von Metrics System kommen
            current_usage = 0
        elif quota_type == "max_concurrent_deployments":
            # Placeholder - würde von Deployment Manager kommen
            current_usage = 0

        usage_percentage = (current_usage / quota_limit * 100) if quota_limit > 0 else 0

        return {
            "quota_type": quota_type,
            "current_usage": current_usage,
            "quota_limit": quota_limit,
            "usage_percentage": usage_percentage,
            "quota_exceeded": current_usage >= quota_limit,
        }

    def get_tenant_statistics(self, tenant_id: str) -> dict[str, Any]:
        """Holt Statistiken für Tenant.

        Args:
            tenant_id: Tenant-ID

        Returns:
            Tenant-Statistiken
        """
        tenant = self.get_tenant(tenant_id)

        # Quota-Nutzung für alle Quota-Typen
        quota_usage = {}
        for quota_type in tenant.resource_quotas.keys():
            quota_usage[quota_type] = self.check_quota_usage(tenant_id, quota_type)

        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.tenant_name,
            "organization": tenant.organization,
            "access_level": tenant.access_level.value,
            "total_agents": len(self._tenant_agents[tenant_id]),
            "quota_usage": quota_usage,
            "created_at": tenant.created_at.isoformat(),
            "updated_at": tenant.updated_at.isoformat(),
            "tags": tenant.tags,
        }

    def update_tenant_quotas(self, tenant_id: str, new_quotas: dict[str, int]) -> None:
        """Aktualisiert Tenant-Quotas.

        Args:
            tenant_id: Tenant-ID
            new_quotas: Neue Quota-Werte
        """
        tenant = self.get_tenant(tenant_id)

        old_quotas = tenant.resource_quotas.copy()
        tenant.resource_quotas.update(new_quotas)
        tenant.updated_at = datetime.now(UTC)

        logger.info(
            f"Tenant-Quotas aktualisiert: {tenant_id}",
            extra={
                "tenant_id": tenant_id,
                "old_quotas": old_quotas,
                "new_quotas": new_quotas,
                "updated_quotas": tenant.resource_quotas,
            },
        )

    def set_cross_tenant_policy(
        self, source_tenant_id: str, target_tenant_id: str, policy: dict[str, Any]
    ) -> None:
        """Setzt Cross-Tenant-Policy.

        Args:
            source_tenant_id: Quell-Tenant
            target_tenant_id: Ziel-Tenant
            policy: Policy-Konfiguration
        """
        policy_key = f"{source_tenant_id}->{target_tenant_id}"
        self._cross_tenant_policies[policy_key] = policy

        logger.info(
            f"Cross-Tenant-Policy gesetzt: {source_tenant_id} -> {target_tenant_id}",
            extra={
                "source_tenant_id": source_tenant_id,
                "target_tenant_id": target_tenant_id,
                "policy": policy,
            },
        )

    def get_cross_tenant_policy(
        self, source_tenant_id: str, target_tenant_id: str
    ) -> dict[str, Any] | None:
        """Holt Cross-Tenant-Policy.

        Args:
            source_tenant_id: Quell-Tenant
            target_tenant_id: Ziel-Tenant

        Returns:
            Policy-Konfiguration oder None
        """
        policy_key = f"{source_tenant_id}->{target_tenant_id}"
        return self._cross_tenant_policies.get(policy_key)

    def cleanup_inactive_tenants(self, max_inactive_days: int = 365) -> int:
        """Entfernt inaktive Tenants.

        Args:
            max_inactive_days: Maximale Inaktivitäts-Tage

        Returns:
            Anzahl entfernter Tenants
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=max_inactive_days)
        removed_count = 0

        tenants_to_remove = []
        for tenant_id, tenant in self._tenants.items():
            # Prüfe ob Tenant inaktiv ist (keine Agents und alte updated_at)
            if not self._tenant_agents[tenant_id] and tenant.updated_at < cutoff_date:
                tenants_to_remove.append(tenant_id)

        for tenant_id in tenants_to_remove:
            del self._tenants[tenant_id]
            if tenant_id in self._tenant_agents:
                del self._tenant_agents[tenant_id]
            removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleanup: {removed_count} inaktive Tenants entfernt")

        return removed_count

    def has_tenant(self, tenant_id: str) -> bool:
        """Public API für Tenant-Existenz-Prüfung.

        Args:
            tenant_id: Tenant-ID

        Returns:
            True wenn Tenant existiert
        """
        return tenant_id in self._tenants

    def get_tenant_ids(self) -> list[str]:
        """Public API für alle Tenant-IDs.

        Returns:
            Liste aller Tenant-IDs
        """
        return list(self._tenants.keys())


# Globale Tenant Manager Instanz
tenant_manager = TenantManager()
