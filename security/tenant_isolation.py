# backend/security/tenant_isolation.py
"""Multi-Tenant-Isolation für Keiko Personal Assistant

Implementiert vollständige Tenant-Isolation mit Datenisolation,
Tenant-spezifischen Konfigurationen und Cross-Tenant-Access-Controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class TenantStatus(str, Enum):
    """Status eines Tenants."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    PENDING = "pending"


class IsolationLevel(str, Enum):
    """Isolations-Level zwischen Tenants."""
    STRICT = "strict"          # Vollständige Isolation
    CONTROLLED = "controlled"  # Kontrollierte Cross-Tenant-Zugriffe
    SHARED = "shared"         # Geteilte Ressourcen erlaubt


@dataclass
class TenantConfig:
    """Konfiguration für einen Tenant."""
    tenant_id: str
    name: str
    description: str = ""
    status: TenantStatus = TenantStatus.ACTIVE
    isolation_level: IsolationLevel = IsolationLevel.STRICT

    # Ressourcen-Limits
    max_agents: int = 100
    max_concurrent_tasks: int = 1000
    max_storage_mb: int = 10240  # 10GB
    max_api_calls_per_hour: int = 10000

    # Konfigurationen
    allowed_capabilities: set[str] = field(default_factory=set)
    blocked_capabilities: set[str] = field(default_factory=set)
    custom_settings: dict[str, Any] = field(default_factory=dict)

    # Cross-Tenant-Zugriffe
    allowed_cross_tenant_access: set[str] = field(default_factory=set)
    trusted_tenants: set[str] = field(default_factory=set)

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_timestamp(self) -> None:
        """Aktualisiert Update-Timestamp."""
        self.updated_at = datetime.now(UTC)

    def is_capability_allowed(self, capability: str) -> bool:
        """Prüft, ob Capability für Tenant erlaubt ist."""
        if self.blocked_capabilities and capability in self.blocked_capabilities:
            return False

        if self.allowed_capabilities:
            return capability in self.allowed_capabilities

        return True  # Standardmäßig erlaubt wenn keine Einschränkungen

    def can_access_tenant(self, target_tenant_id: str) -> bool:
        """Prüft, ob Cross-Tenant-Zugriff erlaubt ist."""
        if self.isolation_level == IsolationLevel.STRICT:
            return False

        if self.isolation_level == IsolationLevel.SHARED:
            return True

        # CONTROLLED: Nur explizit erlaubte Tenants
        return target_tenant_id in self.allowed_cross_tenant_access


@dataclass
class TenantResource:
    """Tenant-spezifische Ressource."""
    resource_id: str
    tenant_id: str
    resource_type: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_accessible_by_tenant(self, requesting_tenant_id: str, tenant_registry: TenantRegistry) -> bool:
        """Prüft, ob Ressource von anderem Tenant zugänglich ist."""
        if self.tenant_id == requesting_tenant_id:
            return True

        # Prüfe Cross-Tenant-Zugriff
        requesting_tenant = tenant_registry.get_tenant(requesting_tenant_id)
        if requesting_tenant:
            return requesting_tenant.can_access_tenant(self.tenant_id)

        return False


@dataclass
class TenantUsageStats:
    """Nutzungsstatistiken für einen Tenant."""
    tenant_id: str

    # Agent-Statistiken
    active_agents: int = 0
    total_agents_created: int = 0

    # Task-Statistiken
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    # Ressourcen-Nutzung
    storage_used_mb: float = 0.0
    api_calls_last_hour: int = 0

    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_stats(self, **kwargs) -> None:
        """Aktualisiert Statistiken."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now(UTC)


class TenantRegistry:
    """Registry für Tenant-Management."""

    def __init__(self) -> None:
        """Initialisiert Tenant Registry."""
        self._tenants: dict[str, TenantConfig] = {}
        self._tenant_resources: dict[str, dict[str, TenantResource]] = {}
        self._usage_stats: dict[str, TenantUsageStats] = {}
        self._default_tenant_id = "default"

        # Erstelle Default-Tenant
        self._create_default_tenant()

    def _create_default_tenant(self) -> None:
        """Erstellt Default-Tenant."""
        default_tenant = TenantConfig(
            tenant_id=self._default_tenant_id,
            name="Default Tenant",
            description="Standard-Tenant für Single-Tenant-Deployments",
            isolation_level=IsolationLevel.SHARED,
            max_agents=1000,
            max_concurrent_tasks=10000,
            max_storage_mb=102400,  # 100GB
            max_api_calls_per_hour=100000
        )

        self._tenants[self._default_tenant_id] = default_tenant
        self._tenant_resources[self._default_tenant_id] = {}
        self._usage_stats[self._default_tenant_id] = TenantUsageStats(self._default_tenant_id)

    def register_tenant(self, tenant_config: TenantConfig) -> bool:
        """Registriert neuen Tenant.

        Args:
            tenant_config: Tenant-Konfiguration

        Returns:
            True wenn erfolgreich registriert
        """
        if tenant_config.tenant_id in self._tenants:
            logger.warning(f"Tenant {tenant_config.tenant_id} bereits registriert")
            return False

        self._tenants[tenant_config.tenant_id] = tenant_config
        self._tenant_resources[tenant_config.tenant_id] = {}
        self._usage_stats[tenant_config.tenant_id] = TenantUsageStats(tenant_config.tenant_id)

        logger.info(f"Tenant {tenant_config.tenant_id} erfolgreich registriert")
        return True

    def get_tenant(self, tenant_id: str) -> TenantConfig | None:
        """Gibt Tenant-Konfiguration zurück."""
        return self._tenants.get(tenant_id)

    def get_all_tenants(self) -> list[TenantConfig]:
        """Gibt alle Tenants zurück."""
        return list(self._tenants.values())

    def get_active_tenants(self) -> list[TenantConfig]:
        """Gibt alle aktiven Tenants zurück."""
        return [tenant for tenant in self._tenants.values() if tenant.status == TenantStatus.ACTIVE]

    def update_tenant(self, tenant_id: str, updates: dict[str, Any]) -> bool:
        """Aktualisiert Tenant-Konfiguration.

        Args:
            tenant_id: Tenant-ID
            updates: Zu aktualisierende Felder

        Returns:
            True wenn erfolgreich aktualisiert
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        tenant.update_timestamp()
        logger.info(f"Tenant {tenant_id} aktualisiert")
        return True

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deaktiviert Tenant.

        Args:
            tenant_id: Tenant-ID

        Returns:
            True wenn erfolgreich deaktiviert
        """
        if tenant_id == self._default_tenant_id:
            logger.error("Default-Tenant kann nicht deaktiviert werden")
            return False

        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.status = TenantStatus.DEACTIVATED
            tenant.update_timestamp()
            logger.info(f"Tenant {tenant_id} deaktiviert")
            return True

        return False

    def add_tenant_resource(self, tenant_id: str, resource: TenantResource) -> bool:
        """Fügt Tenant-Ressource hinzu.

        Args:
            tenant_id: Tenant-ID
            resource: Tenant-Ressource

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        if tenant_id not in self._tenants:
            return False

        if tenant_id not in self._tenant_resources:
            self._tenant_resources[tenant_id] = {}

        self._tenant_resources[tenant_id][resource.resource_id] = resource
        return True

    def get_tenant_resource(self, tenant_id: str, resource_id: str) -> TenantResource | None:
        """Gibt Tenant-Ressource zurück."""
        tenant_resources = self._tenant_resources.get(tenant_id, {})
        return tenant_resources.get(resource_id)

    def get_tenant_resources(self, tenant_id: str, resource_type: str | None = None) -> list[TenantResource]:
        """Gibt alle Ressourcen eines Tenants zurück."""
        tenant_resources = self._tenant_resources.get(tenant_id, {})
        resources = list(tenant_resources.values())

        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]

        return resources

    def remove_tenant_resource(self, tenant_id: str, resource_id: str) -> bool:
        """Entfernt Tenant-Ressource."""
        tenant_resources = self._tenant_resources.get(tenant_id, {})
        if resource_id in tenant_resources:
            del tenant_resources[resource_id]
            return True
        return False

    def get_usage_stats(self, tenant_id: str) -> TenantUsageStats | None:
        """Gibt Nutzungsstatistiken zurück."""
        return self._usage_stats.get(tenant_id)

    def update_usage_stats(self, tenant_id: str, **kwargs) -> None:
        """Aktualisiert Nutzungsstatistiken."""
        stats = self._usage_stats.get(tenant_id)
        if stats:
            stats.update_stats(**kwargs)


class TenantIsolationService:
    """Service für Tenant-Isolation."""

    def __init__(self) -> None:
        """Initialisiert Tenant Isolation Service."""
        self.tenant_registry = TenantRegistry()
        self._isolation_policies: dict[str, Callable] = {}
        self._cross_tenant_audit_log: list[dict[str, Any]] = []
        self._max_audit_entries = 10000

    @trace_function("tenant.validate_access")
    def validate_tenant_access(
        self,
        requesting_tenant_id: str,
        target_tenant_id: str,
        resource_type: str,
        action: str
    ) -> bool:
        """Validiert Cross-Tenant-Zugriff.

        Args:
            requesting_tenant_id: Anfragender Tenant
            target_tenant_id: Ziel-Tenant
            resource_type: Ressourcen-Typ
            action: Aktion

        Returns:
            True wenn Zugriff erlaubt
        """
        # Gleicher Tenant ist immer erlaubt
        if requesting_tenant_id == target_tenant_id:
            return True

        requesting_tenant = self.tenant_registry.get_tenant(requesting_tenant_id)
        target_tenant = self.tenant_registry.get_tenant(target_tenant_id)

        if not requesting_tenant or not target_tenant:
            self._log_cross_tenant_access(
                requesting_tenant_id, target_tenant_id, resource_type, action, False, "Tenant nicht gefunden"
            )
            return False

        # Prüfe Tenant-Status
        if (requesting_tenant.status != TenantStatus.ACTIVE or
            target_tenant.status != TenantStatus.ACTIVE):
            self._log_cross_tenant_access(
                requesting_tenant_id, target_tenant_id, resource_type, action, False, "Tenant nicht aktiv"
            )
            return False

        # Prüfe Isolation-Level
        access_allowed = requesting_tenant.can_access_tenant(target_tenant_id)

        # Prüfe spezielle Isolation-Policies
        policy_key = f"{resource_type}:{action}"
        if policy_key in self._isolation_policies:
            policy_result = self._isolation_policies[policy_key](
                requesting_tenant, target_tenant, resource_type, action
            )
            access_allowed = access_allowed and policy_result

        reason = "Zugriff erlaubt" if access_allowed else "Zugriff verweigert durch Isolation-Policy"
        self._log_cross_tenant_access(
            requesting_tenant_id, target_tenant_id, resource_type, action, access_allowed, reason
        )

        return access_allowed

    def register_isolation_policy(
        self,
        resource_type: str,
        action: str,
        policy_func: Callable[[TenantConfig, TenantConfig, str, str], bool]
    ) -> None:
        """Registriert Isolation-Policy.

        Args:
            resource_type: Ressourcen-Typ
            action: Aktion
            policy_func: Policy-Funktion
        """
        policy_key = f"{resource_type}:{action}"
        self._isolation_policies[policy_key] = policy_func
        logger.info(f"Isolation-Policy für {policy_key} registriert")

    def _log_cross_tenant_access(
        self,
        requesting_tenant_id: str,
        target_tenant_id: str,
        resource_type: str,
        action: str,
        allowed: bool,
        reason: str
    ) -> None:
        """Loggt Cross-Tenant-Zugriff für Audit."""
        audit_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "requesting_tenant_id": requesting_tenant_id,
            "target_tenant_id": target_tenant_id,
            "resource_type": resource_type,
            "action": action,
            "allowed": allowed,
            "reason": reason
        }

        self._cross_tenant_audit_log.append(audit_entry)

        # Begrenze Audit-Log-Größe
        if len(self._cross_tenant_audit_log) > self._max_audit_entries:
            self._cross_tenant_audit_log = self._cross_tenant_audit_log[-self._max_audit_entries:]

    def get_cross_tenant_audit_log(
        self,
        requesting_tenant_id: str | None = None,
        target_tenant_id: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Gibt Cross-Tenant-Audit-Log zurück."""
        filtered_log = self._cross_tenant_audit_log

        if requesting_tenant_id:
            filtered_log = [
                entry for entry in filtered_log
                if entry["requesting_tenant_id"] == requesting_tenant_id
            ]

        if target_tenant_id:
            filtered_log = [
                entry for entry in filtered_log
                if entry["target_tenant_id"] == target_tenant_id
            ]

        return list(reversed(filtered_log))[-limit:]

    def create_tenant_context(self, tenant_id: str) -> TenantContext:
        """Erstellt Tenant-Kontext für Request-Verarbeitung.

        Args:
            tenant_id: Tenant-ID

        Returns:
            Tenant-Kontext
        """
        tenant = self.tenant_registry.get_tenant(tenant_id)
        if not tenant:
            # Fallback auf Default-Tenant
            tenant = self.tenant_registry.get_tenant(self.tenant_registry._default_tenant_id)

        return TenantContext(tenant, self)

    def get_tenant_stats(self) -> dict[str, Any]:
        """Gibt Tenant-Statistiken zurück."""
        all_tenants = self.tenant_registry.get_all_tenants()
        active_tenants = self.tenant_registry.get_active_tenants()

        total_resources = sum(
            len(resources) for resources in self.tenant_registry._tenant_resources.values()
        )

        cross_tenant_accesses = len(self._cross_tenant_audit_log)
        allowed_accesses = sum(1 for entry in self._cross_tenant_audit_log if entry["allowed"])

        return {
            "total_tenants": len(all_tenants),
            "active_tenants": len(active_tenants),
            "total_resources": total_resources,
            "cross_tenant_accesses": cross_tenant_accesses,
            "cross_tenant_success_rate": allowed_accesses / max(cross_tenant_accesses, 1),
            "isolation_policies": len(self._isolation_policies)
        }


class TenantContext:
    """Kontext für Tenant-spezifische Operationen."""

    def __init__(self, tenant_config: TenantConfig, isolation_service: TenantIsolationService):
        """Initialisiert Tenant-Kontext.

        Args:
            tenant_config: Tenant-Konfiguration
            isolation_service: Isolation-Service
        """
        self.tenant_config = tenant_config
        self.isolation_service = isolation_service

    @property
    def tenant_id(self) -> str:
        """Gibt Tenant-ID zurück."""
        return self.tenant_config.tenant_id

    def can_access_tenant(self, target_tenant_id: str, resource_type: str, action: str) -> bool:
        """Prüft Cross-Tenant-Zugriff."""
        return self.isolation_service.validate_tenant_access(
            self.tenant_id, target_tenant_id, resource_type, action
        )

    def is_capability_allowed(self, capability: str) -> bool:
        """Prüft, ob Capability erlaubt ist."""
        return self.tenant_config.is_capability_allowed(capability)

    def get_resource_limit(self, resource_type: str) -> int | None:
        """Gibt Ressourcen-Limit zurück."""
        limit_mapping = {
            "agents": self.tenant_config.max_agents,
            "tasks": self.tenant_config.max_concurrent_tasks,
            "storage_mb": self.tenant_config.max_storage_mb,
            "api_calls_per_hour": self.tenant_config.max_api_calls_per_hour
        }
        return limit_mapping.get(resource_type)

    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Gibt Tenant-spezifische Einstellung zurück."""
        return self.tenant_config.custom_settings.get(key, default)

    def add_resource(self, resource_id: str, resource_type: str, data: dict[str, Any]) -> bool:
        """Fügt Tenant-Ressource hinzu."""
        resource = TenantResource(
            resource_id=resource_id,
            tenant_id=self.tenant_id,
            resource_type=resource_type,
            data=data
        )

        return self.isolation_service.tenant_registry.add_tenant_resource(self.tenant_id, resource)

    def get_resource(self, resource_id: str) -> TenantResource | None:
        """Gibt Tenant-Ressource zurück."""
        return self.isolation_service.tenant_registry.get_tenant_resource(self.tenant_id, resource_id)

    def get_resources(self, resource_type: str | None = None) -> list[TenantResource]:
        """Gibt alle Tenant-Ressourcen zurück."""
        return self.isolation_service.tenant_registry.get_tenant_resources(self.tenant_id, resource_type)

    def update_usage_stats(self, **kwargs) -> None:
        """Aktualisiert Nutzungsstatistiken."""
        self.isolation_service.tenant_registry.update_usage_stats(self.tenant_id, **kwargs)

    def get_usage_stats(self) -> TenantUsageStats | None:
        """Gibt Nutzungsstatistiken zurück."""
        return self.isolation_service.tenant_registry.get_usage_stats(self.tenant_id)


# Globaler Tenant Isolation Service
tenant_isolation_service = TenantIsolationService()
