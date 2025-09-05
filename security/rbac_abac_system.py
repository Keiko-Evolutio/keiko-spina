# backend/security/rbac_abac_system.py
"""RBAC/ABAC-System für Keiko Personal Assistant

Implementiert Role-Based Access Control (RBAC) mit hierarchischen Rollen
und Attribute-Based Access Control (ABAC) für granulare Agent-Berechtigungen.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class ResourceType(str, Enum):
    """Ressourcen-Typen für Authorization."""
    AGENT = "agent"
    CAPABILITY = "capability"
    TASK = "task"
    EVENT = "event"
    LIFECYCLE = "lifecycle"
    DISCOVERY = "discovery"
    RPC = "rpc"
    TENANT = "tenant"
    CONFIGURATION = "configuration"
    AUDIT = "audit"


class Action(str, Enum):
    """Aktionen für Authorization."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    MONITOR = "monitor"
    CONFIGURE = "configure"
    IMPERSONATE = "impersonate"
    DELEGATE = "delegate"


class PermissionEffect(str, Enum):
    """Effekt einer Permission-Regel."""
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class Permission:
    """Einzelne Berechtigung."""
    resource_type: ResourceType
    action: Action
    effect: PermissionEffect = PermissionEffect.ALLOW
    resource_pattern: str = "*"  # Glob-Pattern für Ressourcen-IDs
    conditions: dict[str, Any] = field(default_factory=dict)

    def matches_resource(self, resource_id: str) -> bool:
        """Prüft, ob Permission auf Ressource zutrifft."""
        import fnmatch
        return fnmatch.fnmatch(resource_id, self.resource_pattern)


@dataclass
class Role:
    """Rolle mit Berechtigungen."""
    name: str
    description: str
    permissions: list[Permission] = field(default_factory=list)
    parent_roles: set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_all_permissions(self, role_registry: RoleRegistry) -> list[Permission]:
        """Gibt alle Berechtigungen inklusive vererbte zurück."""
        all_permissions = self.permissions.copy()

        # Berechtigungen von Parent-Rollen hinzufügen
        for parent_role_name in self.parent_roles:
            parent_role = role_registry.get_role(parent_role_name)
            if parent_role:
                all_permissions.extend(parent_role.get_all_permissions(role_registry))

        return all_permissions


@dataclass
class Principal:
    """Authentifizierter Principal (User, Service Account, Agent)."""
    id: str
    type: str  # "user", "service_account", "agent"
    roles: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    scopes: set[str] = field(default_factory=set)

    def has_role(self, role_name: str) -> bool:
        """Prüft, ob Principal eine Rolle hat."""
        return role_name in self.roles

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Gibt Attribut-Wert zurück."""
        return self.attributes.get(key, default)


@dataclass
class AuthorizationContext:
    """Kontext für Authorization-Entscheidungen."""
    principal: Principal
    resource_type: ResourceType
    resource_id: str
    action: Action
    environment: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_environment_value(self, key: str, default: Any = None) -> Any:
        """Gibt Umgebungs-Wert zurück."""
        return self.environment.get(key, default)


@dataclass
class AuthorizationDecision:
    """Ergebnis einer Authorization-Entscheidung."""
    effect: PermissionEffect
    reason: str
    matched_permissions: list[Permission] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    context: AuthorizationContext | None = None


class AttributeEvaluator(ABC):
    """Basis-Klasse für Attribut-Evaluatoren."""

    @abstractmethod
    def evaluate(self, context: AuthorizationContext) -> bool:
        """Evaluiert Attribut-Bedingung."""


class TimeBasedEvaluator(AttributeEvaluator):
    """Zeitbasierte Attribut-Evaluation."""

    def __init__(self, allowed_hours: list[int]):
        """Initialisiert mit erlaubten Stunden (0-23)."""
        self.allowed_hours = set(allowed_hours)

    def evaluate(self, context: AuthorizationContext) -> bool:
        """Prüft, ob aktuelle Zeit erlaubt ist."""
        current_hour = context.timestamp.hour
        return current_hour in self.allowed_hours


class TenantBasedEvaluator(AttributeEvaluator):
    """Tenant-basierte Attribut-Evaluation."""

    def evaluate(self, context: AuthorizationContext) -> bool:
        """Prüft Tenant-Zugehörigkeit."""
        principal_tenant = context.principal.tenant_id
        resource_tenant = context.get_environment_value("tenant_id")

        # Kein Tenant-Check wenn keine Tenant-Info vorhanden
        if not principal_tenant or not resource_tenant:
            return True

        return principal_tenant == resource_tenant


class ScopeBasedEvaluator(AttributeEvaluator):
    """Scope-basierte Attribut-Evaluation."""

    def __init__(self, required_scopes: set[str]):
        """Initialisiert mit erforderlichen Scopes."""
        self.required_scopes = required_scopes

    def evaluate(self, context: AuthorizationContext) -> bool:
        """Prüft, ob Principal erforderliche Scopes hat."""
        return self.required_scopes.issubset(context.principal.scopes)


class RoleRegistry:
    """Registry für Rollen-Management."""

    def __init__(self) -> None:
        """Initialisiert Role Registry."""
        self._roles: dict[str, Role] = {}
        self._role_hierarchy_cache: dict[str, set[str]] = {}
        self._cache_ttl = 300  # 5 Minuten
        self._cache_updated_at: datetime | None = None

        # Standard-Rollen erstellen
        self._create_default_roles()

    def _create_default_roles(self) -> None:
        """Erstellt Standard-Rollen für das System."""
        # Super Admin
        super_admin = Role(
            name="super_admin",
            description="Vollzugriff auf alle Ressourcen",
            permissions=[
                Permission(ResourceType.AGENT, Action.MANAGE),
                Permission(ResourceType.CAPABILITY, Action.MANAGE),
                Permission(ResourceType.TASK, Action.MANAGE),
                Permission(ResourceType.EVENT, Action.MANAGE),
                Permission(ResourceType.LIFECYCLE, Action.MANAGE),
                Permission(ResourceType.DISCOVERY, Action.MANAGE),
                Permission(ResourceType.RPC, Action.MANAGE),
                Permission(ResourceType.TENANT, Action.MANAGE),
                Permission(ResourceType.CONFIGURATION, Action.MANAGE),
                Permission(ResourceType.AUDIT, Action.READ),
            ],
            is_system_role=True
        )

        # Agent Admin
        agent_admin = Role(
            name="agent_admin",
            description="Vollzugriff auf Agent-Management",
            permissions=[
                Permission(ResourceType.AGENT, Action.MANAGE),
                Permission(ResourceType.CAPABILITY, Action.MANAGE),
                Permission(ResourceType.LIFECYCLE, Action.MANAGE),
                Permission(ResourceType.TASK, Action.EXECUTE),
                Permission(ResourceType.EVENT, Action.READ),
                Permission(ResourceType.DISCOVERY, Action.READ),
                Permission(ResourceType.RPC, Action.EXECUTE),
            ],
            is_system_role=True
        )

        # Agent Operator
        agent_operator = Role(
            name="agent_operator",
            description="Operationeller Zugriff auf Agents",
            permissions=[
                Permission(ResourceType.AGENT, Action.READ),
                Permission(ResourceType.AGENT, Action.EXECUTE),
                Permission(ResourceType.CAPABILITY, Action.READ),
                Permission(ResourceType.TASK, Action.EXECUTE),
                Permission(ResourceType.EVENT, Action.READ),
                Permission(ResourceType.LIFECYCLE, Action.READ),
                Permission(ResourceType.DISCOVERY, Action.READ),
                Permission(ResourceType.RPC, Action.EXECUTE),
            ],
            parent_roles=set(),
            is_system_role=True
        )

        # Agent User
        agent_user = Role(
            name="agent_user",
            description="Basis-Zugriff für Agent-Nutzung",
            permissions=[
                Permission(ResourceType.AGENT, Action.READ),
                Permission(ResourceType.CAPABILITY, Action.READ),
                Permission(ResourceType.TASK, Action.EXECUTE, resource_pattern="user:*"),
                Permission(ResourceType.EVENT, Action.READ, resource_pattern="user:*"),
                Permission(ResourceType.DISCOVERY, Action.READ),
            ],
            is_system_role=True
        )

        # Service Account
        service_account = Role(
            name="service_account",
            description="Rolle für Service Accounts",
            permissions=[
                Permission(ResourceType.AGENT, Action.EXECUTE),
                Permission(ResourceType.CAPABILITY, Action.EXECUTE),
                Permission(ResourceType.TASK, Action.EXECUTE),
                Permission(ResourceType.RPC, Action.EXECUTE),
                Permission(ResourceType.EVENT, Action.CREATE),
            ],
            is_system_role=True
        )

        # Rollen registrieren
        for role in [super_admin, agent_admin, agent_operator, agent_user, service_account]:
            self._roles[role.name] = role

    def register_role(self, role: Role) -> None:
        """Registriert neue Rolle."""
        self._roles[role.name] = role
        self._invalidate_cache()
        logger.info(f"Rolle {role.name} registriert")

    def get_role(self, role_name: str) -> Role | None:
        """Gibt Rolle zurück."""
        return self._roles.get(role_name)

    def get_all_roles(self) -> list[Role]:
        """Gibt alle Rollen zurück."""
        return list(self._roles.values())

    def get_role_hierarchy(self, role_name: str) -> set[str]:
        """Gibt alle Rollen in der Hierarchie zurück (inklusive vererbte)."""
        if self._is_cache_valid() and role_name in self._role_hierarchy_cache:
            return self._role_hierarchy_cache[role_name]

        hierarchy = self._compute_role_hierarchy(role_name)
        self._role_hierarchy_cache[role_name] = hierarchy
        return hierarchy

    def _compute_role_hierarchy(self, role_name: str, visited: set[str] | None = None) -> set[str]:
        """Berechnet Rollen-Hierarchie rekursiv."""
        if visited is None:
            visited = set()

        if role_name in visited:
            # Zirkuläre Abhängigkeit erkannt
            logger.warning(f"Zirkuläre Rollen-Abhängigkeit erkannt: {role_name}")
            return {role_name}

        visited.add(role_name)
        hierarchy = {role_name}

        role = self.get_role(role_name)
        if role:
            for parent_role_name in role.parent_roles:
                parent_hierarchy = self._compute_role_hierarchy(parent_role_name, visited.copy())
                hierarchy.update(parent_hierarchy)

        return hierarchy

    def _is_cache_valid(self) -> bool:
        """Prüft, ob Cache gültig ist."""
        if not self._cache_updated_at:
            return False

        age = (datetime.now(UTC) - self._cache_updated_at).total_seconds()
        return age < self._cache_ttl

    def _invalidate_cache(self) -> None:
        """Invalidiert Cache."""
        self._role_hierarchy_cache.clear()
        self._cache_updated_at = datetime.now(UTC)


class PolicyEngine:
    """Policy-Engine für ABAC-Evaluation."""

    def __init__(self) -> None:
        """Initialisiert Policy Engine."""
        self._evaluators: dict[str, AttributeEvaluator] = {}
        self._register_default_evaluators()

    def _register_default_evaluators(self) -> None:
        """Registriert Standard-Evaluatoren."""
        self._evaluators["time_based"] = TimeBasedEvaluator(list(range(24)))  # 24/7
        self._evaluators["tenant_based"] = TenantBasedEvaluator()

    def register_evaluator(self, name: str, evaluator: AttributeEvaluator) -> None:
        """Registriert Attribut-Evaluator."""
        self._evaluators[name] = evaluator
        logger.info(f"Attribut-Evaluator {name} registriert")

    def evaluate_conditions(self, conditions: dict[str, Any], context: AuthorizationContext) -> bool:
        """Evaluiert Bedingungen einer Permission."""
        for condition_name, condition_value in conditions.items():
            evaluator = self._evaluators.get(condition_name)
            if evaluator:
                if not evaluator.evaluate(context):
                    return False
            # Einfache Attribut-Vergleiche
            elif not self._evaluate_simple_condition(condition_name, condition_value, context):
                return False

        return True

    def _evaluate_simple_condition(
        self,
        condition_name: str,
        condition_value: Any,
        context: AuthorizationContext
    ) -> bool:
        """Evaluiert einfache Attribut-Bedingung."""
        # Principal-Attribute
        if condition_name.startswith("principal."):
            attr_name = condition_name[10:]
            principal_value = context.principal.get_attribute(attr_name)
            return principal_value == condition_value

        # Umgebungs-Attribute
        if condition_name.startswith("environment."):
            attr_name = condition_name[12:]
            env_value = context.get_environment_value(attr_name)
            return env_value == condition_value

        # Ressourcen-Attribute
        if condition_name.startswith("resource."):
            attr_name = condition_name[9:]
            if attr_name == "id":
                return context.resource_id == condition_value
            if attr_name == "type":
                return context.resource_type.value == condition_value

        return False


class RBACAuthorizationService:
    """RBAC/ABAC Authorization Service."""

    def __init__(self) -> None:
        """Initialisiert Authorization Service."""
        self.role_registry = RoleRegistry()
        self.policy_engine = PolicyEngine()
        self._audit_enabled = True
        self._audit_log: list[dict[str, Any]] = []
        self._max_audit_entries = 10000

    @trace_function("rbac.authorize")
    def authorize(self, context: AuthorizationContext) -> AuthorizationDecision:
        """Führt Authorization-Entscheidung durch.

        Args:
            context: Authorization-Kontext

        Returns:
            Authorization-Entscheidung
        """
        start_time = time.time()

        try:
            # Sammle alle Berechtigungen des Principals
            all_permissions = self._get_principal_permissions(context.principal)

            # Finde passende Berechtigungen
            matching_permissions = []
            for permission in all_permissions:
                if (permission.resource_type == context.resource_type and
                    permission.action == context.action and
                    permission.matches_resource(context.resource_id)):

                    # Evaluiere Bedingungen
                    if self.policy_engine.evaluate_conditions(permission.conditions, context):
                        matching_permissions.append(permission)

            # Entscheidung treffen (DENY hat Vorrang)
            effect = PermissionEffect.DENY
            reason = "Keine passende Berechtigung gefunden"

            if matching_permissions:
                # Prüfe auf explizite DENY-Regeln
                deny_permissions = [p for p in matching_permissions if p.effect == PermissionEffect.DENY]
                if deny_permissions:
                    effect = PermissionEffect.DENY
                    reason = f"Explizit verweigert durch {len(deny_permissions)} DENY-Regel(n)"
                else:
                    # Alle passenden Permissions sind ALLOW
                    effect = PermissionEffect.ALLOW
                    reason = f"Erlaubt durch {len(matching_permissions)} ALLOW-Regel(n)"

            execution_time = (time.time() - start_time) * 1000

            decision = AuthorizationDecision(
                effect=effect,
                reason=reason,
                matched_permissions=matching_permissions,
                evaluation_time_ms=execution_time,
                context=context
            )

            # Audit-Log
            if self._audit_enabled:
                self._log_authorization_decision(decision)

            return decision

        except Exception as e:
            logger.exception(f"Authorization-Fehler: {e}")
            execution_time = (time.time() - start_time) * 1000

            return AuthorizationDecision(
                effect=PermissionEffect.DENY,
                reason=f"Authorization-Fehler: {e!s}",
                evaluation_time_ms=execution_time,
                context=context
            )

    def _get_principal_permissions(self, principal: Principal) -> list[Permission]:
        """Sammelt alle Berechtigungen eines Principals."""
        all_permissions = []

        for role_name in principal.roles:
            # Hole alle Rollen in der Hierarchie
            role_hierarchy = self.role_registry.get_role_hierarchy(role_name)

            for hierarchical_role_name in role_hierarchy:
                role = self.role_registry.get_role(hierarchical_role_name)
                if role:
                    all_permissions.extend(role.permissions)

        return all_permissions

    def _log_authorization_decision(self, decision: AuthorizationDecision) -> None:
        """Loggt Authorization-Entscheidung für Audit."""
        if not decision.context:
            return

        audit_entry = {
            "timestamp": decision.context.timestamp.isoformat(),
            "principal_id": decision.context.principal.id,
            "principal_type": decision.context.principal.type,
            "resource_type": decision.context.resource_type.value,
            "resource_id": decision.context.resource_id,
            "action": decision.context.action.value,
            "effect": decision.effect.value,
            "reason": decision.reason,
            "evaluation_time_ms": decision.evaluation_time_ms,
            "matched_permissions_count": len(decision.matched_permissions)
        }

        self._audit_log.append(audit_entry)

        # Begrenze Audit-Log-Größe
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

    def get_audit_log(
        self,
        principal_id: str | None = None,
        resource_type: ResourceType | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Gibt Audit-Log zurück."""
        filtered_log = self._audit_log

        # Filter anwenden
        if principal_id:
            filtered_log = [entry for entry in filtered_log if entry["principal_id"] == principal_id]

        if resource_type:
            filtered_log = [entry for entry in filtered_log if entry["resource_type"] == resource_type.value]

        # Limitieren und neueste zuerst
        return list(reversed(filtered_log))[-limit:]

    def get_authorization_stats(self) -> dict[str, Any]:
        """Gibt Authorization-Statistiken zurück."""
        total_decisions = len(self._audit_log)
        allowed_decisions = sum(1 for entry in self._audit_log if entry["effect"] == "allow")
        denied_decisions = total_decisions - allowed_decisions

        avg_evaluation_time = 0.0
        if self._audit_log:
            avg_evaluation_time = sum(entry["evaluation_time_ms"] for entry in self._audit_log) / total_decisions

        return {
            "total_decisions": total_decisions,
            "allowed_decisions": allowed_decisions,
            "denied_decisions": denied_decisions,
            "allow_rate": allowed_decisions / max(total_decisions, 1),
            "avg_evaluation_time_ms": avg_evaluation_time,
            "total_roles": len(self.role_registry.get_all_roles()),
            "audit_enabled": self._audit_enabled
        }


# Globaler Authorization Service
rbac_authorization_service = RBACAuthorizationService()
