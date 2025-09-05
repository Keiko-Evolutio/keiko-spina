# backend/security/scope_permission_manager.py
"""Scope- und Permission-Management für Keiko Personal Assistant

Implementiert granulare Scopes für Agent-Capabilities, dynamische Permission-Evaluation,
Permission-Inheritance und Audit-Logging für alle Authorization-Entscheidungen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class ScopeType(str, Enum):
    """Typen von Scopes."""
    AGENT_CAPABILITY = "agent_capability"
    LIFECYCLE_OPERATION = "lifecycle_operation"
    DISCOVERY_ACCESS = "discovery_access"
    RPC_OPERATION = "rpc_operation"
    TENANT_MANAGEMENT = "tenant_management"
    SYSTEM_ADMINISTRATION = "system_administration"


class PermissionLevel(str, Enum):
    """Permission-Level für granulare Kontrolle."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"


class DelegationType(str, Enum):
    """Typen von Permission-Delegation."""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    CONDITIONAL = "conditional"


@dataclass
class Scope:
    """Definition eines Scopes."""
    name: str
    scope_type: ScopeType
    description: str
    required_permissions: set[str] = field(default_factory=set)
    implied_scopes: set[str] = field(default_factory=set)  # Scopes die automatisch gewährt werden
    conflicting_scopes: set[str] = field(default_factory=set)  # Scopes die sich ausschließen
    conditions: dict[str, Any] = field(default_factory=dict)
    is_system_scope: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def implies_scope(self, other_scope: str) -> bool:
        """Prüft, ob dieser Scope einen anderen impliziert."""
        return other_scope in self.implied_scopes

    def conflicts_with_scope(self, other_scope: str) -> bool:
        """Prüft, ob dieser Scope mit einem anderen in Konflikt steht."""
        return other_scope in self.conflicting_scopes


@dataclass
class PermissionGrant:
    """Gewährte Berechtigung."""
    principal_id: str
    scope: str
    permission_level: PermissionLevel
    resource_pattern: str = "*"  # Glob-Pattern für Ressourcen
    conditions: dict[str, Any] = field(default_factory=dict)

    # Delegation
    delegated_by: str | None = None
    delegation_type: DelegationType = DelegationType.PERMANENT

    # Zeitbeschränkungen
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Metadaten
    granted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    granted_by: str | None = None
    reason: str | None = None

    @property
    def is_valid(self) -> bool:
        """Prüft, ob Permission-Grant gültig ist."""
        now = datetime.now(UTC)

        if self.valid_from and now < self.valid_from:
            return False

        return not (self.valid_until and now > self.valid_until)

    @property
    def is_delegated(self) -> bool:
        """Prüft, ob Permission delegiert wurde."""
        return self.delegated_by is not None


@dataclass
class PermissionEvaluationContext:
    """Kontext für Permission-Evaluation."""
    principal_id: str
    scope: str
    resource_id: str
    action: str
    tenant_id: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PermissionEvaluationResult:
    """Ergebnis einer Permission-Evaluation."""
    granted: bool
    permission_level: PermissionLevel
    effective_scopes: set[str] = field(default_factory=set)
    matched_grants: list[PermissionGrant] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    reason: str = ""
    context: PermissionEvaluationContext | None = None


class ScopeRegistry:
    """Registry für Scope-Management."""

    def __init__(self) -> None:
        """Initialisiert Scope Registry."""
        self._scopes: dict[str, Scope] = {}
        self._scope_hierarchy: dict[str, set[str]] = {}
        self._create_default_scopes()

    def _create_default_scopes(self) -> None:
        """Erstellt Standard-Scopes für das System."""
        default_scopes = [
            # Agent Capability Scopes
            Scope(
                name="agent:capability:data_processing",
                scope_type=ScopeType.AGENT_CAPABILITY,
                description="Zugriff auf Datenverarbeitungs-Capabilities",
                required_permissions={"execute"},
                is_system_scope=True
            ),
            Scope(
                name="agent:capability:nlp",
                scope_type=ScopeType.AGENT_CAPABILITY,
                description="Zugriff auf NLP-Capabilities",
                required_permissions={"execute"},
                is_system_scope=True
            ),
            Scope(
                name="agent:capability:analysis",
                scope_type=ScopeType.AGENT_CAPABILITY,
                description="Zugriff auf Analyse-Capabilities",
                required_permissions={"execute"},
                is_system_scope=True
            ),

            # Lifecycle Operation Scopes
            Scope(
                name="agent:lifecycle:register",
                scope_type=ScopeType.LIFECYCLE_OPERATION,
                description="Agent-Registrierung",
                required_permissions={"create"},
                is_system_scope=True
            ),
            Scope(
                name="agent:lifecycle:manage",
                scope_type=ScopeType.LIFECYCLE_OPERATION,
                description="Vollständiges Agent-Lifecycle-Management",
                required_permissions={"create", "read", "update", "delete", "execute"},
                implied_scopes={"agent:lifecycle:register", "agent:lifecycle:monitor"},
                is_system_scope=True
            ),
            Scope(
                name="agent:lifecycle:monitor",
                scope_type=ScopeType.LIFECYCLE_OPERATION,
                description="Agent-Monitoring und -Status",
                required_permissions={"read"},
                is_system_scope=True
            ),

            # Discovery Access Scopes
            Scope(
                name="discovery:tools:read",
                scope_type=ScopeType.DISCOVERY_ACCESS,
                description="Tool-Discovery-Zugriff",
                required_permissions={"read"},
                is_system_scope=True
            ),
            Scope(
                name="discovery:resources:read",
                scope_type=ScopeType.DISCOVERY_ACCESS,
                description="Resource-Discovery-Zugriff",
                required_permissions={"read"},
                is_system_scope=True
            ),
            Scope(
                name="discovery:prompts:read",
                scope_type=ScopeType.DISCOVERY_ACCESS,
                description="Prompt-Discovery-Zugriff",
                required_permissions={"read"},
                is_system_scope=True
            ),

            # RPC Operation Scopes
            Scope(
                name="rpc:agent:call",
                scope_type=ScopeType.RPC_OPERATION,
                description="Agent-RPC-Aufrufe",
                required_permissions={"execute"},
                is_system_scope=True
            ),
            Scope(
                name="rpc:system:call",
                scope_type=ScopeType.RPC_OPERATION,
                description="System-RPC-Aufrufe",
                required_permissions={"execute"},
                is_system_scope=True
            ),

            # Tenant Management Scopes
            Scope(
                name="tenant:manage",
                scope_type=ScopeType.TENANT_MANAGEMENT,
                description="Tenant-Management",
                required_permissions={"create", "read", "update", "delete"},
                is_system_scope=True
            ),
            Scope(
                name="tenant:read",
                scope_type=ScopeType.TENANT_MANAGEMENT,
                description="Tenant-Informationen lesen",
                required_permissions={"read"},
                is_system_scope=True
            ),

            # System Administration Scopes
            Scope(
                name="system:admin",
                scope_type=ScopeType.SYSTEM_ADMINISTRATION,
                description="Vollständige System-Administration",
                required_permissions={"create", "read", "update", "delete", "execute", "manage"},
                implied_scopes={
                    "agent:lifecycle:manage", "discovery:tools:read", "discovery:resources:read",
                    "discovery:prompts:read", "rpc:system:call", "tenant:manage"
                },
                is_system_scope=True
            ),
            Scope(
                name="system:monitor",
                scope_type=ScopeType.SYSTEM_ADMINISTRATION,
                description="System-Monitoring",
                required_permissions={"read"},
                implied_scopes={"agent:lifecycle:monitor", "tenant:read"},
                is_system_scope=True
            )
        ]

        for scope in default_scopes:
            self._scopes[scope.name] = scope

    def register_scope(self, scope: Scope) -> bool:
        """Registriert neuen Scope.

        Args:
            scope: Scope-Definition

        Returns:
            True wenn erfolgreich registriert
        """
        if scope.name in self._scopes:
            logger.warning(f"Scope {scope.name} bereits registriert")
            return False

        self._scopes[scope.name] = scope
        self._update_scope_hierarchy()

        logger.info(f"Scope {scope.name} registriert")
        return True

    def get_scope(self, scope_name: str) -> Scope | None:
        """Gibt Scope zurück."""
        return self._scopes.get(scope_name)

    def get_scopes_by_type(self, scope_type: ScopeType) -> list[Scope]:
        """Gibt alle Scopes eines Typs zurück."""
        return [scope for scope in self._scopes.values() if scope.scope_type == scope_type]

    def get_implied_scopes(self, scope_name: str) -> set[str]:
        """Gibt alle implizierten Scopes zurück (rekursiv)."""
        if scope_name in self._scope_hierarchy:
            return self._scope_hierarchy[scope_name]

        return self._compute_implied_scopes(scope_name)

    def _compute_implied_scopes(self, scope_name: str, visited: set[str] | None = None) -> set[str]:
        """Berechnet implizierte Scopes rekursiv."""
        if visited is None:
            visited = set()

        if scope_name in visited:
            return set()  # Zirkuläre Abhängigkeit vermeiden

        visited.add(scope_name)
        implied = {scope_name}

        scope = self.get_scope(scope_name)
        if scope:
            for implied_scope in scope.implied_scopes:
                implied.update(self._compute_implied_scopes(implied_scope, visited.copy()))

        return implied

    def _update_scope_hierarchy(self) -> None:
        """Aktualisiert Scope-Hierarchie-Cache."""
        self._scope_hierarchy.clear()

        for scope_name in self._scopes:
            self._scope_hierarchy[scope_name] = self._compute_implied_scopes(scope_name)

    def validate_scope_combination(self, scopes: set[str]) -> tuple[bool, list[str]]:
        """Validiert Scope-Kombination auf Konflikte.

        Args:
            scopes: Set von Scope-Namen

        Returns:
            Tuple (is_valid, conflict_messages)
        """
        conflicts = []

        for scope_name in scopes:
            scope = self.get_scope(scope_name)
            if scope:
                for other_scope in scopes:
                    if other_scope != scope_name and scope.conflicts_with_scope(other_scope):
                        conflicts.append(f"Scope '{scope_name}' steht in Konflikt mit '{other_scope}'")

        return len(conflicts) == 0, conflicts


class PermissionManager:
    """Manager für Permission-Grants und -Evaluation."""

    def __init__(self, scope_registry: ScopeRegistry) -> None:
        """Initialisiert Permission Manager.

        Args:
            scope_registry: Scope Registry
        """
        self.scope_registry = scope_registry
        self._permission_grants: dict[str, list[PermissionGrant]] = {}
        self._evaluation_cache: dict[str, PermissionEvaluationResult] = {}
        self._cache_ttl = 300  # 5 Minuten
        self._audit_log: list[dict[str, Any]] = []
        self._max_audit_entries = 10000

    def grant_permission(
        self,
        principal_id: str,
        scope: str,
        permission_level: PermissionLevel,
        resource_pattern: str = "*",
        granted_by: str | None = None,
        valid_until: datetime | None = None,
        reason: str | None = None
    ) -> bool:
        """Gewährt Permission.

        Args:
            principal_id: Principal-ID
            scope: Scope-Name
            permission_level: Permission-Level
            resource_pattern: Ressourcen-Pattern
            granted_by: Gewährt von
            valid_until: Gültig bis
            reason: Grund für Gewährung

        Returns:
            True wenn erfolgreich gewährt
        """
        # Validiere Scope
        if not self.scope_registry.get_scope(scope):
            logger.error(f"Unbekannter Scope: {scope}")
            return False

        grant = PermissionGrant(
            principal_id=principal_id,
            scope=scope,
            permission_level=permission_level,
            resource_pattern=resource_pattern,
            granted_by=granted_by,
            valid_until=valid_until,
            reason=reason
        )

        if principal_id not in self._permission_grants:
            self._permission_grants[principal_id] = []

        self._permission_grants[principal_id].append(grant)

        # Cache invalidieren
        self._invalidate_cache_for_principal(principal_id)

        # Audit-Log
        self._log_permission_change("grant", grant)

        logger.info(f"Permission gewährt: {principal_id} -> {scope} ({permission_level.value})")
        return True

    def revoke_permission(
        self,
        principal_id: str,
        scope: str,
        revoked_by: str | None = None,
        reason: str | None = None
    ) -> bool:
        """Widerruft Permission.

        Args:
            principal_id: Principal-ID
            scope: Scope-Name
            revoked_by: Widerrufen von
            reason: Grund für Widerruf

        Returns:
            True wenn erfolgreich widerrufen
        """
        grants = self._permission_grants.get(principal_id, [])
        original_count = len(grants)

        # Entferne alle Grants für diesen Scope
        self._permission_grants[principal_id] = [
            grant for grant in grants if grant.scope != scope
        ]

        revoked_count = original_count - len(self._permission_grants[principal_id])

        if revoked_count > 0:
            # Cache invalidieren
            self._invalidate_cache_for_principal(principal_id)

            # Audit-Log
            self._log_permission_change("revoke", None, {
                "principal_id": principal_id,
                "scope": scope,
                "revoked_by": revoked_by,
                "reason": reason,
                "revoked_count": revoked_count
            })

            logger.info(f"Permission widerrufen: {principal_id} -> {scope} ({revoked_count} Grants)")
            return True

        return False

    def delegate_permission(
        self,
        delegator_id: str,
        delegatee_id: str,
        scope: str,
        permission_level: PermissionLevel,
        delegation_type: DelegationType = DelegationType.TEMPORARY,
        valid_until: datetime | None = None,
        reason: str | None = None
    ) -> bool:
        """Delegiert Permission.

        Args:
            delegator_id: Delegierender Principal
            delegatee_id: Empfangender Principal
            scope: Scope-Name
            permission_level: Permission-Level
            delegation_type: Delegations-Typ
            valid_until: Gültig bis
            reason: Grund für Delegation

        Returns:
            True wenn erfolgreich delegiert
        """
        # Prüfe, ob Delegator die Permission hat
        context = PermissionEvaluationContext(
            principal_id=delegator_id,
            scope=scope,
            resource_id="*",
            action="delegate"
        )

        result = self.evaluate_permission(context)
        if not result.granted:
            logger.warning(f"Delegation verweigert: {delegator_id} hat keine Permission für {scope}")
            return False

        # Erstelle delegierte Permission
        grant = PermissionGrant(
            principal_id=delegatee_id,
            scope=scope,
            permission_level=permission_level,
            delegated_by=delegator_id,
            delegation_type=delegation_type,
            valid_until=valid_until,
            reason=reason
        )

        if delegatee_id not in self._permission_grants:
            self._permission_grants[delegatee_id] = []

        self._permission_grants[delegatee_id].append(grant)

        # Cache invalidieren
        self._invalidate_cache_for_principal(delegatee_id)

        # Audit-Log
        self._log_permission_change("delegate", grant)

        logger.info(f"Permission delegiert: {delegator_id} -> {delegatee_id} ({scope})")
        return True

    @trace_function("permission.evaluate")
    def evaluate_permission(self, context: PermissionEvaluationContext) -> PermissionEvaluationResult:
        """Evaluiert Permission für gegebenen Kontext.

        Args:
            context: Evaluation-Kontext

        Returns:
            Evaluation-Ergebnis
        """
        start_time = time.time()

        # Cache-Check
        cache_key = self._get_cache_key(context)
        cached_result = self._evaluation_cache.get(cache_key)
        if cached_result and self._is_cache_valid(cached_result):
            return cached_result

        try:
            # Hole alle Grants für Principal
            grants = self._permission_grants.get(context.principal_id, [])

            # Filtere gültige Grants
            valid_grants = [grant for grant in grants if grant.is_valid]

            # Sammle alle effektiven Scopes (inklusive implizierte)
            effective_scopes = set()
            matched_grants = []

            for grant in valid_grants:
                # Prüfe Scope-Match
                if grant.scope == context.scope or context.scope in self.scope_registry.get_implied_scopes(grant.scope):
                    # Prüfe Ressourcen-Pattern
                    if self._matches_resource_pattern(context.resource_id, grant.resource_pattern):
                        effective_scopes.add(grant.scope)
                        effective_scopes.update(self.scope_registry.get_implied_scopes(grant.scope))
                        matched_grants.append(grant)

            # Bestimme höchstes Permission-Level
            permission_level = PermissionLevel.NONE
            if matched_grants:
                level_hierarchy = [
                    PermissionLevel.NONE, PermissionLevel.READ, PermissionLevel.WRITE,
                    PermissionLevel.EXECUTE, PermissionLevel.MANAGE, PermissionLevel.ADMIN
                ]

                for grant in matched_grants:
                    grant_level_index = level_hierarchy.index(grant.permission_level)
                    current_level_index = level_hierarchy.index(permission_level)

                    if grant_level_index > current_level_index:
                        permission_level = grant.permission_level

            # Entscheidung treffen
            granted = len(matched_grants) > 0 and permission_level != PermissionLevel.NONE
            reason = f"Permission {'gewährt' if granted else 'verweigert'} - Level: {permission_level.value}"

            if not granted:
                reason += f" - Keine passenden Grants für Scope '{context.scope}'"

            execution_time = (time.time() - start_time) * 1000

            result = PermissionEvaluationResult(
                granted=granted,
                permission_level=permission_level,
                effective_scopes=effective_scopes,
                matched_grants=matched_grants,
                evaluation_time_ms=execution_time,
                reason=reason,
                context=context
            )

            # Cache aktualisieren
            self._evaluation_cache[cache_key] = result

            # Audit-Log
            self._log_permission_evaluation(result)

            return result

        except Exception as e:
            logger.exception(f"Permission-Evaluation fehlgeschlagen: {e}")
            execution_time = (time.time() - start_time) * 1000

            return PermissionEvaluationResult(
                granted=False,
                permission_level=PermissionLevel.NONE,
                evaluation_time_ms=execution_time,
                reason=f"Evaluation-Fehler: {e!s}",
                context=context
            )

    def _matches_resource_pattern(self, resource_id: str, pattern: str) -> bool:
        """Prüft, ob Ressource auf Pattern passt."""
        import fnmatch
        return fnmatch.fnmatch(resource_id, pattern)

    def _get_cache_key(self, context: PermissionEvaluationContext) -> str:
        """Generiert Cache-Key für Kontext."""
        return f"{context.principal_id}:{context.scope}:{context.resource_id}:{context.action}"

    def _is_cache_valid(self, result: PermissionEvaluationResult) -> bool:
        """Prüft, ob Cache-Eintrag gültig ist."""
        if not result.context:
            return False

        age = (datetime.now(UTC) - result.context.timestamp).total_seconds()
        return age < self._cache_ttl

    def _invalidate_cache_for_principal(self, principal_id: str) -> None:
        """Invalidiert Cache für Principal."""
        keys_to_remove = [
            key for key in self._evaluation_cache
            if key.startswith(f"{principal_id}:")
        ]

        for key in keys_to_remove:
            del self._evaluation_cache[key]

    def _log_permission_change(self, action: str, grant: PermissionGrant | None, extra_data: dict | None = None) -> None:
        """Loggt Permission-Änderung."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "principal_id": grant.principal_id if grant else extra_data.get("principal_id"),
            "scope": grant.scope if grant else extra_data.get("scope"),
            "permission_level": grant.permission_level.value if grant else None,
            "delegated": grant.is_delegated if grant else False
        }

        if extra_data:
            log_entry.update(extra_data)

        self._audit_log.append(log_entry)

        # Begrenze Audit-Log-Größe
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

    def _log_permission_evaluation(self, result: PermissionEvaluationResult) -> None:
        """Loggt Permission-Evaluation."""
        if not result.context:
            return

        log_entry = {
            "timestamp": result.context.timestamp.isoformat(),
            "action": "evaluate",
            "principal_id": result.context.principal_id,
            "scope": result.context.scope,
            "resource_id": result.context.resource_id,
            "granted": result.granted,
            "permission_level": result.permission_level.value,
            "evaluation_time_ms": result.evaluation_time_ms,
            "matched_grants_count": len(result.matched_grants)
        }

        self._audit_log.append(log_entry)

        # Begrenze Audit-Log-Größe
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

    def get_principal_permissions(self, principal_id: str) -> list[PermissionGrant]:
        """Gibt alle Permissions eines Principals zurück."""
        grants = self._permission_grants.get(principal_id, [])
        return [grant for grant in grants if grant.is_valid]

    def get_audit_log(
        self,
        principal_id: str | None = None,
        action: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Gibt Audit-Log zurück."""
        filtered_log = self._audit_log

        if principal_id:
            filtered_log = [entry for entry in filtered_log if entry.get("principal_id") == principal_id]

        if action:
            filtered_log = [entry for entry in filtered_log if entry.get("action") == action]

        return list(reversed(filtered_log))[-limit:]

    def get_permission_stats(self) -> dict[str, Any]:
        """Gibt Permission-Statistiken zurück."""
        total_grants = sum(len(grants) for grants in self._permission_grants.values())
        total_principals = len(self._permission_grants)

        delegated_grants = sum(
            sum(1 for grant in grants if grant.is_delegated)
            for grants in self._permission_grants.values()
        )

        cache_size = len(self._evaluation_cache)
        audit_entries = len(self._audit_log)

        return {
            "total_grants": total_grants,
            "total_principals": total_principals,
            "delegated_grants": delegated_grants,
            "cache_size": cache_size,
            "audit_entries": audit_entries,
            "registered_scopes": len(self.scope_registry._scopes)
        }


# Globale Instanzen
scope_registry = ScopeRegistry()
permission_manager = PermissionManager(scope_registry)
