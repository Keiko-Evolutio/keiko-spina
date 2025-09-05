# backend/task_management/task_integration.py
"""Task Integration Manager für Keiko Personal Assistant

Implementiert Integration mit Enhanced Security, Policy Engine, Quotas/Limits
und Audit System für vollständige Task-Management-Integration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .core_task_manager import Task, TaskPriority, TaskType

logger = get_logger(__name__)


class IntegrationStatus(str, Enum):
    """Status der Integration."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class IntegrationConfig:
    """Konfiguration für Task-Integration."""
    # Security-Integration
    enable_security_integration: bool = True
    require_authentication: bool = True
    require_authorization: bool = True
    security_check_timeout_seconds: int = 30

    # Policy-Integration
    enable_policy_integration: bool = True
    enforce_execution_policies: bool = True
    enforce_resource_policies: bool = True
    policy_check_timeout_seconds: int = 15

    # Quota-Integration
    enable_quota_integration: bool = True
    enforce_task_quotas: bool = True
    enforce_resource_quotas: bool = True
    quota_check_timeout_seconds: int = 10

    # Audit-Integration
    enable_audit_integration: bool = True
    audit_all_task_events: bool = True
    audit_security_events: bool = True
    audit_policy_events: bool = True
    audit_quota_events: bool = True

    # Error-Handling
    fail_open_on_security_error: bool = False
    fail_open_on_policy_error: bool = True
    fail_open_on_quota_error: bool = True
    fail_open_on_audit_error: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "enable_security_integration": self.enable_security_integration,
            "require_authentication": self.require_authentication,
            "require_authorization": self.require_authorization,
            "security_check_timeout_seconds": self.security_check_timeout_seconds,
            "enable_policy_integration": self.enable_policy_integration,
            "enforce_execution_policies": self.enforce_execution_policies,
            "enforce_resource_policies": self.enforce_resource_policies,
            "policy_check_timeout_seconds": self.policy_check_timeout_seconds,
            "enable_quota_integration": self.enable_quota_integration,
            "enforce_task_quotas": self.enforce_task_quotas,
            "enforce_resource_quotas": self.enforce_resource_quotas,
            "quota_check_timeout_seconds": self.quota_check_timeout_seconds,
            "enable_audit_integration": self.enable_audit_integration,
            "audit_all_task_events": self.audit_all_task_events,
            "audit_security_events": self.audit_security_events,
            "audit_policy_events": self.audit_policy_events,
            "audit_quota_events": self.audit_quota_events,
            "fail_open_on_security_error": self.fail_open_on_security_error,
            "fail_open_on_policy_error": self.fail_open_on_policy_error,
            "fail_open_on_quota_error": self.fail_open_on_quota_error,
            "fail_open_on_audit_error": self.fail_open_on_audit_error
        }


class SecurityIntegration:
    """Integration mit Enhanced Security System."""

    def __init__(self, config: IntegrationConfig):
        """Initialisiert Security Integration.

        Args:
            config: Integration-Konfiguration
        """
        self.config = config
        self.status = IntegrationStatus.ENABLED if config.enable_security_integration else IntegrationStatus.DISABLED

        # Statistiken
        self._authentication_checks = 0
        self._authorization_checks = 0
        self._security_failures = 0

    @trace_function("task_integration.check_authentication")
    async def check_authentication(self, task: Task) -> bool:
        """Prüft Authentifizierung für Task.

        Args:
            task: Task

        Returns:
            True wenn authentifiziert
        """
        if not self.config.enable_security_integration or not self.config.require_authentication:
            return True

        self._authentication_checks += 1

        try:
            # Integration mit Enhanced Security System
            # Vereinfachte Implementierung für Demo
            if not task.context.user_id:
                logger.warning(f"Task {task.task_id}: Keine User-ID für Authentication")
                self._security_failures += 1
                return False

            # Simuliere Authentication-Check
            await asyncio.sleep(0.01)  # Simulierte Latenz

            # Prüfe, ob User-ID gültig ist
            if task.context.user_id.startswith("invalid_"):
                logger.warning(f"Task {task.task_id}: Ungültige User-ID: {task.context.user_id}")
                self._security_failures += 1
                return False

            logger.debug(f"Task {task.task_id}: Authentication erfolgreich für User {task.context.user_id}")
            return True

        except Exception as e:
            logger.exception(f"Authentication-Check fehlgeschlagen für Task {task.task_id}: {e}")
            self._security_failures += 1

            if self.config.fail_open_on_security_error:
                logger.warning(f"Task {task.task_id}: Fail-open bei Security-Fehler")
                return True
            return False

    @trace_function("task_integration.check_authorization")
    async def check_authorization(self, task: Task) -> bool:
        """Prüft Autorisierung für Task.

        Args:
            task: Task

        Returns:
            True wenn autorisiert
        """
        if not self.config.enable_security_integration or not self.config.require_authorization:
            return True

        self._authorization_checks += 1

        try:
            # Integration mit Enhanced Security System
            # Vereinfachte Implementierung für Demo
            if not task.context.user_id:
                logger.warning(f"Task {task.task_id}: Keine User-ID für Authorization")
                self._security_failures += 1
                return False

            # Simuliere Authorization-Check
            await asyncio.sleep(0.01)  # Simulierte Latenz

            # Prüfe Task-Type-spezifische Berechtigungen
            required_permissions = {
                TaskType.AGENT_EXECUTION: ["agent_execute"],
                TaskType.DATA_PROCESSING: ["data_process"],
                TaskType.NLP_ANALYSIS: ["nlp_analyze"],
                TaskType.TOOL_CALL: ["tool_call"],
                TaskType.WORKFLOW: ["workflow_execute"],
                TaskType.BATCH_JOB: ["batch_execute"],
                TaskType.SCHEDULED_TASK: ["schedule_task"],
                TaskType.SYSTEM_MAINTENANCE: ["system_admin"]
            }

            required_perms = required_permissions.get(task.task_type, [])

            # Simuliere Permission-Check
            user_permissions = task.context.security_context.get("permissions", [])

            for perm in required_perms:
                if perm not in user_permissions:
                    logger.warning(f"Task {task.task_id}: Fehlende Berechtigung: {perm}")
                    self._security_failures += 1
                    return False

            logger.debug(f"Task {task.task_id}: Authorization erfolgreich")
            return True

        except Exception as e:
            logger.exception(f"Authorization-Check fehlgeschlagen für Task {task.task_id}: {e}")
            self._security_failures += 1

            if self.config.fail_open_on_security_error:
                logger.warning(f"Task {task.task_id}: Fail-open bei Security-Fehler")
                return True
            return False

    def get_security_statistics(self) -> dict[str, Any]:
        """Gibt Security-Statistiken zurück."""
        return {
            "status": self.status.value,
            "authentication_checks": self._authentication_checks,
            "authorization_checks": self._authorization_checks,
            "security_failures": self._security_failures,
            "failure_rate": self._security_failures / max(self._authentication_checks + self._authorization_checks, 1)
        }


class PolicyIntegration:
    """Integration mit Policy Engine."""

    def __init__(self, config: IntegrationConfig):
        """Initialisiert Policy Integration.

        Args:
            config: Integration-Konfiguration
        """
        self.config = config
        self.status = IntegrationStatus.ENABLED if config.enable_policy_integration else IntegrationStatus.DISABLED

        # Statistiken
        self._policy_checks = 0
        self._policy_violations = 0
        self._policy_enforcements = 0

    @trace_function("task_integration.check_execution_policy")
    async def check_execution_policy(self, task: Task) -> bool:
        """Prüft Execution-Policies für Task.

        Args:
            task: Task

        Returns:
            True wenn Policy erfüllt
        """
        if not self.config.enable_policy_integration or not self.config.enforce_execution_policies:
            return True

        self._policy_checks += 1

        try:
            # Integration mit Policy Engine
            # Vereinfachte Implementierung für Demo
            await asyncio.sleep(0.005)  # Simulierte Latenz

            # Prüfe Task-spezifische Policies
            violations = []

            # Priority-Policy: Kritische Tasks nur für autorisierte User
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.URGENT]:
                if not task.context.user_id or not task.context.user_id.startswith("admin_"):
                    violations.append("critical_task_unauthorized_user")

            # Timeout-Policy: Maximale Timeout-Limits
            max_timeouts = {
                TaskType.AGENT_EXECUTION: 1800,  # 30 Minuten
                TaskType.DATA_PROCESSING: 3600,  # 1 Stunde
                TaskType.BATCH_JOB: 7200,  # 2 Stunden
                TaskType.SYSTEM_MAINTENANCE: 14400  # 4 Stunden
            }

            max_timeout = max_timeouts.get(task.task_type, 300)
            if task.timeout_seconds > max_timeout:
                violations.append(f"timeout_exceeds_limit_{max_timeout}")

            # Resource-Policy: Payload-Größe-Limits
            payload_size = len(str(task.payload))
            if payload_size > 1024 * 1024:  # 1MB
                violations.append("payload_size_exceeds_limit")

            if violations:
                logger.warning(f"Task {task.task_id}: Policy-Violations: {violations}")
                self._policy_violations += 1

                if self.config.fail_open_on_policy_error:
                    logger.warning(f"Task {task.task_id}: Fail-open bei Policy-Violation")
                    return True
                return False

            logger.debug(f"Task {task.task_id}: Execution-Policy erfüllt")
            return True

        except Exception as e:
            logger.exception(f"Policy-Check fehlgeschlagen für Task {task.task_id}: {e}")
            self._policy_violations += 1

            if self.config.fail_open_on_policy_error:
                logger.warning(f"Task {task.task_id}: Fail-open bei Policy-Fehler")
                return True
            return False

    @trace_function("task_integration.enforce_resource_policy")
    async def enforce_resource_policy(self, task: Task) -> dict[str, Any]:
        """Enforced Resource-Policies für Task.

        Args:
            task: Task

        Returns:
            Resource-Limits
        """
        if not self.config.enable_policy_integration or not self.config.enforce_resource_policies:
            return {}

        self._policy_enforcements += 1

        try:
            # Integration mit Policy Engine
            # Vereinfachte Implementierung für Demo
            await asyncio.sleep(0.005)  # Simulierte Latenz

            # Bestimme Resource-Limits basierend auf Task-Type und Priority
            resource_limits = {}

            # CPU-Limits
            cpu_limits = {
                TaskPriority.LOW: 0.5,
                TaskPriority.NORMAL: 1.0,
                TaskPriority.HIGH: 2.0,
                TaskPriority.CRITICAL: 4.0,
                TaskPriority.URGENT: 8.0
            }
            resource_limits["max_cpu_cores"] = cpu_limits.get(task.priority, 1.0)

            # Memory-Limits
            memory_limits = {
                TaskType.AGENT_EXECUTION: 512,  # MB
                TaskType.DATA_PROCESSING: 2048,
                TaskType.NLP_ANALYSIS: 1024,
                TaskType.BATCH_JOB: 4096,
                TaskType.SYSTEM_MAINTENANCE: 1024
            }
            resource_limits["max_memory_mb"] = memory_limits.get(task.task_type, 512)

            # Network-Limits
            resource_limits["max_network_calls"] = 100
            resource_limits["max_network_bandwidth_mbps"] = 10

            # Disk-Limits
            resource_limits["max_disk_usage_mb"] = 1024
            resource_limits["max_disk_io_ops"] = 1000

            # Aktualisiere Task-Context
            task.context.resource_limits.update(resource_limits)

            logger.debug(f"Task {task.task_id}: Resource-Policy enforced: {resource_limits}")

            return resource_limits

        except Exception as e:
            logger.exception(f"Resource-Policy-Enforcement fehlgeschlagen für Task {task.task_id}: {e}")
            return {}

    def get_policy_statistics(self) -> dict[str, Any]:
        """Gibt Policy-Statistiken zurück."""
        return {
            "status": self.status.value,
            "policy_checks": self._policy_checks,
            "policy_violations": self._policy_violations,
            "policy_enforcements": self._policy_enforcements,
            "violation_rate": self._policy_violations / max(self._policy_checks, 1)
        }


class QuotaIntegration:
    """Integration mit Quotas/Limits System."""

    def __init__(self, config: IntegrationConfig):
        """Initialisiert Quota Integration.

        Args:
            config: Integration-Konfiguration
        """
        self.config = config
        self.status = IntegrationStatus.ENABLED if config.enable_quota_integration else IntegrationStatus.DISABLED

        # Statistiken
        self._quota_checks = 0
        self._quota_exceeded = 0
        self._quota_reservations = 0

    @trace_function("task_integration.check_task_quota")
    async def check_task_quota(self, task: Task) -> bool:
        """Prüft Task-Quotas.

        Args:
            task: Task

        Returns:
            True wenn Quota verfügbar
        """
        if not self.config.enable_quota_integration or not self.config.enforce_task_quotas:
            return True

        self._quota_checks += 1

        try:
            # Integration mit Quotas/Limits System
            # Vereinfachte Implementierung für Demo
            await asyncio.sleep(0.005)  # Simulierte Latenz

            # Prüfe verschiedene Quota-Typen
            quota_checks = []

            # User-basierte Task-Quotas
            if task.context.user_id:
                # Simuliere Quota-Check
                current_usage = hash(task.context.user_id) % 50  # 0-49
                quota_limit = 100

                if current_usage >= quota_limit:
                    quota_checks.append(f"user_task_quota_exceeded:{current_usage}/{quota_limit}")

            # Agent-basierte Task-Quotas
            if task.context.agent_id:
                # Simuliere Quota-Check
                current_usage = hash(task.context.agent_id) % 30  # 0-29
                quota_limit = 50

                if current_usage >= quota_limit:
                    quota_checks.append(f"agent_task_quota_exceeded:{current_usage}/{quota_limit}")

            # Task-Type-basierte Quotas
            # Simuliere Quota-Check
            current_usage = hash(task.task_type.value) % 20  # 0-19
            quota_limit = 25

            if current_usage >= quota_limit:
                quota_checks.append(f"task_type_quota_exceeded:{current_usage}/{quota_limit}")

            if quota_checks:
                logger.warning(f"Task {task.task_id}: Quota-Exceeded: {quota_checks}")
                self._quota_exceeded += 1

                if self.config.fail_open_on_quota_error:
                    logger.warning(f"Task {task.task_id}: Fail-open bei Quota-Exceeded")
                    return True
                return False

            logger.debug(f"Task {task.task_id}: Task-Quota verfügbar")
            return True

        except Exception as e:
            logger.exception(f"Quota-Check fehlgeschlagen für Task {task.task_id}: {e}")
            self._quota_exceeded += 1

            if self.config.fail_open_on_quota_error:
                logger.warning(f"Task {task.task_id}: Fail-open bei Quota-Fehler")
                return True
            return False

    @trace_function("task_integration.reserve_resources")
    async def reserve_resources(self, task: Task) -> bool:
        """Reserviert Ressourcen für Task.

        Args:
            task: Task

        Returns:
            True wenn Reservierung erfolgreich
        """
        if not self.config.enable_quota_integration or not self.config.enforce_resource_quotas:
            return True

        self._quota_reservations += 1

        try:
            # Integration mit Quotas/Limits System
            # Vereinfachte Implementierung für Demo
            await asyncio.sleep(0.005)  # Simulierte Latenz

            # Berechne benötigte Ressourcen
            required_resources = {}

            # CPU-Reservation
            required_cpu = task.context.resource_limits.get("max_cpu_cores", 1.0)
            required_resources["cpu_cores"] = required_cpu

            # Memory-Reservation
            required_memory = task.context.resource_limits.get("max_memory_mb", 512)
            required_resources["memory_mb"] = required_memory

            # Network-Reservation
            required_network = task.context.resource_limits.get("max_network_calls", 100)
            required_resources["network_calls"] = required_network

            # Simuliere Resource-Availability-Check
            available_resources = {
                "cpu_cores": 16.0,
                "memory_mb": 32768,
                "network_calls": 10000
            }

            for resource, required in required_resources.items():
                available = available_resources.get(resource, 0)
                if required > available:
                    logger.warning(f"Task {task.task_id}: Unzureichende Ressourcen: {resource} ({required} > {available})")
                    return False

            # Simuliere Reservierung
            logger.debug(f"Task {task.task_id}: Ressourcen reserviert: {required_resources}")

            return True

        except Exception as e:
            logger.exception(f"Resource-Reservation fehlgeschlagen für Task {task.task_id}: {e}")
            return False

    def get_quota_statistics(self) -> dict[str, Any]:
        """Gibt Quota-Statistiken zurück."""
        return {
            "status": self.status.value,
            "quota_checks": self._quota_checks,
            "quota_exceeded": self._quota_exceeded,
            "quota_reservations": self._quota_reservations,
            "exceeded_rate": self._quota_exceeded / max(self._quota_checks, 1)
        }


class AuditIntegration:
    """Integration mit Audit System."""

    def __init__(self, config: IntegrationConfig):
        """Initialisiert Audit Integration.

        Args:
            config: Integration-Konfiguration
        """
        self.config = config
        self.status = IntegrationStatus.ENABLED if config.enable_audit_integration else IntegrationStatus.DISABLED

        # Statistiken
        self._audit_events = 0
        self._audit_failures = 0

    @trace_function("task_integration.audit_task_event")
    async def audit_task_event(
        self,
        task: Task,
        event_type: str,
        event_data: dict[str, Any]
    ) -> bool:
        """Auditiert Task-Event.

        Args:
            task: Task
            event_type: Event-Typ
            event_data: Event-Daten

        Returns:
            True wenn Audit erfolgreich
        """
        if not self.config.enable_audit_integration:
            return True

        # Prüfe, ob Event auditiert werden soll
        should_audit = (
            self.config.audit_all_task_events or
            (event_type.startswith("security_") and self.config.audit_security_events) or
            (event_type.startswith("policy_") and self.config.audit_policy_events) or
            (event_type.startswith("quota_") and self.config.audit_quota_events)
        )

        if not should_audit:
            return True

        self._audit_events += 1

        try:
            # Integration mit Audit System
            # Vereinfachte Implementierung für Demo
            await asyncio.sleep(0.002)  # Simulierte Latenz

            # Erstelle Audit-Event
            {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "task_state": task.state.value,
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": task.context.user_id,
                "agent_id": task.context.agent_id,
                "correlation_id": task.context.correlation_id,
                "request_id": task.context.request_id
            }

            # Simuliere Audit-Logging
            logger.debug(f"Audit-Event: {event_type} für Task {task.task_id}")

            return True

        except Exception as e:
            logger.exception(f"Audit-Event fehlgeschlagen für Task {task.task_id}: {e}")
            self._audit_failures += 1

            if self.config.fail_open_on_audit_error:
                logger.warning(f"Task {task.task_id}: Fail-open bei Audit-Fehler")
                return True
            return False

    def get_audit_statistics(self) -> dict[str, Any]:
        """Gibt Audit-Statistiken zurück."""
        return {
            "status": self.status.value,
            "audit_events": self._audit_events,
            "audit_failures": self._audit_failures,
            "failure_rate": self._audit_failures / max(self._audit_events, 1)
        }


class TaskIntegrationManager:
    """Hauptklasse für Task-Integration."""

    def __init__(self, config: IntegrationConfig | None = None):
        """Initialisiert Task Integration Manager.

        Args:
            config: Integration-Konfiguration
        """
        self.config = config or IntegrationConfig()

        # Integration-Komponenten
        self.security_integration = SecurityIntegration(self.config)
        self.policy_integration = PolicyIntegration(self.config)
        self.quota_integration = QuotaIntegration(self.config)
        self.audit_integration = AuditIntegration(self.config)

        # Statistiken
        self._integrations_performed = 0
        self._integration_failures = 0

    @trace_function("task_integration.validate_task")
    async def validate_task(self, task: Task) -> bool:
        """Validiert Task gegen alle Integration-Systeme.

        Args:
            task: Task

        Returns:
            True wenn alle Validierungen erfolgreich
        """
        self._integrations_performed += 1

        try:
            # Security-Checks
            if self.config.enable_security_integration:
                # Authentication
                if not await self.security_integration.check_authentication(task):
                    await self.audit_integration.audit_task_event(
                        task, "security_authentication_failed", {"reason": "authentication_failed"}
                    )
                    return False

                # Authorization
                if not await self.security_integration.check_authorization(task):
                    await self.audit_integration.audit_task_event(
                        task, "security_authorization_failed", {"reason": "authorization_failed"}
                    )
                    return False

                await self.audit_integration.audit_task_event(
                    task, "security_validation_passed", {"checks": ["authentication", "authorization"]}
                )

            # Policy-Checks
            if self.config.enable_policy_integration:
                # Execution-Policy
                if not await self.policy_integration.check_execution_policy(task):
                    await self.audit_integration.audit_task_event(
                        task, "policy_execution_violation", {"reason": "execution_policy_failed"}
                    )
                    return False

                # Resource-Policy
                resource_limits = await self.policy_integration.enforce_resource_policy(task)
                await self.audit_integration.audit_task_event(
                    task, "policy_resource_enforced", {"resource_limits": resource_limits}
                )

            # Quota-Checks
            if self.config.enable_quota_integration:
                # Task-Quota
                if not await self.quota_integration.check_task_quota(task):
                    await self.audit_integration.audit_task_event(
                        task, "quota_task_exceeded", {"reason": "task_quota_exceeded"}
                    )
                    return False

                # Resource-Reservation
                if not await self.quota_integration.reserve_resources(task):
                    await self.audit_integration.audit_task_event(
                        task, "quota_resource_unavailable", {"reason": "resource_reservation_failed"}
                    )
                    return False

                await self.audit_integration.audit_task_event(
                    task, "quota_validation_passed", {"checks": ["task_quota", "resource_reservation"]}
                )

            # Audit Task-Validation
            await self.audit_integration.audit_task_event(
                task, "task_validation_completed", {"result": "success"}
            )

            logger.info(f"Task-Validation erfolgreich: {task.task_id}")
            return True

        except Exception as e:
            logger.exception(f"Task-Validation fehlgeschlagen für Task {task.task_id}: {e}")
            self._integration_failures += 1

            await self.audit_integration.audit_task_event(
                task, "task_validation_failed", {"error": str(e)}
            )

            return False

    def get_integration_status(self) -> dict[str, Any]:
        """Gibt Integration-Status zurück."""
        return {
            "config": self.config.to_dict(),
            "security_integration": self.security_integration.get_security_statistics(),
            "policy_integration": self.policy_integration.get_policy_statistics(),
            "quota_integration": self.quota_integration.get_quota_statistics(),
            "audit_integration": self.audit_integration.get_audit_statistics(),
            "overall_stats": {
                "integrations_performed": self._integrations_performed,
                "integration_failures": self._integration_failures,
                "failure_rate": self._integration_failures / max(self._integrations_performed, 1)
            }
        }


# Globale Task Integration Manager Instanz
task_integration_manager = TaskIntegrationManager()
