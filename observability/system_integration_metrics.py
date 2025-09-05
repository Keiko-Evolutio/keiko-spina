# backend/observability/system_integration_metrics.py
"""System-Integration-Metriken für Keiko Personal Assistant

Implementiert Integration mit Task Management, Enhanced Security, Audit System
und Policy Engine für umfassende System-Metriken.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .agent_metrics import ErrorCategory, get_agent_metrics_collector
from .base_metrics import MetricsConstants
from .metrics_aggregator import metrics_aggregator

logger = get_logger(__name__)


class SystemComponent(str, Enum):
    """System-Komponenten für Metriken-Integration."""
    TASK_MANAGEMENT = "task_management"
    ENHANCED_SECURITY = "enhanced_security"
    POLICY_ENGINE = "policy_engine"
    AUDIT_SYSTEM = "audit_system"
    QUOTAS_LIMITS = "quotas_limits"
    KEI_BUS = "kei_bus"
    AGENT_LIFECYCLE = "agent_lifecycle"


class SecurityEventType(str, Enum):
    """Security-Event-Typen."""
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_SUCCESS = "authorization_success"
    AUTHORIZATION_FAILURE = "authorization_failure"
    TOKEN_VALIDATION_SUCCESS = "token_validation_success"
    TOKEN_VALIDATION_FAILURE = "token_validation_failure"
    SECURITY_POLICY_VIOLATION = "security_policy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class PolicyEventType(str, Enum):
    """Policy-Event-Typen."""
    POLICY_EVALUATION_SUCCESS = "policy_evaluation_success"
    POLICY_EVALUATION_FAILURE = "policy_evaluation_failure"
    POLICY_VIOLATION = "policy_violation"
    POLICY_ENFORCEMENT_SUCCESS = "policy_enforcement_success"
    POLICY_ENFORCEMENT_FAILURE = "policy_enforcement_failure"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    COMPLIANCE_CHECK_SUCCESS = "compliance_check_success"
    COMPLIANCE_CHECK_FAILURE = "compliance_check_failure"


class AuditEventType(str, Enum):
    """Audit-Event-Typen."""
    AUDIT_LOG_CREATED = "audit_log_created"
    AUDIT_LOG_FAILED = "audit_log_failed"
    COMPLIANCE_REPORT_GENERATED = "compliance_report_generated"
    DATA_RETENTION_APPLIED = "data_retention_applied"
    PII_REDACTION_APPLIED = "pii_redaction_applied"
    TAMPER_DETECTION_TRIGGERED = "tamper_detection_triggered"


@dataclass
class SystemMetricsConfig:
    """Konfiguration für System-Integration-Metriken."""
    # Integration-Flags
    enable_task_management_metrics: bool = True
    enable_security_metrics: bool = True
    enable_policy_metrics: bool = True
    enable_audit_metrics: bool = True
    enable_quota_metrics: bool = True

    # Performance-Konfiguration
    metrics_collection_interval_seconds: int = MetricsConstants.DEFAULT_COLLECTION_INTERVAL_SECONDS
    batch_size: int = MetricsConstants.DEFAULT_BATCH_SIZE
    max_queue_size: int = MetricsConstants.DEFAULT_MAX_QUEUE_SIZE

    # Retention-Konfiguration
    metrics_retention_hours: int = MetricsConstants.DEFAULT_RETENTION_HOURS

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "enable_task_management_metrics": self.enable_task_management_metrics,
            "enable_security_metrics": self.enable_security_metrics,
            "enable_policy_metrics": self.enable_policy_metrics,
            "enable_audit_metrics": self.enable_audit_metrics,
            "enable_quota_metrics": self.enable_quota_metrics,
            "metrics_collection_interval_seconds": self.metrics_collection_interval_seconds,
            "batch_size": self.batch_size,
            "max_queue_size": self.max_queue_size,
            "metrics_retention_hours": self.metrics_retention_hours
        }


class TaskManagementMetricsCollector:
    """Metriken-Collector für Task Management System."""

    def __init__(self):
        """Initialisiert Task Management Metrics Collector."""
        self._task_metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "tasks_retried": 0,
            "tasks_timeout": 0
        }
        self._lock = threading.RLock()

    @trace_function("task_metrics.record_task_submitted")
    def record_task_submitted(
        self,
        task_id: str,
        task_type: str,
        priority: str,
        agent_id: str | None = None,
        user_id: str | None = None
    ) -> None:
        """Zeichnet Task-Submission auf.

        Args:
            task_id: Task-ID
            task_type: Task-Typ
            priority: Task-Priorität
            agent_id: Agent-ID
            user_id: User-ID
        """
        with self._lock:
            self._task_metrics["tasks_submitted"] += 1

        # Sammle Metrik
        metrics_aggregator.collect_metric(
            "task_management.tasks_submitted_total",
            1,
            tags={
                "task_type": task_type,
                "priority": priority,
                "agent_id": agent_id or "unknown",
                "user_id": user_id or "unknown"
            }
        )

        # Agent-spezifische Metriken
        if agent_id:
            agent_collector = get_agent_metrics_collector(agent_id)
            agent_collector.record_queue_enqueue()

    @trace_function("task_metrics.record_task_completed")
    def record_task_completed(
        self,
        task_id: str,
        task_type: str,
        duration_ms: float,
        agent_id: str | None = None,
        success: bool = True
    ) -> None:
        """Zeichnet Task-Completion auf.

        Args:
            task_id: Task-ID
            task_type: Task-Typ
            duration_ms: Ausführungsdauer in Millisekunden
            agent_id: Agent-ID
            success: Erfolg der Ausführung
        """
        with self._lock:
            if success:
                self._task_metrics["tasks_completed"] += 1
            else:
                self._task_metrics["tasks_failed"] += 1

        # Sammle Metriken
        status = "success" if success else "failure"

        metrics_aggregator.collect_metric(
            f"task_management.tasks_{status}_total",
            1,
            tags={
                "task_type": task_type,
                "agent_id": agent_id or "unknown"
            }
        )

        metrics_aggregator.collect_metric(
            "task_management.task_duration_ms",
            duration_ms,
            tags={
                "task_type": task_type,
                "status": status,
                "agent_id": agent_id or "unknown"
            }
        )

        # Agent-spezifische Metriken
        if agent_id:
            agent_collector = get_agent_metrics_collector(agent_id)
            agent_collector.record_task_execution(
                duration_ms=duration_ms,
                success=success,
                error_category=ErrorCategory.BUSINESS_LOGIC_ERROR if not success else None
            )
            agent_collector.record_queue_dequeue(duration_ms)

    @trace_function("task_metrics.record_queue_depth")
    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Zeichnet Queue-Depth auf.

        Args:
            queue_name: Name der Queue
            depth: Aktuelle Queue-Tiefe
        """
        metrics_aggregator.collect_metric(
            "task_management.queue_depth",
            depth,
            tags={"queue_name": queue_name}
        )

    def get_task_metrics(self) -> dict[str, Any]:
        """Gibt Task-Metriken zurück."""
        with self._lock:
            return self._task_metrics.copy()


class SecurityMetricsCollector:
    """Metriken-Collector für Enhanced Security System."""

    def __init__(self):
        """Initialisiert Security Metrics Collector."""
        self._security_metrics = {
            "authentication_attempts": 0,
            "authentication_successes": 0,
            "authentication_failures": 0,
            "authorization_checks": 0,
            "authorization_successes": 0,
            "authorization_failures": 0,
            "security_violations": 0
        }
        self._lock = threading.RLock()

    @trace_function("security_metrics.record_authentication")
    def record_authentication(
        self,
        user_id: str,
        success: bool,
        method: str = "unknown",
        duration_ms: float | None = None
    ) -> None:
        """Zeichnet Authentication-Event auf.

        Args:
            user_id: User-ID
            success: Erfolg der Authentication
            method: Authentication-Methode
            duration_ms: Dauer in Millisekunden
        """
        with self._lock:
            self._security_metrics["authentication_attempts"] += 1
            if success:
                self._security_metrics["authentication_successes"] += 1
            else:
                self._security_metrics["authentication_failures"] += 1

        # Sammle Metriken
        event_type = SecurityEventType.AUTHENTICATION_SUCCESS if success else SecurityEventType.AUTHENTICATION_FAILURE

        metrics_aggregator.collect_metric(
            f"security.{event_type.value}_total",
            1,
            tags={
                "user_id": user_id,
                "method": method
            }
        )

        if duration_ms is not None:
            metrics_aggregator.collect_metric(
                "security.authentication_duration_ms",
                duration_ms,
                tags={
                    "method": method,
                    "success": str(success).lower()
                }
            )

    @trace_function("security_metrics.record_authorization")
    def record_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        duration_ms: float | None = None
    ) -> None:
        """Zeichnet Authorization-Event auf.

        Args:
            user_id: User-ID
            resource: Ressource
            action: Aktion
            success: Erfolg der Authorization
            duration_ms: Dauer in Millisekunden
        """
        with self._lock:
            self._security_metrics["authorization_checks"] += 1
            if success:
                self._security_metrics["authorization_successes"] += 1
            else:
                self._security_metrics["authorization_failures"] += 1

        # Sammle Metriken
        event_type = SecurityEventType.AUTHORIZATION_SUCCESS if success else SecurityEventType.AUTHORIZATION_FAILURE

        metrics_aggregator.collect_metric(
            f"security.{event_type.value}_total",
            1,
            tags={
                "user_id": user_id,
                "resource": resource,
                "action": action
            }
        )

        if duration_ms is not None:
            metrics_aggregator.collect_metric(
                "security.authorization_duration_ms",
                duration_ms,
                tags={
                    "resource": resource,
                    "action": action,
                    "success": str(success).lower()
                }
            )

    @trace_function("security_metrics.record_security_violation")
    def record_security_violation(
        self,
        user_id: str,
        violation_type: str,
        severity: str = "medium",
        details: dict[str, Any] | None = None
    ) -> None:
        """Zeichnet Security-Violation auf.

        Args:
            user_id: User-ID
            violation_type: Typ der Violation
            severity: Schweregrad
            details: Zusätzliche Details
        """
        with self._lock:
            self._security_metrics["security_violations"] += 1

        # Sammle Metriken
        metrics_aggregator.collect_metric(
            "security.security_violations_total",
            1,
            tags={
                "user_id": user_id,
                "violation_type": violation_type,
                "severity": severity
            }
        )

    def get_security_metrics(self) -> dict[str, Any]:
        """Gibt Security-Metriken zurück."""
        with self._lock:
            return self._security_metrics.copy()


class PolicyMetricsCollector:
    """Metriken-Collector für Policy Engine."""

    def __init__(self):
        """Initialisiert Policy Metrics Collector."""
        self._policy_metrics = {
            "policy_evaluations": 0,
            "policy_violations": 0,
            "policy_enforcements": 0,
            "compliance_checks": 0,
            "resource_limit_exceeded": 0
        }
        self._lock = threading.RLock()

    @trace_function("policy_metrics.record_policy_evaluation")
    def record_policy_evaluation(
        self,
        policy_name: str,
        success: bool,
        duration_ms: float,
        user_id: str | None = None,
        resource: str | None = None
    ) -> None:
        """Zeichnet Policy-Evaluation auf.

        Args:
            policy_name: Name der Policy
            success: Erfolg der Evaluation
            duration_ms: Dauer in Millisekunden
            user_id: User-ID
            resource: Ressource
        """
        with self._lock:
            self._policy_metrics["policy_evaluations"] += 1
            if not success:
                self._policy_metrics["policy_violations"] += 1

        # Sammle Metriken
        event_type = PolicyEventType.POLICY_EVALUATION_SUCCESS if success else PolicyEventType.POLICY_VIOLATION

        metrics_aggregator.collect_metric(
            f"policy.{event_type.value}_total",
            1,
            tags={
                "policy_name": policy_name,
                "user_id": user_id or "unknown",
                "resource": resource or "unknown"
            }
        )

        metrics_aggregator.collect_metric(
            "policy.evaluation_duration_ms",
            duration_ms,
            tags={
                "policy_name": policy_name,
                "success": str(success).lower()
            }
        )

    @trace_function("policy_metrics.record_resource_limit")
    def record_resource_limit(
        self,
        resource_type: str,
        limit_type: str,
        current_usage: float,
        limit_value: float,
        user_id: str | None = None
    ) -> None:
        """Zeichnet Resource-Limit-Event auf.

        Args:
            resource_type: Typ der Ressource
            limit_type: Typ des Limits
            current_usage: Aktuelle Nutzung
            limit_value: Limit-Wert
            user_id: User-ID
        """
        exceeded = current_usage > limit_value

        if exceeded:
            with self._lock:
                self._policy_metrics["resource_limit_exceeded"] += 1

        # Sammle Metriken
        metrics_aggregator.collect_metric(
            "policy.resource_usage",
            current_usage,
            tags={
                "resource_type": resource_type,
                "limit_type": limit_type,
                "user_id": user_id or "unknown",
                "exceeded": str(exceeded).lower()
            }
        )

        metrics_aggregator.collect_metric(
            "policy.resource_limit",
            limit_value,
            tags={
                "resource_type": resource_type,
                "limit_type": limit_type
            }
        )

    def get_policy_metrics(self) -> dict[str, Any]:
        """Gibt Policy-Metriken zurück."""
        with self._lock:
            return self._policy_metrics.copy()


class AuditMetricsCollector:
    """Metriken-Collector für Audit System."""

    def __init__(self):
        """Initialisiert Audit Metrics Collector."""
        self._audit_metrics = {
            "audit_logs_created": 0,
            "audit_logs_failed": 0,
            "compliance_reports": 0,
            "pii_redactions": 0,
            "tamper_detections": 0
        }
        self._lock = threading.RLock()

    @trace_function("audit_metrics.record_audit_log")
    def record_audit_log(
        self,
        event_type: str,
        success: bool,
        duration_ms: float,
        user_id: str | None = None,
        resource: str | None = None
    ) -> None:
        """Zeichnet Audit-Log-Event auf.

        Args:
            event_type: Typ des Events
            success: Erfolg des Loggings
            duration_ms: Dauer in Millisekunden
            user_id: User-ID
            resource: Ressource
        """
        with self._lock:
            if success:
                self._audit_metrics["audit_logs_created"] += 1
            else:
                self._audit_metrics["audit_logs_failed"] += 1

        # Sammle Metriken
        audit_event_type = AuditEventType.AUDIT_LOG_CREATED if success else AuditEventType.AUDIT_LOG_FAILED

        metrics_aggregator.collect_metric(
            f"audit.{audit_event_type.value}_total",
            1,
            tags={
                "event_type": event_type,
                "user_id": user_id or "unknown",
                "resource": resource or "unknown"
            }
        )

        metrics_aggregator.collect_metric(
            "audit.log_creation_duration_ms",
            duration_ms,
            tags={
                "event_type": event_type,
                "success": str(success).lower()
            }
        )

    @trace_function("audit_metrics.record_compliance_report")
    def record_compliance_report(
        self,
        report_type: str,
        duration_ms: float,
        records_processed: int
    ) -> None:
        """Zeichnet Compliance-Report-Generation auf.

        Args:
            report_type: Typ des Reports
            duration_ms: Dauer in Millisekunden
            records_processed: Anzahl verarbeiteter Records
        """
        with self._lock:
            self._audit_metrics["compliance_reports"] += 1

        # Sammle Metriken
        metrics_aggregator.collect_metric(
            "audit.compliance_reports_total",
            1,
            tags={"report_type": report_type}
        )

        metrics_aggregator.collect_metric(
            "audit.compliance_report_duration_ms",
            duration_ms,
            tags={"report_type": report_type}
        )

        metrics_aggregator.collect_metric(
            "audit.compliance_report_records",
            records_processed,
            tags={"report_type": report_type}
        )

    def get_audit_metrics(self) -> dict[str, Any]:
        """Gibt Audit-Metriken zurück."""
        with self._lock:
            return self._audit_metrics.copy()


class SystemIntegrationMetrics:
    """Hauptklasse für System-Integration-Metriken."""

    def __init__(self, config: SystemMetricsConfig | None = None):
        """Initialisiert System Integration Metrics.

        Args:
            config: System-Metriken-Konfiguration
        """
        self.config = config or SystemMetricsConfig()

        # Komponenten-Collectors
        self.task_management_collector = TaskManagementMetricsCollector()
        self.security_collector = SecurityMetricsCollector()
        self.policy_collector = PolicyMetricsCollector()
        self.audit_collector = AuditMetricsCollector()

        # Background-Tasks
        self._collection_task: asyncio.Task | None = None
        self._is_running = False

        # Statistiken
        self._total_metrics_collected = 0
        self._collection_cycles = 0

    async def start(self) -> None:
        """Startet System Integration Metrics."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Collection
        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info("System Integration Metrics gestartet")

    async def stop(self) -> None:
        """Stoppt System Integration Metrics."""
        self._is_running = False

        if self._collection_task:
            self._collection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._collection_task

        logger.info("System Integration Metrics gestoppt")

    async def _collection_loop(self) -> None:
        """Background-Loop für Metriken-Sammlung."""
        while self._is_running:
            try:
                await self._collect_system_metrics()
                self._collection_cycles += 1
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
            except Exception as e:
                logger.exception(f"System-Metrics-Collection-Fehler: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self) -> None:
        """Sammelt System-Metriken von allen Komponenten."""
        # Task Management Metriken
        if self.config.enable_task_management_metrics:
            await self._collect_task_management_metrics()

        # Security Metriken
        if self.config.enable_security_metrics:
            await self._collect_security_metrics()

        # Policy Metriken
        if self.config.enable_policy_metrics:
            await self._collect_policy_metrics()

        # Audit Metriken
        if self.config.enable_audit_metrics:
            await self._collect_audit_metrics()

        # Agent-übergreifende Metriken
        await self._collect_agent_aggregate_metrics()

    async def _collect_task_management_metrics(self) -> None:
        """Sammelt Task-Management-Metriken."""
        try:
            # Importiere Task Manager (lazy import um Circular Dependencies zu vermeiden)
            from task_management import task_manager

            if hasattr(task_manager, "get_statistics"):
                stats = task_manager.get_statistics()

                # Sammle Task-State-Metriken
                for state, count in stats.get("tasks_by_state", {}).items():
                    metrics_aggregator.collect_metric(
                        "task_management.tasks_by_state",
                        count,
                        tags={"state": state}
                    )

                # Sammle Task-Priority-Metriken
                for priority, count in stats.get("tasks_by_priority", {}).items():
                    metrics_aggregator.collect_metric(
                        "task_management.tasks_by_priority",
                        count,
                        tags={"priority": priority}
                    )

                # Sammle Task-Type-Metriken
                for task_type, count in stats.get("tasks_by_type", {}).items():
                    metrics_aggregator.collect_metric(
                        "task_management.tasks_by_type",
                        count,
                        tags={"task_type": task_type}
                    )

                self._total_metrics_collected += len(stats.get("tasks_by_state", {}))

        except Exception as e:
            logger.warning(f"Task-Management-Metriken-Collection fehlgeschlagen: {e}")

    async def _collect_security_metrics(self) -> None:
        """Sammelt Security-Metriken."""
        try:
            # Sammle Security-System-Status
            metrics_aggregator.collect_metric(
                "security.system_status",
                1,
                tags={"status": "active"}
            )

            self._total_metrics_collected += 1

        except Exception as e:
            logger.warning(f"Security-Metriken-Collection fehlgeschlagen: {e}")

    async def _collect_policy_metrics(self) -> None:
        """Sammelt Policy-Metriken."""
        try:
            # Sammle Policy-System-Status
            metrics_aggregator.collect_metric(
                "policy.system_status",
                1,
                tags={"status": "active"}
            )

            self._total_metrics_collected += 1

        except Exception as e:
            logger.warning(f"Policy-Metriken-Collection fehlgeschlagen: {e}")

    async def _collect_audit_metrics(self) -> None:
        """Sammelt Audit-Metriken."""
        try:
            # Sammle Audit-System-Status
            metrics_aggregator.collect_metric(
                "audit.system_status",
                1,
                tags={"status": "active"}
            )

            self._total_metrics_collected += 1

        except Exception as e:
            logger.warning(f"Audit-Metriken-Collection fehlgeschlagen: {e}")

    async def _collect_agent_aggregate_metrics(self) -> None:
        """Sammelt aggregierte Agent-Metriken."""
        try:
            from .agent_metrics import get_all_agent_metrics

            all_agent_metrics = get_all_agent_metrics()

            # Aggregiere über alle Agenten
            total_agents = len(all_agent_metrics)
            total_tasks = sum(
                metrics.get("task_metrics", {}).get("success_count", 0) +
                metrics.get("task_metrics", {}).get("failure_count", 0)
                for metrics in all_agent_metrics.values()
            )

            metrics_aggregator.collect_metric(
                "agents.total_active",
                total_agents
            )

            metrics_aggregator.collect_metric(
                "agents.total_tasks_processed",
                total_tasks
            )

            self._total_metrics_collected += 2

        except Exception as e:
            logger.warning(f"Agent-Aggregate-Metriken-Collection fehlgeschlagen: {e}")

    def get_integration_statistics(self) -> dict[str, Any]:
        """Gibt Integration-Statistiken zurück.

        Returns:
            Statistiken-Dictionary
        """
        return {
            "config": self.config.to_dict(),
            "is_running": self._is_running,
            "total_metrics_collected": self._total_metrics_collected,
            "collection_cycles": self._collection_cycles,
            "component_metrics": {
                "task_management": self.task_management_collector.get_task_metrics(),
                "security": self.security_collector.get_security_metrics(),
                "policy": self.policy_collector.get_policy_metrics(),
                "audit": self.audit_collector.get_audit_metrics()
            }
        }


# Globale System Integration Metrics Instanz
system_integration_metrics = SystemIntegrationMetrics()
