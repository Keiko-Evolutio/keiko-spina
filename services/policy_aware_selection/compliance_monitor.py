# backend/services/policy_aware_selection/compliance_monitor.py
"""Compliance Monitor für Policy-aware Agent Selection.

Implementiert Audit-Trail, Compliance-Monitoring und
Policy-Violation-Tracking für Agent-Selection.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger
from services.messaging import BusService, get_bus_service

from .data_models import (
    AgentSelectionContext,
    AuditEvent,
    ComplianceResult,
    PolicyViolation,
    SecurityLevel,
)

logger = get_logger(__name__)


class ComplianceMonitor:
    """Compliance Monitor für Policy-aware Agent Selection."""

    def __init__(self, bus_service: BusService | None = None):
        """Initialisiert Compliance Monitor.

        Args:
            bus_service: Message Bus für Event-Publishing
        """
        self.bus_service = bus_service or get_bus_service()

        # Audit-Konfiguration
        self.enable_audit_trail = True
        self.enable_real_time_monitoring = True
        self.audit_retention_days = 365

        # Event-Storage
        self.audit_events: list[AuditEvent] = []
        self.violation_history: list[PolicyViolation] = []
        self.max_events_in_memory = 10000

        # Compliance-Tracking
        self.compliance_metrics: dict[str, Any] = {}
        self.violation_counts: dict[str, int] = {}

        # Background-Tasks
        self._monitoring_tasks: list[asyncio.Task] = []
        self._is_running = False

        logger.info("Compliance Monitor initialisiert")

    async def start(self) -> None:
        """Startet Compliance Monitor."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._audit_cleanup_loop()),
            asyncio.create_task(self._compliance_metrics_loop())
        ]

        logger.info("Compliance Monitor gestartet")

    async def stop(self) -> None:
        """Stoppt Compliance Monitor."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        self._monitoring_tasks.clear()

        logger.info("Compliance Monitor gestoppt")

    async def track_agent_selection(
        self,
        context: AgentSelectionContext,
        selected_agents: list[str],
        compliance_results: list[ComplianceResult]
    ) -> None:
        """Trackt Agent-Selection für Audit-Trail.

        Args:
            context: Agent-Selection-Kontext
            selected_agents: Liste ausgewählter Agent-IDs
            compliance_results: Compliance-Results der Selection
        """
        try:
            # Erstelle Audit-Event
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type="agent_selection",
                orchestration_id=context.orchestration_id,
                subtask_id=context.subtask_id,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                event_data={
                    "selected_agents": selected_agents,
                    "selection_context": {
                        "task_type": context.task_type,
                        "security_level": context.security_level.value,
                        "data_classification": context.data_classification.value if context.data_classification else None,
                        "required_capabilities": context.required_capabilities,
                        "contains_pii": context.contains_pii,
                        "contains_phi": context.contains_phi
                    },
                    "compliance_summary": {
                        "total_checks": len(compliance_results),
                        "compliant_agents": sum(1 for r in compliance_results if r.is_compliant),
                        "avg_compliance_score": sum(r.compliance_score for r in compliance_results) / len(compliance_results) if compliance_results else 0.0
                    }
                },
                compliance_results=compliance_results,
                sensitivity_level=context.security_level,
                compliance_frameworks=context.compliance_requirements,
                correlation_id=context.correlation_id
            )

            await self._record_audit_event(audit_event)

            # Tracke Violations
            for compliance_result in compliance_results:
                for violation in compliance_result.violations:
                    await self._track_policy_violation(violation, context)

            logger.debug({
                "event": "agent_selection_tracked",
                "context_id": context.request_id,
                "selected_count": len(selected_agents),
                "compliance_checks": len(compliance_results)
            })

        except Exception as e:
            logger.error(f"Agent-Selection-Tracking fehlgeschlagen: {e}")

    async def track_policy_violation(
        self,
        violation: PolicyViolation,
        context: AgentSelectionContext
    ) -> None:
        """Trackt Policy-Violation.

        Args:
            violation: Policy-Violation
            context: Agent-Selection-Kontext
        """
        try:
            # Erstelle Audit-Event für Violation
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type="policy_violation",
                orchestration_id=context.orchestration_id,
                subtask_id=context.subtask_id,
                agent_id=violation.agent_id,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                event_data={
                    "violation_id": violation.violation_id,
                    "policy_id": violation.policy_id,
                    "constraint_id": violation.constraint_id,
                    "violation_type": violation.violation_type,
                    "severity": violation.severity,
                    "description": violation.description,
                    "expected_value": violation.expected_value,
                    "actual_value": violation.actual_value,
                    "can_be_waived": violation.can_be_waived,
                    "remediation_suggestions": violation.remediation_suggestions
                },
                sensitivity_level=SecurityLevel.CONFIDENTIAL,  # Violations sind vertraulich
                compliance_frameworks=context.compliance_requirements,
                correlation_id=context.correlation_id
            )

            await self._record_audit_event(audit_event)

            # Speichere Violation in History
            self.violation_history.append(violation)

            # Update Violation-Counts
            violation_key = f"{violation.policy_id}_{violation.violation_type}"
            self.violation_counts[violation_key] = self.violation_counts.get(violation_key, 0) + 1

            # Real-time Alerting für kritische Violations
            if violation.severity == "critical":
                await self._send_critical_violation_alert(violation, context)

            logger.warning({
                "event": "policy_violation_tracked",
                "violation_id": violation.violation_id,
                "agent_id": violation.agent_id,
                "severity": violation.severity,
                "violation_type": violation.violation_type
            })

        except Exception as e:
            logger.error(f"Policy-Violation-Tracking fehlgeschlagen: {e}")

    async def _track_policy_violation(
        self,
        violation: PolicyViolation,
        context: AgentSelectionContext
    ) -> None:
        """Interne Violation-Tracking-Methode."""
        await self.track_policy_violation(violation, context)

    async def track_compliance_check(
        self,
        agent_id: str,
        context: AgentSelectionContext,
        compliance_result: ComplianceResult
    ) -> None:
        """Trackt Compliance-Check für einzelnen Agent.

        Args:
            agent_id: Agent-ID
            context: Agent-Selection-Kontext
            compliance_result: Compliance-Result
        """
        try:
            # Erstelle Audit-Event
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type="compliance_check",
                orchestration_id=context.orchestration_id,
                subtask_id=context.subtask_id,
                agent_id=agent_id,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                event_data={
                    "compliance_score": compliance_result.compliance_score,
                    "is_compliant": compliance_result.is_compliant,
                    "checked_policies": compliance_result.checked_policies,
                    "passed_constraints": compliance_result.passed_constraints,
                    "failed_constraints": compliance_result.failed_constraints,
                    "check_duration_ms": compliance_result.check_duration_ms,
                    "policies_evaluated": compliance_result.policies_evaluated,
                    "violation_count": len(compliance_result.violations),
                    "warning_count": len(compliance_result.warnings)
                },
                compliance_results=[compliance_result],
                sensitivity_level=context.security_level,
                compliance_frameworks=context.compliance_requirements,
                correlation_id=context.correlation_id
            )

            await self._record_audit_event(audit_event)

        except Exception as e:
            logger.error(f"Compliance-Check-Tracking fehlgeschlagen: {e}")

    async def _record_audit_event(self, audit_event: AuditEvent) -> None:
        """Speichert Audit-Event."""
        if not self.enable_audit_trail:
            return

        try:
            # Speichere in Memory (für schnellen Zugriff)
            self.audit_events.append(audit_event)

            # Memory-Limit prüfen
            if len(self.audit_events) > self.max_events_in_memory:
                # Entferne älteste Events
                self.audit_events = self.audit_events[-self.max_events_in_memory:]

            # Publiziere Event über Message Bus
            if self.enable_real_time_monitoring:
                await self._publish_audit_event(audit_event)

            # TODO: Persistiere in Datenbank für langfristige Speicherung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116

        except Exception as e:
            logger.error(f"Audit-Event-Recording fehlgeschlagen: {e}")

    async def _publish_audit_event(self, audit_event: AuditEvent) -> None:
        """Publiziert Audit-Event über Message Bus."""
        try:
            if self.bus_service:
                subject = f"kei.policy_aware_selection.audit.{audit_event.event_type}.v1"

                await self.bus_service.publish(
                    subject=subject,
                    data={
                        "event_id": audit_event.event_id,
                        "event_type": audit_event.event_type,
                        "orchestration_id": audit_event.orchestration_id,
                        "subtask_id": audit_event.subtask_id,
                        "agent_id": audit_event.agent_id,
                        "user_id": audit_event.user_id,
                        "tenant_id": audit_event.tenant_id,
                        "event_data": audit_event.event_data,
                        "timestamp": audit_event.timestamp.isoformat(),
                        "sensitivity_level": audit_event.sensitivity_level.value,
                        "compliance_frameworks": [cf.value for cf in audit_event.compliance_frameworks],
                        "correlation_id": audit_event.correlation_id
                    }
                )
        except Exception as e:
            logger.error(f"Audit-Event-Publishing fehlgeschlagen: {e}")

    async def _send_critical_violation_alert(
        self,
        violation: PolicyViolation,
        context: AgentSelectionContext
    ) -> None:
        """Sendet Alert für kritische Policy-Violation."""
        try:
            alert_data = {
                "alert_type": "critical_policy_violation",
                "violation_id": violation.violation_id,
                "agent_id": violation.agent_id,
                "policy_id": violation.policy_id,
                "violation_type": violation.violation_type,
                "description": violation.description,
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "critical"
            }

            if self.bus_service:
                await self.bus_service.publish(
                    subject="kei.policy_aware_selection.alerts.critical_violation.v1",
                    data=alert_data
                )

            logger.critical({
                "event": "critical_policy_violation_alert",
                "violation_id": violation.violation_id,
                "agent_id": violation.agent_id,
                "policy_id": violation.policy_id
            })

        except Exception as e:
            logger.error(f"Critical-Violation-Alert fehlgeschlagen: {e}")

    async def get_audit_trail(
        self,
        orchestration_id: str | None = None,
        tenant_id: str | None = None,
        event_types: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100
    ) -> list[AuditEvent]:
        """Holt Audit-Trail basierend auf Filtern.

        Args:
            orchestration_id: Filter nach Orchestration-ID
            tenant_id: Filter nach Tenant-ID
            event_types: Filter nach Event-Types
            start_time: Filter nach Start-Zeit
            end_time: Filter nach End-Zeit
            limit: Maximale Anzahl Events

        Returns:
            Liste von Audit-Events
        """
        try:
            filtered_events = self.audit_events.copy()

            # Anwenden der Filter
            if orchestration_id:
                filtered_events = [e for e in filtered_events if e.orchestration_id == orchestration_id]

            if tenant_id:
                filtered_events = [e for e in filtered_events if e.tenant_id == tenant_id]

            if event_types:
                filtered_events = [e for e in filtered_events if e.event_type in event_types]

            if start_time:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

            if end_time:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

            # Sortiere nach Timestamp (neueste zuerst)
            filtered_events.sort(key=lambda e: e.timestamp, reverse=True)

            # Limitiere Ergebnisse
            return filtered_events[:limit]

        except Exception as e:
            logger.error(f"Audit-Trail-Abfrage fehlgeschlagen: {e}")
            return []

    async def get_compliance_metrics(
        self,
        tenant_id: str | None = None,
        time_range_hours: int = 24
    ) -> dict[str, Any]:
        """Holt Compliance-Metriken.

        Args:
            tenant_id: Filter nach Tenant-ID
            time_range_hours: Zeitraum in Stunden

        Returns:
            Compliance-Metriken
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)

            # Filtere Events nach Zeitraum und Tenant
            relevant_events = [
                e for e in self.audit_events
                if e.timestamp >= cutoff_time and
                (not tenant_id or e.tenant_id == tenant_id)
            ]

            # Berechne Metriken
            total_selections = len([e for e in relevant_events if e.event_type == "agent_selection"])
            total_violations = len([e for e in relevant_events if e.event_type == "policy_violation"])
            total_compliance_checks = len([e for e in relevant_events if e.event_type == "compliance_check"])

            # Violation-Breakdown
            violation_breakdown = {}
            for event in relevant_events:
                if event.event_type == "policy_violation":
                    violation_type = event.event_data.get("violation_type", "unknown")
                    violation_breakdown[violation_type] = violation_breakdown.get(violation_type, 0) + 1

            # Compliance-Rate
            compliance_rate = (
                (total_compliance_checks - total_violations) / total_compliance_checks
                if total_compliance_checks > 0 else 1.0
            )

            return {
                "time_range_hours": time_range_hours,
                "tenant_id": tenant_id,
                "total_agent_selections": total_selections,
                "total_policy_violations": total_violations,
                "total_compliance_checks": total_compliance_checks,
                "compliance_rate": compliance_rate,
                "violation_breakdown": violation_breakdown,
                "metrics_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Compliance-Metriken-Berechnung fehlgeschlagen: {e}")
            return {}

    async def _audit_cleanup_loop(self) -> None:
        """Background-Loop für Audit-Cleanup."""
        while self._is_running:
            try:
                # Entferne alte Events basierend auf Retention-Policy
                cutoff_time = datetime.utcnow() - timedelta(days=self.audit_retention_days)

                original_count = len(self.audit_events)
                self.audit_events = [
                    e for e in self.audit_events
                    if e.timestamp > cutoff_time
                ]

                cleaned_count = original_count - len(self.audit_events)
                if cleaned_count > 0:
                    logger.info(f"Audit-Cleanup: {cleaned_count} alte Events entfernt")

                # Cleanup alle 6 Stunden
                await asyncio.sleep(6 * 3600)

            except Exception as e:
                logger.error(f"Audit-Cleanup-Fehler: {e}")
                await asyncio.sleep(3600)

    async def _compliance_metrics_loop(self) -> None:
        """Background-Loop für Compliance-Metriken-Update."""
        while self._is_running:
            try:
                # Update Compliance-Metriken
                self.compliance_metrics = await self.get_compliance_metrics()

                # Metriken alle 15 Minuten aktualisieren
                await asyncio.sleep(15 * 60)

            except Exception as e:
                logger.error(f"Compliance-Metriken-Update-Fehler: {e}")
                await asyncio.sleep(15 * 60)

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        return {
            "total_audit_events": len(self.audit_events),
            "total_violations": len(self.violation_history),
            "unique_violation_types": len(self.violation_counts),
            "audit_trail_enabled": self.enable_audit_trail,
            "real_time_monitoring_enabled": self.enable_real_time_monitoring,
            "retention_days": self.audit_retention_days,
            "max_events_in_memory": self.max_events_in_memory
        }
