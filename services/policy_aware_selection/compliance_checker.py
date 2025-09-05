# backend/services/policy_aware_selection/compliance_checker.py
"""Agent Compliance Checker.

Prüft Agent-Compliance gegen definierte Policies,
Security-Anforderungen und Compliance-Standards.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from kei_logging import get_logger
from policy_engine.core_policy_engine import PolicyEngine
from services.orchestrator.data_models import SubtaskExecution

from .data_models import (
    AgentSelectionContext,
    ComplianceResult,
    DataClassification,
    PolicyConstraint,
    PolicyType,
    PolicyViolation,
    SecurityLevel,
)

logger = get_logger(__name__)


class AgentComplianceChecker:
    """Agent Compliance Checker für Policy-Enforcement."""

    def __init__(self, policy_engine: PolicyEngine):
        """Initialisiert Agent Compliance Checker.

        Args:
            policy_engine: Policy Engine für Policy-Abfragen
        """
        self.policy_engine = policy_engine

        # Compliance-Konfiguration
        self.strict_mode = True  # Strenge Compliance-Prüfung
        self.enable_warnings = True
        self.max_violations_allowed = 0  # Keine Violations erlaubt

        # Performance-Tracking
        self._check_count = 0
        self._total_check_time_ms = 0.0

        logger.info("Agent Compliance Checker initialisiert")

    async def check_agent_compliance(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution,
        context: AgentSelectionContext
    ) -> ComplianceResult:
        """Prüft Agent-Compliance für Subtask.

        Args:
            agent_info: Agent-Informationen
            subtask: Subtask-Definition
            context: Selection-Kontext

        Returns:
            Compliance-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "compliance_check_started",
                "agent_id": agent_info["agent_id"],
                "subtask_id": subtask.subtask_id,
                "tenant_id": context.tenant_id
            })

            # 1. Hole relevante Policies
            policies = await self._get_relevant_policies(context)

            # 2. Extrahiere Policy-Constraints
            constraints = self._extract_constraints_from_policies(policies)

            # 3. Prüfe Agent gegen alle Constraints
            violations = []
            passed_constraints = []
            warnings = []

            for constraint in constraints:
                violation = await self._check_constraint(
                    agent_info, subtask, context, constraint
                )

                if violation:
                    violations.append(violation)
                else:
                    passed_constraints.append(constraint.constraint_id)

            # 4. Berechne Compliance-Score
            compliance_score = self._calculate_compliance_score(
                len(passed_constraints), len(violations), len(constraints)
            )

            # 5. Bestimme Compliance-Status
            is_compliant = (
                len(violations) <= self.max_violations_allowed and
                compliance_score >= 0.8  # Mindestens 80% Compliance
            )

            # 6. Performance-Tracking
            check_duration_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(check_duration_ms)

            result = ComplianceResult(
                is_compliant=is_compliant,
                compliance_score=compliance_score,
                checked_policies=[p.policy_id for p in policies],
                passed_constraints=passed_constraints,
                failed_constraints=[v.constraint_id for v in violations],
                violations=violations,
                warnings=warnings,
                check_duration_ms=check_duration_ms,
                policies_evaluated=len(policies)
            )

            logger.debug({
                "event": "compliance_check_completed",
                "agent_id": agent_info["agent_id"],
                "is_compliant": is_compliant,
                "compliance_score": compliance_score,
                "violations_count": len(violations),
                "check_duration_ms": check_duration_ms
            })

            return result

        except Exception as e:
            logger.error(f"Compliance-Check fehlgeschlagen für Agent {agent_info.get('agent_id', 'unknown')}: {e}")

            return ComplianceResult(
                is_compliant=False,
                compliance_score=0.0,
                checked_policies=[],
                passed_constraints=[],
                failed_constraints=["check_error"],
                check_duration_ms=(time.time() - start_time) * 1000
            )

    async def _get_relevant_policies(self, context: AgentSelectionContext) -> list[Any]:
        """Holt relevante Policies für Kontext."""
        try:
            # Hole Policies basierend auf Kontext (Mock für Demo)
            policies = []  # TODO: Implementiere echte Policy-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116

            # Filtere aktive Policies
            active_policies = [p for p in policies if getattr(p, "is_active", True)]

            return active_policies

        except Exception as e:
            logger.error(f"Fehler beim Holen relevanter Policies: {e}")
            return []

    def _extract_constraints_from_policies(self, policies: list[Any]) -> list[PolicyConstraint]:
        """Extrahiert Constraints aus Policies."""
        constraints = []

        for policy in policies:
            policy_constraints = getattr(policy, "constraints", [])
            constraints.extend(policy_constraints)

        # Sortiere nach Priorität
        constraints.sort(key=lambda c: c.priority.value)

        return constraints

    async def _check_constraint(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution,
        context: AgentSelectionContext,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft einzelnen Constraint gegen Agent."""
        try:
            # Prüfe je nach Constraint-Type
            if constraint.policy_type == PolicyType.SECURITY:
                return await self._check_security_constraint(agent_info, context, constraint)

            if constraint.policy_type == PolicyType.CAPABILITY:
                return await self._check_capability_constraint(agent_info, subtask, constraint)

            if constraint.policy_type == PolicyType.TENANT:
                return await self._check_tenant_constraint(agent_info, context, constraint)

            if constraint.policy_type == PolicyType.GEOGRAPHIC:
                return await self._check_geographic_constraint(agent_info, context, constraint)

            if constraint.policy_type == PolicyType.DATA_CLASSIFICATION:
                return await self._check_data_classification_constraint(agent_info, context, constraint)

            if constraint.policy_type == PolicyType.COMPLIANCE:
                return await self._check_compliance_constraint(agent_info, context, constraint)

            if constraint.policy_type == PolicyType.PERFORMANCE:
                return await self._check_performance_constraint(agent_info, constraint)

            logger.warning(f"Unbekannter Constraint-Type: {constraint.policy_type}")
            return None

        except Exception as e:
            logger.error(f"Constraint-Check fehlgeschlagen: {e}")
            return None

    async def _check_security_constraint(
        self,
        agent_info: dict[str, Any],
        context: AgentSelectionContext,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Security-Constraint."""
        agent_security_level = SecurityLevel(agent_info.get("security_level", "internal"))

        # Prüfe minimales Security-Level
        if constraint.min_security_level:
            required_level = constraint.min_security_level

            # Security-Level-Hierarchie
            security_hierarchy = {
                SecurityLevel.PUBLIC: 0,
                SecurityLevel.INTERNAL: 1,
                SecurityLevel.CONFIDENTIAL: 2,
                SecurityLevel.RESTRICTED: 3,
                SecurityLevel.TOP_SECRET: 4
            }

            agent_level_value = security_hierarchy.get(agent_security_level, 0)
            required_level_value = security_hierarchy.get(required_level, 0)

            if agent_level_value < required_level_value:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="security_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="insufficient_security_level",
                    severity="high",
                    description=f"Agent Security-Level {agent_security_level.value} ist niedriger als erforderlich {required_level.value}",
                    agent_id=agent_info["agent_id"],
                    context=context,
                    expected_value=required_level.value,
                    actual_value=agent_security_level.value
                )

        # Prüfe erforderliche Clearances
        if constraint.required_clearances:
            agent_clearances = set(agent_info.get("clearances", []))
            required_clearances = set(constraint.required_clearances)

            missing_clearances = required_clearances - agent_clearances

            if missing_clearances:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="security_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="missing_clearances",
                    severity="high",
                    description=f"Agent fehlen erforderliche Clearances: {missing_clearances}",
                    agent_id=agent_info["agent_id"],
                    context=context,
                    expected_value=list(required_clearances),
                    actual_value=list(agent_clearances)
                )

        return None

    async def _check_capability_constraint(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Capability-Constraint."""
        agent_capabilities = set(cap.lower() for cap in agent_info.get("capabilities", []))

        # Prüfe erforderliche Capabilities
        if constraint.required_capabilities:
            required_caps = set(cap.lower() for cap in constraint.required_capabilities)
            missing_caps = required_caps - agent_capabilities

            if missing_caps:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="capability_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="missing_capabilities",
                    severity="medium",
                    description=f"Agent fehlen erforderliche Capabilities: {missing_caps}",
                    agent_id=agent_info["agent_id"],
                    context=subtask,
                    expected_value=list(required_caps),
                    actual_value=list(agent_capabilities)
                )

        # Prüfe verbotene Capabilities
        if constraint.forbidden_capabilities:
            forbidden_caps = set(cap.lower() for cap in constraint.forbidden_capabilities)
            forbidden_present = agent_capabilities.intersection(forbidden_caps)

            if forbidden_present:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="capability_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="forbidden_capabilities",
                    severity="high",
                    description=f"Agent hat verbotene Capabilities: {forbidden_present}",
                    agent_id=agent_info["agent_id"],
                    context=subtask,
                    expected_value=[],
                    actual_value=list(forbidden_present)
                )

        return None

    async def _check_tenant_constraint(
        self,
        agent_info: dict[str, Any],
        context: AgentSelectionContext,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Tenant-Constraint."""
        agent_tenant = agent_info.get("tenant_id")
        context_tenant = context.tenant_id

        # Tenant-Isolation prüfen
        if context_tenant and agent_tenant != context_tenant:
            return PolicyViolation(
                violation_id=str(uuid.uuid4()),
                policy_id="tenant_policy",
                constraint_id=constraint.constraint_id,
                violation_type="tenant_isolation_violation",
                severity="critical",
                description=f"Agent gehört zu anderem Tenant: {agent_tenant} != {context_tenant}",
                agent_id=agent_info["agent_id"],
                context=context,
                expected_value=context_tenant,
                actual_value=agent_tenant
            )

        return None

    async def _check_geographic_constraint(
        self,
        agent_info: dict[str, Any],
        context: AgentSelectionContext,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Geographic-Constraint."""
        agent_region = agent_info.get("deployment_region")

        # Prüfe erlaubte Regionen
        if constraint.allowed_regions and agent_region:
            if agent_region not in constraint.allowed_regions:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="geographic_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="region_not_allowed",
                    severity="high",
                    description=f"Agent-Region {agent_region} nicht in erlaubten Regionen: {constraint.allowed_regions}",
                    agent_id=agent_info["agent_id"],
                    context=context,
                    expected_value=constraint.allowed_regions,
                    actual_value=agent_region
                )

        # Prüfe verbotene Regionen
        if constraint.forbidden_regions and agent_region:
            if agent_region in constraint.forbidden_regions:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="geographic_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="region_forbidden",
                    severity="high",
                    description=f"Agent-Region {agent_region} ist verboten",
                    agent_id=agent_info["agent_id"],
                    context=context,
                    expected_value="not in " + str(constraint.forbidden_regions),
                    actual_value=agent_region
                )

        return None

    async def _check_data_classification_constraint(
        self,
        agent_info: dict[str, Any],
        context: AgentSelectionContext,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Data-Classification-Constraint."""
        # Prüfe ob Agent für Datenklassifizierung geeignet ist
        if context.data_classification:
            agent_clearances = agent_info.get("clearances", [])

            # Vereinfachte Datenklassifizierungs-Prüfung
            classification_requirements = {
                DataClassification.PII: ["pii_handling"],
                DataClassification.PHI: ["phi_handling", "hipaa_compliant"],
                DataClassification.FINANCIAL: ["financial_data_handling"],
                DataClassification.CONFIDENTIAL: ["confidential_clearance"],
                DataClassification.RESTRICTED: ["restricted_clearance"]
            }

            required_clearances = classification_requirements.get(context.data_classification, [])

            for required_clearance in required_clearances:
                if required_clearance not in agent_clearances:
                    return PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        policy_id="data_classification_policy",
                        constraint_id=constraint.constraint_id,
                        violation_type="insufficient_data_clearance",
                        severity="critical",
                        description=f"Agent fehlt Clearance für {context.data_classification.value}: {required_clearance}",
                        agent_id=agent_info["agent_id"],
                        context=context,
                        expected_value=required_clearances,
                        actual_value=agent_clearances
                    )

        return None

    async def _check_compliance_constraint(
        self,
        agent_info: dict[str, Any],
        context: AgentSelectionContext,
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Compliance-Constraint."""
        agent_compliance = agent_info.get("compliance_status", {})

        # Prüfe erforderliche Compliance-Standards
        for standard in constraint.required_compliance_standards:
            compliance_status = agent_compliance.get(standard.value, "unknown")

            if compliance_status != "compliant":
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="compliance_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="compliance_standard_not_met",
                    severity="high",
                    description=f"Agent nicht compliant mit {standard.value}: {compliance_status}",
                    agent_id=agent_info["agent_id"],
                    context=context,
                    expected_value="compliant",
                    actual_value=compliance_status
                )

        return None

    async def _check_performance_constraint(
        self,
        agent_info: dict[str, Any],
        constraint: PolicyConstraint
    ) -> PolicyViolation | None:
        """Prüft Performance-Constraint."""
        # Prüfe maximale Response-Zeit
        if constraint.max_response_time_ms:
            agent_response_time = agent_info.get("avg_response_time_ms", 0.0)

            if agent_response_time > constraint.max_response_time_ms:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="performance_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="response_time_too_high",
                    severity="medium",
                    description=f"Agent Response-Zeit {agent_response_time}ms > {constraint.max_response_time_ms}ms",
                    agent_id=agent_info["agent_id"],
                    context=None,
                    expected_value=constraint.max_response_time_ms,
                    actual_value=agent_response_time
                )

        # Prüfe minimale Success-Rate
        if constraint.min_success_rate:
            agent_success_rate = agent_info.get("success_rate", 0.0)

            if agent_success_rate < constraint.min_success_rate:
                return PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id="performance_policy",
                    constraint_id=constraint.constraint_id,
                    violation_type="success_rate_too_low",
                    severity="medium",
                    description=f"Agent Success-Rate {agent_success_rate} < {constraint.min_success_rate}",
                    agent_id=agent_info["agent_id"],
                    context=None,
                    expected_value=constraint.min_success_rate,
                    actual_value=agent_success_rate
                )

        return None

    def _calculate_compliance_score(
        self,
        passed_count: int,
        violation_count: int,
        total_count: int
    ) -> float:
        """Berechnet Compliance-Score."""
        if total_count == 0:
            return 1.0

        # Basis-Score basierend auf passed Constraints
        base_score = passed_count / total_count

        # Penalty für Violations
        violation_penalty = violation_count * 0.1  # 10% Penalty pro Violation

        # Finaler Score
        final_score = max(0.0, base_score - violation_penalty)

        return min(1.0, final_score)

    def _update_performance_stats(self, check_duration_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._check_count += 1
        self._total_check_time_ms += check_duration_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_check_time = (
            self._total_check_time_ms / self._check_count
            if self._check_count > 0 else 0.0
        )

        return {
            "total_checks": self._check_count,
            "avg_check_time_ms": avg_check_time,
            "total_check_time_ms": self._total_check_time_ms,
            "strict_mode": self.strict_mode,
            "max_violations_allowed": self.max_violations_allowed
        }
