# backend/services/policy_aware_selection/orchestrator_integration.py
"""Integration Layer für Policy-aware Agent Selection mit Orchestrator Service.

Erweitert den Orchestrator Service um Policy-Enforcement
und nahtlose Integration der Policy-aware Agent Selection.
"""

from __future__ import annotations

import time
from typing import Any

from kei_logging import get_logger
from policy_engine.core_policy_engine import PolicyEngine

# from security.enhanced_auth_middleware import SecurityManager
from services.orchestrator.data_models import AgentMatch, SubtaskExecution
from services.orchestrator.execution_engine import ExecutionEngine

from .agent_selector import PolicyAwareAgentSelector
from .compliance_checker import AgentComplianceChecker
from .compliance_monitor import ComplianceMonitor
from .data_models import AgentSelectionContext, DataClassification, SecurityLevel
from .policy_engine import PolicyEnforcementEngine

logger = get_logger(__name__)


class PolicyAwareOrchestrationIntegration:
    """Integration Layer für Policy-aware Agent Selection."""

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        policy_engine: PolicyEngine,
        security_manager: Any,
        enable_policy_enforcement: bool = True
    ):
        """Initialisiert Policy-aware Orchestration Integration.

        Args:
            execution_engine: Orchestrator Execution Engine
            policy_engine: Policy Engine
            security_manager: Security Manager
            enable_policy_enforcement: Policy-Enforcement aktivieren
        """
        self.execution_engine = execution_engine
        self.policy_engine = policy_engine
        self.security_manager = security_manager
        self.enable_policy_enforcement = enable_policy_enforcement

        # Policy-aware Komponenten
        self.compliance_checker = AgentComplianceChecker(policy_engine)
        self.policy_enforcement_engine = PolicyEnforcementEngine(policy_engine)
        self.compliance_monitor = ComplianceMonitor()

        self.policy_aware_selector = PolicyAwareAgentSelector(
            agent_registry=execution_engine.agent_registry,
            policy_engine=policy_engine,
            security_manager=security_manager,
            compliance_checker=self.compliance_checker,
            policy_enforcement_engine=self.policy_enforcement_engine
        )

        # Performance-Tracking
        self._integration_count = 0
        self._total_integration_time_ms = 0.0
        self._policy_overhead_ms = 0.0

        logger.info({
            "event": "policy_aware_integration_initialized",
            "policy_enforcement_enabled": enable_policy_enforcement
        })

    async def start(self) -> None:
        """Startet Policy-aware Integration."""
        await self.compliance_monitor.start()
        logger.info("Policy-aware Orchestration Integration gestartet")

    async def stop(self) -> None:
        """Stoppt Policy-aware Integration."""
        await self.compliance_monitor.stop()
        logger.info("Policy-aware Orchestration Integration gestoppt")

    async def select_agents_with_policy_enforcement(
        self,
        subtasks: list[SubtaskExecution],
        orchestration_id: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        correlation_id: str | None = None
    ) -> dict[str, list[AgentMatch]]:
        """Wählt Agents mit Policy-Enforcement aus.

        Args:
            subtasks: Liste von Subtasks
            orchestration_id: Orchestration-ID
            user_id: User-ID
            tenant_id: Tenant-ID
            session_id: Session-ID
            correlation_id: Correlation-ID

        Returns:
            Dictionary: subtask_id -> [AgentMatch] (policy-compliant)
        """
        start_time = time.time()

        try:
            if not self.enable_policy_enforcement:
                # Fallback auf Standard-Agent-Selection
                return await self._fallback_agent_selection(subtasks)

            logger.info({
                "event": "policy_aware_agent_selection_started",
                "orchestration_id": orchestration_id,
                "subtask_count": len(subtasks),
                "tenant_id": tenant_id,
                "user_id": user_id
            })

            # Erstelle Agent-Selection-Kontext für jeden Subtask
            agent_assignments = {}

            for subtask in subtasks:
                # Erstelle Kontext für Subtask
                context = await self._create_selection_context(
                    subtask, orchestration_id, user_id, tenant_id, session_id, correlation_id
                )

                # Policy-aware Agent-Selection
                policy_start_time = time.time()

                selected_agents = await self.policy_aware_selector.select_agents_with_policies(
                    [subtask], context, max_agents_per_subtask=3
                )

                policy_time_ms = (time.time() - policy_start_time) * 1000
                self._policy_overhead_ms += policy_time_ms

                # Hole Agent-Matches für Subtask
                agent_matches = selected_agents.get(subtask.subtask_id, [])
                agent_assignments[subtask.subtask_id] = agent_matches

                # Compliance-Monitoring
                await self._track_agent_selection(context, agent_matches)

            # Performance-Tracking
            integration_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(integration_time_ms)

            logger.info({
                "event": "policy_aware_agent_selection_completed",
                "orchestration_id": orchestration_id,
                "total_assignments": sum(len(matches) for matches in agent_assignments.values()),
                "integration_time_ms": integration_time_ms,
                "policy_overhead_ms": self._policy_overhead_ms / len(subtasks)
            })

            return agent_assignments

        except Exception as e:
            logger.error(f"Policy-aware Agent-Selection fehlgeschlagen: {e}")

            # Fallback auf Standard-Selection
            return await self._fallback_agent_selection(subtasks)

    async def _create_selection_context(
        self,
        subtask: SubtaskExecution,
        orchestration_id: str,
        user_id: str | None,
        tenant_id: str | None,
        session_id: str | None,
        correlation_id: str | None
    ) -> AgentSelectionContext:
        """Erstellt Agent-Selection-Kontext für Subtask."""
        try:
            # Extrahiere Kontext-Informationen aus Subtask
            task_payload = subtask.payload or {}

            # Bestimme Data-Classification
            data_classification = self._determine_data_classification(task_payload)

            # Bestimme Security-Level
            security_level = self._determine_security_level(task_payload, user_id, tenant_id)

            # Hole User-Informationen
            user_groups, user_clearances = await self._get_user_context(user_id, tenant_id)

            # Bestimme Compliance-Anforderungen
            compliance_requirements = await self._get_compliance_requirements(
                tenant_id, data_classification
            )

            context = AgentSelectionContext(
                request_id=f"{orchestration_id}_{subtask.subtask_id}",
                orchestration_id=orchestration_id,
                subtask_id=subtask.subtask_id,
                user_id=user_id,
                tenant_id=tenant_id,
                user_groups=user_groups,
                user_clearances=user_clearances,
                task_type=subtask.task_type.value,
                task_payload=task_payload,
                required_capabilities=getattr(subtask, "required_capabilities", []),
                data_classification=data_classification,
                contains_pii=self._contains_pii(task_payload),
                contains_phi=self._contains_phi(task_payload),
                geographic_restrictions=self._get_geographic_restrictions(task_payload),
                security_level=security_level,
                compliance_requirements=compliance_requirements,
                max_execution_time_ms=subtask.estimated_duration_minutes * 60 * 1000 if subtask.estimated_duration_minutes else None,
                priority_level=subtask.priority.value,
                correlation_id=correlation_id
            )

            return context

        except Exception as e:
            logger.error(f"Selection-Context-Erstellung fehlgeschlagen: {e}")

            # Fallback-Kontext
            return AgentSelectionContext(
                request_id=f"{orchestration_id}_{subtask.subtask_id}",
                orchestration_id=orchestration_id,
                subtask_id=subtask.subtask_id,
                user_id=user_id,
                tenant_id=tenant_id,
                task_type=subtask.task_type.value,
                task_payload=subtask.payload or {},
                security_level=SecurityLevel.INTERNAL
            )

    def _determine_data_classification(self, task_payload: dict[str, Any]) -> DataClassification | None:
        """Bestimmt Datenklassifizierung aus Task-Payload."""
        # Prüfe explizite Klassifizierung
        if "data_classification" in task_payload:
            try:
                return DataClassification(task_payload["data_classification"])
            except ValueError:
                pass

        # Heuristik-basierte Klassifizierung
        payload_str = str(task_payload).lower()

        if any(keyword in payload_str for keyword in ["pii", "personal", "gdpr", "privacy"]):
            return DataClassification.PII

        if any(keyword in payload_str for keyword in ["phi", "health", "medical", "hipaa"]):
            return DataClassification.PHI

        if any(keyword in payload_str for keyword in ["financial", "payment", "credit", "bank"]):
            return DataClassification.FINANCIAL

        if any(keyword in payload_str for keyword in ["confidential", "secret", "restricted"]):
            return DataClassification.CONFIDENTIAL

        return DataClassification.INTERNAL

    def _determine_security_level(
        self,
        task_payload: dict[str, Any],
        user_id: str | None,
        tenant_id: str | None
    ) -> SecurityLevel:
        """Bestimmt Security-Level für Task."""
        # Prüfe explizites Security-Level
        if "security_level" in task_payload:
            try:
                return SecurityLevel(task_payload["security_level"])
            except ValueError:
                pass

        # Default basierend auf Tenant und Data-Classification
        data_classification = self._determine_data_classification(task_payload)

        if data_classification in [DataClassification.RESTRICTED, DataClassification.PHI]:
            return SecurityLevel.RESTRICTED

        if data_classification in [DataClassification.CONFIDENTIAL, DataClassification.PII]:
            return SecurityLevel.CONFIDENTIAL

        if data_classification == DataClassification.FINANCIAL:
            return SecurityLevel.CONFIDENTIAL

        return SecurityLevel.INTERNAL

    async def _get_user_context(
        self,
        user_id: str | None,
        tenant_id: str | None
    ) -> tuple[list[str], list[str]]:
        """Holt User-Kontext (Groups und Clearances)."""
        try:
            if not user_id:
                return [], []

            # TODO: Implementiere echte User-Context-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116
            # Placeholder für User-Groups und Clearances
            user_groups = ["default_group"]
            user_clearances = ["basic_clearance"]

            if tenant_id:
                user_groups.append(f"tenant_{tenant_id}")

            return user_groups, user_clearances

        except Exception as e:
            logger.error(f"User-Context-Abfrage fehlgeschlagen: {e}")
            return [], []

    async def _get_compliance_requirements(
        self,
        tenant_id: str | None,
        data_classification: DataClassification | None
    ) -> list[Any]:
        """Bestimmt Compliance-Anforderungen."""
        try:
            requirements = []

            # Basis-Compliance für alle Tenants
            from policy_engine.compliance_framework import ComplianceStandard

            if data_classification == DataClassification.PII:
                requirements.append(ComplianceStandard.GDPR)

            if data_classification == DataClassification.PHI:
                requirements.append(ComplianceStandard.HIPAA)

            if data_classification == DataClassification.FINANCIAL:
                requirements.append(ComplianceStandard.PCI_DSS)

            return requirements

        except Exception as e:
            logger.error(f"Compliance-Requirements-Bestimmung fehlgeschlagen: {e}")
            return []

    def _contains_pii(self, task_payload: dict[str, Any]) -> bool:
        """Prüft ob Task PII-Daten enthält."""
        payload_str = str(task_payload).lower()
        pii_keywords = ["email", "phone", "address", "name", "ssn", "personal"]

        return any(keyword in payload_str for keyword in pii_keywords)

    def _contains_phi(self, task_payload: dict[str, Any]) -> bool:
        """Prüft ob Task PHI-Daten enthält."""
        payload_str = str(task_payload).lower()
        phi_keywords = ["medical", "health", "patient", "diagnosis", "treatment"]

        return any(keyword in payload_str for keyword in phi_keywords)

    def _get_geographic_restrictions(self, task_payload: dict[str, Any]) -> list[str]:
        """Extrahiert geografische Beschränkungen."""
        restrictions = []

        if "region" in task_payload:
            restrictions.append(task_payload["region"])

        if "country" in task_payload:
            restrictions.append(task_payload["country"])

        if "data_residency" in task_payload:
            restrictions.extend(task_payload["data_residency"])

        return restrictions

    async def _track_agent_selection(
        self,
        context: AgentSelectionContext,
        agent_matches: list[AgentMatch]
    ) -> None:
        """Trackt Agent-Selection für Compliance-Monitoring."""
        try:
            selected_agents = [match.agent_id for match in agent_matches]

            # Erstelle Mock-Compliance-Results für Tracking
            from .data_models import ComplianceResult

            compliance_results = [
                ComplianceResult(
                    is_compliant=True,
                    compliance_score=match.confidence_score,
                    checked_policies=[],
                    passed_constraints=[],
                    failed_constraints=[]
                )
                for match in agent_matches
            ]

            await self.compliance_monitor.track_agent_selection(
                context, selected_agents, compliance_results
            )

        except Exception as e:
            logger.error(f"Agent-Selection-Tracking fehlgeschlagen: {e}")

    async def _fallback_agent_selection(
        self,
        subtasks: list[SubtaskExecution]
    ) -> dict[str, list[AgentMatch]]:
        """Fallback auf Standard-Agent-Selection ohne Policy-Enforcement."""
        try:
            logger.warning("Fallback auf Standard-Agent-Selection ohne Policy-Enforcement")

            # Verwende Original-Agent-Matching aus Execution Engine
            # TODO: Implementiere Fallback-Logic - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116

            # Placeholder: Leere Agent-Assignments
            return {subtask.subtask_id: [] for subtask in subtasks}

        except Exception as e:
            logger.error(f"Fallback-Agent-Selection fehlgeschlagen: {e}")
            return {}

    def _update_performance_stats(self, integration_time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._integration_count += 1
        self._total_integration_time_ms += integration_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_integration_time = (
            self._total_integration_time_ms / self._integration_count
            if self._integration_count > 0 else 0.0
        )

        avg_policy_overhead = (
            self._policy_overhead_ms / self._integration_count
            if self._integration_count > 0 else 0.0
        )

        # Hole Statistiken von Komponenten
        selector_stats = self.policy_aware_selector.get_performance_stats()
        compliance_stats = self.compliance_checker.get_performance_stats()
        policy_engine_stats = self.policy_engine.get_statistics()
        monitor_stats = self.compliance_monitor.get_performance_stats()

        return {
            "integration": {
                "total_integrations": self._integration_count,
                "avg_integration_time_ms": avg_integration_time,
                "avg_policy_overhead_ms": avg_policy_overhead,
                "meets_policy_sla": avg_policy_overhead < 50.0,  # < 50ms SLA
                "policy_enforcement_enabled": self.enable_policy_enforcement
            },
            "policy_aware_selector": selector_stats,
            "compliance_checker": compliance_stats,
            "policy_engine": policy_engine_stats,
            "compliance_monitor": monitor_stats
        }
