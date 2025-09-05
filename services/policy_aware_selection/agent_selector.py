# backend/services/policy_aware_selection/agent_selector.py
"""Policy-aware Agent Selector.

Erweitert das Orchestrator Service Agent-Matching um Policy-Enforcement,
Compliance-Checks und Multi-Tenant-Isolation.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from agents.registry.dynamic_registry import DynamicAgentRegistry
from kei_logging import get_logger, log_orchestrator_step, training_trace
from policy_engine.core_policy_engine import PolicyEngine

# from security.enhanced_auth_middleware import SecurityManager
from services.orchestrator.data_models import AgentMatch, SubtaskExecution

from .compliance_checker import AgentComplianceChecker
from .data_models import (
    AgentSelectionContext,
    ComplianceResult,
    PolicyEffect,
    PolicyEvaluationResult,
    SecurityLevel,
)
from .policy_engine import PolicyEnforcementEngine

logger = get_logger(__name__)


class PolicyAwareAgentSelector:
    """Policy-aware Agent Selector mit Compliance-Enforcement."""

    def __init__(
        self,
        agent_registry: DynamicAgentRegistry,
        policy_engine: PolicyEngine,
        security_manager: Any,
        compliance_checker: AgentComplianceChecker | None = None,
        policy_enforcement_engine: PolicyEnforcementEngine | None = None
    ):
        """Initialisiert Policy-aware Agent Selector.

        Args:
            agent_registry: Agent Registry für verfügbare Agents
            policy_engine: Policy Engine für Policy-Enforcement
            security_manager: Security Manager für Authentication/Authorization
            compliance_checker: Compliance Checker für Agent-Validation
            policy_enforcement_engine: Policy Enforcement Engine
        """
        self.agent_registry = agent_registry
        self.policy_engine = policy_engine
        self.security_manager = security_manager
        self.compliance_checker = compliance_checker or AgentComplianceChecker(policy_engine)
        self.policy_enforcement_engine = policy_enforcement_engine or PolicyEnforcementEngine(policy_engine)

        # Performance-Konfiguration (Issue #55 Performance Targets)
        self.agent_selection_timeout_seconds = float(os.getenv("KEI_ORCHESTRATOR_AGENT_SELECTION_TIMEOUT_SECONDS", "1.0"))
        self.policy_check_timeout_ms = 50.0  # < 50ms SLA
        self.max_concurrent_checks = 10
        self.enable_policy_caching = True

        # Multi-Tenant-Konfiguration
        self.enable_tenant_isolation = True
        self.default_security_level = SecurityLevel.INTERNAL

        # Performance-Tracking
        self._selection_count = 0
        self._total_selection_time_ms = 0.0
        self._policy_check_time_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("Policy-aware Agent Selector initialisiert")

    @training_trace(context={"component": "policy_aware_selection", "phase": "agent_selection"})
    async def select_agents_with_policies(
        self,
        subtasks: list[SubtaskExecution],
        context: AgentSelectionContext,
        max_agents_per_subtask: int = 3
    ) -> dict[str, list[AgentMatch]]:
        """Wählt Agents mit Policy-Enforcement aus.

        Args:
            subtasks: Liste von Subtasks
            context: Agent-Selection-Kontext
            max_agents_per_subtask: Maximale Anzahl Agents pro Subtask

        Returns:
            Dictionary: subtask_id -> [AgentMatch] (policy-compliant)
        """
        start_time = time.time()

        try:
            # Schritt 1: Policy-aware Agent Selection gestartet
            log_orchestrator_step(
                "Starting Policy-aware Agent Selection",
                "agent_call",
                subtask_count=len(subtasks),
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                security_level=context.security_level.value,
                max_agents_per_subtask=max_agents_per_subtask,
                request_id=context.request_id
            )

            logger.info({
                "event": "policy_aware_selection_started",
                "subtask_count": len(subtasks),
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "security_level": context.security_level.value
            })

            # 1. Hole verfügbare Agents
            log_orchestrator_step(
                "Discovering Available Agents",
                "agent_call",
                context_tenant=context.tenant_id,
                security_level=context.security_level.value
            )

            available_agents = await self._get_available_agents(context)

            log_orchestrator_step(
                "Available Agents Discovered",
                "agent_call",
                agent_count=len(available_agents),
                agent_types=[agent.get("agent_type", "unknown") for agent in available_agents[:5]]  # Erste 5 für Übersicht
            )

            if not available_agents:
                logger.warning("Keine verfügbaren Agents gefunden")
                return {}

            # 2. Policy-Evaluation für Kontext
            log_orchestrator_step(
                "Evaluating Policies for Context",
                "policy_check",
                tenant_id=context.tenant_id,
                security_level=context.security_level.value,
                request_id=context.request_id
            )

            policy_evaluation = await self._evaluate_policies_for_context(context)

            if policy_evaluation.decision == PolicyEffect.DENY:
                log_orchestrator_step(
                    "Policy Denied Agent Selection",
                    "policy_check",
                    decision=policy_evaluation.decision.value,
                    violated_constraints=policy_evaluation.violated_constraints,
                    context=context.request_id
                )
                logger.warning({
                    "event": "policy_denied_agent_selection",
                    "context": context.request_id,
                    "violated_constraints": policy_evaluation.violated_constraints
                })
                return {}

            log_orchestrator_step(
                "Policy Evaluation Successful",
                "policy_check",
                decision=policy_evaluation.decision.value,
                applicable_policies=len(policy_evaluation.evaluated_policies),
                constraints_count=len(policy_evaluation.matched_constraints)
            )

            # 3. Parallel Agent-Selection für alle Subtasks
            selection_tasks = [
                self._select_agents_for_subtask(
                    subtask, available_agents, context, policy_evaluation, max_agents_per_subtask
                )
                for subtask in subtasks
            ]

            selection_results = await asyncio.gather(*selection_tasks, return_exceptions=True)

            # 4. Ergebnisse zusammenfassen
            agent_assignments = {}
            for i, result in enumerate(selection_results):
                if isinstance(result, Exception):
                    logger.error(f"Agent-Selection für Subtask {subtasks[i].subtask_id} fehlgeschlagen: {result}")
                    agent_assignments[subtasks[i].subtask_id] = []
                else:
                    agent_assignments[subtasks[i].subtask_id] = result

            # 5. Performance-Tracking
            selection_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(selection_time_ms)

            logger.info({
                "event": "policy_aware_selection_completed",
                "subtask_count": len(subtasks),
                "total_matches": sum(len(matches) for matches in agent_assignments.values()),
                "selection_time_ms": selection_time_ms,
                "policy_check_time_ms": self._policy_check_time_ms,
                "cache_hit_rate": self._get_cache_hit_rate()
            })

            return agent_assignments

        except Exception as e:
            logger.error(f"Policy-aware Agent-Selection fehlgeschlagen: {e}")
            return {}

    async def _get_available_agents(self, context: AgentSelectionContext) -> list[dict[str, Any]]:
        """Holt verfügbare Agents mit Tenant-Filtering."""
        try:
            # Hole alle Agents
            all_agents_raw = await self.agent_registry.list_agents()

            # Konvertiere zu Dictionary falls Liste zurückgegeben wird
            if isinstance(all_agents_raw, list):
                all_agents = {
                    getattr(agent, "id", f"agent_{i}"): agent
                    for i, agent in enumerate(all_agents_raw)
                }
            else:
                all_agents = all_agents_raw

            available_agents = []
            for agent_id, agent_info in all_agents.items():
                # Tenant-Isolation prüfen
                if self.enable_tenant_isolation and context.tenant_id:
                    agent_tenant = getattr(agent_info, "tenant_id", None)
                    if agent_tenant and agent_tenant != context.tenant_id:
                        continue  # Agent gehört zu anderem Tenant

                # Basis-Agent-Informationen
                agent_data = {
                    "agent_id": agent_id,
                    "agent_type": getattr(agent_info, "agent_type", "unknown"),
                    "capabilities": getattr(agent_info, "capabilities", []),
                    "tenant_id": getattr(agent_info, "tenant_id", None),
                    "security_level": getattr(agent_info, "security_level", self.default_security_level.value),
                    "clearances": getattr(agent_info, "clearances", []),
                    "compliance_status": getattr(agent_info, "compliance_status", {}),
                    "deployment_region": getattr(agent_info, "deployment_region", None),
                    "current_load": self._estimate_agent_load(agent_id),
                    "availability_score": self._calculate_availability_score(agent_info),
                    "avg_response_time_ms": getattr(agent_info, "avg_response_time_ms", 200.0),
                    "success_rate": getattr(agent_info, "success_rate", 0.95)
                }

                available_agents.append(agent_data)

            return available_agents

        except Exception as e:
            logger.error(f"Fehler beim Holen verfügbarer Agents: {e}")
            return []

    async def _evaluate_policies_for_context(self, context: AgentSelectionContext) -> PolicyEvaluationResult:
        """Evaluiert Policies für Selection-Kontext."""
        policy_start_time = time.time()

        try:
            # Hole relevante Policies (Mock für Demo)
            policies = []  # TODO: Implementiere echte Policy-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116

            # Policy-Evaluation
            evaluation_result = await self.policy_enforcement_engine.evaluate_policies(
                policies=policies,
                context=context
            )

            # Performance-Tracking
            policy_time_ms = (time.time() - policy_start_time) * 1000
            self._policy_check_time_ms += policy_time_ms

            return evaluation_result

        except Exception as e:
            logger.error(f"Policy-Evaluation fehlgeschlagen: {e}")

            # Fallback: ALLOW mit niedrigem Confidence
            return PolicyEvaluationResult(
                decision=PolicyEffect.ALLOW,
                confidence=0.5,
                evaluated_policies=[],
                matched_constraints=[],
                violated_constraints=[],
                evaluation_time_ms=(time.time() - policy_start_time) * 1000
            )

    async def _select_agents_for_subtask(
        self,
        subtask: SubtaskExecution,
        available_agents: list[dict[str, Any]],
        context: AgentSelectionContext,
        policy_evaluation: PolicyEvaluationResult,
        max_agents: int
    ) -> list[AgentMatch]:
        """Wählt policy-compliant Agents für einzelnen Subtask."""
        try:
            agent_matches = []

            # Parallel Compliance-Checks für alle Agents
            compliance_tasks = [
                self._check_agent_compliance(agent_info, subtask, context)
                for agent_info in available_agents
            ]

            # Limitiere Concurrent-Checks für Performance
            compliance_results = []
            for i in range(0, len(compliance_tasks), self.max_concurrent_checks):
                batch = compliance_tasks[i:i + self.max_concurrent_checks]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                compliance_results.extend(batch_results)

            # Verarbeite Compliance-Results
            for i, compliance_result in enumerate(compliance_results):
                if isinstance(compliance_result, Exception):
                    logger.warning(f"Compliance-Check für Agent {available_agents[i]['agent_id']} fehlgeschlagen: {compliance_result}")
                    continue

                if not compliance_result.is_compliant:
                    logger.debug(f"Agent {available_agents[i]['agent_id']} nicht policy-compliant")
                    continue

                # Erstelle AgentMatch für compliant Agent
                agent_info = available_agents[i]
                agent_match = await self._create_agent_match(
                    agent_info, subtask, compliance_result, policy_evaluation
                )

                agent_matches.append(agent_match)

            # Sortiere nach Match-Score und limitiere
            agent_matches.sort(key=lambda x: x.match_score, reverse=True)
            return agent_matches[:max_agents]

        except Exception as e:
            logger.error(f"Agent-Selection für Subtask {subtask.subtask_id} fehlgeschlagen: {e}")
            return []

    async def _check_agent_compliance(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution,
        context: AgentSelectionContext
    ) -> ComplianceResult:
        """Prüft Agent-Compliance für Subtask."""
        try:
            # Timeout für Policy-Check
            compliance_task = self.compliance_checker.check_agent_compliance(
                agent_info, subtask, context
            )

            compliance_result = await asyncio.wait_for(
                compliance_task,
                timeout=self.policy_check_timeout_ms / 1000.0
            )

            return compliance_result

        except TimeoutError:
            logger.warning(f"Compliance-Check Timeout für Agent {agent_info['agent_id']}")

            # Fallback: Non-compliant bei Timeout
            return ComplianceResult(
                is_compliant=False,
                compliance_score=0.0,
                checked_policies=[],
                passed_constraints=[],
                failed_constraints=["timeout"],
                check_duration_ms=self.policy_check_timeout_ms
            )

        except Exception as e:
            logger.error(f"Compliance-Check fehlgeschlagen für Agent {agent_info['agent_id']}: {e}")

            return ComplianceResult(
                is_compliant=False,
                compliance_score=0.0,
                checked_policies=[],
                passed_constraints=[],
                failed_constraints=["check_error"],
                check_duration_ms=0.0
            )

    async def _create_agent_match(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution,
        compliance_result: ComplianceResult,
        policy_evaluation: PolicyEvaluationResult
    ) -> AgentMatch:
        """Erstellt AgentMatch mit Policy-Informationen."""
        # Basis-Match-Score aus Capability-Matching
        capability_score = self._calculate_capability_match_score(agent_info, subtask)

        # Policy-Compliance-Score
        compliance_score = compliance_result.compliance_score

        # Kombinierter Match-Score
        match_score = (capability_score * 0.6) + (compliance_score * 0.4)

        # Capability-Matching
        required_caps = set(cap.lower() for cap in subtask.required_capabilities)
        agent_caps = set(cap.lower() for cap in agent_info.get("capabilities", []))

        matched_caps = list(required_caps.intersection(agent_caps))
        missing_caps = list(required_caps.difference(agent_caps))

        coverage = len(matched_caps) / len(required_caps) if required_caps else 1.0

        return AgentMatch(
            agent_id=agent_info["agent_id"],
            agent_type=agent_info["agent_type"],
            match_score=match_score,
            matched_capabilities=matched_caps,
            missing_capabilities=missing_caps,
            capability_coverage=coverage,
            estimated_execution_time_ms=subtask.estimated_duration_minutes * 60 * 1000 if subtask.estimated_duration_minutes else 30000.0,
            confidence_score=compliance_result.compliance_score,
            current_load=agent_info["current_load"],
            availability_score=agent_info["availability_score"],
            queue_length=0,  # TODO: Implementiere echte Queue-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116
            specialization_score=self._calculate_specialization_score(agent_info, subtask),
            historical_success_rate=agent_info["success_rate"]
        )

    def _calculate_capability_match_score(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution
    ) -> float:
        """Berechnet Capability-Match-Score."""
        required_caps = set(cap.lower() for cap in subtask.required_capabilities)
        agent_caps = set(cap.lower() for cap in agent_info.get("capabilities", []))

        if not required_caps:
            return 1.0

        matched_caps = required_caps.intersection(agent_caps)
        return len(matched_caps) / len(required_caps)

    def _calculate_specialization_score(
        self,
        agent_info: dict[str, Any],
        subtask: SubtaskExecution
    ) -> float:
        """Berechnet Spezialisierungs-Score."""
        agent_type = agent_info.get("agent_type", "").lower()
        task_type = subtask.task_type.value.lower()

        # Direkte Matches
        if agent_type == task_type:
            return 1.0

        # Verwandte Matches
        related_matches = {
            "data_processing": ["nlp_analysis", "batch_job"],
            "nlp_analysis": ["data_processing", "agent_execution"],
            "agent_execution": ["tool_call", "workflow"]
        }

        if task_type in related_matches.get(agent_type, []):
            return 0.7

        return 0.5

    def _estimate_agent_load(self, agent_id: str) -> float:
        """Schätzt aktuelle Agent-Load."""
        # TODO: Implementiere echte Load-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116
        return 0.3  # 30% Load

    def _calculate_availability_score(self, agent_info: Any) -> float:
        """Berechnet Verfügbarkeits-Score."""
        # TODO: Implementiere echte Verfügbarkeits-Berechnung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/116
        return 0.9  # 90% verfügbar

    def _update_performance_stats(self, selection_time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._selection_count += 1
        self._total_selection_time_ms += selection_time_ms

    def _get_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate."""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests == 0:
            return 0.0
        return self._cache_hits / total_requests

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_selection_time = (
            self._total_selection_time_ms / self._selection_count
            if self._selection_count > 0 else 0.0
        )

        avg_policy_check_time = (
            self._policy_check_time_ms / self._selection_count
            if self._selection_count > 0 else 0.0
        )

        return {
            "total_selections": self._selection_count,
            "avg_selection_time_ms": avg_selection_time,
            "avg_policy_check_time_ms": avg_policy_check_time,
            "meets_policy_sla": avg_policy_check_time < self.policy_check_timeout_ms,
            "cache_hit_rate": self._get_cache_hit_rate(),
            "policy_check_timeout_ms": self.policy_check_timeout_ms
        }
