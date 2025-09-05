# backend/services/policy_aware_selection/policy_engine.py
"""Policy Enforcement Engine.

Implementiert Real-time Policy-Validation, Multi-Tenant Policy-Management
und Policy-Caching für Performance-Optimierung.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from kei_logging import get_logger
from policy_engine.core_policy_engine import PolicyEngine

from .data_models import (
    AgentPolicy,
    AgentSelectionContext,
    PolicyConstraint,
    PolicyEffect,
    PolicyEvaluationResult,
    PolicyPriority,
    PolicyType,
)

logger = get_logger(__name__)


class PolicyEnforcementEngine:
    """Policy Enforcement Engine für Real-time Policy-Validation."""

    def __init__(self, policy_engine: PolicyEngine):
        """Initialisiert Policy Enforcement Engine.

        Args:
            policy_engine: Policy Engine für Policy-Abfragen
        """
        self.policy_engine = policy_engine

        # Policy-Cache für Performance
        self._policy_cache: dict[str, Any] = {}
        self._cache_ttl_seconds = 300  # 5 Minuten
        self._cache_timestamps: dict[str, float] = {}

        # Performance-Konfiguration
        self.enable_caching = True
        self.max_evaluation_time_ms = 50.0  # < 50ms SLA
        self.parallel_evaluation = True

        # Performance-Tracking
        self._evaluation_count = 0
        self._total_evaluation_time_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("Policy Enforcement Engine initialisiert")

    async def evaluate_policies(
        self,
        policies: list[AgentPolicy],
        context: AgentSelectionContext
    ) -> PolicyEvaluationResult:
        """Evaluiert Policies für Agent-Selection-Kontext.

        Args:
            policies: Liste von Policies
            context: Agent-Selection-Kontext

        Returns:
            Policy-Evaluation-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "policy_evaluation_started",
                "policy_count": len(policies),
                "context_id": context.request_id,
                "tenant_id": context.tenant_id
            })

            # 1. Cache-Check
            cache_key = self._generate_cache_key(policies, context)
            cached_result = self._get_cached_result(cache_key)

            if cached_result:
                self._cache_hits += 1
                logger.debug("Policy-Evaluation aus Cache")
                return cached_result

            self._cache_misses += 1

            # 2. Policy-Evaluation
            if self.parallel_evaluation and len(policies) > 1:
                evaluation_result = await self._evaluate_policies_parallel(policies, context)
            else:
                evaluation_result = await self._evaluate_policies_sequential(policies, context)

            # 3. Cache-Update
            if self.enable_caching:
                self._cache_result(cache_key, evaluation_result)

            # 4. Performance-Tracking
            evaluation_time_ms = (time.time() - start_time) * 1000
            evaluation_result.evaluation_time_ms = evaluation_time_ms
            evaluation_result.cache_hit_rate = self._get_cache_hit_rate()

            self._update_performance_stats(evaluation_time_ms)

            logger.debug({
                "event": "policy_evaluation_completed",
                "decision": evaluation_result.decision.value,
                "confidence": evaluation_result.confidence,
                "evaluation_time_ms": evaluation_time_ms,
                "cache_hit": False
            })

            return evaluation_result

        except Exception as e:
            logger.error(f"Policy-Evaluation fehlgeschlagen: {e}")

            # Fallback: DENY mit niedrigem Confidence
            return PolicyEvaluationResult(
                decision=PolicyEffect.DENY,
                confidence=0.0,
                evaluated_policies=[],
                matched_constraints=[],
                violated_constraints=["evaluation_error"],
                evaluation_time_ms=(time.time() - start_time) * 1000
            )

    async def _evaluate_policies_parallel(
        self,
        policies: list[AgentPolicy],
        context: AgentSelectionContext
    ) -> PolicyEvaluationResult:
        """Evaluiert Policies parallel."""
        # Erstelle Evaluation-Tasks
        evaluation_tasks = [
            self._evaluate_single_policy(policy, context)
            for policy in policies
        ]

        # Führe parallel aus
        policy_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Aggregiere Ergebnisse
        return self._aggregate_policy_results(policies, policy_results, context)

    async def _evaluate_policies_sequential(
        self,
        policies: list[AgentPolicy],
        context: AgentSelectionContext
    ) -> PolicyEvaluationResult:
        """Evaluiert Policies sequenziell."""
        policy_results = []

        for policy in policies:
            try:
                result = await self._evaluate_single_policy(policy, context)
                policy_results.append(result)
            except Exception as e:
                logger.error(f"Policy-Evaluation fehlgeschlagen für {policy.policy_id}: {e}")
                policy_results.append(e)

        return self._aggregate_policy_results(policies, policy_results, context)

    async def _evaluate_single_policy(
        self,
        policy: AgentPolicy,
        context: AgentSelectionContext
    ) -> dict[str, Any]:
        """Evaluiert einzelne Policy."""
        try:
            # Prüfe Policy-Gültigkeit
            if not self._is_policy_valid(policy):
                return {
                    "policy_id": policy.policy_id,
                    "decision": PolicyEffect.ALLOW,
                    "reason": "policy_not_valid",
                    "matched_constraints": [],
                    "violated_constraints": []
                }

            # Prüfe Policy-Scope
            if not self._is_policy_applicable(policy, context):
                return {
                    "policy_id": policy.policy_id,
                    "decision": PolicyEffect.ALLOW,
                    "reason": "policy_not_applicable",
                    "matched_constraints": [],
                    "violated_constraints": []
                }

            # Evaluiere Policy-Constraints
            matched_constraints = []
            violated_constraints = []

            for constraint in policy.constraints:
                if self._evaluate_constraint(constraint, context):
                    matched_constraints.append(constraint.constraint_id)
                else:
                    violated_constraints.append(constraint.constraint_id)

            # Bestimme Policy-Decision
            decision = self._determine_policy_decision(
                policy, matched_constraints, violated_constraints
            )

            return {
                "policy_id": policy.policy_id,
                "decision": decision,
                "matched_constraints": matched_constraints,
                "violated_constraints": violated_constraints
            }

        except Exception as e:
            logger.error(f"Einzelne Policy-Evaluation fehlgeschlagen: {e}")
            return {
                "policy_id": policy.policy_id,
                "decision": PolicyEffect.DENY,
                "reason": "evaluation_error",
                "matched_constraints": [],
                "violated_constraints": ["evaluation_error"]
            }

    def _is_policy_valid(self, policy: AgentPolicy) -> bool:
        """Prüft ob Policy gültig ist."""
        if not policy.is_active:
            return False

        # Prüfe Gültigkeitszeitraum
        from datetime import datetime
        now = datetime.utcnow()

        if policy.valid_from and now < policy.valid_from:
            return False

        if policy.valid_until and now > policy.valid_until:
            return False

        return True

    def _is_policy_applicable(self, policy: AgentPolicy, context: AgentSelectionContext) -> bool:
        """Prüft ob Policy auf Kontext anwendbar ist."""
        # Tenant-Check
        if policy.tenant_id and policy.tenant_id != context.tenant_id:
            return False

        # User-Group-Check
        if policy.user_groups:
            if not any(group in context.user_groups for group in policy.user_groups):
                return False

        # Task-Type-Check
        if policy.task_types:
            if context.task_type not in policy.task_types:
                return False

        return True

    def _evaluate_constraint(self, constraint: PolicyConstraint, context: AgentSelectionContext) -> bool:
        """Evaluiert einzelnen Constraint."""
        try:
            # Basis-Constraint-Evaluation basierend auf Type
            if constraint.policy_type == PolicyType.SECURITY:
                return self._evaluate_security_constraint(constraint, context)

            if constraint.policy_type == PolicyType.TENANT:
                return self._evaluate_tenant_constraint(constraint, context)

            if constraint.policy_type == PolicyType.GEOGRAPHIC:
                return self._evaluate_geographic_constraint(constraint, context)

            if constraint.policy_type == PolicyType.DATA_CLASSIFICATION:
                return self._evaluate_data_classification_constraint(constraint, context)

            # Default: Constraint ist erfüllt
            return True

        except Exception as e:
            logger.error(f"Constraint-Evaluation fehlgeschlagen: {e}")
            return False

    def _evaluate_security_constraint(self, constraint: PolicyConstraint, context: AgentSelectionContext) -> bool:
        """Evaluiert Security-Constraint."""
        # Prüfe Security-Level-Anforderungen
        if constraint.min_security_level:
            # Vereinfachte Security-Level-Prüfung
            context_security_level = context.security_level
            required_level = constraint.min_security_level

            # Security-Level-Hierarchie
            from .data_models import SecurityLevel
            security_hierarchy = {
                SecurityLevel.PUBLIC: 0,
                SecurityLevel.INTERNAL: 1,
                SecurityLevel.CONFIDENTIAL: 2,
                SecurityLevel.RESTRICTED: 3,
                SecurityLevel.TOP_SECRET: 4
            }

            context_level_value = security_hierarchy.get(context_security_level, 0)
            required_level_value = security_hierarchy.get(required_level, 0)

            if context_level_value < required_level_value:
                return False

        # Prüfe Clearance-Anforderungen
        if constraint.required_clearances:
            context_clearances = set(context.user_clearances)
            required_clearances = set(constraint.required_clearances)

            if not required_clearances.issubset(context_clearances):
                return False

        return True

    def _evaluate_tenant_constraint(self, constraint: PolicyConstraint, context: AgentSelectionContext) -> bool:
        """Evaluiert Tenant-Constraint."""
        # Tenant-Isolation prüfen
        if "tenant_isolation" in constraint.conditions:
            if constraint.conditions["tenant_isolation"] and not context.tenant_id:
                return False

        return True

    def _evaluate_geographic_constraint(self, constraint: PolicyConstraint, context: AgentSelectionContext) -> bool:
        """Evaluiert Geographic-Constraint."""
        # Prüfe geografische Beschränkungen
        if constraint.allowed_regions and context.geographic_restrictions:
            allowed_regions = set(constraint.allowed_regions)
            context_restrictions = set(context.geographic_restrictions)

            # Mindestens eine erlaubte Region muss in Kontext-Restrictions sein
            if not allowed_regions.intersection(context_restrictions):
                return False

        return True

    def _evaluate_data_classification_constraint(self, constraint: PolicyConstraint, context: AgentSelectionContext) -> bool:
        """Evaluiert Data-Classification-Constraint."""
        # Prüfe Datenklassifizierungs-Anforderungen
        if constraint.data_classification_constraints and context.data_classification:
            allowed_classifications = constraint.data_classification_constraints

            if context.data_classification not in allowed_classifications:
                return False

        return True

    def _determine_policy_decision(
        self,
        policy: AgentPolicy,
        matched_constraints: list[str],
        violated_constraints: list[str]
    ) -> PolicyEffect:
        """Bestimmt Policy-Decision basierend auf Constraint-Evaluation."""
        # Wenn Violations vorhanden sind
        if violated_constraints:
            # Prüfe Priorität der verletzten Constraints
            critical_violations = [
                c for c in policy.constraints
                if c.constraint_id in violated_constraints and c.priority == PolicyPriority.CRITICAL
            ]

            if critical_violations:
                return PolicyEffect.DENY

            # Hohe Priorität führt zu AVOID
            high_priority_violations = [
                c for c in policy.constraints
                if c.constraint_id in violated_constraints and c.priority == PolicyPriority.HIGH
            ]

            if high_priority_violations:
                return PolicyEffect.AVOID

        # Wenn alle Constraints erfüllt sind
        if len(matched_constraints) == len(policy.constraints):
            return PolicyEffect.ALLOW

        # Partial Match
        return PolicyEffect.PREFER if len(matched_constraints) > len(violated_constraints) else PolicyEffect.AVOID

    def _aggregate_policy_results(
        self,
        policies: list[AgentPolicy],
        policy_results: list[Any],
        context: AgentSelectionContext
    ) -> PolicyEvaluationResult:
        """Aggregiert Policy-Results zu finalem Evaluation-Result."""
        evaluated_policies = []
        matched_constraints = []
        violated_constraints = []

        # Sammle alle Results
        for i, result in enumerate(policy_results):
            if isinstance(result, Exception):
                logger.error(f"Policy-Result-Fehler: {result}")
                violated_constraints.append(f"policy_{i}_error")
                continue

            evaluated_policies.append(result["policy_id"])
            matched_constraints.extend(result["matched_constraints"])
            violated_constraints.extend(result["violated_constraints"])

        # Bestimme finale Decision
        final_decision = self._determine_final_decision(policy_results)

        # Berechne Confidence
        confidence = self._calculate_confidence(policy_results, final_decision)

        return PolicyEvaluationResult(
            decision=final_decision,
            confidence=confidence,
            evaluated_policies=evaluated_policies,
            matched_constraints=matched_constraints,
            violated_constraints=violated_constraints,
            context=context
        )

    def _determine_final_decision(self, policy_results: list[Any]) -> PolicyEffect:
        """Bestimmt finale Decision aus allen Policy-Results."""
        decisions = []

        for result in policy_results:
            if isinstance(result, Exception):
                decisions.append(PolicyEffect.DENY)
            else:
                decisions.append(result["decision"])

        # Prioritäts-basierte Decision
        if PolicyEffect.DENY in decisions:
            return PolicyEffect.DENY

        if PolicyEffect.AVOID in decisions:
            return PolicyEffect.AVOID

        if PolicyEffect.REQUIRE in decisions:
            return PolicyEffect.REQUIRE

        if PolicyEffect.PREFER in decisions:
            return PolicyEffect.PREFER

        return PolicyEffect.ALLOW

    def _calculate_confidence(self, policy_results: list[Any], final_decision: PolicyEffect) -> float:
        """Berechnet Confidence für finale Decision."""
        if not policy_results:
            return 0.0

        # Basis-Confidence basierend auf erfolgreichen Evaluations
        successful_evaluations = sum(
            1 for result in policy_results
            if not isinstance(result, Exception)
        )

        base_confidence = successful_evaluations / len(policy_results)

        # Decision-spezifische Adjustments
        decision_confidence_map = {
            PolicyEffect.DENY: 0.9,      # Hohe Confidence bei DENY
            PolicyEffect.REQUIRE: 0.8,   # Hohe Confidence bei REQUIRE
            PolicyEffect.ALLOW: 0.7,     # Mittlere Confidence bei ALLOW
            PolicyEffect.PREFER: 0.6,    # Niedrigere Confidence bei PREFER
            PolicyEffect.AVOID: 0.5      # Niedrigste Confidence bei AVOID
        }

        decision_factor = decision_confidence_map.get(final_decision, 0.5)

        return base_confidence * decision_factor

    def _generate_cache_key(self, policies: list[AgentPolicy], context: AgentSelectionContext) -> str:
        """Generiert Cache-Key für Policy-Evaluation."""
        policy_ids = sorted([p.policy_id for p in policies])
        context_key = f"{context.tenant_id}_{context.task_type}_{context.security_level.value}"

        return f"policy_eval_{hash(tuple(policy_ids))}_{hash(context_key)}"

    def _get_cached_result(self, cache_key: str) -> PolicyEvaluationResult | None:
        """Holt cached Policy-Evaluation-Result."""
        if not self.enable_caching:
            return None

        if cache_key not in self._policy_cache:
            return None

        # Prüfe Cache-TTL
        cache_time = self._cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > self._cache_ttl_seconds:
            # Cache expired
            del self._policy_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._policy_cache[cache_key]

    def _cache_result(self, cache_key: str, result: PolicyEvaluationResult) -> None:
        """Cached Policy-Evaluation-Result."""
        if not self.enable_caching:
            return

        self._policy_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        # Cache-Cleanup (einfach)
        if len(self._policy_cache) > 1000:  # Max 1000 Cache-Entries
            oldest_key = min(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
            del self._policy_cache[oldest_key]
            del self._cache_timestamps[oldest_key]

    def _get_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate."""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests == 0:
            return 0.0
        return self._cache_hits / total_requests

    def _update_performance_stats(self, evaluation_time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._evaluation_count += 1
        self._total_evaluation_time_ms += evaluation_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_evaluation_time = (
            self._total_evaluation_time_ms / self._evaluation_count
            if self._evaluation_count > 0 else 0.0
        )

        return {
            "total_evaluations": self._evaluation_count,
            "avg_evaluation_time_ms": avg_evaluation_time,
            "meets_sla": avg_evaluation_time < self.max_evaluation_time_ms,
            "cache_hit_rate": self._get_cache_hit_rate(),
            "cache_entries": len(self._policy_cache),
            "parallel_evaluation": self.parallel_evaluation
        }
