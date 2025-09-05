# backend/services/enhanced_security_integration/policy_integration_layer.py
"""Policy Integration Layer für Enhanced Security Integration.

Integriert Enhanced Security mit Policy-aware Agent Selection (TASK 5)
und erweitert alle Services um Enterprise-Grade Security-Features.
"""

from __future__ import annotations

import time
from typing import Any

from kei_logging import get_logger
from security.rbac_abac_system import Action, RBACAuthorizationService, ResourceType
from services.policy_aware_selection import AgentSelectionContext, PolicyAwareAgentSelector
from services.policy_aware_selection import SecurityLevel as PolicySecurityLevel

from .data_models import SecurityContext, SecurityEvent, SecurityLevel
from .security_integration_engine import EnhancedSecurityIntegrationEngine

logger = get_logger(__name__)


class PolicyIntegrationLayer:
    """Policy Integration Layer für Enhanced Security Integration."""

    def __init__(
        self,
        security_integration_engine: EnhancedSecurityIntegrationEngine,
        policy_aware_selector: PolicyAwareAgentSelector,
        rbac_system: RBACAuthorizationService
    ):
        """Initialisiert Policy Integration Layer.

        Args:
            security_integration_engine: Enhanced Security Integration Engine
            policy_aware_selector: Policy-aware Agent Selector
            rbac_system: RBAC System
        """
        self.security_integration_engine = security_integration_engine
        self.policy_aware_selector = policy_aware_selector
        self.rbac_system = rbac_system

        # Integration-Konfiguration
        self.enable_policy_security_integration = True
        self.enable_enhanced_agent_security = True
        self.security_policy_enforcement_level = "strict"

        # Performance-Tracking
        self._integration_count = 0
        self._total_integration_time_ms = 0.0
        self._policy_security_check_time_ms = 0.0

        # Security-Events
        self._integration_security_events: list[SecurityEvent] = []

        logger.info("Policy Integration Layer initialisiert")

    async def start(self) -> None:
        """Startet Policy Integration Layer."""
        try:
            # Starte Security Integration Engine
            await self.security_integration_engine.start()

            logger.info("Policy Integration Layer gestartet")

        except Exception as e:
            logger.error(f"Policy Integration Layer start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Policy Integration Layer."""
        try:
            await self.security_integration_engine.stop()

            logger.info("Policy Integration Layer gestoppt")

        except Exception as e:
            logger.error(f"Policy Integration Layer stop fehlgeschlagen: {e}")

    async def perform_enhanced_agent_selection_with_security(
        self,
        agent_selection_context: AgentSelectionContext,
        subtasks: list[Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Führt Enhanced Agent Selection mit Security-Integration durch.

        Args:
            agent_selection_context: Agent Selection Context
            subtasks: Subtasks für Agent Selection
            security_context: Security Context

        Returns:
            Enhanced Agent Selection Result mit Security-Informationen
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "enhanced_agent_selection_with_security_started",
                "tenant_id": agent_selection_context.tenant_id,
                "user_id": security_context.user_id,
                "subtask_count": len(subtasks)
            })

            # 1. Comprehensive Security Check
            security_check_result = await self.security_integration_engine.perform_comprehensive_security_check(
                security_context=security_context,
                resource_type=ResourceType.AGENT,
                resource_id=f"agent_selection_{agent_selection_context.request_id}",
                action=Action.EXECUTE
            )

            if not security_check_result.is_secure:
                logger.warning({
                    "event": "agent_selection_security_check_failed",
                    "security_score": security_check_result.security_score,
                    "failed_checks": security_check_result.failed_checks
                })

                return {
                    "success": False,
                    "error": "Security check failed",
                    "security_result": security_check_result,
                    "agent_assignments": {}
                }

            # 2. Erweitere Agent Selection Context mit Security-Informationen
            enhanced_context = await self._enhance_agent_selection_context_with_security(
                agent_selection_context, security_context
            )

            # 3. Policy-aware Agent Selection mit Enhanced Security
            policy_start_time = time.time()

            agent_assignments = await self.policy_aware_selector.select_agents_with_policies(
                subtasks=subtasks,
                context=enhanced_context
            )

            policy_time_ms = (time.time() - policy_start_time) * 1000
            self._policy_security_check_time_ms += policy_time_ms

            # 4. Security-Validation der ausgewählten Agents
            validated_assignments = await self._validate_agent_assignments_security(
                agent_assignments, security_context
            )

            # 5. Erstelle Enhanced Result
            integration_time_ms = (time.time() - start_time) * 1000
            self._update_integration_performance_stats(integration_time_ms)

            result = {
                "success": True,
                "agent_assignments": validated_assignments,
                "security_result": security_check_result,
                "security_context": security_context,
                "performance_metrics": {
                    "total_time_ms": integration_time_ms,
                    "policy_security_check_time_ms": policy_time_ms,
                    "security_overhead_ms": security_check_result.overhead_ms
                },
                "security_events": security_check_result.security_events
            }

            logger.debug({
                "event": "enhanced_agent_selection_with_security_completed",
                "success": True,
                "total_assignments": sum(len(agents) for agents in validated_assignments.values()),
                "integration_time_ms": integration_time_ms,
                "security_score": security_check_result.security_score
            })

            return result

        except Exception as e:
            logger.error(f"Enhanced Agent Selection mit Security fehlgeschlagen: {e}")

            return {
                "success": False,
                "error": str(e),
                "agent_assignments": {}
            }

    async def _enhance_agent_selection_context_with_security(
        self,
        agent_context: AgentSelectionContext,
        security_context: SecurityContext
    ) -> AgentSelectionContext:
        """Erweitert Agent Selection Context mit Security-Informationen."""
        try:
            # Mappe Security Level
            security_level_mapping = {
                SecurityLevel.PUBLIC: PolicySecurityLevel.PUBLIC,
                SecurityLevel.INTERNAL: PolicySecurityLevel.INTERNAL,
                SecurityLevel.CONFIDENTIAL: PolicySecurityLevel.CONFIDENTIAL,
                SecurityLevel.RESTRICTED: PolicySecurityLevel.RESTRICTED,
                SecurityLevel.TOP_SECRET: PolicySecurityLevel.TOP_SECRET
            }

            mapped_security_level = security_level_mapping.get(
                security_context.security_level,
                PolicySecurityLevel.INTERNAL
            )

            # Erstelle Enhanced Context
            enhanced_context = AgentSelectionContext(
                request_id=agent_context.request_id,
                orchestration_id=agent_context.orchestration_id,
                subtask_id=agent_context.subtask_id,
                task_type=agent_context.task_type,
                user_id=security_context.user_id or agent_context.user_id,
                tenant_id=security_context.tenant_id or agent_context.tenant_id,
                user_groups=security_context.roles + agent_context.user_groups,
                user_clearances=security_context.clearances + agent_context.user_clearances,
                task_payload=agent_context.task_payload,
                required_capabilities=agent_context.required_capabilities,
                data_classification=agent_context.data_classification,
                contains_pii=agent_context.contains_pii,
                contains_phi=agent_context.contains_phi,
                geographic_restrictions=agent_context.geographic_restrictions,
                security_level=mapped_security_level,
                compliance_requirements=agent_context.compliance_requirements,
                max_execution_time_ms=agent_context.max_execution_time_ms,
                priority_level=agent_context.priority_level,
                correlation_id=security_context.request_id
            )

            return enhanced_context

        except Exception as e:
            logger.error(f"Agent Selection Context enhancement fehlgeschlagen: {e}")
            return agent_context

    async def _validate_agent_assignments_security(
        self,
        agent_assignments: dict[str, list[Any]],
        security_context: SecurityContext
    ) -> dict[str, list[Any]]:
        """Validiert Security der Agent-Assignments."""
        try:
            validated_assignments = {}

            for subtask_id, agent_matches in agent_assignments.items():
                validated_agents = []

                for agent_match in agent_matches:
                    # Security-Validation für jeden Agent
                    agent_security_valid = await self._validate_agent_security(
                        agent_match, security_context
                    )

                    if agent_security_valid:
                        validated_agents.append(agent_match)
                    else:
                        logger.warning({
                            "event": "agent_security_validation_failed",
                            "agent_id": agent_match.agent_id,
                            "subtask_id": subtask_id
                        })

                validated_assignments[subtask_id] = validated_agents

            return validated_assignments

        except Exception as e:
            logger.error(f"Agent assignments security validation fehlgeschlagen: {e}")
            return agent_assignments

    async def _validate_agent_security(
        self,
        agent_match: Any,
        security_context: SecurityContext
    ) -> bool:
        """Validiert Security eines einzelnen Agents."""
        try:
            # Führe Security-Check für Agent durch
            agent_security_check = await self.security_integration_engine.perform_comprehensive_security_check(
                security_context=security_context,
                resource_type=ResourceType.AGENT,
                resource_id=agent_match.agent_id,
                action=Action.EXECUTE
            )

            # Prüfe Security-Score-Threshold
            min_security_score = 0.7  # Mindestens 70% Security-Score

            return (
                agent_security_check.is_secure and
                agent_security_check.security_score >= min_security_score
            )

        except Exception as e:
            logger.error(f"Agent security validation fehlgeschlagen für {agent_match.agent_id}: {e}")
            return False

    async def enhance_orchestrator_service_security(
        self,
        orchestration_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Orchestrator Service um Enhanced Security.

        Args:
            orchestration_request: Orchestration Request
            security_context: Security Context

        Returns:
            Enhanced Orchestration Result
        """
        try:
            logger.debug({
                "event": "enhance_orchestrator_security_started",
                "orchestration_id": orchestration_request.get("orchestration_id"),
                "user_id": security_context.user_id
            })

            # 1. Security-Check für Orchestration
            orchestration_security_check = await self.security_integration_engine.perform_comprehensive_security_check(
                security_context=security_context,
                resource_type=ResourceType.TASK,
                resource_id=orchestration_request.get("orchestration_id", "unknown"),
                action=Action.EXECUTE
            )

            if not orchestration_security_check.is_secure:
                return {
                    "success": False,
                    "error": "Orchestration security check failed",
                    "security_result": orchestration_security_check
                }

            # 2. Erweitere Request mit Security-Informationen
            enhanced_request = orchestration_request.copy()
            enhanced_request["security_context"] = security_context
            enhanced_request["security_validation"] = orchestration_security_check

            # 3. Füge Security-Monitoring hinzu
            enhanced_request["security_monitoring"] = {
                "enable_threat_detection": True,
                "enable_audit_logging": True,
                "security_level": security_context.security_level.value
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "security_result": orchestration_security_check
            }

        except Exception as e:
            logger.error(f"Orchestrator Service Security Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_task_decomposition_security(
        self,
        decomposition_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Task Decomposition Engine um Enhanced Security.

        Args:
            decomposition_request: Task Decomposition Request
            security_context: Security Context

        Returns:
            Enhanced Task Decomposition Result
        """
        try:
            logger.debug({
                "event": "enhance_task_decomposition_security_started",
                "task_id": decomposition_request.get("task_id"),
                "user_id": security_context.user_id
            })

            # 1. Security-Check für Task Decomposition
            decomposition_security_check = await self.security_integration_engine.perform_comprehensive_security_check(
                security_context=security_context,
                resource_type=ResourceType.TASK,
                resource_id=decomposition_request.get("task_id", "unknown"),
                action=Action.CREATE
            )

            if not decomposition_security_check.is_secure:
                return {
                    "success": False,
                    "error": "Task decomposition security check failed",
                    "security_result": decomposition_security_check
                }

            # 2. Erweitere Request mit Security-Constraints
            enhanced_request = decomposition_request.copy()
            enhanced_request["security_constraints"] = {
                "security_level": security_context.security_level.value,
                "tenant_isolation": security_context.tenant_id is not None,
                "data_classification_required": True,
                "compliance_validation": True
            }

            # 3. Füge Security-Validierung für Subtasks hinzu
            enhanced_request["subtask_security_validation"] = {
                "validate_capabilities": True,
                "validate_permissions": True,
                "validate_data_access": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "security_result": decomposition_security_check
            }

        except Exception as e:
            logger.error(f"Task Decomposition Security Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_llm_client_security(
        self,
        llm_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert LLM Client Infrastructure um Enhanced Security.

        Args:
            llm_request: LLM Request
            security_context: Security Context

        Returns:
            Enhanced LLM Request Result
        """
        try:
            logger.debug({
                "event": "enhance_llm_client_security_started",
                "request_id": llm_request.get("request_id"),
                "user_id": security_context.user_id
            })

            # 1. Security-Check für LLM Request
            llm_security_check = await self.security_integration_engine.perform_comprehensive_security_check(
                security_context=security_context,
                resource_type=ResourceType.RPC,
                resource_id=llm_request.get("request_id", "unknown"),
                action=Action.EXECUTE
            )

            if not llm_security_check.is_secure:
                return {
                    "success": False,
                    "error": "LLM request security check failed",
                    "security_result": llm_security_check
                }

            # 2. Erweitere Request mit Security-Features
            enhanced_request = llm_request.copy()
            enhanced_request["security_features"] = {
                "input_sanitization": True,
                "output_filtering": True,
                "pii_detection": True,
                "content_validation": True
            }

            # 3. Füge Security-Monitoring hinzu
            enhanced_request["security_monitoring"] = {
                "log_requests": True,
                "monitor_responses": True,
                "detect_anomalies": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "security_result": llm_security_check
            }

        except Exception as e:
            logger.error(f"LLM Client Security Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _update_integration_performance_stats(self, integration_time_ms: float) -> None:
        """Aktualisiert Integration-Performance-Statistiken."""
        self._integration_count += 1
        self._total_integration_time_ms += integration_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_integration_time = (
            self._total_integration_time_ms / self._integration_count
            if self._integration_count > 0 else 0.0
        )

        avg_policy_security_check_time = (
            self._policy_security_check_time_ms / self._integration_count
            if self._integration_count > 0 else 0.0
        )

        # Hole Statistiken von Komponenten
        security_engine_stats = self.security_integration_engine.get_security_performance_stats()
        policy_selector_stats = self.policy_aware_selector.get_performance_stats()

        return {
            "policy_integration": {
                "total_integrations": self._integration_count,
                "avg_integration_time_ms": avg_integration_time,
                "avg_policy_security_check_time_ms": avg_policy_security_check_time,
                "meets_security_sla": avg_integration_time < 100.0,  # < 100ms SLA
                "policy_security_integration_enabled": self.enable_policy_security_integration,
                "enhanced_agent_security_enabled": self.enable_enhanced_agent_security
            },
            "security_integration_engine": security_engine_stats,
            "policy_aware_selector": policy_selector_stats,
            "integration_security_events": len(self._integration_security_events)
        }
