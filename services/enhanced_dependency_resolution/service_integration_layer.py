# backend/services/enhanced_dependency_resolution/service_integration_layer.py
"""Service Integration Layer für Enhanced Dependency Resolution.

Integriert Enhanced Dependency Resolution mit allen bestehenden Services
und erweitert sie um Enterprise-Grade Dependency-Management-Features.
"""

from __future__ import annotations

import time
from typing import Any

from kei_logging import get_logger
from services.enhanced_quotas_limits_management import EnhancedQuotaManagementEngine
from services.enhanced_security_integration import (
    EnhancedSecurityIntegrationEngine,
    SecurityContext,
)

from .data_models import DependencyRelation, DependencyType, TaskDependencyContext
from .dependency_graph_engine import EnhancedDependencyGraphEngine
from .dependency_resolution_engine import EnhancedDependencyResolutionEngine

logger = get_logger(__name__)


class DependencyServiceIntegrationLayer:
    """Service Integration Layer für Enhanced Dependency Resolution."""

    def __init__(
        self,
        dependency_resolution_engine: EnhancedDependencyResolutionEngine,
        dependency_graph_engine: EnhancedDependencyGraphEngine,
        security_integration_engine: EnhancedSecurityIntegrationEngine | None = None,
        quota_management_engine: EnhancedQuotaManagementEngine | None = None
    ):
        """Initialisiert Service Integration Layer.

        Args:
            dependency_resolution_engine: Dependency Resolution Engine
            dependency_graph_engine: Dependency Graph Engine
            security_integration_engine: Security Integration Engine
            quota_management_engine: Quota Management Engine
        """
        self.dependency_resolution_engine = dependency_resolution_engine
        self.dependency_graph_engine = dependency_graph_engine
        self.security_integration_engine = security_integration_engine
        self.quota_management_engine = quota_management_engine

        # Integration-Konfiguration
        self.enable_task_decomposition_integration = True
        self.enable_orchestrator_integration = True
        self.enable_agent_selection_integration = True
        self.enable_performance_prediction_integration = True
        self.enable_llm_client_integration = True

        # Performance-Tracking
        self._integration_count = 0
        self._total_integration_time_ms = 0.0
        self._dependency_aware_operations = 0

        logger.info("Dependency Service Integration Layer initialisiert")

    async def start(self) -> None:
        """Startet Service Integration Layer."""
        try:
            # Starte Dependency Engines
            await self.dependency_resolution_engine.start()

            logger.info("Dependency Service Integration Layer gestartet")

        except Exception as e:
            logger.error(f"Service Integration Layer start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Service Integration Layer."""
        try:
            await self.dependency_resolution_engine.stop()

            logger.info("Dependency Service Integration Layer gestoppt")

        except Exception as e:
            logger.error(f"Service Integration Layer stop fehlgeschlagen: {e}")

    async def enhance_task_decomposition_with_dependencies(
        self,
        task_decomposition_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Task Decomposition um Dependency-Management.

        Args:
            task_decomposition_request: Task Decomposition Request
            security_context: Security Context

        Returns:
            Enhanced Task Decomposition Result mit Dependency-Informationen
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "enhance_task_decomposition_with_dependencies_started",
                "task_id": task_decomposition_request.get("task_id"),
                "user_id": security_context.user_id,
                "tenant_id": security_context.tenant_id
            })

            # 1. Erstelle Task-Dependency-Context
            task_context = TaskDependencyContext(
                task_id=task_decomposition_request.get("task_id", "unknown"),
                subtask_id=task_decomposition_request.get("subtask_id"),
                orchestration_id=task_decomposition_request.get("orchestration_id"),
                required_tasks=task_decomposition_request.get("required_tasks", []),
                required_agents=task_decomposition_request.get("required_agents", []),
                required_capabilities=task_decomposition_request.get("required_capabilities", []),
                required_resources=task_decomposition_request.get("required_resources", []),
                security_level=security_context.security_level,
                metadata=task_decomposition_request.get("metadata", {})
            )

            # 2. Löse Task-Dependencies auf
            dependency_result = await self.dependency_resolution_engine.resolve_task_dependencies(
                task_context=task_context,
                security_context=security_context
            )

            # 3. Erweitere Request mit Dependency-Constraints
            enhanced_request = task_decomposition_request.copy()
            enhanced_request["dependency_constraints"] = {
                "resolved_dependencies": list(dependency_result.resolved_nodes),
                "failed_dependencies": list(dependency_result.failed_nodes),
                "dependency_resolution_order": dependency_result.resolution_order,
                "circular_dependencies": dependency_result.circular_dependencies,
                "dependency_aware_decomposition": True
            }

            # 4. Füge Dependency-Tracking hinzu
            enhanced_request["dependency_tracking"] = {
                "track_subtask_dependencies": True,
                "track_resource_dependencies": True,
                "track_agent_dependencies": True,
                "dependency_resolution_time_ms": dependency_result.resolution_time_ms
            }

            # Performance-Tracking
            integration_time_ms = (time.time() - start_time) * 1000
            self._update_integration_performance_stats(integration_time_ms)

            return {
                "success": dependency_result.success,
                "enhanced_request": enhanced_request,
                "dependency_result": dependency_result,
                "dependency_constraints": enhanced_request["dependency_constraints"]
            }

        except Exception as e:
            logger.error(f"Task Decomposition Dependency Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_orchestrator_with_dependencies(
        self,
        orchestration_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Orchestrator Service um Dependency-Management.

        Args:
            orchestration_request: Orchestration Request
            security_context: Security Context

        Returns:
            Enhanced Orchestration Result mit Dependency-Management
        """
        try:
            logger.debug({
                "event": "enhance_orchestrator_with_dependencies_started",
                "orchestration_id": orchestration_request.get("orchestration_id"),
                "user_id": security_context.user_id
            })

            # 1. Analysiere Orchestration-Dependencies
            orchestration_dependencies = await self._analyze_orchestration_dependencies(
                orchestration_request, security_context
            )

            # 2. Erstelle Dependency-Graph für Orchestration
            graph_id = f"orchestration_deps_{orchestration_request.get('orchestration_id', 'unknown')}"

            graph = await self.dependency_graph_engine.create_dependency_graph(
                graph_id=graph_id,
                name="Orchestration Dependencies",
                description=f"Dependency graph for orchestration {orchestration_request.get('orchestration_id')}"
            )

            # 3. Baue Orchestration-Dependency-Graph auf
            await self._build_orchestration_dependency_graph(
                graph, orchestration_request, orchestration_dependencies
            )

            # 4. Erweitere Request mit Dependency-Orchestration
            enhanced_request = orchestration_request.copy()
            enhanced_request["dependency_orchestration"] = {
                "dependency_graph_id": graph_id,
                "dependency_aware_scheduling": True,
                "respect_dependency_order": True,
                "handle_circular_dependencies": True,
                "dependency_optimization_enabled": True
            }

            # 5. Füge Dependency-Monitoring hinzu
            enhanced_request["dependency_monitoring"] = {
                "monitor_dependency_resolution": True,
                "track_dependency_performance": True,
                "alert_on_dependency_failures": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "dependency_graph_id": graph_id,
                "orchestration_dependencies": orchestration_dependencies
            }

        except Exception as e:
            logger.error(f"Orchestrator Dependency Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_agent_selection_with_dependencies(
        self,
        agent_selection_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Agent Selection um Dependency-Constraints.

        Args:
            agent_selection_request: Agent Selection Request
            security_context: Security Context

        Returns:
            Enhanced Agent Selection Result mit Dependency-Constraints
        """
        try:
            logger.debug({
                "event": "enhance_agent_selection_with_dependencies_started",
                "request_id": agent_selection_request.get("request_id"),
                "user_id": security_context.user_id
            })

            # 1. Analysiere Agent-Dependencies
            agent_dependencies = await self._analyze_agent_dependencies(
                agent_selection_request, security_context
            )

            # 2. Erweitere Request mit Dependency-Aware Selection
            enhanced_request = agent_selection_request.copy()
            enhanced_request["dependency_aware_selection"] = {
                "consider_agent_dependencies": True,
                "respect_capability_dependencies": True,
                "optimize_for_dependency_resolution": True,
                "agent_dependency_constraints": agent_dependencies
            }

            # 3. Füge Agent-Dependency-Tracking hinzu
            enhanced_request["agent_dependency_tracking"] = {
                "track_agent_capability_dependencies": True,
                "track_agent_resource_dependencies": True,
                "track_agent_performance_dependencies": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "agent_dependencies": agent_dependencies
            }

        except Exception as e:
            logger.error(f"Agent Selection Dependency Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_performance_prediction_with_dependencies(
        self,
        prediction_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Performance Prediction um Dependency-Performance-Vorhersagen.

        Args:
            prediction_request: Performance Prediction Request
            security_context: Security Context

        Returns:
            Enhanced Performance Prediction Result mit Dependency-Performance
        """
        try:
            logger.debug({
                "event": "enhance_performance_prediction_with_dependencies_started",
                "prediction_id": prediction_request.get("prediction_id"),
                "user_id": security_context.user_id
            })

            # 1. Analysiere Dependency-Performance-Impact
            dependency_performance_impact = await self._analyze_dependency_performance_impact(
                prediction_request, security_context
            )

            # 2. Erweitere Request mit Dependency-Performance-Prediction
            enhanced_request = prediction_request.copy()
            enhanced_request["dependency_performance_prediction"] = {
                "predict_dependency_resolution_time": True,
                "predict_dependency_bottlenecks": True,
                "optimize_for_dependency_performance": True,
                "dependency_performance_impact": dependency_performance_impact
            }

            # 3. Füge Dependency-Performance-Analytics hinzu
            enhanced_request["dependency_performance_analytics"] = {
                "analyze_dependency_patterns": True,
                "track_dependency_performance_trends": True,
                "identify_dependency_optimization_opportunities": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "dependency_performance_impact": dependency_performance_impact
            }

        except Exception as e:
            logger.error(f"Performance Prediction Dependency Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_llm_client_with_dependencies(
        self,
        llm_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert LLM Client um Dependency-Management.

        Args:
            llm_request: LLM Request
            security_context: Security Context

        Returns:
            Enhanced LLM Request Result mit Dependency-Management
        """
        try:
            logger.debug({
                "event": "enhance_llm_client_with_dependencies_started",
                "request_id": llm_request.get("request_id"),
                "user_id": security_context.user_id
            })

            # 1. Analysiere LLM-Dependencies
            llm_dependencies = await self._analyze_llm_dependencies(
                llm_request, security_context
            )

            # 2. Erweitere Request mit Dependency-Aware Features
            enhanced_request = llm_request.copy()
            enhanced_request["dependency_features"] = {
                "respect_model_dependencies": True,
                "optimize_for_dependency_chains": True,
                "dependency_aware_caching": True,
                "llm_dependency_constraints": llm_dependencies
            }

            # 3. Füge LLM-Dependency-Tracking hinzu
            enhanced_request["llm_dependency_tracking"] = {
                "track_model_dependencies": True,
                "track_context_dependencies": True,
                "track_token_dependencies": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "llm_dependencies": llm_dependencies
            }

        except Exception as e:
            logger.error(f"LLM Client Dependency Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _analyze_orchestration_dependencies(
        self,
        orchestration_request: dict[str, Any],
        _security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert Orchestration-Dependencies."""
        try:
            dependencies = {
                "task_dependencies": orchestration_request.get("task_dependencies", []),
                "agent_dependencies": orchestration_request.get("agent_dependencies", []),
                "resource_dependencies": orchestration_request.get("resource_dependencies", []),
                "service_dependencies": orchestration_request.get("service_dependencies", []),
                "data_dependencies": orchestration_request.get("data_dependencies", [])
            }

            return dependencies

        except Exception as e:
            logger.error(f"Orchestration dependencies analysis fehlgeschlagen: {e}")
            return {}

    async def _analyze_agent_dependencies(
        self,
        agent_selection_request: dict[str, Any],
        _security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert Agent-Dependencies."""
        try:
            dependencies = {
                "capability_dependencies": agent_selection_request.get("required_capabilities", []),
                "resource_dependencies": agent_selection_request.get("required_resources", []),
                "performance_dependencies": agent_selection_request.get("performance_requirements", {}),
                "security_dependencies": agent_selection_request.get("security_requirements", [])
            }

            return dependencies

        except Exception as e:
            logger.error(f"Agent dependencies analysis fehlgeschlagen: {e}")
            return {}

    async def _analyze_dependency_performance_impact(
        self,
        prediction_request: dict[str, Any],
        _security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert Dependency-Performance-Impact."""
        try:
            impact = {
                "dependency_resolution_overhead_ms": 50.0,  # Geschätzt
                "dependency_chain_length": prediction_request.get("estimated_dependency_chain_length", 5),
                "circular_dependency_risk": prediction_request.get("circular_dependency_risk", "low"),
                "dependency_bottlenecks": prediction_request.get("potential_bottlenecks", [])
            }

            return impact

        except Exception as e:
            logger.error(f"Dependency performance impact analysis fehlgeschlagen: {e}")
            return {}

    async def _analyze_llm_dependencies(
        self,
        llm_request: dict[str, Any],
        _security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert LLM-Dependencies."""
        try:
            dependencies = {
                "model_dependencies": llm_request.get("model_requirements", []),
                "context_dependencies": llm_request.get("context_requirements", []),
                "token_dependencies": llm_request.get("token_requirements", {}),
                "quota_dependencies": llm_request.get("quota_requirements", [])
            }

            return dependencies

        except Exception as e:
            logger.error(f"LLM dependencies analysis fehlgeschlagen: {e}")
            return {}

    async def _build_orchestration_dependency_graph(
        self,
        graph,
        orchestration_request: dict[str, Any],
        dependencies: dict[str, Any]
    ) -> None:
        """Baut Orchestration-Dependency-Graph auf."""
        try:
            orchestration_id = orchestration_request.get("orchestration_id", "unknown")

            # Füge Orchestration-Node hinzu
            await self.dependency_graph_engine.add_dependency_node(
                graph_id=graph.graph_id,
                node_id=orchestration_id,
                node_type=DependencyType.SERVICE,
                name=f"Orchestration {orchestration_id}",
                description="Main orchestration node",
                metadata=orchestration_request.get("metadata", {})
            )

            # Füge Task-Dependencies hinzu
            for i, task_dep in enumerate(dependencies.get("task_dependencies", [])):
                await self.dependency_graph_engine.add_dependency_node(
                    graph_id=graph.graph_id,
                    node_id=task_dep,
                    node_type=DependencyType.TASK,
                    name=f"Task Dependency {task_dep}",
                    description="Required task dependency"
                )

                await self.dependency_graph_engine.add_dependency_edge(
                    graph_id=graph.graph_id,
                    edge_id=f"orch_task_dep_{i}",
                    source_node_id=orchestration_id,
                    target_node_id=task_dep,
                    relation=DependencyRelation.REQUIRES,
                    dependency_type=DependencyType.TASK
                )

            # Füge Agent-Dependencies hinzu
            for i, agent_dep in enumerate(dependencies.get("agent_dependencies", [])):
                await self.dependency_graph_engine.add_dependency_node(
                    graph_id=graph.graph_id,
                    node_id=agent_dep,
                    node_type=DependencyType.AGENT,
                    name=f"Agent Dependency {agent_dep}",
                    description="Required agent dependency"
                )

                await self.dependency_graph_engine.add_dependency_edge(
                    graph_id=graph.graph_id,
                    edge_id=f"orch_agent_dep_{i}",
                    source_node_id=orchestration_id,
                    target_node_id=agent_dep,
                    relation=DependencyRelation.REQUIRES,
                    dependency_type=DependencyType.AGENT
                )

        except Exception as e:
            logger.error(f"Orchestration dependency graph building fehlgeschlagen: {e}")
            raise

    async def track_dependency_usage(
        self,
        service_name: str,
        operation: str,
        _security_context: SecurityContext,
        dependency_usage: dict[str, Any],
        resolution_time_ms: float,
        success: bool
    ) -> None:
        """Trackt Dependency-Usage für Analytics.

        Args:
            service_name: Service-Name
            operation: Operation-Name
            _security_context: Security Context
            dependency_usage: Dependency-Usage-Informationen
            resolution_time_ms: Resolution-Zeit
            success: Erfolg-Status
        """
        try:
            # Tracke Usage in Analytics (falls verfügbar)
            if hasattr(self.dependency_resolution_engine, "analytics_engine"):
                await self.dependency_resolution_engine.analytics_engine.track_dependency_usage(
                    dependency_id=f"{service_name}_{operation}",
                    dependency_type=DependencyType.SERVICE,
                    scope="service",
                    scope_id=service_name,
                    usage_amount=dependency_usage.get("amount", 1),
                    resolution_time_ms=resolution_time_ms,
                    success=success
                )

            self._dependency_aware_operations += 1

        except Exception as e:
            logger.error(f"Dependency usage tracking fehlgeschlagen: {e}")

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

        # Hole Statistiken von Komponenten
        resolution_stats = self.dependency_resolution_engine.get_performance_stats()
        graph_stats = self.dependency_graph_engine.get_performance_stats()

        return {
            "dependency_service_integration": {
                "total_integrations": self._integration_count,
                "avg_integration_time_ms": avg_integration_time,
                "dependency_aware_operations": self._dependency_aware_operations,
                "meets_integration_sla": avg_integration_time < 100.0,  # < 100ms SLA
                "task_decomposition_integration_enabled": self.enable_task_decomposition_integration,
                "orchestrator_integration_enabled": self.enable_orchestrator_integration,
                "agent_selection_integration_enabled": self.enable_agent_selection_integration,
                "performance_prediction_integration_enabled": self.enable_performance_prediction_integration,
                "llm_client_integration_enabled": self.enable_llm_client_integration
            },
            "dependency_resolution_engine": {
                "total_resolutions": resolution_stats["total_resolutions"],
                "avg_resolution_time_ms": resolution_stats["avg_resolution_time_ms"],
                "success_rate": resolution_stats["success_rate"],
                "meets_resolution_sla": resolution_stats["meets_resolution_sla"]
            },
            "dependency_graph_engine": {
                "total_graph_analyses": graph_stats["total_graph_analyses"],
                "avg_graph_analysis_time_ms": graph_stats["avg_graph_analysis_time_ms"],
                "managed_graphs": graph_stats["managed_graphs"],
                "cache_size": graph_stats["cache_size"]
            }
        }
