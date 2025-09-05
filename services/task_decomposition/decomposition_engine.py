# backend/services/task_decomposition/decomposition_engine.py
"""LLM-powered Task Decomposition Engine.

Hauptkomponente für intelligente Task-Zerlegung mit LLM-Integration,
Agent-Matching und Performance-Optimierung.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from agents.registry.dynamic_registry import DynamicAgentRegistry
from kei_logging import get_logger, log_orchestrator_step, training_trace
from services.ml.performance_prediction import PerformancePredictor
from task_management.core_task_manager import TaskManager

from .agent_matcher import AgentCapabilityMatcher
from .data_models import (
    DecompositionPlan,
    DecompositionRequest,
    DecompositionResult,
    DecompositionStrategy,
    SubtaskDefinition,
    ValidationResult,
)
from .fallback_decomposer import FallbackDecomposer
from .llm_analyzer import LLMTaskAnalyzer
from .plan_validator import PlanValidator

logger = get_logger(__name__)


class TaskDecompositionEngine:
    """LLM-powered Task Decomposition Engine."""

    def __init__(
        self,
        task_manager: TaskManager,
        agent_registry: DynamicAgentRegistry,
        performance_predictor: PerformancePredictor | None = None
    ):
        """Initialisiert Task Decomposition Engine.

        Args:
            task_manager: Task Manager für Task-Erstellung
            agent_registry: Agent Registry für verfügbare Agents
            performance_predictor: ML-basierter Performance Predictor
        """
        self.task_manager = task_manager
        self.agent_registry = agent_registry
        self.performance_predictor = performance_predictor

        # Komponenten initialisieren
        self.llm_analyzer = LLMTaskAnalyzer()
        self.agent_matcher = AgentCapabilityMatcher(agent_registry, performance_predictor)
        self.plan_validator = PlanValidator()
        self.fallback_decomposer = FallbackDecomposer()

        # Konfiguration
        self.max_decomposition_time_seconds = 2.0  # SLA: < 2s
        self.enable_llm_analysis = True
        self.enable_fallback = True
        self.enable_validation = True

        # Performance-Tracking
        self._decomposition_count = 0
        self._total_decomposition_time_ms = 0.0
        self._llm_success_count = 0
        self._fallback_count = 0

        logger.info({
            "event": "task_decomposition_engine_initialized",
            "max_decomposition_time": self.max_decomposition_time_seconds,
            "llm_enabled": self.enable_llm_analysis,
            "fallback_enabled": self.enable_fallback
        })

    @training_trace(context={"component": "task_decomposition", "phase": "decomposition"})
    async def decompose_task(self, request: DecompositionRequest) -> DecompositionResult:
        """Zerlegt Task in optimale Subtasks mit Agent-Assignments.

        Args:
            request: Decomposition-Request

        Returns:
            Decomposition-Result mit Plan oder Fallback
        """
        start_time = time.time()

        try:
            # Schritt 1: Task-Decomposition gestartet
            log_orchestrator_step(
                "Starting Task Decomposition",
                "task_decomposition",
                task_id=request.task_id,
                task_type=request.task_type.value,
                task_description_length=len(request.task_description),
                llm_enabled=request.enable_llm_analysis,
                agent_matching_enabled=request.enable_agent_matching,
                max_decomposition_time_seconds=self.max_decomposition_time_seconds
            )

            logger.info({
                "event": "task_decomposition_started",
                "task_id": request.task_id,
                "task_type": request.task_type.value,
                "llm_enabled": request.enable_llm_analysis
            })

            # Timeout-Protection
            decomposition_task = asyncio.create_task(self._decompose_task_internal(request))

            try:
                result = await asyncio.wait_for(
                    decomposition_task,
                    timeout=self.max_decomposition_time_seconds
                )
            except TimeoutError:
                logger.warning(f"Task-Decomposition Timeout für {request.task_id}")
                result = await self._handle_timeout_fallback(request)

            # Performance-Tracking
            decomposition_time_ms = (time.time() - start_time) * 1000
            result.decomposition_time_ms = decomposition_time_ms
            self._update_performance_stats(decomposition_time_ms, result.used_fallback)

            # Schritt 2: Task-Decomposition abgeschlossen
            log_orchestrator_step(
                "Task Decomposition Completed",
                "task_decomposition",
                task_id=request.task_id,
                success=result.success,
                used_fallback=result.used_fallback,
                subtask_count=len(result.plan.subtasks) if result.plan else 0,
                decomposition_time_ms=decomposition_time_ms,
                strategy_used=result.plan.execution_strategy.value if result.plan else "none",
                estimated_total_duration_minutes=result.plan.estimated_total_duration_minutes if result.plan else 0,
                parallelization_efficiency=result.plan.parallelization_efficiency if result.plan else 0
            )

            logger.info({
                "event": "task_decomposition_completed",
                "task_id": request.task_id,
                "success": result.success,
                "used_fallback": result.used_fallback,
                "subtask_count": len(result.plan.subtasks) if result.plan else 0,
                "decomposition_time_ms": decomposition_time_ms
            })

            return result

        except Exception as e:
            logger.error(f"Task-Decomposition fehlgeschlagen für {request.task_id}: {e}")
            return DecompositionResult(
                success=False,
                error_message=str(e),
                decomposition_time_ms=(time.time() - start_time) * 1000
            )

    async def _decompose_task_internal(self, request: DecompositionRequest) -> DecompositionResult:
        """Interne Decomposition-Logic."""
        llm_analysis_start = time.time()

        # 1. LLM-basierte Task-Analyse
        log_orchestrator_step(
            "Starting LLM Task Analysis",
            "llm_call",
            task_id=request.task_id,
            llm_enabled=request.enable_llm_analysis and self.enable_llm_analysis,
            fallback_enabled=self.enable_fallback
        )

        analysis = None
        if request.enable_llm_analysis and self.enable_llm_analysis:
            try:
                analysis = await self.llm_analyzer.analyze_task(request)
                self._llm_success_count += 1

                log_orchestrator_step(
                    "LLM Task Analysis Successful",
                    "llm_call",
                    task_id=request.task_id,
                    complexity_score=analysis.complexity_score,
                    is_decomposable=analysis.is_decomposable,
                    confidence=analysis.decomposition_confidence
                )
            except Exception as e:
                logger.warning(f"LLM-Analyse fehlgeschlagen: {e}")
                log_orchestrator_step(
                    "LLM Task Analysis Failed",
                    "llm_call",
                    task_id=request.task_id,
                    error=str(e),
                    fallback_enabled=self.enable_fallback
                )
                if not self.enable_fallback:
                    raise

        llm_analysis_time_ms = (time.time() - llm_analysis_start) * 1000

        # 2. Subtask-Generierung
        log_orchestrator_step(
            "Starting Subtask Generation",
            "task_decomposition",
            task_id=request.task_id,
            has_analysis=analysis is not None,
            is_decomposable=analysis.is_decomposable if analysis else False
        )

        subtasks = []
        used_fallback = False
        fallback_reason = None

        if analysis and analysis.is_decomposable:
            # LLM-basierte Subtask-Generierung
            try:
                subtasks = await self.llm_analyzer.generate_subtasks(request, analysis)
                log_orchestrator_step(
                    "LLM Subtask Generation Successful",
                    "task_decomposition",
                    task_id=request.task_id,
                    subtask_count=len(subtasks),
                    strategy=analysis.recommended_strategy.value
                )
            except Exception as e:
                logger.warning(f"LLM Subtask-Generierung fehlgeschlagen: {e}")
                if self.enable_fallback:
                    log_orchestrator_step(
                        "Falling back to Rule-based Decomposition",
                        "task_decomposition",
                        task_id=request.task_id,
                        reason="llm_subtask_generation_failed"
                    )
                    subtasks = await self.fallback_decomposer.decompose_task(request)
                    used_fallback = True
                    fallback_reason = "llm_subtask_generation_failed"
                else:
                    raise
        # Fallback auf regelbasierte Decomposition
        elif self.enable_fallback:
            log_orchestrator_step(
                "Using Rule-based Decomposition",
                "task_decomposition",
                task_id=request.task_id,
                reason="no_llm_analysis" if not analysis else "not_decomposable"
            )
            subtasks = await self.fallback_decomposer.decompose_task(request)
            used_fallback = True
            fallback_reason = "task_not_decomposable" if analysis else "llm_analysis_failed"
            self._fallback_count += 1
        else:
            return DecompositionResult(
                success=False,
                error_message="Task ist nicht decomposable und Fallback deaktiviert"
            )

        if not subtasks:
            return DecompositionResult(
                success=False,
                error_message="Keine Subtasks generiert"
            )

        # 3. Agent-Matching
        agent_matching_start = time.time()
        agent_assignments = {}

        if request.enable_agent_matching:
            try:
                agent_matches = await self.agent_matcher.find_best_agents(subtasks)

                # Wähle besten Agent pro Subtask
                for subtask_id, matches in agent_matches.items():
                    if matches:
                        agent_assignments[subtask_id] = matches[0]  # Bester Match

            except Exception as e:
                logger.warning(f"Agent-Matching fehlgeschlagen: {e}")

        agent_matching_time_ms = (time.time() - agent_matching_start) * 1000

        # 4. Decomposition-Plan erstellen
        plan = await self._create_decomposition_plan(
            request, subtasks, agent_assignments, analysis
        )

        # 5. Plan-Validation
        validation_result = None
        if self.enable_validation:
            validation_result = await self.plan_validator.validate_plan(plan, request)

            if not validation_result.is_valid:
                logger.warning({
                    "event": "plan_validation_failed",
                    "task_id": request.task_id,
                    "issues": validation_result.critical_issues
                })

                # Versuche Auto-Fixes
                if validation_result.auto_fixes_applied:
                    plan = await self._apply_auto_fixes(plan, validation_result)

        return DecompositionResult(
            success=True,
            analysis=analysis,
            plan=plan,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            llm_analysis_time_ms=llm_analysis_time_ms,
            agent_matching_time_ms=agent_matching_time_ms
        )

    async def _create_decomposition_plan(
        self,
        request: DecompositionRequest,
        subtasks: list[SubtaskDefinition],
        agent_assignments: dict[str, Any],
        analysis: Any | None
    ) -> DecompositionPlan:
        """Erstellt vollständigen Decomposition-Plan."""
        plan_id = str(uuid.uuid4())

        # Execution-Strategy bestimmen
        strategy = (
            analysis.recommended_strategy
            if analysis
            else request.preferred_strategy or DecompositionStrategy.SEQUENTIAL
        )

        # Execution-Order berechnen
        execution_order = await self._calculate_execution_order(subtasks, strategy)

        # Dependency-Graph erstellen
        dependency_graph = self._build_dependency_graph(subtasks)

        # Kritischen Pfad berechnen
        critical_path = self._calculate_critical_path(subtasks, dependency_graph)

        # Performance-Schätzungen
        total_duration = sum(st.estimated_duration_minutes for st in subtasks)
        parallel_duration = self._estimate_parallel_duration(subtasks, execution_order)
        parallelization_efficiency = (
            1.0 - (parallel_duration / total_duration)
            if total_duration > 0 else 0.0
        )

        return DecompositionPlan(
            plan_id=plan_id,
            original_task_id=request.task_id,
            subtasks=subtasks,
            execution_strategy=strategy,
            execution_order=execution_order,
            agent_assignments=agent_assignments,
            dependency_graph=dependency_graph,
            critical_path=critical_path,
            estimated_total_duration_minutes=total_duration,
            estimated_parallel_duration_minutes=parallel_duration,
            parallelization_efficiency=parallelization_efficiency,
            plan_confidence=analysis.decomposition_confidence if analysis else 0.7,
            validation_results={},
            potential_issues=[],
            llm_model_used=self.llm_analyzer.analysis_model,
            analysis_used=analysis
        )

    async def _calculate_execution_order(
        self,
        subtasks: list[SubtaskDefinition],
        strategy: DecompositionStrategy
    ) -> list[list[str]]:
        """Berechnet optimale Execution-Order."""
        if strategy == DecompositionStrategy.SEQUENTIAL:
            # Sequenzielle Ausführung
            return [[st.subtask_id] for st in subtasks]

        if strategy == DecompositionStrategy.PARALLEL:
            # Maximale Parallelisierung
            parallel_groups = {}
            for subtask in subtasks:
                group = subtask.parallel_group or "default"
                if group not in parallel_groups:
                    parallel_groups[group] = []
                parallel_groups[group].append(subtask.subtask_id)

            return list(parallel_groups.values())

        if strategy == DecompositionStrategy.PIPELINE:
            # Pipeline-Verarbeitung basierend auf Dependencies
            return self._create_pipeline_order(subtasks)

        # HYBRID
        # Intelligente Mischung
        return self._create_hybrid_order(subtasks)

    def _build_dependency_graph(self, subtasks: list[SubtaskDefinition]) -> dict[str, list[str]]:
        """Erstellt Dependency-Graph."""
        graph = {}

        for subtask in subtasks:
            graph[subtask.subtask_id] = subtask.depends_on.copy()

        return graph

    def _calculate_critical_path(
        self,
        subtasks: list[SubtaskDefinition],
        dependency_graph: dict[str, list[str]]
    ) -> list[str]:
        """Berechnet kritischen Pfad durch Dependencies."""
        # Vereinfachte Critical-Path-Berechnung
        # TODO: Implementiere echten Critical-Path-Algorithmus - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115

        # Finde Subtasks ohne Dependencies (Start-Nodes)
        start_nodes = [
            st.subtask_id for st in subtasks
            if not st.depends_on
        ]

        if not start_nodes:
            return [subtasks[0].subtask_id] if subtasks else []

        # Finde längsten Pfad (vereinfacht)
        longest_path = []
        max_duration = 0.0

        for start_node in start_nodes:
            path = [start_node]
            duration = next(
                st.estimated_duration_minutes
                for st in subtasks
                if st.subtask_id == start_node
            )

            if duration > max_duration:
                max_duration = duration
                longest_path = path

        return longest_path

    def _estimate_parallel_duration(
        self,
        subtasks: list[SubtaskDefinition],
        execution_order: list[list[str]]
    ) -> float:
        """Schätzt Dauer bei paralleler Ausführung."""
        total_duration = 0.0

        for group in execution_order:
            # Maximale Dauer in der Gruppe
            group_duration = 0.0
            for subtask_id in group:
                subtask = next(
                    st for st in subtasks
                    if st.subtask_id == subtask_id
                )
                group_duration = max(group_duration, subtask.estimated_duration_minutes)

            total_duration += group_duration

        return total_duration

    def _create_pipeline_order(self, subtasks: list[SubtaskDefinition]) -> list[list[str]]:
        """Erstellt Pipeline-Order basierend auf Dependencies."""
        # Vereinfachte Pipeline-Logic
        return [[st.subtask_id] for st in subtasks]

    def _create_hybrid_order(self, subtasks: list[SubtaskDefinition]) -> list[list[str]]:
        """Erstellt hybride Execution-Order."""
        # Intelligente Mischung aus parallel und sequenziell
        parallel_subtasks = [st for st in subtasks if st.can_run_parallel]
        sequential_subtasks = [st for st in subtasks if not st.can_run_parallel]

        order = []

        # Parallele Subtasks gruppieren
        if parallel_subtasks:
            order.append([st.subtask_id for st in parallel_subtasks])

        # Sequenzielle Subtasks einzeln
        for subtask in sequential_subtasks:
            order.append([subtask.subtask_id])

        return order

    async def _handle_timeout_fallback(self, request: DecompositionRequest) -> DecompositionResult:
        """Behandelt Timeout mit Fallback."""
        logger.warning(f"Decomposition-Timeout für {request.task_id}, verwende Fallback")

        try:
            # Schnelle Fallback-Decomposition
            subtasks = await self.fallback_decomposer.decompose_task(request)

            if subtasks:
                plan = await self._create_decomposition_plan(request, subtasks, {}, None)

                return DecompositionResult(
                    success=True,
                    plan=plan,
                    used_fallback=True,
                    fallback_reason="decomposition_timeout"
                )
        except Exception as e:
            logger.error(f"Fallback nach Timeout fehlgeschlagen: {e}")

        return DecompositionResult(
            success=False,
            error_message="Decomposition-Timeout und Fallback fehlgeschlagen"
        )

    async def _apply_auto_fixes(self, plan: DecompositionPlan, validation: ValidationResult) -> DecompositionPlan:
        """Wendet automatische Fixes auf Plan an."""
        # TODO: Implementiere Auto-Fix-Logic - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        logger.info(f"Auto-Fixes angewendet: {validation.auto_fixes_applied}")
        return plan

    def _update_performance_stats(self, decomposition_time_ms: float, used_fallback: bool) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._decomposition_count += 1
        self._total_decomposition_time_ms += decomposition_time_ms

        if used_fallback:
            self._fallback_count += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_time = (
            self._total_decomposition_time_ms / self._decomposition_count
            if self._decomposition_count > 0 else 0.0
        )

        fallback_rate = (
            self._fallback_count / self._decomposition_count
            if self._decomposition_count > 0 else 0.0
        )

        llm_success_rate = (
            self._llm_success_count / self._decomposition_count
            if self._decomposition_count > 0 else 0.0
        )

        return {
            "total_decompositions": self._decomposition_count,
            "avg_decomposition_time_ms": avg_time,
            "total_decomposition_time_ms": self._total_decomposition_time_ms,
            "fallback_rate": fallback_rate,
            "llm_success_rate": llm_success_rate,
            "meets_sla": avg_time < 2000.0,  # < 2s SLA
            "llm_analyzer_stats": self.llm_analyzer.get_performance_stats(),
            "agent_matcher_stats": self.agent_matcher.get_performance_stats()
        }
