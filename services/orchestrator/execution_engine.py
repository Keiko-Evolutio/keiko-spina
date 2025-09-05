# backend/services/orchestrator/execution_engine.py
"""Execution Engine für Orchestrator Service.

Implementiert intelligente Task-Execution mit Parallelisierung,
Load-Balancing und Real-time Monitoring.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger, log_orchestrator_step, training_trace
from task_management.core_task_manager import TaskManager, TaskState

from .data_models import (
    AgentAssignmentStatus,
    AgentLoadInfo,
    ExecutionPlan,
    OrchestrationProgress,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationState,
    SubtaskExecution,
)

if TYPE_CHECKING:
    from agents.registry.dynamic_registry import DynamicAgentRegistry
    from services.ml.performance_prediction import PerformancePredictor
    from services.task_decomposition import DecompositionPlan, TaskDecompositionEngine

    from .monitoring import OrchestrationMonitor

logger = get_logger(__name__)


class ExecutionEngine:
    """Execution Engine für intelligente Task-Orchestration."""

    def __init__(
        self,
        task_manager: TaskManager,
        agent_registry: DynamicAgentRegistry,
        decomposition_engine: TaskDecompositionEngine,
        performance_predictor: PerformancePredictor | None = None,
        monitor: OrchestrationMonitor | None = None,
        state_store: Any | None = None,
        event_integration: Any | None = None
    ):
        """Initialisiert Execution Engine.

        Args:
            task_manager: Task Manager für Task-Execution
            agent_registry: Agent Registry für verfügbare Agents
            decomposition_engine: Task Decomposition Engine
            performance_predictor: Performance Predictor für Optimierung
            monitor: Orchestration Monitor für Real-time Tracking
            state_store: State Store für Plan-Persistierung
            event_integration: Event Integration für asynchrone Kommunikation
        """
        self.task_manager = task_manager
        self.agent_registry = agent_registry
        self.decomposition_engine = decomposition_engine
        self.performance_predictor = performance_predictor
        self.monitor = monitor
        self.state_store = state_store
        self.event_integration = event_integration

        # Execution-Konfiguration
        self.max_concurrent_orchestrations = 10
        self.max_parallel_subtasks_per_orchestration = 5
        self.subtask_timeout_seconds = 300  # 5 Minuten
        self.orchestration_overhead_target_ms = 100.0  # < 100ms Overhead

        # State-Management
        self.active_orchestrations: dict[str, OrchestrationProgress] = {}
        self.execution_plans: dict[str, ExecutionPlan] = {}
        self.agent_loads: dict[str, AgentLoadInfo] = {}

        # Performance-Tracking
        self._orchestration_count = 0
        self._total_orchestration_time_ms = 0.0
        self._success_count = 0

        # Execution-Tasks
        self._execution_tasks: dict[str, asyncio.Task] = {}
        self._is_running = False

        logger.info("Execution Engine initialisiert")

    async def start(self) -> None:
        """Startet Execution Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        asyncio.create_task(self._agent_load_monitor_loop())
        asyncio.create_task(self._orchestration_cleanup_loop())

        logger.info("Execution Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Execution Engine."""
        self._is_running = False

        # Stoppe alle aktiven Orchestrations
        for orchestration_id in list(self._execution_tasks.keys()):
            await self.cancel_orchestration(orchestration_id)

        logger.info("Execution Engine gestoppt")

    @training_trace(context={"component": "execution_engine", "phase": "orchestration"})
    async def execute_orchestration(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Führt komplette Orchestration aus.

        Args:
            request: Orchestration-Request

        Returns:
            Orchestration-Result
        """
        orchestration_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Schritt 1: Orchestration gestartet
            log_orchestrator_step(
                "Starting Orchestration",
                "orchestration",
                orchestration_id=orchestration_id,
                task_id=request.task_id,
                execution_mode=request.execution_mode.value,
                enable_decomposition=request.enable_decomposition,
                enable_monitoring=request.enable_monitoring,
                enable_recovery=request.enable_recovery
            )

            logger.info({
                "event": "orchestration_started",
                "orchestration_id": orchestration_id,
                "task_id": request.task_id,
                "execution_mode": request.execution_mode.value
            })

            # 1. Initialisiere Progress-Tracking
            progress = OrchestrationProgress(
                orchestration_id=orchestration_id,
                state=OrchestrationState.PLANNING,
                total_subtasks=0,
                completed_subtasks=0,
                failed_subtasks=0,
                running_subtasks=0,
                pending_subtasks=0,
                start_time=datetime.utcnow()
            )

            self.active_orchestrations[orchestration_id] = progress

            # 2. Task-Decomposition (falls aktiviert)
            log_orchestrator_step(
                "Creating Execution Plan",
                "orchestration",
                orchestration_id=orchestration_id,
                decomposition_enabled=request.enable_decomposition
            )

            execution_plan = await self._create_execution_plan(request, orchestration_id)
            self.execution_plans[orchestration_id] = execution_plan

            log_orchestrator_step(
                "Execution Plan Created",
                "orchestration",
                orchestration_id=orchestration_id,
                subtask_count=len(execution_plan.subtasks),
                strategy=execution_plan.strategy.value,
                estimated_duration_minutes=execution_plan.estimated_total_duration_minutes
            )

            # Publiziere Plan-Created Event
            if self.event_integration:
                await self.event_integration.publish_plan_created(
                    plan=execution_plan,
                    orchestration_id=orchestration_id
                )

            # 3. Erstelle Checkpoint für Recovery
            if self.state_store:
                try:
                    checkpoint_data = {
                        "orchestration_id": orchestration_id,
                        "request": {
                            "task_id": request.task_id,
                            "task_type": request.task_type.value,
                            "execution_mode": request.execution_mode.value,
                            "user_id": request.user_id,
                            "session_id": request.session_id,
                            "correlation_id": request.correlation_id
                        },
                        "execution_plan": {
                            "plan_id": execution_plan.plan_id,
                            "subtasks_count": len(execution_plan.subtasks),
                            "estimated_duration_minutes": execution_plan.estimated_total_duration_ms / 60000.0
                        },
                        "progress": {
                            "state": progress.state.value,
                            "start_time": progress.start_time.isoformat()
                        }
                    }

                    checkpoint_id = await self.state_store.create_checkpoint(
                        orchestration_id=orchestration_id,
                        state_data=checkpoint_data,
                        metadata={"checkpoint_type": "orchestration_start"}
                    )

                    logger.debug(f"Checkpoint erstellt: {checkpoint_id}")

                except Exception as e:
                    logger.warning(f"Checkpoint-Erstellung fehlgeschlagen: {e}")

            # 4. Progress aktualisieren
            progress.total_subtasks = len(execution_plan.subtasks)
            progress.pending_subtasks = len(execution_plan.subtasks)
            progress.state = OrchestrationState.SCHEDULED

            # 4. Execution starten
            progress.state = OrchestrationState.EXECUTING

            # 5. Subtasks parallel ausführen
            execution_task = asyncio.create_task(
                self._execute_subtasks(orchestration_id, execution_plan, request)
            )
            self._execution_tasks[orchestration_id] = execution_task

            # 6. Warte auf Completion
            subtask_results = await execution_task

            # 7. Ergebnis aggregieren
            result = await self._aggregate_results(
                orchestration_id, execution_plan, subtask_results, start_time
            )

            # 8. Cleanup
            await self._cleanup_orchestration(orchestration_id)

            # 9. Performance-Tracking
            orchestration_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(orchestration_time_ms, result.success)

            logger.info({
                "event": "orchestration_completed",
                "orchestration_id": orchestration_id,
                "success": result.success,
                "execution_time_ms": orchestration_time_ms,
                "subtasks_completed": len([r for r in subtask_results if r.state == TaskState.COMPLETED])
            })

            return result

        except Exception as e:
            logger.exception(f"Orchestration {orchestration_id} fehlgeschlagen: {e}")

            # Cleanup bei Fehler
            await self._cleanup_orchestration(orchestration_id)

            return OrchestrationResult(
                success=False,
                orchestration_id=orchestration_id,
                state=OrchestrationState.FAILED,
                error_message=str(e),
                total_execution_time_ms=(time.time() - start_time) * 1000
            )

    async def _create_execution_plan(
        self,
        request: OrchestrationRequest,
        orchestration_id: str
    ) -> ExecutionPlan:
        """Erstellt Execution-Plan für Orchestration."""
        if not request.enable_decomposition:
            # Einfacher Plan ohne Decomposition
            return await self._create_simple_execution_plan(request, orchestration_id)

        # Task-Decomposition
        from services.task_decomposition import DecompositionRequest

        decomposition_request = DecompositionRequest(
            task_id=request.task_id,
            task_type=request.task_type,
            task_name=request.task_name,
            task_description=request.task_description,
            task_payload=request.task_payload,
            task_priority=request.priority,
            user_id=request.user_id,
            session_id=request.session_id,
            max_subtasks=20,
            max_parallel_subtasks=request.max_parallel_tasks,
            enable_llm_analysis=True,
            enable_agent_matching=True
        )

        decomposition_result = await self.decomposition_engine.decompose_task(decomposition_request)

        if not decomposition_result.success or not decomposition_result.plan:
            # Fallback auf einfachen Plan
            return await self._create_simple_execution_plan(request, orchestration_id)

        # Konvertiere Decomposition-Plan zu Execution-Plan
        return await self._convert_decomposition_to_execution_plan(
            decomposition_result.plan, orchestration_id
        )

    async def _create_simple_execution_plan(
        self,
        request: OrchestrationRequest,
        orchestration_id: str
    ) -> ExecutionPlan:
        """Erstellt einfachen Execution-Plan ohne Decomposition."""
        # Einzelner Subtask für gesamte Task
        subtask = SubtaskExecution(
            subtask_id=f"{request.task_id}_main",
            name=request.task_name,
            description=request.task_description,
            task_type=request.task_type,
            priority=request.priority,
            payload=request.task_payload
        )

        # Agent-Assignment
        if request.preferred_agents:
            subtask.assigned_agent_id = request.preferred_agents[0]
            subtask.agent_assignment_status = AgentAssignmentStatus.ASSIGNED

        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            subtasks=[subtask],
            execution_groups=[[subtask.subtask_id]],
            critical_path=[subtask.subtask_id],
            agent_assignments={subtask.subtask_id: subtask.assigned_agent_id} if subtask.assigned_agent_id else {},
            load_distribution={},
            estimated_total_duration_ms=30000.0,  # 30s Default
            estimated_parallel_duration_ms=30000.0,
            parallelization_efficiency=0.0,
            peak_resource_usage={},
            resource_timeline=[]
        )

    async def _convert_decomposition_to_execution_plan(
        self,
        decomposition_plan: DecompositionPlan,
        orchestration_id: str
    ) -> ExecutionPlan:
        """Konvertiert Decomposition-Plan zu Execution-Plan."""
        subtasks = []
        agent_assignments = {}

        for subtask_def in decomposition_plan.subtasks:
            subtask = SubtaskExecution(
                subtask_id=subtask_def.subtask_id,
                name=subtask_def.name,
                description=subtask_def.description,
                task_type=subtask_def.task_type,
                priority=subtask_def.priority,
                payload=subtask_def.payload,
                depends_on=subtask_def.depends_on
            )

            # Agent-Assignment aus Decomposition-Plan
            if subtask_def.subtask_id in decomposition_plan.agent_assignments:
                agent_match = decomposition_plan.agent_assignments[subtask_def.subtask_id]
                subtask.assigned_agent_id = agent_match.agent_id
                subtask.agent_assignment_status = AgentAssignmentStatus.ASSIGNED
                subtask.predicted_execution_time_ms = agent_match.estimated_execution_time_ms
                subtask.prediction_confidence = agent_match.confidence_score

                agent_assignments[subtask.subtask_id] = agent_match.agent_id

            subtasks.append(subtask)

        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            subtasks=subtasks,
            execution_groups=decomposition_plan.execution_order,
            critical_path=decomposition_plan.critical_path,
            agent_assignments=agent_assignments,
            load_distribution={},
            estimated_total_duration_ms=decomposition_plan.estimated_total_duration_minutes * 60 * 1000,
            estimated_parallel_duration_ms=decomposition_plan.estimated_parallel_duration_minutes * 60 * 1000,
            parallelization_efficiency=decomposition_plan.parallelization_efficiency,
            peak_resource_usage={},
            resource_timeline=[]
        )

    async def _execute_subtasks(
        self,
        orchestration_id: str,
        execution_plan: ExecutionPlan,
        request: OrchestrationRequest
    ) -> list[SubtaskExecution]:
        """Führt Subtasks gemäß Execution-Plan aus."""
        progress = self.active_orchestrations[orchestration_id]
        completed_subtasks: set[str] = set()

        # Führe Execution-Groups sequenziell aus
        for group in execution_plan.execution_groups:
            # Parallel execution innerhalb der Gruppe
            group_tasks = []

            for subtask_id in group:
                # Prüfe Dependencies
                subtask = next(st for st in execution_plan.subtasks if st.subtask_id == subtask_id)

                if all(dep_id in completed_subtasks for dep_id in subtask.depends_on):
                    # Dependencies erfüllt - starte Subtask
                    task = asyncio.create_task(
                        self._execute_single_subtask(subtask, orchestration_id, request)
                    )
                    group_tasks.append((subtask_id, task))

            # Warte auf Completion der Gruppe
            if group_tasks:
                group_results = await asyncio.gather(
                    *[task for _, task in group_tasks],
                    return_exceptions=True
                )

                # Update completed subtasks
                for i, (subtask_id, _) in enumerate(group_tasks):
                    if not isinstance(group_results[i], Exception):
                        completed_subtasks.add(subtask_id)
                        progress.completed_subtasks += 1
                    else:
                        progress.failed_subtasks += 1
                        logger.error(f"Subtask {subtask_id} fehlgeschlagen: {group_results[i]}")

                # Update Progress
                progress.pending_subtasks = progress.total_subtasks - progress.completed_subtasks - progress.failed_subtasks
                progress.last_updated = datetime.utcnow()

        return execution_plan.subtasks

    async def _execute_single_subtask(
        self,
        subtask: SubtaskExecution,
        orchestration_id: str,
        request: OrchestrationRequest
    ) -> SubtaskExecution:
        """Führt einzelnen Subtask aus."""
        subtask.start_time = datetime.utcnow()
        subtask.state = TaskState.RUNNING

        try:
            # Agent-Assignment falls noch nicht erfolgt
            if not subtask.assigned_agent_id:
                subtask.assigned_agent_id = await self._assign_best_agent(subtask)

            if not subtask.assigned_agent_id:
                raise Exception("Kein verfügbarer Agent gefunden")

            # Task im Task Manager erstellen
            task = await self.task_manager.create_task(
                task_type=subtask.task_type,
                name=subtask.name,
                payload=subtask.payload,
                priority=subtask.priority,
                timeout_seconds=self.subtask_timeout_seconds
            )

            # Task ausführen (vereinfacht - echte Agent-Execution würde hier stattfinden)
            # TODO: Implementiere echte Agent-Execution - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/117
            await asyncio.sleep(0.1)  # Simuliere Execution

            # Erfolgreich abgeschlossen
            subtask.end_time = datetime.utcnow()
            subtask.state = TaskState.COMPLETED
            subtask.execution_time_ms = (subtask.end_time - subtask.start_time).total_seconds() * 1000
            subtask.result = {"status": "completed", "task_id": task.task_id}

            logger.debug(f"Subtask {subtask.subtask_id} erfolgreich abgeschlossen")

        except Exception as e:
            # Fehler-Handling
            subtask.end_time = datetime.utcnow()
            subtask.state = TaskState.FAILED
            subtask.error_message = str(e)

            # Retry-Logic
            if subtask.retry_count < subtask.max_retries:
                subtask.retry_count += 1
                subtask.state = TaskState.PENDING
                logger.warning(f"Subtask {subtask.subtask_id} Retry {subtask.retry_count}/{subtask.max_retries}")

                # Exponential Backoff
                await asyncio.sleep(2 ** subtask.retry_count)
                return await self._execute_single_subtask(subtask, orchestration_id, request)

            logger.exception(f"Subtask {subtask.subtask_id} endgültig fehlgeschlagen: {e}")

        finally:
            subtask.updated_at = datetime.utcnow()

        return subtask

    async def _assign_best_agent(self, subtask: SubtaskExecution) -> str | None:
        """Weist besten verfügbaren Agent zu."""
        try:
            # Hole verfügbare Agents
            log_orchestrator_step(
                "Searching Agent Registry",
                "agent_call",
                subtask_id=subtask.subtask_id,
                required_capabilities=subtask.payload.get("required_capabilities", [])
            )

            available_agents_raw = await self.agent_registry.list_agents()

            if not available_agents_raw:
                log_orchestrator_step(
                    "No Agents Found in Registry",
                    "agent_call",
                    subtask_id=subtask.subtask_id,
                    agent_count=0
                )
                return None

            log_orchestrator_step(
                "Agents Found in Registry",
                "agent_call",
                subtask_id=subtask.subtask_id,
                agent_count=len(available_agents_raw) if isinstance(available_agents_raw, list) else len(available_agents_raw)
            )

            # Konvertiere zu Dictionary falls Liste zurückgegeben wird
            if isinstance(available_agents_raw, list):
                available_agents = {
                    getattr(agent, "id", f"agent_{i}"): agent
                    for i, agent in enumerate(available_agents_raw)
                }
            else:
                available_agents = available_agents_raw

            # Einfache Agent-Selection (TODO: Verbessern mit ML-Prediction) - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/117
            for agent_id, agent_info in available_agents.items():
                agent_capabilities = getattr(agent_info, "capabilities", [])

                # Prüfe Capability-Match
                if any(cap in agent_capabilities for cap in subtask.payload.get("required_capabilities", [])):
                    log_orchestrator_step(
                        "Agent Selected for Subtask",
                        "agent_call",
                        subtask_id=subtask.subtask_id,
                        selected_agent_id=agent_id,
                        agent_capabilities=agent_capabilities,
                        match_reason="capability_match"
                    )
                    return agent_id

            # Fallback: Ersten verfügbaren Agent
            fallback_agent = next(iter(available_agents.keys()))
            log_orchestrator_step(
                "Fallback Agent Selected",
                "agent_call",
                subtask_id=subtask.subtask_id,
                selected_agent_id=fallback_agent,
                match_reason="fallback_first_available"
            )
            return fallback_agent

        except Exception as e:
            logger.exception(f"Agent-Assignment fehlgeschlagen: {e}")
            return None

    async def _aggregate_results(
        self,
        orchestration_id: str,
        execution_plan: ExecutionPlan,
        subtask_results: list[SubtaskExecution],
        start_time: float
    ) -> OrchestrationResult:
        """Aggregiert Subtask-Results zu Orchestration-Result."""
        progress = self.active_orchestrations[orchestration_id]

        # Erfolgs-Status
        success = all(st.state == TaskState.COMPLETED for st in subtask_results)
        failed_subtasks = [st.subtask_id for st in subtask_results if st.state == TaskState.FAILED]

        # Performance-Metriken
        total_execution_time_ms = (time.time() - start_time) * 1000
        orchestration_overhead_ms = total_execution_time_ms - sum(
            st.execution_time_ms or 0 for st in subtask_results
        )

        # Parallelization-Achieved
        sequential_time = sum(st.execution_time_ms or 0 for st in subtask_results)
        parallelization_achieved = (
            1.0 - (total_execution_time_ms / sequential_time)
            if sequential_time > 0 else 0.0
        )

        # Results aggregieren
        results = {st.subtask_id: st.result for st in subtask_results if st.result}

        # Update Progress
        progress.state = OrchestrationState.COMPLETED if success else OrchestrationState.FAILED
        progress.actual_completion_time = datetime.utcnow()

        return OrchestrationResult(
            success=success,
            orchestration_id=orchestration_id,
            state=progress.state,
            results=results,
            total_execution_time_ms=total_execution_time_ms,
            orchestration_overhead_ms=orchestration_overhead_ms,
            parallelization_achieved=parallelization_achieved,
            subtask_results=subtask_results,
            failed_subtasks=failed_subtasks
        )

    async def cancel_orchestration(self, orchestration_id: str) -> bool:
        """Bricht Orchestration ab."""
        if orchestration_id in self._execution_tasks:
            self._execution_tasks[orchestration_id].cancel()
            del self._execution_tasks[orchestration_id]

        if orchestration_id in self.active_orchestrations:
            self.active_orchestrations[orchestration_id].state = OrchestrationState.CANCELLED

        await self._cleanup_orchestration(orchestration_id)
        return True

    async def get_orchestration_progress(self, orchestration_id: str) -> OrchestrationProgress | None:
        """Gibt aktuellen Progress zurück."""
        return self.active_orchestrations.get(orchestration_id)

    async def _cleanup_orchestration(self, orchestration_id: str) -> None:
        """Bereinigt Orchestration-State."""
        self._execution_tasks.pop(orchestration_id, None)
        # Behalte Progress für Monitoring (wird später durch Cleanup-Loop entfernt)

    async def _agent_load_monitor_loop(self) -> None:
        """Background-Loop für Agent-Load-Monitoring."""
        while self._is_running:
            try:
                await self._update_agent_loads()
                await asyncio.sleep(30)  # Update alle 30 Sekunden
            except Exception as e:
                logger.exception(f"Agent-Load-Monitor-Fehler: {e}")
                await asyncio.sleep(60)

    async def _update_agent_loads(self) -> None:
        """Aktualisiert Agent-Load-Informationen."""
        try:
            available_agents_raw = await self.agent_registry.list_agents()

            # Konvertiere zu Dictionary falls Liste zurückgegeben wird
            if isinstance(available_agents_raw, list):
                available_agents = {
                    getattr(agent, "id", f"agent_{i}"): agent
                    for i, agent in enumerate(available_agents_raw)
                }
            else:
                available_agents = available_agents_raw

            for agent_id, agent_info in available_agents.items():
                # TODO: Implementiere echte Load-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/117
                load_info = AgentLoadInfo(
                    agent_id=agent_id,
                    agent_type=getattr(agent_info, "agent_type", "unknown"),
                    capabilities=getattr(agent_info, "capabilities", []),
                    current_load=0.3,  # Mock
                    active_tasks=2,     # Mock
                    queued_tasks=1,     # Mock
                    max_concurrent_tasks=10,  # Mock
                    avg_response_time_ms=200.0,  # Mock
                    success_rate=0.95,  # Mock
                    error_rate=0.05,    # Mock
                    is_available=True,
                    last_heartbeat=datetime.utcnow()
                )

                self.agent_loads[agent_id] = load_info

        except Exception as e:
            logger.exception(f"Agent-Load-Update fehlgeschlagen: {e}")

    async def _orchestration_cleanup_loop(self) -> None:
        """Background-Loop für Orchestration-Cleanup."""
        while self._is_running:
            try:
                # Entferne abgeschlossene Orchestrations nach 1 Stunde
                cutoff_time = datetime.utcnow() - timedelta(hours=1)

                to_remove = []
                for orchestration_id, progress in self.active_orchestrations.items():
                    if (progress.is_completed and
                        progress.last_updated < cutoff_time):
                        to_remove.append(orchestration_id)

                for orchestration_id in to_remove:
                    del self.active_orchestrations[orchestration_id]
                    self.execution_plans.pop(orchestration_id, None)

                await asyncio.sleep(300)  # Cleanup alle 5 Minuten

            except Exception as e:
                logger.exception(f"Orchestration-Cleanup-Fehler: {e}")
                await asyncio.sleep(300)

    def _update_performance_stats(self, orchestration_time_ms: float, success: bool) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._orchestration_count += 1
        self._total_orchestration_time_ms += orchestration_time_ms

        if success:
            self._success_count += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_time = (
            self._total_orchestration_time_ms / self._orchestration_count
            if self._orchestration_count > 0 else 0.0
        )

        success_rate = (
            self._success_count / self._orchestration_count
            if self._orchestration_count > 0 else 0.0
        )

        return {
            "total_orchestrations": self._orchestration_count,
            "avg_orchestration_time_ms": avg_time,
            "success_rate": success_rate,
            "active_orchestrations": len(self.active_orchestrations),
            "meets_overhead_sla": avg_time < self.orchestration_overhead_target_ms,
            "agent_count": len(self.agent_loads)
        }
