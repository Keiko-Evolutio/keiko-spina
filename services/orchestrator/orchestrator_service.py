# backend/services/orchestrator/orchestrator_service.py
"""Eigenständiger Orchestrator Service.

Hauptservice für intelligente Task-Orchestration mit LLM-Integration,
Performance-Optimierung und Real-time Monitoring.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agents.registry.dynamic_registry import DynamicAgentRegistry
from kei_logging import get_logger, log_orchestrator_step, training_trace

# Optional ML-Dependencies
try:
    from services.ml.performance_prediction import PerformancePredictor
    ML_AVAILABLE = True
except ImportError:
    PerformancePredictor = None
    ML_AVAILABLE = False
from services.task_decomposition.data_models import DecompositionRequest, TaskType

from .data_models import ExecutionMode, HealthCheckResult, OrchestrationRequest, OrchestrationResult
from .event_integration import OrchestratorEventIntegration
from .execution_engine import ExecutionEngine
from .monitoring import OrchestrationMonitor
from .state_store import OrchestratorStateStore

if TYPE_CHECKING:
    from services.task_decomposition import TaskDecompositionEngine
    from task_management.core_task_manager import TaskManager

logger = get_logger(__name__)


class OrchestratorService:
    """Eigenständiger Orchestrator Service für intelligente Task-Coordination."""

    def __init__(
        self,
        task_manager: TaskManager | None = None,
        agent_registry: DynamicAgentRegistry | None = None,
        decomposition_engine: TaskDecompositionEngine | None = None,
        performance_predictor: PerformancePredictor | None = None,
        state_store: OrchestratorStateStore | None = None,
        event_integration: OrchestratorEventIntegration | None = None
    ):
        """Initialisiert Orchestrator Service.

        Args:
            task_manager: Task Manager für Task-Execution
            agent_registry: Agent Registry für verfügbare Agents
            decomposition_engine: Task Decomposition Engine aus TASK 3
            performance_predictor: Performance Predictor aus TASK 2
            state_store: State Store für Plan-Persistierung und Recovery
            event_integration: Event Integration für asynchrone Kommunikation
        """
        # Dependencies
        self.task_manager = task_manager or self._get_default_task_manager()
        self.agent_registry = agent_registry or self._get_default_agent_registry()
        self.decomposition_engine = decomposition_engine or self._get_default_decomposition_engine()
        self.performance_predictor = performance_predictor if ML_AVAILABLE else None
        self.state_store = state_store or OrchestratorStateStore()
        self.event_integration = event_integration or OrchestratorEventIntegration()

        # Core-Komponenten
        self.monitor = OrchestrationMonitor()
        self.execution_engine = ExecutionEngine(
            task_manager=self.task_manager,
            agent_registry=self.agent_registry,
            decomposition_engine=self.decomposition_engine,
            performance_predictor=self.performance_predictor if ML_AVAILABLE else None,
            monitor=self.monitor,
            state_store=self.state_store,
            event_integration=self.event_integration
        )

        # Service-Konfiguration
        self.service_id = str(uuid.uuid4())
        self.service_version = "1.0.0"

        # Issue #55 Performance Targets
        self.max_concurrent_orchestrations = int(os.getenv("KEI_ORCHESTRATOR_MAX_CONCURRENT_ORCHESTRATIONS", "20"))
        self.orchestration_timeout_seconds = int(os.getenv("KEI_ORCHESTRATOR_ORCHESTRATION_TIMEOUT_SECONDS", "3600"))
        self.task_analysis_timeout_seconds = float(os.getenv("KEI_ORCHESTRATOR_TASK_ANALYSIS_TIMEOUT_SECONDS", "2.0"))
        self.decomposition_timeout_seconds = float(os.getenv("KEI_ORCHESTRATOR_DECOMPOSITION_TIMEOUT_SECONDS", "5.0"))
        self.agent_selection_timeout_seconds = float(os.getenv("KEI_ORCHESTRATOR_AGENT_SELECTION_TIMEOUT_SECONDS", "1.0"))
        self.plan_persistence_timeout_ms = int(os.getenv("KEI_ORCHESTRATOR_PLAN_PERSISTENCE_TIMEOUT_MS", "500"))

        # Service-State
        self._is_running = False
        self._start_time = datetime.utcnow()
        self._orchestration_count = 0
        self._success_count = 0

        # Background-Tasks
        self._background_tasks: list[asyncio.Task] = []

        logger.info({
            "event": "orchestrator_service_initialized",
            "service_id": self.service_id,
            "version": self.service_version
        })

    async def start(self) -> None:
        """Startet Orchestrator Service."""
        if self._is_running:
            logger.warning("Orchestrator Service bereits gestartet")
            return

        try:
            # Starte Komponenten
            await self.state_store.start()
            await self.event_integration.start()
            await self.monitor.start()
            await self.execution_engine.start()

            # Starte Background-Tasks
            self._background_tasks = [
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._metrics_collection_loop())
            ]

            self._is_running = True
            self._start_time = datetime.utcnow()

            logger.info({
                "event": "orchestrator_service_started",
                "service_id": self.service_id,
                "start_time": self._start_time.isoformat()
            })

        except Exception as e:
            logger.exception(f"Orchestrator Service Start fehlgeschlagen: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stoppt Orchestrator Service."""
        if not self._is_running:
            return

        self._is_running = False

        try:
            # Stoppe Background-Tasks
            for task in self._background_tasks:
                task.cancel()

            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

            # Stoppe Komponenten
            await self.execution_engine.stop()
            await self.monitor.stop()
            await self.event_integration.stop()
            await self.state_store.stop()

            logger.info({
                "event": "orchestrator_service_stopped",
                "service_id": self.service_id,
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds()
            })

        except Exception as e:
            logger.exception(f"Orchestrator Service Stop-Fehler: {e}")

    async def orchestrate_task(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Orchestriert Task-Execution.

        Args:
            request: Orchestration-Request

        Returns:
            Orchestration-Result

        Raises:
            Exception: Bei Service-Fehlern
        """
        if not self._is_running:
            raise Exception("Orchestrator Service nicht gestartet")

        orchestration_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.info({
                "event": "orchestration_request_received",
                "orchestration_id": orchestration_id,
                "task_id": request.task_id,
                "task_type": request.task_type.value,
                "execution_mode": request.execution_mode.value,
                "user_id": request.user_id
            })

            # Validiere Request
            await self._validate_orchestration_request(request)

            # Starte Monitoring
            await self.monitor.track_orchestration_started(
                orchestration_id,
                {
                    "task_id": request.task_id,
                    "task_type": request.task_type.value,
                    "execution_mode": request.execution_mode.value,
                    "max_parallel_tasks": request.max_parallel_tasks,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "correlation_id": request.correlation_id
                }
            )

            # Publiziere Orchestration-Started Event
            await self.event_integration.publish_orchestration_started(
                orchestration_id=orchestration_id,
                request=request
            )

            # Führe Orchestration aus
            result = await self.execution_engine.execute_orchestration(request)

            # Persistiere Plan für Recovery (falls erfolgreich)
            if result.success and result.plan:
                try:
                    state_id = await self.state_store.persist_plan(
                        plan=result.plan,
                        tenant_id=request.tenant_id
                    )
                    logger.debug(f"Plan persistiert: {result.plan.plan_id} -> {state_id}")
                except Exception as e:
                    logger.warning(f"Plan-Persistierung fehlgeschlagen: {e}")

            # Update Service-Statistiken
            orchestration_time_ms = (time.time() - start_time) * 1000
            self._orchestration_count += 1

            if result.success:
                self._success_count += 1

            # Finalisiere Monitoring
            await self.monitor.track_orchestration_completed(
                orchestration_id,
                result.success,
                result.total_execution_time_ms,
                result.orchestration_overhead_ms
            )

            # Publiziere Orchestration-Completed Event
            await self.event_integration.publish_orchestration_completed(
                orchestration_id=orchestration_id,
                result=result
            )

            logger.info({
                "event": "orchestration_completed",
                "orchestration_id": orchestration_id,
                "success": result.success,
                "execution_time_ms": orchestration_time_ms,
                "subtasks_count": len(result.subtask_results),
                "parallelization_achieved": result.parallelization_achieved
            })

            return result

        except Exception as e:
            logger.exception({
                "event": "orchestration_failed",
                "orchestration_id": orchestration_id,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            })

            # Error-Monitoring
            await self.monitor.track_orchestration_completed(
                orchestration_id,
                False,
                (time.time() - start_time) * 1000,
                0.0
            )

            raise

    async def recover_plan(
        self,
        plan_id: str,
        tenant_id: str | None = None
    ) -> Any | None:
        """Stellt Plan aus Persistierung wieder her.

        Args:
            plan_id: Plan-ID
            tenant_id: Tenant-ID für Access-Control

        Returns:
            Wiederhergestellter Plan oder None
        """
        try:
            plan = await self.state_store.recover_plan(plan_id, tenant_id)

            if plan:
                logger.info({
                    "event": "plan_recovered",
                    "plan_id": plan_id,
                    "tenant_id": tenant_id,
                    "subtasks_count": len(plan.subtasks)
                })
            else:
                logger.warning(f"Plan {plan_id} nicht gefunden")

            return plan

        except Exception as e:
            logger.exception(f"Plan-Recovery fehlgeschlagen: {e}")
            return None

    async def resume_orchestration(
        self,
        orchestration_id: str,
        tenant_id: str | None = None
    ) -> OrchestrationResult | None:
        """Setzt unterbrochene Orchestration fort.

        Args:
            orchestration_id: Orchestration-ID
            tenant_id: Tenant-ID

        Returns:
            Orchestration-Result oder None
        """
        try:
            # Erstelle Checkpoint-Konfiguration
            config = {
                "configurable": {
                    "thread_id": f"orchestration_{orchestration_id}",
                    "checkpoint_ns": "orchestrator"
                }
            }

            # Lade Checkpoint
            checkpoint_data = self.state_store.checkpoint_saver.get(config)

            if not checkpoint_data:
                logger.warning(f"Kein Checkpoint für Orchestration {orchestration_id} gefunden")
                return None

            logger.info({
                "event": "orchestration_resume_started",
                "orchestration_id": orchestration_id,
                "tenant_id": tenant_id,
                "checkpoint_found": True
            })

            # Hier würde die eigentliche Resume-Logik implementiert werden
            # Für jetzt geben wir ein Placeholder-Result zurück

            return OrchestrationResult(
                success=True,
                orchestration_id=orchestration_id,
                subtask_results=[],
                total_execution_time_ms=0,
                orchestration_overhead_ms=0,
                parallelization_achieved=0.0,
                plan=None
            )

        except Exception as e:
            logger.exception(f"Orchestration-Resume fehlgeschlagen: {e}")
            return None

    async def get_orchestration_progress(self, orchestration_id: str) -> dict[str, Any] | None:
        """Holt aktuellen Orchestration-Progress.

        Args:
            orchestration_id: Orchestration-ID

        Returns:
            Progress-Informationen oder None
        """
        progress = await self.execution_engine.get_orchestration_progress(orchestration_id)

        if not progress:
            return None

        return {
            "orchestration_id": progress.orchestration_id,
            "state": progress.state.value,
            "completion_percentage": progress.completion_percentage,
            "total_subtasks": progress.total_subtasks,
            "completed_subtasks": progress.completed_subtasks,
            "failed_subtasks": progress.failed_subtasks,
            "running_subtasks": progress.running_subtasks,
            "pending_subtasks": progress.pending_subtasks,
            "start_time": progress.start_time.isoformat() if progress.start_time else None,
            "estimated_completion_time": progress.estimated_completion_time.isoformat() if progress.estimated_completion_time else None,
            "execution_efficiency": progress.execution_efficiency,
            "resource_utilization": progress.resource_utilization,
            "error_count": progress.error_count,
            "last_error": progress.last_error,
            "last_updated": progress.last_updated.isoformat()
        }

    async def cancel_orchestration(self, orchestration_id: str) -> bool:
        """Bricht Orchestration ab.

        Args:
            orchestration_id: Orchestration-ID

        Returns:
            True wenn erfolgreich abgebrochen
        """
        try:
            success = await self.execution_engine.cancel_orchestration(orchestration_id)

            if success:
                logger.info({
                    "event": "orchestration_cancelled",
                    "orchestration_id": orchestration_id
                })

            return success

        except Exception as e:
            logger.exception(f"Orchestration-Cancellation fehlgeschlagen: {e}")
            return False

    async def get_service_health(self) -> HealthCheckResult:
        """Führt Service-Health-Check durch.

        Returns:
            Health-Check-Ergebnis
        """
        try:
            # Service-Status
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()

            # Komponenten-Health-Checks
            task_decomposition_healthy = await self._check_task_decomposition_health()
            performance_prediction_healthy = await self._check_performance_prediction_health()
            agent_registry_healthy = await self._check_agent_registry_health()
            task_manager_healthy = await self._check_task_manager_health()

            # Service-Health
            service_healthy = (
                self._is_running and
                task_decomposition_healthy and
                agent_registry_healthy and
                task_manager_healthy
            )

            # Performance-Metriken
            execution_stats = self.execution_engine.get_performance_stats()
            await self.monitor.get_real_time_stats()

            # Success-Rate
            success_rate = (
                self._success_count / self._orchestration_count
                if self._orchestration_count > 0 else 1.0
            )

            # Resource-Usage (vereinfacht)
            try:
                import psutil
                memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_usage_percent = psutil.Process().cpu_percent()
            except ImportError:
                psutil = None  # Fallback für psutil import
                memory_usage_mb = 0.0
                cpu_usage_percent = 0.0

            return HealthCheckResult(
                service_healthy=service_healthy,
                service_version=self.service_version,
                uptime_seconds=uptime_seconds,
                task_decomposition_healthy=task_decomposition_healthy,
                performance_prediction_healthy=performance_prediction_healthy,
                agent_registry_healthy=agent_registry_healthy,
                task_manager_healthy=task_manager_healthy,
                active_orchestrations=execution_stats.get("active_orchestrations", 0),
                total_orchestrations=self._orchestration_count,
                avg_orchestration_time_ms=execution_stats.get("avg_orchestration_time_ms", 0.0),
                success_rate=success_rate,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                dependencies_healthy={
                    "task_decomposition": task_decomposition_healthy,
                    "performance_prediction": performance_prediction_healthy,
                    "agent_registry": agent_registry_healthy,
                    "task_manager": task_manager_healthy,
                    "monitoring": True  # Monitor ist immer healthy wenn Service läuft
                }
            )

        except Exception as e:
            logger.exception(f"Health-Check fehlgeschlagen: {e}")

            return HealthCheckResult(
                service_healthy=False,
                service_version=self.service_version,
                uptime_seconds=0.0,
                task_decomposition_healthy=False,
                performance_prediction_healthy=False,
                agent_registry_healthy=False,
                task_manager_healthy=False,
                active_orchestrations=0,
                total_orchestrations=0,
                avg_orchestration_time_ms=0.0,
                success_rate=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0
            )

    async def get_service_metrics(self) -> dict[str, Any]:
        """Holt Service-Metriken.

        Returns:
            Service-Metriken
        """
        execution_stats = self.execution_engine.get_performance_stats()
        monitoring_stats = await self.monitor.get_real_time_stats()

        return {
            "service": {
                "service_id": self.service_id,
                "version": self.service_version,
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
                "is_running": self._is_running
            },
            "orchestrations": {
                "total_count": self._orchestration_count,
                "success_count": self._success_count,
                "success_rate": self._success_count / self._orchestration_count if self._orchestration_count > 0 else 0.0,
                "active_count": execution_stats.get("active_orchestrations", 0)
            },
            "performance": {
                "avg_orchestration_time_ms": execution_stats.get("avg_orchestration_time_ms", 0.0),
                "meets_overhead_sla": execution_stats.get("meets_overhead_sla", False),
                "agent_count": execution_stats.get("agent_count", 0)
            },
            "monitoring": monitoring_stats
        }

    async def _validate_orchestration_request(self, request: OrchestrationRequest) -> None:
        """Validiert Orchestration-Request."""
        if not request.task_id:
            raise ValueError("Task-ID ist erforderlich")

        if not request.task_name:
            raise ValueError("Task-Name ist erforderlich")

        if request.max_parallel_tasks <= 0:
            raise ValueError("Max-Parallel-Tasks muss > 0 sein")

        if request.timeout_seconds <= 0:
            raise ValueError("Timeout muss > 0 sein")

    def _get_default_task_manager(self) -> TaskManager:
        """Holt Default Task Manager."""
        try:
            from task_management.core_task_manager import task_manager
            return task_manager
        except ImportError:
            logger.warning("Task Manager nicht verfügbar - verwende Mock")
            return None

    def _get_default_agent_registry(self) -> DynamicAgentRegistry:
        """Holt Default Agent Registry."""
        try:
            return DynamicAgentRegistry()
        except Exception:
            logger.warning("Agent Registry nicht verfügbar - verwende Mock")
            return None

    def _get_default_decomposition_engine(self) -> TaskDecompositionEngine:
        """Holt Default Decomposition Engine."""
        try:
            from services.task_decomposition import create_task_decomposition_engine
            return create_task_decomposition_engine(
                task_manager=self.task_manager,
                agent_registry=self.agent_registry,
                performance_predictor=self.performance_predictor
            )
        except Exception:
            logger.warning("Task Decomposition Engine nicht verfügbar - verwende Mock")
            return None

    async def _check_task_decomposition_health(self) -> bool:
        """Prüft Task Decomposition Engine Health."""
        try:
            if not self.decomposition_engine:
                return False

            # Einfacher Health-Check
            stats = self.decomposition_engine.get_performance_stats()
            return stats.get("meets_sla", False)
        except Exception:
            return False

    async def _check_performance_prediction_health(self) -> bool:
        """Prüft Performance Predictor Health."""
        try:
            if not self.performance_predictor:
                return True  # Optional - Service funktioniert ohne

            # Einfacher Health-Check
            stats = self.performance_predictor.get_performance_stats()
            return stats.get("meets_sla", False)
        except Exception:
            return True  # Optional

    async def _check_agent_registry_health(self) -> bool:
        """Prüft Agent Registry Health."""
        try:
            if not self.agent_registry:
                return False

            # Teste Agent-Listing
            agents = await self.agent_registry.list_agents()
            return len(agents) > 0
        except Exception:
            return False

    async def _check_task_manager_health(self) -> bool:
        """Prüft Task Manager Health."""
        try:
            if not self.task_manager:
                return False

            # Einfacher Health-Check
            return True  # Task Manager ist immer healthy wenn verfügbar
        except Exception:
            return False

    async def _health_check_loop(self) -> None:
        """Background-Loop für regelmäßige Health-Checks."""
        while self._is_running:
            try:
                health = await self.get_service_health()

                if not health.service_healthy:
                    logger.warning({
                        "event": "service_health_degraded",
                        "dependencies": health.dependencies_healthy
                    })

                await asyncio.sleep(60)  # Health-Check alle 60 Sekunden

            except Exception as e:
                logger.exception(f"Health-Check-Loop-Fehler: {e}")
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self) -> None:
        """Background-Loop für Metriken-Collection."""
        while self._is_running:
            try:
                metrics = await self.get_service_metrics()

                # Log wichtige Metriken
                logger.info({
                    "event": "service_metrics",
                    "orchestrations_total": metrics["orchestrations"]["total_count"],
                    "success_rate": metrics["orchestrations"]["success_rate"],
                    "avg_time_ms": metrics["performance"]["avg_orchestration_time_ms"],
                    "active_orchestrations": metrics["orchestrations"]["active_count"]
                })

                await asyncio.sleep(300)  # Metriken alle 5 Minuten

            except Exception as e:
                logger.exception(f"Metriken-Collection-Fehler: {e}")
                await asyncio.sleep(300)

    @training_trace(context={"component": "orchestrator_service", "phase": "voice_request"})
    async def handle_voice_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Verarbeitet Voice-Request über LLM-powered Orchestrator.

        Args:
            request: Voice-Request mit 'text', 'user_id', etc.

        Returns:
            Orchestrator-Response mit success, response_text, etc.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        log_orchestrator_step(
            "Processing Voice Request",
            "orchestration",
            request_id=request_id,
            user_input=request.get("text", ""),
            user_id=request.get("user_id"),
            session_id=request.get("session_id")
        )

        try:
            # 1. Erstelle Decomposition-Request
            decomposition_request = await self._create_decomposition_request(request, request_id)

            # 2. Task-Decomposition
            log_orchestrator_step(
                "Starting Task Decomposition",
                "task_decomposition",
                request_id=request_id,
                task_type=decomposition_request.task_type.value
            )

            decomposition_result = await self.decomposition_engine.decompose_task(decomposition_request)

            if not decomposition_result.success:
                log_orchestrator_step(
                    "Task Decomposition Failed",
                    "task_decomposition",
                    request_id=request_id,
                    error=decomposition_result.error_message
                )
                return self._create_error_response(
                    request_id,
                    "Task-Decomposition fehlgeschlagen",
                    decomposition_result.error_message,
                    time.time() - start_time
                )

            # 3. Orchestration ausführen
            orchestration_request = OrchestrationRequest(
                task_id=decomposition_request.task_id,
                execution_mode=ExecutionMode.PARALLEL,
                enable_decomposition=True,
                enable_monitoring=True,
                enable_recovery=True
            )

            log_orchestrator_step(
                "Starting Task Execution",
                "orchestration",
                request_id=request_id,
                subtask_count=len(decomposition_result.plan.subtasks) if decomposition_result.plan else 0
            )

            orchestration_result = await self.execution_engine.execute_orchestration(orchestration_request)

            # 4. Response erstellen
            execution_time = time.time() - start_time

            if orchestration_result.success:
                log_orchestrator_step(
                    "Voice Request Completed Successfully",
                    "orchestration",
                    request_id=request_id,
                    execution_time_seconds=execution_time,
                    subtasks_completed=len(orchestration_result.completed_subtasks)
                )

                return {
                    "success": True,
                    "response_text": orchestration_result.result_summary or "Aufgabe erfolgreich abgeschlossen",
                    "task_id": decomposition_request.task_id,
                    "request_id": request_id,
                    "execution_time": execution_time,
                    "agent_used": self._extract_primary_agent(orchestration_result),
                    "data": orchestration_result.result_data
                }
            log_orchestrator_step(
                "Voice Request Failed",
                "orchestration",
                request_id=request_id,
                error=orchestration_result.error_message,
                execution_time_seconds=execution_time
            )

            return self._create_error_response(
                request_id,
                "Task-Ausführung fehlgeschlagen",
                orchestration_result.error_message,
                execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Orchestrator Service Fehler: {e}")

            log_orchestrator_step(
                "Voice Request Exception",
                "orchestration",
                request_id=request_id,
                error=str(e),
                execution_time_seconds=execution_time
            )

            return self._create_error_response(
                request_id,
                "Unerwarteter Fehler",
                str(e),
                execution_time
            )

    async def _create_decomposition_request(
        self,
        voice_request: dict[str, Any],
        request_id: str
    ) -> DecompositionRequest:
        """Erstellt Decomposition-Request aus Voice-Request."""
        user_input = voice_request.get("text", "")

        # Einfache Task-Type-Erkennung
        task_type = self._detect_task_type(user_input)

        log_orchestrator_step(
            "Task Type Detected",
            "task_decomposition",
            request_id=request_id,
            detected_type=task_type.value,
            user_input_length=len(user_input)
        )

        return DecompositionRequest(
            task_id=str(uuid.uuid4()),
            task_description=user_input,
            task_type=task_type,
            user_id=voice_request.get("user_id"),
            session_id=voice_request.get("session_id"),
            enable_llm_analysis=True,
            enable_agent_matching=True,
            max_subtasks=5,
            max_decomposition_time_seconds=30
        )

    def _detect_task_type(self, user_input: str) -> TaskType:
        """Erkennt Task-Type aus User-Input."""
        user_input_lower = user_input.lower()

        if any(keyword in user_input_lower for keyword in ["bild", "image", "foto", "erstelle", "generiere", "zeichne"]):
            return TaskType.IMAGE_GENERATION
        if any(keyword in user_input_lower for keyword in ["suche", "recherche", "finde", "web", "internet"]):
            return TaskType.WEB_SEARCH
        return TaskType.CONVERSATION

    def _extract_primary_agent(self, result: OrchestrationResult) -> str | None:
        """Extrahiert primären Agent aus Orchestration-Result."""
        if result.completed_subtasks:
            # Nimm ersten abgeschlossenen Subtask
            first_subtask = result.completed_subtasks[0]
            return first_subtask.get("agent_id")
        return None

    def _create_error_response(
        self,
        request_id: str,
        message: str,
        error: str | None,
        execution_time: float
    ) -> dict[str, Any]:
        """Erstellt Error-Response."""
        return {
            "success": False,
            "response_text": f"❌ {message}: {error}" if error else f"❌ {message}",
            "request_id": request_id,
            "execution_time": execution_time,
            "error": error
        }
