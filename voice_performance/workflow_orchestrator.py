"""Optimized Voice Workflow Orchestrator.
Orchestriert Voice-to-Orchestrator Workflows mit Performance-Optimierung.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .interfaces import (
    IVoiceWorkflowOrchestrator,
    ParallelProcessingResult,
    ProcessingStage,
    VoicePerformanceSettings,
    VoiceWorkflowContext,
)
from .optimizer import VoicePerformanceOptimizer

logger = get_logger(__name__)


class OptimizedVoiceWorkflowOrchestrator(IVoiceWorkflowOrchestrator):
    """Optimized Voice Workflow Orchestrator Implementation.
    Orchestriert Voice Workflows mit maximaler Performance-Optimierung.
    """

    def __init__(self, settings: VoicePerformanceSettings):
        self.settings = settings

        # Core Components
        self.performance_optimizer = VoicePerformanceOptimizer(settings)

        # Workflow Tracking
        self._active_workflows: dict[str, VoiceWorkflowContext] = {}
        self._workflow_results: dict[str, ParallelProcessingResult] = {}
        self._workflow_lock = asyncio.Lock()

        # Performance Metrics
        self._orchestrator_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_latency_ms": 0.0,
            "peak_concurrent_workflows": 0,
            "current_concurrent_workflows": 0,
            "total_processing_time_ms": 0.0,
            "average_speedup_factor": 1.0
        }

        # Circuit Breaker Integration
        self._circuit_breaker_enabled = settings.circuit_breaker_enabled
        self._failure_count = 0
        self._last_failure_time: datetime | None = None

        # Rate Limiting Integration
        self._rate_limiting_enabled = settings.rate_limiting_enabled
        self._request_timestamps: list[datetime] = []

        logger.info("Optimized voice workflow orchestrator initialized")

    async def initialize(self) -> None:
        """Initialisiert Voice Workflow Orchestrator."""
        try:
            # Initialisiere Performance Optimizer
            await self.performance_optimizer.initialize()

            logger.info("Optimized voice workflow orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize voice workflow orchestrator: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt Voice Workflow Orchestrator herunter."""
        try:
            # Stoppe Performance Optimizer
            await self.performance_optimizer.shutdown()

            logger.info("Optimized voice workflow orchestrator shut down successfully")

        except Exception as e:
            logger.error(f"Error during voice workflow orchestrator shutdown: {e}")

    async def process_voice_input(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Verarbeitet Voice Input mit Performance-Optimierung."""
        start_time = time.time()

        try:
            # Circuit Breaker Check
            if not await self._check_circuit_breaker():
                raise Exception("Circuit breaker is open")

            # Rate Limiting Check
            if not await self._check_rate_limits():
                raise Exception("Rate limit exceeded")

            # Registriere Workflow
            await self._register_workflow(context)

            # Führe optimierte Pipeline durch
            result = await self.performance_optimizer.optimize_full_pipeline(context)

            # Update Statistics
            total_time_ms = (time.time() - start_time) * 1000
            await self._update_workflow_stats(total_time_ms, result.speedup_factor, result.success)

            # Circuit Breaker Success
            if result.success:
                await self._record_circuit_breaker_success()
            else:
                await self._record_circuit_breaker_failure()

            # Speichere Result
            self._workflow_results[context.workflow_id] = result

            # Deregistriere Workflow
            await self._deregister_workflow(context)

            return result

        except Exception as e:
            logger.error(f"Voice input processing failed: {e}")

            # Circuit Breaker Failure
            await self._record_circuit_breaker_failure()

            # Update Statistics
            total_time_ms = (time.time() - start_time) * 1000
            await self._update_workflow_stats(total_time_ms, 1.0, False)

            # Deregistriere Workflow
            await self._deregister_workflow(context)

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=ProcessingStage.SPEECH_TO_TEXT,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0
            )

    async def execute_parallel_pipeline(
        self,
        context: VoiceWorkflowContext,
        stages: list[ProcessingStage]
    ) -> list[ParallelProcessingResult]:
        """Führt Pipeline-Stages parallel aus."""
        start_time = time.time()

        try:
            # Erstelle Tasks für alle Stages
            stage_tasks = []

            for stage in stages:
                if stage == ProcessingStage.SPEECH_TO_TEXT:
                    task = asyncio.create_task(
                        self.performance_optimizer.optimize_speech_to_text(context)
                    )
                elif stage == ProcessingStage.AGENT_DISCOVERY:
                    task = asyncio.create_task(
                        self.performance_optimizer.optimize_agent_discovery(context)
                    )
                elif stage == ProcessingStage.AGENT_EXECUTION:
                    # Mock Agents für Parallel Execution
                    mock_agents = ["agent_1", "agent_2"]
                    task = asyncio.create_task(
                        self.performance_optimizer.optimize_agent_execution(context, mock_agents)
                    )
                elif stage == ProcessingStage.RESPONSE_GENERATION:
                    # Mock Results für Response Generation
                    mock_results = [{"result": "data"}]
                    task = asyncio.create_task(
                        self.performance_optimizer.optimize_response_generation(context, mock_results)
                    )
                else:
                    # Unbekannte Stage - Skip
                    continue

                stage_tasks.append(task)

            # Führe alle Stages parallel aus
            stage_results = await asyncio.gather(*stage_tasks, return_exceptions=True)

            # Filtere erfolgreiche Results
            successful_results = []
            for result in stage_results:
                if isinstance(result, ParallelProcessingResult):
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Pipeline stage failed: {result}")

            total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Parallel pipeline completed: {len(successful_results)}/{len(stages)} stages successful in {total_time_ms:.1f}ms")

            return successful_results

        except Exception as e:
            logger.error(f"Parallel pipeline execution failed: {e}")
            return []

    async def handle_concurrent_requests(
        self,
        contexts: list[VoiceWorkflowContext]
    ) -> list[ParallelProcessingResult]:
        """Behandelt mehrere Voice Requests concurrent."""
        start_time = time.time()

        try:
            # Prüfe Concurrent Workflow Limits
            if len(contexts) > self.settings.max_concurrent_workflows:
                logger.warning(f"Too many concurrent workflows: {len(contexts)} > {self.settings.max_concurrent_workflows}")
                contexts = contexts[:self.settings.max_concurrent_workflows]

            # Erstelle Tasks für alle Workflows
            workflow_tasks = []
            for context in contexts:
                task = asyncio.create_task(self.process_voice_input(context))
                workflow_tasks.append(task)

            # Führe alle Workflows parallel aus
            workflow_results = await asyncio.gather(*workflow_tasks, return_exceptions=True)

            # Filtere erfolgreiche Results
            successful_results = []
            for result in workflow_results:
                if isinstance(result, ParallelProcessingResult):
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Concurrent workflow failed: {result}")

            total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Concurrent workflows completed: {len(successful_results)}/{len(contexts)} successful in {total_time_ms:.1f}ms")

            return successful_results

        except Exception as e:
            logger.error(f"Concurrent request handling failed: {e}")
            return []

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Gibt Performance-Metriken zurück."""
        # Performance Optimizer Metrics
        optimizer_metrics = await self.performance_optimizer.get_performance_metrics()

        # Orchestrator-spezifische Metrics
        async with self._workflow_lock:
            orchestrator_metrics = {
                "orchestrator_stats": self._orchestrator_stats.copy(),
                "active_workflows": len(self._active_workflows),
                "workflow_results_cached": len(self._workflow_results),
                "circuit_breaker": {
                    "enabled": self._circuit_breaker_enabled,
                    "failure_count": self._failure_count,
                    "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None
                },
                "rate_limiting": {
                    "enabled": self._rate_limiting_enabled,
                    "recent_requests": len(self._request_timestamps)
                }
            }

        return {
            "orchestrator": orchestrator_metrics,
            "optimizer": optimizer_metrics
        }

    # Helper Methods

    async def _check_circuit_breaker(self) -> bool:
        """Prüft Circuit Breaker Status."""
        if not self._circuit_breaker_enabled:
            return True

        # Einfache Circuit Breaker Logic
        if self._failure_count >= self.settings.circuit_breaker_failure_threshold:
            if self._last_failure_time:
                time_since_failure = (datetime.utcnow() - self._last_failure_time).total_seconds()
                if time_since_failure < self.settings.circuit_breaker_timeout_seconds:
                    logger.warning("Circuit breaker is open")
                    return False
                # Reset Circuit Breaker
                self._failure_count = 0
                self._last_failure_time = None
                logger.info("Circuit breaker reset")

        return True

    async def _check_rate_limits(self) -> bool:
        """Prüft Rate Limits."""
        if not self._rate_limiting_enabled:
            return True

        now = datetime.utcnow()

        # Entferne alte Timestamps (älter als 1 Minute)
        cutoff_time = now - timedelta(minutes=1)
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if ts > cutoff_time
        ]

        # Prüfe Rate Limit
        if len(self._request_timestamps) >= self.settings.target_throughput_rps:
            logger.warning("Rate limit exceeded")
            return False

        # Füge aktuellen Timestamp hinzu
        self._request_timestamps.append(now)
        return True

    async def _record_circuit_breaker_success(self) -> None:
        """Registriert Circuit Breaker Success."""
        if self._circuit_breaker_enabled and self._failure_count > 0:
            self._failure_count = max(0, self._failure_count - 1)

    async def _record_circuit_breaker_failure(self) -> None:
        """Registriert Circuit Breaker Failure."""
        if self._circuit_breaker_enabled:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

    async def _register_workflow(self, context: VoiceWorkflowContext) -> None:
        """Registriert aktiven Workflow."""
        async with self._workflow_lock:
            self._active_workflows[context.workflow_id] = context
            current_count = len(self._active_workflows)

            self._orchestrator_stats["peak_concurrent_workflows"] = max(self._orchestrator_stats["peak_concurrent_workflows"], current_count)

            self._orchestrator_stats["current_concurrent_workflows"] = current_count

    async def _deregister_workflow(self, context: VoiceWorkflowContext) -> None:
        """Deregistriert Workflow."""
        async with self._workflow_lock:
            if context.workflow_id in self._active_workflows:
                del self._active_workflows[context.workflow_id]

            self._orchestrator_stats["current_concurrent_workflows"] = len(self._active_workflows)

    async def _update_workflow_stats(
        self,
        latency_ms: float,
        speedup_factor: float,
        success: bool
    ) -> None:
        """Aktualisiert Workflow-Statistiken."""
        self._orchestrator_stats["total_workflows"] += 1
        self._orchestrator_stats["total_processing_time_ms"] += latency_ms

        if success:
            self._orchestrator_stats["successful_workflows"] += 1
        else:
            self._orchestrator_stats["failed_workflows"] += 1

        # Exponential Moving Average
        alpha = 0.1
        self._orchestrator_stats["average_latency_ms"] = (
            alpha * latency_ms + (1 - alpha) * self._orchestrator_stats["average_latency_ms"]
        )
        self._orchestrator_stats["average_speedup_factor"] = (
            alpha * speedup_factor + (1 - alpha) * self._orchestrator_stats["average_speedup_factor"]
        )

    # Integration Methods

    async def integrate_with_existing_voice_routes(
        self,
        voice_request: dict[str, Any]
    ) -> ParallelProcessingResult:
        """Integriert mit bestehenden Voice Routes."""
        try:
            # Konvertiere Voice Request zu VoiceWorkflowContext
            context = VoiceWorkflowContext(
                workflow_id=str(uuid.uuid4()),
                user_id=voice_request.get("user_id", "unknown"),
                session_id=voice_request.get("session_id", "unknown"),
                text_input=voice_request.get("text", ""),
                language=voice_request.get("language", "de-DE"),
                max_latency_ms=voice_request.get("max_latency_ms", self.settings.target_latency_ms),
                parallel_processing=voice_request.get("parallel_processing", True),
                cache_enabled=voice_request.get("cache_enabled", True)
            )

            # Verarbeite mit optimierter Pipeline
            return await self.process_voice_input(context)

        except Exception as e:
            logger.error(f"Voice route integration failed: {e}")

            return ParallelProcessingResult(
                workflow_id="unknown",
                stage=ProcessingStage.SPEECH_TO_TEXT,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=0.0,
                parallel_time_ms=0.0,
                sequential_time_ms=0.0,
                speedup_factor=1.0
            )

    async def integrate_with_orchestrator_agent(
        self,
        orchestrator_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Integriert mit bestehenden Orchestrator Agent."""
        try:
            # Konvertiere zu Voice Workflow Context
            context = VoiceWorkflowContext(
                workflow_id=orchestrator_request.get("task_id", str(uuid.uuid4())),
                user_id=orchestrator_request.get("user_id", "unknown"),
                session_id=orchestrator_request.get("session_id", "unknown"),
                text_input=orchestrator_request.get("text", ""),
                parallel_processing=True,
                cache_enabled=True
            )

            # Führe Performance-optimierte Verarbeitung durch
            result = await self.process_voice_input(context)

            # Konvertiere zurück zu Orchestrator Format
            return {
                "success": result.success,
                "response_text": f"Optimized response for: {context.text_input}",
                "task_id": context.workflow_id,
                "execution_time": result.total_time_ms / 1000.0,
                "speedup_factor": result.speedup_factor,
                "parallel_processing_used": True,
                "cache_hit": any("cache_hit" in str(r) for r in result.results),
                "performance_metrics": {
                    "total_time_ms": result.total_time_ms,
                    "parallel_time_ms": result.parallel_time_ms,
                    "sequential_time_ms": result.sequential_time_ms,
                    "speedup_factor": result.speedup_factor,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_usage_percent": result.cpu_usage_percent,
                    "success_rate": result.success_rate,
                    "average_confidence": result.average_confidence
                }
            }

        except Exception as e:
            logger.error(f"Orchestrator agent integration failed: {e}")

            return {
                "success": False,
                "response_text": f"Performance optimization failed: {e!s}",
                "task_id": orchestrator_request.get("task_id", "unknown"),
                "execution_time": 0.0,
                "error": str(e)
            }


def create_optimized_voice_workflow_orchestrator(
    settings: VoicePerformanceSettings
) -> OptimizedVoiceWorkflowOrchestrator:
    """Factory-Funktion für Optimized Voice Workflow Orchestrator."""
    return OptimizedVoiceWorkflowOrchestrator(settings)
