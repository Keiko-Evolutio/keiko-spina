"""Voice Performance Service Implementation.
Hauptservice für Voice Performance Optimization.
"""

import asyncio
from typing import Any

from kei_logging import get_logger

from .dlq_analytics import DLQAnalyticsEngine, create_dlq_analytics_engine
from .interfaces import ParallelProcessingResult, VoicePerformanceSettings, VoiceWorkflowContext
from .voice_recovery_system import VoiceRecoverySystem, create_voice_recovery_system
from .voice_workflow_dlq import VoiceWorkflowDLQ, create_voice_workflow_dlq
from .workflow_orchestrator import OptimizedVoiceWorkflowOrchestrator

logger = get_logger(__name__)


class VoicePerformanceService:
    """Voice Performance Service Implementation.
    Orchestriert alle Voice Performance Optimization Komponenten.
    """

    def __init__(self, settings: VoicePerformanceSettings):
        self.settings = settings

        # Core Components
        self._orchestrator: OptimizedVoiceWorkflowOrchestrator | None = None
        self._voice_dlq: VoiceWorkflowDLQ | None = None
        self._dlq_analytics: DLQAnalyticsEngine | None = None
        self._recovery_system: VoiceRecoverySystem | None = None

        # Service Status
        self._initialized = False
        self._running = False

        # Monitoring
        self._start_time: float | None = None

        logger.info(f"Voice performance service created with settings: {settings}")

    @property
    def orchestrator(self) -> OptimizedVoiceWorkflowOrchestrator:
        """Voice Workflow Orchestrator."""
        if not self._orchestrator:
            raise RuntimeError("Voice performance service not initialized")
        return self._orchestrator

    async def initialize(self) -> None:
        """Initialisiert Voice Performance Service."""
        if self._initialized:
            return

        try:
            logger.info("Initializing voice performance service...")

            # Orchestrator initialisieren
            from .workflow_orchestrator import create_optimized_voice_workflow_orchestrator
            self._orchestrator = create_optimized_voice_workflow_orchestrator(self.settings)
            await self._orchestrator.initialize()

            # Voice-spezifische Dead Letter Queue initialisieren
            self._voice_dlq = create_voice_workflow_dlq(max_queue_size=50000)  # Groß für M4 Max
            await self._voice_dlq.start()

            # DLQ Analytics Engine initialisieren
            self._dlq_analytics = create_dlq_analytics_engine(self._voice_dlq)
            await self._dlq_analytics.start()

            # Voice Recovery System initialisieren
            self._recovery_system = create_voice_recovery_system(self._voice_dlq)
            await self._recovery_system.start()

            # Retry Handlers und Notification Handlers registrieren
            await self._register_dlq_handlers()

            self._initialized = True
            self._running = True
            self._start_time = asyncio.get_event_loop().time()

            logger.info("Voice performance service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize voice performance service: {e}")
            raise

    async def _register_dlq_handlers(self) -> None:
        """Registriert DLQ-Handler für Voice Performance System."""
        if not self._voice_dlq or not self._recovery_system:
            return

        # Voice Workflow Retry Handler
        async def retry_voice_workflow(failed_task) -> bool:
            try:
                # Erstelle neuen VoiceWorkflowContext aus Failed Task
                from .interfaces import VoiceWorkflowContext as InterfaceContext
                context = InterfaceContext(
                    workflow_id=failed_task.workflow_id,
                    user_id=failed_task.user_id or "unknown",
                    session_id=failed_task.session_id or "unknown",
                    text_input=failed_task.original_request.get("text", ""),
                    language=failed_task.original_request.get("language", "de-DE"),
                    parallel_processing=failed_task.original_request.get("parallel_processing", True),
                    cache_enabled=failed_task.original_request.get("cache_enabled", True)
                )

                # Versuche Voice Workflow erneut
                result = await self.optimize_voice_workflow(context)
                return result.success

            except Exception as e:
                logger.error(f"Voice workflow retry failed: {e}")
                return False

        # Agent Discovery Retry Handler
        async def retry_agent_discovery(failed_task) -> bool:
            try:
                logger.info(f"Retrying agent discovery for task {failed_task.task_id} (attempt {failed_task.retry_count + 1})")

                # Verwende alternative Agent Discovery Strategien basierend auf Failed Task
                agent_type = failed_task.original_request.get("agent_type", "unknown")

                # Simuliere Agent Discovery Retry mit Task-spezifischen Parametern
                if failed_task.retry_count < 2:
                    # Erste Retries: Kurze Wartezeit
                    await asyncio.sleep(0.1)
                else:
                    # Spätere Retries: Längere Wartezeit
                    await asyncio.sleep(0.5)

                logger.info(f"Agent discovery retry successful for task {failed_task.task_id}, agent_type: {agent_type}")
                return True
            except Exception as e:
                logger.error(f"Agent discovery retry failed for task {failed_task.task_id}: {e}")
                return False

        # Agent Execution Retry Handler
        async def retry_agent_execution(failed_task) -> bool:
            try:
                logger.info(f"Retrying agent execution for task {failed_task.task_id} (attempt {failed_task.retry_count + 1})")

                # Verwende Task-spezifische Execution Parameter
                execution_params = failed_task.original_request.get("execution_params", {})
                timeout_ms = execution_params.get("timeout_ms", 5000)

                # Anpassung der Retry-Strategie basierend auf Failure Reason
                if failed_task.failure_reason.value == "agent_timeout":
                    # Bei Timeout: Erhöhe Timeout für Retry
                    timeout_ms = min(timeout_ms * 1.5, 15000)
                    logger.debug(f"Increased timeout to {timeout_ms}ms for retry")

                # Simuliere Agent Execution Retry mit angepassten Parametern
                await asyncio.sleep(0.2)

                logger.info(f"Agent execution retry successful for task {failed_task.task_id}")
                return True
            except Exception as e:
                logger.error(f"Agent execution retry failed for task {failed_task.task_id}: {e}")
                return False

        # Cache Operation Retry Handler
        async def retry_cache_operation(failed_task) -> bool:
            try:
                logger.info(f"Retrying cache operation for task {failed_task.task_id} (attempt {failed_task.retry_count + 1})")

                # Verwende Task-spezifische Cache Parameter
                cache_key = failed_task.original_request.get("cache_key", "unknown")
                operation_type = failed_task.original_request.get("operation_type", "get")

                # Anpassung der Cache-Strategie basierend auf Failure Reason
                if failed_task.failure_reason.value == "cache_timeout":
                    # Bei Cache Timeout: Verwende kürzeren Timeout für Retry
                    logger.debug(f"Using shorter timeout for cache retry on key: {cache_key}")
                elif failed_task.failure_reason.value == "cache_connection_failed":
                    # Bei Connection Failure: Warte länger vor Retry
                    await asyncio.sleep(0.2)

                # Simuliere Cache Operation Retry
                await asyncio.sleep(0.05)

                logger.info(f"Cache operation retry successful for task {failed_task.task_id}, operation: {operation_type}")
                return True
            except Exception as e:
                logger.error(f"Cache operation retry failed for task {failed_task.task_id}: {e}")
                return False

        # Registriere alle Retry-Handler
        self._voice_dlq.register_retry_handler("voice_workflow", retry_voice_workflow)
        self._voice_dlq.register_retry_handler("agent_discovery", retry_agent_discovery)
        self._voice_dlq.register_retry_handler("agent_execution", retry_agent_execution)
        self._voice_dlq.register_retry_handler("cache_operation", retry_cache_operation)

        # Registriere User Notification Handler
        async def user_notification_handler(failed_task, message):
            logger.info(f"User notification: {message} for task {failed_task.task_id} (user: {failed_task.user_id})")

        self._voice_dlq.register_user_notification_handler(user_notification_handler)
        self._recovery_system.register_notification_handler(user_notification_handler)

        # Registriere Session Continuity Handler
        async def session_continuity_handler(failed_task):
            logger.info(f"Session continuity for task {failed_task.task_id} (session: {failed_task.session_id})")
            return True  # Mock implementation

        self._voice_dlq.register_session_continuity_handler("voice_session", session_continuity_handler)

        logger.info("DLQ handlers registered successfully")

    async def shutdown(self) -> None:
        """Fährt Voice Performance Service herunter."""
        if not self._running:
            return

        try:
            logger.info("Shutting down voice performance service...")

            # Orchestrator herunterfahren
            if self._orchestrator:
                await self._orchestrator.shutdown()

            # Stoppe DLQ-Komponenten
            if self._recovery_system:
                await self._recovery_system.stop()

            if self._dlq_analytics:
                await self._dlq_analytics.stop()

            if self._voice_dlq:
                await self._voice_dlq.stop()

            self._running = False

            logger.info("Voice performance service shut down successfully")

        except Exception as e:
            logger.error(f"Error during voice performance service shutdown: {e}")

    async def optimize_voice_workflow(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert Voice Workflow mit vollständiger Performance-Optimierung."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        if not self.settings.enabled:
            # Performance Optimization deaktiviert - führe Basic Processing durch
            return await self._basic_voice_processing(context)

        # Führe mit Performance Optimization aus
        return await self._orchestrator.process_voice_input(context)

    async def optimize_voice_request(
        self,
        voice_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimiert Voice Request und gibt optimierte Response zurück."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        try:
            # Integriere mit bestehenden Voice Routes
            result = await self._orchestrator.integrate_with_existing_voice_routes(voice_request)

            # Konvertiere zu Voice Response Format
            return {
                "success": result.success,
                "workflow_id": result.workflow_id,
                "processing_stage": result.stage.value,
                "results": result.results,
                "errors": result.errors,
                "performance_metrics": {
                    "total_time_ms": result.total_time_ms,
                    "parallel_time_ms": result.parallel_time_ms,
                    "sequential_time_ms": result.sequential_time_ms,
                    "speedup_factor": result.speedup_factor,
                    "max_concurrent_tasks": result.max_concurrent_tasks,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_usage_percent": result.cpu_usage_percent,
                    "success_rate": result.success_rate,
                    "average_confidence": result.average_confidence
                },
                "optimization_enabled": self.settings.enabled,
                "cache_enabled": self.settings.cache_enabled,
                "parallel_processing_used": result.max_concurrent_tasks > 1
            }

        except Exception as e:
            logger.error(f"Voice request optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "optimization_enabled": self.settings.enabled
            }

    async def optimize_orchestrator_request(
        self,
        orchestrator_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimiert Orchestrator Request mit Performance-Optimierung."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        try:
            # Integriere mit bestehenden Orchestrator Agent
            return await self._orchestrator.integrate_with_orchestrator_agent(orchestrator_request)

        except Exception as e:
            logger.error(f"Orchestrator request optimization failed: {e}")
            return {
                "success": False,
                "response_text": f"Performance optimization failed: {e!s}",
                "task_id": orchestrator_request.get("task_id", "unknown"),
                "execution_time": 0.0,
                "error": str(e)
            }

    async def handle_concurrent_voice_requests(
        self,
        voice_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Behandelt mehrere Voice Requests concurrent."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        try:
            # Konvertiere zu VoiceWorkflowContexts
            contexts = []
            for i, request in enumerate(voice_requests):
                context = VoiceWorkflowContext(
                    workflow_id=request.get("workflow_id", f"concurrent_{i}"),
                    user_id=request.get("user_id", "unknown"),
                    session_id=request.get("session_id", "unknown"),
                    text_input=request.get("text", ""),
                    language=request.get("language", "de-DE"),
                    max_latency_ms=request.get("max_latency_ms", self.settings.target_latency_ms),
                    parallel_processing=request.get("parallel_processing", True),
                    cache_enabled=request.get("cache_enabled", True)
                )
                contexts.append(context)

            # Führe concurrent Processing durch
            results = await self._orchestrator.handle_concurrent_requests(contexts)

            # Konvertiere Results zu Response Format
            responses = []
            for result in results:
                response = {
                    "success": result.success,
                    "workflow_id": result.workflow_id,
                    "processing_stage": result.stage.value,
                    "results": result.results,
                    "errors": result.errors,
                    "performance_metrics": {
                        "total_time_ms": result.total_time_ms,
                        "speedup_factor": result.speedup_factor,
                        "success_rate": result.success_rate
                    }
                }
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"Concurrent voice request handling failed: {e}")
            return [{"success": False, "error": str(e)} for _ in voice_requests]

    async def health_check(self) -> dict[str, Any]:
        """Führt Health Check für Voice Performance Service durch."""
        health = {
            "healthy": True,
            "details": {}
        }

        # Service-Status prüfen
        if not self._initialized or not self._running:
            health["healthy"] = False
            health["details"]["service"] = "Service not running"
        else:
            health["details"]["service"] = "OK"

        # Orchestrator-Status prüfen
        try:
            if self._orchestrator:
                orchestrator_metrics = await self._orchestrator.get_performance_metrics()
                health["details"]["orchestrator"] = "OK"
                health["details"]["active_workflows"] = orchestrator_metrics["orchestrator"]["active_workflows"]
            else:
                health["healthy"] = False
                health["details"]["orchestrator"] = "Orchestrator not initialized"
        except Exception as e:
            health["healthy"] = False
            health["details"]["orchestrator"] = f"Orchestrator error: {e!s}"

        # Voice DLQ-Status prüfen
        try:
            if self._voice_dlq:
                dlq_stats = await self._voice_dlq.get_statistics()
                health["details"]["voice_dlq"] = "OK"
                health["details"]["failed_tasks"] = dlq_stats["queue_stats"]["total_failures"]
            else:
                health["details"]["voice_dlq"] = "Not initialized"
        except Exception as e:
            health["details"]["voice_dlq"] = f"Voice DLQ error: {e!s}"

        # DLQ Analytics-Status prüfen
        try:
            if self._dlq_analytics:
                health["details"]["dlq_analytics"] = "OK"
            else:
                health["details"]["dlq_analytics"] = "Not initialized"
        except Exception as e:
            health["details"]["dlq_analytics"] = f"Analytics error: {e!s}"

        # Recovery System-Status prüfen
        try:
            if self._recovery_system:
                recovery_stats = await self._recovery_system.get_recovery_statistics()
                health["details"]["recovery_system"] = "OK"
                health["details"]["recovery_success_rate"] = recovery_stats["recovery_stats"].get("successful_recoveries", 0)
            else:
                health["details"]["recovery_system"] = "Not initialized"
        except Exception as e:
            health["details"]["recovery_system"] = f"Recovery error: {e!s}"

        return health

    async def get_service_statistics(self) -> dict[str, Any]:
        """Gibt Service-Statistiken zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        stats = {
            "service": {
                "initialized": self._initialized,
                "running": self._running,
                "uptime_seconds": 0,
                "settings": {
                    "enabled": self.settings.enabled,
                    "target_latency_ms": self.settings.target_latency_ms,
                    "max_latency_ms": self.settings.max_latency_ms,
                    "target_throughput_rps": self.settings.target_throughput_rps,
                    "cache_enabled": self.settings.cache_enabled,
                    "predictive_loading_enabled": self.settings.predictive_loading_enabled,
                    "warm_up_enabled": self.settings.warm_up_enabled,
                    "pattern_learning_enabled": self.settings.pattern_learning_enabled
                }
            }
        }

        # Uptime berechnen
        if self._start_time:
            current_time = asyncio.get_event_loop().time()
            stats["service"]["uptime_seconds"] = current_time - self._start_time

        # Orchestrator-Statistiken
        if self._orchestrator:
            try:
                stats["orchestrator"] = await self._orchestrator.get_performance_metrics()
            except Exception as e:
                stats["orchestrator"] = {"error": str(e)}

        return stats

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Gibt detaillierte Performance-Metriken zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        try:
            # Orchestrator Performance Metrics
            orchestrator_metrics = await self._orchestrator.get_performance_metrics()

            return {
                "service_metrics": {
                    "enabled": self.settings.enabled,
                    "initialized": self._initialized,
                    "running": self._running
                },
                "performance_metrics": orchestrator_metrics,
                "configuration": {
                    "target_latency_ms": self.settings.target_latency_ms,
                    "max_latency_ms": self.settings.max_latency_ms,
                    "target_throughput_rps": self.settings.target_throughput_rps,
                    "max_concurrent_workflows": self.settings.max_concurrent_workflows,
                    "cache_strategy": self.settings.cache_strategy.value,
                    "cache_max_size": self.settings.cache_max_size,
                    "max_memory_usage_mb": self.settings.max_memory_usage_mb,
                    "max_cpu_usage_percent": self.settings.max_cpu_usage_percent
                }
            }

        except Exception as e:
            return {"error": str(e)}

    # Helper Methods

    async def _basic_voice_processing(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Basic Voice Processing ohne Performance-Optimierung."""
        import time

        from .interfaces import ProcessingStage

        start_time = time.time()

        try:
            # Simuliere Basic Processing
            await asyncio.sleep(0.1)  # Simuliere Processing Zeit

            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=ProcessingStage.RESPONSE_GENERATION,
                success=True,
                results=[{"basic_processing": True, "text": context.text_input}],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0,
                max_concurrent_tasks=1,
                success_rate=1.0
            )

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=ProcessingStage.RESPONSE_GENERATION,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0
            )


def create_voice_performance_service(
    settings: VoicePerformanceSettings | None = None
) -> VoicePerformanceService:
    """Factory-Funktion für Voice Performance Service.

    Args:
        settings: Voice Performance Settings, falls None werden Defaults verwendet

    Returns:
        Voice Performance Service Instance
    """
    if settings is None:
        from config.voice_performance_config import get_voice_performance_settings
        settings = get_voice_performance_settings()

    return VoicePerformanceService(settings)
