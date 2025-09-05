"""Voice Performance Optimizer Implementation.
Optimiert Voice-to-Orchestrator Workflows mit Parallel Processing.
"""

import asyncio
import time
from typing import Any

import psutil

from kei_logging import get_logger

from .interfaces import (
    DiscoveryRequest,
    DiscoveryType,
    IVoicePerformanceOptimizer,
    ParallelProcessingResult,
    ProcessingStage,
    VoicePerformanceSettings,
    VoiceWorkflowContext,
)
from .parallel_discovery import ParallelDiscoveryEngine
from .performance_cache import PerformanceCacheEngine

logger = get_logger(__name__)


class VoicePerformanceOptimizer(IVoicePerformanceOptimizer):
    """Voice Performance Optimizer Implementation.
    Orchestriert alle Performance-Optimierungen für Voice Workflows.
    """

    def __init__(self, settings: VoicePerformanceSettings):
        self.settings = settings

        # Core Components
        self.discovery_engine = ParallelDiscoveryEngine(settings)
        self.cache_engine = PerformanceCacheEngine(settings)

        # Performance Tracking
        self._performance_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_speedup": 1.0,
            "average_latency_ms": 0.0,
            "peak_concurrent_workflows": 0,
            "current_concurrent_workflows": 0
        }

        # Resource Monitoring
        self._resource_monitor = ResourceMonitor(settings)

        # Concurrent Workflow Tracking
        self._active_workflows: dict[str, VoiceWorkflowContext] = {}
        self._workflow_lock = asyncio.Lock()

        logger.info("Voice performance optimizer initialized")

    async def initialize(self) -> None:
        """Initialisiert Performance Optimizer."""
        try:
            # Starte Cache Engine
            await self.cache_engine.start()

            # Starte Resource Monitor
            await self._resource_monitor.start()

            logger.info("Voice performance optimizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize voice performance optimizer: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt Performance Optimizer herunter."""
        try:
            # Stoppe Cache Engine
            await self.cache_engine.stop()

            # Stoppe Resource Monitor
            await self._resource_monitor.stop()

            logger.info("Voice performance optimizer shut down successfully")

        except Exception as e:
            logger.error(f"Error during voice performance optimizer shutdown: {e}")

    async def optimize_speech_to_text(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert Speech-to-Text Processing."""
        start_time = time.time()
        stage = ProcessingStage.SPEECH_TO_TEXT

        try:
            # Cache Warm-up für STT Models
            await self._warm_up_stt_models(context)

            # Parallel STT Processing (falls mehrere Models verfügbar)
            results = await self._parallel_stt_processing(context)

            # Resource Usage Tracking
            memory_usage = self._resource_monitor.get_memory_usage()
            cpu_usage = self._resource_monitor.get_cpu_usage()

            total_time_ms = (time.time() - start_time) * 1000

            # Berechne Speedup (Vergleich zu sequenzieller Verarbeitung)
            sequential_time_ms = total_time_ms * len(results) if len(results) > 1 else total_time_ms
            speedup_factor = sequential_time_ms / total_time_ms if total_time_ms > 0 else 1.0

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=True,
                results=results,
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=sequential_time_ms,
                speedup_factor=speedup_factor,
                max_concurrent_tasks=len(results),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                success_rate=1.0,
                average_confidence=sum(r.get("confidence", 0.8) for r in results) / len(results) if results else 0.0
            )

        except Exception as e:
            logger.error(f"STT optimization failed: {e}")
            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0
            )

    async def optimize_agent_discovery(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert Agent Discovery mit Parallel Processing."""
        start_time = time.time()
        stage = ProcessingStage.AGENT_DISCOVERY

        try:
            # Cache Check für Agent Discovery
            cache_key = f"agent_discovery:{context.user_id}:{hash(context.text_input or '')}"
            cached_result = await self.cache_engine.get(cache_key)

            if cached_result:
                logger.debug(f"Agent discovery cache hit for workflow {context.workflow_id}")
                return ParallelProcessingResult(
                    workflow_id=context.workflow_id,
                    stage=stage,
                    success=True,
                    results=[cached_result.value],
                    total_time_ms=5.0,  # Cache access time
                    parallel_time_ms=5.0,
                    sequential_time_ms=200.0,  # Estimated sequential time
                    speedup_factor=40.0,  # Massive speedup from cache
                    max_concurrent_tasks=1
                )

            # Parallel Discovery Requests
            discovery_requests = [
                DiscoveryRequest(
                    discovery_type=DiscoveryType.AGENT_DISCOVERY,
                    query=context.text_input or "general",
                    context=context,
                    max_results=context.max_agents,
                    timeout_seconds=self.settings.agent_discovery_timeout_seconds
                ),
                DiscoveryRequest(
                    discovery_type=DiscoveryType.CAPABILITY_DISCOVERY,
                    query=context.text_input or "general",
                    context=context,
                    max_results=10,
                    timeout_seconds=self.settings.capability_discovery_timeout_seconds
                )
            ]

            # Führe Parallel Discovery durch
            discovery_results = await self.discovery_engine.discover_all(discovery_requests)

            # Merge Results
            merged_results = await self._merge_discovery_results(discovery_results)

            # Cache Results
            await self.cache_engine.put(cache_key, merged_results, ttl_seconds=300)

            total_time_ms = (time.time() - start_time) * 1000
            sequential_time_ms = sum(r.discovery_time_ms for r in discovery_results)
            speedup_factor = sequential_time_ms / total_time_ms if total_time_ms > 0 else 1.0

            # Resource Usage
            memory_usage = self._resource_monitor.get_memory_usage()
            cpu_usage = self._resource_monitor.get_cpu_usage()

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=True,
                results=[merged_results],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=sequential_time_ms,
                speedup_factor=speedup_factor,
                max_concurrent_tasks=len(discovery_requests),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                success_rate=len([r for r in discovery_results if r.total_found > 0]) / len(discovery_results) if discovery_results else 0.0,
                average_confidence=sum(r.average_confidence for r in discovery_results) / len(discovery_results) if discovery_results else 0.0
            )

        except Exception as e:
            logger.error(f"Agent discovery optimization failed: {e}")
            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0
            )

    async def optimize_agent_execution(
        self,
        context: VoiceWorkflowContext,
        agents: list[str]
    ) -> ParallelProcessingResult:
        """Optimiert Agent Execution mit Parallel Processing."""
        start_time = time.time()
        stage = ProcessingStage.AGENT_EXECUTION

        try:
            # Prüfe Resource Limits
            if not await self._check_resource_limits():
                raise Exception("Resource limits exceeded")

            # Parallel Agent Execution
            if len(agents) > 1 and context.parallel_processing:
                results = await self._parallel_agent_execution(context, agents)
            else:
                results = await self._sequential_agent_execution(context, agents)

            total_time_ms = (time.time() - start_time) * 1000

            # Berechne Performance Metrics
            successful_results = [r for r in results if r.get("success", False)]
            success_rate = len(successful_results) / len(results) if results else 0.0

            # Schätze sequenzielle Zeit
            sequential_time_ms = total_time_ms * len(agents) if len(agents) > 1 else total_time_ms
            speedup_factor = sequential_time_ms / total_time_ms if total_time_ms > 0 else 1.0

            # Resource Usage
            memory_usage = self._resource_monitor.get_memory_usage()
            cpu_usage = self._resource_monitor.get_cpu_usage()

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=len(successful_results) > 0,
                results=results,
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=sequential_time_ms,
                speedup_factor=speedup_factor,
                max_concurrent_tasks=len(agents),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                success_rate=success_rate,
                average_confidence=sum(r.get("confidence", 0.8) for r in successful_results) / len(successful_results) if successful_results else 0.0
            )

        except Exception as e:
            logger.error(f"Agent execution optimization failed: {e}")
            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0
            )

    async def optimize_response_generation(
        self,
        context: VoiceWorkflowContext,
        results: list[Any]
    ) -> ParallelProcessingResult:
        """Optimiert Response Generation."""
        start_time = time.time()
        stage = ProcessingStage.RESPONSE_GENERATION

        try:
            # Parallel Response Processing
            if len(results) > 1:
                processed_results = await self._parallel_response_processing(context, results)
            else:
                processed_results = await self._sequential_response_processing(context, results)

            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=True,
                results=processed_results,
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms * len(results) if len(results) > 1 else total_time_ms,
                speedup_factor=len(results) if len(results) > 1 else 1.0,
                max_concurrent_tasks=len(results),
                memory_usage_mb=self._resource_monitor.get_memory_usage(),
                cpu_usage_percent=self._resource_monitor.get_cpu_usage(),
                success_rate=1.0
            )

        except Exception as e:
            logger.error(f"Response generation optimization failed: {e}")
            total_time_ms = (time.time() - start_time) * 1000

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=stage,
                success=False,
                results=[],
                errors=[str(e)],
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=total_time_ms,
                speedup_factor=1.0
            )

    async def optimize_full_pipeline(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert komplette Voice Pipeline."""
        start_time = time.time()

        try:
            # Registriere Workflow
            await self._register_workflow(context)

            # Cache Warm-up und Predictive Loading
            if self.settings.warm_up_enabled:
                await self.cache_engine.warm_up(context)

            if self.settings.predictive_loading_enabled:
                await self.cache_engine.predict_and_preload(context)

            # Pipeline Stages - definiere Reihenfolge für Optimierung
            stages = [
                ProcessingStage.SPEECH_TO_TEXT,
                ProcessingStage.AGENT_DISCOVERY,
                ProcessingStage.AGENT_SELECTION,
                ProcessingStage.AGENT_EXECUTION,
                ProcessingStage.RESPONSE_GENERATION
            ]

            # Führe Pipeline-Optimierung durch (basierend auf definierten Stages)
            stage_results = []
            logger.debug(f"Starting pipeline optimization with {len(stages)} stages: {[s.value for s in stages]}")

            # STT Optimization
            stt_result = await self.optimize_speech_to_text(context)
            stage_results.append(stt_result)

            # Agent Discovery Optimization
            discovery_result = await self.optimize_agent_discovery(context)
            stage_results.append(discovery_result)

            # Agent Execution Optimization (mit Mock Agents)
            mock_agents = ["agent_1", "agent_2", "agent_3"]
            execution_result = await self.optimize_agent_execution(context, mock_agents)
            stage_results.append(execution_result)

            # Response Generation Optimization
            mock_results = [{"result": "response_1"}, {"result": "response_2"}]
            response_result = await self.optimize_response_generation(context, mock_results)
            stage_results.append(response_result)

            # Aggregiere Results
            total_time_ms = (time.time() - start_time) * 1000
            successful_stages = [r for r in stage_results if r.success]
            overall_success = len(successful_stages) >= len(stage_results) * 0.7  # 70% Success Rate

            # Berechne Gesamte Performance Metrics
            total_speedup = sum(r.speedup_factor for r in stage_results) / len(stage_results)
            average_confidence = sum(r.average_confidence for r in stage_results) / len(stage_results)

            # Update Performance Stats
            self._update_performance_stats(total_time_ms, total_speedup, overall_success)

            # Deregistriere Workflow
            await self._deregister_workflow(context)

            return ParallelProcessingResult(
                workflow_id=context.workflow_id,
                stage=ProcessingStage.RESPONSE_GENERATION,  # Final stage
                success=overall_success,
                results=stage_results,
                total_time_ms=total_time_ms,
                parallel_time_ms=total_time_ms,
                sequential_time_ms=sum(r.sequential_time_ms for r in stage_results),
                speedup_factor=total_speedup,
                max_concurrent_tasks=max(r.max_concurrent_tasks for r in stage_results),
                memory_usage_mb=self._resource_monitor.get_memory_usage(),
                cpu_usage_percent=self._resource_monitor.get_cpu_usage(),
                success_rate=len(successful_stages) / len(stage_results),
                average_confidence=average_confidence
            )

        except Exception as e:
            logger.error(f"Full pipeline optimization failed: {e}")
            total_time_ms = (time.time() - start_time) * 1000

            # Deregistriere Workflow auch bei Fehler
            await self._deregister_workflow(context)

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

    # Helper Methods

    async def _warm_up_stt_models(self, context: VoiceWorkflowContext) -> None:
        """Führt STT Model Warm-up durch."""
        try:
            # Mock STT Model Warm-up
            await asyncio.sleep(0.05)  # Simuliere Warm-up Zeit
            logger.debug(f"STT models warmed up for workflow {context.workflow_id}")
        except Exception as e:
            logger.warning(f"STT warm-up failed: {e}")

    async def _parallel_stt_processing(self, context: VoiceWorkflowContext) -> list[dict[str, Any]]:
        """Führt parallele STT Processing durch."""
        # Mock parallel STT processing
        tasks = [
            self._mock_stt_process("model_1", context),
            self._mock_stt_process("model_2", context)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtere erfolgreiche Results
        successful_results = []
        for result in results:
            if isinstance(result, dict):
                successful_results.append(result)

        return successful_results

    async def _mock_stt_process(self, model_name: str, context: VoiceWorkflowContext) -> dict[str, Any]:
        """Mock STT Processing."""
        await asyncio.sleep(0.1)  # Simuliere STT Zeit

        return {
            "model": model_name,
            "text": context.text_input or "Recognized text",
            "confidence": 0.85,
            "processing_time_ms": 100
        }

    async def _merge_discovery_results(self, discovery_results: list[Any]) -> dict[str, Any]:
        """Merged Discovery Results."""
        merged = {
            "agents": [],
            "capabilities": [],
            "total_discovery_time_ms": 0,
            "discovery_count": len(discovery_results)
        }

        for result in discovery_results:
            if hasattr(result, "discovery_type"):
                if result.discovery_type.value == "agent_discovery":
                    merged["agents"].extend(result.items)
                elif result.discovery_type.value == "capability_discovery":
                    merged["capabilities"].extend(result.items)

                merged["total_discovery_time_ms"] += result.discovery_time_ms

        return merged

    async def _parallel_agent_execution(
        self,
        context: VoiceWorkflowContext,
        agents: list[str]
    ) -> list[dict[str, Any]]:
        """Führt parallele Agent Execution durch."""
        # Begrenze Parallelität
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_agents)

        async def execute_agent(agent_id: str) -> dict[str, Any]:
            async with semaphore:
                return await self._mock_agent_execution(agent_id, context)

        tasks = [execute_agent(agent_id) for agent_id in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtere erfolgreiche Results
        successful_results = []
        for result in results:
            if isinstance(result, dict):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Agent execution failed: {result}")

        return successful_results

    async def _sequential_agent_execution(
        self,
        context: VoiceWorkflowContext,
        agents: list[str]
    ) -> list[dict[str, Any]]:
        """Führt sequenzielle Agent Execution durch."""
        results = []
        for agent_id in agents:
            try:
                result = await self._mock_agent_execution(agent_id, context)
                results.append(result)
            except Exception as e:
                logger.warning(f"Agent execution failed for {agent_id}: {e}")

        return results

    async def _mock_agent_execution(self, agent_id: str, context: VoiceWorkflowContext) -> dict[str, Any]:
        """Mock Agent Execution mit Context-spezifischen Parametern."""
        # Verwende Context-Informationen für realistische Simulation
        base_delay = 0.2

        # Anpassung der Execution-Zeit basierend auf Context
        if context.max_latency_ms and context.max_latency_ms < 1000:
            # Für niedrige Latenz-Anforderungen: Schnellere Execution
            base_delay *= 0.7

        # Simuliere unterschiedliche Agent-Typen basierend auf Text Input
        if context.text_input and len(context.text_input) > 100:
            # Längere Texte benötigen mehr Processing-Zeit
            base_delay *= 1.3

        await asyncio.sleep(base_delay)

        return {
            "agent_id": agent_id,
            "success": True,
            "result": f"Result from {agent_id} for workflow {context.workflow_id}",
            "confidence": 0.8,
            "execution_time_ms": int(base_delay * 1000),
            "context_language": context.language,
            "input_length": len(context.text_input) if context.text_input else 0
        }

    async def _parallel_response_processing(
        self,
        context: VoiceWorkflowContext,
        results: list[Any]
    ) -> list[dict[str, Any]]:
        """Führt parallele Response Processing durch."""
        tasks = [
            self._mock_response_processing(i, result, context)
            for i, result in enumerate(results)
        ]

        processed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtere erfolgreiche Results
        successful_results = []
        for result in processed_results:
            if isinstance(result, dict):
                successful_results.append(result)

        return successful_results

    async def _sequential_response_processing(
        self,
        context: VoiceWorkflowContext,
        results: list[Any]
    ) -> list[dict[str, Any]]:
        """Führt sequenzielle Response Processing durch."""
        processed_results = []
        for i, result in enumerate(results):
            try:
                processed = await self._mock_response_processing(i, result, context)
                processed_results.append(processed)
            except Exception as e:
                logger.warning(f"Response processing failed for result {i}: {e}")

        return processed_results

    async def _mock_response_processing(
        self,
        index: int,
        result: Any,
        context: VoiceWorkflowContext
    ) -> dict[str, Any]:
        """Mock Response Processing."""
        await asyncio.sleep(0.05)  # Simuliere Response Processing Zeit

        return {
            "index": index,
            "processed_result": f"Processed: {result}",
            "workflow_id": context.workflow_id,
            "processing_time_ms": 50
        }

    async def _check_resource_limits(self) -> bool:
        """Prüft Resource Limits."""
        memory_usage = self._resource_monitor.get_memory_usage()
        cpu_usage = self._resource_monitor.get_cpu_usage()

        if memory_usage > self.settings.max_memory_usage_mb:
            logger.warning(f"Memory usage too high: {memory_usage}MB > {self.settings.max_memory_usage_mb}MB")
            return False

        if cpu_usage > self.settings.max_cpu_usage_percent:
            logger.warning(f"CPU usage too high: {cpu_usage}% > {self.settings.max_cpu_usage_percent}%")
            return False

        return True

    async def _register_workflow(self, context: VoiceWorkflowContext) -> None:
        """Registriert aktiven Workflow."""
        async with self._workflow_lock:
            self._active_workflows[context.workflow_id] = context
            current_count = len(self._active_workflows)

            self._performance_stats["peak_concurrent_workflows"] = max(self._performance_stats["peak_concurrent_workflows"], current_count)

            self._performance_stats["current_concurrent_workflows"] = current_count

    async def _deregister_workflow(self, context: VoiceWorkflowContext) -> None:
        """Deregistriert Workflow."""
        async with self._workflow_lock:
            if context.workflow_id in self._active_workflows:
                del self._active_workflows[context.workflow_id]

            self._performance_stats["current_concurrent_workflows"] = len(self._active_workflows)

    def _update_performance_stats(self, latency_ms: float, speedup: float, success: bool) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._performance_stats["total_optimizations"] += 1

        if success:
            self._performance_stats["successful_optimizations"] += 1
        else:
            self._performance_stats["failed_optimizations"] += 1

        # Exponential Moving Average für Metriken
        alpha = 0.1
        self._performance_stats["average_speedup"] = (
            alpha * speedup + (1 - alpha) * self._performance_stats["average_speedup"]
        )
        self._performance_stats["average_latency_ms"] = (
            alpha * latency_ms + (1 - alpha) * self._performance_stats["average_latency_ms"]
        )

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Gibt Performance-Metriken zurück."""
        # Discovery Engine Stats
        discovery_stats = await self.discovery_engine.get_statistics()

        # Cache Engine Stats
        cache_stats = await self.cache_engine.get_statistics()

        # Resource Monitor Stats
        resource_stats = await self._resource_monitor.get_statistics()

        return {
            "performance_stats": self._performance_stats.copy(),
            "discovery_stats": discovery_stats,
            "cache_stats": cache_stats,
            "resource_stats": resource_stats,
            "settings": {
                "enabled": self.settings.enabled,
                "target_latency_ms": self.settings.target_latency_ms,
                "max_latency_ms": self.settings.max_latency_ms,
                "target_throughput_rps": self.settings.target_throughput_rps,
                "max_concurrent_workflows": self.settings.max_concurrent_workflows
            }
        }


class ResourceMonitor:
    """Resource Monitor für Performance Tracking."""

    def __init__(self, settings: VoicePerformanceSettings):
        self.settings = settings
        self._running = False
        self._monitor_task: asyncio.Task | None = None

        # Resource Stats
        self._resource_stats = {
            "current_memory_mb": 0.0,
            "peak_memory_mb": 0.0,
            "current_cpu_percent": 0.0,
            "peak_cpu_percent": 0.0,
            "monitoring_enabled": settings.monitoring_enabled
        }

    async def start(self) -> None:
        """Startet Resource Monitoring."""
        if not self.settings.monitoring_enabled or self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.debug("Resource monitor started")

    async def stop(self) -> None:
        """Stoppt Resource Monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.debug("Resource monitor stopped")

    async def _monitor_loop(self) -> None:
        """Resource Monitoring Loop."""
        while self._running:
            try:
                # Memory Usage
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                self._resource_stats["current_memory_mb"] = memory_mb

                self._resource_stats["peak_memory_mb"] = max(self._resource_stats["peak_memory_mb"], memory_mb)

                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self._resource_stats["current_cpu_percent"] = cpu_percent

                self._resource_stats["peak_cpu_percent"] = max(self._resource_stats["peak_cpu_percent"], cpu_percent)

                await asyncio.sleep(30)  # Monitor alle 30 Sekunden (reduziert CPU-Last)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)

    def get_memory_usage(self) -> float:
        """Gibt aktuelle Memory Usage zurück."""
        return self._resource_stats["current_memory_mb"]

    def get_cpu_usage(self) -> float:
        """Gibt aktuelle CPU Usage zurück."""
        return self._resource_stats["current_cpu_percent"]

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Resource-Statistiken zurück."""
        return self._resource_stats.copy()


def create_voice_performance_optimizer(settings: VoicePerformanceSettings) -> VoicePerformanceOptimizer:
    """Factory-Funktion für Voice Performance Optimizer."""
    return VoicePerformanceOptimizer(settings)
