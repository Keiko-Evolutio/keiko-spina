"""Voice-Workflow Failed Task Recovery System.
Intelligente Recovery-Strategien für Voice-to-Orchestrator Failed Tasks.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger

from .dead_letter_queue import FailedTask, FailureReason
from .voice_workflow_dlq import VoiceWorkflowDLQ

logger = get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery-Strategien für Failed Tasks."""
    IMMEDIATE_RETRY = "immediate_retry"
    DELAYED_RETRY = "delayed_retry"
    ALTERNATIVE_AGENT = "alternative_agent"
    FALLBACK_PROCESSING = "fallback_processing"
    USER_INTERVENTION = "user_intervention"
    SESSION_RECOVERY = "session_recovery"
    PARTIAL_RECOVERY = "partial_recovery"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class RecoveryPriority(Enum):
    """Recovery-Prioritäten."""
    CRITICAL = "critical"  # Sofortige Recovery erforderlich
    HIGH = "high"         # Recovery innerhalb 1 Minute
    NORMAL = "normal"     # Recovery innerhalb 5 Minuten
    LOW = "low"          # Recovery innerhalb 15 Minuten
    BACKGROUND = "background"  # Recovery wenn Ressourcen verfügbar


class VoiceRecoverySystem:
    """Voice-Workflow Failed Task Recovery System.
    Intelligente Recovery mit Context-aware Strategien.
    """

    def __init__(self, dlq: VoiceWorkflowDLQ):
        self.dlq = dlq

        # Recovery Strategies
        self._recovery_strategies: dict[FailureReason, RecoveryStrategy] = {
            # Voice-spezifische Strategies
            FailureReason.VOICE_PROCESSING_TIMEOUT: RecoveryStrategy.DELAYED_RETRY,
            FailureReason.SPEECH_TO_TEXT_FAILED: RecoveryStrategy.ALTERNATIVE_AGENT,
            FailureReason.INTENT_RECOGNITION_FAILED: RecoveryStrategy.FALLBACK_PROCESSING,
            FailureReason.VOICE_ORCHESTRATOR_TIMEOUT: RecoveryStrategy.SESSION_RECOVERY,

            # Agent-spezifische Strategies
            FailureReason.AGENT_DISCOVERY_FAILED: RecoveryStrategy.ALTERNATIVE_AGENT,
            FailureReason.AGENT_EXECUTION_FAILED: RecoveryStrategy.ALTERNATIVE_AGENT,
            FailureReason.AGENT_TIMEOUT: RecoveryStrategy.DELAYED_RETRY,
            FailureReason.AGENT_CIRCUIT_BREAKER_OPEN: RecoveryStrategy.GRACEFUL_DEGRADATION,

            # Resource-spezifische Strategies
            FailureReason.RESOURCE_EXHAUSTED: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureReason.MEMORY_LIMIT_EXCEEDED: RecoveryStrategy.PARTIAL_RECOVERY,
            FailureReason.CPU_LIMIT_EXCEEDED: RecoveryStrategy.DELAYED_RETRY,
            FailureReason.RATE_LIMIT_EXCEEDED: RecoveryStrategy.DELAYED_RETRY,

            # System-spezifische Strategies
            FailureReason.NETWORK_ERROR: RecoveryStrategy.IMMEDIATE_RETRY,
            FailureReason.CACHE_ERROR: RecoveryStrategy.FALLBACK_PROCESSING,
            FailureReason.VALIDATION_ERROR: RecoveryStrategy.USER_INTERVENTION
        }

        # Recovery Handlers
        self._recovery_handlers: dict[RecoveryStrategy, Callable[[FailedTask], Awaitable[bool]]] = {}

        # Alternative Agents/Services
        self._alternative_agents: dict[str, list[str]] = {
            "speech_to_text": ["azure_stt", "google_stt", "whisper_local"],
            "intent_recognition": ["azure_luis", "google_dialogflow", "local_nlp"],
            "voice_orchestrator": ["primary_orchestrator", "fallback_orchestrator"]
        }

        # Recovery Statistics
        self._recovery_stats = {
            "total_recovery_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "strategy_success_rates": {},
            "average_recovery_time_ms": 0.0,
            "user_satisfaction_after_recovery": 0.0
        }

        # Background Recovery
        self._recovery_task: asyncio.Task | None = None
        self._running = False

        # User Notification System
        self._user_notification_enabled = True
        self._notification_handlers: list[Callable[[FailedTask, str], Awaitable[None]]] = []

        logger.info("Voice recovery system initialized")

    async def start(self) -> None:
        """Startet Recovery System."""
        if self._running:
            return

        self._running = True

        # Registriere Recovery Handlers
        await self._register_recovery_handlers()

        # Starte Background Recovery Task
        self._recovery_task = asyncio.create_task(self._recovery_loop())

        logger.info("Voice recovery system started")

    async def stop(self) -> None:
        """Stoppt Recovery System."""
        self._running = False

        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        logger.info("Voice recovery system stopped")

    async def recover_failed_task(
        self,
        task_id: str,
        strategy_override: RecoveryStrategy | None = None
    ) -> bool:
        """Führt Recovery für spezifischen Failed Task durch."""
        start_time = time.time()

        try:
            # Hole Failed Task
            if task_id not in self.dlq._failed_tasks:
                logger.error(f"Failed task not found: {task_id}")
                return False

            failed_task = self.dlq._failed_tasks[task_id]

            # Bestimme Recovery Strategy
            strategy = strategy_override or self._determine_recovery_strategy(failed_task)

            # Prüfe Recovery-Berechtigung
            if not await self._can_attempt_recovery(failed_task):
                logger.warning(f"Recovery not allowed for task: {task_id}")
                return False

            # Führe Recovery durch
            success = await self._execute_recovery_strategy(failed_task, strategy)

            # Update Statistics
            recovery_time_ms = (time.time() - start_time) * 1000
            await self._update_recovery_stats(strategy, success, recovery_time_ms)

            # User Notification
            if self._user_notification_enabled:
                await self._notify_user_recovery_result(failed_task, success)

            if success:
                logger.info(f"Task recovery successful: {task_id} using {strategy.value} in {recovery_time_ms:.1f}ms")
            else:
                logger.warning(f"Task recovery failed: {task_id} using {strategy.value}")

            return success

        except Exception as e:
            logger.error(f"Recovery error for task {task_id}: {e}")
            return False

    async def recover_voice_workflow(
        self,
        workflow_id: str,
        recovery_mode: str = "intelligent"
    ) -> dict[str, Any]:
        """Führt komplette Voice Workflow Recovery durch."""
        start_time = time.time()

        try:
            # Finde alle Failed Tasks für Workflow
            workflow_tasks = [
                task for task in self.dlq._failed_tasks.values()
                if task.workflow_id == workflow_id
            ]

            if not workflow_tasks:
                return {"success": False, "error": "No failed tasks found for workflow"}

            # Sortiere Tasks nach Processing Stage
            workflow_tasks.sort(key=lambda t: t.created_at)

            # Recovery basierend auf Mode
            if recovery_mode == "intelligent":
                result = await self._intelligent_workflow_recovery(workflow_tasks)
            elif recovery_mode == "sequential":
                result = await self._sequential_workflow_recovery(workflow_tasks)
            elif recovery_mode == "parallel":
                result = await self._parallel_workflow_recovery(workflow_tasks)
            elif recovery_mode == "critical_only":
                result = await self._critical_only_recovery(workflow_tasks)
            else:
                return {"success": False, "error": f"Unknown recovery mode: {recovery_mode}"}

            # Recovery Time
            total_recovery_time_ms = (time.time() - start_time) * 1000
            result["total_recovery_time_ms"] = total_recovery_time_ms

            logger.info(f"Workflow recovery completed: {workflow_id} in {total_recovery_time_ms:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"Workflow recovery error: {workflow_id}: {e}")
            return {"success": False, "error": str(e)}

    async def implement_session_continuity(
        self,
        session_id: str,
        continuity_strategy: str = "seamless"
    ) -> bool:
        """Implementiert Session Continuity nach Failed Tasks."""
        try:
            # Finde alle Failed Tasks für Session
            session_tasks = [
                task for task in self.dlq._failed_tasks.values()
                if task.session_id == session_id
            ]

            if not session_tasks:
                return True  # Keine Failed Tasks = Continuity bereits gewährleistet

            # Session Continuity Strategy
            if continuity_strategy == "seamless":
                success = await self._seamless_session_continuity(session_tasks)
            elif continuity_strategy == "transparent":
                success = await self._transparent_session_continuity(session_tasks)
            elif continuity_strategy == "informed":
                success = await self._informed_session_continuity(session_tasks)
            else:
                logger.error(f"Unknown continuity strategy: {continuity_strategy}")
                return False

            if success:
                self._recovery_stats["session_continuity_maintained"] = (
                    self._recovery_stats.get("session_continuity_maintained", 0) + 1
                )

            return success

        except Exception as e:
            logger.error(f"Session continuity error: {session_id}: {e}")
            return False

    async def get_recovery_recommendations(
        self,
        task_id: str
    ) -> list[dict[str, Any]]:
        """Gibt Recovery-Empfehlungen für Failed Task zurück."""
        if task_id not in self.dlq._failed_tasks:
            return []

        failed_task = self.dlq._failed_tasks[task_id]
        recommendations = []

        # Primary Strategy
        primary_strategy = self._determine_recovery_strategy(failed_task)
        recommendations.append({
            "strategy": primary_strategy.value,
            "priority": "primary",
            "success_probability": self._get_strategy_success_rate(primary_strategy),
            "estimated_time_ms": self._estimate_recovery_time(failed_task, primary_strategy),
            "description": self._get_strategy_description(primary_strategy)
        })

        # Alternative Strategies
        alternative_strategies = self._get_alternative_strategies(failed_task)
        for strategy in alternative_strategies:
            recommendations.append({
                "strategy": strategy.value,
                "priority": "alternative",
                "success_probability": self._get_strategy_success_rate(strategy),
                "estimated_time_ms": self._estimate_recovery_time(failed_task, strategy),
                "description": self._get_strategy_description(strategy)
            })

        return recommendations

    def register_notification_handler(
        self,
        handler: Callable[[FailedTask, str], Awaitable[None]]
    ) -> None:
        """Registriert User Notification Handler."""
        self._notification_handlers.append(handler)
        logger.info("User notification handler registered")

    async def get_recovery_statistics(self) -> dict[str, Any]:
        """Gibt Recovery-Statistiken zurück."""
        return {
            "recovery_stats": self._recovery_stats.copy(),
            "strategy_performance": {
                strategy.value: {
                    "success_rate": self._get_strategy_success_rate(strategy),
                    "average_time_ms": self._get_strategy_average_time(strategy),
                    "usage_count": self._get_strategy_usage_count(strategy)
                }
                for strategy in RecoveryStrategy
            },
            "alternative_agents": self._alternative_agents.copy()
        }

    # Private Methods

    async def _recovery_loop(self) -> None:
        """Background Recovery Loop."""
        while self._running:
            try:
                await self._process_recovery_queues()
                await asyncio.sleep(10)  # Alle 10 Sekunden
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(10)

    async def _process_recovery_queues(self) -> None:
        """Verarbeitet Recovery Queues nach Priorität."""
        # Critical Queue zuerst
        await self._process_priority_queue(self.dlq._critical_voice_queue, RecoveryPriority.CRITICAL)

        # Voice Processing Queue
        await self._process_priority_queue(self.dlq._voice_processing_queue, RecoveryPriority.HIGH)

        # Agent Execution Queue
        await self._process_priority_queue(self.dlq._agent_execution_queue, RecoveryPriority.NORMAL)

        # Standard Retry Queue
        await self._process_priority_queue(self.dlq._retry_queue, RecoveryPriority.LOW)

    async def _process_priority_queue(
        self,
        queue: any,
        priority: RecoveryPriority
    ) -> None:
        """Verarbeitet spezifische Priority Queue."""
        processed_count = 0
        max_batch_size = 5 if priority == RecoveryPriority.CRITICAL else 3

        while queue and processed_count < max_batch_size:
            try:
                task_id = queue.popleft()

                if task_id in self.dlq._failed_tasks:
                    failed_task = self.dlq._failed_tasks[task_id]

                    # Prüfe ob Ready für Recovery
                    if await self._is_ready_for_recovery(failed_task):
                        success = await self.recover_failed_task(task_id)
                        if not success and failed_task.retry_count < failed_task.max_retries:
                            # Zurück in Queue für späteren Retry
                            queue.append(task_id)

                processed_count += 1

            except Exception as e:
                logger.error(f"Priority queue processing error: {e}")
                break

    def _determine_recovery_strategy(self, failed_task: FailedTask) -> RecoveryStrategy:
        """Bestimmt optimale Recovery Strategy."""
        # Basis-Strategy basierend auf Failure Reason
        base_strategy = self._recovery_strategies.get(
            failed_task.failure_reason,
            RecoveryStrategy.DELAYED_RETRY
        )

        # Anpassung basierend auf Kontext
        if failed_task.criticality == "critical":
            if base_strategy == RecoveryStrategy.DELAYED_RETRY:
                return RecoveryStrategy.IMMEDIATE_RETRY

        if failed_task.retry_count >= 2:
            if base_strategy in [RecoveryStrategy.IMMEDIATE_RETRY, RecoveryStrategy.DELAYED_RETRY]:
                return RecoveryStrategy.ALTERNATIVE_AGENT

        return base_strategy

    async def _can_attempt_recovery(self, failed_task: FailedTask) -> bool:
        """Prüft ob Recovery versucht werden kann."""
        # Retry Limit Check
        if failed_task.retry_count >= failed_task.max_retries:
            return False

        # Time-based Check
        if failed_task.next_retry_at and datetime.utcnow() < failed_task.next_retry_at:
            return False

        # Resource Check
        if failed_task.failure_category == "resource":
            # Prüfe aktuelle Resource-Verfügbarkeit
            return await self._check_resource_availability()

        return True

    async def _is_ready_for_recovery(self, failed_task: FailedTask) -> bool:
        """Prüft ob Task ready für Recovery ist."""
        if not await self._can_attempt_recovery(failed_task):
            return False

        # Circuit Breaker Check
        if failed_task.failure_reason == FailureReason.AGENT_CIRCUIT_BREAKER_OPEN:
            return await self._check_circuit_breaker_status(failed_task)

        return True

    async def _register_recovery_handlers(self) -> None:
        """Registriert Recovery Strategy Handlers."""
        self._recovery_handlers = {
            RecoveryStrategy.IMMEDIATE_RETRY: self._immediate_retry_handler,
            RecoveryStrategy.DELAYED_RETRY: self._delayed_retry_handler,
            RecoveryStrategy.ALTERNATIVE_AGENT: self._alternative_agent_handler,
            RecoveryStrategy.FALLBACK_PROCESSING: self._fallback_processing_handler,
            RecoveryStrategy.USER_INTERVENTION: self._user_intervention_handler,
            RecoveryStrategy.SESSION_RECOVERY: self._session_recovery_handler,
            RecoveryStrategy.PARTIAL_RECOVERY: self._partial_recovery_handler,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation_handler
        }

    async def _execute_recovery_strategy(
        self,
        failed_task: FailedTask,
        strategy: RecoveryStrategy
    ) -> bool:
        """Führt spezifische Recovery Strategy aus."""
        handler = self._recovery_handlers.get(strategy)
        if not handler:
            logger.error(f"No handler for recovery strategy: {strategy.value}")
            return False

        try:
            return await handler(failed_task)
        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {strategy.value}: {e}")
            return False

    # Recovery Strategy Handlers

    async def _immediate_retry_handler(self, failed_task: FailedTask) -> bool:
        """Immediate Retry Handler."""
        # Führe sofortigen Retry durch
        return await self.dlq.retry_task(failed_task.task_id)

    async def _delayed_retry_handler(self, failed_task: FailedTask) -> bool:
        """Delayed Retry Handler."""
        # Warte kurz und führe dann Retry durch
        await asyncio.sleep(1.0)
        return await self.dlq.retry_task(failed_task.task_id)

    async def _alternative_agent_handler(self, failed_task: FailedTask) -> bool:
        """Alternative Agent Handler."""
        # Versuche mit alternativem Agent
        task_type = failed_task.task_type
        alternatives = self._alternative_agents.get(task_type, [])

        for alternative in alternatives:
            try:
                # Mock: Versuche mit alternativem Agent
                await asyncio.sleep(0.1)
                logger.info(f"Trying alternative agent: {alternative} for task {failed_task.task_id}")
                return True  # Mock success
            except Exception as e:
                logger.warning(f"Alternative agent {alternative} failed: {e}")
                continue

        return False

    async def _fallback_processing_handler(self, failed_task: FailedTask) -> bool:
        """Fallback Processing Handler."""
        # Implementiere Fallback-Verarbeitung
        try:
            # Mock Fallback Processing
            await asyncio.sleep(0.2)
            logger.info(f"Fallback processing for task {failed_task.task_id}")
            return True
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return False

    async def _user_intervention_handler(self, failed_task: FailedTask) -> bool:
        """User Intervention Handler."""
        # Benachrichtige User und warte auf Intervention
        await self._notify_user_intervention_required(failed_task)
        return False  # Requires manual intervention

    async def _session_recovery_handler(self, failed_task: FailedTask) -> bool:
        """Session Recovery Handler."""
        # Implementiere Session Recovery
        return await self.implement_session_continuity(failed_task.session_id, "seamless")

    async def _partial_recovery_handler(self, failed_task: FailedTask) -> bool:
        """Partial Recovery Handler."""
        # Implementiere partielle Recovery
        if failed_task.voice_context and failed_task.voice_context.partial_results:
            # Verwende partielle Ergebnisse
            logger.info(f"Partial recovery using existing results for task {failed_task.task_id}")
            return True
        return False

    async def _graceful_degradation_handler(self, failed_task: FailedTask) -> bool:
        """Graceful Degradation Handler."""
        # Implementiere Graceful Degradation
        try:
            # Reduziere Funktionalität aber halte Service am Laufen
            await asyncio.sleep(0.1)
            logger.info(f"Graceful degradation for task {failed_task.task_id}")
            return True
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False

    # Helper Methods

    async def _intelligent_workflow_recovery(self, workflow_tasks: list[FailedTask]) -> dict[str, Any]:
        """Intelligente Workflow Recovery."""
        # Analysiere Dependencies und führe optimale Recovery durch
        recovery_plan = self._create_recovery_plan(workflow_tasks)

        successful_recoveries = 0
        for task in recovery_plan:
            success = await self.recover_failed_task(task.task_id)
            if success:
                successful_recoveries += 1

        return {
            "success": successful_recoveries > 0,
            "total_tasks": len(workflow_tasks),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / len(workflow_tasks)
        }

    async def _sequential_workflow_recovery(self, workflow_tasks: list[FailedTask]) -> dict[str, Any]:
        """Sequenzielle Workflow Recovery."""
        successful_recoveries = 0

        for task in workflow_tasks:
            success = await self.recover_failed_task(task.task_id)
            if success:
                successful_recoveries += 1
            else:
                break  # Stoppe bei erstem Fehler

        return {
            "success": successful_recoveries == len(workflow_tasks),
            "total_tasks": len(workflow_tasks),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / len(workflow_tasks)
        }

    async def _parallel_workflow_recovery(self, workflow_tasks: list[FailedTask]) -> dict[str, Any]:
        """Parallele Workflow Recovery."""
        tasks = [
            asyncio.create_task(self.recover_failed_task(task.task_id))
            for task in workflow_tasks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_recoveries = sum(1 for result in results if result is True)

        return {
            "success": successful_recoveries > 0,
            "total_tasks": len(workflow_tasks),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / len(workflow_tasks)
        }

    async def _critical_only_recovery(self, workflow_tasks: list[FailedTask]) -> dict[str, Any]:
        """Recovery nur für kritische Tasks."""
        critical_tasks = [task for task in workflow_tasks if task.criticality in ["high", "critical"]]

        if not critical_tasks:
            return {"success": True, "total_tasks": 0, "successful_recoveries": 0, "recovery_rate": 1.0}

        successful_recoveries = 0
        for task in critical_tasks:
            success = await self.recover_failed_task(task.task_id)
            if success:
                successful_recoveries += 1

        return {
            "success": successful_recoveries > 0,
            "total_tasks": len(critical_tasks),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / len(critical_tasks)
        }

    def _create_recovery_plan(self, workflow_tasks: list[FailedTask]) -> list[FailedTask]:
        """Erstellt optimalen Recovery-Plan."""
        # Sortiere nach Criticality und Dependencies
        plan = sorted(workflow_tasks, key=lambda t: (
            {"critical": 0, "high": 1, "normal": 2, "low": 3}.get(t.criticality, 2),
            t.created_at
        ))
        return plan

    async def _seamless_session_continuity(self, session_tasks: list[FailedTask]) -> bool:
        """Seamless Session Continuity - User merkt nichts."""
        # Implementiere transparente Recovery für alle Session Tasks
        logger.debug(f"Seamless session continuity for {len(session_tasks)} tasks")

        # Führe Recovery für alle Tasks durch ohne User-Benachrichtigung
        recovery_success = True
        for task in session_tasks:
            try:
                success = await self.recover_failed_task(task.task_id)
                if not success:
                    recovery_success = False
                    logger.warning(f"Seamless recovery failed for task {task.task_id}")
            except Exception as e:
                logger.error(f"Error in seamless recovery for task {task.task_id}: {e}")
                recovery_success = False

        return recovery_success

    async def _transparent_session_continuity(self, session_tasks: list[FailedTask]) -> bool:
        """Transparent Session Continuity - Minimale User-Störung."""
        logger.debug(f"Transparent session continuity for {len(session_tasks)} tasks")

        # Führe Recovery durch mit minimaler User-Benachrichtigung
        recovery_success = True
        for task in session_tasks:
            try:
                # Kurze Benachrichtigung über Recovery-Versuch
                if self._user_notification_enabled:
                    await self._notify_user_recovery_attempt(task)

                success = await self.recover_failed_task(task.task_id)
                if not success:
                    recovery_success = False
                    logger.warning(f"Transparent recovery failed for task {task.task_id}")
            except Exception as e:
                logger.error(f"Error in transparent recovery for task {task.task_id}: {e}")
                recovery_success = False

        return recovery_success

    async def _informed_session_continuity(self, session_tasks: list[FailedTask]) -> bool:
        """Informed Session Continuity - User wird informiert."""
        # Informiere User über Recovery
        for task in session_tasks:
            await self._notify_user_recovery_attempt(task)
        return True

    async def _check_resource_availability(self) -> bool:
        """Prüft Resource-Verfügbarkeit."""
        # Mock implementation
        return True

    async def _check_circuit_breaker_status(self, failed_task: FailedTask) -> bool:
        """Prüft Circuit Breaker Status."""
        # Prüfe Circuit Breaker Status für den spezifischen Service/Agent
        service_name = failed_task.voice_context.agent_name if failed_task.voice_context else "unknown"
        logger.debug(f"Checking circuit breaker status for service: {service_name}")

        # In einer echten Implementierung würde hier der Circuit Breaker Status geprüft
        # Für jetzt nehmen wir an, dass der Circuit Breaker geschlossen ist
        return True

    async def _update_recovery_stats(
        self,
        strategy: RecoveryStrategy,
        success: bool,
        recovery_time_ms: float
    ) -> None:
        """Aktualisiert Recovery-Statistiken."""
        self._recovery_stats["total_recovery_attempts"] += 1

        if success:
            self._recovery_stats["successful_recoveries"] += 1
        else:
            self._recovery_stats["failed_recoveries"] += 1

        # Strategy-spezifische Stats
        strategy_key = strategy.value
        if strategy_key not in self._recovery_stats["strategy_success_rates"]:
            self._recovery_stats["strategy_success_rates"][strategy_key] = {"attempts": 0, "successes": 0}

        self._recovery_stats["strategy_success_rates"][strategy_key]["attempts"] += 1
        if success:
            self._recovery_stats["strategy_success_rates"][strategy_key]["successes"] += 1

        # Average Recovery Time
        alpha = 0.1
        current_avg = self._recovery_stats["average_recovery_time_ms"]
        self._recovery_stats["average_recovery_time_ms"] = (
            alpha * recovery_time_ms + (1 - alpha) * current_avg
        )

    def _get_strategy_success_rate(self, strategy: RecoveryStrategy) -> float:
        """Gibt Success Rate für Strategy zurück."""
        strategy_stats = self._recovery_stats["strategy_success_rates"].get(strategy.value, {})
        attempts = strategy_stats.get("attempts", 0)
        successes = strategy_stats.get("successes", 0)
        return successes / attempts if attempts > 0 else 0.0

    def _get_strategy_average_time(self, strategy: RecoveryStrategy) -> float:
        """Gibt durchschnittliche Recovery-Zeit für Strategy zurück."""
        # Berechne durchschnittliche Zeit basierend auf Strategy-spezifischen Daten
        strategy_stats = self._recovery_stats["strategy_success_rates"].get(strategy.value, {})
        total_time = strategy_stats.get("total_time_ms", 0.0)
        attempts = strategy_stats.get("attempts", 0)

        if attempts > 0:
            return total_time / attempts

        # Fallback auf geschätzte Zeiten wenn keine Daten vorhanden
        base_times = {
            RecoveryStrategy.IMMEDIATE_RETRY: 100.0,
            RecoveryStrategy.DELAYED_RETRY: 2000.0,
            RecoveryStrategy.ALTERNATIVE_AGENT: 500.0,
            RecoveryStrategy.FALLBACK_PROCESSING: 300.0,
            RecoveryStrategy.USER_INTERVENTION: 30000.0,
            RecoveryStrategy.SESSION_RECOVERY: 1000.0,
            RecoveryStrategy.PARTIAL_RECOVERY: 200.0,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 150.0
        }
        return base_times.get(strategy, 150.0)

    def _get_strategy_usage_count(self, strategy: RecoveryStrategy) -> int:
        """Gibt Usage Count für Strategy zurück."""
        strategy_stats = self._recovery_stats["strategy_success_rates"].get(strategy.value, {})
        return strategy_stats.get("attempts", 0)

    def _estimate_recovery_time(self, failed_task: FailedTask, strategy: RecoveryStrategy) -> float:
        """Schätzt Recovery-Zeit für Strategy basierend auf Task-Kontext."""
        base_times = {
            RecoveryStrategy.IMMEDIATE_RETRY: 100.0,
            RecoveryStrategy.DELAYED_RETRY: 2000.0,
            RecoveryStrategy.ALTERNATIVE_AGENT: 500.0,
            RecoveryStrategy.FALLBACK_PROCESSING: 300.0,
            RecoveryStrategy.USER_INTERVENTION: 30000.0,
            RecoveryStrategy.SESSION_RECOVERY: 1000.0,
            RecoveryStrategy.PARTIAL_RECOVERY: 200.0,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 150.0
        }

        base_time = base_times.get(strategy, 1000.0)

        # Anpassung basierend auf Task-Eigenschaften
        multiplier = 1.0

        # Criticality-Anpassung
        if failed_task.criticality == "critical":
            multiplier *= 0.8  # Schnellere Recovery für kritische Tasks
        elif failed_task.criticality == "low":
            multiplier *= 1.5  # Langsamere Recovery für niedrige Priorität

        # Retry-Count Anpassung
        if failed_task.retry_count > 2:
            multiplier *= 1.3  # Mehr Zeit für oft fehlgeschlagene Tasks

        # Complexity-Anpassung basierend auf Voice Context
        if failed_task.voice_context and hasattr(failed_task.voice_context, "complexity"):
            if failed_task.voice_context.complexity == "high":
                multiplier *= 1.4

        return base_time * multiplier

    def _get_strategy_description(self, strategy: RecoveryStrategy) -> str:
        """Gibt Beschreibung für Strategy zurück."""
        descriptions = {
            RecoveryStrategy.IMMEDIATE_RETRY: "Sofortiger Retry ohne Delay",
            RecoveryStrategy.DELAYED_RETRY: "Retry mit exponential backoff",
            RecoveryStrategy.ALTERNATIVE_AGENT: "Verwendung alternativer Agents/Services",
            RecoveryStrategy.FALLBACK_PROCESSING: "Fallback zu einfacherer Verarbeitung",
            RecoveryStrategy.USER_INTERVENTION: "Manuelle User-Intervention erforderlich",
            RecoveryStrategy.SESSION_RECOVERY: "Session-weite Recovery-Maßnahmen",
            RecoveryStrategy.PARTIAL_RECOVERY: "Verwendung partieller Ergebnisse",
            RecoveryStrategy.GRACEFUL_DEGRADATION: "Reduzierte Funktionalität"
        }
        return descriptions.get(strategy, "Unbekannte Recovery-Strategie")

    def _get_alternative_strategies(self, failed_task: FailedTask) -> list[RecoveryStrategy]:
        """Gibt alternative Strategies für Failed Task zurück."""
        primary = self._determine_recovery_strategy(failed_task)

        alternatives = []
        for strategy in RecoveryStrategy:
            if strategy != primary:
                alternatives.append(strategy)

        # Sortiere nach Erfolgswahrscheinlichkeit
        alternatives.sort(key=lambda s: self._get_strategy_success_rate(s), reverse=True)

        return alternatives[:3]  # Top 3 Alternativen

    async def _notify_user_recovery_result(self, failed_task: FailedTask, success: bool) -> None:
        """Benachrichtigt User über Recovery-Ergebnis."""
        message = "Recovery successful" if success else "Recovery failed"

        for handler in self._notification_handlers:
            try:
                await handler(failed_task, message)
            except Exception as e:
                logger.error(f"User notification failed: {e}")

    async def _notify_user_intervention_required(self, failed_task: FailedTask) -> None:
        """Benachrichtigt User über erforderliche Intervention."""
        message = "Manual intervention required"

        for handler in self._notification_handlers:
            try:
                await handler(failed_task, message)
            except Exception as e:
                logger.error(f"User intervention notification failed: {e}")

    async def _notify_user_recovery_attempt(self, failed_task: FailedTask) -> None:
        """Benachrichtigt User über Recovery-Versuch."""
        message = "Attempting task recovery"

        for handler in self._notification_handlers:
            try:
                await handler(failed_task, message)
            except Exception as e:
                logger.error(f"Recovery attempt notification failed: {e}")


def create_voice_recovery_system(dlq: VoiceWorkflowDLQ) -> VoiceRecoverySystem:
    """Factory-Funktion für Voice Recovery System."""
    return VoiceRecoverySystem(dlq)
