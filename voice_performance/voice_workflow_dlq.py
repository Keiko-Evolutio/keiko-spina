"""Voice-Workflow-spezifische Dead Letter Queue Implementation.
Erweiterte DLQ für Voice-to-Orchestrator Failed Task Management.
"""

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .dead_letter_queue import DeadLetterQueue, FailedTask, FailureReason, RetryPolicy, TaskStatus
from .interfaces import VoiceWorkflowContext

logger = get_logger(__name__)


class VoiceWorkflowDLQ(DeadLetterQueue):
    """Voice-Workflow-spezifische Dead Letter Queue.
    Erweitert die Base DLQ um Voice-spezifische Features.
    """

    def __init__(self, max_queue_size: int = 50000):  # Größer für M4 Max
        super().__init__(max_queue_size)

        # Voice-spezifische Queues
        self._voice_processing_queue: deque = deque(maxlen=max_queue_size // 4)
        self._agent_execution_queue: deque = deque(maxlen=max_queue_size // 4)
        self._critical_voice_queue: deque = deque(maxlen=1000)  # Hohe Priorität

        # Voice-spezifische Retry Policies
        self._voice_retry_policies = {
            "voice_processing": RetryPolicy(
                max_retries=2,
                initial_delay_seconds=1.0,
                voice_processing_retries=2,
                circuit_breaker_aware=True
            ),
            "speech_to_text": RetryPolicy(
                max_retries=3,
                initial_delay_seconds=0.5,
                voice_processing_retries=3,
                timeout_retry_multiplier=1.2
            ),
            "intent_recognition": RetryPolicy(
                max_retries=2,
                initial_delay_seconds=1.0,
                voice_processing_retries=2
            ),
            "voice_orchestrator": RetryPolicy(
                max_retries=3,
                initial_delay_seconds=2.0,
                agent_execution_retries=3,
                circuit_breaker_aware=True
            ),
            "agent_discovery": RetryPolicy(
                max_retries=5,
                initial_delay_seconds=1.0,
                agent_execution_retries=5,
                resource_retry_multiplier=1.5
            ),
            "agent_execution": RetryPolicy(
                max_retries=3,
                initial_delay_seconds=3.0,
                agent_execution_retries=3,
                circuit_breaker_aware=True,
                timeout_retry_multiplier=2.0
            )
        }

        # Voice-spezifische Statistics
        self._voice_stats = {
            "voice_processing_failures": 0,
            "agent_execution_failures": 0,
            "critical_voice_failures": 0,
            "voice_recovery_success_rate": 0.0,
            "average_voice_recovery_time_ms": 0.0,
            "user_notification_sent": 0,
            "session_continuity_maintained": 0
        }

        # User Notification Handlers
        self._user_notification_handlers: list[Callable[[FailedTask], Awaitable[None]]] = []

        # Session Continuity Handlers
        self._session_continuity_handlers: dict[str, Callable[[FailedTask], Awaitable[bool]]] = {}

        # Voice Context Recovery
        self._voice_context_cache: dict[str, VoiceWorkflowContext] = {}

        logger.info("Voice workflow DLQ initialized with enhanced features")

    async def add_voice_failed_task(
        self,
        task_id: str,
        workflow_id: str,
        task_type: str,
        failure_reason: FailureReason,
        error_message: str,
        voice_context: VoiceWorkflowContext,
        original_request: dict[str, Any],
        user_id: str,
        session_id: str,
        priority: int = 0,
        criticality: str = "normal",
        stack_trace: str | None = None
    ) -> None:
        """Fügt einen Voice-spezifischen Failed Task hinzu."""
        # Bestimme Failure Category
        failure_category = self._categorize_failure(failure_reason)

        # Erstelle Enhanced Failed Task
        failed_task = FailedTask(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type=task_type,
            failure_reason=failure_reason,
            error_message=error_message,
            stack_trace=stack_trace,
            failure_category=failure_category,
            original_request=original_request,
            voice_context=voice_context,
            user_id=user_id,
            session_id=session_id,
            priority=priority,
            criticality=criticality
        )

        # Voice-spezifische Retry Policy
        retry_policy = self._voice_retry_policies.get(task_type, self._retry_policies["default"])
        failed_task.max_retries = retry_policy.max_retries

        # Berechne Retry-Delay mit Voice-spezifischen Faktoren
        delay = self._calculate_voice_retry_delay(retry_policy, failed_task)
        failed_task.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

        # Speichere Voice Context
        self._voice_context_cache[workflow_id] = voice_context

        # Füge zu entsprechender Queue hinzu
        await self._route_to_voice_queue(failed_task)

        # Update Voice-spezifische Statistics
        self._update_voice_stats(failure_category)

        # User Notification für kritische Failures
        if criticality in ["high", "critical"]:
            await self._send_user_notification(failed_task)

        logger.warning(f"Voice failed task added to DLQ: {task_id} ({failure_reason.value}, category: {failure_category})")

    async def recover_voice_workflow(
        self,
        workflow_id: str,
        recovery_strategy: str = "full_recovery"
    ) -> bool:
        """Führt Voice Workflow Recovery durch."""
        start_time = time.time()

        try:
            # Finde alle Failed Tasks für Workflow
            workflow_tasks = [
                task for task in self._failed_tasks.values()
                if task.workflow_id == workflow_id
            ]

            if not workflow_tasks:
                logger.warning(f"No failed tasks found for workflow: {workflow_id}")
                return False

            # Sortiere nach Processing Stage
            workflow_tasks.sort(key=lambda t: t.created_at)

            # Recovery basierend auf Strategy
            if recovery_strategy == "full_recovery":
                success = await self._full_voice_recovery(workflow_tasks)
            elif recovery_strategy == "partial_recovery":
                success = await self._partial_voice_recovery(workflow_tasks)
            elif recovery_strategy == "session_continuity":
                success = await self._maintain_session_continuity(workflow_tasks)
            else:
                logger.error(f"Unknown recovery strategy: {recovery_strategy}")
                return False

            # Update Recovery Statistics
            recovery_time_ms = (time.time() - start_time) * 1000
            self._update_recovery_stats(success, recovery_time_ms)

            if success:
                logger.info(f"Voice workflow recovery successful: {workflow_id} in {recovery_time_ms:.1f}ms")
            else:
                logger.warning(f"Voice workflow recovery failed: {workflow_id}")

            return success

        except Exception as e:
            logger.error(f"Voice workflow recovery error: {workflow_id}: {e}")
            return False

    async def get_voice_failure_analytics(self) -> dict[str, Any]:
        """Gibt Voice-spezifische Failure Analytics zurück."""
        # Failure Pattern Analysis
        failure_patterns = defaultdict(int)
        recovery_patterns = defaultdict(list)

        for task in self._failed_tasks.values():
            if task.voice_context:
                # Failure Patterns
                pattern_key = f"{task.failure_category}:{task.failure_reason.value}"
                failure_patterns[pattern_key] += 1

                # Recovery Patterns
                if task.status == TaskStatus.RECOVERED:
                    recovery_patterns[task.failure_reason.value].append(task.retry_count)

        # Success Rate Analysis
        total_voice_failures = sum(1 for t in self._failed_tasks.values() if t.voice_context)
        recovered_voice_tasks = sum(1 for t in self._failed_tasks.values()
                                  if t.voice_context and t.status == TaskStatus.RECOVERED)

        voice_recovery_rate = recovered_voice_tasks / total_voice_failures if total_voice_failures > 0 else 0.0

        # Average Recovery Time
        recovery_times = [
            sum(t.retry_latencies_ms) for t in self._failed_tasks.values()
            if t.voice_context and t.status == TaskStatus.RECOVERED and t.retry_latencies_ms
        ]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0.0

        return {
            "voice_failure_analytics": {
                "total_voice_failures": total_voice_failures,
                "recovered_voice_tasks": recovered_voice_tasks,
                "voice_recovery_rate": voice_recovery_rate,
                "average_recovery_time_ms": avg_recovery_time,
                "failure_patterns": dict(failure_patterns),
                "recovery_patterns": {
                    reason: {
                        "average_retries": sum(retries) / len(retries) if retries else 0,
                        "max_retries": max(retries) if retries else 0,
                        "success_count": len(retries)
                    }
                    for reason, retries in recovery_patterns.items()
                }
            },
            "voice_stats": self._voice_stats.copy(),
            "queue_distribution": {
                "voice_processing_queue": len(self._voice_processing_queue),
                "agent_execution_queue": len(self._agent_execution_queue),
                "critical_voice_queue": len(self._critical_voice_queue)
            }
        }

    def register_user_notification_handler(
        self,
        handler: Callable[[FailedTask], Awaitable[None]]
    ) -> None:
        """Registriert User Notification Handler."""
        self._user_notification_handlers.append(handler)
        logger.info("User notification handler registered")

    def register_session_continuity_handler(
        self,
        session_type: str,
        handler: Callable[[FailedTask], Awaitable[bool]]
    ) -> None:
        """Registriert Session Continuity Handler."""
        self._session_continuity_handlers[session_type] = handler
        logger.info(f"Session continuity handler registered for: {session_type}")

    # Private Methods

    def _categorize_failure(self, failure_reason: FailureReason) -> str:
        """Kategorisiert Failure-Grund."""
        voice_failures = {
            FailureReason.VOICE_PROCESSING_TIMEOUT,
            FailureReason.SPEECH_TO_TEXT_FAILED,
            FailureReason.INTENT_RECOGNITION_FAILED,
            FailureReason.VOICE_ORCHESTRATOR_TIMEOUT
        }

        agent_failures = {
            FailureReason.AGENT_DISCOVERY_FAILED,
            FailureReason.AGENT_EXECUTION_FAILED,
            FailureReason.AGENT_TIMEOUT,
            FailureReason.AGENT_CIRCUIT_BREAKER_OPEN
        }

        resource_failures = {
            FailureReason.RESOURCE_EXHAUSTED,
            FailureReason.MEMORY_LIMIT_EXCEEDED,
            FailureReason.CPU_LIMIT_EXCEEDED,
            FailureReason.RATE_LIMIT_EXCEEDED
        }

        if failure_reason in voice_failures:
            return "voice"
        if failure_reason in agent_failures:
            return "agent"
        if failure_reason in resource_failures:
            return "resource"
        return "system"

    def _calculate_voice_retry_delay(
        self,
        policy: RetryPolicy,
        failed_task: FailedTask
    ) -> float:
        """Berechnet Voice-spezifischen Retry-Delay."""
        base_delay = self._calculate_retry_delay(policy, failed_task.retry_count)

        # Priority-basierte Anpassung
        if failed_task.priority > 5:  # Hohe Priorität
            base_delay *= policy.high_priority_multiplier
        elif failed_task.priority < 3:  # Niedrige Priorität
            base_delay *= policy.low_priority_multiplier

        # Criticality-basierte Anpassung
        if failed_task.criticality == "critical":
            base_delay *= 0.3  # Sehr schnelle Retries
        elif failed_task.criticality == "high":
            base_delay *= 0.6
        elif failed_task.criticality == "low":
            base_delay *= 1.5

        # Failure-spezifische Anpassung
        if failed_task.failure_reason in [FailureReason.TIMEOUT, FailureReason.VOICE_PROCESSING_TIMEOUT]:
            base_delay *= policy.timeout_retry_multiplier
        elif failed_task.failure_category == "resource":
            base_delay *= policy.resource_retry_multiplier
        elif failed_task.failure_reason == FailureReason.NETWORK_ERROR:
            base_delay *= policy.network_retry_multiplier

        return min(base_delay, policy.max_delay_seconds)

    async def _route_to_voice_queue(self, failed_task: FailedTask) -> None:
        """Routet Failed Task zu entsprechender Voice Queue."""
        if failed_task.criticality in ["high", "critical"]:
            self._critical_voice_queue.append(failed_task.task_id)
        elif failed_task.failure_category == "voice":
            self._voice_processing_queue.append(failed_task.task_id)
        elif failed_task.failure_category == "agent":
            self._agent_execution_queue.append(failed_task.task_id)
        else:
            # Fallback zu Standard-Queue
            self._retry_queue.append(failed_task.task_id)

        # Speichere in Haupt-Dictionary
        self._failed_tasks[failed_task.task_id] = failed_task

    def _update_voice_stats(self, failure_category: str) -> None:
        """Aktualisiert Voice-spezifische Statistiken."""
        if failure_category == "voice":
            self._voice_stats["voice_processing_failures"] += 1
        elif failure_category == "agent":
            self._voice_stats["agent_execution_failures"] += 1

        # Update Gesamt-Stats
        self._stats["total_failures"] += 1

    def _update_recovery_stats(self, success: bool, recovery_time_ms: float) -> None:
        """Aktualisiert Recovery-Statistiken."""
        if success:
            # Exponential Moving Average für Recovery Time
            alpha = 0.1
            current_avg = self._voice_stats["average_voice_recovery_time_ms"]
            self._voice_stats["average_voice_recovery_time_ms"] = (
                alpha * recovery_time_ms + (1 - alpha) * current_avg
            )

            # Update Success Rate
            total_recoveries = self._voice_stats.get("total_recovery_attempts", 0) + 1
            successful_recoveries = self._voice_stats.get("successful_recoveries", 0) + 1
            self._voice_stats["voice_recovery_success_rate"] = successful_recoveries / total_recoveries
            self._voice_stats["successful_recoveries"] = successful_recoveries

        self._voice_stats["total_recovery_attempts"] = self._voice_stats.get("total_recovery_attempts", 0) + 1

    async def _send_user_notification(self, failed_task: FailedTask) -> None:
        """Sendet User Notification für kritische Failures."""
        try:
            for handler in self._user_notification_handlers:
                await handler(failed_task)

            self._voice_stats["user_notification_sent"] += 1

        except Exception as e:
            logger.error(f"User notification failed: {e}")

    async def _full_voice_recovery(self, workflow_tasks: list[FailedTask]) -> bool:
        """Führt vollständige Voice Workflow Recovery durch."""
        try:
            # Retry alle Tasks in der richtigen Reihenfolge
            for task in workflow_tasks:
                success = await self.retry_task(task.task_id)
                if not success:
                    return False

            return True

        except Exception as e:
            logger.error(f"Full voice recovery failed: {e}")
            return False

    async def _partial_voice_recovery(self, workflow_tasks: list[FailedTask]) -> bool:
        """Führt partielle Voice Workflow Recovery durch."""
        try:
            # Retry nur kritische Tasks
            critical_tasks = [t for t in workflow_tasks if t.criticality in ["high", "critical"]]

            success_count = 0
            for task in critical_tasks:
                success = await self.retry_task(task.task_id)
                if success:
                    success_count += 1

            # Erfolg wenn mindestens 50% der kritischen Tasks erfolgreich
            return success_count >= len(critical_tasks) * 0.5

        except Exception as e:
            logger.error(f"Partial voice recovery failed: {e}")
            return False

    async def _maintain_session_continuity(self, workflow_tasks: list[FailedTask]) -> bool:
        """Erhält Session Continuity nach Failed Tasks."""
        try:
            # Finde Session Continuity Handler
            session_task = workflow_tasks[0]  # Verwende ersten Task für Session Info
            session_type = session_task.context.get("session_type", "default")

            handler = self._session_continuity_handlers.get(session_type)
            if not handler:
                logger.warning(f"No session continuity handler for type: {session_type}")
                return False

            # Führe Session Continuity durch
            success = await handler(session_task)

            if success:
                self._voice_stats["session_continuity_maintained"] += 1

            return success

        except Exception as e:
            logger.error(f"Session continuity maintenance failed: {e}")
            return False


def create_voice_workflow_dlq(max_queue_size: int = 50000) -> VoiceWorkflowDLQ:
    """Factory-Funktion für Voice Workflow DLQ."""
    return VoiceWorkflowDLQ(max_queue_size)
