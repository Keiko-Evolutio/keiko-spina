"""Dead Letter Queue Implementation für Voice Performance System.
Behandelt Failed Tasks und ermöglicht Retry-Mechanismen.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

# Import Pydantic Models
from .models import FailedTask, FailureReason, RetryPolicy, TaskStatus

logger = get_logger(__name__)

# Note: FailureReason, TaskStatus, VoiceWorkflowContext und FailedTask sind jetzt in models/ package definiert









class DeadLetterQueue:
    """Dead Letter Queue Implementation für Voice Performance System.
    Sammelt, verwaltet und versucht Failed Tasks erneut.
    """

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size

        # Queue Storage mit Pydantic Models
        self._failed_tasks: dict[str, FailedTask] = {}
        self._retry_queue: deque = deque(maxlen=max_queue_size)
        self._permanently_failed: deque = deque(maxlen=1000)

        # Retry Policies per Task Type mit Pydantic Models
        self._retry_policies: dict[str, RetryPolicy] = {
            "voice_workflow": RetryPolicy(max_retries=3, initial_delay_seconds=2.0),
            "agent_discovery": RetryPolicy(max_retries=5, initial_delay_seconds=1.0),
            "agent_execution": RetryPolicy(max_retries=2, initial_delay_seconds=5.0),
            "cache_operation": RetryPolicy(max_retries=3, initial_delay_seconds=0.5),
            "default": RetryPolicy(max_retries=3, initial_delay_seconds=1.0)
        }

        # Retry Handlers
        self._retry_handlers: dict[str, Callable[[FailedTask], Awaitable[bool]]] = {}

        # Background Tasks
        self._retry_processor_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._stats = {
            "total_failures": 0,
            "total_retries": 0,
            "successful_retries": 0,
            "permanent_failures": 0,
            "current_queue_size": 0
        }

        logger.info("Dead letter queue initialized")

    async def start(self) -> None:
        """Startet Dead Letter Queue Background-Tasks."""
        if self._running:
            return

        self._running = True

        # Starte Retry Processor
        self._retry_processor_task = asyncio.create_task(self._retry_processor_loop())

        # Starte Cleanup Task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Dead letter queue background tasks started")

    async def stop(self) -> None:
        """Stoppt Dead Letter Queue Background-Tasks."""
        self._running = False

        if self._retry_processor_task:
            self._retry_processor_task.cancel()
            try:
                await self._retry_processor_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Dead letter queue background tasks stopped")

    async def add_failed_task(
        self,
        task_id: str,
        workflow_id: str,
        task_type: str,
        failure_reason: FailureReason,
        error_message: str,
        original_request: dict[str, Any],
        context: dict[str, Any] | None = None,
        stack_trace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        priority: int = 0
    ) -> None:
        """Fügt einen fehlgeschlagenen Task zur Dead Letter Queue hinzu."""
        # Erstelle Failed Task
        failed_task = FailedTask(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type=task_type,
            failure_reason=failure_reason,
            error_message=error_message,
            stack_trace=stack_trace,
            original_request=original_request,
            context=context or {},
            user_id=user_id,
            session_id=session_id,
            priority=priority
        )

        # Bestimme Retry Policy
        retry_policy = self._retry_policies.get(task_type, self._retry_policies["default"])
        failed_task.max_retries = retry_policy.max_retries

        # Berechne nächsten Retry-Zeitpunkt
        if failed_task.retry_count < failed_task.max_retries:
            delay = self._calculate_retry_delay(retry_policy, failed_task.retry_count)
            failed_task.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
            failed_task.status = TaskStatus.FAILED
        else:
            failed_task.status = TaskStatus.PERMANENTLY_FAILED

        # Speichere Task
        self._failed_tasks[task_id] = failed_task

        if failed_task.status == TaskStatus.FAILED:
            self._retry_queue.append(task_id)
        else:
            self._permanently_failed.append(task_id)

        # Update Statistics
        self._stats["total_failures"] += 1
        self._stats["current_queue_size"] = len(self._retry_queue)

        if failed_task.status == TaskStatus.PERMANENTLY_FAILED:
            self._stats["permanent_failures"] += 1

        logger.warning(f"Task added to dead letter queue: {task_id} ({failure_reason.value})")

    def register_retry_handler(
        self,
        task_type: str,
        handler: Callable[[FailedTask], Awaitable[bool]]
    ) -> None:
        """Registriert einen Retry-Handler für einen Task-Typ."""
        self._retry_handlers[task_type] = handler
        logger.info(f"Retry handler registered for task type: {task_type}")

    async def retry_task(self, task_id: str) -> bool:
        """Versucht einen spezifischen Task erneut."""
        if task_id not in self._failed_tasks:
            logger.warning(f"Task not found in dead letter queue: {task_id}")
            return False

        failed_task = self._failed_tasks[task_id]

        # Prüfe ob Retry möglich
        if failed_task.retry_count >= failed_task.max_retries:
            logger.warning(f"Task exceeded max retries: {task_id}")
            return False

        # Prüfe ob Retry-Handler verfügbar
        handler = self._retry_handlers.get(failed_task.task_type)
        if not handler:
            logger.warning(f"No retry handler for task type: {failed_task.task_type}")
            return False

        try:
            # Update Task Status
            failed_task.status = TaskStatus.RETRYING
            failed_task.retry_count += 1
            failed_task.last_retry_at = datetime.utcnow()

            # Führe Retry durch
            success = await handler(failed_task)

            if success:
                # Retry erfolgreich
                failed_task.status = TaskStatus.RECOVERED
                self._stats["successful_retries"] += 1
                logger.info(f"Task successfully retried: {task_id}")
                return True
            # Retry fehlgeschlagen
            if failed_task.retry_count >= failed_task.max_retries:
                failed_task.status = TaskStatus.PERMANENTLY_FAILED
                self._permanently_failed.append(task_id)
                self._stats["permanent_failures"] += 1
            else:
                # Plane nächsten Retry
                retry_policy = self._retry_policies.get(
                    failed_task.task_type,
                    self._retry_policies["default"]
                )
                delay = self._calculate_retry_delay(retry_policy, failed_task.retry_count)
                failed_task.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
                failed_task.status = TaskStatus.FAILED

            self._stats["total_retries"] += 1
            logger.warning(f"Task retry failed: {task_id} (attempt {failed_task.retry_count})")
            return False

        except Exception as e:
            logger.error(f"Error during task retry: {task_id}: {e}")
            failed_task.status = TaskStatus.FAILED
            return False

    async def get_failed_tasks(
        self,
        task_type: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 100
    ) -> list[FailedTask]:
        """Gibt Failed Tasks zurück."""
        tasks = []

        for task in self._failed_tasks.values():
            if task_type and task.task_type != task_type:
                continue
            if status and task.status != status:
                continue

            tasks.append(task)

            if len(tasks) >= limit:
                break

        # Sortiere nach Priorität und Zeitstempel
        tasks.sort(key=lambda t: (-t.priority, t.failed_at))

        return tasks

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Dead Letter Queue Statistiken zurück."""
        return {
            "queue_stats": self._stats.copy(),
            "queue_sizes": {
                "retry_queue": len(self._retry_queue),
                "permanently_failed": len(self._permanently_failed),
                "total_failed_tasks": len(self._failed_tasks)
            },
            "retry_policies": {
                task_type: {
                    "max_retries": policy.max_retries,
                    "initial_delay_seconds": policy.initial_delay_seconds,
                    "exponential_backoff": policy.exponential_backoff
                }
                for task_type, policy in self._retry_policies.items()
            }
        }

    # Private Methods

    async def _retry_processor_loop(self) -> None:
        """Background-Task für Retry Processing."""
        while self._running:
            try:
                await self._process_retry_queue()
                await asyncio.sleep(10)  # Alle 10 Sekunden
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry processor loop: {e}")
                await asyncio.sleep(10)

    async def _process_retry_queue(self) -> None:
        """Verarbeitet Retry Queue."""
        now = datetime.utcnow()
        tasks_to_retry = []

        # Sammle Tasks die für Retry bereit sind
        while self._retry_queue:
            task_id = self._retry_queue.popleft()

            if task_id not in self._failed_tasks:
                continue

            failed_task = self._failed_tasks[task_id]

            if failed_task.next_retry_at and failed_task.next_retry_at <= now:
                tasks_to_retry.append(task_id)
            else:
                # Zurück in Queue
                self._retry_queue.append(task_id)
                break

        # Führe Retries durch
        for task_id in tasks_to_retry:
            try:
                await self.retry_task(task_id)
            except Exception as e:
                logger.error(f"Error retrying task {task_id}: {e}")

        # Update Statistics
        self._stats["current_queue_size"] = len(self._retry_queue)

    async def _cleanup_loop(self) -> None:
        """Background-Task für Cleanup."""
        while self._running:
            try:
                await self._cleanup_old_tasks()
                await asyncio.sleep(3600)  # Alle Stunde
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_tasks(self) -> None:
        """Entfernt alte Tasks."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)  # 7 Tage

        tasks_to_remove = []
        for task_id, failed_task in self._failed_tasks.items():
            if failed_task.failed_at < cutoff_time:
                if failed_task.status in [TaskStatus.RECOVERED, TaskStatus.PERMANENTLY_FAILED]:
                    tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self._failed_tasks[task_id]

        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks from dead letter queue")

    def _calculate_retry_delay(self, policy: RetryPolicy, retry_count: int) -> float:
        """Berechnet Retry-Delay basierend auf Policy."""
        if policy.exponential_backoff:
            delay = policy.initial_delay_seconds * (policy.backoff_multiplier ** retry_count)
        else:
            delay = policy.initial_delay_seconds

        # Begrenze auf max_delay
        delay = min(delay, policy.max_delay_seconds)

        # Füge Jitter hinzu
        if policy.jitter:
            import random
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter

        return delay


def create_dead_letter_queue(max_queue_size: int = 10000) -> DeadLetterQueue:
    """Factory-Funktion für Dead Letter Queue."""
    return DeadLetterQueue(max_queue_size)
