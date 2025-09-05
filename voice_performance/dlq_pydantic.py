"""Pydantic-basierte Dead Letter Queue Implementation.
Vollständige Migration zu Pydantic Models mit Type Safety und Validation.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

# Import Pydantic Models
from .models import (
    Criticality,
    FailedTask,
    FailureCategory,
    FailureReason,
    ModelFactory,
    ModelValidator,
    RetryPolicy,
    TaskStatus,
    VoiceWorkflowContext,
)

logger = get_logger(__name__)


class PydanticDeadLetterQueue:
    """Pydantic-basierte Dead Letter Queue für Voice Performance System.
    Vollständige Type Safety und Validation für alle Operations.
    """

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size

        # Pydantic-basierte Storage
        self._failed_tasks: dict[str, FailedTask] = {}

        # Queue Management mit Type Safety
        self._retry_queue: deque = deque()
        self._permanently_failed: deque = deque()

        # Retry Handlers mit Type Safety
        self._retry_handlers: dict[str, Callable[[FailedTask], Awaitable[bool]]] = {}

        # Background Processing
        self._background_task: asyncio.Task | None = None
        self._running = False

        # Statistics mit Pydantic Validation
        self._statistics = {
            "total_failures": 0,
            "total_retries": 0,
            "successful_retries": 0,
            "permanent_failures": 0
        }

        # Validation Pipeline
        self._validator = ModelValidator()

        logger.info(f"Pydantic DLQ initialized with max_queue_size={max_queue_size}")

    async def start(self) -> None:
        """Startet Dead Letter Queue mit Pydantic Validation."""
        if self._running:
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_processor())

        logger.info("Pydantic DLQ started")

    async def stop(self) -> None:
        """Stoppt Dead Letter Queue."""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        logger.info("Pydantic DLQ stopped")

    async def add_failed_task(
        self,
        task_id: str,
        workflow_id: str,
        task_type: str,
        failure_reason: FailureReason,
        error_message: str,
        voice_context: VoiceWorkflowContext | None = None,
        original_request: dict[str, Any] | None = None,
        **kwargs
    ) -> FailedTask:
        """Fügt Failed Task mit vollständiger Pydantic Validation hinzu.

        Returns:
            FailedTask: Validierter und erstellter Failed Task
        """
        try:
            # Erstelle Failed Task mit ModelFactory
            failed_task = ModelFactory.create_failed_task(
                task_id=task_id,
                workflow_id=workflow_id,
                task_type=task_type,
                failure_reason=failure_reason,
                error_message=error_message,
                voice_context=voice_context,
                original_request=original_request or {},
                **kwargs
            )

            # Validiere Failed Task
            validation_result = self._validator.validate_model(failed_task)
            if not validation_result.valid:
                logger.error(f"Failed task validation failed: {validation_result.errors}")
                raise ValueError(f"Failed task validation failed: {validation_result.errors}")

            # Prüfe Queue Capacity
            if len(self._failed_tasks) >= self.max_queue_size:
                logger.warning("DLQ at capacity, removing oldest task")
                await self._cleanup_oldest_task()

            # Speichere Failed Task
            self._failed_tasks[task_id] = failed_task

            # Füge zu Retry Queue hinzu
            self._retry_queue.append(task_id)

            # Update Statistics
            self._statistics["total_failures"] += 1

            logger.info(f"Failed task added: {task_id} (reason: {failure_reason.value})")

            return failed_task

        except Exception as e:
            logger.error(f"Failed to add task to DLQ: {e}")
            raise

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
        criticality: Criticality = Criticality.NORMAL
    ) -> FailedTask:
        """Fügt Voice-spezifischen Failed Task hinzu.

        Returns:
            FailedTask: Validierter Voice Failed Task
        """
        return await self.add_failed_task(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type=task_type,
            failure_reason=failure_reason,
            error_message=error_message,
            voice_context=voice_context,
            original_request=original_request,
            user_id=user_id,
            session_id=session_id,
            priority=priority,
            criticality=criticality,
            failure_category=FailureCategory.VOICE
        )

    async def retry_task(self, task_id: str) -> bool:
        """Führt Retry für Failed Task durch mit Pydantic Validation.

        Returns:
            bool: True wenn Retry erfolgreich
        """
        if task_id not in self._failed_tasks:
            logger.error(f"Task not found for retry: {task_id}")
            return False

        failed_task = self._failed_tasks[task_id]

        # Validiere Task vor Retry
        validation_result = self._validator.validate_model(failed_task)
        if not validation_result.valid:
            logger.error(f"Task validation failed before retry: {validation_result.errors}")
            return False

        # Prüfe Retry-Berechtigung
        if not failed_task.is_retry_ready():
            logger.warning(f"Task not ready for retry: {task_id}")
            return False

        try:
            # Hole Retry Handler
            handler = self._retry_handlers.get(failed_task.task_type)
            if not handler:
                logger.error(f"No retry handler for task type: {failed_task.task_type}")
                return False

            # Update Retry Information
            failed_task.retry_count += 1
            failed_task.last_retry_at = datetime.utcnow()
            failed_task.status = TaskStatus.RETRYING

            # Führe Retry durch
            retry_start = time.time()
            success = await handler(failed_task)
            retry_duration = (time.time() - retry_start) * 1000

            # Update Task basierend auf Retry-Ergebnis
            if success:
                failed_task.status = TaskStatus.RECOVERED
                failed_task.recovery_success = True
                failed_task.recovery_notes.append(f"Successful retry at attempt {failed_task.retry_count}")

                # Entferne aus Retry Queue
                if task_id in self._retry_queue:
                    self._retry_queue.remove(task_id)

                self._statistics["successful_retries"] += 1
                logger.info(f"Task retry successful: {task_id} in {retry_duration:.1f}ms")

            else:
                failed_task.status = TaskStatus.FAILED
                failed_task.recovery_notes.append(f"Failed retry at attempt {failed_task.retry_count}")

                # Prüfe ob permanent failed
                if failed_task.retry_count >= failed_task.max_retries:
                    failed_task.status = TaskStatus.PERMANENTLY_FAILED
                    self._permanently_failed.append(task_id)

                    if task_id in self._retry_queue:
                        self._retry_queue.remove(task_id)

                    self._statistics["permanent_failures"] += 1
                    logger.warning(f"Task permanently failed: {task_id}")

                else:
                    # Berechne nächsten Retry-Zeitpunkt
                    retry_policy = RetryPolicy()  # Default policy
                    delay = failed_task.calculate_next_retry_delay(retry_policy)
                    failed_task.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

                    logger.info(f"Task retry failed: {task_id}, next retry in {delay:.1f}s")

            # Validiere Updated Task
            validation_result = self._validator.validate_model(failed_task)
            if not validation_result.valid:
                logger.error(f"Task validation failed after retry: {validation_result.errors}")

            self._statistics["total_retries"] += 1
            return success

        except Exception as e:
            logger.error(f"Retry failed for task {task_id}: {e}")
            failed_task.status = TaskStatus.FAILED
            failed_task.recovery_notes.append(f"Retry exception: {e!s}")
            return False

    async def get_failed_tasks(
        self,
        task_type: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 100
    ) -> list[FailedTask]:
        """Gibt Failed Tasks mit Filtering zurück.

        Returns:
            List[FailedTask]: Gefilterte und validierte Failed Tasks
        """
        tasks = []
        count = 0

        for failed_task in self._failed_tasks.values():
            if count >= limit:
                break

            # Apply Filters
            if task_type and failed_task.task_type != task_type:
                continue

            if status and failed_task.status != status:
                continue

            # Validiere Task vor Rückgabe
            validation_result = self._validator.validate_model(failed_task)
            if validation_result.valid:
                tasks.append(failed_task)
                count += 1
            else:
                logger.warning(f"Invalid task found in storage: {failed_task.task_id}")

        return tasks

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt DLQ-Statistiken zurück.

        Returns:
            Dict: Validierte Statistiken
        """
        return {
            "queue_stats": self._statistics.copy(),
            "queue_sizes": {
                "retry_queue": len(self._retry_queue),
                "permanently_failed": len(self._permanently_failed),
                "total_failed_tasks": len(self._failed_tasks)
            },
            "validation_stats": {
                "total_validations": getattr(self._validator, "_total_validations", 0),
                "validation_errors": getattr(self._validator, "_validation_errors", 0)
            }
        }

    def register_retry_handler(
        self,
        task_type: str,
        handler: Callable[[FailedTask], Awaitable[bool]]
    ) -> None:
        """Registriert Retry Handler mit Type Safety.

        Args:
            task_type: Task-Typ
            handler: Async Handler Function die FailedTask akzeptiert
        """
        self._retry_handlers[task_type] = handler
        logger.info(f"Retry handler registered for task type: {task_type}")

    async def _background_processor(self) -> None:
        """Background Processor für Retry Queue."""
        while self._running:
            try:
                await self._process_retry_queue()
                await asyncio.sleep(10)  # Process every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                await asyncio.sleep(10)

    async def _process_retry_queue(self) -> None:
        """Verarbeitet Retry Queue mit Pydantic Validation."""
        processed_count = 0
        max_batch_size = 5

        while self._retry_queue and processed_count < max_batch_size:
            try:
                task_id = self._retry_queue.popleft()

                if task_id in self._failed_tasks:
                    failed_task = self._failed_tasks[task_id]

                    # Prüfe ob Ready für Retry
                    if failed_task.is_retry_ready():
                        await self.retry_task(task_id)
                    else:
                        # Zurück in Queue für späteren Retry
                        self._retry_queue.append(task_id)

                processed_count += 1

            except Exception as e:
                logger.error(f"Retry queue processing error: {e}")
                break

    async def _cleanup_oldest_task(self) -> None:
        """Entfernt ältesten Task bei Queue Overflow."""
        if not self._failed_tasks:
            return

        # Finde ältesten Task
        oldest_task_id = min(
            self._failed_tasks.keys(),
            key=lambda tid: self._failed_tasks[tid].created_at
        )

        # Entferne Task
        del self._failed_tasks[oldest_task_id]

        # Entferne aus Queues
        if oldest_task_id in self._retry_queue:
            self._retry_queue.remove(oldest_task_id)

        if oldest_task_id in self._permanently_failed:
            self._permanently_failed.remove(oldest_task_id)

        logger.info(f"Removed oldest task due to capacity: {oldest_task_id}")


def create_pydantic_dead_letter_queue(max_queue_size: int = 10000) -> PydanticDeadLetterQueue:
    """Factory-Funktion für Pydantic Dead Letter Queue."""
    return PydanticDeadLetterQueue(max_queue_size=max_queue_size)
