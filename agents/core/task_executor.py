# backend/agents/core/task_executor.py
"""Task-Executor für Keiko Personal Assistant

Implementiert saubere Task-Ausführung mit:
- Async-First Design
- Resource Management
- Error Handling
- Performance Monitoring
"""

import asyncio
import time
import uuid
from abc import ABC
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .utils import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    MetricsCollector,
    ValidationError,
    get_module_logger,
    validate_positive_number,
    validate_required_field,
)

logger = get_module_logger(__name__)

T = TypeVar("T")
TaskExecutorFunc = Callable[[dict[str, Any]], Awaitable[T]]


class TaskStatus:
    """Status-Konstanten für Tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskResult(Generic[T]):
    """Ergebnis einer Task-Ausführung."""

    task_id: str
    success: bool
    result: T | None = None
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Post-Initialisierung."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskContext:
    """Kontext für Task-Ausführung."""

    task_id: str
    agent_id: str
    task_data: dict[str, Any] = None
    task_name: str = ""
    capability: str | None = None
    timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    retry_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Post-Initialisierung mit Validierung."""
        try:
            validate_required_field(self.task_id, "task_id")
            validate_required_field(self.agent_id, "agent_id")
            validate_positive_number(self.timeout_seconds, "timeout_seconds")
            validate_positive_number(self.max_retries, "max_retries")
        except ValidationError as e:
            raise ValueError(str(e)) from e

        if self.metadata is None:
            self.metadata = {}
        if self.task_data is None:
            self.task_data = {}
        if not self.task_name:
            self.task_name = f"task_{self.task_id}"


class TaskExecutionError(Exception):
    """Fehler bei Task-Ausführung."""

    def __init__(
        self, message: str, task_context: TaskContext, original_error: Exception | None = None
    ):
        super().__init__(message)
        self.task_context = task_context
        self.original_error = original_error


class BaseTaskExecutor(ABC, MetricsCollector):
    """Abstrakte Basis-Klasse für Task-Executors."""

    def __init__(self, max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS):
        """Initialisiert Task-Executor.

        Args:
            max_concurrent_tasks: Maximale Anzahl paralleler Tasks
        """
        MetricsCollector.__init__(self)
        validate_positive_number(max_concurrent_tasks, "max_concurrent_tasks")

        self.max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._active_tasks: dict[str, asyncio.Task] = {}

    @property
    def total_tasks(self) -> int:
        """Alias für total_operations für Backward-Compatibility."""
        return self.total_operations

    @property
    def successful_tasks(self) -> int:
        """Alias für successful_operations für Backward-Compatibility."""
        return self.successful_operations

    @property
    def failed_tasks(self) -> int:
        """Alias für failed_operations für Backward-Compatibility."""
        return self.failed_operations

    async def execute_task(
        self, task_context: TaskContext, executor: TaskExecutorFunc[T], task_data: dict[str, Any]
    ) -> TaskResult[T]:
        """Führt Task mit Resource-Management aus.

        Args:
            task_context: Task-Kontext
            executor: Task-Executor-Funktion
            task_data: Task-Daten

        Returns:
            Task-Ergebnis
        """
        async with self._semaphore:
            return await self._execute_with_monitoring(task_context, executor, task_data)

    async def _execute_with_monitoring(
        self, task_context: TaskContext, executor: TaskExecutorFunc[T], task_data: dict[str, Any]
    ) -> TaskResult[T]:
        """Führt Task mit vollständigem Monitoring aus."""
        start_time = time.time()
        # Verwende total_operations statt total_tasks Property für direkten Zugriff
        self.total_operations += 1

        try:
            result = await asyncio.wait_for(
                self._execute_task_safe(task_context, executor, task_data),
                timeout=task_context.timeout_seconds,
            )

            execution_time = time.time() - start_time
            self.record_operation(True, execution_time)

            logger.debug(f"Task {task_context.task_name} erfolgreich ({execution_time:.2f}s)")

            return TaskResult(
                task_id=task_context.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"task_context": task_context},
            )

        except TimeoutError:
            execution_time = time.time() - start_time
            self.record_operation(False, execution_time)

            error_msg = f"Task {task_context.task_name} Timeout nach {task_context.timeout_seconds}s"
            logger.error(error_msg)

            return TaskResult(
                task_id=task_context.task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={"task_context": task_context, "error_type": "timeout"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_operation(False, execution_time)

            error_msg = f"Task {task_context.task_name} fehlgeschlagen: {e}"
            logger.error(error_msg)

            return TaskResult(
                task_id=task_context.task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={"task_context": task_context, "error_type": "execution_error"},
            )

    async def _execute_task_safe(
        self, task_context: TaskContext, executor: TaskExecutorFunc[T], task_data: dict[str, Any]
    ) -> T:
        """Führt Task sicher mit Error-Handling aus."""
        try:
            await self._pre_execution_hook(task_context, task_data)
            result = await executor(task_data)
            await self._post_execution_hook(task_context, task_data, result)
            return result

        except Exception as e:
            await self._error_hook(task_context, task_data, e)
            raise TaskExecutionError(f"Task-Ausführung fehlgeschlagen: {e}", task_context, e)

    @staticmethod
    async def _pre_execution_hook(
        task_context: TaskContext, _task_data: dict[str, Any]
    ) -> None:
        """Hook vor Task-Ausführung.

        Args:
            task_context: Task-Kontext mit Metadaten
            _task_data: Task-Daten für Ausführung (ungenutzt in Basis-Implementation)

        Note:
            Statische Methode - kann von Subklassen überschrieben werden
        """
        logger.debug(f"Starte Task {task_context.task_name} (ID: {task_context.task_id})")

    @staticmethod
    async def _post_execution_hook(
        task_context: TaskContext, _task_data: dict[str, Any], _result: Any
    ) -> None:
        """Hook nach erfolgreicher Task-Ausführung.

        Args:
            task_context: Task-Kontext mit Metadaten
            _task_data: Task-Daten für Ausführung (ungenutzt in Basis-Implementation)
            _result: Ergebnis der Task-Ausführung (ungenutzt in Basis-Implementation)

        Note:
            Statische Methode - kann von Subklassen überschrieben werden
        """
        logger.debug(f"Task {task_context.task_name} erfolgreich abgeschlossen")

    @staticmethod
    async def _error_hook(
        task_context: TaskContext, _task_data: dict[str, Any], error: Exception
    ) -> None:
        """Hook bei Task-Fehler.

        Args:
            task_context: Task-Kontext mit Metadaten
            _task_data: Task-Daten für Ausführung (ungenutzt in Basis-Implementation)
            error: Aufgetretener Fehler

        Note:
            Statische Methode - kann von Subklassen überschrieben werden
        """
        logger.error(f"Task {task_context.task_name} fehlgeschlagen: {error}")

    @asynccontextmanager
    async def task_batch(self, batch_name: str):
        """Context-Manager für Batch-Task-Ausführung."""
        batch_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.info(f"Starte Task-Batch {batch_name} (ID: {batch_id})")
            yield batch_id

        finally:
            execution_time = time.time() - start_time
            logger.info(f"Task-Batch {batch_name} abgeschlossen ({execution_time:.2f}s)")

    async def execute_batch(
        self,
        tasks: list[tuple[TaskContext, TaskExecutorFunc[T], dict[str, Any]]],
        max_concurrent: int | None = None,
    ) -> list[TaskResult[T]]:
        """Führt mehrere Tasks parallel aus.

        Args:
            tasks: Liste von (TaskContext, ExecutorFunc, TaskData)
            max_concurrent: Maximale parallele Tasks

        Returns:
            Liste von Task-Ergebnissen
        """
        max_concurrent = max_concurrent or self.max_concurrent_tasks

        async with self.task_batch(f"batch_{len(tasks)}_tasks"):
            semaphore = asyncio.Semaphore(max_concurrent)

            async def execute_single(task_info):
                async with semaphore:
                    task_context, executor, task_data = task_info
                    return await self.execute_task(task_context, executor, task_data)

            return await asyncio.gather(
                *[execute_single(task_info) for task_info in tasks], return_exceptions=False
            )

    def get_performance_stats(self) -> dict[str, Any]:
        """Holt Performance-Statistiken.

        Returns:
            Performance-Statistiken
        """
        success_rate = self.successful_tasks / max(self.total_tasks, 1)
        avg_execution_time = self.total_execution_time / max(self.total_tasks, 1)

        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "active_tasks": len(self._active_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }

    async def wait_for_completion(self) -> None:
        """Wartet auf Abschluss aller aktiven Tasks."""
        if self._active_tasks:
            logger.info(f"Warte auf {len(self._active_tasks)} aktive Tasks")
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
            self._active_tasks.clear()


class RetryTaskExecutor(BaseTaskExecutor):
    """Task-Executor mit Retry-Mechanismus."""

    def __init__(
        self,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        default_max_retries: int = DEFAULT_MAX_RETRIES
    ):
        """Initialisiert Retry-Task-Executor.

        Args:
            max_concurrent_tasks: Maximale Anzahl paralleler Tasks
            default_max_retries: Standard-Anzahl Retry-Versuche
        """
        super().__init__(max_concurrent_tasks)
        self.default_max_retries = default_max_retries

    async def _execute_task_safe(
        self, task_context: TaskContext, executor: TaskExecutorFunc[T], task_data: dict[str, Any]
    ) -> T:
        """Führt Task mit Retry-Mechanismus aus."""
        last_error = None
        max_retries = task_context.max_retries or self.default_max_retries

        for attempt in range(max_retries + 1):
            try:
                task_context.retry_count = attempt
                return await super()._execute_task_safe(task_context, executor, task_data)

            except TaskExecutionError as e:
                last_error = e

                if attempt < max_retries:
                    retry_delay = min(
                        DEFAULT_RETRY_BASE_DELAY * (DEFAULT_RETRY_BACKOFF_FACTOR ** attempt),
                        DEFAULT_RETRY_MAX_DELAY
                    )
                    logger.warning(
                        f"Task {task_context.task_name} Retry {attempt + 1}/{max_retries} in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Task {task_context.task_name} endgültig fehlgeschlagen nach {max_retries} Retries"
                    )
                    raise

        raise last_error or TaskExecutionError("Unbekannter Retry-Fehler", task_context)


class TaskExecutor(BaseTaskExecutor):
    """Standard Task-Executor für KEI-Agent Framework.

    Direkte Implementierung ohne zusätzliche Features.
    """


class CircuitBreakerTaskExecutor(BaseTaskExecutor):
    """Task-Executor mit Circuit-Breaker-Pattern."""

    def __init__(
        self,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    ):
        """Initialisiert Circuit-Breaker-Task-Executor.

        Args:
            max_concurrent_tasks: Maximale Anzahl paralleler Tasks
            failure_threshold: Anzahl Fehler bis Circuit öffnet
            recovery_timeout: Zeit bis Recovery-Versuch
        """
        super().__init__(max_concurrent_tasks)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.circuit_open = False

    async def _execute_task_safe(
        self, task_context: TaskContext, executor: TaskExecutorFunc[T], task_data: dict[str, Any]
    ) -> T:
        """Führt Task mit Circuit-Breaker aus."""
        if self.circuit_open:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.circuit_open = False
                self.failure_count = 0
                logger.info("Circuit-Breaker Recovery-Versuch")
            else:
                raise TaskExecutionError("Circuit-Breaker offen - Task abgelehnt", task_context)

        try:
            result = await super()._execute_task_safe(task_context, executor, task_data)

            if self.failure_count > 0:
                self.failure_count = 0
                logger.debug("Circuit-Breaker Failure-Count zurückgesetzt")

            return result

        except TaskExecutionError:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                logger.warning(f"Circuit-Breaker geöffnet nach {self.failure_count} Fehlern")

            raise
