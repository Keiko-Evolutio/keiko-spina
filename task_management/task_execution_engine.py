# backend/task_management/task_execution_engine.py
"""Task Execution Engine für Keiko Personal Assistant

Implementiert asynchrone Task-Execution mit Worker-Pools, Task-Scheduling,
Load-Balancing und horizontaler Skalierung.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

from .constants import (
    QUEUE_POLL_TIMEOUT_SECONDS,
    SIMULATED_TASK_EXECUTION_TIME_MS,
)
from .core_task_manager import Task, TaskPriority, TaskResult, TaskState, task_manager
from .utils import (
    calculate_execution_time_ms,
    get_current_utc_datetime,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)


class WorkerState(str, Enum):
    """Worker-Status."""
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class QueueStrategy(str, Enum):
    """Queue-Strategien."""
    FIFO = "fifo"
    PRIORITY = "priority"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    DEADLINE_FIRST = "deadline_first"


@dataclass
class ExecutionResult:
    """Ergebnis einer Task-Ausführung."""
    task_id: str
    success: bool

    # Execution-Details
    started_at: datetime
    completed_at: datetime
    execution_time_ms: float

    # Result-Data
    result_data: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None

    # Resource-Usage
    cpu_time_ms: float | None = None
    memory_usage_mb: float | None = None
    network_calls: int | None = None

    # Worker-Information
    worker_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "network_calls": self.network_calls,
            "worker_id": self.worker_id
        }


@dataclass
class WorkerMetrics:
    """Metriken für Task-Worker."""
    worker_id: str

    # Execution-Metriken
    tasks_executed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0

    # Performance-Metriken
    avg_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0

    # Resource-Metriken
    avg_cpu_usage_percent: float = 0.0
    avg_memory_usage_mb: float = 0.0

    # Status
    current_state: WorkerState = WorkerState.IDLE
    last_task_at: datetime | None = None

    # Uptime
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.tasks_executed == 0:
            return 0.0
        return self.tasks_succeeded / self.tasks_executed

    @property
    def uptime_seconds(self) -> float:
        """Berechnet Uptime in Sekunden."""
        return (datetime.now(UTC) - self.started_at).total_seconds()

    def update_execution_metrics(self, execution_result: ExecutionResult) -> None:
        """Aktualisiert Execution-Metriken."""
        self.tasks_executed += 1

        if execution_result.success:
            self.tasks_succeeded += 1
        else:
            self.tasks_failed += 1

        # Aktualisiere Durchschnitts-Execution-Zeit
        self.total_execution_time_ms += execution_result.execution_time_ms
        self.avg_execution_time_ms = self.total_execution_time_ms / self.tasks_executed

        self.last_task_at = execution_result.completed_at

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "worker_id": self.worker_id,
            "tasks_executed": self.tasks_executed,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "total_execution_time_ms": self.total_execution_time_ms,
            "avg_cpu_usage_percent": self.avg_cpu_usage_percent,
            "avg_memory_usage_mb": self.avg_memory_usage_mb,
            "current_state": self.current_state.value,
            "last_task_at": self.last_task_at.isoformat() if self.last_task_at else None,
            "started_at": self.started_at.isoformat(),
            "success_rate": self.success_rate,
            "uptime_seconds": self.uptime_seconds
        }


class TaskWorker:
    """Einzelner Task-Worker."""

    def __init__(self, worker_id: str, executor_function: Callable[[Task], Awaitable[ExecutionResult]]):
        """Initialisiert Task Worker.

        Args:
            worker_id: Worker-ID
            executor_function: Task-Executor-Funktion
        """
        self.worker_id = worker_id
        self.executor_function = executor_function

        # Worker-State
        self.metrics = WorkerMetrics(worker_id=worker_id)
        self._current_task: Task | None = None
        self._worker_task: asyncio.Task | None = None
        self._is_running = False
        self._stop_event = asyncio.Event()

        # Task-Queue für diesen Worker
        self._task_queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> None:
        """Startet Worker."""
        if self._is_running:
            return

        self._is_running = True
        self.metrics.current_state = WorkerState.IDLE
        self._worker_task = asyncio.create_task(self._worker_loop())

        logger.info(f"Task Worker gestartet: {self.worker_id}")

    async def stop(self, graceful: bool = True) -> None:
        """Stoppt Worker.

        Args:
            graceful: Graceful Shutdown
        """
        self._is_running = False
        self.metrics.current_state = WorkerState.STOPPING

        if graceful:
            # Warte auf aktuellen Task
            self._stop_event.set()

            if self._worker_task:
                try:
                    await asyncio.wait_for(self._worker_task, timeout=30.0)
                except TimeoutError:
                    logger.warning(f"Worker {self.worker_id} Graceful-Shutdown-Timeout")
                    self._worker_task.cancel()
        # Forceful Stop
        elif self._worker_task:
            self._worker_task.cancel()

        self.metrics.current_state = WorkerState.STOPPED
        logger.info(f"Task Worker gestoppt: {self.worker_id}")

    async def submit_task(self, task: Task) -> None:
        """Reicht Task an Worker weiter.

        Args:
            task: Auszuführende Task
        """
        if not self._is_running:
            raise RuntimeError(f"Worker {self.worker_id} ist nicht aktiv")

        await self._task_queue.put(task)

    @property
    def is_idle(self) -> bool:
        """Prüft, ob Worker idle ist."""
        return self.metrics.current_state == WorkerState.IDLE

    @property
    def is_busy(self) -> bool:
        """Prüft, ob Worker busy ist."""
        return self.metrics.current_state == WorkerState.BUSY

    @property
    def queue_size(self) -> int:
        """Gibt Queue-Größe zurück."""
        return self._task_queue.qsize()

    async def _worker_loop(self) -> None:
        """Haupt-Worker-Loop."""
        while self._is_running:
            try:
                # Warte auf Task oder Stop-Signal
                try:
                    task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0
                    )
                except TimeoutError:
                    # Timeout ist normal - prüfe Stop-Event
                    if self._stop_event.is_set():
                        break
                    continue

                # Führe Task aus
                await self._execute_task(task)

            except Exception as e:
                logger.exception(f"Worker {self.worker_id} Loop-Fehler: {e}")
                self.metrics.current_state = WorkerState.ERROR
                await asyncio.sleep(1.0)
                self.metrics.current_state = WorkerState.IDLE

    @trace_function("task_worker.execute_task")
    async def _execute_task(self, task: Task) -> None:
        """Führt einzelne Task aus."""
        self._current_task = task
        self.metrics.current_state = WorkerState.BUSY

        # Aktualisiere Task-State
        await task_manager.update_task_state(task.task_id, TaskState.RUNNING)

        # Erstelle Task-Result
        if not task.result:
            task.result = TaskResult(task_id=task.task_id, state=TaskState.RUNNING)

        task.result.started_at = datetime.now(UTC)

        try:
            # Führe Task aus
            execution_result = await self.executor_function(task)

            # Aktualisiere Task-Result
            task.result.completed_at = execution_result.completed_at
            task.result.execution_time_ms = execution_result.execution_time_ms
            task.result.result_data = execution_result.result_data
            task.result.error_message = execution_result.error_message
            task.result.error_code = execution_result.error_code
            task.result.error_details = execution_result.error_details
            task.result.cpu_time_ms = execution_result.cpu_time_ms
            task.result.memory_usage_mb = execution_result.memory_usage_mb
            task.result.network_calls = execution_result.network_calls

            # Aktualisiere Task-State
            if execution_result.success:
                await task_manager.update_task_state(task.task_id, TaskState.COMPLETED)
                task.result.state = TaskState.COMPLETED
            else:
                await task_manager.update_task_state(task.task_id, TaskState.FAILED)
                task.result.state = TaskState.FAILED

            # Aktualisiere Worker-Metriken
            self.metrics.update_execution_metrics(execution_result)

            logger.info(f"Task ausgeführt: {task.task_id} von Worker {self.worker_id} ({'Erfolg' if execution_result.success else 'Fehler'})")

        except Exception as e:
            # Task-Execution fehlgeschlagen
            task.result.completed_at = datetime.now(UTC)
            task.result.execution_time_ms = (task.result.completed_at - task.result.started_at).total_seconds() * 1000
            task.result.error_message = str(e)
            task.result.error_code = "EXECUTION_ERROR"
            task.result.state = TaskState.FAILED

            await task_manager.update_task_state(task.task_id, TaskState.FAILED)

            # Erstelle Execution-Result für Metriken
            execution_result = ExecutionResult(
                task_id=task.task_id,
                success=False,
                started_at=task.result.started_at,
                completed_at=task.result.completed_at,
                execution_time_ms=task.result.execution_time_ms,
                error_message=str(e),
                worker_id=self.worker_id
            )

            self.metrics.update_execution_metrics(execution_result)

            logger.exception(f"Task-Execution fehlgeschlagen: {task.task_id} von Worker {self.worker_id}: {e}")

        finally:
            self._current_task = None
            self.metrics.current_state = WorkerState.IDLE


class TaskQueue:
    """Task-Queue mit verschiedenen Scheduling-Strategien."""

    def __init__(self, strategy: QueueStrategy = QueueStrategy.PRIORITY):
        """Initialisiert Task Queue.

        Args:
            strategy: Queue-Strategie
        """
        self.strategy = strategy

        # Queue-Storage
        self._tasks: deque = deque()
        self._priority_queues: dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }

        # Locks
        self._queue_lock = asyncio.Lock()

        # Statistiken
        self._tasks_queued = 0
        self._tasks_dequeued = 0

    async def enqueue(self, task: Task) -> None:
        """Fügt Task zur Queue hinzu.

        Args:
            task: Task
        """
        async with self._queue_lock:
            if self.strategy == QueueStrategy.PRIORITY:
                self._priority_queues[task.priority].append(task)
            else:
                self._tasks.append(task)

            self._tasks_queued += 1

            # Aktualisiere Task-State
            await task_manager.update_task_state(task.task_id, TaskState.QUEUED)

    async def dequeue(self) -> Task | None:
        """Holt nächste Task aus Queue.

        Returns:
            Nächste Task oder None
        """
        async with self._queue_lock:
            task = None

            if self.strategy == QueueStrategy.PRIORITY:
                # Höchste Priorität zuerst
                for priority in [TaskPriority.URGENT, TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
                    if self._priority_queues[priority]:
                        task = self._priority_queues[priority].popleft()
                        break

            elif self.strategy == QueueStrategy.FIFO:
                if self._tasks:
                    task = self._tasks.popleft()

            elif self.strategy == QueueStrategy.SHORTEST_JOB_FIRST:
                # Vereinfachte Implementierung - sortiere nach geschätzter Dauer
                if self._tasks:
                    # Für Demo: Verwende Payload-Größe als Proxy für Execution-Zeit
                    sorted_tasks = sorted(self._tasks, key=lambda t: len(str(t.payload)))
                    task = sorted_tasks[0]
                    self._tasks.remove(task)

            elif self.strategy == QueueStrategy.DEADLINE_FIRST:
                # Sortiere nach Deadline (falls vorhanden)
                if self._tasks:
                    tasks_with_deadlines = [t for t in self._tasks if t.schedule and t.schedule.latest_start]
                    if tasks_with_deadlines:
                        task = min(tasks_with_deadlines, key=lambda t: t.schedule.latest_start)
                        self._tasks.remove(task)
                    else:
                        task = self._tasks.popleft()

            if task:
                self._tasks_dequeued += 1

            return task

    @property
    def size(self) -> int:
        """Gibt Queue-Größe zurück."""
        if self.strategy == QueueStrategy.PRIORITY:
            return sum(len(queue) for queue in self._priority_queues.values())
        return len(self._tasks)

    @property
    def is_empty(self) -> bool:
        """Prüft, ob Queue leer ist."""
        return self.size == 0

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Queue-Statistiken zurück."""
        stats = {
            "strategy": self.strategy.value,
            "total_size": self.size,
            "tasks_queued": self._tasks_queued,
            "tasks_dequeued": self._tasks_dequeued,
            "is_empty": self.is_empty
        }

        if self.strategy == QueueStrategy.PRIORITY:
            stats["priority_distribution"] = {
                priority.value: len(queue)
                for priority, queue in self._priority_queues.items()
            }

        return stats


class TaskWorkerPool:
    """Pool von Task-Workern."""

    def __init__(
        self,
        pool_size: int = 10,
        executor_function: Callable[[Task], Awaitable[ExecutionResult]] | None = None
    ):
        """Initialisiert Worker Pool.

        Args:
            pool_size: Anzahl Worker
            executor_function: Task-Executor-Funktion
        """
        self.pool_size = pool_size
        self.executor_function = executor_function or self._default_executor

        # Worker-Management
        self._workers: dict[str, TaskWorker] = {}
        self._worker_assignment: dict[str, str] = {}  # task_id -> worker_id

        # Load-Balancing
        self._round_robin_index = 0

        # Statistiken
        self._tasks_assigned = 0

    async def start(self) -> None:
        """Startet Worker Pool."""
        for i in range(self.pool_size):
            worker_id = f"worker_{i:03d}"
            worker = TaskWorker(worker_id, self.executor_function)
            self._workers[worker_id] = worker
            await worker.start()

        logger.info(f"Worker Pool gestartet mit {self.pool_size} Workern")

    async def stop(self, graceful: bool = True) -> None:
        """Stoppt Worker Pool.

        Args:
            graceful: Graceful Shutdown
        """
        stop_tasks = []
        for worker in self._workers.values():
            stop_tasks.append(worker.stop(graceful))

        await asyncio.gather(*stop_tasks, return_exceptions=True)

        logger.info("Worker Pool gestoppt")

    async def submit_task(self, task: Task) -> str:
        """Reicht Task an Worker weiter.

        Args:
            task: Task

        Returns:
            Worker-ID
        """
        # Wähle Worker (Round-Robin Load-Balancing)
        worker_id = self._select_worker()
        worker = self._workers[worker_id]

        # Reiche Task weiter
        await worker.submit_task(task)

        # Tracking
        self._worker_assignment[task.task_id] = worker_id
        self._tasks_assigned += 1

        logger.debug(f"Task {task.task_id} an Worker {worker_id} zugewiesen")

        return worker_id

    def _select_worker(self) -> str:
        """Wählt Worker für Load-Balancing."""
        # Round-Robin-Auswahl
        worker_ids = list(self._workers.keys())
        worker_id = worker_ids[self._round_robin_index % len(worker_ids)]
        self._round_robin_index += 1

        return worker_id

    def get_worker_metrics(self) -> dict[str, WorkerMetrics]:
        """Gibt Worker-Metriken zurück."""
        return {worker_id: worker.metrics for worker_id, worker in self._workers.items()}

    def get_pool_statistics(self) -> dict[str, Any]:
        """Gibt Pool-Statistiken zurück."""
        idle_workers = sum(1 for worker in self._workers.values() if worker.is_idle)
        busy_workers = sum(1 for worker in self._workers.values() if worker.is_busy)

        total_tasks_executed = sum(worker.metrics.tasks_executed for worker in self._workers.values())
        total_tasks_succeeded = sum(worker.metrics.tasks_succeeded for worker in self._workers.values())

        return {
            "pool_size": self.pool_size,
            "idle_workers": idle_workers,
            "busy_workers": busy_workers,
            "tasks_assigned": self._tasks_assigned,
            "total_tasks_executed": total_tasks_executed,
            "total_tasks_succeeded": total_tasks_succeeded,
            "overall_success_rate": total_tasks_succeeded / max(total_tasks_executed, 1),
            "worker_utilization": busy_workers / self.pool_size
        }

    async def _default_executor(self, task: Task) -> ExecutionResult:
        """Standard-Task-Executor."""
        start_time = get_current_utc_datetime()

        try:
            # Simuliere Task-Execution
            await asyncio.sleep(SIMULATED_TASK_EXECUTION_TIME_MS / 1000)  # Simulierte Verarbeitungszeit

            end_time = get_current_utc_datetime()
            execution_time = calculate_execution_time_ms(start_time, end_time)

            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                started_at=start_time,
                completed_at=end_time,
                execution_time_ms=execution_time,
                result_data={"message": f"Task {task.task_id} erfolgreich ausgeführt"}
            )

        except Exception as e:
            end_time = datetime.now(UTC)
            execution_time = (end_time - start_time).total_seconds() * 1000

            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                started_at=start_time,
                completed_at=end_time,
                execution_time_ms=execution_time,
                error_message=str(e),
                error_code="DEFAULT_EXECUTOR_ERROR"
            )


class TaskScheduler:
    """Task-Scheduler für zeitbasierte Ausführung."""

    def __init__(self, worker_pool: TaskWorkerPool):
        """Initialisiert Task Scheduler.

        Args:
            worker_pool: Worker Pool
        """
        self.worker_pool = worker_pool

        # Scheduling
        self._scheduled_tasks: dict[str, Task] = {}
        self._scheduler_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Startet Scheduler."""
        if self._is_running:
            return

        self._is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info("Task Scheduler gestartet")

    async def stop(self) -> None:
        """Stoppt Scheduler."""
        self._is_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task

        logger.info("Task Scheduler gestoppt")

    async def schedule_task(self, task: Task) -> None:
        """Plant Task für spätere Ausführung.

        Args:
            task: Task
        """
        if not task.schedule:
            # Keine Scheduling-Information - direkt ausführen
            await self.worker_pool.submit_task(task)
            return

        # Berechne Ausführungszeit
        execution_time = None

        if task.schedule.scheduled_at:
            execution_time = task.schedule.scheduled_at
        elif task.schedule.delay_seconds:
            execution_time = datetime.now(UTC) + timedelta(seconds=task.schedule.delay_seconds)

        if execution_time:
            self._scheduled_tasks[task.task_id] = task
            logger.info(f"Task geplant: {task.task_id} für {execution_time.isoformat()}")
        else:
            # Keine gültige Scheduling-Zeit - direkt ausführen
            await self.worker_pool.submit_task(task)

    async def _scheduler_loop(self) -> None:
        """Scheduler-Loop."""
        while self._is_running:
            try:
                now = datetime.now(UTC)
                ready_tasks = []

                # Finde bereite Tasks
                for task_id, task in list(self._scheduled_tasks.items()):
                    if task.schedule:
                        execution_time = task.schedule.scheduled_at
                        if not execution_time and task.schedule.delay_seconds:
                            execution_time = task.created_at + timedelta(seconds=task.schedule.delay_seconds)

                        if execution_time and now >= execution_time:
                            ready_tasks.append(task)
                            del self._scheduled_tasks[task_id]

                # Führe bereite Tasks aus
                for task in ready_tasks:
                    await self.worker_pool.submit_task(task)

                # Warte 1 Sekunde
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.exception(f"Scheduler-Loop-Fehler: {e}")
                await asyncio.sleep(1.0)


class TaskExecutionEngine:
    """Hauptklasse für Task-Execution."""

    def __init__(
        self,
        pool_size: int = 10,
        queue_strategy: QueueStrategy = QueueStrategy.PRIORITY,
        executor_function: Callable[[Task], Awaitable[ExecutionResult]] | None = None
    ):
        """Initialisiert Task Execution Engine.

        Args:
            pool_size: Worker-Pool-Größe
            queue_strategy: Queue-Strategie
            executor_function: Task-Executor-Funktion
        """
        # Komponenten
        self.task_queue = TaskQueue(queue_strategy)
        self.worker_pool = TaskWorkerPool(pool_size, executor_function)
        self.scheduler = TaskScheduler(self.worker_pool)

        # Engine-State
        self._is_running = False
        self._dispatcher_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Startet Execution Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Komponenten
        await self.worker_pool.start()
        await self.scheduler.start()

        # Starte Dispatcher
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())

        logger.info("Task Execution Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Execution Engine."""
        self._is_running = False

        # Stoppe Dispatcher
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher_task

        # Stoppe Komponenten
        await self.scheduler.stop()
        await self.worker_pool.stop()

        logger.info("Task Execution Engine gestoppt")

    async def submit_task(self, task: Task) -> None:
        """Reicht Task zur Ausführung ein.

        Args:
            task: Task
        """
        if task.schedule:
            # Geplante Task
            await self.scheduler.schedule_task(task)
        else:
            # Sofortige Ausführung
            await self.task_queue.enqueue(task)

    async def _dispatcher_loop(self) -> None:
        """Dispatcher-Loop für Queue-Processing."""
        while self._is_running:
            try:
                # Hole Task aus Queue
                task = await self.task_queue.dequeue()

                if task:
                    # Reiche an Worker Pool weiter
                    await self.worker_pool.submit_task(task)
                else:
                    # Keine Tasks - kurz warten
                    await asyncio.sleep(QUEUE_POLL_TIMEOUT_SECONDS)

            except Exception as e:
                logger.exception(f"Dispatcher-Loop-Fehler: {e}")
                await asyncio.sleep(1.0)

    def get_engine_statistics(self) -> dict[str, Any]:
        """Gibt Engine-Statistiken zurück."""
        return {
            "is_running": self._is_running,
            "queue_stats": self.task_queue.get_statistics(),
            "pool_stats": self.worker_pool.get_pool_statistics(),
            "scheduled_tasks": len(self.scheduler._scheduled_tasks)
        }


# Globale Task Execution Engine Instanz
task_execution_engine = TaskExecutionEngine()
