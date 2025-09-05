"""Distributed Task Processor für Keiko Personal Assistant.

Dieses Modul implementiert den verteilten Task-Processor für die Verarbeitung
von Edge-Computing-Tasks im verteilten System.
"""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
from .edge_types import (
    EdgeConfiguration,
    EdgeProcessingCapability,
    EdgeTask,
    EdgeTaskPriority,
    EdgeTaskResult,
    EdgeTaskStatus,
    EdgeTaskType,
)

logger = get_logger(__name__)


class TaskDistributionStrategy(str, Enum):
    """Task-Verteilungsstrategien."""
    ROUND_ROBIN = "round-robin"
    LOAD_BASED = "load-based"
    CAPABILITY_BASED = "capability-based"
    LATENCY_OPTIMIZED = "latency-optimized"


@dataclass
class TaskExecutionContext:
    """Kontext für Task-Ausführung."""
    task_id: str
    assigned_node_id: str | None = None
    execution_start: datetime | None = None
    execution_end: datetime | None = None
    retry_count: int = 0
    error_message: str | None = None
    performance_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessorMetrics:
    """Metriken für Task-Processor."""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time_ms: float = 0.0
    active_tasks: int = 0
    queue_size: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


class DistributedTaskProcessor:
    """Enterprise Distributed Task Processor für Edge Computing.

    Implementiert intelligente Task-Verteilung mit:
    - Automatische Load-Balancing zwischen Edge-Nodes
    - Capability-basierte Task-Zuordnung
    - Retry-Mechanismen und Fehlerbehandlung
    - Performance-Monitoring und Metriken
    - Prioritäts-basierte Task-Scheduling
    """

    def __init__(self, config: EdgeConfiguration | None = None):
        """Initialisiert den Distributed Task Processor.

        Args:
            config: Edge-Konfiguration
        """
        self.config = config or EdgeConfiguration()

        # Task-Management
        self._task_queue: asyncio.Queue[EdgeTask] = asyncio.Queue(
            maxsize=self.config.task_queue_size
        )
        self._active_tasks: dict[str, TaskExecutionContext] = {}
        self._completed_tasks: dict[str, EdgeTaskResult] = {}
        self._task_lock = asyncio.Lock()

        # Worker-Management
        self._worker_tasks: list[asyncio.Task] = []
        self._running = False
        self._max_workers = self.config.max_concurrent_tasks_per_node

        # Konfiguration
        self.distribution_strategy = TaskDistributionStrategy.CAPABILITY_BASED
        self.task_timeout = timedelta(seconds=self.config.task_timeout_seconds)
        self.max_retries = self.config.task_retry_attempts

        # Metriken
        self.metrics = ProcessorMetrics()

        # Callbacks
        self._task_completion_callbacks: list[Callable[[EdgeTaskResult], None]] = []
        self._task_failure_callbacks: list[Callable[[str, str], None]] = []

        logger.info("Distributed Task Processor initialisiert")

    async def start(self) -> None:
        """Startet den Task-Processor und Worker."""
        if self._running:
            logger.warning("Task-Processor bereits gestartet")
            return

        self._running = True

        # Worker-Tasks starten
        for i in range(self._max_workers):
            worker_task = asyncio.create_task(
                self._worker_loop(f"worker-{i}"),
                name=f"task-processor-worker-{i}"
            )
            self._worker_tasks.append(worker_task)

        # Monitoring-Task starten
        monitoring_task = asyncio.create_task(
            self._monitoring_loop(),
            name="task-processor-monitoring"
        )
        self._worker_tasks.append(monitoring_task)

        logger.info(f"Task-Processor gestartet mit {self._max_workers} Workern")

    async def stop(self) -> None:
        """Stoppt den Task-Processor und alle Worker."""
        if not self._running:
            return

        self._running = False

        # Alle Worker-Tasks stoppen
        for task in self._worker_tasks:
            task.cancel()

        # Warten bis alle Tasks beendet sind
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks.clear()

        logger.info("Task-Processor gestoppt")

    async def submit_task(
        self,
        task_type: EdgeTaskType,
        input_data: bytes,
        processing_params: dict[str, Any] | None = None,
        priority: EdgeTaskPriority = EdgeTaskPriority.NORMAL,
        required_capabilities: list[EdgeProcessingCapability] | None = None,
        deadline: datetime | None = None
    ) -> str:
        """Reicht eine neue Task zur Verarbeitung ein.

        Args:
            task_type: Typ der Task
            input_data: Eingabedaten
            processing_params: Verarbeitungsparameter
            priority: Task-Priorität
            required_capabilities: Erforderliche Capabilities
            deadline: Deadline für Verarbeitung

        Returns:
            Task-ID

        Raises:
            asyncio.QueueFull: Wenn Task-Queue voll ist
        """
        task_id = str(uuid.uuid4())

        # Task erstellen
        task = EdgeTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            input_data=input_data,
            processing_params=processing_params or {},
            expected_output_format="json",
            deadline=deadline,
            required_capabilities=required_capabilities or []
        )

        try:
            # Task in Queue einreihen
            await self._task_queue.put(task)

            # Metriken aktualisieren
            self.metrics.queue_size = self._task_queue.qsize()

            logger.info(f"Task eingereicht: {task_id} (Typ: {task_type.value}, Priorität: {priority.value})")

            return task_id

        except asyncio.QueueFull:
            logger.error(f"Task-Queue voll - Task {task_id} abgelehnt")
            raise

    async def get_task_status(self, task_id: str) -> EdgeTaskStatus | None:
        """Gibt Status einer Task zurück.

        Args:
            task_id: Task-ID

        Returns:
            Task-Status oder None wenn nicht gefunden
        """
        async with self._task_lock:
            # Aktive Tasks prüfen
            if task_id in self._active_tasks:
                context = self._active_tasks[task_id]
                if context.execution_start:
                    return EdgeTaskStatus.RUNNING
                return EdgeTaskStatus.ASSIGNED

            # Abgeschlossene Tasks prüfen
            if task_id in self._completed_tasks:
                result = self._completed_tasks[task_id]
                return EdgeTaskStatus.COMPLETED if result.success else EdgeTaskStatus.FAILED

            return None

    async def get_task_result(self, task_id: str) -> EdgeTaskResult | None:
        """Gibt Ergebnis einer abgeschlossenen Task zurück.

        Args:
            task_id: Task-ID

        Returns:
            Task-Ergebnis oder None wenn nicht gefunden
        """
        async with self._task_lock:
            return self._completed_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Bricht eine Task ab.

        Args:
            task_id: Task-ID

        Returns:
            True wenn erfolgreich abgebrochen
        """
        async with self._task_lock:
            if task_id in self._active_tasks:
                context = self._active_tasks[task_id]
                context.error_message = "Task cancelled by user"

                # Task als abgebrochen markieren
                result = EdgeTaskResult(
                    task_id=task_id,
                    success=False,
                    output_data=b"",
                    error_message="Task cancelled",
                    processing_time_ms=0,
                    completed_at=datetime.now(UTC)
                )

                self._completed_tasks[task_id] = result
                del self._active_tasks[task_id]

                logger.info(f"Task abgebrochen: {task_id}")
                return True

        return False

    def add_completion_callback(self, callback: Callable[[EdgeTaskResult], None]) -> None:
        """Fügt Callback für Task-Completion hinzu.

        Args:
            callback: Callback-Funktion
        """
        self._task_completion_callbacks.append(callback)

    def add_failure_callback(self, callback: Callable[[str, str], None]) -> None:
        """Fügt Callback für Task-Failure hinzu.

        Args:
            callback: Callback-Funktion (task_id, error_message)
        """
        self._task_failure_callbacks.append(callback)

    async def _worker_loop(self, worker_id: str) -> None:
        """Worker-Loop für Task-Verarbeitung.

        Args:
            worker_id: Worker-ID
        """
        logger.debug(f"Worker {worker_id} gestartet")

        while self._running:
            try:
                # Task aus Queue holen (mit Timeout)
                try:
                    task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Task verarbeiten
                await self._process_task(task, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in Worker {worker_id}: {e}")
                await asyncio.sleep(1)

        logger.debug(f"Worker {worker_id} gestoppt")

    async def _process_task(self, task: EdgeTask, worker_id: str) -> None:
        """Verarbeitet eine einzelne Task.

        Args:
            task: Zu verarbeitende Task
            worker_id: Worker-ID
        """
        start_time = datetime.now(UTC)

        # Task-Kontext erstellen
        context = TaskExecutionContext(
            task_id=task.task_id,
            execution_start=start_time
        )

        async with self._task_lock:
            self._active_tasks[task.task_id] = context
            self.metrics.active_tasks += 1

        try:
            # Task-Status aktualisieren
            task.status = EdgeTaskStatus.RUNNING
            task.started_at = start_time

            logger.debug(f"Worker {worker_id} verarbeitet Task {task.task_id}")

            # Simulierte Task-Verarbeitung (hier würde echte Verarbeitung stattfinden)
            await self._execute_task_logic(task, context)

            # Erfolgreiches Ergebnis erstellen
            end_time = datetime.now(UTC)
            processing_time = (end_time - start_time).total_seconds() * 1000

            result = EdgeTaskResult(
                task_id=task.task_id,
                node_id=worker_id,
                status=EdgeTaskStatus.COMPLETED,
                success=True,
                output_data=b"processed_data",  # Placeholder
                processing_time_ms=int(processing_time),
                completed_at=end_time
            )

            # Ergebnis speichern
            async with self._task_lock:
                self._completed_tasks[task.task_id] = result
                del self._active_tasks[task.task_id]
                self.metrics.active_tasks -= 1
                self.metrics.successful_tasks += 1
                self.metrics.total_tasks_processed += 1

            # Callbacks aufrufen
            for callback in self._task_completion_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Fehler in Completion-Callback: {e}")

            logger.info(f"Task erfolgreich verarbeitet: {task.task_id} ({processing_time:.1f}ms)")

        except Exception as e:
            # Fehlerbehandlung
            end_time = datetime.now(UTC)
            processing_time = (end_time - start_time).total_seconds() * 1000

            error_message = str(e)

            # Retry-Logik
            context.retry_count += 1
            if context.retry_count <= self.max_retries:
                logger.warning(f"Task {task.task_id} fehlgeschlagen, Retry {context.retry_count}/{self.max_retries}")
                # Task wieder in Queue einreihen
                await self._task_queue.put(task)
                return

            # Fehler-Ergebnis erstellen
            result = EdgeTaskResult(
                task_id=task.task_id,
                node_id=worker_id,
                status=EdgeTaskStatus.FAILED,
                success=False,
                output_data=b"",
                error_message=error_message,
                processing_time_ms=int(processing_time),
                completed_at=end_time
            )

            # Ergebnis speichern
            async with self._task_lock:
                self._completed_tasks[task.task_id] = result
                del self._active_tasks[task.task_id]
                self.metrics.active_tasks -= 1
                self.metrics.failed_tasks += 1
                self.metrics.total_tasks_processed += 1

            # Callbacks aufrufen
            for callback in self._task_failure_callbacks:
                try:
                    callback(task.task_id, error_message)
                except Exception as cb_error:
                    logger.error(f"Fehler in Failure-Callback: {cb_error}")

            logger.error(f"Task fehlgeschlagen: {task.task_id} - {error_message}")

    async def _execute_task_logic(self, task: EdgeTask, context: TaskExecutionContext) -> None:
        """Führt die eigentliche Task-Logik aus.

        Args:
            task: Task
            context: Ausführungskontext
        """
        # Simulierte Verarbeitung basierend auf Task-Typ
        if task.task_type == EdgeTaskType.AUDIO_PROCESSING:
            await asyncio.sleep(0.1)  # Simuliere Audio-Verarbeitung
        elif task.task_type == EdgeTaskType.AI_INFERENCE:
            await asyncio.sleep(0.5)  # Simuliere AI-Inferenz
        elif task.task_type == EdgeTaskType.DATA_TRANSFORMATION:
            await asyncio.sleep(0.05)  # Simuliere Daten-Transformation
        else:
            await asyncio.sleep(0.02)  # Standard-Verarbeitung

    async def _monitoring_loop(self) -> None:
        """Monitoring-Loop für Metriken und Cleanup."""
        while self._running:
            try:
                # Metriken aktualisieren
                await self._update_metrics()

                # Cleanup alter abgeschlossener Tasks
                await self._cleanup_old_tasks()

                # Timeout-Prüfung für aktive Tasks
                await self._check_task_timeouts()

                # Warten bis zum nächsten Check
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Monitoring-Loop: {e}")
                await asyncio.sleep(5)

    async def _update_metrics(self) -> None:
        """Aktualisiert Performance-Metriken."""
        async with self._task_lock:
            self.metrics.queue_size = self._task_queue.qsize()
            self.metrics.last_updated = datetime.now(UTC)

            # Durchschnittliche Verarbeitungszeit berechnen
            if self.metrics.total_tasks_processed > 0:
                total_time = sum(
                    result.processing_time_ms
                    for result in self._completed_tasks.values()
                )
                self.metrics.average_processing_time_ms = (
                    total_time / self.metrics.total_tasks_processed
                )

    async def _cleanup_old_tasks(self) -> None:
        """Bereinigt alte abgeschlossene Tasks."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=1)

        async with self._task_lock:
            old_task_ids = [
                task_id for task_id, result in self._completed_tasks.items()
                if result.completed_at < cutoff_time
            ]

            for task_id in old_task_ids:
                del self._completed_tasks[task_id]

            if old_task_ids:
                logger.debug(f"Bereinigt {len(old_task_ids)} alte Tasks")

    async def _check_task_timeouts(self) -> None:
        """Prüft auf Task-Timeouts."""
        current_time = datetime.now(UTC)

        async with self._task_lock:
            timeout_tasks = []

            for task_id, context in self._active_tasks.items():
                if context.execution_start:
                    elapsed = current_time - context.execution_start
                    if elapsed > self.task_timeout:
                        timeout_tasks.append(task_id)

            # Timeout-Tasks abbrechen
            for task_id in timeout_tasks:
                context = self._active_tasks[task_id]
                result = EdgeTaskResult(
                    task_id=task_id,
                    node_id="timeout",
                    status=EdgeTaskStatus.TIMEOUT,
                    success=False,
                    output_data=b"",
                    error_message="Task timeout",
                    processing_time_ms=int(self.task_timeout.total_seconds() * 1000),
                    completed_at=current_time
                )

                self._completed_tasks[task_id] = result
                del self._active_tasks[task_id]
                self.metrics.active_tasks -= 1
                self.metrics.failed_tasks += 1
                self.metrics.total_tasks_processed += 1

                logger.warning(f"Task-Timeout: {task_id}")

    async def get_processor_status(self) -> dict[str, Any]:
        """Gibt Processor-Status zurück.

        Returns:
            Dictionary mit Processor-Statistiken
        """
        async with self._task_lock:
            return {
                "running": self._running,
                "active_workers": len(self._worker_tasks),
                "queue_size": self.metrics.queue_size,
                "active_tasks": self.metrics.active_tasks,
                "total_processed": self.metrics.total_tasks_processed,
                "successful_tasks": self.metrics.successful_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "average_processing_time_ms": self.metrics.average_processing_time_ms,
                "last_updated": self.metrics.last_updated.isoformat()
            }


def create_task_processor(config: EdgeConfiguration | None = None) -> DistributedTaskProcessor:
    """Factory-Funktion für Distributed Task Processor.

    Args:
        config: Edge-Konfiguration

    Returns:
        Neue DistributedTaskProcessor-Instanz
    """
    return DistributedTaskProcessor(config)
