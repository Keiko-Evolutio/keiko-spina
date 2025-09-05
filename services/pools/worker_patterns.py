"""Worker-Pattern-Abstraktion für Keiko Personal Assistant.

Bietet gemeinsame Worker-Base-Klassen und -Patterns für verschiedene
Worker-Implementierungen (Task-Worker, Webhook-Worker, etc.).
"""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from kei_logging import get_logger

logger = get_logger(__name__)

# Type Variables
T = TypeVar("T")  # Work Item Type
R = TypeVar("R")  # Result Type


class WorkerState(Enum):
    """Worker-Status-Enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class WorkerConfig:
    """Konfiguration für Worker."""
    worker_id: str
    max_concurrent_tasks: int = 1
    task_timeout: float = 30.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3

    def __post_init__(self) -> None:
        """Validiert Worker-Konfiguration."""
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks muss positiv sein")
        if self.task_timeout <= 0:
            raise ValueError("task_timeout muss positiv sein")
        if self.max_restart_attempts < 0:
            raise ValueError("max_restart_attempts muss nicht-negativ sein")


@dataclass
class WorkerMetrics:
    """Metriken für Worker-Monitoring."""
    worker_id: str
    state: WorkerState = WorkerState.STOPPED
    tasks_processed: int = 0
    tasks_failed: int = 0
    restart_count: int = 0
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        total = self.tasks_processed + self.tasks_failed
        return self.tasks_processed / total if total > 0 else 0.0


class BaseWorker(ABC, Generic[T, R]):
    """Basis-Worker-Klasse für verschiedene Worker-Implementierungen.

    Bietet gemeinsame Worker-Patterns wie Lifecycle-Management,
    Error-Handling, Restart-Logic und Monitoring.
    """

    def __init__(self, config: WorkerConfig) -> None:
        """Initialisiert Worker.

        Args:
            config: Worker-Konfiguration
        """
        self.config = config
        self.metrics = WorkerMetrics(config.worker_id)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Startet Worker.

        Raises:
            RuntimeError: Wenn Worker bereits läuft
        """
        if self._running:
            raise RuntimeError(f"Worker {self.config.worker_id} läuft bereits")

        self.metrics.state = WorkerState.STARTING
        self._running = True
        self._shutdown_event.clear()

        # Starte Worker-Task
        self._task = asyncio.create_task(self._worker_loop())

        logger.info(f"Worker {self.config.worker_id} gestartet")
        self.metrics.state = WorkerState.RUNNING

    async def stop(self, graceful: bool = True, timeout: float = 10.0) -> None:
        """Stoppt Worker.

        Args:
            graceful: Graceful Shutdown (wartet auf aktuelle Tasks)
            timeout: Timeout für Shutdown
        """
        if not self._running:
            logger.debug(f"Worker {self.config.worker_id} bereits gestoppt")
            return

        self.metrics.state = WorkerState.STOPPING
        self._running = False
        self._shutdown_event.set()

        if self._task and not self._task.done():
            if graceful:
                try:
                    await asyncio.wait_for(self._task, timeout=timeout)
                except TimeoutError:
                    logger.warning(f"Worker {self.config.worker_id} Graceful Shutdown Timeout")
                    self._task.cancel()
            else:
                self._task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        self.metrics.state = WorkerState.STOPPED
        logger.info(f"Worker {self.config.worker_id} gestoppt")

    async def _worker_loop(self) -> None:
        """Haupt-Worker-Loop mit Error-Handling und Restart-Logic."""
        restart_attempts = 0

        while self._running:
            try:
                await self._process_work()
                restart_attempts = 0  # Reset bei erfolgreichem Durchlauf

            except asyncio.CancelledError:
                logger.debug(f"Worker {self.config.worker_id} wurde abgebrochen")
                break

            except Exception as e:
                self.metrics.last_error = str(e)
                logger.exception(f"Worker {self.config.worker_id} Fehler: {e}")

                if self.config.restart_on_failure and restart_attempts < self.config.max_restart_attempts:
                    restart_attempts += 1
                    self.metrics.restart_count += 1

                    logger.info(
                        f"Worker {self.config.worker_id} Restart-Versuch "
                        f"{restart_attempts}/{self.config.max_restart_attempts}"
                    )

                    # Kurze Pause vor Restart
                    await asyncio.sleep(min(restart_attempts * 2, 10))
                    continue
                logger.exception(f"Worker {self.config.worker_id} endgültig fehlgeschlagen")
                self.metrics.state = WorkerState.FAILED
                break

        self._running = False

    @abstractmethod
    async def _process_work(self) -> None:
        """Verarbeitet Work Items.

        Diese Methode muss von Subklassen implementiert werden.
        Sollte kontinuierlich Work Items verarbeiten bis _running False ist.
        """

    @abstractmethod
    async def process_item(self, item: T) -> R:
        """Verarbeitet einzelnes Work Item.

        Args:
            item: Zu verarbeitendes Item

        Returns:
            Verarbeitungsresultat

        Raises:
            Exception: Bei Verarbeitungsfehlern
        """

    def get_metrics(self) -> WorkerMetrics:
        """Ruft Worker-Metriken ab.

        Returns:
            Aktuelle Worker-Metriken
        """
        return self.metrics

    def is_running(self) -> bool:
        """Prüft ob Worker läuft.

        Returns:
            True wenn Worker läuft
        """
        return self._running and self.metrics.state == WorkerState.RUNNING


@dataclass
class WorkerPoolConfig:
    """Konfiguration für Worker-Pools."""
    pool_name: str
    pool_size: int = 4
    worker_config_template: WorkerConfig | None = None
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded

    def __post_init__(self) -> None:
        """Validiert Pool-Konfiguration."""
        if self.pool_size <= 0:
            raise ValueError("pool_size muss positiv sein")
        if self.load_balancing_strategy not in ["round_robin", "least_loaded"]:
            raise ValueError("Ungültige Load-Balancing-Strategie")


class BaseWorkerPool(ABC, Generic[T, R]):
    """Basis-Worker-Pool für verschiedene Worker-Pool-Implementierungen.

    Verwaltet mehrere Worker-Instanzen mit Load-Balancing,
    Health-Monitoring und automatischem Failover.
    """

    def __init__(self, config: WorkerPoolConfig) -> None:
        """Initialisiert Worker-Pool.

        Args:
            config: Pool-Konfiguration
        """
        self.config = config
        self._workers: dict[str, BaseWorker[T, R]] = {}
        self._round_robin_index = 0
        self._running = False

    async def start(self) -> None:
        """Startet Worker-Pool.

        Erstellt und startet alle Worker-Instanzen.
        """
        if self._running:
            logger.warning(f"Worker-Pool {self.config.pool_name} läuft bereits")
            return

        # Erstelle Worker
        for i in range(self.config.pool_size):
            worker_id = f"{self.config.pool_name}_worker_{i:03d}"
            worker_config = self._create_worker_config(worker_id)
            worker = self._create_worker(worker_config)

            self._workers[worker_id] = worker
            await worker.start()

        self._running = True
        logger.info(f"Worker-Pool {self.config.pool_name} gestartet mit {len(self._workers)} Workern")

    async def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Stoppt Worker-Pool.

        Args:
            graceful: Graceful Shutdown
            timeout: Timeout für Shutdown
        """
        if not self._running:
            logger.debug(f"Worker-Pool {self.config.pool_name} bereits gestoppt")
            return

        # Stoppe alle Worker parallel
        stop_tasks = []
        for worker in self._workers.values():
            stop_tasks.append(worker.stop(graceful, timeout / len(self._workers)))

        await asyncio.gather(*stop_tasks, return_exceptions=True)

        self._workers.clear()
        self._running = False
        logger.info(f"Worker-Pool {self.config.pool_name} gestoppt")

    def _create_worker_config(self, worker_id: str) -> WorkerConfig:
        """Erstellt Worker-Konfiguration.

        Args:
            worker_id: Worker-ID

        Returns:
            Worker-Konfiguration
        """
        if self.config.worker_config_template:
            # Kopiere Template und setze Worker-ID
            template = self.config.worker_config_template
            return WorkerConfig(
                worker_id=worker_id,
                max_concurrent_tasks=template.max_concurrent_tasks,
                task_timeout=template.task_timeout,
                restart_on_failure=template.restart_on_failure,
                max_restart_attempts=template.max_restart_attempts
            )
        return WorkerConfig(worker_id=worker_id)

    @abstractmethod
    def _create_worker(self, config: WorkerConfig) -> BaseWorker[T, R]:
        """Erstellt Worker-Instanz.

        Args:
            config: Worker-Konfiguration

        Returns:
            Worker-Instanz
        """

    def get_pool_metrics(self) -> dict[str, Any]:
        """Ruft Pool-Metriken ab.

        Returns:
            Dictionary mit Pool-Metriken
        """
        worker_metrics = {
            worker_id: worker.get_metrics()
            for worker_id, worker in self._workers.items()
        }

        total_processed = sum(m.tasks_processed for m in worker_metrics.values())
        total_failed = sum(m.tasks_failed for m in worker_metrics.values())

        return {
            "pool_name": self.config.pool_name,
            "pool_size": len(self._workers),
            "running": self._running,
            "total_tasks_processed": total_processed,
            "total_tasks_failed": total_failed,
            "success_rate": total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0.0,
            "workers": {
                worker_id: {
                    "state": metrics.state.value,
                    "tasks_processed": metrics.tasks_processed,
                    "tasks_failed": metrics.tasks_failed,
                    "success_rate": metrics.success_rate,
                    "restart_count": metrics.restart_count,
                    "last_error": metrics.last_error
                }
                for worker_id, metrics in worker_metrics.items()
            }
        }


__all__ = [
    "BaseWorker",
    "BaseWorkerPool",
    "WorkerConfig",
    "WorkerMetrics",
    "WorkerPoolConfig",
    "WorkerState",
]
