"""Basis-Worker-Klasse für das KEI-Webhook System.

Stellt gemeinsame Funktionalität für alle Worker bereit: Start/Stop-Lifecycle,
Health-Checks, Task-Management und standardisierte Fehlerbehandlung.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger

from ..constants import (
    WORKER_POLL_INTERVAL_SECONDS,
    WORKER_SHUTDOWN_TIMEOUT_SECONDS,
    WORKER_SUPERVISION_INTERVAL_SECONDS,
)
from ..exceptions import safe_execute_debug

logger = get_logger(__name__)


class WorkerStatus(str, Enum):
    """Status eines Workers."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class WorkerConfig:
    """Konfiguration für Worker."""

    name: str
    poll_interval_seconds: float = WORKER_POLL_INTERVAL_SECONDS
    shutdown_timeout_seconds: float = WORKER_SHUTDOWN_TIMEOUT_SECONDS
    supervision_interval_seconds: float = WORKER_SUPERVISION_INTERVAL_SECONDS
    auto_restart: bool = True
    max_restart_attempts: int = 3


class BaseWorker(ABC):
    """Abstrakte Basis-Klasse für alle Webhook-Worker.

    Stellt standardisierte Lifecycle-Methoden, Health-Checks und
    Task-Management bereit.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._status = WorkerStatus.STOPPED
        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._last_error: Exception | None = None
        self._restart_count = 0
        self._started_at: datetime | None = None
        self._logger = get_logger(f"{__name__}.{config.name}")

    @property
    def status(self) -> WorkerStatus:
        """Aktueller Worker-Status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Prüft, ob Worker läuft."""
        return self._status == WorkerStatus.RUNNING

    @property
    def uptime_seconds(self) -> float:
        """Uptime des Workers in Sekunden."""
        if self._started_at is None:
            return 0.0
        return (datetime.now(UTC) - self._started_at).total_seconds()

    @property
    def health_info(self) -> dict[str, Any]:
        """Health-Informationen des Workers."""
        return {
            "name": self.config.name,
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            "restart_count": self._restart_count,
            "last_error": str(self._last_error) if self._last_error else None,
            "task_alive": self._task is not None and not self._task.done(),
        }

    async def start(self) -> None:
        """Startet den Worker."""
        if self._status != WorkerStatus.STOPPED:
            self._logger.warning(f"Worker bereits gestartet (Status: {self._status})")
            return

        self._logger.info(f"Starte Worker '{self.config.name}'")
        self._status = WorkerStatus.STARTING
        self._shutdown_event.clear()
        self._last_error = None

        try:
            await self._pre_start()
            self._task = asyncio.create_task(self._run_with_supervision())
            self._status = WorkerStatus.RUNNING
            self._started_at = datetime.now(UTC)
            self._logger.info(f"Worker '{self.config.name}' gestartet")
        except Exception as exc:
            self._status = WorkerStatus.ERROR
            self._last_error = exc
            self._logger.exception(f"Worker-Start fehlgeschlagen: {exc}")
            raise

    async def stop(self) -> None:
        """Stoppt den Worker graceful."""
        if self._status == WorkerStatus.STOPPED:
            return

        self._logger.info(f"Stoppe Worker '{self.config.name}'")
        self._status = WorkerStatus.STOPPING
        self._shutdown_event.set()

        if self._task:
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self.config.shutdown_timeout_seconds
                )
            except TimeoutError:
                self._logger.warning("Worker-Shutdown timeout, forciere Abbruch")
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            except Exception as exc:
                self._logger.exception(f"Fehler beim Worker-Shutdown: {exc}")

        await self._post_stop()
        self._status = WorkerStatus.STOPPED
        self._task = None
        self._started_at = None
        self._logger.info(f"Worker '{self.config.name}' gestoppt")

    async def restart(self) -> None:
        """Startet den Worker neu."""
        await self.stop()
        await self.start()

    async def _run_with_supervision(self) -> None:
        """Haupt-Loop mit Supervision und Auto-Restart."""
        while not self._shutdown_event.is_set():
            try:
                await self._run_cycle()

                # Kurze Pause zwischen Zyklen
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.poll_interval_seconds
                    )
                    break  # Shutdown angefordert
                except TimeoutError:
                    continue  # Normaler Zyklus

            except asyncio.CancelledError:
                self._logger.debug("Worker-Task wurde abgebrochen")
                break
            except Exception as exc:
                self._last_error = exc
                self._logger.exception(f"Worker-Zyklus fehlgeschlagen: {exc}")

                if self.config.auto_restart and self._restart_count < self.config.max_restart_attempts:
                    self._restart_count += 1
                    self._logger.info(f"Auto-Restart {self._restart_count}/{self.config.max_restart_attempts}")
                    await asyncio.sleep(min(2.0 ** self._restart_count, 30.0))  # Exponential backoff
                    continue
                self._status = WorkerStatus.ERROR
                self._logger.exception("Maximale Restart-Versuche erreicht, Worker gestoppt")
                break

    @abstractmethod
    async def _run_cycle(self) -> None:
        """Führt einen Worker-Zyklus aus.

        Diese Methode muss von Subklassen implementiert werden und enthält
        die eigentliche Worker-Logik.
        """

    async def _pre_start(self) -> None:
        """Hook für Initialisierung vor Worker-Start.

        Kann von Subklassen überschrieben werden.
        """

    async def _post_stop(self) -> None:
        """Hook für Cleanup nach Worker-Stop.

        Kann von Subklassen überschrieben werden.
        """

    def _safe_execute(self, fn, error_code: str, **context) -> Any:
        """Sichere Ausführung mit Logging."""
        return safe_execute_debug(
            fn,
            logger=self._logger,
            error_code=error_code,
            context=context
        )


class WorkerManager:
    """Manager für mehrere Worker."""

    def __init__(self) -> None:
        self._workers: dict[str, BaseWorker] = {}

    def register(self, worker: BaseWorker) -> None:
        """Registriert einen Worker."""
        self._workers[worker.config.name] = worker

    def unregister(self, name: str) -> None:
        """Entfernt einen Worker."""
        self._workers.pop(name, None)

    async def start_all(self) -> None:
        """Startet alle registrierten Worker."""
        for worker in self._workers.values():
            try:
                await worker.start()
            except Exception as exc:
                logger.exception(f"Fehler beim Starten von Worker '{worker.config.name}': {exc}")

    async def stop_all(self) -> None:
        """Stoppt alle registrierten Worker."""
        tasks = []
        for worker in self._workers.values():
            if worker.is_running:
                tasks.append(worker.stop())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_health_status(self) -> dict[str, Any]:
        """Holt Health-Status aller Worker."""
        return {
            name: worker.health_info
            for name, worker in self._workers.items()
        }

    def get_worker(self, name: str) -> BaseWorker | None:
        """Holt Worker nach Name."""
        return self._workers.get(name)


# Globaler Worker-Manager
_worker_manager: WorkerManager | None = None


def get_worker_manager() -> WorkerManager:
    """Holt die globale WorkerManager-Instanz."""
    global _worker_manager
    if _worker_manager is None:
        _worker_manager = WorkerManager()
    return _worker_manager


__all__ = [
    "BaseWorker",
    "WorkerConfig",
    "WorkerManager",
    "WorkerStatus",
    "get_worker_manager"
]
