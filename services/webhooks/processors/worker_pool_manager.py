"""Worker-Pool-Manager für das KEI-Webhook System.

Verwaltet den Lifecycle von Delivery-Workern mit automatischer Supervision
und koordiniertem Shutdown.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from ..constants import WORKER_SHUTDOWN_TIMEOUT_SECONDS
from ..delivery_worker import DeliveryWorker
from ..prometheus_metrics import WEBHOOK_ACTIVE_WORKERS

if TYPE_CHECKING:
    from ..models import WorkerPoolConfig

logger = get_logger(__name__)


class WebhookWorkerPoolManager:
    """Verwaltet Worker-Pool-Lifecycle mit Supervision und Health-Monitoring."""

    def __init__(self, config: WorkerPoolConfig) -> None:
        """Initialisiert den Worker-Pool-Manager.

        Args:
            config: Worker-Pool-Konfiguration
        """
        self.config = config
        self.workers: list[DeliveryWorker] = []
        self.running = False

    async def start_pool(self) -> None:
        """Startet alle Worker mit automatischer Supervision."""
        if self.running:
            logger.warning("Worker-Pool bereits gestartet")
            return

        logger.info(
            "Starte Worker-Pool: %d Worker für Shards %s",
            self.config.worker_count,
            self.config.shard_names
        )

        # Erstelle Worker für jeden Shard
        self.workers = [
            DeliveryWorker(
                queue_name=shard,
                poll_interval=self.config.poll_interval
            )
            for shard in self.config.shard_names
        ]

        # Starte alle Worker parallel
        try:
            await asyncio.gather(*(worker.start() for worker in self.workers))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Fehler beim Starten der Worker: %s", exc)
            # Cleanup bei Fehler
            await self._cleanup_workers()
            raise

        # Supervision für automatischen Neustart einrichten
        for worker in self.workers:
            self._attach_supervision(worker)

        self.running = True
        logger.info("Worker-Pool erfolgreich gestartet")

    async def stop_pool(
        self,
        timeout_seconds: float = WORKER_SHUTDOWN_TIMEOUT_SECONDS
    ) -> None:
        """Stoppt alle Worker koordiniert mit Timeout.

        Args:
            timeout_seconds: Maximale Shutdown-Zeit
        """
        if not self.running:
            logger.info("Worker-Pool bereits gestoppt")
            return

        logger.info(
            "Stoppe Worker-Pool: %d Worker (Timeout: %.1fs)",
            len(self.workers),
            timeout_seconds
        )

        self.running = False

        # Erstelle Stop-Tasks für alle Worker
        stop_tasks = [
            asyncio.create_task(worker.stop(timeout=timeout_seconds))
            for worker in self.workers
        ]

        # Warte auf alle Stops mit globalem Timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*stop_tasks, return_exceptions=True),
                timeout=timeout_seconds
            )
        except TimeoutError:
            logger.warning("Worker-Pool Shutdown-Timeout - erzwinge Abbruch")
            await self._force_stop_workers()

        # Prüfe Stop-Ergebnisse
        await self._check_stop_results(stop_tasks)

        # Cleanup
        self.workers.clear()
        logger.info("Worker-Pool erfolgreich gestoppt")

    def get_pool_status(self) -> dict[str, Any]:
        """Gibt detaillierten Status des Worker-Pools zurück.

        Returns:
            Dictionary mit Pool-Status-Informationen
        """
        worker_statuses = []
        active_count = 0

        for worker in self.workers:
            try:
                status = worker.get_status()
                worker_statuses.append(status)
                if status.get("active", False):
                    active_count += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Worker-Status-Abfrage fehlgeschlagen: %s", exc)
                worker_statuses.append({
                    "queue_name": getattr(worker, "queue_name", "unknown"),
                    "active": False,
                    "error": str(exc)
                })

        # Prometheus Metric aktualisieren
        try:
            WEBHOOK_ACTIVE_WORKERS.set(active_count)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Prometheus metric update failed: %s", exc)

        return {
            "configured_workers": self.config.worker_count,
            "active_workers": active_count,
            "queue_base": self.config.queue_name,
            "poll_interval": self.config.poll_interval,
            "running": self.running,
            "workers": worker_statuses,
        }

    def _attach_supervision(self, worker: DeliveryWorker) -> None:
        """Registriert Supervision für automatischen Worker-Neustart.

        Args:
            worker: Worker-Instanz für Supervision
        """
        task = getattr(worker, "_task", None)
        if not isinstance(task, asyncio.Task):
            logger.warning("Worker hat keine Task für Supervision: %s", worker.queue_name)
            return

        def _on_worker_done(completed_task: asyncio.Task) -> None:
            """Callback für beendete Worker-Tasks."""
            if not self.running:
                return

            should_restart = False
            try:
                exc = completed_task.exception()
                should_restart = exc is not None
                if exc:
                    logger.warning(
                        "Worker %s beendet mit Exception: %s",
                        worker.queue_name, exc
                    )
            except asyncio.CancelledError:
                logger.info("Worker %s wurde abgebrochen", worker.queue_name)
                should_restart = False
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception(
                    "Fehler beim Prüfen der Worker-Exception: %s", exc
                )
                should_restart = True

            if should_restart:
                asyncio.create_task(self._restart_worker(worker))

        task.add_done_callback(_on_worker_done)

    async def _restart_worker(self, failed_worker: DeliveryWorker) -> None:
        """Startet einen fehlgeschlagenen Worker neu.

        Args:
            failed_worker: Der fehlgeschlagene Worker
        """
        try:
            # Neuen Worker erstellen
            new_worker = DeliveryWorker(
                queue_name=failed_worker.queue_name,
                poll_interval=self.config.poll_interval
            )

            # In Worker-Liste ersetzen
            for i, worker in enumerate(self.workers):
                if worker is failed_worker:
                    self.workers[i] = new_worker
                    break

            # Neuen Worker starten und überwachen
            await new_worker.start()
            self._attach_supervision(new_worker)

            logger.info("Worker für Shard '%s' erfolgreich neu gestartet", new_worker.queue_name)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception(
                "Worker-Neustart fehlgeschlagen für Shard '%s': %s",
                failed_worker.queue_name, exc
            )

    async def _cleanup_workers(self) -> None:
        """Bereinigt Worker bei Startup-Fehlern."""
        for worker in self.workers:
            try:
                await worker.stop(timeout=5.0)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Worker-Cleanup-Fehler: %s", exc)
        self.workers.clear()

    async def _force_stop_workers(self) -> None:
        """Erzwingt das Stoppen aller Worker-Tasks."""
        for worker in self.workers:
            task = getattr(worker, "_task", None)
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()

    async def _check_stop_results(self, stop_tasks: list[asyncio.Task]) -> None:
        """Prüft die Ergebnisse der Stop-Tasks.

        Args:
            stop_tasks: Liste der Stop-Tasks
        """
        for idx, task in enumerate(stop_tasks):
            try:
                result = await asyncio.shield(task)
                if isinstance(result, Exception):
                    logger.warning(
                        "Worker-Stop meldete Exception (Index %d): %s",
                        idx, result
                    )
            except asyncio.CancelledError:
                logger.info("Worker-Stop Task (Index %d) wurde abgebrochen", idx)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Fehler beim Warten auf Stop-Task (Index %d): %s",
                    idx, exc
                )


__all__ = ["WebhookWorkerPoolManager"]
