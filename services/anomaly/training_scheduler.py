"""Scheduler/Worker für periodisches LSTM-Training und Baseline-Lernen.

Verwendet Redis als einfache Job-Queue (Listen/Keys). In Production kann
dies durch Celery/Arq/Temporal ersetzt werden. Hier focus auf Minimalismus
und Integration in bestehende Services.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import settings
from kei_logging import get_logger

from .common import TrainingConfig, redis_helper, safe_ml_operation
from .lstm_service import LSTMAnomalyService
from .service import AnomalyDetectionService

logger = get_logger(__name__)


class AnomalyTrainingScheduler:
    """Einfacher Async-Scheduler für LSTM-Training und Baseline-Learning."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._task: asyncio.Task | None = None
        self._running = False
        self._lstm_service = LSTMAnomalyService()
        self._anomaly_service = AnomalyDetectionService()

    async def start(self) -> None:
        """Startet den Training-Scheduler."""
        if self._running or not getattr(settings, "anomaly_training_enabled", True):
            return
        self._running = True
        self._task = asyncio.create_task(self._training_loop())

    async def stop(self) -> None:
        """Stoppt den Training-Scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Fehler beim Stoppen des Schedulers: {e}")
            self._task = None

    async def _training_loop(self) -> None:
        """Hauptschleife für Training-Job-Verarbeitung."""
        interval = getattr(settings, "anomaly_training_interval_minutes", 5) * 60
        interval = max(self.config.interval_seconds, interval)

        while self._running:
            try:
                await self._process_training_jobs()
            except Exception as exc:
                logger.warning(f"TrainingScheduler Fehler: {exc}")
            await asyncio.sleep(interval)

    async def _process_training_jobs(self) -> None:
        """Verarbeitet alle verfügbaren Training-Jobs aus der Queue."""
        while self._running:
            job_data = await redis_helper.dequeue_training_job()
            if not job_data:
                break
            await self._execute_training_job(job_data)

    async def _execute_training_job(self, job_data: dict[str, Any]) -> None:
        """Führt einen einzelnen Training-Job aus."""
        tenant = job_data.get("tenant", "default")
        metric = job_data.get("metric", "unknown")
        values = job_data.get("values", [])

        if not values:
            logger.warning(f"Leerer Training-Job für {tenant}:{metric}")
            return

        # 1) Baseline-Statistiken aktualisieren
        baseline_success = await self._anomaly_service.learn_baseline(
            tenant=tenant,
            metric_name=metric,
            values=values
        )

        # 2) LSTM-Training (falls verfügbar)
        lstm_success = safe_ml_operation(
            "lstm_training_job",
            self._lstm_service.train,
            values,
            default_return=False
        )

        logger.info(
            f"Training-Job abgeschlossen für {tenant}:{metric} "
            f"(Baseline: {baseline_success}, LSTM: {lstm_success})"
        )

    async def enqueue_training(self, tenant: str, metric: str, values: list[float]) -> bool:
        """Fügt Training-Job zur Queue hinzu."""
        return await redis_helper.enqueue_training_job(tenant, metric, values)
