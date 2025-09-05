"""Unit-Tests für anomaly/training_scheduler.py.

Testet AnomalyTrainingScheduler mit Async-Mocking und Job-Verarbeitung.
"""

from unittest.mock import AsyncMock, patch

import pytest

from services.anomaly.common import TrainingConfig
from services.anomaly.training_scheduler import AnomalyTrainingScheduler


class TestAnomalyTrainingScheduler:
    """Tests für AnomalyTrainingScheduler-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.config = TrainingConfig(
            interval_seconds=60,
            queue_key="test:queue"
        )
        self.scheduler = AnomalyTrainingScheduler(self.config)

    def test_init_with_config(self):
        """Testet Initialisierung mit Konfiguration."""
        assert self.scheduler.config == self.config
        assert self.scheduler._task is None
        assert self.scheduler._running is False
        assert self.scheduler._lstm_service is not None
        assert self.scheduler._anomaly_service is not None

    def test_init_without_config(self):
        """Testet Initialisierung ohne Konfiguration (Standard-Werte)."""
        scheduler = AnomalyTrainingScheduler()
        assert isinstance(scheduler.config, TrainingConfig)
        assert scheduler.config.interval_seconds == 300  # Standard-Wert

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Testet start wenn Scheduler bereits läuft."""
        self.scheduler._running = True

        await self.scheduler.start()

        # Task sollte nicht erstellt werden
        assert self.scheduler._task is None

    @pytest.mark.asyncio
    async def test_start_training_disabled(self):
        """Testet start wenn Training deaktiviert ist."""
        with patch("services.anomaly.training_scheduler.settings") as mock_settings:
            mock_settings.anomaly_training_enabled = False

            await self.scheduler.start()

            assert self.scheduler._running is False
            assert self.scheduler._task is None

    @pytest.mark.asyncio
    async def test_start_success(self):
        """Testet start bei erfolgreichem Aufruf."""
        with patch("services.anomaly.training_scheduler.settings") as mock_settings:
            mock_settings.anomaly_training_enabled = True

            # Mock _training_loop als Coroutine
            async def mock_training_loop():
                pass

            with patch.object(self.scheduler, "_training_loop", side_effect=mock_training_loop):
                await self.scheduler.start()

                assert self.scheduler._running is True
                assert self.scheduler._task is not None

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Testet stop wenn Scheduler nicht läuft."""
        await self.scheduler.stop()

        assert self.scheduler._running is False
        assert self.scheduler._task is None

    @pytest.mark.asyncio
    async def test_stop_with_task(self):
        """Testet stop mit laufendem Task."""
        mock_task = AsyncMock()
        self.scheduler._task = mock_task
        self.scheduler._running = True

        await self.scheduler.stop()

        assert self.scheduler._running is False
        assert self.scheduler._task is None
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_task_exception(self):
        """Testet stop mit Task-Exception."""
        mock_task = AsyncMock()
        mock_task.side_effect = Exception("Task error")
        self.scheduler._task = mock_task
        self.scheduler._running = True

        with patch("services.anomaly.training_scheduler.logger") as mock_logger:
            await self.scheduler.stop()

            assert self.scheduler._running is False
            assert self.scheduler._task is None
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_training_loop_interval_calculation(self):
        """Testet _training_loop Intervall-Berechnung."""
        with patch("services.anomaly.training_scheduler.settings") as mock_settings, \
             patch.object(self.scheduler, "_process_training_jobs") as mock_process, \
             patch("asyncio.sleep") as mock_sleep:

            mock_settings.anomaly_training_interval_minutes = 2  # 2 Minuten = 120 Sekunden
            self.scheduler._running = True

            # Stoppe nach erstem Durchlauf
            def stop_after_first():
                self.scheduler._running = False

            mock_process.side_effect = stop_after_first

            await self.scheduler._training_loop()

            # Sollte max(config.interval_seconds, settings * 60) verwenden
            expected_interval = max(60, 120)  # 120 Sekunden
            mock_sleep.assert_called_once_with(expected_interval)

    @pytest.mark.asyncio
    async def test_training_loop_exception_handling(self):
        """Testet _training_loop Exception-Handling."""
        with patch("services.anomaly.training_scheduler.settings") as mock_settings, \
             patch.object(self.scheduler, "_process_training_jobs") as mock_process, \
             patch("asyncio.sleep") as mock_sleep, \
             patch("services.anomaly.training_scheduler.logger") as mock_logger:

            mock_settings.anomaly_training_interval_minutes = 1
            mock_process.side_effect = [Exception("Process error"), None]
            self.scheduler._running = True

            # Stoppe nach zweitem Durchlauf
            call_count = 0
            async def stop_after_second(_interval):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    self.scheduler._running = False

            mock_sleep.side_effect = stop_after_second

            await self.scheduler._training_loop()

            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_training_jobs_empty_queue(self):
        """Testet _process_training_jobs mit leerer Queue."""
        self.scheduler._running = True

        with patch("services.anomaly.training_scheduler.redis_helper") as mock_redis:
            mock_redis.dequeue_training_job = AsyncMock(return_value=None)

            await self.scheduler._process_training_jobs()

            mock_redis.dequeue_training_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_training_jobs_with_jobs(self):
        """Testet _process_training_jobs mit Jobs in der Queue."""
        job1 = {"tenant": "tenant1", "metric": "cpu", "values": [1.0, 2.0]}
        job2 = {"tenant": "tenant2", "metric": "memory", "values": [3.0, 4.0]}

        self.scheduler._running = True

        with patch("services.anomaly.training_scheduler.redis_helper") as mock_redis, \
             patch.object(self.scheduler, "_execute_training_job") as mock_execute:

            # Erste zwei Aufrufe geben Jobs zurück, dritter gibt None zurück
            mock_redis.dequeue_training_job = AsyncMock(side_effect=[job1, job2, None])

            await self.scheduler._process_training_jobs()

            assert mock_redis.dequeue_training_job.call_count == 3
            assert mock_execute.call_count == 2
            mock_execute.assert_any_call(job1)
            mock_execute.assert_any_call(job2)

    @pytest.mark.asyncio
    async def test_execute_training_job_empty_values(self):
        """Testet _execute_training_job mit leeren Werten."""
        job_data = {"tenant": "tenant1", "metric": "cpu", "values": []}

        with patch("services.anomaly.training_scheduler.logger") as mock_logger:
            await self.scheduler._execute_training_job(job_data)

            mock_logger.warning.assert_called_once_with(
                "Leerer Training-Job für tenant1:cpu"
            )

    @pytest.mark.asyncio
    async def test_execute_training_job_success(self):
        """Testet _execute_training_job bei erfolgreichem Aufruf."""
        job_data = {"tenant": "tenant1", "metric": "cpu", "values": [1.0, 2.0, 3.0]}

        with patch.object(self.scheduler._anomaly_service, "learn_baseline") as mock_baseline, \
             patch("services.anomaly.training_scheduler.safe_ml_operation") as mock_safe_ml, \
             patch("services.anomaly.training_scheduler.logger") as mock_logger:

            mock_baseline.return_value = True
            mock_safe_ml.return_value = True

            await self.scheduler._execute_training_job(job_data)

            mock_baseline.assert_called_once_with(
                tenant="tenant1",
                metric_name="cpu",
                values=[1.0, 2.0, 3.0]
            )
            mock_safe_ml.assert_called_once()
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_training_job_baseline_failed(self):
        """Testet _execute_training_job wenn Baseline-Learning fehlschlägt."""
        job_data = {"tenant": "tenant1", "metric": "cpu", "values": [1.0, 2.0, 3.0]}

        with patch.object(self.scheduler._anomaly_service, "learn_baseline") as mock_baseline, \
             patch("services.anomaly.training_scheduler.safe_ml_operation") as mock_safe_ml, \
             patch("services.anomaly.training_scheduler.logger") as mock_logger:

            mock_baseline.return_value = False
            mock_safe_ml.return_value = True

            await self.scheduler._execute_training_job(job_data)

            # Sollte trotzdem LSTM-Training versuchen
            mock_safe_ml.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Training-Job abgeschlossen für tenant1:cpu (Baseline: False, LSTM: True)"
            )

    @pytest.mark.asyncio
    async def test_execute_training_job_lstm_failed(self):
        """Testet _execute_training_job wenn LSTM-Training fehlschlägt."""
        job_data = {"tenant": "tenant1", "metric": "cpu", "values": [1.0, 2.0, 3.0]}

        with patch.object(self.scheduler._anomaly_service, "learn_baseline") as mock_baseline, \
             patch("services.anomaly.training_scheduler.safe_ml_operation") as mock_safe_ml, \
             patch("services.anomaly.training_scheduler.logger") as mock_logger:

            mock_baseline.return_value = True
            mock_safe_ml.return_value = False

            await self.scheduler._execute_training_job(job_data)

            mock_logger.info.assert_called_once_with(
                "Training-Job abgeschlossen für tenant1:cpu (Baseline: True, LSTM: False)"
            )

    @pytest.mark.asyncio
    async def test_execute_training_job_missing_fields(self):
        """Testet _execute_training_job mit fehlenden Feldern aber gültigen Werten."""
        job_data = {"metric": "cpu", "values": [1.0, 2.0]}  # tenant fehlt, aber values vorhanden

        with patch.object(self.scheduler._anomaly_service, "learn_baseline") as mock_baseline, \
             patch("services.anomaly.training_scheduler.safe_ml_operation") as mock_safe_ml, \
             patch("services.anomaly.training_scheduler.logger"):

            mock_baseline.return_value = True
            mock_safe_ml.return_value = True

            await self.scheduler._execute_training_job(job_data)

            # Sollte mit Default-Werten arbeiten
            mock_baseline.assert_called_once_with(
                tenant="default",  # Default-Wert
                metric_name="cpu",
                values=[1.0, 2.0]
            )

    @pytest.mark.asyncio
    async def test_enqueue_training_success(self):
        """Testet enqueue_training bei erfolgreichem Aufruf."""
        with patch("services.anomaly.training_scheduler.redis_helper") as mock_redis:
            mock_redis.enqueue_training_job = AsyncMock(return_value=True)

            result = await self.scheduler.enqueue_training("tenant1", "cpu", [1.0, 2.0])

            assert result is True
            mock_redis.enqueue_training_job.assert_called_once_with(
                "tenant1", "cpu", [1.0, 2.0]
            )

    @pytest.mark.asyncio
    async def test_enqueue_training_failed(self):
        """Testet enqueue_training bei fehlgeschlagenem Aufruf."""
        with patch("services.anomaly.training_scheduler.redis_helper") as mock_redis:
            mock_redis.enqueue_training_job = AsyncMock(return_value=False)

            result = await self.scheduler.enqueue_training("tenant1", "cpu", [1.0, 2.0])

            assert result is False


class TestTrainingSchedulerIntegration:
    """Integrationstests für AnomalyTrainingScheduler."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.scheduler = AnomalyTrainingScheduler()

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Testet kompletten Lebenszyklus des Schedulers."""
        with patch("services.anomaly.training_scheduler.settings") as mock_settings:
            mock_settings.anomaly_training_enabled = True
            mock_settings.anomaly_training_interval_minutes = 1

            # Mock _training_loop um sofort zu stoppen
            async def mock_training_loop():
                self.scheduler._running = False

            with patch.object(self.scheduler, "_training_loop", side_effect=mock_training_loop):
                # Start
                await self.scheduler.start()
                assert self.scheduler._running is True
                assert self.scheduler._task is not None

                # Warte auf Task-Completion
                await self.scheduler._task

                # Stop
                await self.scheduler.stop()
                assert self.scheduler._running is False
                assert self.scheduler._task is None
