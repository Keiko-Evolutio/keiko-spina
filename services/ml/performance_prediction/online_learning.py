# backend/services/ml/performance_prediction/online_learning.py
"""Online Learning Pipeline für kontinuierliche Model-Updates.

Implementiert Online-Learning mit neuen Performance-Daten,
Model-Drift-Detection und automatisches Re-Training.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from kei_logging import get_logger

from .data_models import OnlineLearningUpdate, PerformanceDataPoint
from .model_trainer import ModelTrainer
from .performance_predictor import PerformancePredictor

logger = get_logger(__name__)


class OnlineLearningPipeline:
    """Online Learning Pipeline für kontinuierliche Model-Verbesserung."""

    def __init__(
        self,
        model_trainer: ModelTrainer,
        performance_predictor: PerformancePredictor
    ):
        """Initialisiert Online Learning Pipeline.

        Args:
            model_trainer: Model Trainer für Re-Training
            performance_predictor: Performance Predictor für Feedback
        """
        self.model_trainer = model_trainer
        self.performance_predictor = performance_predictor

        # Feedback-Queue für Online Learning
        self.feedback_queue: deque[OnlineLearningUpdate] = deque(maxlen=10000)

        # Drift-Detection-Parameter
        self.drift_detection_window = 100  # Anzahl Predictions für Drift-Check
        self.drift_threshold = 0.2  # 20% Verschlechterung triggert Re-Training
        self.min_feedback_for_retraining = 50  # Minimum Feedback für Re-Training

        # Re-Training-Parameter
        self.auto_retrain_enabled = True
        self.retrain_interval_hours = 24  # Automatisches Re-Training alle 24h
        self.last_retrain_time = datetime.utcnow()

        # Performance-Tracking
        self.recent_errors: deque[float] = deque(maxlen=self.drift_detection_window)
        self.baseline_mae = None  # Baseline MAE für Drift-Detection

        # Online Learning Task
        self._learning_task: asyncio.Task | None = None
        self._running = False

        logger.info({
            "event": "online_learning_initialized",
            "drift_detection_window": self.drift_detection_window,
            "drift_threshold": self.drift_threshold,
            "auto_retrain_enabled": self.auto_retrain_enabled
        })

    async def start_online_learning(self) -> None:
        """Startet Online Learning Pipeline."""
        if self._running:
            logger.warning("Online Learning bereits gestartet")
            return

        self._running = True
        self._learning_task = asyncio.create_task(self._online_learning_loop())

        logger.info("Online Learning Pipeline gestartet")

    async def stop_online_learning(self) -> None:
        """Stoppt Online Learning Pipeline."""
        self._running = False

        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass

        logger.info("Online Learning Pipeline gestoppt")

    async def add_feedback(self, feedback: OnlineLearningUpdate) -> None:
        """Fügt Feedback für Online Learning hinzu.

        Args:
            feedback: Performance-Feedback für Prediction
        """
        self.feedback_queue.append(feedback)

        # Update Error-Tracking für Drift-Detection
        self.recent_errors.append(feedback.absolute_error)

        logger.debug({
            "event": "feedback_added",
            "prediction_error": feedback.prediction_error,
            "absolute_error": feedback.absolute_error,
            "percentage_error": feedback.percentage_error,
            "queue_size": len(self.feedback_queue)
        })

        # Prüfe auf Model-Drift
        await self._check_model_drift()

    async def _online_learning_loop(self) -> None:
        """Haupt-Loop für Online Learning."""
        while self._running:
            try:
                # Prüfe auf automatisches Re-Training
                await self._check_auto_retrain()

                # Verarbeite Feedback-Queue
                await self._process_feedback_queue()

                # Warte vor nächster Iteration
                await asyncio.sleep(60)  # 1 Minute

            except Exception as e:
                logger.error(f"Fehler in Online Learning Loop: {e}")
                await asyncio.sleep(60)

    async def _check_model_drift(self) -> None:
        """Prüft auf Model-Drift und triggert Re-Training falls nötig."""
        if len(self.recent_errors) < self.drift_detection_window:
            return

        # Berechne aktuelle MAE
        current_mae = np.mean(list(self.recent_errors))

        # Setze Baseline falls noch nicht gesetzt
        if self.baseline_mae is None:
            self.baseline_mae = current_mae
            logger.info(f"Baseline MAE gesetzt: {self.baseline_mae:.2f}")
            return

        # Prüfe auf Drift
        mae_increase = (current_mae - self.baseline_mae) / self.baseline_mae

        if mae_increase > self.drift_threshold:
            logger.warning({
                "event": "model_drift_detected",
                "baseline_mae": self.baseline_mae,
                "current_mae": current_mae,
                "mae_increase": mae_increase,
                "threshold": self.drift_threshold
            })

            # Triggere Re-Training
            await self._trigger_retraining("model_drift")

    async def _check_auto_retrain(self) -> None:
        """Prüft auf automatisches Re-Training basierend auf Zeit."""
        if not self.auto_retrain_enabled:
            return

        time_since_last_retrain = datetime.utcnow() - self.last_retrain_time

        if time_since_last_retrain.total_seconds() > self.retrain_interval_hours * 3600:
            if len(self.feedback_queue) >= self.min_feedback_for_retraining:
                await self._trigger_retraining("scheduled")

    async def _trigger_retraining(self, reason: str) -> None:
        """Triggert Model-Re-Training."""
        logger.info({
            "event": "retraining_triggered",
            "reason": reason,
            "feedback_count": len(self.feedback_queue)
        })

        try:
            # Konvertiere Feedback zu Training-Daten
            training_data = self._convert_feedback_to_training_data()

            if len(training_data) < self.min_feedback_for_retraining:
                logger.warning(f"Nicht genug Training-Daten: {len(training_data)}")
                return

            # Re-Training
            await self._retrain_models(training_data)

            # Update Baseline und Reset
            self.last_retrain_time = datetime.utcnow()
            self.baseline_mae = None  # Reset für neue Baseline

            logger.info({
                "event": "retraining_completed",
                "training_data_size": len(training_data),
                "reason": reason
            })

        except Exception as e:
            logger.error(f"Fehler beim Re-Training: {e}")

    def _convert_feedback_to_training_data(self) -> list[PerformanceDataPoint]:
        """Konvertiert Feedback zu Training-Daten."""
        training_data = []

        for feedback in self.feedback_queue:
            # Erstelle PerformanceDataPoint aus Feedback
            data_point = PerformanceDataPoint(
                execution_id=f"feedback_{int(time.time())}_{len(training_data)}",
                task_id=f"feedback_task_{len(training_data)}",
                agent_id=feedback.prediction_request.agent_characteristics.agent_id,
                timestamp=feedback.feedback_timestamp,
                task_characteristics=feedback.prediction_request.task_characteristics,
                agent_characteristics=feedback.prediction_request.agent_characteristics,
                system_load=feedback.prediction_request.system_load,
                concurrent_executions=feedback.prediction_request.concurrent_executions,
                time_of_day_hour=feedback.feedback_timestamp.hour,
                day_of_week=feedback.feedback_timestamp.weekday(),
                actual_execution_time_ms=feedback.actual_execution_time_ms,
                success=feedback.actual_success,
                error_type=feedback.actual_error_type
            )

            training_data.append(data_point)

        return training_data

    async def _retrain_models(self, training_data: list[PerformanceDataPoint]) -> None:
        """Re-Training der ML-Modelle."""
        # Bestimme welche Modelle re-trainiert werden sollen
        active_model_info = self.model_trainer.get_active_model()

        if active_model_info:
            model_id, model, metadata = active_model_info
            model_types = [metadata.model_type]
        else:
            # Trainiere alle verfügbaren Modelle
            model_types = ["random_forest", "xgboost"] if hasattr(self.model_trainer, "available_models") else ["random_forest"]

        # Re-Training
        new_metadata = await self.model_trainer.train_models(
            data_points=training_data,
            model_types=model_types,
            enable_hyperparameter_tuning=False,  # Schnelleres Training
            test_size=0.2,
            cv_folds=3  # Weniger Folds für Geschwindigkeit
        )

        # Aktiviere bestes neues Modell
        if new_metadata:
            best_model_type = max(
                new_metadata.items(),
                key=lambda x: x[1].r2_score
            )[0]

            # Finde Model-ID für besten Typ
            for model_id, metadata in self.model_trainer.model_metadata.items():
                if metadata.model_type == best_model_type and not metadata.is_active:
                    self.model_trainer.activate_model(model_id)
                    break

    async def _process_feedback_queue(self) -> None:
        """Verarbeitet Feedback-Queue für kontinuierliches Learning."""
        if len(self.feedback_queue) == 0:
            return

        # Analysiere Feedback-Patterns
        recent_feedback = list(self.feedback_queue)[-100:]  # Letzte 100 Feedback-Items

        # Berechne Metriken
        absolute_errors = [f.absolute_error for f in recent_feedback]
        percentage_errors = [f.percentage_error for f in recent_feedback]

        avg_absolute_error = np.mean(absolute_errors)
        avg_percentage_error = np.mean(percentage_errors)

        # Identifiziere problematische Patterns
        await self._analyze_error_patterns(recent_feedback)

        logger.debug({
            "event": "feedback_queue_processed",
            "feedback_count": len(recent_feedback),
            "avg_absolute_error": avg_absolute_error,
            "avg_percentage_error": avg_percentage_error
        })

    async def _analyze_error_patterns(self, feedback_list: list[OnlineLearningUpdate]) -> None:
        """Analysiert Fehler-Patterns für Insights."""
        # Gruppiere nach Agent-ID
        agent_errors = {}
        for feedback in feedback_list:
            agent_id = feedback.prediction_request.agent_characteristics.agent_id
            if agent_id not in agent_errors:
                agent_errors[agent_id] = []
            agent_errors[agent_id].append(feedback.percentage_error)

        # Identifiziere Agents mit hohen Fehlern
        problematic_agents = []
        for agent_id, errors in agent_errors.items():
            avg_error = np.mean(errors)
            if avg_error > 0.5:  # 50% Fehler
                problematic_agents.append((agent_id, avg_error))

        if problematic_agents:
            logger.warning({
                "event": "problematic_agents_detected",
                "agents": problematic_agents
            })

        # Gruppiere nach Task-Type
        task_errors = {}
        for feedback in feedback_list:
            task_type = feedback.prediction_request.task_characteristics.task_type
            if task_type not in task_errors:
                task_errors[task_type] = []
            task_errors[task_type].append(feedback.percentage_error)

        # Identifiziere Task-Types mit hohen Fehlern
        problematic_tasks = []
        for task_type, errors in task_errors.items():
            avg_error = np.mean(errors)
            if avg_error > 0.4:  # 40% Fehler
                problematic_tasks.append((task_type, avg_error))

        if problematic_tasks:
            logger.warning({
                "event": "problematic_task_types_detected",
                "task_types": problematic_tasks
            })

    def get_learning_statistics(self) -> dict[str, Any]:
        """Gibt Online Learning Statistiken zurück."""
        if not self.feedback_queue:
            return {"feedback_count": 0}

        # Berechne Statistiken
        recent_feedback = list(self.feedback_queue)[-100:]
        absolute_errors = [f.absolute_error for f in recent_feedback]
        percentage_errors = [f.percentage_error for f in recent_feedback]

        return {
            "feedback_count": len(self.feedback_queue),
            "recent_feedback_count": len(recent_feedback),
            "avg_absolute_error": np.mean(absolute_errors) if absolute_errors else 0.0,
            "avg_percentage_error": np.mean(percentage_errors) if percentage_errors else 0.0,
            "baseline_mae": self.baseline_mae,
            "current_mae": np.mean(list(self.recent_errors)) if self.recent_errors else 0.0,
            "last_retrain_time": self.last_retrain_time.isoformat(),
            "auto_retrain_enabled": self.auto_retrain_enabled,
            "drift_detection_active": len(self.recent_errors) >= self.drift_detection_window,
            "running": self._running
        }

    def configure_learning_parameters(
        self,
        drift_threshold: float | None = None,
        retrain_interval_hours: int | None = None,
        auto_retrain_enabled: bool | None = None
    ) -> None:
        """Konfiguriert Online Learning Parameter."""
        if drift_threshold is not None:
            self.drift_threshold = drift_threshold

        if retrain_interval_hours is not None:
            self.retrain_interval_hours = retrain_interval_hours

        if auto_retrain_enabled is not None:
            self.auto_retrain_enabled = auto_retrain_enabled

        logger.info({
            "event": "learning_parameters_updated",
            "drift_threshold": self.drift_threshold,
            "retrain_interval_hours": self.retrain_interval_hours,
            "auto_retrain_enabled": self.auto_retrain_enabled
        })
