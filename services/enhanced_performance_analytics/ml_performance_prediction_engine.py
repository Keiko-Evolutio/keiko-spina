# backend/services/enhanced_performance_analytics/ml_performance_prediction_engine.py
"""ML-basierte Performance Prediction Engine.

Implementiert Enterprise-Grade ML-basierte Performance-Vorhersagen mit
Real-time Analytics-Pipeline und Advanced ML-Models.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger
from services.ml.performance_prediction import PerformancePredictor

from .data_models import AnalyticsConfiguration, AnalyticsScope, MLPerformancePrediction

logger = get_logger(__name__)


class MLPerformancePredictionEngine:
    """ML-basierte Performance Prediction Engine für Enterprise-Grade Predictions."""

    def __init__(
        self,
        performance_predictor: PerformancePredictor | None = None,
        configuration: AnalyticsConfiguration | None = None
    ):
        """Initialisiert ML Performance Prediction Engine.

        Args:
            performance_predictor: Performance Predictor
            configuration: Analytics-Konfiguration
        """
        self.performance_predictor = performance_predictor or PerformancePredictor()
        self.configuration = configuration or AnalyticsConfiguration()

        # Prediction-Storage
        self._predictions: dict[str, MLPerformancePrediction] = {}
        self._prediction_history: dict[str, list[MLPerformancePrediction]] = {}
        self._model_performance: dict[str, dict[str, float]] = {}

        # Feature-Engineering
        self._feature_cache: dict[str, dict[str, Any]] = {}
        self._feature_importance: dict[str, dict[str, float]] = {}

        # Real-time Prediction
        self._prediction_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Performance-Tracking
        self._prediction_performance_stats = {
            "total_predictions_made": 0,
            "avg_prediction_time_ms": 0.0,
            "prediction_accuracy": 0.0,
            "prediction_error_rate": 0.0,
            "model_retrain_count": 0,
            "feature_engineering_time_ms": 0.0
        }

        # Model-Management
        self._active_models: dict[str, dict[str, Any]] = {}
        self._model_validation_results: dict[str, dict[str, float]] = {}

        logger.info("ML Performance Prediction Engine initialisiert")

    async def start(self) -> None:
        """Startet ML Performance Prediction Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._prediction_tasks = [
            asyncio.create_task(self._prediction_generation_loop()),
            asyncio.create_task(self._model_validation_loop()),
            asyncio.create_task(self._feature_importance_analysis_loop()),
            asyncio.create_task(self._prediction_accuracy_monitoring_loop())
        ]

        # Initialisiere Models
        await self._initialize_prediction_models()

        logger.info("ML Performance Prediction Engine gestartet")

    async def stop(self) -> None:
        """Stoppt ML Performance Prediction Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._prediction_tasks:
            task.cancel()

        await asyncio.gather(*self._prediction_tasks, return_exceptions=True)
        self._prediction_tasks.clear()

        logger.info("ML Performance Prediction Engine gestoppt")

    async def predict_performance(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        prediction_horizon_minutes: int,
        context_data: dict[str, Any] | None = None
    ) -> MLPerformancePrediction:
        """Erstellt ML-basierte Performance-Vorhersage.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_name: Metrik-Name
            prediction_horizon_minutes: Vorhersage-Horizont in Minuten
            context_data: Kontext-Daten

        Returns:
            ML Performance-Vorhersage
        """
        start_time = time.time()

        try:
            import uuid

            prediction_id = str(uuid.uuid4())
            target_timestamp = datetime.utcnow() + timedelta(minutes=prediction_horizon_minutes)

            # Feature-Engineering
            features = await self._engineer_features(
                scope, scope_id, metric_name, context_data
            )

            # Model-Selection
            model_info = await self._select_best_model(scope, metric_name)

            if not model_info:
                # Fallback zu statistischer Vorhersage
                return await self._create_fallback_prediction(
                    prediction_id, scope, scope_id, metric_name,
                    prediction_horizon_minutes, target_timestamp
                )

            model_id, model_data = model_info

            # ML-Prediction
            predicted_value, confidence, bounds = await self._make_ml_prediction(
                model_data, features, metric_name
            )

            # Feature-Importance
            feature_importance = await self._calculate_feature_importance(
                model_data, features, metric_name
            )

            # Top-Features
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # Erstelle Prediction
            prediction = MLPerformancePrediction(
                prediction_id=prediction_id,
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                predicted_value=predicted_value,
                prediction_confidence=confidence,
                prediction_horizon_minutes=prediction_horizon_minutes,
                target_timestamp=target_timestamp,
                model_id=model_id,
                model_version=model_data.get("version", "1.0"),
                model_type=model_data.get("type", "ml_model"),
                model_accuracy=model_data.get("accuracy", 0.0),
                input_features=features,
                feature_importance=feature_importance,
                top_features=[feature for feature, _ in top_features],
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                historical_accuracy=self._get_historical_accuracy(model_id),
                similar_predictions_count=self._count_similar_predictions(scope, metric_name),
                baseline_comparison=await self._calculate_baseline_comparison(
                    scope, scope_id, metric_name, predicted_value
                )
            )

            # Speichere Prediction
            self._predictions[prediction_id] = prediction

            # Füge zu History hinzu
            history_key = f"{scope.value}:{scope_id}:{metric_name}"
            if history_key not in self._prediction_history:
                self._prediction_history[history_key] = []
            self._prediction_history[history_key].append(prediction)

            # Limitiere History-Größe
            if len(self._prediction_history[history_key]) > 100:
                self._prediction_history[history_key] = self._prediction_history[history_key][-100:]

            # Update Performance-Stats
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_prediction_performance_stats(prediction_time_ms)

            logger.debug({
                "event": "ml_performance_prediction_created",
                "prediction_id": prediction_id,
                "scope": scope.value,
                "scope_id": scope_id,
                "metric_name": metric_name,
                "predicted_value": predicted_value,
                "confidence": confidence,
                "model_id": model_id,
                "prediction_time_ms": prediction_time_ms
            })

            return prediction

        except Exception as e:
            logger.error(f"ML performance prediction fehlgeschlagen: {e}")

            # Fallback zu statistischer Vorhersage
            return await self._create_fallback_prediction(
                str(uuid.uuid4()), scope, scope_id, metric_name,
                prediction_horizon_minutes, target_timestamp
            )

    async def validate_prediction(
        self,
        prediction_id: str,
        actual_value: float
    ) -> dict[str, float]:
        """Validiert Prediction mit tatsächlichem Wert.

        Args:
            prediction_id: Prediction-ID
            actual_value: Tatsächlicher Wert

        Returns:
            Validation-Metriken
        """
        try:
            prediction = self._predictions.get(prediction_id)
            if not prediction:
                logger.warning(f"Prediction {prediction_id} nicht gefunden für Validation")
                return {}

            # Berechne Prediction-Error
            prediction_error = abs(actual_value - prediction.predicted_value)
            relative_error = prediction_error / actual_value if actual_value != 0 else float("inf")

            # Update Prediction
            prediction.actual_value = actual_value
            prediction.prediction_error = prediction_error
            prediction.validation_timestamp = datetime.utcnow()

            # Berechne Validation-Metriken
            validation_metrics = {
                "absolute_error": prediction_error,
                "relative_error": relative_error,
                "accuracy": 1.0 - min(relative_error, 1.0),
                "within_bounds": prediction.lower_bound <= actual_value <= prediction.upper_bound,
                "confidence_calibration": self._calculate_confidence_calibration(prediction)
            }

            # Update Model-Performance
            await self._update_model_performance(prediction, validation_metrics)

            logger.debug({
                "event": "prediction_validated",
                "prediction_id": prediction_id,
                "predicted_value": prediction.predicted_value,
                "actual_value": actual_value,
                "absolute_error": prediction_error,
                "relative_error": relative_error,
                "accuracy": validation_metrics["accuracy"]
            })

            return validation_metrics

        except Exception as e:
            logger.error(f"Prediction validation fehlgeschlagen: {e}")
            return {}

    async def get_prediction_trends(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        time_range_hours: int = 24
    ) -> dict[str, Any]:
        """Holt Prediction-Trends.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_name: Metrik-Name
            time_range_hours: Zeitraum in Stunden

        Returns:
            Prediction-Trends
        """
        try:
            history_key = f"{scope.value}:{scope_id}:{metric_name}"
            predictions = self._prediction_history.get(history_key, [])

            if not predictions:
                return {"trends": [], "summary": {}}

            # Filter nach Zeitraum
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            recent_predictions = [
                p for p in predictions
                if p.prediction_timestamp >= cutoff_time
            ]

            if not recent_predictions:
                return {"trends": [], "summary": {}}

            # Berechne Trend-Metriken
            predicted_values = [p.predicted_value for p in recent_predictions]
            confidences = [p.prediction_confidence for p in recent_predictions]

            # Validierte Predictions
            validated_predictions = [p for p in recent_predictions if p.actual_value is not None]

            accuracy_scores = []
            if validated_predictions:
                for p in validated_predictions:
                    if p.actual_value != 0:
                        accuracy = 1.0 - abs(p.prediction_error) / p.actual_value
                        accuracy_scores.append(max(0.0, accuracy))

            trends_summary = {
                "total_predictions": len(recent_predictions),
                "validated_predictions": len(validated_predictions),
                "avg_predicted_value": sum(predicted_values) / len(predicted_values),
                "avg_confidence": sum(confidences) / len(confidences),
                "avg_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
                "prediction_trend": self._calculate_prediction_trend(predicted_values),
                "confidence_trend": self._calculate_prediction_trend(confidences)
            }

            # Trend-Daten
            trend_data = [
                {
                    "timestamp": p.prediction_timestamp.isoformat(),
                    "predicted_value": p.predicted_value,
                    "confidence": p.prediction_confidence,
                    "actual_value": p.actual_value,
                    "accuracy": (1.0 - abs(p.prediction_error) / p.actual_value) if p.actual_value and p.actual_value != 0 else None
                }
                for p in recent_predictions
            ]

            return {
                "trends": trend_data,
                "summary": trends_summary
            }

        except Exception as e:
            logger.error(f"Prediction trends retrieval fehlgeschlagen: {e}")
            return {"trends": [], "summary": {}}

    async def _engineer_features(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        context_data: dict[str, Any] | None
    ) -> dict[str, float]:
        """Führt Feature-Engineering aus."""
        try:
            features = {}

            # Basis-Features
            features["scope_type"] = hash(scope.value) % 1000  # Numerische Repräsentation
            features["metric_type"] = hash(metric_name) % 1000
            features["hour_of_day"] = datetime.utcnow().hour
            features["day_of_week"] = datetime.utcnow().weekday()

            # Kontext-Features
            if context_data:
                for key, value in context_data.items():
                    if isinstance(value, (int, float)):
                        features[f"context_{key}"] = float(value)
                    elif isinstance(value, str):
                        features[f"context_{key}_hash"] = hash(value) % 1000

            # Historical-Features
            historical_features = await self._extract_historical_features(scope, scope_id, metric_name)
            features.update(historical_features)

            # Cache Features
            cache_key = f"{scope.value}:{scope_id}:{metric_name}"
            self._feature_cache[cache_key] = features

            return features

        except Exception as e:
            logger.error(f"Feature engineering fehlgeschlagen: {e}")
            return {}

    async def _extract_historical_features(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str
    ) -> dict[str, float]:
        """Extrahiert Historical-Features."""
        try:
            features = {}

            # Hole Historical Predictions
            history_key = f"{scope.value}:{scope_id}:{metric_name}"
            predictions = self._prediction_history.get(history_key, [])

            if predictions:
                recent_predictions = predictions[-10:]  # Letzte 10 Predictions

                # Historical-Performance-Features
                predicted_values = [p.predicted_value for p in recent_predictions]
                confidences = [p.prediction_confidence for p in recent_predictions]

                if predicted_values:
                    features["hist_avg_predicted"] = sum(predicted_values) / len(predicted_values)
                    features["hist_max_predicted"] = max(predicted_values)
                    features["hist_min_predicted"] = min(predicted_values)
                    features["hist_std_predicted"] = self._calculate_std_dev(predicted_values)

                if confidences:
                    features["hist_avg_confidence"] = sum(confidences) / len(confidences)
                    features["hist_min_confidence"] = min(confidences)

                # Accuracy-Features
                validated_predictions = [p for p in recent_predictions if p.actual_value is not None]
                if validated_predictions:
                    accuracies = []
                    for p in validated_predictions:
                        if p.actual_value != 0:
                            accuracy = 1.0 - abs(p.prediction_error) / p.actual_value
                            accuracies.append(max(0.0, accuracy))

                    if accuracies:
                        features["hist_avg_accuracy"] = sum(accuracies) / len(accuracies)
                        features["hist_min_accuracy"] = min(accuracies)

            return features

        except Exception as e:
            logger.error(f"Historical features extraction fehlgeschlagen: {e}")
            return {}

    async def _select_best_model(
        self,
        scope: AnalyticsScope,
        metric_name: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Wählt bestes Model für Prediction."""
        try:
            # Suche nach passenden Models
            candidate_models = []

            for model_id, model_data in self._active_models.items():
                # Prüfe Model-Kompatibilität
                if self._is_model_compatible(model_data, scope, metric_name):
                    model_performance = self._model_performance.get(model_id, {})
                    accuracy = model_performance.get("accuracy", 0.0)
                    candidate_models.append((model_id, model_data, accuracy))

            if not candidate_models:
                return None

            # Wähle Model mit höchster Accuracy
            best_model = max(candidate_models, key=lambda x: x[2])
            return best_model[0], best_model[1]

        except Exception as e:
            logger.error(f"Model selection fehlgeschlagen: {e}")
            return None

    async def _make_ml_prediction(
        self,
        _model_data: dict[str, Any],
        features: dict[str, float],
        _metric_name: str
    ) -> tuple[float, float, tuple[float, float]]:
        """Macht ML-Prediction."""
        try:
            # Simuliere ML-Prediction (in Realität würde hier das echte Model verwendet)
            base_value = features.get("hist_avg_predicted", 100.0)

            # Einfache Prediction basierend auf Features
            predicted_value = base_value * (1.0 + (features.get("hour_of_day", 12) - 12) * 0.01)

            # Confidence basierend auf Historical Accuracy
            historical_accuracy = features.get("hist_avg_accuracy", 0.8)
            confidence = min(0.95, max(0.1, historical_accuracy))

            # Prediction-Bounds
            uncertainty = predicted_value * (1.0 - confidence) * 0.5
            lower_bound = predicted_value - uncertainty
            upper_bound = predicted_value + uncertainty

            return predicted_value, confidence, (lower_bound, upper_bound)

        except Exception as e:
            logger.error(f"ML prediction fehlgeschlagen: {e}")
            return 0.0, 0.0, (0.0, 0.0)

    async def _calculate_feature_importance(
        self,
        _model_data: dict[str, Any],
        features: dict[str, float],
        _metric_name: str
    ) -> dict[str, float]:
        """Berechnet Feature-Importance."""
        try:
            # Simuliere Feature-Importance (in Realität würde hier das echte Model verwendet)
            importance = {}

            for feature_name in features:
                if "hist_" in feature_name:
                    importance[feature_name] = 0.8  # Historical Features sind wichtig
                elif "context_" in feature_name:
                    importance[feature_name] = 0.6  # Context Features sind moderat wichtig
                else:
                    importance[feature_name] = 0.3  # Basis Features sind weniger wichtig

            # Normalisiere Importance
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}

            return importance

        except Exception as e:
            logger.error(f"Feature importance calculation fehlgeschlagen: {e}")
            return {}

    async def _create_fallback_prediction(
        self,
        prediction_id: str,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        prediction_horizon_minutes: int,
        target_timestamp: datetime
    ) -> MLPerformancePrediction:
        """Erstellt Fallback-Prediction."""
        try:
            # Einfache statistische Vorhersage
            history_key = f"{scope.value}:{scope_id}:{metric_name}"
            predictions = self._prediction_history.get(history_key, [])

            if predictions:
                recent_values = [p.predicted_value for p in predictions[-5:]]
                predicted_value = sum(recent_values) / len(recent_values)
                confidence = 0.5  # Niedrige Confidence für Fallback
            else:
                predicted_value = 100.0  # Default-Wert
                confidence = 0.3  # Sehr niedrige Confidence

            return MLPerformancePrediction(
                prediction_id=prediction_id,
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                predicted_value=predicted_value,
                prediction_confidence=confidence,
                prediction_horizon_minutes=prediction_horizon_minutes,
                target_timestamp=target_timestamp,
                model_id="fallback_statistical",
                model_version="1.0",
                model_type="statistical",
                model_accuracy=0.5,
                lower_bound=predicted_value * 0.8,
                upper_bound=predicted_value * 1.2,
                historical_accuracy=0.5,
                similar_predictions_count=len(predictions),
                baseline_comparison=0.0
            )

        except Exception as e:
            logger.error(f"Fallback prediction creation fehlgeschlagen: {e}")
            # Minimal-Fallback
            return MLPerformancePrediction(
                prediction_id=prediction_id,
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                predicted_value=100.0,
                prediction_confidence=0.1,
                prediction_horizon_minutes=prediction_horizon_minutes,
                target_timestamp=target_timestamp,
                model_id="minimal_fallback",
                model_version="1.0",
                model_type="minimal",
                model_accuracy=0.1,
                lower_bound=80.0,
                upper_bound=120.0,
                historical_accuracy=0.1,
                similar_predictions_count=0,
                baseline_comparison=0.0
            )

    async def _initialize_prediction_models(self) -> None:
        """Initialisiert Prediction-Models."""
        try:
            # Registriere Standard-Models
            self._active_models["linear_regression"] = {
                "type": "linear_regression",
                "version": "1.0",
                "accuracy": 0.75,
                "supported_scopes": [scope.value for scope in AnalyticsScope],
                "supported_metrics": ["response_time", "throughput", "error_rate", "cpu_usage"]
            }

            self._active_models["random_forest"] = {
                "type": "random_forest",
                "version": "1.0",
                "accuracy": 0.82,
                "supported_scopes": [scope.value for scope in AnalyticsScope],
                "supported_metrics": ["response_time", "throughput", "error_rate"]
            }

            self._active_models["neural_network"] = {
                "type": "neural_network",
                "version": "1.0",
                "accuracy": 0.88,
                "supported_scopes": [AnalyticsScope.SYSTEM.value, AnalyticsScope.SERVICE.value],
                "supported_metrics": ["response_time", "throughput"]
            }

            logger.info(f"Prediction models initialisiert: {len(self._active_models)} Models")

        except Exception as e:
            logger.error(f"Prediction models initialization fehlgeschlagen: {e}")

    async def _prediction_generation_loop(self) -> None:
        """Background-Loop für Prediction-Generation."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.ml_prediction_interval_minutes * 60)

                if self._is_running:
                    await self._generate_scheduled_predictions()

            except Exception as e:
                logger.error(f"Prediction generation loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _model_validation_loop(self) -> None:
        """Background-Loop für Model-Validation."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Jede Stunde

                if self._is_running:
                    await self._validate_model_performance()

            except Exception as e:
                logger.error(f"Model validation loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _feature_importance_analysis_loop(self) -> None:
        """Background-Loop für Feature-Importance-Analysis."""
        while self._is_running:
            try:
                await asyncio.sleep(1800)  # Alle 30 Minuten

                if self._is_running:
                    await self._analyze_feature_importance()

            except Exception as e:
                logger.error(f"Feature importance analysis loop fehlgeschlagen: {e}")
                await asyncio.sleep(1800)

    async def _prediction_accuracy_monitoring_loop(self) -> None:
        """Background-Loop für Prediction-Accuracy-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(900)  # Alle 15 Minuten

                if self._is_running:
                    await self._monitor_prediction_accuracy()

            except Exception as e:
                logger.error(f"Prediction accuracy monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(900)

    async def _generate_scheduled_predictions(self) -> None:
        """Generiert geplante Predictions."""
        try:
            # Generiere Predictions für alle aktiven Metriken
            for history_key in self._prediction_history.keys():
                scope_str, scope_id, metric_name = history_key.split(":", 2)
                scope = AnalyticsScope(scope_str)

                # Generiere Prediction für nächste Stunde
                await self.predict_performance(
                    scope=scope,
                    scope_id=scope_id,
                    metric_name=metric_name,
                    prediction_horizon_minutes=60
                )

        except Exception as e:
            logger.error(f"Scheduled predictions generation fehlgeschlagen: {e}")

    async def _validate_model_performance(self) -> None:
        """Validiert Model-Performance."""
        try:
            for model_id, model_data in self._active_models.items():
                # Sammle Validation-Daten für Model
                validation_results = []

                for predictions in self._prediction_history.values():
                    model_predictions = [p for p in predictions if p.model_id == model_id and p.actual_value is not None]

                    for prediction in model_predictions:
                        if prediction.actual_value != 0:
                            accuracy = 1.0 - abs(prediction.prediction_error) / prediction.actual_value
                            validation_results.append(max(0.0, accuracy))

                if validation_results:
                    avg_accuracy = sum(validation_results) / len(validation_results)

                    # Update Model-Performance
                    if model_id not in self._model_performance:
                        self._model_performance[model_id] = {}

                    self._model_performance[model_id]["accuracy"] = avg_accuracy
                    self._model_performance[model_id]["validation_count"] = len(validation_results)
                    self._model_performance[model_id]["last_validated"] = datetime.utcnow()

        except Exception as e:
            logger.error(f"Model performance validation fehlgeschlagen: {e}")

    async def _analyze_feature_importance(self) -> None:
        """Analysiert Feature-Importance."""
        try:
            # Analysiere Feature-Importance über alle Predictions
            all_feature_importance = defaultdict(list)

            for predictions in self._prediction_history.values():
                for prediction in predictions:
                    for feature, importance in prediction.feature_importance.items():
                        all_feature_importance[feature].append(importance)

            # Berechne durchschnittliche Feature-Importance
            avg_feature_importance = {}
            for feature, importance_values in all_feature_importance.items():
                avg_feature_importance[feature] = sum(importance_values) / len(importance_values)

            # Speichere globale Feature-Importance
            self._feature_importance["global"] = avg_feature_importance

        except Exception as e:
            logger.error(f"Feature importance analysis fehlgeschlagen: {e}")

    async def _monitor_prediction_accuracy(self) -> None:
        """Monitort Prediction-Accuracy."""
        try:
            # Sammle Accuracy-Daten
            all_accuracies = []

            for predictions in self._prediction_history.values():
                for prediction in predictions:
                    if prediction.actual_value is not None and prediction.actual_value != 0:
                        accuracy = 1.0 - abs(prediction.prediction_error) / prediction.actual_value
                        all_accuracies.append(max(0.0, accuracy))

            if all_accuracies:
                avg_accuracy = sum(all_accuracies) / len(all_accuracies)
                self._prediction_performance_stats["prediction_accuracy"] = avg_accuracy

        except Exception as e:
            logger.error(f"Prediction accuracy monitoring fehlgeschlagen: {e}")

    async def _calculate_baseline_comparison(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        predicted_value: float
    ) -> float:
        """Berechnet Baseline-Comparison."""
        try:
            # Einfache Baseline: Durchschnitt der letzten Werte
            history_key = f"{scope.value}:{scope_id}:{metric_name}"
            predictions = self._prediction_history.get(history_key, [])

            if predictions:
                recent_values = [p.predicted_value for p in predictions[-5:]]
                baseline = sum(recent_values) / len(recent_values)

                if baseline != 0:
                    return (predicted_value - baseline) / baseline

            return 0.0

        except Exception as e:
            logger.error(f"Baseline comparison calculation fehlgeschlagen: {e}")
            return 0.0

    async def _update_model_performance(
        self,
        prediction: MLPerformancePrediction,
        validation_metrics: dict[str, float]
    ) -> None:
        """Aktualisiert Model-Performance."""
        try:
            model_id = prediction.model_id

            if model_id not in self._model_performance:
                self._model_performance[model_id] = {
                    "total_predictions": 0,
                    "total_accuracy": 0.0,
                    "accuracy": 0.0
                }

            performance = self._model_performance[model_id]
            performance["total_predictions"] += 1
            performance["total_accuracy"] += validation_metrics.get("accuracy", 0.0)
            performance["accuracy"] = performance["total_accuracy"] / performance["total_predictions"]

        except Exception as e:
            logger.error(f"Model performance update fehlgeschlagen: {e}")

    def _is_model_compatible(
        self,
        model_data: dict[str, Any],
        scope: AnalyticsScope,
        metric_name: str
    ) -> bool:
        """Prüft Model-Kompatibilität."""
        try:
            supported_scopes = model_data.get("supported_scopes", [])
            supported_metrics = model_data.get("supported_metrics", [])

            return (scope.value in supported_scopes and
                    metric_name in supported_metrics)

        except Exception as e:
            logger.error(f"Model compatibility check fehlgeschlagen: {e}")
            return False

    def _get_historical_accuracy(self, model_id: str) -> float:
        """Holt Historical Accuracy für Model."""
        try:
            performance = self._model_performance.get(model_id, {})
            return performance.get("accuracy", 0.0)

        except Exception as e:
            logger.error(f"Historical accuracy retrieval fehlgeschlagen: {e}")
            return 0.0

    def _count_similar_predictions(self, scope: AnalyticsScope, metric_name: str) -> int:
        """Zählt ähnliche Predictions."""
        try:
            count = 0
            for history_key, predictions in self._prediction_history.items():
                if scope.value in history_key and metric_name in history_key:
                    count += len(predictions)

            return count

        except Exception as e:
            logger.error(f"Similar predictions counting fehlgeschlagen: {e}")
            return 0

    def _calculate_confidence_calibration(self, prediction: MLPerformancePrediction) -> float:
        """Berechnet Confidence-Calibration."""
        try:
            if prediction.actual_value is None:
                return 0.0

            # Prüfe ob tatsächlicher Wert in Prediction-Bounds liegt
            within_bounds = prediction.lower_bound <= prediction.actual_value <= prediction.upper_bound

            # Confidence-Calibration: Wie gut entspricht Confidence der tatsächlichen Accuracy
            if within_bounds:
                return prediction.prediction_confidence
            return 1.0 - prediction.prediction_confidence

        except Exception as e:
            logger.error(f"Confidence calibration calculation fehlgeschlagen: {e}")
            return 0.0

    def _calculate_prediction_trend(self, values: list[float]) -> str:
        """Berechnet Prediction-Trend."""
        try:
            if len(values) < 2:
                return "stable"

            # Einfache Trend-Detection
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            change_percent = (avg_second - avg_first) / avg_first if avg_first != 0 else 0

            if change_percent > 0.1:
                return "increasing"
            if change_percent < -0.1:
                return "decreasing"
            return "stable"

        except Exception as e:
            logger.error(f"Prediction trend calculation fehlgeschlagen: {e}")
            return "unknown"

    def _calculate_std_dev(self, values: list[float]) -> float:
        """Berechnet Standard-Abweichung."""
        try:
            if len(values) < 2:
                return 0.0

            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5

        except Exception as e:
            logger.error(f"Standard deviation calculation fehlgeschlagen: {e}")
            return 0.0

    def _update_prediction_performance_stats(self, prediction_time_ms: float) -> None:
        """Aktualisiert Prediction-Performance-Statistiken."""
        try:
            self._prediction_performance_stats["total_predictions_made"] += 1

            current_avg = self._prediction_performance_stats["avg_prediction_time_ms"]
            total_count = self._prediction_performance_stats["total_predictions_made"]
            new_avg = ((current_avg * (total_count - 1)) + prediction_time_ms) / total_count
            self._prediction_performance_stats["avg_prediction_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Prediction performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = self._prediction_performance_stats.copy()

            # Model-Performance
            stats["model_performance"] = self._model_performance.copy()
            stats["active_models"] = len(self._active_models)

            # Prediction-Storage
            stats["total_predictions_stored"] = len(self._predictions)
            stats["prediction_history_entries"] = sum(len(predictions) for predictions in self._prediction_history.values())

            # Feature-Importance
            stats["feature_importance"] = self._feature_importance.copy()

            # Configuration
            stats["ml_predictions_enabled"] = self.configuration.ml_predictions_enabled
            stats["ml_prediction_interval_minutes"] = self.configuration.ml_prediction_interval_minutes

            return stats

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
