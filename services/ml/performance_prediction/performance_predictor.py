# backend/services/ml/performance_prediction/performance_predictor.py
"""Performance Predictor - Hauptklasse für ML-basierte Performance-Vorhersage.

Implementiert Real-time Performance-Prediction mit < 50ms Response-Zeit,
Confidence-Scoring und Fallback-Mechanismen.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np

from kei_logging import get_logger
from storage.cache.redis_cache import get_cache_client

from .data_models import PredictionRequest, PredictionResult
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer

logger = get_logger(__name__)


class PerformancePredictor:
    """ML-basierte Performance-Vorhersage für Agent-Execution-Zeit."""

    def __init__(self, model_trainer: ModelTrainer):
        """Initialisiert Performance Predictor.

        Args:
            model_trainer: Trainierter Model Trainer
        """
        self.model_trainer = model_trainer
        self.feature_engineer = FeatureEngineer()

        # Cache für Performance
        self._prediction_cache: dict[str, PredictionResult] = {}
        self._cache_ttl_seconds = 300  # 5 Minuten

        # Fallback-Statistiken
        self._fallback_stats: dict[str, dict[str, float]] = {}

        # Performance-Tracking
        self._prediction_count = 0
        self._total_prediction_time_ms = 0.0
        self._cache_hits = 0

        logger.info("Performance Predictor initialisiert")

    async def predict_execution_time(
        self,
        request: PredictionRequest
    ) -> PredictionResult:
        """Sagt Agent-Execution-Zeit vorher.

        Args:
            request: Prediction-Request

        Returns:
            Prediction-Result mit Execution-Zeit und Confidence
        """
        start_time = time.time()

        try:
            # Cache-Lookup
            cache_key = self._create_cache_key(request)
            cached_result = await self._get_cached_prediction(cache_key)
            if cached_result:
                self._cache_hits += 1
                return cached_result

            # ML-Prediction
            result = await self._make_ml_prediction(request)

            # Cache-Speicherung
            await self._cache_prediction(cache_key, result)

            # Performance-Tracking
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(prediction_time_ms)

            logger.debug({
                "event": "prediction_completed",
                "agent_id": request.agent_characteristics.agent_id,
                "task_type": request.task_characteristics.task_type,
                "predicted_time_ms": result.predicted_execution_time_ms,
                "confidence": result.confidence_score,
                "prediction_time_ms": prediction_time_ms,
                "used_fallback": result.used_fallback
            })

            return result

        except Exception as e:
            logger.error(f"Fehler bei Performance-Prediction: {e}")

            # Fallback auf statistische Schätzung
            return await self._fallback_prediction(request)

    async def _make_ml_prediction(self, request: PredictionRequest) -> PredictionResult:
        """Macht ML-basierte Prediction."""
        # Aktives Modell holen
        active_model_info = self.model_trainer.get_active_model()

        if not active_model_info:
            # Fallback wenn kein aktives Modell
            return await self._fallback_prediction(request)

        model_id, model, metadata = active_model_info

        try:
            # Features für Prediction erstellen
            features = self.feature_engineer.transform_single_prediction(
                request.task_characteristics,
                request.agent_characteristics,
                request.system_load,
                request.concurrent_executions
            )

            if len(features) == 0:
                return await self._fallback_prediction(request)

            # ML-Prediction
            features_reshaped = features.reshape(1, -1)
            predicted_time = model.predict(features_reshaped)[0]

            # Confidence-Score berechnen
            confidence_score = self._calculate_confidence_score(
                model, features_reshaped, metadata, request
            )

            # Feature-Importance für Top-Features
            top_features = self._get_top_features(features, metadata)

            return PredictionResult(
                predicted_execution_time_ms=max(0.0, float(predicted_time)),
                confidence_score=confidence_score,
                model_id=model_id,
                model_version=metadata.model_version,
                prediction_timestamp=datetime.utcnow(),
                top_features=top_features,
                used_fallback=False
            )

        except Exception as e:
            logger.warning(f"ML-Prediction fehlgeschlagen: {e}")
            return await self._fallback_prediction(request)

    def _calculate_confidence_score(
        self,
        model: Any,
        features: np.ndarray,
        metadata: Any,
        request: PredictionRequest
    ) -> float:
        """Berechnet Confidence-Score für Prediction."""
        confidence_factors = []

        # 1. Model-Performance (R² Score)
        model_performance = metadata.r2_score
        confidence_factors.append(min(1.0, max(0.0, model_performance)))

        # 2. Feature-Qualität
        feature_quality = self._assess_feature_quality(features, request)
        confidence_factors.append(feature_quality)

        # 3. Agent-Vertrautheit (wie oft haben wir diesen Agent gesehen?)
        agent_familiarity = self._assess_agent_familiarity(request.agent_characteristics.agent_id)
        confidence_factors.append(agent_familiarity)

        # 4. Task-Type-Vertrautheit
        task_familiarity = self._assess_task_familiarity(request.task_characteristics.task_type)
        confidence_factors.append(task_familiarity)

        # 5. System-Load-Normalität
        load_normalcy = self._assess_load_normalcy(request.system_load)
        confidence_factors.append(load_normalcy)

        # Gewichteter Durchschnitt
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Model-Performance am wichtigsten
        confidence = sum(w * f for w, f in zip(weights, confidence_factors, strict=False))

        return min(1.0, max(0.0, confidence))

    def _assess_feature_quality(self, features: np.ndarray, request: PredictionRequest) -> float:
        """Bewertet Qualität der Features."""
        # Prüfe auf extreme Werte oder NaN
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return 0.3

        # Prüfe auf realistische Werte
        if request.task_characteristics.complexity_score < 1 or request.task_characteristics.complexity_score > 10:
            return 0.5

        if request.agent_characteristics.current_load < 0 or request.agent_characteristics.current_load > 1:
            return 0.5

        return 0.9

    def _assess_agent_familiarity(self, agent_id: str) -> float:
        """Bewertet Vertrautheit mit Agent."""
        # TODO: Implementiere basierend auf historischen Daten - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
        # Für jetzt: Vereinfachte Logik
        return 0.8

    def _assess_task_familiarity(self, task_type: str) -> float:
        """Bewertet Vertrautheit mit Task-Type."""
        # TODO: Implementiere basierend auf Training-Daten - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
        # Für jetzt: Vereinfachte Logik
        common_task_types = ["simple_query", "data_processing", "llm_generation"]
        return 0.9 if task_type in common_task_types else 0.6

    def _assess_load_normalcy(self, system_load: float) -> float:
        """Bewertet Normalität der System-Load."""
        # Normal Load: 0.2 - 0.8
        if 0.2 <= system_load <= 0.8:
            return 0.9
        if 0.0 <= system_load <= 1.0:
            return 0.7
        return 0.3

    def _get_top_features(self, features: np.ndarray, metadata: Any) -> dict[str, float]:
        """Gibt Top-Features für Debugging zurück."""
        if not metadata.feature_importance:
            return {}

        # Sortiere Features nach Importance
        sorted_features = sorted(
            metadata.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Gib Top 5 Features zurück
        return dict(sorted_features[:5])

    async def _fallback_prediction(self, request: PredictionRequest) -> PredictionResult:
        """Fallback auf statistische Schätzung."""
        # Basis-Schätzung basierend auf Task-Komplexität
        base_time = request.task_characteristics.complexity_score * 1000  # ms

        # Agent-Load-Modifikator
        load_multiplier = 1.0 + request.agent_characteristics.current_load

        # System-Load-Modifikator
        system_multiplier = 1.0 + (request.system_load * 0.5)

        # Token-basierte Schätzung
        token_time = request.task_characteristics.estimated_tokens * 0.1  # 0.1ms pro Token

        # Finale Schätzung
        estimated_time = (base_time + token_time) * load_multiplier * system_multiplier

        # Update Fallback-Statistiken
        self._update_fallback_stats(request.task_characteristics.task_type, estimated_time)

        return PredictionResult(
            predicted_execution_time_ms=estimated_time,
            confidence_score=0.5,  # Niedrige Confidence für Fallback
            model_id="statistical_fallback",
            model_version="v1.0.0",
            prediction_timestamp=datetime.utcnow(),
            used_fallback=True,
            fallback_reason="No active ML model available",
            statistical_estimate=estimated_time
        )

    def _update_fallback_stats(self, task_type: str, estimated_time: float) -> None:
        """Aktualisiert Fallback-Statistiken."""
        if task_type not in self._fallback_stats:
            self._fallback_stats[task_type] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0
            }

        stats = self._fallback_stats[task_type]
        stats["count"] += 1
        stats["total_time"] += estimated_time
        stats["avg_time"] = stats["total_time"] / stats["count"]

    def _create_cache_key(self, request: PredictionRequest) -> str:
        """Erstellt Cache-Key für Request."""
        # Vereinfachter Cache-Key basierend auf wichtigsten Parametern
        key_components = [
            request.agent_characteristics.agent_id,
            request.task_characteristics.task_type,
            f"{request.task_characteristics.complexity_score:.1f}",
            f"{request.agent_characteristics.current_load:.2f}",
            f"{request.system_load:.2f}",
            str(request.concurrent_executions)
        ]

        return f"perf_pred:{'_'.join(key_components)}"

    async def _get_cached_prediction(self, cache_key: str) -> PredictionResult | None:
        """Holt Prediction aus Cache."""
        try:
            redis = await get_cache_client()
            cached_data = await redis.get(cache_key)

            if cached_data:
                import json
                data = json.loads(cached_data)

                # Prüfe TTL
                cached_time = datetime.fromisoformat(data["prediction_timestamp"])
                age_seconds = (datetime.utcnow() - cached_time).total_seconds()

                if age_seconds < self._cache_ttl_seconds:
                    return PredictionResult(
                        predicted_execution_time_ms=data["predicted_execution_time_ms"],
                        confidence_score=data["confidence_score"],
                        model_id=data["model_id"],
                        model_version=data["model_version"],
                        prediction_timestamp=cached_time,
                        top_features=data.get("top_features", {}),
                        used_fallback=data.get("used_fallback", False)
                    )
        except Exception as e:
            logger.warning(f"Cache-Lookup fehlgeschlagen: {e}")

        return None

    async def _cache_prediction(self, cache_key: str, result: PredictionResult) -> None:
        """Speichert Prediction im Cache."""
        try:
            redis = await get_cache_client()

            cache_data = {
                "predicted_execution_time_ms": result.predicted_execution_time_ms,
                "confidence_score": result.confidence_score,
                "model_id": result.model_id,
                "model_version": result.model_version,
                "prediction_timestamp": result.prediction_timestamp.isoformat(),
                "top_features": result.top_features,
                "used_fallback": result.used_fallback
            }

            import json
            await redis.setex(
                cache_key,
                self._cache_ttl_seconds,
                json.dumps(cache_data)
            )

        except Exception as e:
            logger.warning(f"Cache-Speicherung fehlgeschlagen: {e}")

    def _update_performance_stats(self, prediction_time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._prediction_count += 1
        self._total_prediction_time_ms += prediction_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_prediction_time = (
            self._total_prediction_time_ms / self._prediction_count
            if self._prediction_count > 0 else 0.0
        )

        cache_hit_rate = (
            self._cache_hits / self._prediction_count
            if self._prediction_count > 0 else 0.0
        )

        return {
            "total_predictions": self._prediction_count,
            "avg_prediction_time_ms": avg_prediction_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_hits,
            "fallback_stats": self._fallback_stats,
            "meets_sla": avg_prediction_time < 50.0  # < 50ms SLA
        }
