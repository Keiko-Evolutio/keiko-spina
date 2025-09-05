# backend/services/ml/performance_prediction/model_trainer.py
"""ML-Model-Training für Performance Prediction.

Implementiert Training verschiedener ML-Modelle mit Cross-Validation,
Hyperparameter-Tuning und Model-Evaluation.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None  # Fallback für xgboost import
    XGBOOST_AVAILABLE = False

from kei_logging import get_logger

from .data_models import ModelMetadata, PerformanceDataPoint
from .feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ModelTrainer:
    """ML-Model-Trainer für Performance Prediction."""

    def __init__(self, models_directory: str = "models/performance_prediction"):
        """Initialisiert Model Trainer.

        Args:
            models_directory: Verzeichnis für Model-Speicherung
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()

        # Verfügbare Modelle
        self.available_models = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }

        # XGBoost falls verfügbar
        if XGBOOST_AVAILABLE:
            self.available_models["xgboost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )

        # Hyperparameter-Grids für Tuning
        self.param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            } if XGBOOST_AVAILABLE else {},
            "ridge_regression": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            }
        }

        # Trainierte Modelle
        self.trained_models: dict[str, Any] = {}
        self.model_metadata: dict[str, ModelMetadata] = {}

        logger.info({
            "event": "model_trainer_initialized",
            "available_models": list(self.available_models.keys()),
            "xgboost_available": XGBOOST_AVAILABLE,
            "models_directory": str(self.models_directory)
        })

    async def train_models(
        self,
        data_points: list[PerformanceDataPoint],
        model_types: list[str] | None = None,
        enable_hyperparameter_tuning: bool = True,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> dict[str, ModelMetadata]:
        """Trainiert ML-Modelle auf Performance-Daten.

        Args:
            data_points: Training-Daten
            model_types: Zu trainierende Modell-Typen (None = alle)
            enable_hyperparameter_tuning: Hyperparameter-Tuning aktivieren
            test_size: Größe des Test-Sets
            cv_folds: Anzahl Cross-Validation-Folds

        Returns:
            Dictionary mit Model-Metadaten
        """
        if len(data_points) < 100:
            raise ValueError(f"Mindestens 100 Datenpunkte erforderlich, {len(data_points)} gegeben")

        logger.info(f"Starte Model-Training mit {len(data_points)} Datenpunkten")

        # Feature Engineering
        X, y, feature_names = self.feature_engineer.engineer_features(
            data_points,
            fit_transformers=True
        )

        if X.shape[0] == 0:
            raise ValueError("Feature Engineering ergab keine Features")

        # Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Modell-Typen bestimmen
        if model_types is None:
            model_types = list(self.available_models.keys())

        trained_metadata = {}

        for model_type in model_types:
            if model_type not in self.available_models:
                logger.warning(f"Modell-Typ {model_type} nicht verfügbar")
                continue

            try:
                metadata = await self._train_single_model(
                    model_type=model_type,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=feature_names,
                    enable_hyperparameter_tuning=enable_hyperparameter_tuning,
                    cv_folds=cv_folds
                )

                trained_metadata[model_type] = metadata

            except Exception as e:
                logger.error(f"Fehler beim Training von {model_type}: {e}")

        logger.info({
            "event": "model_training_completed",
            "trained_models": list(trained_metadata.keys()),
            "best_model": self._get_best_model(trained_metadata)
        })

        return trained_metadata

    async def _train_single_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        enable_hyperparameter_tuning: bool,
        cv_folds: int
    ) -> ModelMetadata:
        """Trainiert einzelnes ML-Modell."""
        training_start = datetime.utcnow()

        # Basis-Modell
        base_model = self.available_models[model_type]

        # Hyperparameter-Tuning
        if enable_hyperparameter_tuning and model_type in self.param_grids:
            logger.info(f"Starte Hyperparameter-Tuning für {model_type}")

            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grids[model_type],
                cv=cv_folds,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            logger.info({
                "event": "hyperparameter_tuning_completed",
                "model_type": model_type,
                "best_params": grid_search.best_params_,
                "best_score": -grid_search.best_score_
            })
        else:
            # Standard-Training
            model = base_model
            model.fit(X_train, y_train)

        training_end = datetime.utcnow()
        training_duration = (training_end - training_start).total_seconds()

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metriken berechnen
        mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # MAPE (Mean Absolute Percentage Error)
        test_mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-6))) * 100

        # Cross-Validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring="neg_mean_absolute_error"
        )
        cv_scores = -cv_scores  # Konvertiere zu positiven Werten

        # Feature Importance
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance_values, strict=False))
        elif hasattr(model, "coef_"):
            importance_values = np.abs(model.coef_)
            feature_importance = dict(zip(feature_names, importance_values, strict=False))

        # Model-ID und Version generieren
        model_id = f"{model_type}_{int(time.time())}"
        model_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Modell speichern
        model_path = self.models_directory / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        # Metadata erstellen
        metadata = ModelMetadata(
            model_id=model_id,
            model_version=model_version,
            model_type=model_type,
            training_data_size=len(X_train),
            training_start_time=training_start,
            training_end_time=training_end,
            training_duration_seconds=training_duration,
            mae=test_mae,
            rmse=test_rmse,
            r2_score=test_r2,
            mape=test_mape,
            feature_count=len(feature_names),
            feature_names=feature_names,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores.tolist(),
            test_set_score=test_r2,
            is_active=False
        )

        # Speichere Modell und Metadata
        self.trained_models[model_id] = model
        self.model_metadata[model_id] = metadata

        logger.info({
            "event": "model_training_completed",
            "model_id": model_id,
            "model_type": model_type,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mape": test_mape,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        })

        return metadata

    def _get_best_model(self, trained_metadata: dict[str, ModelMetadata]) -> str | None:
        """Bestimmt bestes Modell basierend auf Metriken."""
        if not trained_metadata:
            return None

        # Sortiere nach R² Score (höher ist besser)
        best_model = max(
            trained_metadata.items(),
            key=lambda x: x[1].r2_score
        )

        return best_model[0]

    def load_model(self, model_id: str) -> Any | None:
        """Lädt trainiertes Modell."""
        if model_id in self.trained_models:
            return self.trained_models[model_id]

        # Versuche von Disk zu laden
        model_path = self.models_directory / f"{model_id}.joblib"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                self.trained_models[model_id] = model
                return model
            except Exception as e:
                logger.error(f"Fehler beim Laden von Modell {model_id}: {e}")

        return None

    def get_model_metadata(self, model_id: str) -> ModelMetadata | None:
        """Gibt Modell-Metadaten zurück."""
        return self.model_metadata.get(model_id)

    def list_trained_models(self) -> list[dict[str, Any]]:
        """Listet alle trainierten Modelle auf."""
        models_info = []

        for model_id, metadata in self.model_metadata.items():
            models_info.append({
                "model_id": model_id,
                "model_type": metadata.model_type,
                "model_version": metadata.model_version,
                "training_data_size": metadata.training_data_size,
                "mae": metadata.mae,
                "rmse": metadata.rmse,
                "r2_score": metadata.r2_score,
                "mape": metadata.mape,
                "is_active": metadata.is_active,
                "created_at": metadata.created_at.isoformat()
            })

        # Sortiere nach R² Score
        models_info.sort(key=lambda x: x["r2_score"], reverse=True)

        return models_info

    def activate_model(self, model_id: str) -> bool:
        """Aktiviert Modell für Production."""
        if model_id not in self.model_metadata:
            return False

        # Deaktiviere alle anderen Modelle
        for metadata in self.model_metadata.values():
            metadata.is_active = False

        # Aktiviere gewähltes Modell
        self.model_metadata[model_id].is_active = True
        self.model_metadata[model_id].deployed_at = datetime.utcnow()

        logger.info({
            "event": "model_activated",
            "model_id": model_id,
            "model_type": self.model_metadata[model_id].model_type
        })

        return True

    def get_active_model(self) -> tuple[str, Any, ModelMetadata] | None:
        """Gibt aktives Modell zurück."""
        for model_id, metadata in self.model_metadata.items():
            if metadata.is_active:
                model = self.load_model(model_id)
                if model:
                    return model_id, model, metadata

        return None

    def evaluate_model_performance(
        self,
        model_id: str,
        test_data_points: list[PerformanceDataPoint]
    ) -> dict[str, float]:
        """Evaluiert Modell-Performance auf neuen Daten."""
        model = self.load_model(model_id)
        if not model:
            raise ValueError(f"Modell {model_id} nicht gefunden")

        # Feature Engineering (ohne Fitting)
        X_test, y_test, _ = self.feature_engineer.engineer_features(
            test_data_points,
            fit_transformers=False
        )

        if X_test.shape[0] == 0:
            raise ValueError("Keine Test-Daten nach Feature Engineering")

        # Predictions
        y_pred = model.predict(X_test)

        # Metriken
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

        return {
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
            "sample_count": len(test_data_points)
        }

    def cleanup_old_models(self, keep_latest: int = 5) -> None:
        """Bereinigt alte Modelle und behält nur die neuesten."""
        if len(self.model_metadata) <= keep_latest:
            return

        # Sortiere nach Erstellungsdatum
        sorted_models = sorted(
            self.model_metadata.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )

        # Behalte nur die neuesten
        models_to_keep = set(model_id for model_id, _ in sorted_models[:keep_latest])

        # Lösche alte Modelle
        models_to_delete = []
        for model_id in self.model_metadata:
            if model_id not in models_to_keep and not self.model_metadata[model_id].is_active:
                models_to_delete.append(model_id)

        for model_id in models_to_delete:
            # Lösche von Disk
            model_path = self.models_directory / f"{model_id}.joblib"
            if model_path.exists():
                model_path.unlink()

            # Lösche aus Memory
            self.trained_models.pop(model_id, None)
            self.model_metadata.pop(model_id, None)

        logger.info({
            "event": "model_cleanup_completed",
            "deleted_models": len(models_to_delete),
            "remaining_models": len(self.model_metadata)
        })
