# backend/services/ml/performance_prediction/feature_engineering.py
"""Feature Engineering für Performance Prediction ML-Pipeline.

Implementiert erweiterte Feature-Engineering-Techniken für bessere
ML-Modell-Performance und Feature-Auswahl.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback für fehlende sklearn
    PCA = None
    SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_FEATURE_AVAILABLE = True
except ImportError:
    SelectKBest = None
    f_regression = None
    StandardScaler = None
    LabelEncoder = None
    SKLEARN_FEATURE_AVAILABLE = False

from kei_logging import get_logger

from .data_models import PerformanceDataPoint

logger = get_logger(__name__)


class FeatureEngineer:
    """Feature Engineering für Performance Prediction."""

    def __init__(self):
        """Initialisiert Feature Engineer."""
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_selector: SelectKBest | None = None
        self.pca: PCA | None = None

        # Feature-Konfiguration
        self.max_features = 50  # Maximum Features für Modell
        self.pca_components = 0.95  # Erkläre 95 % der Varianz

        # Feature-Namen-Cache
        self._feature_names: list[str] = []
        self._engineered_feature_names: list[str] = []

        logger.info("Feature Engineer initialisiert")

    def engineer_features(
        self,
        data_points: list[PerformanceDataPoint],
        fit_transformers: bool = True
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Führt komplettes Feature Engineering durch.

        Args:
            data_points: Liste von Performance-Datenpunkten
            fit_transformers: Ob Transformer gefittet werden sollen

        Returns:
            Tuple von (X_features, y_target, feature_names)
        """
        if not data_points:
            return np.array([]), np.array([]), []

        logger.info(f"Starte Feature Engineering für {len(data_points)} Datenpunkte")

        # 1. Basis-Features extrahieren
        df = self._create_base_dataframe(data_points)

        # 2. Erweiterte Features erstellen
        df = self._create_advanced_features(df)

        # 3. Zeitbasierte Features
        df = self._create_temporal_features(df)

        # 4. Interaktions-Features
        df = self._create_interaction_features(df)

        # 5. Aggregations-Features
        df = self._create_aggregation_features(df, data_points)

        # 6. Target-Variable extrahieren
        y = df["target_execution_time_ms"].values

        # 7. Features für ML vorbereiten
        X, feature_names = self._prepare_features_for_ml(df, fit_transformers)

        logger.info({
            "event": "feature_engineering_completed",
            "original_features": len(df.columns) - 1,  # -1 für target
            "final_features": X.shape[1],
            "samples": X.shape[0]
        })

        return X, y, feature_names

    def _create_base_dataframe(self, data_points: list[PerformanceDataPoint]) -> pd.DataFrame:
        """Erstellt Basis-DataFrame aus Datenpunkten."""
        feature_data = []

        for dp in data_points:
            features = dp.to_feature_vector()
            features["target_execution_time_ms"] = dp.actual_execution_time_ms
            features["timestamp"] = dp.timestamp
            features["agent_id"] = dp.agent_id
            features["task_type"] = dp.task_characteristics.task_type
            feature_data.append(features)

        df = pd.DataFrame(feature_data)
        self._feature_names = [col for col in df.columns if col != "target_execution_time_ms"]

        return df

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt erweiterte Features."""
        # Ratio-Features
        df["load_to_capacity_ratio"] = df["agent_current_load"] / (df["agent_max_concurrent"] + 1e-6)
        df["queue_to_capacity_ratio"] = df["agent_queue_length"] / (df["agent_max_concurrent"] + 1e-6)
        df["active_to_capacity_ratio"] = df["agent_active_tasks"] / (df["agent_max_concurrent"] + 1e-6)

        # Performance-Ratios
        df["success_to_error_ratio"] = df["agent_success_rate"] / (df["agent_error_rate"] + 1e-6)
        df["response_time_efficiency"] = 1.0 / (df["agent_avg_response_time"] + 1e-6)

        # Task-Komplexitäts-Features
        df["complexity_tokens_ratio"] = df["task_complexity_score"] / (df["task_estimated_tokens"] + 1e-6)
        df["tokens_per_capability"] = df["task_estimated_tokens"] / (df["task_capability_count"] + 1e-6)

        # System-Load-Features
        df["system_load_impact"] = df["system_load"] * df["concurrent_executions"]
        df["load_pressure"] = df["system_load"] * df["agent_current_load"]

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt zeitbasierte Features."""
        # Zyklische Zeit-Features (für bessere ML-Performance)
        df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Arbeitszeit-Features
        df["is_business_hours"] = ((df["time_of_day_hour"] >= 9) &
                                  (df["time_of_day_hour"] <= 17) &
                                  (df["day_of_week"] < 5)).astype(float)

        df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
        df["is_night"] = ((df["time_of_day_hour"] < 6) |
                         (df["time_of_day_hour"] > 22)).astype(float)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Interaktions-Features zwischen wichtigen Variablen."""
        # Task-Agent-Interaktionen
        df["complexity_load_interaction"] = (df["task_complexity_score"] *
                                           df["agent_current_load"])

        df["tokens_response_time_interaction"] = (df["task_estimated_tokens"] *
                                                df["agent_avg_response_time"])

        df["urgency_load_interaction"] = (df["task_deadline_urgency"] *
                                        df["agent_current_load"])

        # System-Performance-Interaktionen
        df["system_agent_load_interaction"] = (df["system_load"] *
                                             df["agent_current_load"])

        df["concurrent_complexity_interaction"] = (df["concurrent_executions"] *
                                                 df["task_complexity_score"])

        # Capability-Performance-Interaktionen
        df["specialization_success_interaction"] = (df["agent_specialization_score"] *
                                                  df["agent_success_rate"])

        return df

    def _create_aggregation_features(
        self,
        df: pd.DataFrame,
        data_points: list[PerformanceDataPoint]
    ) -> pd.DataFrame:
        """Erstellt Aggregations-Features basierend auf historischen Daten."""
        # Agent-basierte Aggregationen
        agent_stats = df.groupby("agent_id").agg({
            "target_execution_time_ms": ["mean", "std", "min", "max"],
            "agent_success_rate": "mean",
            "agent_current_load": "mean"
        }).reset_index()

        # Flatten column names
        agent_stats.columns = ["agent_id"] + [
            f"agent_historical_{col[0]}_{col[1]}" if col[1] else f"agent_historical_{col[0]}"
            for col in agent_stats.columns[1:]
        ]

        # Task-Type-basierte Aggregationen
        task_stats = df.groupby("task_type").agg({
            "target_execution_time_ms": ["mean", "std"],
            "task_complexity_score": "mean"
        }).reset_index()

        task_stats.columns = ["task_type"] + [
            f"task_type_historical_{col[0]}_{col[1]}" if col[1] else f"task_type_historical_{col[0]}"
            for col in task_stats.columns[1:]
        ]

        # Merge zurück zum Haupt-DataFrame
        df = df.merge(agent_stats, on="agent_id", how="left")
        df = df.merge(task_stats, on="task_type", how="left")

        # Fülle NaN-Werte mit Defaults
        for col in df.columns:
            if "historical" in col:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _prepare_features_for_ml(
        self,
        df: pd.DataFrame,
        fit_transformers: bool
    ) -> tuple[np.ndarray, list[str]]:
        """Bereitet Features für ML-Modell vor."""
        # Entferne nicht-numerische Spalten
        exclude_cols = ["target_execution_time_ms", "timestamp", "agent_id", "task_type"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()

        # Handle kategorische Features (falls vorhanden)
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            if fit_transformers:
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Check if encoder is fitted before using it
                encoder = self.label_encoders.get(col)
                if encoder is None or not hasattr(encoder, "classes_"):
                    logger.warning(f"LabelEncoder for column '{col}' not fitted. Skipping transformation.")
                    # Convert to numeric if possible, otherwise use ordinal encoding
                    X[col] = pd.factorize(X[col].astype(str))[0]
                    continue

                # Handle unseen categories
                X[col] = X[col].astype(str)
                known_classes = set(encoder.classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else "unknown")

                # Add "unknown" to encoder classes if not present
                if "unknown" not in known_classes:
                    # Create new encoder with unknown class
                    extended_classes = list(encoder.classes_) + ["unknown"]
                    encoder.classes_ = np.array(extended_classes)

                X[col] = encoder.transform(X[col])

        # Konvertiere zu numpy array
        X_array = X.values.astype(np.float32)

        # Handle NaN/Inf values
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e6, neginf=-1e6)

        # Feature Scaling
        if fit_transformers:
            X_array = self.scaler.fit_transform(X_array)
        else:
            X_array = self.scaler.transform(X_array)

        # Feature Selection
        if fit_transformers and X_array.shape[1] > self.max_features:
            self.feature_selector = SelectKBest(
                score_func=f_regression,
                k=min(self.max_features, X_array.shape[1])
            )
            X_array = self.feature_selector.fit_transform(X_array, df["target_execution_time_ms"])

            # Update feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            feature_cols = [feature_cols[i] for i in selected_indices]
        elif self.feature_selector:
            X_array = self.feature_selector.transform(X_array)
            selected_indices = self.feature_selector.get_support(indices=True)
            feature_cols = [feature_cols[i] for i in selected_indices]

        # Optional: PCA für Dimensionalitäts-Reduktion
        if fit_transformers and X_array.shape[1] > 20:
            self.pca = PCA(n_components=self.pca_components)
            X_array = self.pca.fit_transform(X_array)

            # Update feature names für PCA
            feature_cols = [f"pca_component_{i}" for i in range(X_array.shape[1])]
        elif self.pca:
            X_array = self.pca.transform(X_array)
            feature_cols = [f"pca_component_{i}" for i in range(X_array.shape[1])]

        self._engineered_feature_names = feature_cols

        return X_array, feature_cols

    def get_feature_importance_mapping(self, feature_importance: np.ndarray) -> dict[str, float]:
        """Mappt Feature-Importance zurück zu Original-Feature-Namen."""
        if len(feature_importance) != len(self._engineered_feature_names):
            logger.warning("Feature-Importance-Länge stimmt nicht mit Feature-Namen überein")
            return {}

        return dict(zip(self._engineered_feature_names, feature_importance, strict=False))

    def transform_single_prediction(
        self,
        task_characteristics: Any,
        agent_characteristics: Any,
        system_load: float,
        concurrent_executions: int
    ) -> np.ndarray:
        """Transformiert einzelne Prediction-Request zu Features."""
        from .data_models import PerformanceDataPoint

        # Erstelle temporären DataPoint
        temp_dp = PerformanceDataPoint(
            execution_id="temp",
            task_id="temp",
            agent_id=agent_characteristics.agent_id,
            timestamp=datetime.utcnow(),
            task_characteristics=task_characteristics,
            agent_characteristics=agent_characteristics,
            system_load=system_load,
            concurrent_executions=concurrent_executions,
            time_of_day_hour=datetime.utcnow().hour,
            day_of_week=datetime.utcnow().weekday(),
            actual_execution_time_ms=0.0,  # Dummy-Wert
            success=True
        )

        # Feature Engineering (ohne Fitting)
        X, _, _ = self.engineer_features([temp_dp], fit_transformers=False)

        return X[0] if len(X) > 0 else np.array([])

    def get_feature_statistics(self) -> dict[str, Any]:
        """Gibt Feature-Engineering-Statistiken zurück."""
        return {
            "original_feature_count": len(self._feature_names),
            "engineered_feature_count": len(self._engineered_feature_names),
            "feature_selection_enabled": self.feature_selector is not None,
            "pca_enabled": self.pca is not None,
            "pca_components": self.pca.n_components_ if self.pca else None,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist() if self.pca else None,
            "scaler_fitted": hasattr(self.scaler, "mean_"),
            "label_encoders_count": len(self.label_encoders)
        }
