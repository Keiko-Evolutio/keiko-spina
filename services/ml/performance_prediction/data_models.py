# backend/services/ml/performance_prediction/data_models.py
"""Datenmodelle für Performance Prediction ML-Pipeline.

Definiert alle Datenstrukturen für historische Performance-Daten,
Features und ML-Modell-Metadaten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task-Komplexitäts-Level."""

    TRIVIAL = "trivial"      # 1-2 Komplexität
    SIMPLE = "simple"        # 3-4 Komplexität
    MODERATE = "moderate"    # 5-6 Komplexität
    COMPLEX = "complex"      # 7-8 Komplexität
    CRITICAL = "critical"    # 9-10 Komplexität


class AgentLoadLevel(Enum):
    """Agent-Load-Level."""

    IDLE = "idle"           # 0-20% Load
    LOW = "low"             # 21-40% Load
    MODERATE = "moderate"   # 41-60% Load
    HIGH = "high"           # 61-80% Load
    OVERLOADED = "overloaded"  # 81-100% Load


@dataclass
class TaskCharacteristics:
    """Task-Charakteristika für Feature-Engineering."""

    # Basis-Eigenschaften
    task_type: str
    complexity_score: float  # 1-10 Skala
    estimated_tokens: int
    required_capabilities: list[str]

    # Kontext-Eigenschaften
    user_priority: str = "normal"
    deadline_urgency: float = 0.0  # 0-1 Skala
    dependency_count: int = 0
    parallel_execution_possible: bool = True

    # Payload-Eigenschaften
    payload_size_bytes: int = 0
    contains_files: bool = False
    requires_external_api: bool = False

    @property
    def complexity_level(self) -> TaskComplexity:
        """Gibt Komplexitäts-Level zurück."""
        if self.complexity_score <= 2:
            return TaskComplexity.TRIVIAL
        if self.complexity_score <= 4:
            return TaskComplexity.SIMPLE
        if self.complexity_score <= 6:
            return TaskComplexity.MODERATE
        if self.complexity_score <= 8:
            return TaskComplexity.COMPLEX
        return TaskComplexity.CRITICAL


@dataclass
class AgentCharacteristics:
    """Agent-Charakteristika für Feature-Engineering."""

    # Basis-Eigenschaften
    agent_id: str
    agent_type: str
    capabilities: list[str]

    # Performance-Eigenschaften
    current_load: float  # 0-1 Skala
    avg_response_time_ms: float
    success_rate: float  # 0-1 Skala
    error_rate: float    # 0-1 Skala

    # Kapazitäts-Eigenschaften
    max_concurrent_tasks: int
    current_active_tasks: int
    queue_length: int

    # Historische Eigenschaften
    total_completed_tasks: int
    avg_task_completion_time_ms: float
    specialization_score: float  # 0-1 Skala für Task-Type

    @property
    def load_level(self) -> AgentLoadLevel:
        """Gibt Load-Level zurück."""
        if self.current_load <= 0.2:
            return AgentLoadLevel.IDLE
        if self.current_load <= 0.4:
            return AgentLoadLevel.LOW
        if self.current_load <= 0.6:
            return AgentLoadLevel.MODERATE
        if self.current_load <= 0.8:
            return AgentLoadLevel.HIGH
        return AgentLoadLevel.OVERLOADED

    @property
    def availability_score(self) -> float:
        """Berechnet Verfügbarkeits-Score (0-1)."""
        if self.max_concurrent_tasks == 0:
            return 0.0

        utilization = self.current_active_tasks / self.max_concurrent_tasks
        return max(0.0, 1.0 - utilization)


@dataclass
class PerformanceDataPoint:
    """Einzelner Performance-Datenpunkt für ML-Training."""

    # Identifikation
    execution_id: str
    task_id: str
    agent_id: str
    timestamp: datetime

    # Task-Features
    task_characteristics: TaskCharacteristics

    # Agent-Features
    agent_characteristics: AgentCharacteristics

    # Umgebungs-Features
    system_load: float  # 0-1 Skala
    concurrent_executions: int
    time_of_day_hour: int  # 0-23
    day_of_week: int       # 0-6 (Montag=0)

    # Target-Variable (was wir vorhersagen wollen)
    actual_execution_time_ms: float

    # Zusätzliche Metriken
    success: bool
    error_type: str | None = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_feature_vector(self) -> dict[str, float]:
        """Konvertiert zu Feature-Vector für ML-Modell."""
        features = {
            # Task-Features
            "task_complexity_score": self.task_characteristics.complexity_score,
            "task_estimated_tokens": float(self.task_characteristics.estimated_tokens),
            "task_capability_count": float(len(self.task_characteristics.required_capabilities)),
            "task_deadline_urgency": self.task_characteristics.deadline_urgency,
            "task_dependency_count": float(self.task_characteristics.dependency_count),
            "task_parallel_possible": float(self.task_characteristics.parallel_execution_possible),
            "task_payload_size": float(self.task_characteristics.payload_size_bytes),
            "task_has_files": float(self.task_characteristics.contains_files),
            "task_needs_external_api": float(self.task_characteristics.requires_external_api),

            # Agent-Features
            "agent_current_load": self.agent_characteristics.current_load,
            "agent_avg_response_time": self.agent_characteristics.avg_response_time_ms,
            "agent_success_rate": self.agent_characteristics.success_rate,
            "agent_error_rate": self.agent_characteristics.error_rate,
            "agent_queue_length": float(self.agent_characteristics.queue_length),
            "agent_active_tasks": float(self.agent_characteristics.current_active_tasks),
            "agent_max_concurrent": float(self.agent_characteristics.max_concurrent_tasks),
            "agent_total_completed": float(self.agent_characteristics.total_completed_tasks),
            "agent_avg_completion_time": self.agent_characteristics.avg_task_completion_time_ms,
            "agent_specialization_score": self.agent_characteristics.specialization_score,
            "agent_availability_score": self.agent_characteristics.availability_score,

            # Umgebungs-Features
            "system_load": self.system_load,
            "concurrent_executions": float(self.concurrent_executions),
            "time_of_day_hour": float(self.time_of_day_hour),
            "day_of_week": float(self.day_of_week),

            # Kategorische Features (One-Hot-Encoded)
            "task_complexity_trivial": float(self.task_characteristics.complexity_level == TaskComplexity.TRIVIAL),
            "task_complexity_simple": float(self.task_characteristics.complexity_level == TaskComplexity.SIMPLE),
            "task_complexity_moderate": float(self.task_characteristics.complexity_level == TaskComplexity.MODERATE),
            "task_complexity_complex": float(self.task_characteristics.complexity_level == TaskComplexity.COMPLEX),
            "task_complexity_critical": float(self.task_characteristics.complexity_level == TaskComplexity.CRITICAL),

            "agent_load_idle": float(self.agent_characteristics.load_level == AgentLoadLevel.IDLE),
            "agent_load_low": float(self.agent_characteristics.load_level == AgentLoadLevel.LOW),
            "agent_load_moderate": float(self.agent_characteristics.load_level == AgentLoadLevel.MODERATE),
            "agent_load_high": float(self.agent_characteristics.load_level == AgentLoadLevel.HIGH),
            "agent_load_overloaded": float(self.agent_characteristics.load_level == AgentLoadLevel.OVERLOADED),

            "priority_low": float(self.task_characteristics.user_priority == "low"),
            "priority_normal": float(self.task_characteristics.user_priority == "normal"),
            "priority_high": float(self.task_characteristics.user_priority == "high"),
            "priority_urgent": float(self.task_characteristics.user_priority == "urgent"),
        }

        return features


@dataclass
class ModelMetadata:
    """Metadaten für ML-Modell."""

    model_id: str
    model_version: str
    model_type: str  # "xgboost", "random_forest", "linear_regression"

    # Training-Informationen
    training_data_size: int
    training_start_time: datetime
    training_end_time: datetime
    training_duration_seconds: float

    # Performance-Metriken
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r2_score: float  # R² Score
    mape: float  # Mean Absolute Percentage Error

    # Feature-Informationen
    feature_count: int
    feature_names: list[str]
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Validierung
    cross_validation_scores: list[float] = field(default_factory=list)
    test_set_score: float = 0.0

    # Deployment-Informationen
    deployed_at: datetime | None = None
    is_active: bool = False
    prediction_count: int = 0

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionRequest:
    """Request für Performance-Prediction."""

    task_characteristics: TaskCharacteristics
    agent_characteristics: AgentCharacteristics
    system_load: float
    concurrent_executions: int

    # Optional: Für Batch-Predictions
    request_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionResult:
    """Ergebnis einer Performance-Prediction."""

    # Prediction
    predicted_execution_time_ms: float
    confidence_score: float  # 0-1 Skala

    # Metadaten
    model_id: str
    model_version: str
    prediction_timestamp: datetime

    # Feature-Importance (für Debugging)
    top_features: dict[str, float] = field(default_factory=dict)

    # Fallback-Information
    used_fallback: bool = False
    fallback_reason: str | None = None
    statistical_estimate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für API-Response."""
        return {
            "predicted_execution_time_ms": self.predicted_execution_time_ms,
            "confidence_score": self.confidence_score,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "top_features": self.top_features,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "statistical_estimate": self.statistical_estimate
        }


@dataclass
class OnlineLearningUpdate:
    """Update für Online-Learning-System."""

    # Original Prediction
    prediction_request: PredictionRequest
    prediction_result: PredictionResult

    # Actual Outcome
    actual_execution_time_ms: float
    actual_success: bool
    actual_error_type: str | None = None

    # Feedback-Metriken
    prediction_error: float = field(init=False)
    absolute_error: float = field(init=False)
    percentage_error: float = field(init=False)

    # Metadaten
    feedback_timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Berechnet Fehler-Metriken nach Initialisierung."""
        predicted = self.prediction_result.predicted_execution_time_ms
        actual = self.actual_execution_time_ms

        self.prediction_error = predicted - actual
        self.absolute_error = abs(self.prediction_error)

        if actual > 0:
            self.percentage_error = abs(self.prediction_error) / actual
        else:
            self.percentage_error = 0.0
