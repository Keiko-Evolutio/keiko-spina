# backend/services/ml/performance_prediction/__init__.py
"""Performance Prediction ML-Pipeline Package.

Implementiert ML-basierte Performance-Vorhersage für Agent-Execution-Zeit
mit Online-Learning, Model-Versioning und A/B-Testing.
"""

from __future__ import annotations

from .data_collector import PerformanceDataCollector
from .data_models import (
    AgentCharacteristics,
    ModelMetadata,
    OnlineLearningUpdate,
    PerformanceDataPoint,
    PredictionRequest,
    PredictionResult,
    TaskCharacteristics,
)
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from .online_learning import OnlineLearningPipeline
from .performance_predictor import PerformancePredictor

__all__ = [
    # Data Models
    "TaskCharacteristics",
    "AgentCharacteristics",
    "PerformanceDataPoint",
    "ModelMetadata",
    "PredictionRequest",
    "PredictionResult",
    "OnlineLearningUpdate",

    # Core Components
    "PerformanceDataCollector",
    "FeatureEngineer",
    "ModelTrainer",
    "PerformancePredictor",
    "OnlineLearningPipeline",

    # Factory Functions
    "create_performance_prediction_pipeline",
    "create_data_collector",
    "create_model_trainer",
    "create_performance_predictor",
]

__version__ = "1.0.0"


def create_performance_prediction_pipeline(
    performance_monitor=None,
    agent_registry=None,
    task_manager=None,
    models_directory: str = "models/performance_prediction"
) -> tuple[PerformanceDataCollector, ModelTrainer, PerformancePredictor, OnlineLearningPipeline]:
    """Factory-Funktion für komplette Performance Prediction Pipeline.
    
    Args:
        performance_monitor: Performance Monitor Instanz
        agent_registry: Agent Registry Instanz  
        task_manager: Task Manager Instanz
        models_directory: Verzeichnis für Model-Speicherung
        
    Returns:
        Tuple von (DataCollector, ModelTrainer, PerformancePredictor, OnlineLearning)
    """
    # Data Collector
    data_collector = None
    if performance_monitor and agent_registry and task_manager:
        data_collector = PerformanceDataCollector(
            performance_monitor=performance_monitor,
            agent_registry=agent_registry,
            task_manager=task_manager
        )

    # Model Trainer
    model_trainer = ModelTrainer(models_directory=models_directory)

    # Performance Predictor
    performance_predictor = PerformancePredictor(model_trainer=model_trainer)

    # Online Learning Pipeline
    online_learning = OnlineLearningPipeline(
        model_trainer=model_trainer,
        performance_predictor=performance_predictor
    )

    return data_collector, model_trainer, performance_predictor, online_learning


def create_data_collector(
    performance_monitor,
    agent_registry,
    task_manager
) -> PerformanceDataCollector:
    """Factory-Funktion für Data Collector.
    
    Args:
        performance_monitor: Performance Monitor Instanz
        agent_registry: Agent Registry Instanz
        task_manager: Task Manager Instanz
        
    Returns:
        Konfigurierter Data Collector
    """
    return PerformanceDataCollector(
        performance_monitor=performance_monitor,
        agent_registry=agent_registry,
        task_manager=task_manager
    )


def create_model_trainer(models_directory: str = "models/performance_prediction") -> ModelTrainer:
    """Factory-Funktion für Model Trainer.
    
    Args:
        models_directory: Verzeichnis für Model-Speicherung
        
    Returns:
        Konfigurierter Model Trainer
    """
    return ModelTrainer(models_directory=models_directory)


def create_performance_predictor(model_trainer: ModelTrainer) -> PerformancePredictor:
    """Factory-Funktion für Performance Predictor.
    
    Args:
        model_trainer: Model Trainer Instanz
        
    Returns:
        Konfigurierter Performance Predictor
    """
    return PerformancePredictor(model_trainer=model_trainer)
