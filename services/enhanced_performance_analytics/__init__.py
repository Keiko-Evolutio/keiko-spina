# backend/services/enhanced_performance_analytics/__init__.py
"""Enhanced Performance Analytics Package.

Implementiert Enterprise-Grade Performance Analytics mit Event-driven Integration,
ML-basierte Performance-Vorhersagen und automatische Optimization-Empfehlungen.
"""

from __future__ import annotations

from .analytics_service_integration_layer import AnalyticsServiceIntegrationLayer
from .data_models import (
    AdvancedPerformanceMetrics,
    AnalyticsConfiguration,
    AnalyticsPerformanceMetrics,
    AnalyticsScope,
    AnomalyDetection,
    AnomalyType,
    EventType,
    MetricDimension,
    MLPerformancePrediction,
    OptimizationType,
    PerformanceDataPoint,
    PerformanceEvent,
    PerformanceOptimizationRecommendation,
    TrendAnalysis,
    TrendDirection,
)
from .enhanced_performance_analytics_engine import EnhancedPerformanceAnalyticsEngine
from .event_driven_analytics_engine import EventDrivenAnalyticsEngine
from .ml_performance_prediction_engine import MLPerformancePredictionEngine
from .performance_optimization_engine import PerformanceOptimizationEngine
from .trend_analysis_anomaly_detection_engine import TrendAnalysisAnomalyDetectionEngine

__all__ = [
    # Core Components
    "EnhancedPerformanceAnalyticsEngine",
    "EventDrivenAnalyticsEngine",
    "MLPerformancePredictionEngine",
    "TrendAnalysisAnomalyDetectionEngine",
    "PerformanceOptimizationEngine",
    "AnalyticsServiceIntegrationLayer",

    # Data Models
    "PerformanceDataPoint",
    "AdvancedPerformanceMetrics",
    "TrendAnalysis",
    "AnomalyDetection",
    "PerformanceOptimizationRecommendation",
    "MLPerformancePrediction",
    "PerformanceEvent",
    "AnalyticsConfiguration",
    "AnalyticsPerformanceMetrics",

    # Enums
    "AnalyticsScope",
    "MetricDimension",
    "TrendDirection",
    "AnomalyType",
    "OptimizationType",
    "EventType",

    # Factory Functions
    "create_enhanced_performance_analytics_engine",
    "create_event_driven_analytics_engine",
    "create_ml_performance_prediction_engine",
    "create_trend_analysis_anomaly_detection_engine",
    "create_performance_optimization_engine",
    "create_analytics_service_integration_layer",
    "create_integrated_performance_analytics_system",
]

__version__ = "1.0.0"


def create_enhanced_performance_analytics_engine(
    monitoring_engine=None,
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    performance_predictor=None,
    configuration=None
) -> EnhancedPerformanceAnalyticsEngine:
    """Factory-Funktion für Enhanced Performance Analytics Engine.

    Args:
        monitoring_engine: Real-time Monitoring Engine (optional)
        dependency_resolution_engine: Dependency Resolution Engine (optional)
        quota_management_engine: Quota Management Engine (optional)
        security_integration_engine: Security Integration Engine (optional)
        performance_predictor: Performance Predictor (optional)
        configuration: Analytics-Konfiguration (optional)

    Returns:
        Konfigurierte Enhanced Performance Analytics Engine
    """
    return EnhancedPerformanceAnalyticsEngine(
        monitoring_engine=monitoring_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        performance_predictor=performance_predictor,
        configuration=configuration
    )


def create_event_driven_analytics_engine(
    monitoring_engine=None,
    configuration=None
) -> EventDrivenAnalyticsEngine:
    """Factory-Funktion für Event-driven Analytics Engine.

    Args:
        monitoring_engine: Real-time Monitoring Engine (optional)
        configuration: Analytics-Konfiguration (optional)

    Returns:
        Konfigurierte Event-driven Analytics Engine
    """
    return EventDrivenAnalyticsEngine(
        monitoring_engine=monitoring_engine,
        configuration=configuration
    )


def create_ml_performance_prediction_engine(
    performance_predictor=None,
    configuration=None
) -> MLPerformancePredictionEngine:
    """Factory-Funktion für ML Performance Prediction Engine.

    Args:
        performance_predictor: Performance Predictor (optional)
        configuration: Analytics-Konfiguration (optional)

    Returns:
        Konfigurierte ML Performance Prediction Engine
    """
    return MLPerformancePredictionEngine(
        performance_predictor=performance_predictor,
        configuration=configuration
    )


def create_trend_analysis_anomaly_detection_engine(
    configuration=None
) -> TrendAnalysisAnomalyDetectionEngine:
    """Factory-Funktion für Trend Analysis und Anomaly Detection Engine.

    Args:
        configuration: Analytics-Konfiguration (optional)

    Returns:
        Konfigurierte Trend Analysis und Anomaly Detection Engine
    """
    return TrendAnalysisAnomalyDetectionEngine(
        configuration=configuration
    )


def create_performance_optimization_engine(
    configuration=None
) -> PerformanceOptimizationEngine:
    """Factory-Funktion für Performance Optimization Engine.

    Args:
        configuration: Analytics-Konfiguration (optional)

    Returns:
        Konfigurierte Performance Optimization Engine
    """
    return PerformanceOptimizationEngine(
        configuration=configuration
    )


def create_analytics_service_integration_layer(
    event_driven_analytics_engine: EventDrivenAnalyticsEngine,
    ml_prediction_engine: MLPerformancePredictionEngine,
    trend_anomaly_engine: TrendAnalysisAnomalyDetectionEngine,
    optimization_engine: PerformanceOptimizationEngine,
    monitoring_engine=None,
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    performance_predictor=None,
    configuration=None
) -> AnalyticsServiceIntegrationLayer:
    """Factory-Funktion für Analytics Service Integration Layer.

    Args:
        event_driven_analytics_engine: Event-driven Analytics Engine
        ml_prediction_engine: ML Performance Prediction Engine
        trend_anomaly_engine: Trend Analysis und Anomaly Detection Engine
        optimization_engine: Performance Optimization Engine
        monitoring_engine: Real-time Monitoring Engine (optional)
        dependency_resolution_engine: Dependency Resolution Engine (optional)
        quota_management_engine: Quota Management Engine (optional)
        security_integration_engine: Security Integration Engine (optional)
        performance_predictor: Performance Predictor (optional)
        configuration: Analytics-Konfiguration (optional)

    Returns:
        Konfigurierte Analytics Service Integration Layer
    """
    return AnalyticsServiceIntegrationLayer(
        event_driven_analytics_engine=event_driven_analytics_engine,
        ml_prediction_engine=ml_prediction_engine,
        trend_anomaly_engine=trend_anomaly_engine,
        optimization_engine=optimization_engine,
        monitoring_engine=monitoring_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        performance_predictor=performance_predictor,
        configuration=configuration
    )


def create_integrated_performance_analytics_system(
    monitoring_engine=None,
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    performance_predictor=None,
    analytics_configuration=None
) -> dict:
    """Factory-Funktion für integriertes Performance Analytics System.

    Args:
        monitoring_engine: Real-time Monitoring Engine (optional)
        dependency_resolution_engine: Dependency Resolution Engine (optional)
        quota_management_engine: Quota Management Engine (optional)
        security_integration_engine: Security Integration Engine (optional)
        performance_predictor: Performance Predictor (optional)
        analytics_configuration: Analytics-Konfiguration (optional)

    Returns:
        Dictionary mit allen konfigurierten Komponenten
    """
    # Erstelle Konfiguration
    configuration = analytics_configuration or AnalyticsConfiguration()

    # Erstelle alle Analytics-Engines
    event_driven_analytics_engine = create_event_driven_analytics_engine(
        monitoring_engine=monitoring_engine,
        configuration=configuration
    )

    ml_prediction_engine = create_ml_performance_prediction_engine(
        performance_predictor=performance_predictor,
        configuration=configuration
    )

    trend_anomaly_engine = create_trend_analysis_anomaly_detection_engine(
        configuration=configuration
    )

    optimization_engine = create_performance_optimization_engine(
        configuration=configuration
    )

    # Erstelle Service Integration Layer
    service_integration_layer = create_analytics_service_integration_layer(
        event_driven_analytics_engine=event_driven_analytics_engine,
        ml_prediction_engine=ml_prediction_engine,
        trend_anomaly_engine=trend_anomaly_engine,
        optimization_engine=optimization_engine,
        monitoring_engine=monitoring_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        performance_predictor=performance_predictor,
        configuration=configuration
    )

    # Erstelle Haupt-Engine
    enhanced_performance_analytics_engine = create_enhanced_performance_analytics_engine(
        monitoring_engine=monitoring_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        performance_predictor=performance_predictor,
        configuration=configuration
    )

    return {
        "enhanced_performance_analytics_engine": enhanced_performance_analytics_engine,
        "event_driven_analytics_engine": event_driven_analytics_engine,
        "ml_prediction_engine": ml_prediction_engine,
        "trend_anomaly_engine": trend_anomaly_engine,
        "optimization_engine": optimization_engine,
        "service_integration_layer": service_integration_layer,
        "configuration": configuration
    }
