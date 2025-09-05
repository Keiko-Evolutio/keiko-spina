# backend/services/enhanced_performance_analytics/data_models.py
"""Datenmodelle für Enhanced Performance Analytics.

Definiert alle Datenstrukturen für Enterprise-Grade Performance Analytics,
Event-driven Integration und ML-basierte Performance-Vorhersagen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from services.enhanced_security_integration import SecurityLevel


class AnalyticsScope(Enum):
    """Analytics-Scope für verschiedene Analysebereiche."""

    SYSTEM = "system"                    # System-weite Analytics
    SERVICE = "service"                  # Service-spezifische Analytics
    AGENT = "agent"                      # Agent-spezifische Analytics
    TASK = "task"                        # Task-spezifische Analytics
    ORCHESTRATION = "orchestration"      # Orchestration-Analytics
    DEPENDENCY = "dependency"            # Dependency-Analytics
    RESOURCE = "resource"                # Resource-Analytics
    SECURITY = "security"                # Security-Analytics
    LLM = "llm"                         # LLM-spezifische Analytics


class MetricDimension(Enum):
    """Metric-Dimensionen für Multi-dimensional Analytics."""

    TIME = "time"                        # Zeit-Dimension
    SERVICE = "service"                  # Service-Dimension
    USER = "user"                        # User-Dimension
    TENANT = "tenant"                    # Tenant-Dimension
    OPERATION = "operation"              # Operation-Dimension
    RESOURCE_TYPE = "resource_type"      # Resource-Type-Dimension
    SECURITY_LEVEL = "security_level"    # Security-Level-Dimension
    GEOGRAPHIC = "geographic"            # Geografische Dimension


class TrendDirection(Enum):
    """Trend-Richtungen für Trend-Analysis."""

    INCREASING = "increasing"            # Steigender Trend
    DECREASING = "decreasing"            # Fallender Trend
    STABLE = "stable"                    # Stabiler Trend
    VOLATILE = "volatile"                # Volatiler Trend
    SEASONAL = "seasonal"                # Saisonaler Trend
    ANOMALOUS = "anomalous"              # Anomaler Trend


class AnomalyType(Enum):
    """Anomaly-Typen für Anomaly-Detection."""

    SPIKE = "spike"                      # Performance-Spike
    DIP = "dip"                          # Performance-Dip
    DRIFT = "drift"                      # Performance-Drift
    OUTLIER = "outlier"                  # Performance-Outlier
    PATTERN_BREAK = "pattern_break"      # Pattern-Break
    THRESHOLD_BREACH = "threshold_breach" # Threshold-Breach


class OptimizationType(Enum):
    """Optimization-Typen für Performance-Optimization."""

    RESOURCE_SCALING = "resource_scaling"        # Resource-Scaling
    CACHE_OPTIMIZATION = "cache_optimization"   # Cache-Optimization
    LOAD_BALANCING = "load_balancing"           # Load-Balancing
    QUERY_OPTIMIZATION = "query_optimization"   # Query-Optimization
    ALGORITHM_TUNING = "algorithm_tuning"       # Algorithm-Tuning
    CONFIGURATION_TUNING = "configuration_tuning" # Configuration-Tuning


class EventType(Enum):
    """Event-Typen für Event-driven Architecture."""

    PERFORMANCE_METRIC = "performance_metric"   # Performance-Metrik-Event
    ANOMALY_DETECTED = "anomaly_detected"       # Anomaly-Detection-Event
    TREND_CHANGE = "trend_change"               # Trend-Change-Event
    OPTIMIZATION_TRIGGER = "optimization_trigger" # Optimization-Trigger-Event
    THRESHOLD_BREACH = "threshold_breach"       # Threshold-Breach-Event
    PREDICTION_UPDATE = "prediction_update"     # Prediction-Update-Event


@dataclass
class PerformanceDataPoint:
    """Performance-Datenpunkt für Analytics."""

    # Datenpunkt-Identifikation
    data_point_id: str
    metric_name: str
    scope: AnalyticsScope
    scope_id: str

    # Werte
    value: int | float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Dimensionen
    dimensions: dict[MetricDimension, str] = field(default_factory=dict)

    # Kontext
    service_name: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    orchestration_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None

    # Labels und Tags
    labels: dict[str, str] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvancedPerformanceMetrics:
    """Advanced Performance-Metriken für Multi-dimensional Analytics."""

    # Metrics-Identifikation
    metrics_id: str
    scope: AnalyticsScope
    scope_id: str

    # Zeitraum
    period_start: datetime
    period_end: datetime
    sample_count: int = 0

    # Response-Time-Metriken (erweitert)
    avg_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p75_response_time_ms: float = 0.0
    p90_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    p999_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    std_dev_response_time_ms: float = 0.0

    # Throughput-Metriken (erweitert)
    avg_throughput_rps: float = 0.0
    peak_throughput_rps: float = 0.0
    min_throughput_rps: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0

    # Error-Rate-Metriken (erweitert)
    error_rate: float = 0.0
    success_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0

    # Resource-Utilization-Metriken
    avg_cpu_usage_percent: float = 0.0
    peak_cpu_usage_percent: float = 0.0
    avg_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    avg_disk_io_mbps: float = 0.0
    avg_network_io_mbps: float = 0.0

    # Concurrency-Metriken
    avg_concurrent_requests: float = 0.0
    peak_concurrent_requests: int = 0
    queue_depth_avg: float = 0.0
    queue_depth_max: int = 0

    # Efficiency-Metriken
    efficiency_score: float = 0.0  # 0.0 - 1.0
    resource_efficiency: float = 0.0
    cost_efficiency: float = 0.0

    # SLA-Compliance
    sla_compliance_percent: float = 0.0
    sla_violations: int = 0
    availability_percent: float = 0.0

    # Multi-dimensional Breakdowns
    dimension_breakdowns: dict[MetricDimension, dict[str, float]] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend-Analysis für Performance-Metriken."""

    # Trend-Identifikation
    analysis_id: str
    metric_name: str
    scope: AnalyticsScope
    scope_id: str

    # Trend-Details
    trend_direction: TrendDirection
    trend_strength: float  # 0.0 - 1.0
    trend_confidence: float  # 0.0 - 1.0

    # Zeitraum
    analysis_period_start: datetime
    analysis_period_end: datetime
    data_points_count: int

    # Trend-Werte
    trend_slope: float
    trend_intercept: float
    r_squared: float

    # Vorhersagen
    predicted_next_value: float
    predicted_next_timestamp: datetime
    prediction_confidence: float

    # Seasonal-Analysis
    seasonal_pattern_detected: bool = False
    seasonal_period_hours: float | None = None
    seasonal_amplitude: float | None = None

    # Change-Points
    change_points: list[datetime] = field(default_factory=list)
    significant_changes: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """Anomaly-Detection für Performance-Metriken."""

    # Anomaly-Identifikation
    anomaly_id: str
    metric_name: str
    scope: AnalyticsScope
    scope_id: str

    # Anomaly-Details
    anomaly_type: AnomalyType
    severity: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0

    # Anomaly-Werte
    anomalous_value: float
    expected_value: float
    deviation: float
    deviation_percent: float

    # Zeitpunkt
    anomaly_start: datetime

    # Kontext
    baseline_period_start: datetime
    baseline_period_end: datetime
    baseline_sample_count: int

    # Detection-Algorithmus
    detection_algorithm: str

    # Optional fields with defaults
    detected_at: datetime = field(default_factory=datetime.utcnow)
    anomaly_end: datetime | None = None
    duration_seconds: float | None = None
    algorithm_parameters: dict[str, Any] = field(default_factory=dict)

    # Impact-Assessment
    impact_score: float = 0.0  # 0.0 - 1.0
    affected_services: list[str] = field(default_factory=list)
    affected_users: int = 0

    # Root-Cause-Hints
    potential_causes: list[str] = field(default_factory=list)
    correlated_anomalies: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceOptimizationRecommendation:
    """Performance-Optimization-Empfehlung."""

    # Recommendation-Identifikation
    recommendation_id: str
    scope: AnalyticsScope
    scope_id: str

    # Optimization-Details
    optimization_type: OptimizationType
    priority: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0

    # Problem-Description
    problem_description: str
    current_performance: dict[str, float]
    target_performance: dict[str, float]

    # Recommendation-Details
    recommendation_title: str
    recommendation_description: str
    implementation_steps: list[str]

    # Impact-Estimation
    estimated_improvement_percent: float
    estimated_cost: float
    estimated_effort_hours: float
    estimated_risk: float  # 0.0 - 1.0

    # Prerequisites
    prerequisites: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Validation
    validation_metrics: list[str] = field(default_factory=list)
    rollback_plan: str | None = None

    # Timing
    recommended_implementation_time: datetime | None = None
    estimated_duration_hours: float = 0.0

    # Status
    status: str = "pending"  # pending, approved, implemented, rejected
    implemented_at: datetime | None = None
    implementation_result: dict[str, Any] | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MLPerformancePrediction:
    """ML-basierte Performance-Vorhersage."""

    # Prediction-Identifikation
    prediction_id: str
    scope: AnalyticsScope
    scope_id: str

    # Prediction-Details
    metric_name: str
    predicted_value: float
    prediction_confidence: float  # 0.0 - 1.0

    # Zeitraum
    prediction_horizon_minutes: int
    target_timestamp: datetime

    # Model-Information
    model_id: str
    model_version: str
    model_type: str
    model_accuracy: float

    # Prediction-Bands
    lower_bound: float
    upper_bound: float

    # Historical-Context
    historical_accuracy: float
    similar_predictions_count: int
    baseline_comparison: float

    # Optional fields with defaults
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Features
    input_features: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)
    prediction_interval: float = 0.95  # 95% Confidence Interval

    # Uncertainty-Factors
    uncertainty_factors: list[str] = field(default_factory=list)
    data_quality_score: float = 1.0

    # Validation
    actual_value: float | None = None
    prediction_error: float | None = None
    validation_timestamp: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceEvent:
    """Performance-Event für Event-driven Architecture."""

    # Event-Identifikation
    event_id: str
    event_type: EventType
    event_name: str

    # Event-Source
    source_service: str
    source_scope: AnalyticsScope
    source_scope_id: str

    # Event-Payload
    payload: dict[str, Any] = field(default_factory=dict)

    # Event-Context
    correlation_id: str | None = None
    causation_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None

    # Event-Timing
    event_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_deadline: datetime | None = None

    # Event-Priority
    priority: int = 5  # 1 (highest) - 10 (lowest)
    requires_immediate_processing: bool = False

    # Event-Routing
    target_services: list[str] = field(default_factory=list)
    routing_key: str | None = None

    # Event-Metadata
    version: str = "1.0"
    schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsConfiguration:
    """Analytics-Konfiguration."""

    # Analytics-Einstellungen
    analytics_enabled: bool = True
    real_time_analytics_enabled: bool = True
    ml_predictions_enabled: bool = True
    trend_analysis_enabled: bool = True
    anomaly_detection_enabled: bool = True
    optimization_recommendations_enabled: bool = True

    # Performance-Einstellungen
    analytics_processing_timeout_ms: float = 100.0
    data_collection_interval_seconds: int = 10
    aggregation_interval_seconds: int = 60

    # ML-Einstellungen
    ml_prediction_interval_minutes: int = 15
    ml_model_retrain_interval_hours: int = 24
    ml_prediction_horizon_minutes: int = 60

    # Trend-Analysis-Einstellungen
    trend_analysis_window_hours: int = 24
    trend_detection_sensitivity: float = 0.1
    seasonal_detection_enabled: bool = True

    # Anomaly-Detection-Einstellungen
    anomaly_detection_sensitivity: float = 0.05
    anomaly_baseline_window_hours: int = 168  # 1 Woche
    anomaly_algorithms: list[str] = field(default_factory=lambda: ["statistical", "ml_based", "isolation_forest"])

    # Optimization-Einstellungen
    optimization_recommendation_threshold: float = 0.1  # 10% Verbesserung
    optimization_confidence_threshold: float = 0.8
    auto_optimization_enabled: bool = False

    # Event-driven-Einstellungen
    event_processing_enabled: bool = True
    event_batch_size: int = 100
    event_processing_interval_ms: int = 1000

    # Data-Retention-Einstellungen
    raw_data_retention_days: int = 30
    aggregated_data_retention_days: int = 365
    prediction_data_retention_days: int = 90

    # Security-Einstellungen
    analytics_security_level: SecurityLevel = SecurityLevel.INTERNAL
    audit_trail_enabled: bool = True
    compliance_monitoring_enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalyticsPerformanceMetrics:
    """Performance-Metriken für Analytics-System selbst."""

    # Analytics-Performance
    total_data_points_processed: int = 0
    avg_data_processing_time_ms: float = 0.0
    data_processing_error_rate: float = 0.0

    # ML-Prediction-Performance
    total_predictions_made: int = 0
    avg_prediction_time_ms: float = 0.0
    prediction_accuracy: float = 0.0
    prediction_error_rate: float = 0.0

    # Trend-Analysis-Performance
    total_trend_analyses: int = 0
    avg_trend_analysis_time_ms: float = 0.0
    trend_detection_accuracy: float = 0.0

    # Anomaly-Detection-Performance
    total_anomaly_detections: int = 0
    avg_anomaly_detection_time_ms: float = 0.0
    anomaly_detection_precision: float = 0.0
    anomaly_detection_recall: float = 0.0
    false_positive_rate: float = 0.0

    # Optimization-Performance
    total_recommendations_generated: int = 0
    avg_recommendation_generation_time_ms: float = 0.0
    recommendation_acceptance_rate: float = 0.0
    recommendation_effectiveness: float = 0.0

    # Event-Processing-Performance
    total_events_processed: int = 0
    avg_event_processing_time_ms: float = 0.0
    event_processing_throughput_eps: float = 0.0  # Events per second
    event_processing_error_rate: float = 0.0

    # SLA-Compliance
    meets_analytics_sla: bool = True
    analytics_sla_threshold_ms: float = 100.0

    # Zeitraum
    measurement_period_start: datetime = field(default_factory=datetime.utcnow)
    measurement_period_end: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0
