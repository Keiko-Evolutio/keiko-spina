"""Edge Computing Types für Backend

Definiert alle Datentypen für Edge Computing-Backend-Komponenten.
Kompatibel mit Frontend Edge Types für einheitliche API.

@version 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

# =============================================================================
# Edge Node Types
# =============================================================================

class EdgeNodeType(str, Enum):
    """Edge-Node-Typen."""
    AUDIO_PROCESSOR = "audio-processor"
    AI_INFERENCE = "ai-inference"
    CACHE_NODE = "cache-node"
    LOAD_BALANCER = "load-balancer"
    GATEWAY = "gateway"

class EdgeNodeStatus(str, Enum):
    """Edge-Node-Status."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class EdgeProcessingCapability(str, Enum):
    """Edge-Processing-Fähigkeiten."""
    VOICE_ACTIVITY_DETECTION = "vad"
    NOISE_REDUCTION = "noise-reduction"
    AUDIO_ENHANCEMENT = "audio-enhancement"
    SPEECH_RECOGNITION = "speech-recognition"
    SPEAKER_IDENTIFICATION = "speaker-identification"
    EMOTION_DETECTION = "emotion-detection"
    LANGUAGE_DETECTION = "language-detection"

# =============================================================================
# Edge Node Information
# =============================================================================

@dataclass
class EdgeNodeInfo:
    """Edge-Node-Informationen."""
    node_id: str
    node_type: EdgeNodeType
    endpoint: str
    region: str
    status: EdgeNodeStatus = EdgeNodeStatus.INITIALIZING

    # Performance-Metriken
    latency_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_bandwidth_mbps: float = 0.0

    # Kapazitäten
    max_concurrent_tasks: int = 10
    current_task_count: int = 0
    available_capacity: float = 1.0

    # Fähigkeiten
    supported_capabilities: list[EdgeProcessingCapability] = field(default_factory=list)
    supported_models: list[str] = field(default_factory=list)
    supported_formats: list[str] = field(default_factory=list)

    # Metadaten
    version: str = "1.0.0"
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeNodeHealthCheck:
    """Edge-Node-Health-Check-Ergebnis."""
    node_id: str
    timestamp: datetime
    status: EdgeNodeStatus
    response_time_ms: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    error_rate: float
    last_error: str | None = None

# =============================================================================
# Task Processing Types
# =============================================================================

class EdgeTaskType(str, Enum):
    """Edge-Task-Typen."""
    AUDIO_PROCESSING = "audio-processing"
    AI_INFERENCE = "ai-inference"
    DATA_TRANSFORMATION = "data-transformation"
    CACHE_OPERATION = "cache-operation"
    HEALTH_CHECK = "health-check"

class EdgeTaskPriority(str, Enum):
    """Edge-Task-Prioritäten."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class EdgeTaskStatus(str, Enum):
    """Edge-Task-Status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class EdgeTask:
    """Edge-Task-Definition."""
    task_id: str
    task_type: EdgeTaskType
    priority: EdgeTaskPriority

    # Task-Daten
    input_data: bytes
    processing_params: dict[str, Any]
    expected_output_format: str

    # Scheduling
    assigned_node_id: str | None = None
    status: EdgeTaskStatus = EdgeTaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    assigned_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Constraints
    deadline: datetime | None = None
    max_processing_time_ms: int = 30000  # 30 Sekunden default
    required_capabilities: list[EdgeProcessingCapability] = field(default_factory=list)
    preferred_regions: list[str] = field(default_factory=list)

    # Dependencies
    dependencies: list[str] = field(default_factory=list)

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeTaskResult:
    """Edge-Task-Ergebnis."""
    task_id: str
    node_id: str
    status: EdgeTaskStatus

    # Success-Flag für einfache Überprüfung
    success: bool = False

    # Ergebnis-Daten
    output_data: bytes | None = None
    output_metadata: dict[str, Any] = field(default_factory=dict)

    # Performance-Metriken
    processing_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Ressourcen-Nutzung
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    network_io_mb: float = 0.0

    # Error-Informationen
    error_message: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

# =============================================================================
# Load Balancing Types
# =============================================================================

class LoadBalancingStrategy(str, Enum):
    """Load-Balancing-Strategien."""
    ROUND_ROBIN = "round-robin"
    LEAST_CONNECTIONS = "least-connections"
    WEIGHTED_ROUND_ROBIN = "weighted-round-robin"
    LATENCY_BASED = "latency-based"
    CAPACITY_BASED = "capacity-based"
    ADAPTIVE = "adaptive"
    GEOGRAPHIC = "geographic"

@dataclass
class LoadBalancingWeights:
    """Gewichtungen für Load Balancing."""
    latency: float = 0.4
    capacity: float = 0.3
    reliability: float = 0.2
    cost: float = 0.1

@dataclass
class RoutingDecision:
    """Routing-Entscheidung."""
    task_id: str
    selected_node_id: str
    strategy_used: LoadBalancingStrategy
    decision_time_ms: float
    confidence_score: float
    alternative_nodes: list[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

# =============================================================================
# Performance Monitoring Types
# =============================================================================

@dataclass
class EdgeMetrics:
    """Edge-Performance-Metriken."""
    timestamp: datetime

    # Latenz-Metriken
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Durchsatz-Metriken
    tasks_per_second: float = 0.0
    bytes_per_second: float = 0.0
    requests_per_second: float = 0.0

    # Erfolgsraten
    success_rate: float = 1.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    # Ressourcen-Nutzung
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    avg_disk_usage: float = 0.0
    avg_network_usage: float = 0.0

    # Node-Metriken
    total_nodes: int = 0
    healthy_nodes: int = 0
    active_tasks: int = 0
    queued_tasks: int = 0

    # Cache-Metriken
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_size_mb: float = 0.0

@dataclass
class EdgeAlert:
    """Edge-Alert."""
    alert_id: str
    alert_type: str
    severity: str  # info, warning, error, critical
    message: str
    node_id: str | None = None
    metric_name: str | None = None
    threshold_value: float | None = None
    current_value: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved: bool = False
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Auto-Scaling Types
# =============================================================================

class ScalingAction(str, Enum):
    """Scaling-Aktionen."""
    SCALE_UP = "scale-up"
    SCALE_DOWN = "scale-down"
    SCALE_OUT = "scale-out"
    SCALE_IN = "scale-in"
    NO_ACTION = "no-action"

@dataclass
class ScalingRule:
    """Auto-Scaling-Regel."""
    rule_id: str
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==
    action: ScalingAction
    cooldown_period_seconds: int = 300
    enabled: bool = True

    # Scaling-Parameter
    scale_factor: float = 1.5
    min_nodes: int = 1
    max_nodes: int = 10

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingEvent:
    """Scaling-Event."""
    event_id: str
    rule_id: str
    action: ScalingAction
    trigger_metric: str
    trigger_value: float
    threshold_value: float
    nodes_before: int
    nodes_after: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    success: bool = True
    error_message: str | None = None

# =============================================================================
# Configuration Types
# =============================================================================

@dataclass
class EdgeConfiguration:
    """Edge-System-Konfiguration."""

    # Node Management
    node_registry_url: str = "http://localhost:8080"
    node_heartbeat_interval_seconds: int = 30
    node_timeout_seconds: int = 120
    max_nodes_per_region: int = 50

    # Task Processing
    max_concurrent_tasks_per_node: int = 10
    task_timeout_seconds: int = 30
    task_retry_attempts: int = 3
    task_queue_size: int = 1000

    # Load Balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    load_balancing_weights: LoadBalancingWeights = field(default_factory=LoadBalancingWeights)
    routing_cache_ttl_seconds: int = 60

    # Performance Monitoring
    metrics_collection_interval_seconds: int = 10
    metrics_retention_days: int = 7
    alert_evaluation_interval_seconds: int = 30

    # Auto-Scaling
    auto_scaling_enabled: bool = True
    scaling_evaluation_interval_seconds: int = 60
    scaling_cooldown_period_seconds: int = 300
    min_nodes_global: int = 2
    max_nodes_global: int = 100

    # Caching
    cache_enabled: bool = True
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 3600

    # Security
    enable_authentication: bool = True
    enable_encryption: bool = True
    api_key_required: bool = True

    # Debugging
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

# =============================================================================
# API Request/Response Types
# =============================================================================

class EdgeNodeRegistrationRequest(BaseModel):
    """Edge-Node-Registrierungs-Request."""
    node_type: EdgeNodeType
    endpoint: str
    region: str
    capabilities: list[EdgeProcessingCapability]
    supported_models: list[str] = []
    supported_formats: list[str] = []
    max_concurrent_tasks: int = 10
    metadata: dict[str, Any] = {}

class EdgeTaskSubmissionRequest(BaseModel):
    """Edge-Task-Submission-Request."""
    task_type: EdgeTaskType
    priority: EdgeTaskPriority = EdgeTaskPriority.NORMAL
    input_data: bytes
    processing_params: dict[str, Any] = {}
    expected_output_format: str = "json"
    deadline: datetime | None = None
    max_processing_time_ms: int = 30000
    required_capabilities: list[EdgeProcessingCapability] = []
    preferred_regions: list[str] = []
    metadata: dict[str, Any] = {}

class EdgeTaskStatusResponse(BaseModel):
    """Edge-Task-Status-Response."""
    task_id: str
    status: EdgeTaskStatus
    assigned_node_id: str | None = None
    progress_percentage: float = 0.0
    estimated_completion_time: datetime | None = None
    created_at: datetime
    updated_at: datetime

class EdgeMetricsResponse(BaseModel):
    """Edge-Metrics-Response."""
    timestamp: datetime
    metrics: EdgeMetrics
    node_metrics: dict[str, dict[str, float]] = {}
    alerts: list[EdgeAlert] = []

# =============================================================================
# Error Types
# =============================================================================

class EdgeComputingError(Exception):
    """Basis Edge Computing Error."""

    def __init__(
        self,
        message: str,
        code: str = "EDGE_ERROR",
        context: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.now(UTC)

class EdgeNodeError(EdgeComputingError):
    """Edge-Node-spezifischer Error."""

    def __init__(self, message: str, node_id: str, context: dict[str, Any] | None = None):
        super().__init__(message, "EDGE_NODE_ERROR", {"node_id": node_id, **(context or {})})
        self.node_id = node_id

class EdgeTaskError(EdgeComputingError):
    """Edge-Task-spezifischer Error."""

    def __init__(self, message: str, task_id: str, context: dict[str, Any] | None = None):
        super().__init__(message, "EDGE_TASK_ERROR", {"task_id": task_id, **(context or {})})
        self.task_id = task_id

class EdgeLoadBalancingError(EdgeComputingError):
    """Load-Balancing-spezifischer Error."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message, "EDGE_LOAD_BALANCING_ERROR", context)

class EdgeScalingError(EdgeComputingError):
    """Auto-Scaling-spezifischer Error."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message, "EDGE_SCALING_ERROR", context)

# =============================================================================
# Utility Functions
# =============================================================================

def create_edge_node_id(node_type: EdgeNodeType, region: str) -> str:
    """Erstellt eine eindeutige Edge-Node-ID."""
    import uuid
    timestamp = int(datetime.now(UTC).timestamp())
    return f"{node_type.value}-{region}-{timestamp}-{uuid.uuid4().hex[:8]}"

def create_edge_task_id() -> str:
    """Erstellt eine eindeutige Edge-Task-ID."""
    import uuid
    return f"task-{uuid.uuid4().hex}"

def calculate_node_score(
    node: EdgeNodeInfo,
    weights: LoadBalancingWeights
) -> float:
    """Berechnet einen Score für eine Edge-Node basierend auf Gewichtungen."""
    # Normalisierte Scores (0-1, höher = besser)
    latency_score = max(0, 1 - (node.latency_ms / 1000))  # <1s = gut
    capacity_score = node.available_capacity
    reliability_score = 1.0 if node.status == EdgeNodeStatus.HEALTHY else 0.5
    cost_score = 1.0  # Vereinfacht, könnte basierend auf Node-Typ variieren

    return (
        latency_score * weights.latency +
        capacity_score * weights.capacity +
        reliability_score * weights.reliability +
        cost_score * weights.cost
    )

def is_node_suitable_for_task(node: EdgeNodeInfo, task: EdgeTask) -> bool:
    """Prüft, ob eine Node für eine Task geeignet ist."""
    # Kapazitäts-Check
    if node.current_task_count >= node.max_concurrent_tasks:
        return False

    # Status-Check
    if node.status != EdgeNodeStatus.HEALTHY:
        return False

    # Capability-Check
    if task.required_capabilities:
        if not all(cap in node.supported_capabilities for cap in task.required_capabilities):
            return False

    # Region-Check
    if task.preferred_regions:
        if node.region not in task.preferred_regions:
            return False

    return True
