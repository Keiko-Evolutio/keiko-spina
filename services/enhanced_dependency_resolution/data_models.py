# backend/services/enhanced_dependency_resolution/data_models.py
"""Datenmodelle für Enhanced Dependency Resolution.

Definiert alle Datenstrukturen für Enterprise-Grade Dependency-Management,
Intelligent Dependency Graph Analysis und Real-time Dependency-Tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from services.enhanced_security_integration import SecurityLevel


class DependencyType(Enum):
    """Dependency-Typen für verschiedene Abhängigkeiten."""

    TASK = "task"                        # Task-Dependencies
    RESOURCE = "resource"                # Resource-Dependencies
    AGENT = "agent"                      # Agent-Dependencies
    DATA = "data"                        # Data-Dependencies
    SERVICE = "service"                  # Service-Dependencies
    CAPABILITY = "capability"            # Capability-Dependencies
    QUOTA = "quota"                      # Quota-Dependencies
    SECURITY = "security"                # Security-Dependencies


class DependencyRelation(Enum):
    """Dependency-Relationen."""

    REQUIRES = "requires"                # Benötigt
    PROVIDES = "provides"                # Stellt bereit
    CONFLICTS = "conflicts"              # Konflikt
    ENHANCES = "enhances"                # Verbessert
    REPLACES = "replaces"                # Ersetzt
    OPTIONAL = "optional"                # Optional
    CONDITIONAL = "conditional"          # Bedingt


class DependencyStatus(Enum):
    """Dependency-Status."""

    PENDING = "pending"                  # Wartend
    RESOLVING = "resolving"              # Wird aufgelöst
    RESOLVED = "resolved"                # Aufgelöst
    FAILED = "failed"                    # Fehlgeschlagen
    CIRCULAR = "circular"                # Zirkulär
    TIMEOUT = "timeout"                  # Timeout
    CACHED = "cached"                    # Gecacht


class ResolutionStrategy(Enum):
    """Dependency-Resolution-Strategien."""

    EAGER = "eager"                      # Sofortige Auflösung
    LAZY = "lazy"                        # Verzögerte Auflösung
    PARALLEL = "parallel"                # Parallele Auflösung
    SEQUENTIAL = "sequential"            # Sequenzielle Auflösung
    CACHED = "cached"                    # Cache-basierte Auflösung
    FALLBACK = "fallback"                # Fallback-Strategie


class CircularResolutionStrategy(Enum):
    """Strategien für Circular-Dependency-Resolution."""

    BREAK_WEAKEST = "break_weakest"      # Schwächste Dependency brechen
    BREAK_OPTIONAL = "break_optional"    # Optionale Dependencies brechen
    MERGE_CYCLES = "merge_cycles"        # Zyklen zusammenführen
    FAIL_FAST = "fail_fast"              # Sofort fehlschlagen
    IGNORE = "ignore"                    # Ignorieren (gefährlich)


@dataclass
class DependencyNode:
    """Dependency-Graph-Node."""

    # Node-Identifikation
    node_id: str
    node_type: DependencyType
    name: str
    description: str

    # Node-Eigenschaften
    version: str | None = None
    priority: int = 0
    weight: float = 1.0

    # Dependencies
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

    # Status
    status: DependencyStatus = DependencyStatus.PENDING

    # Zeitstempel
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DependencyEdge:
    """Dependency-Graph-Edge."""

    # Edge-Identifikation
    edge_id: str
    source_node_id: str
    target_node_id: str

    # Edge-Eigenschaften
    relation: DependencyRelation
    dependency_type: DependencyType
    weight: float = 1.0
    priority: int = 0

    # Constraints
    version_constraint: str | None = None
    condition: str | None = None
    timeout_seconds: int | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Status
    status: DependencyStatus = DependencyStatus.PENDING

    # Zeitstempel
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DependencyGraph:
    """Dependency-Graph."""

    # Graph-Identifikation
    graph_id: str
    name: str
    description: str

    # Graph-Komponenten
    nodes: dict[str, DependencyNode] = field(default_factory=dict)
    edges: dict[str, DependencyEdge] = field(default_factory=dict)

    # Graph-Eigenschaften
    is_acyclic: bool = True
    has_circular_dependencies: bool = False
    circular_cycles: list[list[str]] = field(default_factory=list)

    # Resolution-Konfiguration
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.EAGER
    circular_resolution_strategy: CircularResolutionStrategy = CircularResolutionStrategy.BREAK_WEAKEST

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Zeitstempel
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DependencyResolutionRequest:
    """Dependency-Resolution-Request."""

    # Request-Identifikation
    request_id: str
    graph_id: str

    # Resolution-Parameter
    target_nodes: list[str]
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.EAGER
    max_depth: int = 10
    timeout_seconds: int = 300

    # Constraints
    include_optional: bool = True
    break_circular: bool = True
    use_cache: bool = True

    # Security-Kontext
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    user_id: str | None = None
    tenant_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Zeitstempel
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DependencyResolutionResult:
    """Dependency-Resolution-Result."""

    # Result-Identifikation
    result_id: str
    request_id: str

    # Resolution-Status
    success: bool
    resolution_order: list[str] = field(default_factory=list)
    resolved_nodes: set[str] = field(default_factory=set)
    failed_nodes: set[str] = field(default_factory=set)

    # Circular-Dependencies
    circular_dependencies: list[list[str]] = field(default_factory=list)
    broken_dependencies: list[str] = field(default_factory=list)

    # Performance-Metriken
    resolution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    # Errors und Warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Zeitstempel
    resolved_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CircularDependency:
    """Circular-Dependency-Definition."""

    # Cycle-Identifikation
    cycle_id: str
    graph_id: str

    # Cycle-Komponenten
    cycle_nodes: list[str]
    cycle_edges: list[str]
    cycle_length: int

    # Cycle-Eigenschaften
    is_strong_cycle: bool = True
    cycle_weight: float = 0.0
    weakest_edge_id: str | None = None

    # Resolution-Informationen
    resolution_strategy: CircularResolutionStrategy = CircularResolutionStrategy.BREAK_WEAKEST
    can_be_broken: bool = True
    break_cost: float = 0.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Zeitstempel
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DependencyCache:
    """Dependency-Cache-Entry."""

    # Cache-Identifikation
    cache_key: str
    graph_id: str

    # Cache-Daten
    resolution_result: DependencyResolutionResult
    cache_version: str = "1.0"

    # Cache-Eigenschaften
    ttl_seconds: int = 3600
    access_count: int = 0

    # Invalidation-Triggers
    invalidation_triggers: set[str] = field(default_factory=set)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Zeitstempel
    cached_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DependencyAnalytics:
    """Dependency-Analytics-Daten."""

    # Analytics-Identifikation
    analytics_id: str
    graph_id: str

    # Zeitraum
    period_start: datetime
    period_end: datetime

    # Resolution-Statistiken
    total_resolutions: int = 0
    successful_resolutions: int = 0
    failed_resolutions: int = 0
    circular_resolutions: int = 0

    # Performance-Statistiken
    avg_resolution_time_ms: float = 0.0
    p95_resolution_time_ms: float = 0.0
    p99_resolution_time_ms: float = 0.0

    # Cache-Statistiken
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0

    # Graph-Statistiken
    avg_graph_size: float = 0.0
    max_graph_depth: int = 0
    most_common_dependencies: list[str] = field(default_factory=list)

    # Trends
    resolution_trend: str = "stable"  # increasing, decreasing, stable, volatile

    # Metadaten
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyPerformanceMetrics:
    """Performance-Metriken für Dependency-Management."""

    # Resolution-Performance
    total_resolutions: int = 0
    avg_resolution_time_ms: float = 0.0
    p95_resolution_time_ms: float = 0.0
    p99_resolution_time_ms: float = 0.0

    # Graph-Analysis-Performance
    total_graph_analyses: int = 0
    avg_graph_analysis_time_ms: float = 0.0

    # Circular-Detection-Performance
    total_circular_detections: int = 0
    avg_circular_detection_time_ms: float = 0.0

    # Cache-Performance
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_size: int = 0

    # SLA-Compliance
    meets_resolution_sla: bool = True
    resolution_sla_threshold_ms: float = 100.0

    # Error-Rates
    resolution_error_rate: float = 0.0
    circular_dependency_rate: float = 0.0

    # Metadaten
    measurement_period_start: datetime = field(default_factory=datetime.utcnow)
    measurement_period_end: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0


@dataclass
class TaskDependencyContext:
    """Task-Dependency-Context für Integration mit Task Decomposition Engine."""

    # Task-Kontext
    task_id: str
    subtask_id: str | None = None
    orchestration_id: str | None = None

    # Task-Dependencies
    required_tasks: list[str] = field(default_factory=list)
    required_agents: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    required_resources: list[str] = field(default_factory=list)

    # Quota-Dependencies
    required_quotas: list[str] = field(default_factory=list)
    quota_constraints: dict[str, Any] = field(default_factory=dict)

    # Security-Dependencies
    security_requirements: list[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.INTERNAL

    # Performance-Requirements
    max_resolution_time_ms: float = 100.0
    priority: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceDependencyContext:
    """Resource-Dependency-Context für Integration mit Quotas & Limits Management."""

    # Resource-Kontext
    resource_id: str
    resource_type: str
    tenant_id: str | None = None

    # Resource-Dependencies
    required_resources: list[str] = field(default_factory=list)
    resource_constraints: dict[str, Any] = field(default_factory=dict)

    # Quota-Dependencies
    quota_requirements: list[str] = field(default_factory=list)
    rate_limit_requirements: list[str] = field(default_factory=list)

    # Availability-Requirements
    min_availability: float = 0.99
    max_latency_ms: float = 100.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
