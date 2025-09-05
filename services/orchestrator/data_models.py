# backend/services/orchestrator/data_models.py
"""Datenmodelle für Orchestrator Service.

Definiert alle Datenstrukturen für Task-Orchestration,
Agent-Assignment, Execution-Monitoring und State-Management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from task_management.core_task_manager import TaskPriority, TaskState, TaskType


class OrchestrationState(Enum):
    """Orchestration-Status."""

    PENDING = "pending"           # Warten auf Start
    PLANNING = "planning"         # Task-Decomposition läuft
    SCHEDULED = "scheduled"       # Execution-Plan erstellt
    EXECUTING = "executing"       # Tasks werden ausgeführt
    PAUSED = "paused"            # Execution pausiert
    COMPLETED = "completed"       # Erfolgreich abgeschlossen
    FAILED = "failed"            # Fehlgeschlagen
    CANCELLED = "cancelled"       # Abgebrochen


class ExecutionMode(Enum):
    """Execution-Modi für Orchestration."""

    SEQUENTIAL = "sequential"     # Sequenzielle Ausführung
    PARALLEL = "parallel"        # Maximale Parallelisierung
    OPTIMIZED = "optimized"      # Intelligente Optimierung
    MANUAL = "manual"            # Manuelle Kontrolle


class AgentAssignmentStatus(Enum):
    """Status der Agent-Zuweisung."""

    PENDING = "pending"          # Warten auf Assignment
    ASSIGNED = "assigned"        # Agent zugewiesen
    EXECUTING = "executing"      # Agent führt aus
    COMPLETED = "completed"      # Erfolgreich abgeschlossen
    FAILED = "failed"           # Fehlgeschlagen
    RETRYING = "retrying"       # Wird wiederholt


@dataclass
class OrchestrationRequest:
    """Request für Task-Orchestration."""

    # Basis-Task-Information
    task_id: str
    task_type: TaskType = TaskType.AGENT_EXECUTION
    task_name: str = ""
    task_description: str = ""
    task_payload: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL

    # Orchestration-Konfiguration
    execution_mode: ExecutionMode = ExecutionMode.OPTIMIZED
    max_parallel_tasks: int = 5
    timeout_seconds: int = 3600  # 1 Stunde Default

    # Constraints
    required_capabilities: list[str] = field(default_factory=list)
    preferred_agents: list[str] = field(default_factory=list)
    excluded_agents: list[str] = field(default_factory=list)
    resource_constraints: dict[str, Any] = field(default_factory=dict)

    # Context
    user_id: str | None = None
    session_id: str | None = None
    tenant_id: str | None = None

    # Options
    enable_decomposition: bool = True
    enable_performance_prediction: bool = True
    enable_monitoring: bool = True
    enable_recovery: bool = True

    # Metadaten
    request_timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str | None = None


@dataclass
class AgentMatch:
    """Agent-Match für Subtask-Assignment."""

    # Agent-Identifikation
    agent_id: str
    agent_type: str

    # Match-Bewertung
    match_score: float  # 0.0 - 1.0
    confidence_score: float = 0.0

    # Capability-Matching
    matched_capabilities: list[str] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    capability_coverage: float = 0.0  # 0.0 - 1.0

    # Performance-Schätzung
    estimated_execution_time_ms: float = 0.0
    current_load: float = 0.0  # 0.0 - 1.0
    availability_score: float = 1.0  # 0.0 - 1.0
    queue_length: int = 0

    # Spezialisierung
    specialization_score: float = 0.0  # 0.0 - 1.0
    historical_success_rate: float = 1.0  # 0.0 - 1.0

    # Metadaten
    match_timestamp: datetime = field(default_factory=datetime.utcnow)
    match_reason: str | None = None


@dataclass
class SubtaskExecution:
    """Execution-Status eines Subtasks."""

    # Subtask-Information
    subtask_id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    payload: dict[str, Any]
    required_capabilities: list[str] = field(default_factory=list)
    estimated_duration_minutes: float | None = None

    # Agent-Assignment
    assigned_agent_id: str | None = None
    agent_assignment_status: AgentAssignmentStatus = AgentAssignmentStatus.PENDING
    assignment_timestamp: datetime | None = None

    # Execution-Status
    state: TaskState = TaskState.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    execution_time_ms: float | None = None

    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    blocking_subtasks: list[str] = field(default_factory=list)

    # Results
    result: dict[str, Any] | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    # Performance-Metriken
    predicted_execution_time_ms: float | None = None
    prediction_confidence: float | None = None
    performance_variance: float | None = None

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentLoadInfo:
    """Agent-Load-Informationen für Load-Balancing."""

    agent_id: str
    agent_type: str
    capabilities: list[str]

    # Load-Metriken
    current_load: float  # 0-1 Skala
    active_tasks: int
    queued_tasks: int
    max_concurrent_tasks: int

    # Performance-Metriken
    avg_response_time_ms: float
    success_rate: float
    error_rate: float

    # Availability
    is_available: bool
    last_heartbeat: datetime
    health_status: str = "healthy"

    # Specialization
    specialization_scores: dict[str, float] = field(default_factory=dict)

    # Metadaten
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionPlan:
    """Execution-Plan für Orchestration."""

    # Plan-Identifikation
    plan_id: str
    orchestration_id: str

    # Subtasks
    subtasks: list[SubtaskExecution]

    # Execution-Reihenfolge
    execution_groups: list[list[str]]  # Gruppen von parallel ausführbaren Subtask-IDs
    critical_path: list[str]

    # Agent-Assignments
    agent_assignments: dict[str, str]  # subtask_id -> agent_id
    load_distribution: dict[str, AgentLoadInfo]

    # Performance-Schätzungen
    estimated_total_duration_ms: float
    estimated_parallel_duration_ms: float
    parallelization_efficiency: float

    # Resource-Planung
    peak_resource_usage: dict[str, float]
    resource_timeline: list[dict[str, Any]]

    # Execution-Strategy (mit Default-Wert)
    strategy: ExecutionMode = ExecutionMode.OPTIMIZED

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "orchestrator_service"

    @property
    def estimated_total_duration_minutes(self) -> float:
        """Konvertiert Millisekunden zu Minuten für Kompatibilität."""
        return self.estimated_total_duration_ms / 60000.0


@dataclass
class OrchestrationProgress:
    """Progress-Tracking für Orchestration."""

    orchestration_id: str
    state: OrchestrationState

    # Progress-Metriken
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    running_subtasks: int
    pending_subtasks: int

    # Timing
    start_time: datetime | None = None
    estimated_completion_time: datetime | None = None
    actual_completion_time: datetime | None = None

    # Performance
    execution_efficiency: float = 0.0  # 0-1 Skala
    resource_utilization: float = 0.0  # 0-1 Skala

    # Errors
    error_count: int = 0
    last_error: str | None = None

    # Metadaten
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def completion_percentage(self) -> float:
        """Berechnet Completion-Percentage."""
        if self.total_subtasks == 0:
            return 0.0
        return (self.completed_subtasks / self.total_subtasks) * 100.0

    @property
    def is_completed(self) -> bool:
        """Prüft ob Orchestration abgeschlossen ist."""
        return self.state in [OrchestrationState.COMPLETED, OrchestrationState.FAILED, OrchestrationState.CANCELLED]


@dataclass
class OrchestrationResult:
    """Ergebnis einer Orchestration."""

    # Status
    success: bool
    orchestration_id: str
    state: OrchestrationState

    # Results
    results: dict[str, Any] = field(default_factory=dict)  # subtask_id -> result
    aggregated_result: dict[str, Any] | None = None

    # Additional result attributes for orchestrator service compatibility
    result_summary: str | None = None
    result_data: dict[str, Any] | None = None
    completed_subtasks: list[dict[str, Any]] = field(default_factory=list)

    # Execution Plan
    plan: ExecutionPlan | None = None

    # Performance-Metriken
    total_execution_time_ms: float = 0.0
    orchestration_overhead_ms: float = 0.0
    parallelization_achieved: float = 0.0

    # Subtask-Details
    subtask_results: list[SubtaskExecution] = field(default_factory=list)
    failed_subtasks: list[str] = field(default_factory=list)

    # Agent-Performance
    agent_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    # Errors
    error_message: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)

    # Metadaten
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HealthCheckResult:
    """Health-Check-Ergebnis für Orchestrator Service."""

    # Service-Status
    service_healthy: bool
    service_version: str
    uptime_seconds: float

    # Komponenten-Status
    task_decomposition_healthy: bool
    performance_prediction_healthy: bool
    agent_registry_healthy: bool
    task_manager_healthy: bool

    # Performance-Metriken
    active_orchestrations: int
    total_orchestrations: int
    avg_orchestration_time_ms: float
    success_rate: float

    # Resource-Status
    memory_usage_mb: float
    cpu_usage_percent: float

    # Dependencies
    dependencies_healthy: dict[str, bool] = field(default_factory=dict)

    # Metadaten
    check_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrchestrationEvent:
    """Event für Orchestration-Monitoring."""

    # Event-Identifikation
    event_id: str
    orchestration_id: str
    event_type: str  # "started", "subtask_completed", "failed", etc.

    # Event-Daten
    subtask_id: str | None = None
    agent_id: str | None = None
    event_data: dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    user_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
