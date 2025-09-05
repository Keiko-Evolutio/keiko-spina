# backend/services/task_decomposition/data_models.py
"""Datenmodelle für LLM-powered Task Decomposition Engine.

Definiert alle Datenstrukturen für Task-Analyse, Decomposition,
Agent-Matching und Dependency-Resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from task_management.core_task_manager import TaskPriority, TaskType


class ComplexityLevel(Enum):
    """Task-Komplexitäts-Level."""

    TRIVIAL = "trivial"      # 1-2 Komplexität
    SIMPLE = "simple"        # 3-4 Komplexität
    MODERATE = "moderate"    # 5-6 Komplexität
    COMPLEX = "complex"      # 7-8 Komplexität
    CRITICAL = "critical"    # 9-10 Komplexität


class DecompositionStrategy(Enum):
    """Decomposition-Strategien."""

    SEQUENTIAL = "sequential"        # Sequenzielle Ausführung
    PARALLEL = "parallel"           # Parallele Ausführung
    PIPELINE = "pipeline"           # Pipeline-Verarbeitung
    HYBRID = "hybrid"               # Gemischte Strategie


class DependencyType(Enum):
    """Dependency-Typen zwischen Subtasks."""

    HARD = "hard"                   # Harte Abhängigkeit (blockierend)
    SOFT = "soft"                   # Weiche Abhängigkeit (bevorzugt)
    DATA = "data"                   # Daten-Abhängigkeit
    RESOURCE = "resource"           # Resource-Abhängigkeit
    TEMPORAL = "temporal"           # Zeit-Abhängigkeit


@dataclass
class TaskAnalysis:
    """Ergebnis der LLM-basierten Task-Analyse."""

    # Basis-Analyse
    complexity_score: float  # 1-10 Skala
    complexity_level: ComplexityLevel
    estimated_duration_minutes: float

    # Decomposition-Empfehlung
    is_decomposable: bool
    recommended_strategy: DecompositionStrategy
    decomposition_confidence: float  # 0-1 Skala

    # Capability-Anforderungen
    required_capabilities: list[str]
    optional_capabilities: list[str]
    specialized_skills: list[str]

    # Resource-Schätzung
    estimated_cpu_usage: float  # 0-1 Skala
    estimated_memory_mb: int
    estimated_network_io: bool
    estimated_disk_io: bool

    # Parallelisierung
    parallel_potential: float  # 0-1 Skala
    max_parallel_subtasks: int
    bottleneck_factors: list[str]

    # Risk-Assessment
    risk_factors: list[str]
    failure_probability: float  # 0-1 Skala
    rollback_complexity: float  # 0-1 Skala

    # LLM-Metadaten
    analysis_model: str
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    analysis_confidence: float = 0.0  # 0-1 Skala


@dataclass
class SubtaskDefinition:
    """Definition eines Subtasks."""

    # Identifikation
    subtask_id: str
    name: str
    description: str

    # Task-Properties
    task_type: TaskType
    priority: TaskPriority
    estimated_duration_minutes: float
    required_capabilities: list[str]
    optional_capabilities: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    # Execution-Eigenschaften
    preferred_agent_types: list[str] = field(default_factory=list)

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Subtask-IDs
    dependency_types: dict[str, DependencyType] = field(default_factory=dict)

    # Parallelisierung
    can_run_parallel: bool = True
    parallel_group: str | None = None

    # Validation
    success_criteria: list[str] = field(default_factory=list)
    validation_rules: dict[str, Any] = field(default_factory=dict)

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentMatch:
    """Agent-Matching-Ergebnis für Subtask."""

    agent_id: str
    agent_type: str
    match_score: float  # 0-1 Skala

    # Capability-Matching
    matched_capabilities: list[str]
    missing_capabilities: list[str]
    capability_coverage: float  # 0-1 Skala

    # Performance-Schätzung
    estimated_execution_time_ms: float
    confidence_score: float  # 0-1 Skala

    # Load-Informationen
    current_load: float  # 0-1 Skala
    availability_score: float  # 0-1 Skala
    queue_length: int

    # Spezialisierung
    specialization_score: float  # 0-1 Skala für Task-Type
    historical_success_rate: float  # 0-1 Skala

    # Metadaten
    match_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecompositionPlan:
    """Vollständiger Decomposition-Plan."""

    # Plan-Identifikation
    plan_id: str
    original_task_id: str

    # Subtasks
    subtasks: list[SubtaskDefinition]

    # Execution-Plan
    execution_strategy: DecompositionStrategy
    execution_order: list[list[str]]  # Execution-Gruppen (parallel innerhalb, sequenziell zwischen)

    # Agent-Assignments
    agent_assignments: dict[str, AgentMatch]  # subtask_id -> AgentMatch

    # Dependencies
    dependency_graph: dict[str, list[str]]  # subtask_id -> [dependent_subtask_ids]
    critical_path: list[str]  # Kritischer Pfad durch Dependencies

    # Performance-Schätzung
    estimated_total_duration_minutes: float
    estimated_parallel_duration_minutes: float
    parallelization_efficiency: float  # 0-1 Skala

    # Validation
    plan_confidence: float  # 0-1 Skala
    validation_results: dict[str, bool]
    potential_issues: list[str]

    # Fallback
    fallback_strategy: str | None = None
    rollback_plan: list[str] = field(default_factory=list)

    # Metadaten
    created_by: str = "llm_decomposition_engine"
    created_at: datetime = field(default_factory=datetime.utcnow)
    llm_model_used: str = ""
    analysis_used: TaskAnalysis | None = None


@dataclass
class DecompositionRequest:
    """Request für Task-Decomposition."""

    # Original Task
    task_id: str
    task_type: TaskType
    task_name: str
    task_description: str
    task_payload: dict[str, Any]
    task_priority: TaskPriority

    # Context
    user_id: str | None = None
    session_id: str | None = None
    deadline: datetime | None = None

    # Preferences
    preferred_strategy: DecompositionStrategy | None = None
    max_subtasks: int = 20
    max_parallel_subtasks: int = 5

    # Constraints
    available_agents: list[str] | None = None
    required_capabilities: list[str] | None = None
    resource_constraints: dict[str, Any] = field(default_factory=dict)

    # Options
    enable_llm_analysis: bool = True
    enable_performance_prediction: bool = True
    enable_agent_matching: bool = True

    # Metadaten
    request_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecompositionResult:
    """Ergebnis der Task-Decomposition."""

    # Status
    success: bool
    error_message: str | None = None

    # Ergebnisse
    analysis: TaskAnalysis | None = None
    plan: DecompositionPlan | None = None

    # Fallback-Information
    used_fallback: bool = False
    fallback_reason: str | None = None
    rule_based_plan: dict[str, Any] | None = None

    # Performance-Metriken
    decomposition_time_ms: float = 0.0
    llm_analysis_time_ms: float = 0.0
    agent_matching_time_ms: float = 0.0

    # Metadaten
    engine_version: str = "1.0.0"
    result_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Ergebnis der Plan-Validation."""

    is_valid: bool
    validation_score: float  # 0-1 Skala

    # Validation-Checks
    completeness_check: bool = False
    dependency_check: bool = False
    capability_check: bool = False
    resource_check: bool = False
    timing_check: bool = False

    # Issues
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # Fixes
    auto_fixes_applied: list[str] = field(default_factory=list)
    manual_fixes_required: list[str] = field(default_factory=list)

    # Metadaten
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FallbackRule:
    """Regel für regelbasierte Fallback-Decomposition."""

    rule_id: str
    name: str
    description: str

    # Trigger-Bedingungen
    task_type_patterns: list[str]
    complexity_range: tuple[float, float]  # (min, max)
    keyword_triggers: list[str]

    # Decomposition-Logic
    subtask_templates: list[dict[str, Any]]
    default_strategy: DecompositionStrategy

    # Priorität
    priority: int = 100  # Niedrigere Zahl = höhere Priorität

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime | None = None
    usage_count: int = 0
