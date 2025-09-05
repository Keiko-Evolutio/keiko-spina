# backend/services/enhanced_real_time_monitoring/data_models.py
"""Datenmodelle für Enhanced Real-time Monitoring.

Definiert alle Datenstrukturen für Enterprise-Grade Real-time Monitoring,
Saga Coordination und Distributed Tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..enhanced_security_integration import SecurityLevel


class MonitoringScope(Enum):
    """Monitoring-Scope für verschiedene Überwachungsebenen."""

    SYSTEM = "system"                    # System-weites Monitoring
    SERVICE = "service"                  # Service-spezifisches Monitoring
    AGENT = "agent"                      # Agent-spezifisches Monitoring
    TASK = "task"                        # Task-spezifisches Monitoring
    ORCHESTRATION = "orchestration"      # Orchestration-Monitoring
    DEPENDENCY = "dependency"            # Dependency-Monitoring
    RESOURCE = "resource"                # Resource-Monitoring
    SECURITY = "security"                # Security-Monitoring


class MetricType(Enum):
    """Metric-Typen für verschiedene Messungen."""

    COUNTER = "counter"                  # Zähler-Metrik
    GAUGE = "gauge"                      # Momentaufnahme-Metrik
    HISTOGRAM = "histogram"              # Verteilungs-Metrik
    TIMER = "timer"                      # Zeit-Metrik
    RATE = "rate"                        # Rate-Metrik
    PERCENTAGE = "percentage"            # Prozent-Metrik


class AlertSeverity(Enum):
    """Alert-Schweregrade."""

    INFO = "info"                        # Informational
    WARNING = "warning"                  # Warnung
    CRITICAL = "critical"                # Kritisch
    EMERGENCY = "emergency"              # Notfall


class SagaStatus(Enum):
    """Saga-Status für Compensation-Logic."""

    PENDING = "pending"                  # Wartend
    RUNNING = "running"                  # Läuft
    COMPENSATING = "compensating"        # Kompensiert
    COMPLETED = "completed"              # Abgeschlossen
    FAILED = "failed"                    # Fehlgeschlagen
    CANCELLED = "cancelled"              # Abgebrochen


class CompensationAction(Enum):
    """Compensation-Aktionen für Saga-Pattern."""

    ROLLBACK = "rollback"                # Rollback
    RETRY = "retry"                      # Wiederholen
    SKIP = "skip"                        # Überspringen
    ESCALATE = "escalate"                # Eskalieren
    MANUAL = "manual"                    # Manuell


class TraceStatus(Enum):
    """Trace-Status für Distributed Tracing."""

    ACTIVE = "active"                    # Aktiv
    COMPLETED = "completed"              # Abgeschlossen
    ERROR = "error"                      # Fehler
    TIMEOUT = "timeout"                  # Timeout


@dataclass
class MonitoringMetric:
    """Real-time Monitoring-Metrik."""

    # Metric-Identifikation
    metric_id: str
    metric_name: str
    metric_type: MetricType
    scope: MonitoringScope

    # Metric-Werte
    value: int | float | str
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Kontext
    service_name: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    orchestration_id: str | None = None

    # Labels und Tags
    labels: dict[str, str] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance-Metriken für SLA-Tracking."""

    # Performance-Identifikation
    metrics_id: str
    scope: MonitoringScope
    scope_id: str

    # Response-Time-Metriken
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0

    # Throughput-Metriken
    requests_per_second: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Error-Rate-Metriken
    error_rate: float = 0.0
    success_rate: float = 0.0

    # Resource-Metriken
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mbps: float = 0.0

    # SLA-Compliance
    sla_compliance_percent: float = 0.0
    sla_violations: int = 0

    # Zeitstempel
    measurement_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    measurement_end: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert-Regel für Monitoring."""

    # Rule-Identifikation
    rule_id: str
    rule_name: str
    description: str

    # Rule-Konfiguration
    metric_name: str
    scope: MonitoringScope
    scope_pattern: str  # Regex-Pattern

    # Schwellwerte
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    emergency_threshold: float | None = None

    # Evaluation
    evaluation_window_seconds: int = 300  # 5 Minuten
    consecutive_violations: int = 1

    # Aktionen
    notification_channels: list[str] = field(default_factory=list)
    auto_remediation: bool = False
    escalation_policy: str | None = None

    # Status
    enabled: bool = True
    last_triggered: datetime | None = None
    cooldown_seconds: int = 1800  # 30 Minuten

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringAlert:
    """Monitoring-Alert."""

    # Alert-Identifikation
    alert_id: str
    rule_id: str
    alert_name: str

    # Alert-Details
    severity: AlertSeverity
    message: str
    description: str

    # Kontext
    scope: MonitoringScope
    scope_id: str
    metric_name: str
    metric_value: int | float | str
    threshold_value: int | float | str

    # Status
    status: str = "active"  # active, acknowledged, resolved
    acknowledged_by: str | None = None
    resolved_by: str | None = None

    # Zeitstempel
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaStep:
    """Saga-Step für Compensation-Logic."""

    # Step-Identifikation
    step_id: str
    saga_id: str
    step_name: str
    step_order: int

    # Step-Konfiguration
    service_name: str
    operation: str
    action: str = ""  # Action name for the step
    parameters: dict[str, Any] = field(default_factory=dict)

    # Compensation
    compensation_operation: str | None = None
    compensation_parameters: dict[str, Any] = field(default_factory=dict)
    compensation_action: CompensationAction = CompensationAction.ROLLBACK

    # Status
    status: SagaStatus = SagaStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

    # Timing
    timeout_seconds: int = 300
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    result: dict[str, Any] | None = None
    error: str | None = None
    error_message: str | None = None

    # Context and Metadata
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaTransaction:
    """Saga-Transaction für Enterprise-Reliability."""

    # Saga-Identifikation
    saga_id: str
    saga_name: str
    description: str

    # Saga-Konfiguration
    steps: list[SagaStep] = field(default_factory=list)
    compensation_strategy: str = "reverse_order"  # reverse_order, parallel, custom

    # Status
    status: SagaStatus = SagaStatus.PENDING
    current_step: int = 0
    current_step_index: int = 0
    completed_steps: set[str] = field(default_factory=set)
    compensated_steps: set[str] = field(default_factory=set)

    # Error Handling
    error_message: str | None = None
    compensation_executed: bool = False

    # Timing
    timeout_seconds: int = 1800  # 30 Minuten
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Context
    orchestration_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL

    # Results
    final_result: dict[str, Any] | None = None
    compensation_result: dict[str, Any] | None = None
    error: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed Tracing Span."""

    # Span-Identifikation
    span_id: str
    trace_id: str
    operation_name: str
    service_name: str
    component: str
    parent_span_id: str | None = None

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float | None = None

    # Status
    status: TraceStatus = TraceStatus.ACTIVE
    status_code: int | None = None
    error: str | None = None

    # Tags und Logs
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    # Kontext
    agent_id: str | None = None
    task_id: str | None = None
    orchestration_id: str | None = None
    user_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTrace:
    """Distributed Trace für Observability."""

    # Trace-Identifikation
    trace_id: str
    trace_name: str
    root_span_id: str

    # Trace-Komponenten
    spans: dict[str, TraceSpan] = field(default_factory=dict)
    span_hierarchy: dict[str, list[str]] = field(default_factory=dict)  # parent -> children

    # Trace-Status
    status: TraceStatus = TraceStatus.ACTIVE
    total_spans: int = 0
    completed_spans: int = 0
    error_spans: int = 0

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    total_duration_ms: float | None = None

    # Kontext
    service_names: set[str] = field(default_factory=set)
    orchestration_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveDashboardData:
    """Live-Dashboard-Daten für Real-time Monitoring."""

    # Dashboard-Identifikation
    dashboard_id: str
    dashboard_name: str

    # System-Übersicht
    system_health: str = "healthy"  # healthy, degraded, unhealthy
    active_alerts: int = 0
    total_services: int = 0
    healthy_services: int = 0

    # Performance-Übersicht
    avg_response_time_ms: float = 0.0
    total_requests_per_second: float = 0.0
    overall_error_rate: float = 0.0
    overall_success_rate: float = 0.0

    # Resource-Übersicht
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    total_disk_usage: float = 0.0

    # Service-Details
    service_metrics: dict[str, PerformanceMetrics] = field(default_factory=dict)

    # Active Sagas
    active_sagas: int = 0
    completed_sagas: int = 0
    failed_sagas: int = 0

    # Active Traces
    active_traces: int = 0
    completed_traces: int = 0
    error_traces: int = 0

    # Zeitstempel
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfiguration:
    """Monitoring-Konfiguration."""

    # Monitoring-Einstellungen
    monitoring_enabled: bool = True
    real_time_monitoring_enabled: bool = True
    saga_coordination_enabled: bool = True
    distributed_tracing_enabled: bool = True

    # Performance-Einstellungen
    metrics_collection_interval_seconds: int = 10
    performance_monitoring_overhead_ms: float = 50.0
    alert_evaluation_interval_seconds: int = 30

    # Saga-Einstellungen
    saga_timeout_seconds: int = 1800
    saga_retry_attempts: int = 3
    compensation_timeout_seconds: int = 300

    # Tracing-Einstellungen
    trace_sampling_rate: float = 1.0
    trace_retention_hours: int = 24
    span_timeout_seconds: int = 300

    # Dashboard-Einstellungen
    dashboard_refresh_interval_seconds: int = 5
    dashboard_data_retention_hours: int = 24

    # Alert-Einstellungen
    alert_cooldown_seconds: int = 1800
    alert_escalation_enabled: bool = True
    auto_remediation_enabled: bool = False

    # Security-Einstellungen
    monitoring_security_level: SecurityLevel = SecurityLevel.INTERNAL
    audit_trail_enabled: bool = True
    compliance_monitoring_enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MonitoringPerformanceMetrics:
    """Performance-Metriken für Monitoring-System selbst."""

    # Monitoring-Performance
    total_metrics_collected: int = 0
    avg_metric_collection_time_ms: float = 0.0
    metrics_collection_error_rate: float = 0.0

    # Alert-Performance
    total_alerts_processed: int = 0
    avg_alert_processing_time_ms: float = 0.0
    alert_processing_error_rate: float = 0.0

    # Saga-Performance
    total_sagas_executed: int = 0
    avg_saga_execution_time_ms: float = 0.0
    saga_success_rate: float = 0.0
    saga_compensation_rate: float = 0.0

    # Tracing-Performance
    total_traces_created: int = 0
    avg_trace_duration_ms: float = 0.0
    trace_completion_rate: float = 0.0

    # Dashboard-Performance
    dashboard_generation_time_ms: float = 0.0
    dashboard_update_frequency: float = 0.0

    # SLA-Compliance
    meets_monitoring_sla: bool = True
    monitoring_sla_threshold_ms: float = 50.0

    # Zeitraum
    measurement_period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    measurement_period_end: datetime = field(default_factory=lambda: datetime.now(UTC))
    sample_count: int = 0
