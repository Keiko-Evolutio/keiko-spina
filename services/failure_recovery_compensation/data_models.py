# backend/services/failure_recovery_compensation/data_models.py
"""Data Models für Failure Recovery & Compensation System.

Implementiert Enterprise-Grade Data Models für Failure Recovery, Compensation Framework,
Saga Pattern Transactions und Distributed System Resilience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from services.enhanced_security_integration import SecurityLevel


class FailureType(Enum):
    """Failure-Typen für Recovery System."""

    # Service Failures
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_TIMEOUT = "service_timeout"
    SERVICE_ERROR = "service_error"
    SERVICE_OVERLOAD = "service_overload"

    # Network Failures
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_UNREACHABLE = "network_unreachable"
    CONNECTION_REFUSED = "connection_refused"
    DNS_RESOLUTION_FAILED = "dns_resolution_failed"

    # Resource Failures
    RESOURCE_EXHAUSTED = "resource_exhausted"
    MEMORY_EXHAUSTED = "memory_exhausted"
    CPU_EXHAUSTED = "cpu_exhausted"
    DISK_FULL = "disk_full"

    # Authentication/Authorization Failures
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    TOKEN_EXPIRED = "token_expired"
    PERMISSION_DENIED = "permission_denied"

    # Data Failures
    DATA_CORRUPTION = "data_corruption"
    DATA_VALIDATION_FAILED = "data_validation_failed"
    SERIALIZATION_FAILED = "serialization_failed"

    # Business Logic Failures
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INVALID_STATE = "invalid_state"

    # External System Failures
    EXTERNAL_API_FAILED = "external_api_failed"
    THIRD_PARTY_SERVICE_DOWN = "third_party_service_down"

    # Unknown/Generic Failures
    UNKNOWN_ERROR = "unknown_error"
    INTERNAL_ERROR = "internal_error"


class RecoveryStrategy(Enum):
    """Recovery-Strategien für Failure Recovery."""

    # Retry Strategies
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"

    # Fallback Strategies
    FALLBACK_SERVICE = "fallback_service"
    CACHED_RESPONSE = "cached_response"
    DEFAULT_RESPONSE = "default_response"
    DEGRADED_SERVICE = "degraded_service"

    # Circuit Breaker Strategies
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    RATE_LIMITING = "rate_limiting"

    # Compensation Strategies
    SAGA_COMPENSATION = "saga_compensation"
    ROLLBACK_TRANSACTION = "rollback_transaction"
    MANUAL_INTERVENTION = "manual_intervention"

    # Advanced Strategies
    ADAPTIVE_RECOVERY = "adaptive_recovery"
    ML_BASED_RECOVERY = "ml_based_recovery"
    CHAOS_ENGINEERING = "chaos_engineering"


class CompensationAction(Enum):
    """Compensation-Aktionen für Saga Pattern."""

    # Data Compensation
    UNDO_DATA_CHANGE = "undo_data_change"
    RESTORE_BACKUP = "restore_backup"
    DELETE_CREATED_RECORD = "delete_created_record"
    UPDATE_STATUS = "update_status"

    # Service Compensation
    CANCEL_OPERATION = "cancel_operation"
    REVERSE_OPERATION = "reverse_operation"
    NOTIFY_CANCELLATION = "notify_cancellation"

    # Financial Compensation
    REFUND_PAYMENT = "refund_payment"
    REVERSE_CHARGE = "reverse_charge"
    CREDIT_ACCOUNT = "credit_account"

    # Resource Compensation
    RELEASE_RESOURCES = "release_resources"
    DEALLOCATE_MEMORY = "deallocate_memory"
    CLOSE_CONNECTIONS = "close_connections"

    # Notification Compensation
    SEND_FAILURE_NOTIFICATION = "send_failure_notification"
    UPDATE_USER_STATUS = "update_user_status"
    LOG_COMPENSATION = "log_compensation"

    # Custom Compensation
    CUSTOM_COMPENSATION = "custom_compensation"
    NO_COMPENSATION = "no_compensation"


class SagaState(Enum):
    """Saga-Transaction-States."""

    CREATED = "created"
    EXECUTING = "executing"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RecoveryState(Enum):
    """Recovery-Process-States."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RETRYING = "retrying"
    RECOVERED = "recovered"
    FAILED = "failed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"


@dataclass
class FailureContext:
    """Kontext-Informationen für Failure."""

    # Failure-Identifikation
    failure_id: str
    failure_type: FailureType
    service_name: str
    operation_name: str

    # Failure-Details
    error_message: str
    error_code: str | None
    stack_trace: str | None

    # Timing
    occurred_at: datetime

    # Request-Context
    request_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None

    # Service-Context
    service_version: str | None = None
    deployment_id: str | None = None
    instance_id: str | None = None

    # Network-Context
    source_ip: str | None = None
    target_endpoint: str | None = None
    http_status_code: int | None = None

    # Performance-Context
    response_time_ms: float | None = None
    cpu_usage_percent: float | None = None
    memory_usage_mb: float | None = None

    # Additional Context
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class RecoveryConfiguration:
    """Konfiguration für Recovery-Strategien."""

    # Recovery-Identifikation
    config_id: str
    service_name: str
    operation_name: str

    # Recovery-Strategien
    primary_strategy: RecoveryStrategy
    fallback_strategies: list[RecoveryStrategy] = field(default_factory=list)

    # Retry-Konfiguration
    max_retry_attempts: int = 3
    initial_retry_delay_ms: int = 1000
    max_retry_delay_ms: int = 30000
    retry_multiplier: float = 2.0
    retry_jitter: bool = True

    # Timeout-Konfiguration
    operation_timeout_ms: int = 30000
    recovery_timeout_ms: int = 300000

    # Circuit Breaker-Konfiguration
    failure_threshold: int = 5
    success_threshold: int = 3
    circuit_timeout_ms: int = 60000

    # Fallback-Konfiguration
    fallback_service_url: str | None = None
    fallback_cache_ttl_seconds: int = 300
    default_response: dict[str, Any] | None = None

    # Compensation-Konfiguration
    compensation_enabled: bool = True
    compensation_timeout_ms: int = 60000

    # Monitoring-Konfiguration
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    metrics_collection_enabled: bool = True

    # Security-Konfiguration
    security_level: SecurityLevel = SecurityLevel.INTERNAL

    # Additional Configuration
    custom_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaStep:
    """Einzelner Schritt in Saga-Transaction."""

    # Step-Identifikation
    step_id: str
    step_name: str
    step_order: int

    # Service-Informationen
    service_name: str
    operation_name: str
    endpoint_url: str

    # Request-Daten
    request_data: dict[str, Any]

    # Compensation-Informationen
    compensation_action: CompensationAction

    # Optional fields with defaults
    headers: dict[str, str] = field(default_factory=dict)
    compensation_service: str | None = None
    compensation_operation: str | None = None
    compensation_data: dict[str, Any] = field(default_factory=dict)

    # Timeout-Konfiguration
    timeout_ms: int = 30000

    # Retry-Konfiguration
    retry_enabled: bool = True
    max_retries: int = 3

    # Dependencies
    depends_on: list[str] = field(default_factory=list)

    # Execution-State
    state: str | None = None
    executed_at: datetime | None = None
    completed_at: datetime | None = None
    response_data: dict[str, Any] | None = None
    error_message: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaTransaction:
    """Saga-Transaction für Distributed Transactions."""

    # Saga-Identifikation
    saga_id: str
    saga_name: str
    description: str

    # Saga-Steps
    steps: list[SagaStep]

    # Saga-State
    state: SagaState = SagaState.CREATED

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Execution-Context
    orchestration_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    security_level: SecurityLevel | None = None

    # Compensation-Konfiguration
    compensation_strategy: str = "automatic"
    compensation_timeout_ms: int = 300000

    # Timeout-Konfiguration
    total_timeout_ms: int = 1800000  # 30 Minuten

    # Execution-Results
    executed_steps: list[str] = field(default_factory=list)
    compensated_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)

    # Error-Informationen
    error_message: str | None = None
    error_step_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class RecoveryAttempt:
    """Recovery-Versuch für Failed Operation."""

    # Recovery-Identifikation
    attempt_id: str
    failure_id: str

    # Recovery-Strategie
    strategy: RecoveryStrategy
    strategy_config: dict[str, Any] = field(default_factory=dict)

    # Recovery-State
    state: RecoveryState = RecoveryState.PENDING

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Retry-Informationen
    attempt_number: int = 1
    max_attempts: int = 3
    next_retry_at: datetime | None = None

    # Results
    success: bool = False
    error_message: str | None = None
    recovery_data: dict[str, Any] | None = None

    # Performance-Metriken
    recovery_time_ms: float | None = None
    resource_usage: dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedSystemHealth:
    """Health-Status für Distributed System."""

    # System-Identifikation
    system_id: str
    system_name: str

    # Health-Status
    overall_health: str = "healthy"  # healthy, degraded, unhealthy, critical
    health_score: float = 1.0  # 0.0 - 1.0

    # Service-Health
    service_health: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Infrastructure-Health
    infrastructure_health: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Network-Health
    network_health: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Performance-Metriken
    response_time_p95_ms: float = 0.0
    error_rate_percent: float = 0.0
    throughput_rps: float = 0.0
    availability_percent: float = 100.0

    # Resource-Utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_usage_percent: float = 0.0

    # Active Issues
    active_failures: list[str] = field(default_factory=list)
    active_recoveries: list[str] = field(default_factory=list)
    active_compensations: list[str] = field(default_factory=list)

    # Timing
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureRecoveryMetrics:
    """Metriken für Failure Recovery System."""

    # System-Identifikation
    system_id: str

    # Zeitraum (required fields)
    period_start: datetime
    period_end: datetime

    # Failure-Metriken (optional fields with defaults)
    total_failures: int = 0
    failures_by_type: dict[str, int] = field(default_factory=dict)
    failures_by_service: dict[str, int] = field(default_factory=dict)

    # Recovery-Metriken
    total_recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_success_rate: float = 0.0
    avg_recovery_time_ms: float = 0.0

    # Saga-Metriken
    total_sagas: int = 0
    successful_sagas: int = 0
    compensated_sagas: int = 0
    failed_sagas: int = 0
    saga_success_rate: float = 0.0
    avg_saga_duration_ms: float = 0.0

    # Performance-Metriken
    system_availability: float = 100.0
    mean_time_to_recovery_ms: float = 0.0
    mean_time_between_failures_ms: float = 0.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
