# backend/services/enhanced_quotas_limits_management/data_models.py
"""Datenmodelle für Enhanced Quotas & Limits Management.

Definiert alle Datenstrukturen für Enterprise-Grade Resource-Management,
Multi-Tenant Quota-Enforcement und Real-time Monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from services.enhanced_security_integration import SecurityLevel


class ResourceType(Enum):
    """Resource-Typen für Quota-Management."""

    AGENT = "agent"                      # Agent-Instanzen
    TASK = "task"                        # Task-Executions
    API_CALL = "api_call"               # API-Aufrufe
    LLM_REQUEST = "llm_request"         # LLM-Requests
    STORAGE = "storage"                  # Storage-Nutzung
    COMPUTE = "compute"                  # Compute-Ressourcen
    NETWORK = "network"                  # Network-Bandwidth
    MEMORY = "memory"                    # Memory-Nutzung


class QuotaScope(Enum):
    """Quota-Scope für Hierarchical Limits."""

    GLOBAL = "global"                    # Global für alle Tenants
    TENANT = "tenant"                    # Tenant-spezifisch
    USER = "user"                        # User-spezifisch
    AGENT = "agent"                      # Agent-spezifisch
    SERVICE = "service"                  # Service-spezifisch
    CAPABILITY = "capability"            # Capability-spezifisch


class QuotaPeriod(Enum):
    """Quota-Zeiträume."""

    SECOND = "second"                    # Pro Sekunde
    MINUTE = "minute"                    # Pro Minute
    HOUR = "hour"                        # Pro Stunde
    DAY = "day"                          # Pro Tag
    WEEK = "week"                        # Pro Woche
    MONTH = "month"                      # Pro Monat
    YEAR = "year"                        # Pro Jahr


class QuotaStatus(Enum):
    """Quota-Status."""

    ACTIVE = "active"                    # Aktiv
    SUSPENDED = "suspended"              # Suspendiert
    EXCEEDED = "exceeded"                # Überschritten
    WARNING = "warning"                  # Warnung
    CRITICAL = "critical"                # Kritisch


class EnforcementAction(Enum):
    """Enforcement-Aktionen bei Quota-Verletzungen."""

    ALLOW = "allow"                      # Erlauben
    DENY = "deny"                        # Verweigern
    THROTTLE = "throttle"                # Drosseln
    QUEUE = "queue"                      # In Warteschlange
    ALERT = "alert"                      # Nur Alert
    SCALE = "scale"                      # Auto-Scaling


class HealthStatus(Enum):
    """Health-Status für API Contracts."""

    HEALTHY = "healthy"                  # Gesund
    DEGRADED = "degraded"                # Beeinträchtigt
    UNHEALTHY = "unhealthy"              # Ungesund
    CRITICAL = "critical"                # Kritisch
    UNKNOWN = "unknown"                  # Unbekannt


@dataclass
class ResourceQuota:
    """Resource-Quota-Definition."""

    # Quota-Identifikation
    quota_id: str
    name: str
    description: str

    # Resource-Definition
    resource_type: ResourceType
    scope: QuotaScope
    scope_id: str  # Tenant-ID, User-ID, Agent-ID, etc.

    # Quota-Limits
    limit: int
    period: QuotaPeriod
    burst_limit: int | None = None

    # Hierarchical Limits
    parent_quota_id: str | None = None
    child_quota_ids: set[str] = field(default_factory=set)

    # Security-Integration
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    required_clearances: list[str] = field(default_factory=list)

    # Enforcement
    enforcement_action: EnforcementAction = EnforcementAction.DENY
    grace_period_seconds: int = 0

    # Status
    status: QuotaStatus = QuotaStatus.ACTIVE

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class QuotaUsage:
    """Quota-Usage-Tracking."""

    # Usage-Identifikation
    usage_id: str
    quota_id: str

    # Usage-Details
    resource_type: ResourceType
    scope: QuotaScope
    scope_id: str

    # Usage-Metriken
    current_usage: int
    peak_usage: int
    average_usage: float

    # Zeitfenster
    period_start: datetime
    period_end: datetime
    measurement_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Zusätzliche Metriken
    request_count: int = 0
    error_count: int = 0
    throttled_count: int = 0
    denied_count: int = 0

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaViolation:
    """Quota-Verletzung."""

    # Violation-Identifikation
    violation_id: str
    quota_id: str

    # Violation-Details
    resource_type: ResourceType
    scope: QuotaScope
    scope_id: str

    # Violation-Metriken
    limit_value: int
    actual_value: int
    excess_amount: int

    # Enforcement
    enforcement_action: EnforcementAction
    action_taken: bool = False

    # Zeitstempel
    violation_timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved_timestamp: datetime | None = None

    # Kontext
    request_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None

    # Metadaten
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimit:
    """Rate-Limit-Definition."""

    # Rate-Limit-Identifikation
    rate_limit_id: str
    name: str
    description: str

    # Rate-Limit-Konfiguration
    resource_type: ResourceType
    scope: QuotaScope
    scope_id: str

    # Rate-Limit-Parameter
    requests_per_period: int
    period: QuotaPeriod
    burst_capacity: int = 0

    # Multi-Tenant-Isolation
    tenant_id: str | None = None
    isolation_level: str = "strict"  # strict, relaxed, shared

    # Algorithmus
    algorithm: str = "sliding_window"  # sliding_window, token_bucket, fixed_window

    # Status
    enabled: bool = True

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QuotaCheckResult:
    """Ergebnis einer Quota-Prüfung."""

    # Check-Status
    allowed: bool
    quota_id: str

    # Usage-Informationen
    current_usage: int
    limit: int
    remaining: int

    # Rate-Limiting
    rate_limited: bool = False
    retry_after_seconds: int | None = None

    # Enforcement
    enforcement_action: EnforcementAction = EnforcementAction.ALLOW
    grace_period_remaining: int = 0

    # Performance-Metriken
    check_duration_ms: float = 0.0

    # Violations
    violations: list[QuotaViolation] = field(default_factory=list)

    # Metadaten
    check_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIContract:
    """API-Contract-Definition für Service-Health-Checks."""

    # Contract-Identifikation
    contract_id: str
    service_name: str
    version: str

    # Endpoint-Definition
    endpoint_path: str
    http_method: str

    # SLA-Definition
    max_response_time_ms: int
    min_success_rate: float  # 0.0 - 1.0
    max_error_rate: float    # 0.0 - 1.0

    # Quota-Integration
    quota_requirements: list[str] = field(default_factory=list)
    rate_limit_requirements: list[str] = field(default_factory=list)

    # Health-Check-Konfiguration
    health_check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 30

    # Status
    enabled: bool = True

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HealthCheckResult:
    """Ergebnis eines Health-Checks."""

    # Check-Identifikation
    check_id: str
    contract_id: str
    service_name: str

    # Health-Status
    status: HealthStatus

    # Performance-Metriken
    response_time_ms: float
    success_rate: float
    error_rate: float

    # Quota-Status
    quota_compliance: bool = True
    rate_limit_compliance: bool = True

    # Details
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Zeitstempel
    check_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaAnalytics:
    """Quota-Analytics-Daten."""

    # Analytics-Identifikation
    analytics_id: str
    quota_id: str

    # Zeitraum
    period_start: datetime
    period_end: datetime

    # Usage-Statistiken
    total_requests: int
    successful_requests: int
    failed_requests: int
    throttled_requests: int
    denied_requests: int

    # Performance-Statistiken
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    # Quota-Statistiken
    peak_usage: int
    average_usage: float
    quota_utilization: float  # 0.0 - 1.0

    # Trends
    usage_trend: str = "stable"  # increasing, decreasing, stable, volatile
    predicted_exhaustion: datetime | None = None

    # Violations
    violation_count: int = 0
    violation_rate: float = 0.0

    # Metadaten
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaPerformanceMetrics:
    """Performance-Metriken für Quota-Management."""

    # Quota-Check-Performance
    total_quota_checks: int = 0
    avg_quota_check_time_ms: float = 0.0
    p95_quota_check_time_ms: float = 0.0
    p99_quota_check_time_ms: float = 0.0

    # Rate-Limiting-Performance
    total_rate_limit_checks: int = 0
    avg_rate_limit_check_time_ms: float = 0.0

    # Enforcement-Performance
    total_enforcements: int = 0
    avg_enforcement_time_ms: float = 0.0

    # Health-Check-Performance
    total_health_checks: int = 0
    avg_health_check_time_ms: float = 0.0

    # SLA-Compliance
    meets_quota_sla: bool = True
    quota_sla_threshold_ms: float = 50.0

    # Cache-Performance
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0

    # Error-Rates
    quota_check_error_rate: float = 0.0
    enforcement_error_rate: float = 0.0

    # Metadaten
    measurement_period_start: datetime = field(default_factory=datetime.utcnow)
    measurement_period_end: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0
