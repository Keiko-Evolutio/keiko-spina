"""Gemeinsame Konfigurationsstrukturen für das Monitoring-System.

Konsolidiert alle @dataclass Konfigurationen aus dem Monitoring-Modul
in einer einzigen, gut organisierten Datei. Eliminiert Code-Duplikation
und stellt konsistente Konfigurationsmuster sicher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ============================================================================
# KONSTANTEN FÜR MAGIC NUMBERS
# ============================================================================

# Timeout-Konstanten (Sekunden)
DEFAULT_TIMEOUT_SECONDS: float = 5.0
WARNING_THRESHOLD_MS: float = 1000.0
CRITICAL_THRESHOLD_MS: float = 3000.0
SLA_LATENCY_TARGET_MS: float = 200.0

# Verfügbarkeits-Konstanten
SLA_AVAILABILITY_TARGET: float = 0.999  # 99.9%
SLA_ERROR_RATE_TARGET_PCT: float = 1.0  # 1%

# Metriken-Konstanten
MAX_METRICS_IN_MEMORY: int = 10000
RETAIN_POINTS_AFTER_TRIM: int = 50
MAX_POINTS_PER_METRIC: int = 100
RETENTION_HOURS: int = 24
CLEANUP_INTERVAL_MINUTES: int = 60

# Performance-Konstanten
HIGH_VOLUME_THRESHOLD: int = 1000  # Events/Minute
METRICS_SAMPLING_RATE: float = 1.0  # 100% = keine Sampling
TIME_WINDOW_SECONDS: int = 300  # 5 Minuten Rolling-Window
RING_BUFFER_CAPACITY: int = 24 * 60  # 24h in Minuten

# SLO-Konstanten
P95_LATENCY_TARGET_MS: float = 500.0
ERROR_RATE_TARGET_PCT: float = 1.0

# ============================================================================
# BASIS-KONFIGURATION
# ============================================================================

@dataclass
class BaseMonitoringConfig:
    """Basis-Konfiguration für alle Monitoring-Komponenten."""
    enabled: bool = True
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def validate(self) -> None:
        """Validiert die Konfiguration."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds muss positiv sein")

# ============================================================================
# HEALTH-CHECK-KONFIGURATION
# ============================================================================

@dataclass
class HealthCheckConfig(BaseMonitoringConfig):
    """Erweiterte Health Check Konfiguration."""
    warning_threshold_ms: float = WARNING_THRESHOLD_MS
    critical_threshold_ms: float = CRITICAL_THRESHOLD_MS
    sla_latency_ms: float = SLA_LATENCY_TARGET_MS
    sla_availability_target: float = SLA_AVAILABILITY_TARGET
    sla_error_rate_pct: float = SLA_ERROR_RATE_TARGET_PCT

    def validate(self) -> None:
        """Validiert Health-Check-spezifische Konfiguration."""
        super().validate()
        if self.warning_threshold_ms >= self.critical_threshold_ms:
            raise ValueError("warning_threshold_ms muss kleiner als critical_threshold_ms sein")
        if not 0 <= self.sla_availability_target <= 1:
            raise ValueError("sla_availability_target muss zwischen 0 und 1 liegen")

# ============================================================================
# METRIKEN-KONFIGURATION
# ============================================================================

@dataclass
class MetricsConfig(BaseMonitoringConfig):
    """Erweiterte Konfiguration für Metrics mit Observability-Integration."""
    max_metrics_in_memory: int = MAX_METRICS_IN_MEMORY
    retention_hours: int = RETENTION_HOURS
    cleanup_interval_minutes: int = CLEANUP_INTERVAL_MINUTES

    # Observability-Features
    enable_agent_metrics: bool = True
    enable_task_rps_tracking: bool = True
    enable_latency_percentiles: bool = True
    enable_error_categorization: bool = True
    enable_queue_depth_monitoring: bool = True
    enable_tool_call_tracking: bool = True

    # Performance-Konfiguration
    metrics_sampling_rate: float = METRICS_SAMPLING_RATE
    high_volume_threshold: int = HIGH_VOLUME_THRESHOLD
    async_processing: bool = True

    # Integration-Flags
    integrate_with_task_management: bool = True
    integrate_with_security: bool = True
    integrate_with_policy_engine: bool = True
    integrate_with_audit_system: bool = True

    def validate(self) -> None:
        """Validiert Metriken-spezifische Konfiguration."""
        super().validate()
        if self.max_metrics_in_memory <= 0:
            raise ValueError("max_metrics_in_memory muss positiv sein")
        if not 0 <= self.metrics_sampling_rate <= 1:
            raise ValueError("metrics_sampling_rate muss zwischen 0 und 1 liegen")

# ============================================================================
# SLO-KONFIGURATION
# ============================================================================

@dataclass
class SLOTargets(BaseMonitoringConfig):
    """Zielwerte für SLOs."""
    p95_latency_ms: float = P95_LATENCY_TARGET_MS
    error_rate_pct: float = ERROR_RATE_TARGET_PCT
    window_seconds: int = TIME_WINDOW_SECONDS

    def validate(self) -> None:
        """Validiert SLO-spezifische Konfiguration."""
        super().validate()
        if self.p95_latency_ms <= 0:
            raise ValueError("p95_latency_ms muss positiv sein")
        if not 0 <= self.error_rate_pct <= 100:
            raise ValueError("error_rate_pct muss zwischen 0 und 100 liegen")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds muss positiv sein")

# ============================================================================
# AZURE-MONITORING-KONFIGURATION
# ============================================================================

@dataclass
class AzureMonitoringConfig(BaseMonitoringConfig):
    """Konfiguration für Azure Monitor Integration."""
    connection_string: str | None = None
    application_name: str = "keiko-backend"
    environment: str = "development"
    enable_custom_metrics: bool = True
    enable_dependency_tracking: bool = True
    enable_request_tracking: bool = True
    enable_exception_tracking: bool = True

    def validate(self) -> None:
        """Validiert Azure-spezifische Konfiguration."""
        super().validate()
        if self.enabled and not self.connection_string:
            raise ValueError("connection_string ist erforderlich wenn Azure Monitoring aktiviert ist")

# ============================================================================
# LANGSMITH-KONFIGURATION
# ============================================================================

@dataclass
class LangSmithConfig(BaseMonitoringConfig):
    """Konfiguration für LangSmith Integration."""
    api_key: str | None = None
    project_name: str = "keiko-monitoring"
    environment: str = "development"
    enable_tracing: bool = True
    enable_metrics_export: bool = True
    sampling_rate: float = 1.0

    def validate(self) -> None:
        """Validiert LangSmith-spezifische Konfiguration."""
        super().validate()
        if self.enabled and not self.api_key:
            raise ValueError("api_key ist erforderlich wenn LangSmith aktiviert ist")
        if not 0 <= self.sampling_rate <= 1:
            raise ValueError("sampling_rate muss zwischen 0 und 1 liegen")

# ============================================================================
# PERFORMANCE-TRACKER-KONFIGURATION
# ============================================================================

@dataclass
class PerformanceConfig(BaseMonitoringConfig):
    """Konfiguration für Performance-Tracking."""
    enable_detailed_timing: bool = True
    enable_memory_tracking: bool = True
    enable_cpu_tracking: bool = False
    max_tracked_operations: int = 1000
    operation_timeout_seconds: float = 30.0

    def validate(self) -> None:
        """Validiert Performance-spezifische Konfiguration."""
        super().validate()
        if self.max_tracked_operations <= 0:
            raise ValueError("max_tracked_operations muss positiv sein")
        if self.operation_timeout_seconds <= 0:
            raise ValueError("operation_timeout_seconds muss positiv sein")

# ============================================================================
# MONITORING-MANAGER-KONFIGURATION
# ============================================================================

@dataclass
class MonitoringManagerConfig(BaseMonitoringConfig):
    """Zentrale Konfiguration für den MonitoringManager."""
    service_name: str = "keiko-backend"
    enable_metrics: bool = True
    enable_health: bool = True
    enable_tracing: bool = True
    enable_azure_integration: bool = False
    enable_langsmith_integration: bool = False

    # Sub-Konfigurationen
    health_config: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
    slo_config: SLOTargets = field(default_factory=SLOTargets)
    azure_config: AzureMonitoringConfig = field(default_factory=AzureMonitoringConfig)
    langsmith_config: LangSmithConfig = field(default_factory=LangSmithConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)

    def validate(self) -> None:
        """Validiert die gesamte MonitoringManager-Konfiguration."""
        super().validate()

        # Validiere Sub-Konfigurationen
        self.health_config.validate()
        self.metrics_config.validate()
        self.slo_config.validate()

        if self.enable_azure_integration:
            self.azure_config.enabled = True
            self.azure_config.validate()

        if self.enable_langsmith_integration:
            self.langsmith_config.enabled = True
            self.langsmith_config.validate()

        self.performance_config.validate()

# ============================================================================
# FACTORY-FUNKTIONEN
# ============================================================================

def create_development_config() -> MonitoringManagerConfig:
    """Erstellt Konfiguration für Development-Umgebung."""
    config = MonitoringManagerConfig(
        service_name="keiko-backend-dev",
        enable_azure_integration=False,
        enable_langsmith_integration=False,
    )

    # Development-spezifische Anpassungen
    config.health_config.sla_availability_target = 0.95  # Weniger streng
    config.metrics_config.max_metrics_in_memory = 5000  # Weniger Memory
    config.slo_config.p95_latency_ms = 1000.0  # Weniger streng

    return config

def create_production_config() -> MonitoringManagerConfig:
    """Erstellt Konfiguration für Production-Umgebung."""
    config = MonitoringManagerConfig(
        service_name="keiko-backend-prod",
        enable_azure_integration=True,
        enable_langsmith_integration=True,
    )

    # Production-spezifische Anpassungen
    config.health_config.sla_availability_target = 0.999  # Sehr streng
    config.metrics_config.max_metrics_in_memory = 20000  # Mehr Memory
    config.slo_config.p95_latency_ms = 200.0  # Sehr streng
    config.azure_config.environment = "production"
    config.langsmith_config.environment = "production"

    return config

def create_testing_config() -> MonitoringManagerConfig:
    """Erstellt Konfiguration für Testing-Umgebung."""
    config = MonitoringManagerConfig(
        service_name="keiko-backend-test",
        enable_azure_integration=False,
        enable_langsmith_integration=False,
    )

    # Testing-spezifische Anpassungen
    config.health_config.timeout_seconds = 1.0  # Schnelle Tests
    config.metrics_config.async_processing = False  # Synchron für Tests
    config.metrics_config.cleanup_interval_minutes = 1  # Häufiger Cleanup

    return config

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CRITICAL_THRESHOLD_MS",
    # Konstanten
    "DEFAULT_TIMEOUT_SECONDS",
    "HIGH_VOLUME_THRESHOLD",
    "MAX_METRICS_IN_MEMORY",
    "P95_LATENCY_TARGET_MS",
    "SLA_AVAILABILITY_TARGET",
    "SLA_LATENCY_TARGET_MS",
    "WARNING_THRESHOLD_MS",
    "AzureMonitoringConfig",
    # Basis-Konfigurationen
    "BaseMonitoringConfig",
    "HealthCheckConfig",
    "LangSmithConfig",
    "MetricsConfig",
    "MonitoringManagerConfig",
    "PerformanceConfig",
    "SLOTargets",
    # Factory-Funktionen
    "create_development_config",
    "create_production_config",
    "create_testing_config",
]
