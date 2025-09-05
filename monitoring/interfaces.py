"""Monitoring System Interfaces für Keiko Personal Assistant.
Definiert abstrakte Interfaces für alle Monitoring-Komponenten.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MetricType(Enum):
    """Typen von Metriken."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert-Schweregrade."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Metrik-Wert mit Metadaten."""
    name: str
    value: float
    metric_type: MetricType
    labels: dict[str, str]
    timestamp: datetime
    help_text: str | None = None


@dataclass
class HealthStatus:
    """Health-Status eines Services."""
    service_name: str
    is_healthy: bool
    status: str
    details: dict[str, Any]
    last_check: datetime
    response_time_ms: float | None = None


@dataclass
class Alert:
    """Alert-Definition."""
    name: str
    severity: AlertSeverity
    message: str
    labels: dict[str, str]
    timestamp: datetime
    resolved: bool = False


@dataclass
class VoiceWorkflowMetrics:
    """Voice-Workflow spezifische Metriken."""
    workflow_id: str
    user_id: str
    session_id: str

    # Timing-Metriken
    total_duration_ms: float
    speech_to_text_duration_ms: float
    orchestrator_duration_ms: float
    agent_selection_duration_ms: float
    agent_execution_duration_ms: float

    # Status-Metriken
    success: bool
    error_type: str | None = None
    error_message: str | None = None

    # Quality-Metriken
    speech_confidence: float | None = None
    agent_confidence: float | None = None

    # Business-Metriken
    agents_used: list[str] = None
    tools_called: list[str] = None

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.agents_used is None:
            self.agents_used = []
        if self.tools_called is None:
            self.tools_called = []


# =============================================================================
# MONITORING INTERFACES
# =============================================================================

@runtime_checkable
class IMetricsCollector(Protocol):
    """Interface für Metrik-Sammlung."""

    def increment_counter(self, name: str, value: float = 1.0, labels: dict[str, str] = None) -> None:
        """Erhöht einen Counter."""
        ...

    def set_gauge(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Setzt einen Gauge-Wert."""
        ...

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Fügt Wert zu Histogram hinzu."""
        ...

    def record_summary(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Zeichnet Summary-Wert auf."""
        ...

    def get_metrics(self) -> list[MetricValue]:
        """Gibt alle gesammelten Metriken zurück."""
        ...


@runtime_checkable
class IHealthChecker(Protocol):
    """Interface für Health-Checks."""

    async def check_health(self, service_name: str) -> HealthStatus:
        """Führt Health-Check für Service durch."""
        ...

    async def check_all_services(self) -> list[HealthStatus]:
        """Führt Health-Check für alle Services durch."""
        ...

    def register_health_check(self, service_name: str, check_func: Callable[[], bool]) -> None:
        """Registriert Health-Check-Funktion."""
        ...

    def get_overall_health(self) -> dict[str, Any]:
        """Gibt Gesamtstatus aller Services zurück."""
        ...


@runtime_checkable
class IAlertManager(Protocol):
    """Interface für Alert-Management."""

    async def send_alert(self, alert: Alert) -> None:
        """Sendet einen Alert."""
        ...

    async def resolve_alert(self, alert_name: str) -> None:
        """Markiert Alert als gelöst."""
        ...

    def get_active_alerts(self) -> list[Alert]:
        """Gibt aktive Alerts zurück."""
        ...


@runtime_checkable
class IVoiceWorkflowMonitor(Protocol):
    """Interface für Voice-Workflow-Monitoring."""

    async def start_workflow_tracking(self, workflow_id: str, user_id: str, session_id: str) -> None:
        """Startet Workflow-Tracking."""
        ...

    async def track_speech_to_text(self, workflow_id: str, duration_ms: float, confidence: float = None) -> None:
        """Trackt Speech-to-Text Phase."""
        ...

    async def track_orchestrator(self, workflow_id: str, duration_ms: float) -> None:
        """Trackt Orchestrator Phase."""
        ...

    async def track_agent_selection(self, workflow_id: str, duration_ms: float, agents: list[str]) -> None:
        """Trackt Agent-Selection Phase."""
        ...

    async def track_agent_execution(self, workflow_id: str, duration_ms: float, success: bool, tools: list[str] = None) -> None:
        """Trackt Agent-Execution Phase."""
        ...

    async def complete_workflow(self, workflow_id: str, success: bool, error: str = None) -> VoiceWorkflowMetrics:
        """Beendet Workflow-Tracking und gibt Metriken zurück."""
        ...

    def get_workflow_statistics(self) -> dict[str, Any]:
        """Gibt Voice-Workflow-Statistiken zurück."""
        ...


@runtime_checkable
class IPerformanceMonitor(Protocol):
    """Interface für Performance-Monitoring."""

    def track_response_time(self, endpoint: str, duration_ms: float) -> None:
        """Trackt Response-Zeit für Endpoint."""
        ...

    def track_throughput(self, endpoint: str, requests_count: int = 1) -> None:
        """Trackt Throughput für Endpoint."""
        ...

    def track_resource_usage(self, cpu_percent: float, memory_mb: float, network_kb: float = None) -> None:
        """Trackt Resource-Nutzung."""
        ...

    def track_queue_length(self, queue_name: str, length: int) -> None:
        """Trackt Queue-Länge."""
        ...

    def get_response_time_stats(self, endpoint: str = None) -> list[Any]:
        """Gibt Response-Zeit-Statistiken zurück."""
        ...

    def get_throughput_stats(self, endpoint: str = None, window_seconds: int = 60) -> list[Any]:
        """Gibt Throughput-Statistiken zurück."""
        ...

    def get_resource_stats(self, window_seconds: int = 300) -> dict[str, Any]:
        """Gibt Resource-Statistiken zurück."""
        ...


@runtime_checkable
class ICircuitBreaker(Protocol):
    """Interface für Circuit Breaker Pattern."""

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Führt Funktion mit Circuit Breaker aus."""
        ...

    def get_state(self) -> str:
        """Gibt aktuellen Circuit Breaker State zurück."""
        ...

    def get_failure_rate(self) -> float:
        """Gibt aktuelle Failure-Rate zurück."""
        ...


@runtime_checkable
class IMonitoringService(Protocol):
    """Haupt-Interface für Monitoring-Service."""

    @property
    def metrics_collector(self) -> IMetricsCollector:
        """Metrics Collector."""
        ...

    @property
    def health_checker(self) -> IHealthChecker:
        """Health Checker."""
        ...

    @property
    def alert_manager(self) -> IAlertManager:
        """Alert Manager."""
        ...

    @property
    def voice_monitor(self) -> IVoiceWorkflowMonitor:
        """Voice Workflow Monitor."""
        ...

    @property
    def performance_monitor(self) -> IPerformanceMonitor:
        """Performance Monitor."""
        ...

    @property
    def circuit_breaker_manager(self) -> Any:
        """Circuit Breaker Manager."""
        ...

    async def initialize(self) -> None:
        """Initialisiert Monitoring-Service."""
        ...

    async def shutdown(self) -> None:
        """Fährt Monitoring-Service herunter."""
        ...

    def get_monitoring_status(self) -> dict[str, Any]:
        """Gibt aktuellen Monitoring-Status zurück."""
        ...

    def get_dashboard_data(self) -> dict[str, Any]:
        """Gibt Dashboard-Daten zurück."""
        ...

    async def export_metrics(self, output_format: str = "prometheus") -> str:
        """Exportiert Metriken in gewünschtem Format."""
        ...


# =============================================================================
# UTILITY TYPES
# =============================================================================

@dataclass
class MonitoringConfig:
    """Monitoring-Konfiguration."""
    enabled: bool = True
    metrics_enabled: bool = True
    health_checks_enabled: bool = True
    alerts_enabled: bool = True
    voice_monitoring_enabled: bool = True
    performance_monitoring_enabled: bool = True

    # Export-Konfiguration
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True

    # Health-Check-Konfiguration
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5

    # Alert-Konfiguration
    alert_webhook_url: str | None = None
    alert_email_enabled: bool = False
    alert_slack_enabled: bool = False

    # Performance-Thresholds
    response_time_threshold_ms: float = 1000.0
    error_rate_threshold_percent: float = 5.0
    cpu_threshold_percent: float = 80.0
    memory_threshold_mb: float = 1024.0
