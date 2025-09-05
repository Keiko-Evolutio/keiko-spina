"""Main Monitoring Service Implementation.
Zentraler Service der alle Monitoring-Komponenten orchestriert.
"""

from datetime import datetime
from typing import Any

from kei_logging import get_logger

from .alert_manager import AlertManager
from .circuit_breaker import CircuitBreakerManager
from .health_checker import HealthChecker
from .interfaces import (
    IAlertManager,
    IHealthChecker,
    IMetricsCollector,
    IMonitoringService,
    IPerformanceMonitor,
    IVoiceWorkflowMonitor,
    MonitoringConfig,
)
from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor
from .voice_metrics import VoiceWorkflowMonitor

logger = get_logger(__name__)


class MonitoringService(IMonitoringService):
    """Haupt-Monitoring-Service Implementation.
    Orchestriert alle Monitoring-Komponenten und bietet einheitliche API.
    """

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()

        # Monitoring-Komponenten
        self._metrics_collector = MetricsCollector()
        self._health_checker = HealthChecker(self._metrics_collector)
        self._alert_manager = AlertManager(self._metrics_collector)
        self._voice_monitor = VoiceWorkflowMonitor(self._metrics_collector)
        self._performance_monitor = PerformanceMonitor(self._metrics_collector)
        self._circuit_breaker_manager = CircuitBreakerManager(self._metrics_collector)

        # Service-Status
        self._initialized = False
        self._running = False

        # Registriere Standard-Metriken
        self._register_standard_metrics()

        logger.info(f"Monitoring service created with config: {self.config}")

    def _register_standard_metrics(self) -> None:
        """Registriert Standard-Metrik-Hilfe-Texte."""
        help_texts = {
            # Voice Workflow Metriken
            "voice_workflow_total": "Total number of voice workflows started",
            "voice_workflow_duration_seconds": "Duration of voice workflows in seconds",
            "voice_workflow_success_rate": "Success rate of voice workflows",
            "voice_stt_duration_seconds": "Speech-to-text processing duration in seconds",
            "voice_stt_confidence": "Speech-to-text confidence score",
            "voice_orchestrator_duration_seconds": "Orchestrator processing duration in seconds",
            "voice_agent_selection_duration_seconds": "Agent selection duration in seconds",
            "voice_agent_execution_duration_seconds": "Agent execution duration in seconds",
            "voice_active_workflows": "Number of currently active voice workflows",
            "voice_workflow_error_rate": "Error rate of voice workflows",

            # Performance Metriken
            "http_request_duration_seconds": "HTTP request duration in seconds",
            "http_requests_total": "Total number of HTTP requests",
            "http_requests_per_second": "HTTP requests per second",
            "http_slow_requests_total": "Total number of slow HTTP requests",
            "system_cpu_usage_percent": "System CPU usage percentage",
            "system_memory_usage_mb": "System memory usage in MB",
            "system_network_usage_kb": "System network usage in KB",
            "process_memory_usage_mb": "Process memory usage in MB",
            "process_cpu_usage_percent": "Process CPU usage percentage",
            "queue_length": "Queue length",

            # Health Check Metriken
            "health_check_status": "Health check status (1=healthy, 0=unhealthy)",
            "health_check_duration_seconds": "Health check duration in seconds",
            "health_check_total_services": "Total number of services being monitored",
            "health_check_healthy_services": "Number of healthy services",
            "health_check_overall_ratio": "Overall health ratio",

            # Alert Metriken
            "alerts_total": "Total number of alerts sent",
            "alerts_resolved_total": "Total number of alerts resolved",
            "active_alerts": "Number of currently active alerts",

            # Circuit Breaker Metriken
            "circuit_breaker_requests_total": "Total number of circuit breaker requests",
            "circuit_breaker_request_duration_seconds": "Circuit breaker request duration in seconds",
            "circuit_breaker_state": "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            "circuit_breaker_opened_total": "Total number of times circuit breaker opened"
        }

        for metric_name, help_text in help_texts.items():
            self._metrics_collector.register_metric_help(metric_name, help_text)

    @property
    def metrics_collector(self) -> IMetricsCollector:
        """Metrics Collector."""
        return self._metrics_collector

    @property
    def health_checker(self) -> IHealthChecker:
        """Health Checker."""
        return self._health_checker

    @property
    def alert_manager(self) -> IAlertManager:
        """Alert Manager."""
        return self._alert_manager

    @property
    def voice_monitor(self) -> IVoiceWorkflowMonitor:
        """Voice Workflow Monitor."""
        return self._voice_monitor

    @property
    def performance_monitor(self) -> IPerformanceMonitor:
        """Performance Monitor."""
        return self._performance_monitor

    @property
    def circuit_breaker_manager(self) -> CircuitBreakerManager:
        """Circuit Breaker Manager."""
        return self._circuit_breaker_manager

    async def initialize(self) -> None:
        """Initialisiert Monitoring-Service."""
        if self._initialized:
            return

        try:
            logger.info("Initializing monitoring service...")

            # Health Checks starten (falls aktiviert)
            if self.config.health_checks_enabled:
                await self._health_checker.start_periodic_checks()
                logger.debug("Health checks started")

            # Alert Monitoring starten (falls aktiviert)
            if self.config.alerts_enabled:
                await self._alert_manager.start_alert_monitoring()
                logger.debug("Alert monitoring started")

            # Performance Monitoring starten (falls aktiviert)
            if self.config.performance_monitoring_enabled:
                await self._performance_monitor.start_system_monitoring()
                logger.debug("Performance monitoring started")

            # Standard Circuit Breaker für externe Services erstellen
            self._setup_default_circuit_breakers()

            # Service-Metriken initialisieren
            self._metrics_collector.set_gauge("monitoring_service_initialized", 1.0)
            self._metrics_collector.set_gauge("monitoring_service_start_time", datetime.utcnow().timestamp())

            self._initialized = True
            self._running = True

            logger.info("Monitoring service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt Monitoring-Service herunter."""
        if not self._running:
            return

        try:
            logger.info("Shutting down monitoring service...")

            # Alle Monitoring-Komponenten stoppen
            await self._health_checker.stop_periodic_checks()
            await self._alert_manager.stop_alert_monitoring()
            await self._performance_monitor.stop_system_monitoring()

            # Service-Metriken aktualisieren
            self._metrics_collector.set_gauge("monitoring_service_initialized", 0.0)

            self._running = False

            logger.info("Monitoring service shut down successfully")

        except Exception as e:
            logger.error(f"Error during monitoring service shutdown: {e}")

    def _setup_default_circuit_breakers(self) -> None:
        """Erstellt Standard Circuit Breaker für externe Services."""
        from .circuit_breaker import CircuitBreakerConfig

        # Azure OpenAI Circuit Breaker
        azure_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=60,
            success_threshold=3,
            timeout_seconds=30.0
        )
        self._circuit_breaker_manager.create_circuit_breaker("azure_openai", azure_config)

        # Redis Circuit Breaker
        redis_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=30,
            success_threshold=2,
            timeout_seconds=5.0
        )
        self._circuit_breaker_manager.create_circuit_breaker("redis", redis_config)

        logger.debug("Default circuit breakers created")

    async def export_metrics(self, output_format: str = "prometheus") -> str:
        """Exportiert Metriken in gewünschtem Format."""
        if output_format.lower() == "prometheus":
            return self._metrics_collector.export_prometheus_format()
        raise ValueError(f"Unsupported export format: {output_format}")

    def get_monitoring_status(self) -> dict[str, Any]:
        """Gibt aktuellen Monitoring-Status zurück."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "config": {
                "metrics_enabled": self.config.metrics_enabled,
                "health_checks_enabled": self.config.health_checks_enabled,
                "alerts_enabled": self.config.alerts_enabled,
                "voice_monitoring_enabled": self.config.voice_monitoring_enabled,
                "performance_monitoring_enabled": self.config.performance_monitoring_enabled
            },
            "components": {
                "metrics_collector": {
                    "metric_count": self._metrics_collector.get_metric_count()
                },
                "health_checker": {
                    "overall_health": self._health_checker.get_overall_health()
                },
                "alert_manager": {
                    "statistics": self._alert_manager.get_alert_statistics()
                },
                "voice_monitor": {
                    "statistics": self._voice_monitor.get_workflow_statistics()
                },
                "performance_monitor": {
                    "resource_stats": self._performance_monitor.get_resource_stats()
                },
                "circuit_breakers": {
                    "statistics": self._circuit_breaker_manager.get_all_statistics()
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Gibt Dashboard-Daten zurück."""
        return {
            "overview": {
                "status": "healthy" if self._running else "stopped",
                "uptime_seconds": (datetime.utcnow().timestamp() -
                                 self._metrics_collector.get_metrics()[0].timestamp.timestamp()
                                 if self._metrics_collector.get_metrics() else 0),
                "total_metrics": len(self._metrics_collector.get_metrics()),
                "active_alerts": len(self._alert_manager.get_active_alerts())
            },
            "voice_workflows": self._voice_monitor.get_workflow_statistics(),
            "performance": {
                "response_times": [
                    {
                        "endpoint": stats.endpoint,
                        "avg_ms": stats.avg_time_ms,
                        "p95_ms": stats.p95_time_ms
                    }
                    for stats in self._performance_monitor.get_response_time_stats()
                ],
                "resource_usage": self._performance_monitor.get_resource_stats()
            },
            "health": self._health_checker.get_overall_health(),
            "alerts": {
                "active": [
                    {
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self._alert_manager.get_active_alerts()
                ]
            }
        }
