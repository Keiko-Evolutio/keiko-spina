# backend/observability/__init__.py
"""Vollständiges Observability-Metrics-System für Keiko Personal Assistant

Stellt zentrale Funktionen für Tracing, Metriken und erweiterte Observability bereit.
Implementiert Enterprise-Grade Observability mit Agent-spezifischen Metriken,
Real-time-Aggregation, System-Integration und Prometheus-Export.
"""

from __future__ import annotations

from kei_logging import get_logger

# Base Classes und Common Patterns
from .base_metrics import (
    BaseErrorTracker,
    BaseLatencyTracker,
    BaseMetricsCollector,
    BaseRateTracker,
    LatencyTrackerProtocol,
    MetricDataPoint,
    MetricsCollectorProtocol,
    MetricsConstants,
    build_headers_with_context,
    calculate_percentiles,
    sanitize_metric_name,
)

# Bestehende Tracing-Funktionalität
from .tracing import (
    # Verfügbarkeits-Flag
    OPENTELEMETRY_AVAILABLE,
    # Konfiguration
    ObservabilityConfig,
    add_span_attributes,
    get_current_trace_id,
    get_observability_status,
    get_tracing_status,
    is_observability_available,
    # Statusfunktionen
    is_tracing_healthy,
    record_exception_in_span,
    # Kernfunktionen
    setup_opentelemetry,
    shutdown_tracing,
    trace_function,
    trace_span,
)

# Neue Agent-spezifische Metriken (optional import)
try:
    from .agent_metrics import (
        AgentMetricsCollector,
        ErrorCategory,
        ErrorMetrics,
        LatencyTracker,
        PercentileMetrics,
        QueueMetrics,
        RateMetrics,
        RateTracker,
        ToolCallMetrics,
        get_agent_metrics_collector,
        get_all_agent_metrics,
        reset_all_agent_metrics,
    )
    AGENT_METRICS_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Agent-Metriken
    AgentMetricsCollector = None
    ErrorCategory = None
    ErrorMetrics = None
    LatencyTracker = None
    PercentileMetrics = None
    QueueMetrics = None
    RateMetrics = None
    RateTracker = None
    ToolCallMetrics = None
    get_agent_metrics_collector = None
    get_all_agent_metrics = None
    reset_all_agent_metrics = None
    AGENT_METRICS_AVAILABLE = False

# Neue Metriken-Aggregation (optional import)
try:
    from .metrics_aggregator import (
        AggregatedMetric,
        AggregationType,
        AggregationWindow,
        MetricDataPoint,
        MetricsAggregator,
        MetricsConfig,
        TimeSeriesBuffer,
        metrics_aggregator,
    )
    METRICS_AGGREGATOR_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Metriken-Aggregation
    AggregatedMetric = None
    AggregationType = None
    AggregationWindow = None
    MetricDataPoint = None
    MetricsAggregator = None
    MetricsConfig = None
    TimeSeriesBuffer = None
    metrics_aggregator = None
    METRICS_AGGREGATOR_AVAILABLE = False

# System-Integration (optional import)
try:
    from .system_integration_metrics import (
        AuditEventType,
        AuditMetricsCollector,
        PolicyEventType,
        PolicyMetricsCollector,
        SecurityEventType,
        SecurityMetricsCollector,
        SystemComponent,
        SystemIntegrationMetrics,
        SystemMetricsConfig,
        TaskManagementMetricsCollector,
        system_integration_metrics,
    )
    SYSTEM_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende System-Integration-Metriken
    AuditEventType = None
    AuditMetricsCollector = None
    PolicyEventType = None
    PolicyMetricsCollector = None
    SecurityEventType = None
    SecurityMetricsCollector = None
    SystemComponent = None
    SystemIntegrationMetrics = None
    SystemMetricsConfig = None
    TaskManagementMetricsCollector = None
    system_integration_metrics = None
    SYSTEM_INTEGRATION_AVAILABLE = False

# Export und Alerting (optional import)
try:
    from .metrics_export import (
        AlertEvent,
        AlertManager,
        AlertRule,
        AlertRuleRequest,
        AlertSeverity,
        DashboardConfig,
        MetricsExportAPI,
        MetricsFormat,
        MetricsQueryRequest,
        MetricsResponse,
        PrometheusExporter,
        setup_metrics_export_api,
    )
    METRICS_EXPORT_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Metriken-Export-Funktionalität
    AlertEvent = None
    AlertManager = None
    AlertRule = None
    AlertRuleRequest = None
    AlertSeverity = None
    DashboardConfig = None
    MetricsExportAPI = None
    MetricsFormat = None
    MetricsQueryRequest = None
    MetricsResponse = None
    PrometheusExporter = None
    setup_metrics_export_api = None
    METRICS_EXPORT_AVAILABLE = False

# Logfire-Integration (optional import)
try:
    from .logfire_config import (
        LogfireEnvironment,
        LogfireMode,
        LogfirePIIRedactionConfig,
        LogfireSettings,
        get_logfire_settings,
        validate_logfire_config,
    )
    from .logfire_integration import (
        LogfireManager,
        get_logfire_manager,
        initialize_logfire,
        shutdown_logfire,
    )
    from .logfire_llm_tracking import (
        LLMCallContext,
        LLMCallMetrics,
        LLMCallTracker,
        get_llm_tracker,
        track_llm_call,
    )
    LOGFIRE_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Logfire-Integration
    LogfireManager = None
    get_logfire_manager = None
    initialize_logfire = None
    shutdown_logfire = None
    LogfireSettings = None
    LogfireMode = None
    LogfireEnvironment = None
    LogfirePIIRedactionConfig = None
    get_logfire_settings = None
    validate_logfire_config = None
    LLMCallMetrics = None
    LLMCallTracker = None
    LLMCallContext = None
    get_llm_tracker = None
    track_llm_call = None
    LOGFIRE_INTEGRATION_AVAILABLE = False

logger = get_logger(__name__)

# Version und Metadata
__version__ = "1.0.0"
__author__ = "KEI-Agent-Framework Team"
__description__ = "Enterprise-Grade Observability-Metrics-System"

# Alle verfügbaren Exports
__all__ = [
    "OPENTELEMETRY_AVAILABLE",
    "BaseErrorTracker",
    "BaseLatencyTracker",
    "BaseMetricsCollector",
    "BaseRateTracker",
    "LatencyTrackerProtocol",
    "MetricDataPoint",
    "MetricsCollectorProtocol",
    # Base Classes und Common Patterns
    "MetricsConstants",
    # Bestehende Tracing-Funktionalität
    "ObservabilityConfig",
    "add_span_attributes",
    "build_headers_with_context",
    "calculate_percentiles",
    "get_current_trace_id",
    "get_observability_status",
    "get_tracing_status",
    "is_observability_available",
    "is_tracing_healthy",
    "record_exception_in_span",
    "sanitize_metric_name",
    "setup_opentelemetry",
    "shutdown_tracing",
    "trace_function",
    "trace_span",
]

# Erweitere __all__ basierend auf verfügbaren Komponenten
if AGENT_METRICS_AVAILABLE:
    __all__.extend([
        "AgentMetricsCollector",
        "ErrorCategory",
        "ErrorMetrics",
        "LatencyTracker",
        "PercentileMetrics",
        "QueueMetrics",
        "RateMetrics",
        "RateTracker",
        "ToolCallMetrics",
        "get_agent_metrics_collector",
        "get_all_agent_metrics",
        "reset_all_agent_metrics",
    ])

if METRICS_AGGREGATOR_AVAILABLE:
    __all__.extend([
        "AggregatedMetric",
        "AggregationType",
        "AggregationWindow",
        "MetricDataPoint",
        "MetricsAggregator",
        "MetricsConfig",
        "TimeSeriesBuffer",
        "metrics_aggregator",
    ])

if SYSTEM_INTEGRATION_AVAILABLE:
    __all__.extend([
        "AuditEventType",
        "AuditMetricsCollector",
        "PolicyEventType",
        "PolicyMetricsCollector",
        "SecurityEventType",
        "SecurityMetricsCollector",
        "SystemComponent",
        "SystemIntegrationMetrics",
        "SystemMetricsConfig",
        "TaskManagementMetricsCollector",
        "system_integration_metrics",
    ])

if METRICS_EXPORT_AVAILABLE:
    __all__.extend([
        "AlertEvent",
        "AlertManager",
        "AlertRule",
        "AlertRuleRequest",
        "AlertSeverity",
        "DashboardConfig",
        "MetricsExportAPI",
        "MetricsFormat",
        "MetricsQueryRequest",
        "MetricsResponse",
        "PrometheusExporter",
        "setup_metrics_export_api",
    ])

if LOGFIRE_INTEGRATION_AVAILABLE:
    __all__.extend([
        "LLMCallContext",
        "LLMCallMetrics",
        "LLMCallTracker",
        "LogfireEnvironment",
        "LogfireManager",
        "LogfireMode",
        "LogfirePIIRedactionConfig",
        "LogfireSettings",
        "get_llm_tracker",
        "get_logfire_manager",
        "get_logfire_settings",
        "initialize_logfire",
        "shutdown_logfire",
        "track_llm_call",
        "validate_logfire_config",
    ])

# Utility-Funktionen
__all__.extend([
    "get_enhanced_observability_status",
    "start_observability_system",
    "stop_observability_system",
])


def get_enhanced_observability_status() -> dict:
    """Gibt erweiterten Status des Observability-Systems zurück.

    Returns:
        Status-Dictionary mit allen Komponenten
    """
    try:
        status = {
            "version": __version__,
            "status": "operational",
            "components": {
                "tracing": {
                    "available": OPENTELEMETRY_AVAILABLE,
                    "status": "running" if is_tracing_healthy() else "stopped"
                },
                "agent_metrics": {
                    "available": AGENT_METRICS_AVAILABLE,
                    "status": "not_implemented"
                },
                "metrics_aggregator": {
                    "available": METRICS_AGGREGATOR_AVAILABLE,
                    "status": "not_implemented"
                },
                "system_integration": {
                    "available": SYSTEM_INTEGRATION_AVAILABLE,
                    "status": "not_implemented"
                },
                "metrics_export": {
                    "available": METRICS_EXPORT_AVAILABLE,
                    "status": "not_implemented"
                }
            },
            "features": {
                "distributed_tracing": OPENTELEMETRY_AVAILABLE,
                "agent_specific_metrics": AGENT_METRICS_AVAILABLE,
                "latency_percentiles": AGENT_METRICS_AVAILABLE,
                "error_categorization": AGENT_METRICS_AVAILABLE,
                "queue_depth_monitoring": AGENT_METRICS_AVAILABLE,
                "tool_call_tracking": AGENT_METRICS_AVAILABLE,
                "real_time_aggregation": METRICS_AGGREGATOR_AVAILABLE,
                "prometheus_export": METRICS_EXPORT_AVAILABLE,
                "alerting": METRICS_EXPORT_AVAILABLE,
                "dashboard_integration": METRICS_EXPORT_AVAILABLE,
                "system_integration": SYSTEM_INTEGRATION_AVAILABLE,
                "logfire_integration": LOGFIRE_INTEGRATION_AVAILABLE,
                "llm_call_tracking": LOGFIRE_INTEGRATION_AVAILABLE,
                "pii_redaction": LOGFIRE_INTEGRATION_AVAILABLE
            }
        }

        # Erweitere Status mit verfügbaren Komponenten
        if AGENT_METRICS_AVAILABLE:
            try:
                all_agents = get_all_agent_metrics()
                status["components"]["agent_metrics"]["active_agents"] = len(all_agents)
                status["components"]["agent_metrics"]["status"] = "running" if len(all_agents) > 0 else "idle"
            except:
                pass

        if METRICS_AGGREGATOR_AVAILABLE:
            try:
                aggregator_stats = metrics_aggregator.get_aggregator_statistics()
                status["components"]["metrics_aggregator"]["status"] = "running" if aggregator_stats["is_running"] else "stopped"
                status["components"]["metrics_aggregator"]["metrics_collected"] = aggregator_stats["metrics_collected"]
            except:
                pass

        if SYSTEM_INTEGRATION_AVAILABLE:
            try:
                integration_stats = system_integration_metrics.get_integration_statistics()
                status["components"]["system_integration"]["status"] = "running" if integration_stats["is_running"] else "stopped"
                status["components"]["system_integration"]["collection_cycles"] = integration_stats["collection_cycles"]
            except:
                pass

        return status

    except Exception as e:
        return {
            "version": __version__,
            "status": "error",
            "error": str(e),
            "components": {},
            "features": {}
        }


async def start_observability_system() -> bool:
    """Startet das vollständige Observability-System.

    Returns:
        True wenn erfolgreich gestartet
    """
    try:
        success = True

        # Starte Tracing (inklusive Logfire-Integration)
        if OPENTELEMETRY_AVAILABLE:
            try:
                setup_opentelemetry()
                logger.info("OpenTelemetry-Tracing gestartet")
            except Exception as e:
                logger.warning(f"OpenTelemetry-Start fehlgeschlagen: {e}")
                success = False

        # Starte Logfire-Integration (falls verfügbar)
        if LOGFIRE_INTEGRATION_AVAILABLE:
            try:
                logfire_settings = get_logfire_settings()
                if initialize_logfire(logfire_settings):
                    logger.info("Logfire-Integration gestartet")
                else:
                    logger.warning("Logfire-Integration konnte nicht gestartet werden")
            except Exception as e:
                logger.warning(f"Logfire-Start fehlgeschlagen: {e}")

        # Starte Metriken-Aggregator
        if METRICS_AGGREGATOR_AVAILABLE:
            try:
                await metrics_aggregator.start()
                logger.info("Metrics Aggregator gestartet")
            except Exception as e:
                logger.warning(f"Metrics-Aggregator-Start fehlgeschlagen: {e}")
                success = False

        # Starte System-Integration
        if SYSTEM_INTEGRATION_AVAILABLE:
            try:
                await system_integration_metrics.start()
                logger.info("System Integration Metrics gestartet")
            except Exception as e:
                logger.warning(f"System-Integration-Start fehlgeschlagen: {e}")
                success = False

        if success:
            logger.info("Observability-System erfolgreich gestartet")
        else:
            logger.warning("Observability-System teilweise gestartet")

        return success

    except Exception as e:
        logger.exception(f"Observability-System-Start fehlgeschlagen: {e}")
        return False


async def stop_observability_system() -> bool:
    """Stoppt das vollständige Observability-System.

    Returns:
        True wenn erfolgreich gestoppt
    """
    try:
        success = True

        # Stoppe System-Integration
        if SYSTEM_INTEGRATION_AVAILABLE:
            try:
                await system_integration_metrics.stop()
                logger.info("System Integration Metrics gestoppt")
            except Exception as e:
                logger.warning(f"System-Integration-Stop fehlgeschlagen: {e}")
                success = False

        # Stoppe Metriken-Aggregator
        if METRICS_AGGREGATOR_AVAILABLE:
            try:
                await metrics_aggregator.stop()
                logger.info("Metrics Aggregator gestoppt")
            except Exception as e:
                logger.warning(f"Metrics-Aggregator-Stop fehlgeschlagen: {e}")
                success = False

        # Stoppe Logfire-Integration
        if LOGFIRE_INTEGRATION_AVAILABLE:
            try:
                shutdown_logfire()
                logger.info("Logfire-Integration gestoppt")
            except Exception as e:
                logger.warning(f"Logfire-Stop fehlgeschlagen: {e}")

        # Stoppe Tracing
        if OPENTELEMETRY_AVAILABLE:
            try:
                import asyncio
                # Prüfe ob bereits ein Event Loop läuft
                try:
                    loop = asyncio.get_running_loop()
                    # Event Loop läuft bereits - erstelle Task
                    task = loop.create_task(shutdown_tracing())
                    # Warte nicht auf Completion, da wir im Shutdown sind
                    logger.info("OpenTelemetry-Tracing Shutdown-Task erstellt")
                except RuntimeError:
                    # Kein Event Loop läuft - erstelle neuen
                    asyncio.run(shutdown_tracing())
                    logger.info("OpenTelemetry-Tracing gestoppt")
            except Exception as e:
                logger.warning(f"OpenTelemetry-Stop fehlgeschlagen: {e}")
                success = False

        if success:
            logger.info("Observability-System erfolgreich gestoppt")
        else:
            logger.warning("Observability-System teilweise gestoppt")

        return success

    except Exception as e:
        logger.exception(f"Observability-System-Stop fehlgeschlagen: {e}")
        return False

# Flag auf Paket-Ebene, um die Verfügbarkeit des Tracing-Moduls anzuzeigen
TRACING_MODULE_AVAILABLE: bool = True


def record_custom_metric(name: str, value: float, tags: dict | None = None) -> None:
    """Zeichnet eine benutzerdefinierte Metrik auf.

    Args:
        name: Name der Metrik
        value: Wert der Metrik
        tags: Optionale Tags für die Metrik
    """
    logger.debug(f"Custom metric recorded: {name}={value}, tags={tags}")

# Hinweis-Log beim Laden des Pakets
logger.debug("Observability-Paket geladen")
