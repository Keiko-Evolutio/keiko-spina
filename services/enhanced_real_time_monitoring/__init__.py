# backend/services/enhanced_real_time_monitoring/__init__.py
"""Enhanced Real-time Monitoring Package.

Implementiert Enterprise-Grade Real-time Monitoring mit Saga Coordination,
Distributed Tracing und Live-Dashboards für alle Services.
"""

from __future__ import annotations

from .data_models import (
    AlertRule,
    AlertSeverity,
    CompensationAction,
    DistributedTrace,
    LiveDashboardData,
    MetricType,
    MonitoringAlert,
    MonitoringConfiguration,
    MonitoringMetric,
    MonitoringPerformanceMetrics,
    MonitoringScope,
    PerformanceMetrics,
    SagaStatus,
    SagaStep,
    SagaTransaction,
    TraceSpan,
    TraceStatus,
)
from .distributed_tracing_engine import DistributedTracingEngine
from .live_dashboard_engine import LiveDashboardEngine
from .real_time_monitoring_engine import EnhancedRealTimeMonitoringEngine
from .saga_coordinator_engine import SagaCoordinatorEngine
from .service_integration_layer import MonitoringServiceIntegrationLayer

__all__ = [
    # Core Components
    "EnhancedRealTimeMonitoringEngine",
    "SagaCoordinatorEngine",
    "DistributedTracingEngine",
    "LiveDashboardEngine",
    "MonitoringServiceIntegrationLayer",

    # Data Models
    "MonitoringMetric",
    "PerformanceMetrics",
    "AlertRule",
    "MonitoringAlert",
    "SagaStep",
    "SagaTransaction",
    "TraceSpan",
    "DistributedTrace",
    "LiveDashboardData",
    "MonitoringConfiguration",
    "MonitoringPerformanceMetrics",

    # Enums
    "MonitoringScope",
    "MetricType",
    "AlertSeverity",
    "SagaStatus",
    "CompensationAction",
    "TraceStatus",

    # Factory Functions
    "create_enhanced_real_time_monitoring_engine",
    "create_saga_coordinator_engine",
    "create_distributed_tracing_engine",
    "create_live_dashboard_engine",
    "create_monitoring_service_integration_layer",
    "create_integrated_real_time_monitoring_system",
]

__version__ = "1.0.0"


def create_enhanced_real_time_monitoring_engine(
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    configuration=None
) -> EnhancedRealTimeMonitoringEngine:
    """Factory-Funktion für Enhanced Real-time Monitoring Engine.

    Args:
        dependency_resolution_engine: Dependency Resolution Engine (optional)
        quota_management_engine: Quota Management Engine (optional)
        security_integration_engine: Security Integration Engine (optional)
        configuration: Monitoring-Konfiguration (optional)

    Returns:
        Konfigurierte Enhanced Real-time Monitoring Engine
    """
    return EnhancedRealTimeMonitoringEngine(
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        configuration=configuration
    )


def create_saga_coordinator_engine() -> SagaCoordinatorEngine:
    """Factory-Funktion für Saga Coordinator Engine.

    Returns:
        Konfigurierte Saga Coordinator Engine
    """
    return SagaCoordinatorEngine()


def create_distributed_tracing_engine() -> DistributedTracingEngine:
    """Factory-Funktion für Distributed Tracing Engine.

    Returns:
        Konfigurierte Distributed Tracing Engine
    """
    return DistributedTracingEngine()


def create_live_dashboard_engine() -> LiveDashboardEngine:
    """Factory-Funktion für Live Dashboard Engine.

    Returns:
        Konfigurierte Live Dashboard Engine
    """
    return LiveDashboardEngine()


def create_monitoring_service_integration_layer(
    monitoring_engine: EnhancedRealTimeMonitoringEngine,
    saga_coordinator: SagaCoordinatorEngine,
    tracing_engine: DistributedTracingEngine,
    dashboard_engine: LiveDashboardEngine,
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None
) -> MonitoringServiceIntegrationLayer:
    """Factory-Funktion für Monitoring Service Integration Layer.

    Args:
        monitoring_engine: Real-time Monitoring Engine
        saga_coordinator: Saga Coordinator Engine
        tracing_engine: Distributed Tracing Engine
        dashboard_engine: Live Dashboard Engine
        dependency_resolution_engine: Dependency Resolution Engine (optional)
        quota_management_engine: Quota Management Engine (optional)
        security_integration_engine: Security Integration Engine (optional)

    Returns:
        Konfigurierte Monitoring Service Integration Layer
    """
    return MonitoringServiceIntegrationLayer(
        monitoring_engine=monitoring_engine,
        saga_coordinator=saga_coordinator,
        tracing_engine=tracing_engine,
        dashboard_engine=dashboard_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine
    )


def create_integrated_real_time_monitoring_system(
    dependency_resolution_engine=None,
    quota_management_engine=None,
    security_integration_engine=None,
    monitoring_configuration=None
) -> dict:
    """Factory-Funktion für integriertes Real-time Monitoring System.

    Args:
        dependency_resolution_engine: Dependency Resolution Engine (optional)
        quota_management_engine: Quota Management Engine (optional)
        security_integration_engine: Security Integration Engine (optional)
        monitoring_configuration: Monitoring-Konfiguration (optional)

    Returns:
        Dictionary mit allen konfigurierten Komponenten
    """
    # Erstelle alle Komponenten
    monitoring_engine = create_enhanced_real_time_monitoring_engine(
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine,
        configuration=monitoring_configuration
    )

    saga_coordinator = create_saga_coordinator_engine()

    tracing_engine = create_distributed_tracing_engine()

    dashboard_engine = create_live_dashboard_engine()

    service_integration_layer = create_monitoring_service_integration_layer(
        monitoring_engine=monitoring_engine,
        saga_coordinator=saga_coordinator,
        tracing_engine=tracing_engine,
        dashboard_engine=dashboard_engine,
        dependency_resolution_engine=dependency_resolution_engine,
        quota_management_engine=quota_management_engine,
        security_integration_engine=security_integration_engine
    )

    return {
        "monitoring_engine": monitoring_engine,
        "saga_coordinator": saga_coordinator,
        "tracing_engine": tracing_engine,
        "dashboard_engine": dashboard_engine,
        "service_integration_layer": service_integration_layer
    }
