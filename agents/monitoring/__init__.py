# backend/agents/monitoring/__init__.py
"""Monitoring-Module für das Agent-Framework.

Stellt umfassende Monitoring- und Observability-Funktionen bereit:
- Performance-Metriken
- Health-Monitoring
- Alerting-System
- Distributed Tracing
- Custom Metrics
"""

from __future__ import annotations

from .alert_manager import (
    Alert,
    AlertChannel,
    AlertConfig,
    AlertLevel,
    AlertManager,
    AlertRule,
)
from .health_monitor import (
    HealthCheck,
    HealthCheckResult,
    HealthConfig,
    HealthMonitor,
    HealthStatus,
)
from .metrics_collector import (
    CustomMetric,
    MetricAggregation,
    MetricsCollector,
    MetricsConfig,
    MetricValue,
)
from .performance_monitor import (
    MetricType,
    PerformanceConfig,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReport,
)
from .tracing_manager import (
    Span,
    SpanStatus,
    TraceContext,
    TracingConfig,
    TracingManager,
)

# Versionsinformationen
__version__ = "1.0.0"
__author__ = "Monitoring Team"

# Paket-Exporte
__all__ = [
    # Performance-Monitoring
    "PerformanceMonitor",
    "PerformanceConfig",
    "MetricType",
    "PerformanceMetric",
    "PerformanceReport",

    # Health-Monitoring
    "HealthMonitor",
    "HealthConfig",
    "HealthStatus",
    "HealthCheck",
    "HealthCheckResult",

    # Alert-Management
    "AlertManager",
    "AlertConfig",
    "AlertLevel",
    "Alert",
    "AlertRule",
    "AlertChannel",

    # Metrics-Collection
    "MetricsCollector",
    "MetricsConfig",
    "CustomMetric",
    "MetricValue",
    "MetricAggregation",

    # Distributed-Tracing
    "TracingManager",
    "TracingConfig",
    "TraceContext",
    "Span",
    "SpanStatus",
]
