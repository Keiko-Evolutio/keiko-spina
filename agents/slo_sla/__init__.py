# backend/agents/slo_sla/__init__.py
"""Vollständiges SLO/SLA Management-System

Implementiert Enterprise-Grade Service Level Objectives (SLOs) und
Service Level Agreements (SLAs) mit Real-time-Monitoring, automatischem
Alerting und proaktivem Capacity-Management.
"""

from .breach_manager import (
    BreachNotification,
    EscalationWorkflow,
    RecoveryTracker,
    SLABreachManager,
)
from .capacity_planner import (
    CapacityPlanner,
    PerformanceBudget,
    ScalingRecommendation,
    TrendAnalyzer,
)
from .config import CapabilitySLAConfig, CapabilitySLOConfig, SLAConfig, SLOConfig, SLOSLAConfig
from .coordinator import SLOSLACoordinator, SLOSLAPolicy, SLOSLAReport
from .models import (
    SLABreach,
    SLADefinition,
    SLAMetrics,
    SLAType,
    SLODefinition,
    SLOMetrics,
    SLOType,
    SLOViolation,
    TimeWindow,
    ViolationSeverity,
)
from .monitor import ErrorRateTracker, PercentileCalculator, SlidingWindowStats, SLOSLAMonitor

# Version Information
__version__ = "1.0.0"
__author__ = "SLO/SLA Management Team"

# Export All Public APIs
__all__ = [
    # Core Models
    "SLODefinition",
    "SLADefinition",
    "SLOMetrics",
    "SLAMetrics",
    "SLOViolation",
    "SLABreach",
    "SLOType",
    "SLAType",
    "TimeWindow",
    "ViolationSeverity",
    # Configuration
    "SLOConfig",
    "SLAConfig",
    "SLOSLAConfig",
    "CapabilitySLOConfig",
    "CapabilitySLAConfig",
    # Monitoring
    "SLOSLAMonitor",
    "PercentileCalculator",
    "ErrorRateTracker",
    "SlidingWindowStats",
    # Breach Management
    "SLABreachManager",
    "EscalationWorkflow",
    "BreachNotification",
    "RecoveryTracker",
    # Capacity Planning
    "CapacityPlanner",
    "PerformanceBudget",
    "ScalingRecommendation",
    "TrendAnalyzer",
    # Coordination
    "SLOSLACoordinator",
    "SLOSLAPolicy",
    "SLOSLAReport",
    # Version
    "__version__",
]

# Logging-Konfiguration
from kei_logging import get_logger

logger = get_logger(__name__)
logger.info(f"KEI-Agent SLO/SLA Management System v{__version__} initialisiert")
