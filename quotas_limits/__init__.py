# backend/quotas_limits/__init__.py
"""Vollständiges Quotas/Limits System für Keiko Personal Assistant

Implementiert Agent-spezifische Rate-Limiting, Budget-Propagation,
erweiterte Quota-Management-Features und Integration mit bestehenden Systemen.
"""

from __future__ import annotations

from kei_logging import get_logger

# Agent Rate Limiting
from .agent_rate_limiter import (
    AgentRateLimit,
    AgentRateLimiter,
    BurstConfig,
    CapabilityRateLimit,
    PriorityLevel,
    RateLimitResult,
    RateLimitWindow,
    agent_rate_limiter,
)

# Budget Management
from .budget_manager import (
    Budget,
    BudgetAllocation,
    BudgetManager,
    BudgetPropagationRule,
    BudgetTransfer,
    BudgetUsage,
    CostTracker,
    OperationCost,
    budget_manager,
)

# Core Quota System
from .core_quota_manager import (
    QuotaAllocation,
    QuotaCheckResult,
    QuotaManager,
    QuotaPolicy,
    QuotaScope,
    QuotaType,
    QuotaUsage,
    QuotaViolation,
    quota_manager,
)

# Quota Analytics
from .quota_analytics import (
    AlertRule,
    PredictiveAnalysis,
    QuotaAnalytics,
    QuotaMetrics,
    QuotaReport,
    UsagePattern,
    UsageTrend,
    quota_analytics,
)

# Quota Middleware
from .quota_middleware import (
    QuotaConfig,
    QuotaEnforcementMiddleware,
    QuotaEnforcementResult,
    check_api_quota,
    enforce_agent_quotas,
    enforce_budget_limits,
    require_quota_compliance,
)

# Monitoring & Alerting
from .quota_monitoring import (
    AlertSeverity,
    MonitoringRule,
    PerformanceMetrics,
    QuotaAlert,
    QuotaHealthCheck,
    QuotaMonitor,
    quota_monitor,
)

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "AgentRateLimit",
    # Agent Rate Limiting
    "AgentRateLimiter",
    "AlertRule",
    "AlertSeverity",
    "Budget",
    "BudgetAllocation",
    # Budget Management
    "BudgetManager",
    "BudgetPropagationRule",
    "BudgetTransfer",
    "BudgetUsage",
    "BurstConfig",
    "CapabilityRateLimit",
    "CostTracker",
    "MonitoringRule",
    "OperationCost",
    "PerformanceMetrics",
    "PredictiveAnalysis",
    "PriorityLevel",
    "QuotaAlert",
    "QuotaAllocation",
    # Quota Analytics
    "QuotaAnalytics",
    "QuotaCheckResult",
    "QuotaConfig",
    # Quota Middleware
    "QuotaEnforcementMiddleware",
    "QuotaEnforcementResult",
    "QuotaHealthCheck",
    # Core Quota System
    "QuotaManager",
    "QuotaMetrics",
    # Monitoring & Alerting
    "QuotaMonitor",
    "QuotaPolicy",
    "QuotaReport",
    "QuotaScope",
    "QuotaType",
    "QuotaUsage",
    "QuotaViolation",
    "RateLimitResult",
    "RateLimitWindow",
    "UsagePattern",
    "UsageTrend",
    "agent_rate_limiter",
    "budget_manager",
    "enforce_agent_quotas",
    "enforce_budget_limits",
    "quota_analytics",
    "quota_manager",
    "quota_monitor",
    "require_quota_compliance",
]

# Quota-System Status
def get_quota_system_status() -> dict:
    """Gibt Status des Quota-Systems zurück."""
    return {
        "package": "backend.quotas_limits",
        "version": "1.0.0",
        "components": {
            "core_quota_manager": True,
            "agent_rate_limiter": True,
            "budget_manager": True,
            "quota_analytics": True,
            "quota_middleware": True,
            "quota_monitoring": True,
        },
        "features": {
            "agent_specific_rate_limiting": True,
            "capability_specific_limits": True,
            "configurable_time_windows": True,
            "budget_propagation": True,
            "cost_tracking": True,
            "timeout_propagation": True,
            "burst_limits": True,
            "priority_based_allocation": True,
            "quota_sharing": True,
            "automatic_quota_adjustment": True,
            "real_time_monitoring": True,
            "predictive_analytics": True,
            "alerting_system": True,
            "usage_reporting": True,
            "circuit_breaker_pattern": True,
            "async_processing": True,
            "batch_operations": True,
            "caching_optimization": True,
        },
        "quota_types": [
            "request_rate",
            "operation_count",
            "data_volume",
            "compute_time",
            "memory_usage",
            "storage_quota",
            "bandwidth_limit",
            "concurrent_operations"
        ],
        "rate_limit_algorithms": [
            "sliding_window",
            "token_bucket",
            "fixed_window",
            "leaky_bucket"
        ],
        "budget_types": [
            "monetary_budget",
            "compute_credits",
            "api_calls",
            "data_transfer",
            "storage_usage"
        ]
    }

logger.info(f"Quotas/Limits System geladen - Status: {get_quota_system_status()}")
