# backend/services/enhanced_quotas_limits_management/__init__.py
"""Enhanced Quotas & Limits Management Package.

Implementiert Enterprise-Grade Resource-Management mit Multi-Tenant-Isolation,
Real-time Monitoring, Rate Limiting und Integration mit Enhanced Security.
"""

from __future__ import annotations

from .api_contracts_health_engine import APIContractsHealthEngine
from .data_models import (
    APIContract,
    EnforcementAction,
    HealthCheckResult,
    HealthStatus,
    QuotaAnalytics,
    QuotaCheckResult,
    QuotaPerformanceMetrics,
    QuotaPeriod,
    QuotaScope,
    QuotaStatus,
    QuotaUsage,
    QuotaViolation,
    RateLimit,
    ResourceQuota,
    ResourceType,
)
from .quota_analytics_engine import QuotaAnalyticsEngine
from .quota_enforcement_engine import QuotaEnforcementEngine
from .quota_management_engine import EnhancedQuotaManagementEngine
from .rate_limiting_engine import RateLimitingEngine
from .service_integration_layer import ServiceIntegrationLayer

__all__ = [
    # Core Components
    "EnhancedQuotaManagementEngine",
    "RateLimitingEngine",
    "QuotaEnforcementEngine",
    "QuotaAnalyticsEngine",
    "APIContractsHealthEngine",
    "ServiceIntegrationLayer",

    # Data Models
    "ResourceQuota",
    "QuotaUsage",
    "QuotaViolation",
    "RateLimit",
    "QuotaCheckResult",
    "APIContract",
    "HealthCheckResult",
    "QuotaAnalytics",
    "QuotaPerformanceMetrics",

    # Enums
    "ResourceType",
    "QuotaScope",
    "QuotaPeriod",
    "QuotaStatus",
    "EnforcementAction",
    "HealthStatus",

    # Factory Functions
    "create_enhanced_quota_management_engine",
    "create_rate_limiting_engine",
    "create_quota_enforcement_engine",
    "create_quota_analytics_engine",
    "create_api_contracts_health_engine",
    "create_integrated_quotas_limits_system",
]

__version__ = "1.0.0"


def create_enhanced_quota_management_engine(
    security_integration_engine,
    existing_quota_manager=None,
    rate_limiting_engine=None,
    enforcement_engine=None,
    analytics_engine=None
) -> EnhancedQuotaManagementEngine:
    """Factory-Funktion für Enhanced Quota Management Engine.

    Args:
        security_integration_engine: Enhanced Security Integration Engine
        existing_quota_manager: Bestehender Quota Manager (optional)
        rate_limiting_engine: Rate Limiting Engine (optional)
        enforcement_engine: Quota Enforcement Engine (optional)
        analytics_engine: Quota Analytics Engine (optional)

    Returns:
        Konfigurierte Enhanced Quota Management Engine
    """
    return EnhancedQuotaManagementEngine(
        security_integration_engine=security_integration_engine,
        existing_quota_manager=existing_quota_manager,
        rate_limiting_engine=rate_limiting_engine,
        enforcement_engine=enforcement_engine,
        analytics_engine=analytics_engine
    )


def create_rate_limiting_engine(
    existing_agent_rate_limiter=None,
    redis_rate_limiter=None
) -> RateLimitingEngine:
    """Factory-Funktion für Rate Limiting Engine.

    Args:
        existing_agent_rate_limiter: Bestehender Agent Rate Limiter (optional)
        redis_rate_limiter: Redis Rate Limiter (optional)

    Returns:
        Konfigurierte Rate Limiting Engine
    """
    return RateLimitingEngine(
        existing_agent_rate_limiter=existing_agent_rate_limiter,
        redis_rate_limiter=redis_rate_limiter
    )


def create_quota_enforcement_engine() -> QuotaEnforcementEngine:
    """Factory-Funktion für Quota Enforcement Engine.

    Returns:
        Konfigurierte Quota Enforcement Engine
    """
    return QuotaEnforcementEngine()


def create_quota_analytics_engine() -> QuotaAnalyticsEngine:
    """Factory-Funktion für Quota Analytics Engine.

    Returns:
        Konfigurierte Quota Analytics Engine
    """
    return QuotaAnalyticsEngine()


def create_api_contracts_health_engine() -> APIContractsHealthEngine:
    """Factory-Funktion für API Contracts & Health Engine.

    Returns:
        Konfigurierte API Contracts & Health Engine
    """
    return APIContractsHealthEngine()


def create_integrated_quotas_limits_system(
    security_integration_engine,
    existing_quota_manager=None,
    redis_rate_limiter=None
) -> dict:
    """Factory-Funktion für integriertes Quotas & Limits System.

    Args:
        security_integration_engine: Enhanced Security Integration Engine
        existing_quota_manager: Bestehender Quota Manager (optional)
        redis_rate_limiter: Redis Rate Limiter (optional)

    Returns:
        Dictionary mit allen konfigurierten Komponenten
    """
    # Erstelle alle Komponenten
    rate_limiting_engine = create_rate_limiting_engine(
        redis_rate_limiter=redis_rate_limiter
    )

    quota_enforcement_engine = create_quota_enforcement_engine()

    quota_analytics_engine = create_quota_analytics_engine()

    api_contracts_health_engine = create_api_contracts_health_engine()

    enhanced_quota_management_engine = create_enhanced_quota_management_engine(
        security_integration_engine=security_integration_engine,
        existing_quota_manager=existing_quota_manager,
        rate_limiting_engine=rate_limiting_engine,
        enforcement_engine=quota_enforcement_engine,
        analytics_engine=quota_analytics_engine
    )

    return {
        "quota_management_engine": enhanced_quota_management_engine,
        "rate_limiting_engine": rate_limiting_engine,
        "enforcement_engine": quota_enforcement_engine,
        "analytics_engine": quota_analytics_engine,
        "api_contracts_health_engine": api_contracts_health_engine
    }
