# backend/agents/capabilities/__init__.py
"""Enhanced Capabilities System.

Dieses Paket implementiert das vollständige Capabilities-System mit
Health/Readiness-Checks, Versionierung und kategorie-spezifischen
Capability-Typen.
"""

from __future__ import annotations

from kei_logging import get_logger

from ..metadata.agent_metadata import (
    HealthStatus,
    ReadinessStatus,
)
from .capability_manager import (
    CapabilityManager,
    CapabilityValidationError,
)
from .capability_matching import (
    AgentMatch,
    CapabilityMatcher,
    assign_capability,
    find_agents_with_capability,
    get_capability_matcher,
    match_capability,
)
from .capability_utils import (
    CapabilityConstants,
    CapabilityFactory,
    CapabilityHelper,
    CapabilityMetrics,
    CapabilityValidator,
    VersionCompatibility,
    VersionInfo,
)
from .enhanced_capabilities import (
    EnhancedCapability,
)
from .health_interfaces import (
    BaseHealthChecker as HealthChecker,
)
from .health_interfaces import (
    BaseReadinessChecker as ReadinessChecker,
)
from .health_interfaces import (
    DefaultHealthChecker,
    DefaultReadinessChecker,
    HealthCheckConstants,
    HealthCheckResult,
    ReadinessCheckResult,
)
from .specialized_capabilities import (
    DomainEntity,
    DomainRelationship,
    DomainsCapability,
    DomainType,
    PoliciesCapability,
    PolicyRule,
    PolicySeverity,
    PolicyType,
    PolicyViolation,
    SkillLevel,
    SkillMetric,
    SkillsCapability,
    ToolEndpoint,
    ToolParameter,
    ToolsCapability,
    ToolType,
)

logger = get_logger(__name__)


_global_capability_manager: CapabilityManager | None = None


def get_capability_manager() -> CapabilityManager:
    """Gibt den globalen CapabilityManager zurück."""
    global _global_capability_manager
    if _global_capability_manager is None:
        _global_capability_manager = CapabilityManager()
    return _global_capability_manager


try:
    from ..registry.dynamic_registry import AgentCapability
except ImportError:

    from dataclasses import dataclass
    from enum import Enum

    class AgentCategory(str, Enum):
        """Agent-Kategorien."""
        GENERAL = "general"
        SPECIALIZED = "specialized"
        CUSTOM = "custom"

    @dataclass
    class AgentCapability:
        """Agent-Fähigkeit mit Metadaten."""
        name: str
        description: str
        category: AgentCategory = AgentCategory.GENERAL
        confidence_score: float = 1.0


__all__ = [
    "AgentCapability",
    "AgentMatch",
    "CapabilityConstants",
    "CapabilityFactory",
    "CapabilityHelper",
    "CapabilityManager",
    "CapabilityMatcher",
    "CapabilityMetrics",
    "CapabilityValidationError",
    "CapabilityValidator",
    "DefaultHealthChecker",
    "DefaultReadinessChecker",
    "DomainEntity",
    "DomainRelationship",
    "DomainType",
    "DomainsCapability",
    "EnhancedCapability",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
    "PoliciesCapability",
    "PolicyRule",
    "PolicySeverity",
    "PolicyType",
    "PolicyViolation",
    "ReadinessCheckResult",
    "ReadinessChecker",
    "ReadinessStatus",
    "SkillLevel",
    "SkillMetric",
    "SkillsCapability",
    "ToolEndpoint",
    "ToolParameter",
    "ToolType",
    "ToolsCapability",
    "VersionCompatibility",
    "VersionInfo",
    "assign_capability",
    "find_agents_with_capability",
    "get_capability_manager",
    "get_capability_matcher",
    "match_capability",
]


def get_capabilities_status() -> dict:
    """Gibt den Status des Capabilities-Systems zurück."""
    return {
        "package": "agents.capabilities",
        "version": "1.0.0",
        "components": {
            "enhanced_capabilities": True,
            "capability_manager": True,
            "specialized_capabilities": True,
        },
        "features": {
            "health_checks": True,
            "readiness_checks": True,
            "versioning": True,
            "category_validation": True,
            "metrics_tracking": True,
        },
    }


logger.info(f"Enhanced Capabilities System geladen - Status: {get_capabilities_status()}")
