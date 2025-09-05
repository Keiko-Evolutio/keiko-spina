# backend/kei_mcp/discovery/__init__.py
"""KEI-MCP Discovery-System für vollständige Tool/Resource/Prompt-Discovery.

Implementiert automatische Discovery, Katalogisierung und Integration
von MCP-Server-Komponenten mit dem KEI-Agent-System.
"""

from __future__ import annotations

from kei_logging import get_logger

# MCP Integration
from .mcp_integration import (
    DiscoveryMetrics,
    IntegrationConfig,
    IntegrationStatus,
    MCPIntegrationEngine,
    ServerHealthMetrics,
    ServerHealthStatus,
    mcp_integration_engine,
)

# Prompt Discovery
from .prompt_discovery import (
    DiscoveredPrompt,
    PromptAnalyzer,
    PromptCategory,
    PromptComplexity,
    PromptDiscoveryEngine,
    PromptMetadata,
    PromptParameter,
    PromptStatus,
    PromptUsageStats,
    prompt_discovery_engine,
)

# Resource Discovery
from .resource_discovery import (
    DiscoveredResource,
    ResourceAccessController,
    ResourceAccessLevel,
    ResourceCache,
    ResourceDiscoveryEngine,
    ResourceMetadata,
    ResourceStatus,
    ResourceType,
    resource_discovery_engine,
)

# Tool Discovery
from .tool_discovery import (
    DiscoveredTool,
    ToolAvailabilityStatus,
    ToolCategory,
    ToolDiscoveryEngine,
    ToolMetadata,
    ToolSchema,
    tool_discovery_engine,
)

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "DiscoveredPrompt",
    "DiscoveredResource",
    "DiscoveredTool",
    "DiscoveryMetrics",
    "IntegrationConfig",
    "IntegrationStatus",
    # MCP Integration
    "MCPIntegrationEngine",
    "PromptAnalyzer",
    "PromptCategory",
    "PromptComplexity",
    # Prompt Discovery
    "PromptDiscoveryEngine",
    "PromptMetadata",
    "PromptParameter",
    "PromptStatus",
    "PromptUsageStats",
    "ResourceAccessController",
    "ResourceAccessLevel",
    "ResourceCache",
    # Resource Discovery
    "ResourceDiscoveryEngine",
    "ResourceMetadata",
    "ResourceStatus",
    "ResourceType",
    "ServerHealthMetrics",
    "ServerHealthStatus",
    "ToolAvailabilityStatus",
    "ToolCategory",
    # Tool Discovery
    "ToolDiscoveryEngine",
    "ToolMetadata",
    "ToolSchema",
    "mcp_integration_engine",
    "prompt_discovery_engine",
    "resource_discovery_engine",
    "tool_discovery_engine",
]

# Discovery-System Status
def get_discovery_system_status() -> dict:
    """Gibt Status des Discovery-Systems zurück."""
    return {
        "package": "kei_mcp.discovery",
        "version": "1.0.0",
        "components": {
            "tool_discovery": True,
            "resource_discovery": True,
            "prompt_discovery": True,
            "mcp_integration": True,
        },
        "features": {
            "automatic_discovery": True,
            "schema_validation": True,
            "parameter_mapping": True,
            "availability_checking": True,
            "resource_caching": True,
            "access_control": True,
            "prompt_categorization": True,
            "usage_tracking": True,
            "health_monitoring": True,
            "failover_support": True,
        },
        "engines": {
            "tool_discovery_engine": tool_discovery_engine is not None,
            "resource_discovery_engine": resource_discovery_engine is not None,
            "prompt_discovery_engine": prompt_discovery_engine is not None,
            "mcp_integration_engine": mcp_integration_engine is not None,
        }
    }

logger.info(f"KEI-MCP Discovery-System geladen - Status: {get_discovery_system_status()}")
