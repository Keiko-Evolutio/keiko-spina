# backend/agents/factory/__init__.py
"""Factory-Modul für Agent-Erstellung und -Management.

Modulare Factory-Implementierung mit:
- Agent-Erstellung für verschiedene Frameworks
- MCP-Client-Integration
- Error-Handling und Monitoring
- Singleton-Pattern für Ressourcen-Management
"""
from __future__ import annotations

# Hauptkomponenten
from .agent_factory import AgentFactory

# Konstanten und Konfiguration
from .constants import (
    DEFAULT_FEATURE_FLAGS,
    DEFAULT_FRAMEWORK,
    SUPPORTED_FRAMEWORKS,
    AgentFramework,
    FactoryState,
)

# Fehlerbehandlung
from .error_handlers import (
    AgentCreationError,
    AgentFactoryErrorHandler,
    MCPClientError,
    error_context,
    retry_on_error,
)
from .factory_api import (
    cleanup_agent,
    cleanup_all_agents,
    create_agent,
    create_agent_with_mcp,
    create_azure_foundry_agent_with_mcp,
    get_factory,
    get_factory_status,
    is_factory_available,
)
from .factory_utils import create_foundry_adapter

# Metriken und Überwachung
from .metrics_manager import MetricsManager, MetricType

# Singleton-Muster
from .singleton_mixin import SingletonMixin

# Kompatibilität für bestehende Tests
from .unified_mcp_client import (
    MCPClientFactory,  # Backward compatibility
    UnifiedMCPClient,
    UnifiedMCPClientFactory,
    get_unified_mcp_client_factory,
)

# Feature-Flags für Tests
AVAILABLE_FEATURES: dict[str, bool] = DEFAULT_FEATURE_FLAGS.copy()


# =============================================================================
# Export für einfachen Import
# =============================================================================

__all__ = [
    "AVAILABLE_FEATURES",
    "DEFAULT_FRAMEWORK",
    "SUPPORTED_FRAMEWORKS",
    "AgentCreationError",
    "AgentFactory",
    "AgentFactoryErrorHandler",
    "AgentFramework",
    "FactoryState",
    "MCPClientError",
    "MCPClientFactory",
    "MetricType",
    "MetricsManager",
    "SingletonMixin",
    "UnifiedMCPClient",
    "UnifiedMCPClientFactory",
    "cleanup_agent",
    "cleanup_all_agents",
    "create_agent",
    "create_agent_with_mcp",
    "create_azure_foundry_agent_with_mcp",
    "create_foundry_adapter",
    "error_context",
    "get_factory",
    "get_factory_status",
    "get_unified_mcp_client_factory",
    "is_factory_available",
    "retry_on_error",
]
