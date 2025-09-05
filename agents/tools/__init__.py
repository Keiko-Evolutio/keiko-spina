"""Tools-Bridge Paket mit konsolidiertem MCP-Support.

Konsolidiert MCP-Funktionalitäten aus kei_mcp/ nach agents/tools/mcp/.
"""

# MCP-Komponenten (migriert aus kei_mcp/)
from .mcp import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerException,
    CircuitState,
    ExternalMCPConfig,
    KEIMCPClient,
    KEIMCPRegistry,
    MCPResourceResult,
    MCPToolDefinition,
    MCPToolResult,
    RegisteredMCPServer,
    SchemaValidationError,
    ValidationResult,
    schema_validator,
)
from .mcp import (
    kei_mcp_registry as mcp_registry,
)
from .mcp_langchain_bridge import BridgeConfig, create_langchain_tool_from_mcp

__all__ = [
    # LangChain Bridge
    "BridgeConfig",
    "create_langchain_tool_from_mcp",

    # MCP Core Components
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerException",
    "CircuitState",
    "ExternalMCPConfig",
    "KEIMCPClient",
    "KEIMCPRegistry",
    "MCPResourceResult",
    "MCPToolDefinition",
    "MCPToolResult",
    "RegisteredMCPServer",
    "SchemaValidationError",
    "ValidationResult",
    "mcp_registry",
    "schema_validator",
]
