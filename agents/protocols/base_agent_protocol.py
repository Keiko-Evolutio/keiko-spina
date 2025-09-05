"""BaseAgentProtocol - Enterprise Multi-Agent System Interface.

DEPRECATED: Diese Datei wurde refactored und aufgeteilt.
Verwende stattdessen: backend/agents/protocols/core/
"""

from __future__ import annotations

# Deprecation Warning
import warnings

# Backward Compatibility Re-exports
from .core import (
    A2AInteropProtocol,
    AgentCapabilityType,
    AgentExecutionContext,
    AgentLifecycleState,
    AgentOperationResult,
    BaseAgentProtocol,
    ConnectedAgentProtocol,
    MCPCapability,
    MCPCapabilityType,
    MCPClientProtocol,
    MCPPrimitiveType,
    MCPTransportType,
    agent_execution_session,
    create_agent_context,
    validate_mcp_capability,
)

warnings.warn(
    "base_agent_protocol.py ist deprecated. "
    "Verwende stattdessen: from agents.protocols.core import ...",
    DeprecationWarning,
    stacklevel=2
)


# Exports
__all__ = [
    "A2AInteropProtocol",
    "AgentCapabilityType",
    "AgentExecutionContext",
    "AgentLifecycleState",
    "AgentOperationResult",
    "BaseAgentProtocol",
    "ConnectedAgentProtocol",
    "MCPCapability",
    "MCPCapabilityType",
    "MCPClientProtocol",
    "MCPPrimitiveType",
    "MCPTransportType",
    "agent_execution_session",
    "create_agent_context",
    "validate_mcp_capability",
]
