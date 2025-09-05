"""Core Protocol Components."""

from __future__ import annotations

from .context import agent_execution_session, create_agent_context
from .dataclasses import (
    AgentExecutionContext,
    AgentOperationResult,
    MCPCapability,
)
from .enums import (
    AgentCapabilityType,
    AgentLifecycleState,
    MCPCapabilityType,
    MCPPrimitiveType,
    MCPTransportType,
)
from .interfaces import (
    A2AInteropProtocol,
    BaseAgentProtocol,
    ConnectedAgentProtocol,
    MCPClientProtocol,
)
from .utils import validate_mcp_capability

__all__ = [
    # Enums
    "AgentCapabilityType",
    "AgentLifecycleState",
    "MCPCapabilityType",
    "MCPPrimitiveType",
    "MCPTransportType",
    # Dataclasses
    "AgentExecutionContext",
    "AgentOperationResult",
    "MCPCapability",
    # Interfaces
    "A2AInteropProtocol",
    "BaseAgentProtocol",
    "ConnectedAgentProtocol",
    "MCPClientProtocol",
    # Context & Utils
    "agent_execution_session",
    "create_agent_context",
    "validate_mcp_capability",
]
