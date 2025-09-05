"""Protocol Enums."""

from __future__ import annotations

from enum import Enum, auto


class MCPTransportType(Enum):
    """MCP Transport-Mechanismen gemäß Spezifikation 2025-03-26."""

    STDIO = "stdio"
    HTTP_SSE = "http_sse"
    STREAMABLE_HTTP = "streamable_http"


class MCPPrimitiveType(Enum):
    """MCP Primitive-Typen als Kern-Capabilities."""

    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"


class AgentLifecycleState(Enum):
    """Agent-Lifecycle-States."""

    INITIALIZING = auto()
    READY = auto()
    EXECUTING = auto()
    SUSPENDED = auto()
    ERROR = auto()
    TERMINATED = auto()


class AgentCapabilityType(Enum):
    """Agent-Capability-Kategorien."""

    MCP_CLIENT = "mcp_client"
    CONNECTED_AGENT = "connected_agent"
    A2A_INTEROP = "a2a_interop"
    DEEP_RESEARCH = "deep_research"
    CUSTOM_TOOL = "custom_tool"


class MCPCapabilityType(Enum):
    """MCP-Capability-Typen."""

    TOOL_INVOCATION = "tool_invocation"
    RESOURCE_ACCESS = "resource_access"


__all__ = [
    "AgentCapabilityType",
    "AgentLifecycleState",
    "MCPCapabilityType",
    "MCPPrimitiveType",
    "MCPTransportType",
]
