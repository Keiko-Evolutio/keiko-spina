"""Agent Protocols Package.

Enterprise Multi-Agent System Interfaces mit MCP-Unterstützung.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .._compat import (
    UnavailableClass as _UnavailableComponent,
)
from .._compat import (
    unavailable_function_factory as _unavailable_function_factory,
)

logger = get_logger(__name__)

# Konstanten
PROTOCOL_VERSION = "0.0.1"
MCP_SPEC_VERSION = "2025-03-26"
A2A_PROTOCOL_VERSION = "v1"
DEFAULT_AGENT_LIFECYCLE_TIMEOUT = 300.0
DEFAULT_MCP_SESSION_TIMEOUT = 60.0


_unavailable_function = _unavailable_function_factory("Agent Protocol Function")


# Core Protocol Imports
try:
    from .core import (
        A2AInteropProtocol,
        AgentCapabilityType,
        AgentExecutionContext,
        AgentLifecycleState,
        AgentOperationResult,
        BaseAgentProtocol,
        ConnectedAgentProtocol,
        MCPCapability,
        MCPClientProtocol,
        MCPPrimitiveType,
        MCPTransportType,
        agent_execution_session,
        create_agent_context,
        validate_mcp_capability,
    )

    DEFAULT_MCP_TRANSPORT = MCPTransportType.HTTP_SSE
    logger.debug("Base Agent Protocol geladen")

except ImportError as e:
    logger.error(f"Base Agent Protocol nicht verfügbar: {e}")

    # Fallbacks
    BaseAgentProtocol = _UnavailableComponent
    MCPClientProtocol = _UnavailableComponent
    ConnectedAgentProtocol = _UnavailableComponent
    A2AInteropProtocol = _UnavailableComponent
    MCPCapability = _UnavailableComponent
    AgentExecutionContext = _UnavailableComponent
    AgentOperationResult = _UnavailableComponent
    MCPTransportType = _UnavailableComponent
    MCPPrimitiveType = _UnavailableComponent
    AgentLifecycleState = _UnavailableComponent
    AgentCapabilityType = _UnavailableComponent
    agent_execution_session = _unavailable_function
    validate_mcp_capability = _unavailable_function
    create_agent_context = _unavailable_function
    DEFAULT_MCP_TRANSPORT = None

# Semantic Kernel nicht verfügbar
SemanticKernelAgentProtocol = _UnavailableComponent
SKAgentWrapper = _UnavailableComponent
semantic_kernel_session = _unavailable_function

try:
    from .foundry_protocols import (
        DeepResearchProtocol,
        FoundryProtocolFactory,
        foundry_agent_session,
    )
    logger.debug("Azure Foundry Protocols geladen")
except ImportError:
    FoundryProtocolFactory = _UnavailableComponent
    DeepResearchProtocol = _UnavailableComponent
    foundry_agent_session = _unavailable_function

# A2A Bus Interop nicht verfügbar
BusA2AInterop = _UnavailableComponent


# Protocol Factory
def create_base_agent_protocol(
    protocol_type: str = "default", config: dict[str, Any] | None = None
) -> BaseAgentProtocol:
    """Erstellt eine konkrete `BaseAgentProtocol`-Instanz.

    Args:
        protocol_type: Protokolltyp ("langchain", "deep_research", "default")
        config: Optionale Konfiguration für das Zielprotokoll

    Returns:
        Instanz eines konkreten `BaseAgentProtocol`

    Raises:
        ImportError: Wenn das gewünschte Protokoll nicht verfügbar ist
        ValueError: Bei unbekanntem Protokolltyp
    """
    protocol = (protocol_type or "default").strip().lower()

    if protocol in {"default", "langchain"}:
        return _create_langchain_protocol()

    if protocol in {"deep_research", "foundry_deep_research"}:
        return _create_foundry_protocol(config or {})

    raise ValueError(f"Unbekannter Protokolltyp: {protocol_type}")


def _create_langchain_protocol() -> BaseAgentProtocol:
    """Erstellt LangChain Protocol-Instanz."""
    try:
        from .langchain_agent_protocol import LangChainAgentProtocol
        return LangChainAgentProtocol()
    except Exception as protocol_error:
        logger.error(f"LangChainAgentProtocol nicht verfügbar: {protocol_error}")
        raise ImportError("LangChainAgentProtocol nicht verfügbar") from protocol_error


def _create_foundry_protocol(config: dict[str, Any]) -> BaseAgentProtocol:
    """Erstellt Foundry Protocol-Instanz."""
    if FoundryProtocolFactory is _UnavailableComponent:
        raise ImportError("Foundry Protocols nicht verfügbar")

    try:
        return FoundryProtocolFactory.create_protocol("deep_research", **config)  # type: ignore[arg-type]
    except Exception as foundry_error:
        logger.error(f"Erstellung des DeepResearchProtocol fehlgeschlagen: {foundry_error}")
        raise


def get_available_protocols() -> dict[str, bool]:
    """Übersicht über verfügbare Protocol-Implementierungen."""
    return {
        "base_agent_protocol": BaseAgentProtocol != _UnavailableComponent,
        "semantic_kernel": SemanticKernelAgentProtocol != _UnavailableComponent,
        "azure_foundry": DeepResearchProtocol != _UnavailableComponent,
    }


def validate_protocol_environment() -> list[str]:
    """Validiert Protocol-Environment."""
    warnings_list = []
    if BaseAgentProtocol == _UnavailableComponent:
        warnings_list.append("Base Agent Protocol nicht verfügbar")
    return warnings_list


# Exports
__all__ = [
    "A2A_PROTOCOL_VERSION",
    "DEFAULT_AGENT_LIFECYCLE_TIMEOUT",
    "DEFAULT_MCP_SESSION_TIMEOUT",
    "DEFAULT_MCP_TRANSPORT",
    "MCP_SPEC_VERSION",
    "PROTOCOL_VERSION",
    "A2AInteropProtocol",
    "AgentCapabilityType",
    "AgentExecutionContext",
    "AgentLifecycleState",
    "AgentOperationResult",
    "BaseAgentProtocol",
    "BusA2AInterop",
    "ConnectedAgentProtocol",
    "DeepResearchProtocol",
    "FoundryProtocolFactory",
    "MCPCapability",
    "MCPClientProtocol",
    "MCPPrimitiveType",
    "MCPTransportType",
    "SKAgentWrapper",
    "SemanticKernelAgentProtocol",
    "agent_execution_session",
    "create_agent_context",
    "create_base_agent_protocol",
    "foundry_agent_session",
    "get_available_protocols",
    "semantic_kernel_session",
    "validate_mcp_capability",
    "validate_protocol_environment",
]

# Environment Validation beim Import
_validation_warnings = validate_protocol_environment()
for warning in _validation_warnings:
    logger.warning(warning)

# Package-Metadaten
__version__ = "0.0.1"
__author__ = "Keiko Development Team"
