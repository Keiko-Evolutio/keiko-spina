# backend/agents/metadata/__init__.py
"""Agent Metadata Package für Multi-Agent Systems.

Metadata-Management für Agent-Architekturen.

Hauptkomponenten:
- AgentMetadata: Core Metadata-Model mit MCP-Integration
- AgentMetadataService: Service für Factory und Management
- create_metadata: Generische Funktion für alle Frameworks

Unterstützte Frameworks:
- Azure AI Foundry (Connected Agents)
- Semantic Kernel (Plugin-basiert)
- AutoGen (Multi-Agent Conversations)
- Custom MCP (Direktintegration)
"""

from __future__ import annotations

# Package-Metadaten
__version__ = "0.0.1"
__author__ = "Development Team"
__mcp_spec_version__ = "2025-06-18"

from kei_logging import get_logger

logger = get_logger(__name__)

# Core Imports
try:
    from .agent_metadata import (
        AgentMetadata,
        AuthMethod,
        CapabilityStatus,
        FrameworkType,
        MCPCapabilityDescriptor,
        MCPServerDescriptor,
        MCPServerEndpoint,
        MCPSpecVersion,
        TransportType,
    )

    logger.info("Agent Metadata Core geladen")
except ImportError as e:
    logger.error(f"Agent Metadata Core nicht verfügbar: {e}")
    raise

# Service Imports
try:
    from .service import (
        AgentMetadataService,
        create_metadata,
        metadata_service,
    )

    logger.info("Agent Metadata Service geladen")
except ImportError as e:
    logger.error(f"Agent Metadata Service nicht verfügbar: {e}")
    raise

# Public API
__all__ = [
    # Core Model
    "AgentMetadata",
    "MCPServerDescriptor",
    "MCPCapabilityDescriptor",
    "MCPServerEndpoint",
    # Enums
    "MCPSpecVersion",
    "FrameworkType",
    "TransportType",
    "AuthMethod",
    "CapabilityStatus",
    # Service
    "AgentMetadataService",
    "metadata_service",
    "create_metadata",
]
