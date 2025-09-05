# backend/agents/__init__.py
"""Konsolidiertes Agents Package - Enterprise Multi-Agent Framework.
==================================================================

Dieses Package konsolidiert alle Agent-Funktionalitäten aus:
- kei_agents/ (Enterprise-Framework mit Security, Monitoring, Resilience)
- agents/ (Spezifische Implementierungen, Tools, Workflows)

Stellt eine einheitliche API bereit für:
- Core Agent Framework (BaseAgent, ComponentManager)
- Common Operations (Adapter-System, Multi-Agent-Sessions)
- Capabilities (Agent-Fähigkeiten-Management)
- Custom Agents (Native Python Implementierungen)
- Memory (Cosmos DB Integration)
- Workflows (LangGraph Integration)
- Tools (MCP Bridge, Utilities)
"""

from __future__ import annotations

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

# Core Framework
try:
    from .core import (
        AgentConfig,
        AgentMetrics,
        BaseAgent,
        ComponentManager,
        get_component_manager,
    )
    _CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core Framework nicht verfügbar: {e}")
    _CORE_AVAILABLE = False

# Metadata
try:
    from .metadata import AgentMetadata
    _METADATA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Metadata nicht verfügbar: {e}")
    _METADATA_AVAILABLE = False

# Common Operations
try:
    from .common import (
        clear_adapters,
        create_agent,
        execute_agent_task,
        find_best_agent_for_task,
        get_adapter,
        get_agents,
        get_all_agents,
        get_foundry_agents,
        initialize_adapters,
        multi_agent_session,
    )
    _COMMON_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Common Operations nicht verfügbar: {e}")
    _COMMON_AVAILABLE = False

# Capabilities
try:
    from .capabilities import (
        AgentCapability,
        AgentMatch,
        CapabilityManager,
        assign_capability,
        find_agents_with_capability,
        match_capability,
    )
    _CAPABILITIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Capabilities nicht verfügbar: {e}")
    _CAPABILITIES_AVAILABLE = False

# Custom Agents
try:
    from .custom import (
        ImageGeneratorAgent,
        ImageTask,
    )
    _CUSTOM_AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Custom Agents nicht verfügbar: {e}")
    _CUSTOM_AGENTS_AVAILABLE = False

# Constants
try:
    from .constants import (
        AgentFramework,
        AgentStatus,
        AgentType,
        DefaultValues,
        TimeoutConstants,
    )
    _CONSTANTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Constants nicht verfügbar: {e}")
    _CONSTANTS_AVAILABLE = False

# Orchestrator Tools
try:
    from .orchestrator import get_orchestrator_tools
    _ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Orchestrator Tools nicht verfügbar: {e}")
    _ORCHESTRATOR_AVAILABLE = False

# Version und Metadaten
__version__ = "2.0.0"
__author__ = "Keiko Agents Team"
__description__ = "Konsolidiertes Enterprise Multi-Agent Framework"

# Conditional Exports basierend auf Verfügbarkeit
__all__ = ["__author__", "__description__", "__version__"]

if _CORE_AVAILABLE:
    __all__.extend([
        "AgentConfig",
        "AgentMetrics",
        "BaseAgent",
        "ComponentManager",
        "get_component_manager",
    ])

if _METADATA_AVAILABLE:
    __all__.extend([
        "AgentMetadata",
    ])

if _COMMON_AVAILABLE:
    __all__.extend([
        "clear_adapters",
        "create_agent",
        "execute_agent_task",
        "find_best_agent_for_task",
        "get_adapter",
        "get_agents",
        "get_all_agents",
        "get_foundry_agents",
        "initialize_adapters",
        "multi_agent_session",
    ])

if _CAPABILITIES_AVAILABLE:
    __all__.extend([
        "AgentCapability",
        "AgentMatch",
        "CapabilityManager",
        "assign_capability",
        "find_agents_with_capability",
        "match_capability",
    ])

if _CUSTOM_AGENTS_AVAILABLE:
    __all__.extend([
        "ImageGeneratorAgent",
        "ImageTask",
    ])

if _CONSTANTS_AVAILABLE:
    __all__.extend([
        "AgentFramework",
        "AgentStatus",
        "AgentType",
        "DefaultValues",
        "TimeoutConstants",
    ])

if _ORCHESTRATOR_AVAILABLE:
    __all__.extend([
        "get_orchestrator_tools",
    ])

# System Status
def get_system_status() -> dict[str, bool]:
    """Gibt den Status aller Subsysteme zurück."""
    return {
        "core_functional": _CORE_AVAILABLE,
        "common_available": _COMMON_AVAILABLE,
        "capabilities_available": _CAPABILITIES_AVAILABLE,
        "custom_agents_available": _CUSTOM_AGENTS_AVAILABLE,
        "constants_available": _CONSTANTS_AVAILABLE,
    }

# Package-Level Status
status = get_system_status()
functional_modules = sum(status.values())
total_modules = len(status)

logger.info(f"Agents Package geladen - {functional_modules}/{total_modules} Module verfügbar")

if not status["core_functional"]:
    logger.critical("KRITISCH: Core Agent-Funktionalität nicht verfügbar!")
elif functional_modules < total_modules:
    logger.warning(f"WARNUNG: {total_modules - functional_modules} Module nicht verfügbar")
else:
    logger.info("Alle Agent-Module erfolgreich geladen")
