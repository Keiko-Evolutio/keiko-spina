"""Custom Loader für Prompty-basierte Agents.

Lädt und verwaltet Prompty-basierte Custom Agents mit Enterprise-Caching
und Singleton-Pattern für optimale Performance.
"""

from .custom_loader import (
    CustomAgentLoader,
    get_client_agents,
    get_custom_agent_loader,
    load_custom_agents,
    prompty_to_agent,
)

__all__ = [
    "CustomAgentLoader",
    "get_client_agents",
    "get_custom_agent_loader",
    "load_custom_agents",
    "prompty_to_agent",
]

__version__ = "1.0.0"
__author__ = "Development Team"
