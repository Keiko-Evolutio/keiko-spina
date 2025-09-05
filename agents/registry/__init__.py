"""Agent Registry Package für dynamische Agent-Discovery."""

from .dynamic_registry import (
    AgentCapability,
    AgentMatch,
    DynamicAgentRegistry,
    dynamic_registry,
)

__all__ = ["AgentCapability", "AgentMatch", "DynamicAgentRegistry", "dynamic_registry"]

__version__ = "0.0.1"
