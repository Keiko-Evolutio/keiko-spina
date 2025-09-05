"""Orchestrator Package für Agent-Koordination.

Dieses Modul stellt die Hauptfunktionalitäten für Agent-Orchestrierung bereit:
- Agent Discovery und Selection
- Task Delegation und Monitoring
- Tool-Konfiguration für Orchestrator-Operationen
"""

from .agent_operations import (
    delegate_to_agent_implementation,
    delegate_to_best_agent_implementation,
    discover_agents_implementation,
    get_agent_details_implementation,
    monitor_execution_implementation,
)
from .orchestrator_tools import get_orchestrator_tools

__all__ = [
    "delegate_to_agent_implementation",
    "delegate_to_best_agent_implementation",
    "discover_agents_implementation",
    "get_agent_details_implementation",
    "get_orchestrator_tools",
    "monitor_execution_implementation",
]

__version__ = "0.1.0"
