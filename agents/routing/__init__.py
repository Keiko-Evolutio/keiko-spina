"""Agent-Routing-Modul.

Capabilities-basierte Agent-Auswahl mit Policy-Integration.
"""

from __future__ import annotations

from .conditional_agent_router import AgentRouter, agent_router, route_to_best_agent
from .constants import (
    AGENT_FIELD_CAPABILITIES,
    AGENT_FIELD_DESCRIPTION,
    AGENT_FIELD_ID,
    AGENT_FIELD_NAME,
    DEFAULT_SCORE_WEIGHT,
    LOG_EVENT_AGENT_SELECTED,
    LOG_EVENT_NO_AGENT_FOUND,
    LOG_EVENT_POLICY_DENIED,
    MIN_SCORE_THRESHOLD,
)
from .utils import calculate_intersection_score, normalize_string_list, validate_agent_structure

__version__ = "2.0.0"

__all__ = [
    # Konstanten
    "AGENT_FIELD_CAPABILITIES",
    "AGENT_FIELD_DESCRIPTION",
    "AGENT_FIELD_ID",
    "AGENT_FIELD_NAME",
    "DEFAULT_SCORE_WEIGHT",
    "LOG_EVENT_AGENT_SELECTED",
    "LOG_EVENT_NO_AGENT_FOUND",
    "LOG_EVENT_POLICY_DENIED",
    "MIN_SCORE_THRESHOLD",
    # Hauptfunktionen und Klassen
    "route_to_best_agent",
    "AgentRouter",
    "agent_router",
    # Utility-Funktionen
    "normalize_string_list",
    "calculate_intersection_score",
    "validate_agent_structure",
]
