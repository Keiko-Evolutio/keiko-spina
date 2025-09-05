"""Konstanten für das Agent-Routing-Modul."""

from __future__ import annotations

from typing import Final

# Agent-Datenstruktur Felder
AGENT_FIELD_CAPABILITIES: Final[str] = "capabilities"
AGENT_FIELD_ID: Final[str] = "id"

# Logging Events
LOG_EVENT_AGENT_SELECTED: Final[str] = "agent_route_selected"

# Scoring-Konstanten
MIN_SCORE_THRESHOLD: Final[int] = 0

# API-Kompatibilität
AGENT_FIELD_NAME: Final[str] = "name"
AGENT_FIELD_DESCRIPTION: Final[str] = "description"
LOG_EVENT_NO_AGENT_FOUND: Final[str] = "no_agent_found"
LOG_EVENT_POLICY_DENIED: Final[str] = "agent_policy_denied"
DEFAULT_SCORE_WEIGHT: Final[float] = 1.0

__all__ = [
    "AGENT_FIELD_CAPABILITIES",
    "AGENT_FIELD_DESCRIPTION",
    "AGENT_FIELD_ID",
    "AGENT_FIELD_NAME",
    "DEFAULT_SCORE_WEIGHT",
    "LOG_EVENT_AGENT_SELECTED",
    "LOG_EVENT_NO_AGENT_FOUND",
    "LOG_EVENT_POLICY_DENIED",
    "MIN_SCORE_THRESHOLD",
]
