"""Conditional Agent Routing basierend auf Capabilities und Policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from kei_logging import get_logger

from .constants import (
    AGENT_FIELD_CAPABILITIES,
    AGENT_FIELD_ID,
    LOG_EVENT_AGENT_SELECTED,
    MIN_SCORE_THRESHOLD,
)
from .utils import calculate_intersection_score, normalize_string_list, validate_agent_structure

if TYPE_CHECKING:
    from collections.abc import Callable

# Type Aliases
AgentDescriptor: TypeAlias = dict[str, Any]
ScoredAgent: TypeAlias = tuple[AgentDescriptor, int]

if TYPE_CHECKING:
    POLICY_FILTER: TypeAlias = Callable[[AgentDescriptor], bool]
else:
    POLICY_FILTER = "Callable[[AgentDescriptor], bool]"

PolicyFilter = POLICY_FILTER

logger = get_logger(__name__)


def route_to_best_agent(
    *,
    agents: list[AgentDescriptor],
    required_capabilities: list[str] | None = None,
    policy_allow: PolicyFilter | None = None,
) -> AgentDescriptor | None:
    """Wählt den besten Agenten anhand Capabilities-basierter Heuristik.

    Args:
        agents: Liste von Agent-Deskriptoren.
        required_capabilities: Liste erforderlicher Capabilities.
        policy_allow: Optionale Policy-Funktion für Filterung.

    Returns:
        Agent-Deskriptor mit höchstem Score oder None.
    """
    if not agents:
        return None

    valid_agents = [agent for agent in agents if validate_agent_structure(agent)]
    if not valid_agents:
        return None

    required_caps = normalize_string_list(required_capabilities)
    scored_agents = _score_agents(valid_agents, required_caps)

    return _select_best_agent(scored_agents, required_caps, policy_allow)


def _normalize_capabilities(capabilities: list[str] | None) -> set[str]:
    """Normalisiert Capabilities zu lowercase Set.

    Args:
        capabilities: Liste von Capability-Strings.

    Returns:
        Set von normalisierten Capabilities.
    """
    return normalize_string_list(capabilities)


def _score_agents(
    agents: list[AgentDescriptor],
    required_caps: set[str]
) -> list[ScoredAgent]:
    """Berechnet Scores für alle Agenten.

    Args:
        agents: Liste von Agent-Deskriptoren.
        required_caps: Set von erforderlichen Capabilities.

    Returns:
        Liste von (agent, score) Tupeln, sortiert nach Score.
    """
    scored = []
    for agent in agents:
        agent_caps = normalize_string_list(agent.get(AGENT_FIELD_CAPABILITIES, []))
        score = calculate_intersection_score(required_caps, agent_caps)
        scored.append((agent, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def _select_best_agent(
    scored_agents: list[ScoredAgent],
    required_caps: set[str],
    policy_allow: PolicyFilter | None,
) -> AgentDescriptor | None:
    """Wählt besten Agent unter Berücksichtigung von Policy und Requirements.

    Args:
        scored_agents: Liste von (agent, score) Tupeln.
        required_caps: Set von erforderlichen Capabilities.
        policy_allow: Optionaler Policy-Filter.

    Returns:
        Bester Agent oder None.
    """
    for agent, score in scored_agents:
        if not _is_agent_valid(agent, score, required_caps, policy_allow):
            continue

        _log_agent_selection(agent, score, required_caps)
        return agent

    return None


def _is_agent_valid(
    agent: AgentDescriptor,
    score: int,
    required_caps: set[str],
    policy_allow: PolicyFilter | None,
) -> bool:
    """Prüft ob Agent alle Validierungskriterien erfüllt.

    Args:
        agent: Agent-Deskriptor.
        score: Berechneter Capability-Score.
        required_caps: Erforderliche Capabilities.
        policy_allow: Optionaler Policy-Filter.

    Returns:
        True wenn Agent gültig ist.
    """
    # Policy-Check
    if policy_allow and not policy_allow(agent):
        return False

    # Capability-Check (nur wenn Requirements vorhanden)
    return not (required_caps and score <= MIN_SCORE_THRESHOLD)


def _log_agent_selection(
    agent: AgentDescriptor,
    score: int,
    required_caps: set[str]
) -> None:
    """Loggt Agent-Auswahl für Debugging.

    Args:
        agent: Gewählter Agent.
        score: Capability-Score.
        required_caps: Erforderliche Capabilities.
    """
    logger.debug({
        "event": LOG_EVENT_AGENT_SELECTED,
        "agent_id": agent.get(AGENT_FIELD_ID),
        "score": score,
        "required": list(required_caps),
    })


class AgentRouter:
    """Enterprise-Grade Agent Router mit Capabilities-basierter Auswahl.

    Bietet eine objektorientierte Schnittstelle für Agent-Routing mit
    Policy-Integration und erweiterten Konfigurationsmöglichkeiten.
    """

    def __init__(self, *, default_policy: PolicyFilter | None = None) -> None:
        """Initialisiert den Agent Router.

        Args:
            default_policy: Standard-Policy für Agent-Filterung.
        """
        self._default_policy = default_policy
        logger.debug("Agent Router initialisiert")

    def route_request(
        self,
        *,
        agents: list[AgentDescriptor],
        required_capabilities: list[str] | None = None,
        policy_allow: PolicyFilter | None = None,
    ) -> AgentDescriptor | None:
        """Routet Request zum besten verfügbaren Agent.

        Args:
            agents: Liste verfügbarer Agent-Deskriptoren.
            required_capabilities: Erforderliche Capabilities.
            policy_allow: Optionale Policy (überschreibt default_policy).

        Returns:
            Agent-Deskriptor mit bestem Match oder None.
        """
        effective_policy = policy_allow or self._default_policy
        return route_to_best_agent(
            agents=agents,
            required_capabilities=required_capabilities,
            policy_allow=effective_policy,
        )


agent_router = AgentRouter()


__all__ = ["AgentRouter", "agent_router", "route_to_best_agent"]
