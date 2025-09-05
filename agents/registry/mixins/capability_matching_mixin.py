# backend/agents/registry/mixins/capability_matching_mixin.py
"""Mixin für Capability-Matching-Funktionalität.

Konsolidiert die Matching-Algorithmen und eliminiert Code-Duplikation.
"""

from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from ..utils.constants import MatchingConstants
from ..utils.helpers import calculate_match_score, extract_capabilities
from ..utils.types import (
    AgentInstance,
    CapabilityList,
    MatchScore,
)

logger = get_logger(__name__)


@dataclass
class AgentMatch:
    """Ergebnis eines Agent-Matches mit Validierung."""

    agent_id: str
    agent_name: str
    match_score: MatchScore
    capabilities: CapabilityList
    agent_type: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Post-Initialisierung mit Validierung."""
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("Agent-ID darf nicht leer sein")

        if not self.agent_name or not self.agent_name.strip():
            raise ValueError("Agent-Name darf nicht leer sein")

        if not (MatchingConstants.MIN_MATCH_SCORE <= self.match_score <= MatchingConstants.MAX_MATCH_SCORE):
            raise ValueError(
                f"match_score muss zwischen {MatchingConstants.MIN_MATCH_SCORE} "
                f"und {MatchingConstants.MAX_MATCH_SCORE} liegen, erhalten: {self.match_score}"
            )

        if not isinstance(self.capabilities, list):
            raise ValueError("capabilities muss eine Liste sein")


class CapabilityMatchingMixin:
    """Mixin für Capability-Matching-Funktionalität.

    Konsolidiert die verschiedenen Matching-Algorithmen und
    eliminiert Code-Duplikation zwischen Registry-Klassen.
    """

    def __init__(self, *args, **kwargs):
        """Initialisiert das Mixin."""
        super().__init__(*args, **kwargs)

        # Matching-Metriken
        self._matching_metrics = {
            "total_matches": 0,
            "successful_matches": 0,
            "capability_matches": 0,
            "text_matches": 0,
            "category_matches": 0,
        }

    def create_agent_matches(
        self,
        agents: dict[str, AgentInstance],
        query: str | None = None,
        capabilities: CapabilityList | None = None,
        category: str | None = None,
        limit: int = MatchingConstants.DEFAULT_MATCH_LIMIT,
    ) -> list[AgentMatch]:
        """Erstellt Agent-Matches basierend auf Kriterien.

        Args:
            agents: Dictionary der verfügbaren Agents
            query: Suchtext
            capabilities: Erforderliche Capabilities
            category: Gewünschte Kategorie
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Sortierte Liste von Agent-Matches
        """
        matches = []
        self._matching_metrics["total_matches"] += 1

        for agent_id, agent in agents.items():
            score = calculate_match_score(agent, query, capabilities, category)

            if score >= MatchingConstants.MIN_VIABLE_SCORE:
                match = self._create_agent_match(agent_id, agent, score)
                matches.append(match)
                self._update_matching_metrics(query, capabilities, category)

        # Sortiere nach Score (höchste zuerst)
        matches.sort(key=lambda x: x.match_score, reverse=True)

        if matches:
            self._matching_metrics["successful_matches"] += 1

        return matches[:limit]

    def _create_agent_match(
        self,
        agent_id: str,
        agent: AgentInstance,
        score: MatchScore,
    ) -> AgentMatch:
        """Erstellt ein AgentMatch-Objekt.

        Args:
            agent_id: Agent-ID
            agent: Agent-Instanz
            score: Match-Score

        Returns:
            AgentMatch-Objekt
        """
        return AgentMatch(
            agent_id=agent_id,
            agent_name=getattr(agent, "name", "Unknown Agent"),
            match_score=score,
            capabilities=extract_capabilities(agent),
            agent_type=type(agent).__name__,
            metadata={
                "last_updated": getattr(self, "_last_refresh", None),
                "agent_status": getattr(agent, "status", "unknown"),
            },
        )

    def _update_matching_metrics(
        self,
        query: str | None,
        capabilities: CapabilityList | None,
        category: str | None,
    ) -> None:
        """Aktualisiert Matching-Metriken.

        Args:
            query: Suchtext
            capabilities: Capabilities
            category: Kategorie
        """
        if query:
            self._matching_metrics["text_matches"] += 1
        if capabilities:
            self._matching_metrics["capability_matches"] += 1
        if category:
            self._matching_metrics["category_matches"] += 1

    def find_agents_by_capability(
        self,
        capability: str,
        agents: dict[str, AgentInstance] | None = None,
    ) -> list[AgentMatch]:
        """Findet Agents mit einer spezifischen Capability.

        Args:
            capability: Gesuchte Capability
            agents: Optional Agent-Dictionary

        Returns:
            Liste von Matches
        """
        if agents is None:
            agents = getattr(self, "_agents", {})

        return self.create_agent_matches(
            agents=agents,
            capabilities=[capability],
            limit=MatchingConstants.DEFAULT_MATCH_LIMIT,
        )

    def find_agents_by_category(
        self,
        category: str,
        agents: dict[str, AgentInstance] | None = None,
    ) -> list[AgentMatch]:
        """Findet Agents einer spezifischen Kategorie.

        Args:
            category: Gesuchte Kategorie
            agents: Optional Agent-Dictionary

        Returns:
            Liste von Matches
        """
        if agents is None:
            agents = getattr(self, "_agents", {})

        return self.create_agent_matches(
            agents=agents,
            category=category,
            limit=MatchingConstants.DEFAULT_MATCH_LIMIT,
        )

    def get_agent_capabilities(
        self,
        agent_id: str,
        agents: dict[str, AgentInstance] | None = None,
    ) -> CapabilityList:
        """Holt Capabilities eines spezifischen Agents.

        Args:
            agent_id: Agent-ID
            agents: Optional Agent-Dictionary

        Returns:
            Liste der Capabilities
        """
        if agents is None:
            agents = getattr(self, "_agents", {})

        agent = agents.get(agent_id)
        if not agent:
            return []

        return extract_capabilities(agent)

    def get_all_capabilities(
        self,
        agents: dict[str, AgentInstance] | None = None,
    ) -> set[str]:
        """Holt alle verfügbaren Capabilities.

        Args:
            agents: Optional Agent-Dictionary

        Returns:
            Set aller Capabilities
        """
        if agents is None:
            agents = getattr(self, "_agents", {})

        all_capabilities: set[str] = set()

        for agent in agents.values():
            capabilities = extract_capabilities(agent)
            all_capabilities.update(capabilities)

        return all_capabilities

    def get_agents_by_type(
        self,
        agent_type: str,
        agents: dict[str, AgentInstance] | None = None,
    ) -> list[AgentInstance]:
        """Holt Agents eines spezifischen Typs.

        Args:
            agent_type: Agent-Typ
            agents: Optional Agent-Dictionary

        Returns:
            Liste von Agents
        """
        if agents is None:
            agents = getattr(self, "_agents", {})

        return [
            agent for agent in agents.values()
            if type(agent).__name__ == agent_type
        ]

    def get_matching_metrics(self) -> dict[str, int]:
        """Gibt Matching-Metriken zurück.

        Returns:
            Metriken-Dictionary
        """
        return self._matching_metrics.copy()

    def reset_matching_metrics(self) -> None:
        """Setzt Matching-Metriken zurück."""
        self._matching_metrics = {
            "total_matches": 0,
            "successful_matches": 0,
            "capability_matches": 0,
            "text_matches": 0,
            "category_matches": 0,
        }
