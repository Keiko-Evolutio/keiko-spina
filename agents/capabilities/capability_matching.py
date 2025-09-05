# backend/agents/capabilities/capability_matching.py
"""Capability-Matching und Agent-Discovery-Funktionen.

Konsolidiert alle Capability-Matching-Logik aus verschiedenen Modulen
für bessere Wartbarkeit und Konsistenz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentMatch:
    """Ergebnis eines Agent-Capability-Matches."""

    agent_id: str
    capability: str
    confidence_score: float
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validiert Match-Daten."""
        if self.metadata is None:
            self.metadata = {}

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence Score muss zwischen 0.0 und 1.0 liegen: {self.confidence_score}")


class CapabilityMatcher:
    """Zentrale Klasse für Capability-Matching und Agent-Discovery."""

    def __init__(self) -> None:
        """Initialisiert Capability-Matcher."""
        self._agent_capabilities: dict[str, set[str]] = {}
        self._capability_agents: dict[str, set[str]] = {}
        self._semantic_mappings: dict[str, list[str]] = {}

    def register_agent_capability(self, agent_id: str, capability: str) -> bool:
        """Registriert eine Capability für einen Agent.

        Args:
            agent_id: ID des Agents
            capability: Name der Capability

        Returns:
            True wenn erfolgreich registriert
        """
        try:
            if agent_id not in self._agent_capabilities:
                self._agent_capabilities[agent_id] = set()
            self._agent_capabilities[agent_id].add(capability)

            if capability not in self._capability_agents:
                self._capability_agents[capability] = set()
            self._capability_agents[capability].add(agent_id)

            logger.info(f"Capability '{capability}' für Agent {agent_id} registriert")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Registrieren der Capability: {e}")
            return False

    def unregister_agent_capability(self, agent_id: str, capability: str) -> bool:
        """Entfernt eine Capability von einem Agent.

        Args:
            agent_id: ID des Agents
            capability: Name der Capability

        Returns:
            True wenn erfolgreich entfernt
        """
        try:
            if agent_id in self._agent_capabilities:
                self._agent_capabilities[agent_id].discard(capability)
                if not self._agent_capabilities[agent_id]:
                    del self._agent_capabilities[agent_id]

            if capability in self._capability_agents:
                self._capability_agents[capability].discard(agent_id)
                if not self._capability_agents[capability]:
                    del self._capability_agents[capability]

            logger.info(f"Capability '{capability}' von Agent {agent_id} entfernt")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Entfernen der Capability: {e}")
            return False

    def find_agents_with_capability(self, capability: str) -> list[str]:
        """Findet alle Agents mit einer bestimmten Capability.

        Args:
            capability: Name der gesuchten Capability

        Returns:
            Liste der Agent-IDs
        """
        try:
            direct_matches = list(self._capability_agents.get(capability, set()))

            semantic_matches = []
            for related_capability in self._semantic_mappings.get(capability, []):
                semantic_matches.extend(self._capability_agents.get(related_capability, set()))

            all_matches = list(set(direct_matches + semantic_matches))

            logger.debug(f"Gefunden {len(all_matches)} Agents mit Capability '{capability}'")
            return all_matches

        except Exception as e:
            logger.error(f"Fehler bei Agent-Suche: {e}")
            return []

    def match_capability_requirement(self, requirement: str) -> list[AgentMatch]:
        """Matched Capability-Anforderung mit verfügbaren Agents.

        Args:
            requirement: Capability-Anforderung

        Returns:
            Liste von AgentMatch-Objekten, sortiert nach Confidence Score
        """
        try:
            matches = []

            agent_ids = self.find_agents_with_capability(requirement)

            for agent_id in agent_ids:
                confidence = self._calculate_confidence_score(agent_id, requirement)

                match = AgentMatch(
                    agent_id=agent_id,
                    capability=requirement,
                    confidence_score=confidence,
                    metadata={
                        "match_type": "direct" if requirement in self._agent_capabilities.get(agent_id, set()) else "semantic",
                        "agent_capabilities": list(self._agent_capabilities.get(agent_id, set()))
                    }
                )
                matches.append(match)

            matches.sort(key=lambda x: x.confidence_score, reverse=True)

            logger.debug(f"Gefunden {len(matches)} Matches für Requirement '{requirement}'")
            return matches

        except Exception as e:
            logger.error(f"Fehler beim Capability-Matching: {e}")
            return []

    def assign_capability_to_agent(self, agent_id: str, capability: str) -> bool:
        """Weist einem Agent eine Capability zu.

        Args:
            agent_id: ID des Agents
            capability: Name der Capability

        Returns:
            True wenn erfolgreich zugewiesen
        """
        return self.register_agent_capability(agent_id, capability)

    def get_agent_capabilities(self, agent_id: str) -> list[str]:
        """Gibt alle Capabilities eines Agents zurück.

        Args:
            agent_id: ID des Agents

        Returns:
            Liste der Capability-Namen
        """
        return list(self._agent_capabilities.get(agent_id, set()))

    def get_capability_agents(self, capability: str) -> list[str]:
        """Gibt alle Agents mit einer Capability zurück.

        Args:
            capability: Name der Capability

        Returns:
            Liste der Agent-IDs
        """
        return list(self._capability_agents.get(capability, set()))

    def add_semantic_mapping(self, capability: str, related_capabilities: list[str]) -> None:
        """Fügt semantische Mappings für Capabilities hinzu.

        Args:
            capability: Haupt-Capability
            related_capabilities: Liste verwandter Capabilities
        """
        self._semantic_mappings[capability] = related_capabilities
        logger.debug(f"Semantische Mappings für '{capability}' hinzugefügt: {related_capabilities}")

    def _calculate_confidence_score(self, agent_id: str, capability: str) -> float:
        """Berechnet Confidence Score für Agent-Capability-Match.

        Args:
            agent_id: ID des Agents
            capability: Name der Capability

        Returns:
            Confidence Score zwischen 0.0 und 1.0
        """
        agent_capabilities = self._agent_capabilities.get(agent_id, set())

        if capability in agent_capabilities:
            return 1.0

        for related_capability in self._semantic_mappings.get(capability, []):
            if related_capability in agent_capabilities:
                return 0.7

        return 0.0

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Statistiken über registrierte Capabilities zurück.

        Returns:
            Dictionary mit Statistiken
        """
        return {
            "total_agents": len(self._agent_capabilities),
            "total_capabilities": len(self._capability_agents),
            "total_mappings": sum(len(caps) for caps in self._agent_capabilities.values()),
            "semantic_mappings": len(self._semantic_mappings),
            "agents_per_capability": {
                cap: len(agents) for cap, agents in self._capability_agents.items()
            },
            "capabilities_per_agent": {
                agent: len(caps) for agent, caps in self._agent_capabilities.items()
            }
        }


_global_capability_matcher: CapabilityMatcher | None = None


def get_capability_matcher() -> CapabilityMatcher:
    """Gibt den globalen Capability-Matcher zurück."""
    global _global_capability_matcher
    if _global_capability_matcher is None:
        _global_capability_matcher = CapabilityMatcher()
    return _global_capability_matcher



def assign_capability(agent_id: str, capability: str) -> bool:
    """Weist einem Agent eine Capability zu."""
    return get_capability_matcher().assign_capability_to_agent(agent_id, capability)


def find_agents_with_capability(capability: str) -> list[str]:
    """Findet Agents mit einer bestimmten Capability."""
    return get_capability_matcher().find_agents_with_capability(capability)


def match_capability(requirement: str) -> list[AgentMatch]:
    """Matched Capability-Anforderungen mit verfügbaren Agents."""
    return get_capability_matcher().match_capability_requirement(requirement)



__all__ = [
    "AgentMatch",
    "CapabilityMatcher",
    "assign_capability",
    "find_agents_with_capability",
    "get_capability_matcher",
    "match_capability",
]
