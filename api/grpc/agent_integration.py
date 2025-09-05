# backend/api/grpc/agent_integration.py
"""Agent-Integration für gRPC API Service.

Implementiert Capability-basiertes Routing, Agent-Discovery und
Fallback-Mechanismen für standardisierte Agent-Operationen.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .models import OperationType

logger = get_logger(__name__)


class AgentIntegrationMixin:
    """Mixin für Agent-Integration in KEI-RPC Service."""

    def __init__(self, *args, **kwargs):
        """Initialisiert das Mixin mit erforderlichen Attributen."""
        super().__init__(*args, **kwargs)
        self._agent_registry = None
        self._capability_manager = None

    async def _get_agent_by_id(self, agent_id: str) -> Any | None:
        """Gibt Agent anhand ID zurück."""
        if not self._agent_registry:
            logger.warning("Agent Registry nicht verfügbar")
            return None

        try:
            # Agent aus Registry abrufen
            agents = self._agent_registry.agents
            return agents.get(agent_id)

        except Exception as e:
            logger.exception(f"Fehler beim Agent-Abruf: {e}")
            return None

    @staticmethod
    def _get_agent_framework(agent: Any) -> str:
        """Ermittelt Framework eines Agents."""
        try:
            # Framework aus Agent-Metadaten extrahieren
            if hasattr(agent, "framework"):
                return agent.framework
            if hasattr(agent, "type"):
                # Framework aus Agent-Typ ableiten
                agent_type = agent.type.lower()
                if "foundry" in agent_type or "azure" in agent_type:
                    return "foundry"
                if "autogen" in agent_type:
                    return "autogen"
                if "semantic" in agent_type:
                    return "semantic_kernel"

            # Standard-Framework
            return "foundry"

        except Exception as e:
            logger.warning(f"Framework-Ermittlung fehlgeschlagen: {e}")
            return "foundry"

    @staticmethod
    def _get_default_capabilities_for_operation(operation_type: OperationType) -> list[str]:
        """Gibt Standard-Capabilities für Operation zurück."""
        capability_mapping = {
            OperationType.PLAN: ["planning", "reasoning", "task_decomposition", "goal_setting"],
            OperationType.ACT: [
                "action_execution",
                "tool_usage",
                "function_calling",
                "task_execution",
            ],
            OperationType.OBSERVE: [
                "observation",
                "monitoring",
                "data_analysis",
                "pattern_recognition",
            ],
            OperationType.EXPLAIN: [
                "explanation",
                "reasoning",
                "knowledge_synthesis",
                "communication",
            ],
        }

        return capability_mapping.get(operation_type, ["general"])

    async def _search_agents_by_capabilities(self, capabilities: list[str]) -> list[Any]:
        """Sucht Agents basierend auf Capabilities."""
        if not self._agent_registry:
            logger.warning("Agent Registry nicht verfügbar")
            return []

        try:
            # Agents mit passenden Capabilities suchen
            matches = await self._agent_registry.search_agents(capabilities=capabilities, limit=5)

            # Nach Match-Score sortieren
            return sorted(matches, key=lambda x: x.match_score, reverse=True)

        except Exception as e:
            logger.exception(f"Agent-Suche fehlgeschlagen: {e}")
            return []

    async def _get_fallback_agents(self, operation_type: OperationType) -> list[Any]:
        """Gibt Fallback-Agents für Operation zurück."""
        if not self._agent_registry:
            logger.warning("Agent Registry nicht verfügbar")
            return []

        try:
            # Alle verfügbaren Agents abrufen
            all_agents = await self._agent_registry.search_agents(limit=10)

            # Nach Verfügbarkeit und Typ filtern
            fallback_agents = []
            for agent_match in all_agents:
                agent = await self._get_agent_by_id(agent_match.agent_id)
                if agent and self._is_agent_available(agent):
                    # Priorisierung basierend auf Operation
                    priority = self._calculate_fallback_priority(agent, operation_type)
                    if priority > 0:
                        agent_match.match_score = priority
                        fallback_agents.append(agent_match)

            # Nach Priorität sortieren
            return sorted(fallback_agents, key=lambda x: x.match_score, reverse=True)

        except Exception as e:
            logger.exception(f"Fallback-Agent-Suche fehlgeschlagen: {e}")
            return []

    @staticmethod
    def _get_agent_framework_from_match(agent_match: Any) -> str:
        """Ermittelt Framework aus Agent-Match."""
        try:
            # Framework aus Agent-Typ ableiten
            if hasattr(agent_match, "agent_type"):
                agent_type = agent_match.agent_type.lower()
                if "foundry" in agent_type:
                    return "foundry"
                if "autogen" in agent_type:
                    return "autogen"
                if "semantic" in agent_type:
                    return "semantic_kernel"

            # Framework aus Metadaten
            if hasattr(agent_match, "metadata") and agent_match.metadata:
                framework = agent_match.metadata.get("framework")
                if framework:
                    return framework

            # Standard-Framework
            return "foundry"

        except Exception as e:
            logger.warning(f"Framework-Ermittlung aus Match fehlgeschlagen: {e}")
            return "foundry"

    def _is_agent_available(self, agent: Any) -> bool:
        """Prüft, ob Agent verfügbar ist."""
        try:
            # Status-Prüfung
            if hasattr(agent, "status"):
                return agent.status.lower() in ["available", "ready", "active"]

            # Capability-Prüfung über Capability Manager
            if self._capability_manager and hasattr(agent, "id"):
                # Prüfe, ob Agent-Capabilities verfügbar sind
                agent_capabilities = AgentIntegrationMixin._get_agent_capabilities(agent)
                for cap_id in agent_capabilities:
                    capability = self._capability_manager.get_capability(cap_id)
                    if capability and capability.is_available():
                        return True

            # Standard: verfügbar
            return True

        except Exception as e:
            logger.warning(f"Agent-Verfügbarkeitsprüfung fehlgeschlagen: {e}")
            return True

    @staticmethod
    def _get_agent_capabilities(agent: Any) -> list[str]:
        """Extrahiert Capabilities eines Agents."""
        try:
            capabilities = []

            # Capabilities aus Agent-Attributen
            if hasattr(agent, "capabilities"):
                if isinstance(agent.capabilities, list):
                    capabilities.extend(agent.capabilities)
                elif isinstance(agent.capabilities, str):
                    capabilities.append(agent.capabilities)

            # Capabilities aus Agent-Typ ableiten
            if hasattr(agent, "type"):
                agent_type = agent.type.lower()
                if "research" in agent_type:
                    capabilities.extend(["research", "web_search", "data_analysis"])
                elif "planning" in agent_type:
                    capabilities.extend(["planning", "reasoning", "task_decomposition"])
                elif "execution" in agent_type:
                    capabilities.extend(["action_execution", "tool_usage"])
                elif "analysis" in agent_type:
                    capabilities.extend(["observation", "data_analysis", "pattern_recognition"])

            # Standard-Capabilities
            if not capabilities:
                capabilities = ["general"]

            return capabilities

        except Exception as e:
            logger.warning(f"Capability-Extraktion fehlgeschlagen: {e}")
            return ["general"]

    def _calculate_fallback_priority(self, agent: Any, operation_type: OperationType) -> float:
        """Berechnet Fallback-Priorität für Agent."""
        try:
            priority = 0.0

            # Basis-Priorität basierend auf Agent-Typ
            if hasattr(agent, "type"):
                agent_type = agent.type.lower()

                # Operation-spezifische Prioritäten
                if operation_type == OperationType.PLAN:
                    if any(
                        keyword in agent_type for keyword in ["planning", "strategy", "reasoning"]
                    ):
                        priority += 0.8
                    elif any(keyword in agent_type for keyword in ["general", "assistant"]):
                        priority += 0.5

                elif operation_type == OperationType.ACT:
                    if any(keyword in agent_type for keyword in ["execution", "action", "tool"]):
                        priority += 0.8
                    elif any(keyword in agent_type for keyword in ["general", "assistant"]):
                        priority += 0.6

                elif operation_type == OperationType.OBSERVE:
                    if any(
                        keyword in agent_type
                        for keyword in ["analysis", "monitoring", "observation"]
                    ):
                        priority += 0.8
                    elif any(keyword in agent_type for keyword in ["research", "data"]):
                        priority += 0.7
                    elif any(keyword in agent_type for keyword in ["general", "assistant"]):
                        priority += 0.4

                elif operation_type == OperationType.EXPLAIN:
                    if any(
                        keyword in agent_type
                        for keyword in ["explanation", "teaching", "communication"]
                    ):
                        priority += 0.8
                    elif any(keyword in agent_type for keyword in ["reasoning", "knowledge"]):
                        priority += 0.7
                    elif any(keyword in agent_type for keyword in ["general", "assistant"]):
                        priority += 0.6

            # Verfügbarkeits-Bonus
            if self._is_agent_available(agent):
                priority += 0.2

            # Framework-Bonus (bevorzuge stabile Frameworks)
            framework = AgentIntegrationMixin._get_agent_framework(agent)
            if framework == "foundry":
                priority += 0.1

            return min(priority, 1.0)  # Maximal 1.0

        except Exception as e:
            logger.warning(f"Prioritäts-Berechnung fehlgeschlagen: {e}")
            return 0.1  # Minimale Priorität

    def _match_capabilities_to_operation(
        self, agent_capabilities: list[str], required_capabilities: list[str]
    ) -> float:
        """Berechnet Capability-Match-Score."""
        if not required_capabilities:
            return 0.5  # Neutral bei fehlenden Anforderungen

        if not agent_capabilities:
            return 0.0  # Keine Capabilities verfügbar

        # Exakte Matches
        exact_matches = len(set(agent_capabilities) & set(required_capabilities))

        # Semantische Matches (vereinfacht)
        semantic_matches = 0
        for req_cap in required_capabilities:
            for agent_cap in agent_capabilities:
                if AgentIntegrationMixin._are_capabilities_related(req_cap, agent_cap):
                    semantic_matches += 0.5
                    break

        total_score = (exact_matches + semantic_matches) / len(required_capabilities)
        return min(total_score, 1.0)

    @staticmethod
    def _are_capabilities_related(cap1: str, cap2: str) -> bool:
        """Prüft, ob zwei Capabilities semantisch verwandt sind."""
        # Vereinfachte semantische Verwandtschaft
        related_groups = [
            {"planning", "reasoning", "strategy", "goal_setting"},
            {"action_execution", "tool_usage", "function_calling", "task_execution"},
            {"observation", "monitoring", "analysis", "pattern_recognition"},
            {"explanation", "reasoning", "communication", "knowledge_synthesis"},
            {"research", "web_search", "data_analysis", "information_retrieval"},
        ]

        cap1_lower = cap1.lower()
        cap2_lower = cap2.lower()

        return any(cap1_lower in group and cap2_lower in group for group in related_groups)
