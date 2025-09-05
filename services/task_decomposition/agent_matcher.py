# backend/services/task_decomposition/agent_matcher.py
"""Agent-Capability-Matching für optimale Task-Zuordnung.

Integriert Performance Prediction ML-Pipeline aus TASK 2 für
intelligente Agent-Selection basierend auf Capabilities und Performance.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agents.registry.dynamic_registry import DynamicAgentRegistry
from kei_logging import get_logger, log_orchestrator_step, training_trace
from services.ml.performance_prediction import (
    AgentCharacteristics,
    PerformancePredictor,
    PredictionRequest,
    TaskCharacteristics,
)

from .data_models import AgentMatch, SubtaskDefinition

logger = get_logger(__name__)


class AgentCapabilityMatcher:
    """Agent-Capability-Matching mit ML-basierter Performance-Prediction."""

    def __init__(
        self,
        agent_registry: DynamicAgentRegistry,
        performance_predictor: PerformancePredictor | None = None
    ):
        """Initialisiert Agent Matcher.

        Args:
            agent_registry: Agent Registry für verfügbare Agents
            performance_predictor: ML-basierter Performance Predictor
        """
        self.agent_registry = agent_registry
        self.performance_predictor = performance_predictor

        # Matching-Konfiguration
        self.min_capability_coverage = 0.7  # 70% Capability-Abdeckung erforderlich
        self.max_load_threshold = 0.8  # 80% maximale Agent-Load
        self.specialization_weight = 0.3  # Gewichtung für Spezialisierung
        self.performance_weight = 0.4  # Gewichtung für Performance-Prediction
        self.availability_weight = 0.3  # Gewichtung für Verfügbarkeit

        # Performance-Tracking
        self._matching_count = 0
        self._total_matching_time_ms = 0.0

        logger.info("Agent Capability Matcher initialisiert")

    @training_trace(context={"component": "agent_matcher", "phase": "capability_matching"})
    async def find_best_agents(
        self,
        subtasks: list[SubtaskDefinition],
        max_agents_per_subtask: int = 3
    ) -> dict[str, list[AgentMatch]]:
        """Findet beste Agents für alle Subtasks.

        Args:
            subtasks: Liste von Subtasks
            max_agents_per_subtask: Maximale Anzahl Agent-Kandidaten pro Subtask

        Returns:
            Dictionary: subtask_id -> [AgentMatch]
        """
        start_time = time.time()

        try:
            # Schritt 1: Agent Capability Matching gestartet
            log_orchestrator_step(
                "Starting Agent Capability Matching",
                "agent_call",
                subtask_count=len(subtasks),
                max_agents_per_subtask=max_agents_per_subtask,
                required_capabilities=[st.required_capabilities for st in subtasks]
            )

            # Hole alle verfügbaren Agents
            log_orchestrator_step(
                "Discovering Available Agents",
                "agent_call",
                discovery_method="agent_registry"
            )

            available_agents = await self._get_available_agents()

            if not available_agents:
                log_orchestrator_step(
                    "No Available Agents Found",
                    "agent_call",
                    agent_count=0
                )
                logger.warning("Keine verfügbaren Agents gefunden")
                return {}

            log_orchestrator_step(
                "Available Agents Discovered",
                "agent_call",
                agent_count=len(available_agents),
                agent_types=[agent.get("agent_type", "unknown") for agent in available_agents[:5]]
            )

            # Schritt 2: Parallel Matching für alle Subtasks
            log_orchestrator_step(
                "Starting Parallel Agent Matching",
                "agent_call",
                subtask_count=len(subtasks),
                available_agent_count=len(available_agents),
                matching_strategy="parallel"
            )

            matching_tasks = [
                self._match_agents_for_subtask(subtask, available_agents, max_agents_per_subtask)
                for subtask in subtasks
            ]

            matching_results = await asyncio.gather(*matching_tasks, return_exceptions=True)

            # Ergebnisse zusammenfassen
            agent_assignments = {}
            for i, result in enumerate(matching_results):
                if isinstance(result, Exception):
                    logger.error(f"Agent-Matching für Subtask {subtasks[i].subtask_id} fehlgeschlagen: {result}")
                    agent_assignments[subtasks[i].subtask_id] = []
                else:
                    agent_assignments[subtasks[i].subtask_id] = result

            # Performance-Tracking
            matching_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(matching_time_ms)

            logger.info({
                "event": "agent_matching_completed",
                "subtask_count": len(subtasks),
                "total_matches": sum(len(matches) for matches in agent_assignments.values()),
                "matching_time_ms": matching_time_ms
            })

            return agent_assignments

        except Exception as e:
            logger.error(f"Agent-Matching fehlgeschlagen: {e}")
            return {}

    async def _match_agents_for_subtask(
        self,
        subtask: SubtaskDefinition,
        available_agents: list[dict[str, Any]],
        max_agents: int
    ) -> list[AgentMatch]:
        """Matched Agents für einzelnen Subtask."""
        try:
            agent_matches = []

            for agent_info in available_agents:
                # Capability-Matching
                capability_match = await self._calculate_capability_match(subtask, agent_info)

                if capability_match["coverage"] < self.min_capability_coverage:
                    continue  # Unzureichende Capability-Abdeckung

                # Load-Check
                if agent_info.get("current_load", 0.0) > self.max_load_threshold:
                    continue  # Agent überlastet

                # Performance-Prediction (falls verfügbar)
                performance_prediction = await self._predict_performance(subtask, agent_info)

                # Gesamtscore berechnen
                match_score = self._calculate_match_score(
                    capability_match,
                    performance_prediction,
                    agent_info
                )

                # AgentMatch erstellen
                agent_match = AgentMatch(
                    agent_id=agent_info["agent_id"],
                    agent_type=agent_info.get("agent_type", "unknown"),
                    match_score=match_score,
                    matched_capabilities=capability_match["matched"],
                    missing_capabilities=capability_match["missing"],
                    capability_coverage=capability_match["coverage"],
                    estimated_execution_time_ms=performance_prediction.get("execution_time_ms", 0.0),
                    confidence_score=performance_prediction.get("confidence", 0.5),
                    current_load=agent_info.get("current_load", 0.0),
                    availability_score=agent_info.get("availability_score", 1.0),
                    queue_length=agent_info.get("queue_length", 0),
                    specialization_score=capability_match.get("specialization", 0.5),
                    historical_success_rate=agent_info.get("success_rate", 0.9)
                )

                agent_matches.append(agent_match)

            # Sortiere nach Match-Score und limitiere
            agent_matches.sort(key=lambda x: x.match_score, reverse=True)
            return agent_matches[:max_agents]

        except Exception as e:
            logger.error(f"Agent-Matching für Subtask {subtask.subtask_id} fehlgeschlagen: {e}")
            return []

    async def _get_available_agents(self) -> list[dict[str, Any]]:
        """Holt verfügbare Agents aus Registry."""
        try:
            # Hole alle Agents aus Registry
            all_agents_raw = await self.agent_registry.list_agents()

            # Konvertiere zu Dictionary falls Liste zurückgegeben wird
            if isinstance(all_agents_raw, list):
                all_agents = {
                    getattr(agent, "id", f"agent_{i}"): agent
                    for i, agent in enumerate(all_agents_raw)
                }
            else:
                all_agents = all_agents_raw

            available_agents = []
            for agent_id, agent_info in all_agents.items():
                # Extrahiere Agent-Informationen
                agent_data = {
                    "agent_id": agent_id,
                    "agent_type": getattr(agent_info, "agent_type", "unknown"),
                    "capabilities": getattr(agent_info, "capabilities", []),
                    "current_load": self._estimate_agent_load(agent_id),
                    "availability_score": self._calculate_availability_score(agent_info),
                    "queue_length": self._get_agent_queue_length(agent_id),
                    "success_rate": self._get_agent_success_rate(agent_id),
                    "avg_response_time_ms": self._get_agent_avg_response_time(agent_id)
                }

                available_agents.append(agent_data)

            return available_agents

        except Exception as e:
            logger.error(f"Fehler beim Holen verfügbarer Agents: {e}")
            return []

    async def _calculate_capability_match(
        self,
        subtask: SubtaskDefinition,
        agent_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Berechnet Capability-Matching zwischen Subtask und Agent."""
        required_caps = set(cap.lower() for cap in subtask.required_capabilities)
        agent_caps = set(cap.lower() for cap in agent_info.get("capabilities", []))

        # Matched und Missing Capabilities
        matched_caps = list(required_caps.intersection(agent_caps))
        missing_caps = list(required_caps.difference(agent_caps))

        # Coverage-Score
        coverage = len(matched_caps) / len(required_caps) if required_caps else 1.0

        # Spezialisierungs-Score (basierend auf Task-Type)
        specialization = self._calculate_specialization_score(subtask, agent_info)

        return {
            "matched": matched_caps,
            "missing": missing_caps,
            "coverage": coverage,
            "specialization": specialization
        }

    def _calculate_specialization_score(
        self,
        subtask: SubtaskDefinition,
        agent_info: dict[str, Any]
    ) -> float:
        """Berechnet Spezialisierungs-Score für Task-Type."""
        # Vereinfachte Spezialisierungs-Logic
        agent_type = agent_info.get("agent_type", "").lower()
        task_type = subtask.task_type.value.lower()

        # Direkte Matches
        if agent_type == task_type:
            return 1.0

        # Verwandte Matches
        related_matches = {
            "data_processing": ["nlp_analysis", "batch_job"],
            "nlp_analysis": ["data_processing", "agent_execution"],
            "agent_execution": ["tool_call", "workflow"],
            "tool_call": ["agent_execution", "system_maintenance"]
        }

        if task_type in related_matches.get(agent_type, []):
            return 0.7

        # Default
        return 0.5

    async def _predict_performance(
        self,
        subtask: SubtaskDefinition,
        agent_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Sagt Performance mit ML-Pipeline vorher."""
        if not self.performance_predictor:
            # Fallback auf einfache Schätzung
            return {
                "execution_time_ms": subtask.estimated_duration_minutes * 60 * 1000,
                "confidence": 0.5
            }

        try:
            # Erstelle Task-Charakteristika
            task_chars = TaskCharacteristics(
                task_type=subtask.task_type.value,
                complexity_score=5.0,  # Default-Komplexität
                estimated_tokens=len(str(subtask.payload)) * 4,  # Grobe Token-Schätzung
                required_capabilities=subtask.required_capabilities
            )

            # Erstelle Agent-Charakteristika
            agent_chars = AgentCharacteristics(
                agent_id=agent_info["agent_id"],
                agent_type=agent_info["agent_type"],
                capabilities=agent_info["capabilities"],
                current_load=agent_info["current_load"],
                avg_response_time_ms=agent_info["avg_response_time_ms"],
                success_rate=agent_info["success_rate"],
                error_rate=1.0 - agent_info["success_rate"],
                max_concurrent_tasks=10,  # Default
                current_active_tasks=int(agent_info["current_load"] * 10),
                queue_length=agent_info["queue_length"],
                total_completed_tasks=100,  # Default
                avg_task_completion_time_ms=agent_info["avg_response_time_ms"],
                specialization_score=0.8  # Default
            )

            # Performance-Prediction
            prediction_request = PredictionRequest(
                task_characteristics=task_chars,
                agent_characteristics=agent_chars,
                system_load=0.5,  # Default System-Load
                concurrent_executions=5  # Default
            )

            prediction_result = await self.performance_predictor.predict_execution_time(prediction_request)

            return {
                "execution_time_ms": prediction_result.predicted_execution_time_ms,
                "confidence": prediction_result.confidence_score
            }

        except Exception as e:
            logger.warning(f"Performance-Prediction fehlgeschlagen: {e}")
            return {
                "execution_time_ms": subtask.estimated_duration_minutes * 60 * 1000,
                "confidence": 0.3
            }

    def _calculate_match_score(
        self,
        capability_match: dict[str, Any],
        performance_prediction: dict[str, Any],
        agent_info: dict[str, Any]
    ) -> float:
        """Berechnet Gesamt-Match-Score."""
        # Capability-Score
        capability_score = capability_match["coverage"]

        # Performance-Score (basierend auf Confidence)
        performance_score = performance_prediction.get("confidence", 0.5)

        # Availability-Score
        availability_score = agent_info.get("availability_score", 1.0)

        # Spezialisierungs-Score
        specialization_score = capability_match.get("specialization", 0.5)

        # Gewichteter Gesamtscore
        total_score = (
            capability_score * 0.4 +
            performance_score * self.performance_weight +
            availability_score * self.availability_weight +
            specialization_score * self.specialization_weight
        )

        return min(1.0, max(0.0, total_score))

    def _estimate_agent_load(self, agent_id: str) -> float:
        """Schätzt aktuelle Agent-Load."""
        # TODO: Implementiere echte Load-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        # Für jetzt: Vereinfachte Schätzung
        return 0.3  # 30% Load

    def _calculate_availability_score(self, agent_info: Any) -> float:
        """Berechnet Verfügbarkeits-Score."""
        # TODO: Implementiere echte Verfügbarkeits-Berechnung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        return 0.9  # 90% verfügbar

    def _get_agent_queue_length(self, agent_id: str) -> int:
        """Holt Agent-Queue-Länge."""
        # TODO: Implementiere echte Queue-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        return 2  # 2 Tasks in Queue

    def _get_agent_success_rate(self, agent_id: str) -> float:
        """Holt Agent-Success-Rate."""
        # TODO: Implementiere echte Success-Rate-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        return 0.92  # 92% Success-Rate

    def _get_agent_avg_response_time(self, agent_id: str) -> float:
        """Holt Agent-durchschnittliche Response-Zeit."""
        # TODO: Implementiere echte Response-Zeit-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        return 250.0  # 250ms durchschnittlich

    def _update_performance_stats(self, matching_time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._matching_count += 1
        self._total_matching_time_ms += matching_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_time = (
            self._total_matching_time_ms / self._matching_count
            if self._matching_count > 0 else 0.0
        )

        return {
            "total_matchings": self._matching_count,
            "avg_matching_time_ms": avg_time,
            "total_matching_time_ms": self._total_matching_time_ms,
            "min_capability_coverage": self.min_capability_coverage,
            "max_load_threshold": self.max_load_threshold
        }
