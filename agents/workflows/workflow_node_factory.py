"""Zentrale Factory für Workflow-Node-Funktionen.

Dieses Modul konsolidiert alle Node-Implementierungen, die in verschiedenen
Workflow-Buildern verwendet werden, um Code-Duplikation zu eliminieren.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kei_logging import get_logger
from observability import trace_span

from .workflows_utils import run_sync

logger = get_logger(__name__)


class WorkflowNodeFactory:
    """Factory für wiederverwendbare Workflow-Node-Funktionen.

    Diese Klasse stellt alle Standard-Node-Implementierungen bereit,
    die in verschiedenen Workflow-Konfigurationen verwendet werden können.
    """

    @staticmethod
    def create_analyze_node(
        required_capabilities: list[str] | None = None
    ) -> Callable[[Any], Any]:
        """Erstellt eine Intent-Analyse-Node.

        Args:
            required_capabilities: Standard-Capabilities falls keine erkannt werden

        Returns:
            Node-Funktion für Intent-Analyse
        """
        def node_analyze(state: Any) -> Any:
            """Analysiert das Anliegen und bereitet Routing vor."""
            try:
                from agents.orchestrator.intent_recognition import (
                    detect_image_intent,
                    detect_photo_intent,
                )

                text = getattr(state, "message", "") or ""

                # Stelle sicher, dass extras existiert
                if not hasattr(state, "extras") or not isinstance(state.extras, dict):
                    state.extras = {}

                if detect_photo_intent(text).is_photo:
                    state.extras["required_capabilities"] = ["camera", "photo"]
                elif detect_image_intent(text).is_image:
                    state.extras["required_capabilities"] = ["image_generation"]
                else:
                    state.extras["required_capabilities"] = required_capabilities or ["assistant"]
                return state
            except Exception as e:  # pragma: no cover - defensiv
                state.error = str(e)
                return state

        return node_analyze

    @staticmethod
    def create_route_node() -> Callable[[Any], Any]:
        """Erstellt eine Routing-Node für Agent-Auswahl."""
        def node_route(state: Any) -> Any:
            """Wählt passenden Agenten über Dynamic Registry und Policy."""
            required = (
                state.extras.get("required_capabilities", [])
                if hasattr(state, "extras") and isinstance(state.extras, dict)
                else []
            )
            try:
                from agents.capabilities import get_capability_manager
                from agents.routing.conditional_agent_router import route_to_best_agent

                # Sammle verfügbare Agents in vereinfachter dict-Form
                agents: list[dict[str, Any]] = []
                for agent in get_capability_manager()._agent_capabilities.values():
                    agents.append({
                        "id": getattr(agent, "id", None) or getattr(agent, "name", "unknown"),
                        "name": getattr(agent, "name", "unknown"),
                        "capabilities": getattr(agent, "capabilities", []),
                        "description": getattr(agent, "description", ""),
                    })

                chosen = route_to_best_agent(agents=agents, required_capabilities=required)
                if chosen:
                    state.target_agent_id = chosen.get("id")
                else:
                    state.error = "no_agent_found"
                return state
            except Exception as e:  # pragma: no cover - defensiv
                state.error = str(e)
                return state

        return node_route

    @staticmethod
    def create_invoke_node() -> Callable[[Any], Any]:
        """Erstellt eine Agent-Invoke-Node."""
        def node_invoke(state: Any) -> Any:
            """Ruft den gewählten Agenten über Unified Protocol auf."""
            if not getattr(state, "target_agent_id", None):
                return state
            try:
                from agents.common.operations import execute_agent_task

                result = run_sync(
                    execute_agent_task(
                        agent_id=state.target_agent_id,
                        task=getattr(state, "message", ""),
                        framework=None,
                    )
                )
                if isinstance(result, dict) and result.get("error"):
                    state.error = str(result.get("error"))
                else:
                    state.last_output = str(result)
                return state
            except Exception as e:  # pragma: no cover - defensiv
                state.error = str(e)
                return state

        return node_invoke

    @staticmethod
    def create_decide_node() -> Callable[[Any], Any]:
        """Erstellt eine Entscheidungs-Node für Human-in-the-Loop."""
        def node_decide(state: Any) -> Any:
            """Entscheidet, ob Human-in-the-Loop erforderlich ist."""
            with trace_span("workflow.decide", {"human_required": getattr(state, "human_required", False)}):
                human_required = getattr(state, "human_required", False)
                state.branch = "human_review" if human_required else "route"
                return state

        return node_decide

    @staticmethod
    def create_human_review_node() -> Callable[[Any], Any]:
        """Erstellt eine Human-Review-Node."""
        def node_human_review(state: Any) -> Any:
            """Human-Gate: wartet auf Feedback, parkt falls nicht vorhanden."""
            with trace_span("workflow.human_review", None):
                human_feedback = getattr(state, "human_feedback", None)
                if not human_feedback:
                    state.branch = "wait"
                else:
                    state.branch = "route"
                return state

        return node_human_review

    @staticmethod
    def create_parallel_invoke_node() -> Callable[[Any], Any]:
        """Erstellt eine Parallel-Invoke-Node."""
        def node_parallel_invoke(state: Any) -> Any:
            """Führt mehrere Agent-Aufrufe parallel aus (falls konfiguriert)."""
            with trace_span("workflow.parallel_invoke", None):
                extras = getattr(state, "extras", {})
                targets = extras.get("parallel_targets", []) if isinstance(extras, dict) else []

                if not targets:
                    state.parallel_results = []
                    return state

                try:
                    import asyncio

                    from agents.common.operations import execute_agent_task

                    async def _gather() -> list[str]:
                        async def _call(agent_id: str) -> str:
                            try:
                                res = await execute_agent_task(
                                    agent_id=agent_id,
                                    task=getattr(state, "message", ""),
                                    framework=None
                                )
                                return str(res)
                            except Exception as ee:  # pragma: no cover - defensiv
                                return f"error:{ee}"

                        return await asyncio.gather(*[_call(t) for t in targets])

                    state.parallel_results = run_sync(_gather())
                    return state
                except Exception as e:  # pragma: no cover - defensiv
                    state.error = str(e)
                    return state

        return node_parallel_invoke

    @staticmethod
    def create_aggregate_node() -> Callable[[Any], Any]:
        """Erstellt eine Aggregations-Node."""
        def node_aggregate(state: Any) -> Any:
            """Aggregiert parallele Ergebnisse in `last_output`."""
            with trace_span("workflow.aggregate", None):
                parallel_results = getattr(state, "parallel_results", None)
                if parallel_results:
                    state.last_output = "\n".join(parallel_results)
                return state

        return node_aggregate

    @staticmethod
    def create_retry_decide_node(max_retries: int = 2) -> Callable[[Any], Any]:
        """Erstellt eine Retry-Entscheidungs-Node.

        Args:
            max_retries: Maximale Anzahl von Wiederholungsversuchen
        """
        def node_retry_decide(state: Any) -> Any:
            """Entscheidet über erneuten Versuch oder Abschluss basierend auf Fehlern."""
            try:
                error = getattr(state, "error", None)
                retry_count = getattr(state, "retry_count", 0)

                if error and retry_count < max_retries:
                    state.retry_count = retry_count + 1
                    state.branch = "invoke_agent"
                else:
                    state.branch = "end"
                return state
            except Exception as e:  # pragma: no cover - defensiv
                state.branch = "end"
                state.error = str(e)
                return state

        return node_retry_decide

    @staticmethod
    def create_post_process_node() -> Callable[[Any], Any]:
        """Erstellt eine Post-Processing-Node."""
        def node_post_process(state: Any) -> Any:
            """Optionales Post-Processing nach Agent-Calls."""
            with trace_span("workflow.post_process", None):
                return state

        return node_post_process

    @staticmethod
    def create_loop_decide_node() -> Callable[[Any], Any]:
        """Erstellt eine Loop-Entscheidungs-Node."""
        def node_loop_decide(state: Any) -> Any:
            """Entscheidet, ob eine weitere Iteration erfolgen soll."""
            try:
                extras = getattr(state, "extras", {})
                iterate = (
                    bool(extras.get("should_iterate"))
                    if isinstance(extras, dict)
                    else False
                )
                state.branch = "invoke_agent" if iterate else "end"
                return state
            except Exception as e:  # pragma: no cover - defensiv
                state.branch = "end"
                state.error = str(e)
                return state

        return node_loop_decide


__all__ = ["WorkflowNodeFactory"]
