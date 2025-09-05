"""LangGraph Workflow - Kern-Workflow-Engine auf Basis von LangGraph StateGraph.

Stellt eine Produktionsbasis bereit für Multi-Agent-Workflows inklusive:
- StateGraph-Integration
- Agent-to-Agent Kommunikation via `BaseAgentProtocol`
- Integration mit `AgentExecutionContext`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from agents.capabilities import get_capability_manager
from agents.memory.langgraph_cosmos_checkpointer import CosmosCheckpointSaver
from agents.state.langgraph_state_bridge import (
    WorkflowState,
    replace_bool,
    replace_dict,
    replace_int,
)
from kei_logging import get_logger


# Fehlende replace-Funktionen definieren
def replace_optional_str(left: str | None, right: str | None) -> str | None:
    """Ersetzt optionalen String-Wert."""
    return right if right is not None else left

def replace_list(left: list | None, right: list | None) -> list | None:
    """Ersetzt Listen-Wert."""
    return right if right is not None else left
from agents.workflows.advanced_patterns import SubgraphComposer
from agents.workflows.dynamic_node_factory import DynamicNodeFactory, DynamicNodeFactoryConfig
from agents.workflows.graph_export import WorkflowVisualizer
from agents.workflows.workflow_node_factory import WorkflowNodeFactory
from agents.workflows.workflows_utils import handle_langgraph_import, run_sync

StateGraph, END = handle_langgraph_import()

logger = get_logger(__name__)


@dataclass
class OrchestrationState(WorkflowState):
    """Erweiterter State für Multi-Agent-Orchestrierung."""

    target_agent_id: Annotated[str | None, replace_optional_str] = None
    last_output: Annotated[str | None, replace_optional_str] = None
    error: Annotated[str | None, replace_optional_str] = None
    retry_count: Annotated[int, replace_int] = 0
    max_retries: Annotated[int, replace_int] = 2
    human_required: Annotated[bool, replace_bool] = False
    human_feedback: Annotated[str | None, replace_optional_str] = None
    branch: Annotated[str | None, replace_optional_str] = None
    parallel_results: Annotated[list[str] | None, replace_list] = None
    extras: Annotated[dict[str, Any], replace_dict] = field(default_factory=dict)

    def __init__(self, message: str = "", **kwargs):
        """Initialisiert OrchestrationState mit erforderlicher message."""
        super().__init__(message=message)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LangGraphWorkflow:
    """Builder für StateGraph basierte Multi-Agent-Workflows."""

    def __init__(self) -> None:
        """Initialisiert den Workflow-Builder."""
        self.nodes: list[str] = ["analyze_intent", "route", "invoke_agent", "end"]
        self.edges: list[tuple[str, str]] = [
            ("analyze_intent", "route"),
            ("route", "invoke_agent"),
            ("invoke_agent", "end"),
        ]

    def build(self) -> dict[str, Any]:
        """Gibt eine einfache serialisierbare Graph-Repräsentation zurück."""
        return {"nodes": self.nodes.copy(), "edges": self.edges.copy()}

    @staticmethod
    def build_extended() -> dict[str, Any]:
        """Gibt eine erweiterte Graph-Repräsentation mit Node-Typen und Bedingungen zurück."""
        nodes: list[dict[str, Any]] = [
            {"id": "analyze_intent", "label": "Analyze Intent", "type": "start"},
            {"id": "decision", "label": "Decision", "type": "decision"},
            {"id": "route", "label": "Router", "type": "router"},
            {"id": "human_review", "label": "Human Review", "type": "human"},
            {"id": "invoke_agent", "label": "Invoke Agent", "type": "task"},
            {"id": "parallel_invoke", "label": "Parallel Invoke", "type": "parallel"},
            {"id": "aggregate", "label": "Aggregate Results", "type": "task"},
            {"id": "retry_node", "label": "Retry", "type": "retry"},
            {"id": "loop_decide", "label": "Loop Decide", "type": "loop"},
            {"id": "post_process", "label": "Post Process", "type": "task"},
            {"id": "end", "label": "End", "type": "end"},
        ]

        edges: list[dict[str, Any]] = [
            {"source": "analyze_intent", "target": "decision"},
            {"source": "decision", "target": "route", "condition": "auto"},
            {"source": "decision", "target": "human_review", "condition": "human"},
            {"source": "human_review", "target": "route", "condition": "approved"},
            {"source": "route", "target": "parallel_invoke"},
            {"source": "parallel_invoke", "target": "aggregate"},
            {"source": "aggregate", "target": "post_process"},
            {"source": "post_process", "target": "loop_decide"},
            {"source": "loop_decide", "target": "invoke_agent", "condition": "iterate"},
            {"source": "loop_decide", "target": "end", "condition": "done"},
            {"source": "invoke_agent", "target": "retry_node"},
            {"source": "retry_node", "target": "invoke_agent", "condition": "retry"},
            {"source": "retry_node", "target": "end", "condition": "giveup"},
        ]

        return {"nodes": nodes, "edges": edges}

    def export_to_dot(self, *, extended: bool = True) -> str:
        """Exportiert aktuellen Workflow in DOT-Format."""
        graph = self.build_extended() if extended else self.build()
        return WorkflowVisualizer(graph).export_to_dot()

    def export_to_mermaid(self, *, extended: bool = True) -> str:
        """Exportiert aktuellen Workflow in Mermaid-Format."""
        graph = self.build_extended() if extended else self.build()
        return WorkflowVisualizer(graph).export_to_mermaid()

    def generate_html_preview(
        self, *, extended: bool = True, mermaid_theme: str = "default"
    ) -> str:
        """Erzeugt HTML-Vorschau für den Workflow."""
        graph = self.build_extended() if extended else self.build()
        return WorkflowVisualizer(graph).generate_html_preview(mermaid_theme=mermaid_theme)

    @staticmethod
    def compile_graph() -> Any:
        """Baut und kompiliert den StateGraph mit Cosmos-Checkpointing."""
        if StateGraph is None:
            raise RuntimeError("LangGraph nicht installiert")

        graph = StateGraph(OrchestrationState)

        # Erstelle Workflow-Nodes
        node_analyze = WorkflowNodeFactory.create_analyze_node()
        node_decide = WorkflowNodeFactory.create_decide_node()
        node_human_review = WorkflowNodeFactory.create_human_review_node()
        node_route = WorkflowNodeFactory.create_route_node()
        node_parallel_invoke = WorkflowNodeFactory.create_parallel_invoke_node()
        node_aggregate = WorkflowNodeFactory.create_aggregate_node()
        node_invoke = WorkflowNodeFactory.create_invoke_node()
        node_retry_decide = WorkflowNodeFactory.create_retry_decide_node(max_retries=3)
        node_post_process = WorkflowNodeFactory.create_post_process_node()
        node_loop_decide = WorkflowNodeFactory.create_loop_decide_node()

        # Registriere Nodes im Graph
        graph.add_node("analyze_intent", node_analyze)
        graph.add_node("decide", node_decide)
        graph.add_node("human_review", node_human_review)
        graph.add_node("route", node_route)
        graph.add_node("parallel_invoke", node_parallel_invoke)
        graph.add_node("aggregate", node_aggregate)
        graph.add_node("invoke_agent", node_invoke)
        graph.add_node("post_process", node_post_process)
        graph.add_node("loop_decide", node_loop_decide)
        graph.add_node("retry_decide", node_retry_decide)

        graph.set_entry_point("analyze_intent")
        graph.add_edge("analyze_intent", "decide")

        # Hilfsfunktion für bedingte Kanten
        def _pick_branch(s: dict[str, Any] | OrchestrationState) -> str:
            return str(s.get("branch") if isinstance(s, dict) else getattr(s, "branch", "end"))

        # Bedingte Kanten: decide -> human_review oder route
        try:  # pragma: no cover - optional
            if hasattr(graph, "add_conditional_edges"):
                graph.add_conditional_edges(
                    "decide", _pick_branch, {"human_review": "human_review", "route": "route"}
                )
            else:
                graph.add_edge("decide", "route")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Conditional edges nicht verfügbar: {e}")
            graph.add_edge("decide", "route")

        # Bedingte Kanten: human_review -> wait (END) oder route
        try:  # pragma: no cover - optional
            if hasattr(graph, "add_conditional_edges"):
                graph.add_conditional_edges(
                    "human_review", _pick_branch, {"wait": END, "route": "route"}
                )
            else:
                graph.add_edge("human_review", "route")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Conditional edges nicht verfügbar: {e}")
            graph.add_edge("human_review", "route")

        # Routing-Logik: route -> parallel_invoke oder invoke_agent
        def _fanout(s: dict[str, Any] | OrchestrationState) -> str:
            ex = s.get("extras") if isinstance(s, dict) else getattr(s, "extras", {})
            if isinstance(ex, dict) and ex.get("parallel_targets"):
                return "parallel_invoke"
            return "invoke_agent"

        try:  # pragma: no cover - optional
            if hasattr(graph, "add_conditional_edges"):
                graph.add_conditional_edges(
                    "route",
                    _fanout,
                    {"parallel_invoke": "parallel_invoke", "invoke_agent": "invoke_agent"},
                )
            else:
                graph.add_edge("route", "invoke_agent")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Conditional edges nicht verfügbar: {e}")
            graph.add_edge("route", "invoke_agent")

        # Verbinde Ausführungs-Pfade
        graph.add_edge("parallel_invoke", "aggregate")
        graph.add_edge("aggregate", "post_process")
        graph.add_edge("invoke_agent", "post_process")
        graph.add_edge("post_process", "loop_decide")

        # Bedingte Kanten: loop_decide -> invoke_agent oder end
        try:  # pragma: no cover - optional
            if hasattr(graph, "add_conditional_edges"):
                graph.add_conditional_edges(
                    "loop_decide", _pick_branch, {"invoke_agent": "invoke_agent", "end": END}
                )
            else:
                graph.add_edge("post_process", END)
        except (AttributeError, TypeError) as e:
            logger.warning(f"Conditional edges nicht verfügbar: {e}")
            graph.add_edge("post_process", END)

        # Fehlerbehandlung und Retry-Logik
        graph.add_edge("invoke_agent", "retry_decide")
        try:  # pragma: no cover - optional
            if hasattr(graph, "add_conditional_edges"):
                graph.add_conditional_edges(
                    "retry_decide", _pick_branch, {"invoke_agent": "invoke_agent", "end": END}
                )
            else:
                graph.add_edge("retry_decide", END)
        except Exception:
            graph.add_edge("retry_decide", END)

        saver = CosmosCheckpointSaver()
        return graph.compile(checkpointer=saver)

    @staticmethod
    def compile_graph_advanced(
        *,
        dynamic: bool = True,
        required_capabilities: list[str] | None = None,
    ) -> Any:
        """Baut einen erweiterten Graphen mit dynamischen Nodes und Subgraph-Patterns.

        Args:
            dynamic: Ob dynamische Nodes aus Registry eingebunden werden sollen
            required_capabilities: Capabilities-Filter für dynamische Node-Erzeugung
        """
        if StateGraph is None:
            raise RuntimeError("LangGraph nicht installiert")

        graph = StateGraph(OrchestrationState)

        # Erstelle Workflow-Nodes
        node_analyze = WorkflowNodeFactory.create_analyze_node(required_capabilities)
        node_route = WorkflowNodeFactory.create_route_node()
        node_invoke = WorkflowNodeFactory.create_invoke_node()

        # Registriere Basis-Nodes
        graph.add_node("analyze_intent", node_analyze)
        graph.add_node("route", node_route)
        graph.add_node("invoke_agent", node_invoke)
        graph.set_entry_point("analyze_intent")
        graph.add_edge("analyze_intent", "route")

        # Füge dynamische Nodes hinzu (optional)
        next_after_dynamic = "post_process"
        if dynamic:
            factory = DynamicNodeFactory(
                config=DynamicNodeFactoryConfig(
                    max_dynamic_nodes=10, connect_from="route", connect_to=next_after_dynamic
                )
            )
            try:
                # Verwende capability_manager als Registry
                capability_manager = get_capability_manager()
                dyn_nodes, dyn_edges = run_sync(
                    factory.create_nodes(
                        required_capabilities=required_capabilities, registry=capability_manager
                    )
                )
                for node_id, fn in dyn_nodes:
                    graph.add_node(node_id, fn)
                for src, dst in dyn_edges:
                    graph.add_edge(src, dst)
            except (ValueError, TypeError) as e:  # pragma: no cover
                logger.warning(f"Dynamische Nodes konnten nicht erstellt werden - Validierungsfehler: {e}")
                graph.add_edge("route", "invoke_agent")
            except (ConnectionError, TimeoutError) as e:  # pragma: no cover
                logger.warning(f"Dynamische Nodes konnten nicht erstellt werden - Verbindungsproblem: {e}")
                graph.add_edge("route", "invoke_agent")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Dynamische Nodes konnten nicht erstellt werden - Unerwarteter Fehler: {e}")
                graph.add_edge("route", "invoke_agent")
        else:
            graph.add_edge("route", "invoke_agent")

        # Füge Post-Processing hinzu
        def _post_process_orchestration_state(state: OrchestrationState) -> OrchestrationState:
            """Führt Post-Processing der Orchestration-State durch.

            Args:
                state: Aktueller Orchestration-State

            Returns:
                Verarbeiteter State (aktuell unverändert)
            """
            # Post-Processing-Logik kann hier erweitert werden
            return state

        graph.add_node("post_process", _post_process_orchestration_state)

        # Integriere Retry-Mechanismus mit konfigurierbaren Werten
        DEFAULT_MAX_RETRY_ATTEMPTS = 3  # Fallback-Wert
        composer = SubgraphComposer(graph)
        composer.insert_retry(
            task=("invoke_agent", node_invoke),
            max_retries=DEFAULT_MAX_RETRY_ATTEMPTS,
            exit_to="end"
        )

        # Verbinde finale Kanten mit spezifischer Exception-Behandlung
        try:  # pragma: no cover
            graph.add_edge("post_process", "invoke_agent")
        except ValueError as e:
            # Edge bereits vorhanden oder ungültige Konfiguration
            logger.warning(f"Konnte finale Kante nicht hinzufügen - Validierungsfehler: {e}")
        except (AttributeError, TypeError) as e:
            # Graph-Objekt oder Methoden nicht verfügbar
            logger.warning(f"Konnte finale Kante nicht hinzufügen - Attribut-/Typ-Fehler: {e}")
        except RuntimeError as e:
            # Unerwarteter Fehler beim Hinzufügen der Kante
            logger.error(f"Unerwarteter Fehler beim Graph-Setup: {e}")
            raise

        saver = CosmosCheckpointSaver()
        return graph.compile(checkpointer=saver)

__all__ = ["LangGraphWorkflow", "OrchestrationState"]
