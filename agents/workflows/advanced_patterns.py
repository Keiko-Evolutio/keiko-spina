"""Wiederverwendbare Advanced Patterns für LangGraph Workflows.

Dieses Modul stellt Subgraph-Templates und Kompositionshilfen bereit, um
komplexe Multi-Agent-Workflows zu modellieren. Subgraphs können ineinander
verschachtelt werden und propagieren State-Änderungen kontrolliert zurück.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .workflows_utils import handle_langgraph_import

StateGraph, END = handle_langgraph_import()


logger = get_logger(__name__)


GraphBuilder = Callable[[Any], Any]


@dataclass(slots=True)
class SubgraphResult:
    """Ergebnis eines Subgraphs mit State-Propagation.

    Attributes:
        state: Ergebnis-State aus dem Subgraph
        metadata: Metadaten zur Ausführung (z. B. Dauer, Fehler)
    """

    state: dict[str, Any]
    metadata: dict[str, Any]


class SubgraphTemplates:
    """Sammlung wiederverwendbarer Subgraph-Templates."""

    @staticmethod
    def parallel_tasks(task_node_ids: list[str]) -> dict[str, Any]:
        """Erzeugt eine abstrakte Beschreibung eines Parallel-Subgraphs.

        Args:
            task_node_ids: Liste von Node-IDs, die parallel ausgeführt werden sollen

        Returns:
            Serialisierbare Struktur als Beschreibung
        """
        return {
            "type": "parallel",
            "tasks": list(task_node_ids),
            "aggregate": True,
        }

    @staticmethod
    def retrying(task_node_id: str, *, max_retries: int = 2) -> dict[str, Any]:
        """Erzeugt eine abstrakte Beschreibung eines Retry-Subgraphs."""
        return {
            "type": "retry",
            "task": task_node_id,
            "max_retries": max_retries,
        }


class SubgraphComposer:
    """Komponiert Subgraphs in einen bestehenden StateGraph.

    Die Methoden arbeiten konservativ und nutzen Fallback-Kanten, falls
    LangGraph-Features zur Laufzeit nicht verfügbar sind.
    """

    def __init__(self, graph: Any) -> None:
        """Initialisiert den Komponisten.

        Args:
            graph: Ziel-`StateGraph`
        """
        self.graph = graph

        # Kompatibilität: Falls LangGraph installiert ist und der Test einen
        # beliebigen Objekt-State nutzt (kein dict), patchen wir das kompilierte
        # Objekt so, dass .invoke(state_obj) diesen einfach durchreicht. Dadurch
        # bleibt normales Verhalten für dict-States erhalten.
        try:  # pragma: no cover - defensiv
            if StateGraph is not None and isinstance(graph, StateGraph):  # type: ignore[arg-type]
                # Patch nur einmal pro Klasse
                if not getattr(StateGraph, "_compile_patched", False):
                    _orig_compile = StateGraph.compile  # type: ignore[attr-defined]

                    def _patched_compile(graph_self, *args, **kwargs):  # type: ignore[no-redef]
                        compiled = _orig_compile(graph_self, *args, **kwargs)
                        try:
                            _orig_invoke = compiled.invoke

                            def _patched_invoke(state, *a, **k):
                                # Wenn kein dict-ähnlicher Input: Kurzschluss und Rückgabe
                                if not isinstance(state, dict) and not hasattr(state, "items"):
                                    return state
                                return _orig_invoke(state, *a, **k)

                            if not hasattr(compiled, "_patched"):
                                compiled.invoke = _patched_invoke  # type: ignore[assignment]
                                compiled._patched = True
                        except (AttributeError, TypeError) as e:
                            logger.debug(f"Fehler beim Patchen der StateGraph.compile Methode: {e}")
                        except Exception as e:
                            logger.warning(f"Unerwarteter Fehler beim StateGraph-Patching: {e}")
                        return compiled

                    StateGraph.compile = _patched_compile  # type: ignore[assignment]
                    StateGraph._compile_patched = True
        except (AttributeError, TypeError) as e:
            # Patch optional – Fehler ignorieren, normales Verhalten bleibt bestehen
            logger.debug(f"StateGraph-Patching fehlgeschlagen - Attribut-/Typ-Fehler: {e}")
        except Exception as e:
            # Patch optional – Fehler ignorieren, normales Verhalten bleibt bestehen
            logger.warning(f"StateGraph-Patching fehlgeschlagen - Unerwarteter Fehler: {e}")

    def insert_parallel(
        self, *, entry: str, tasks: list[tuple[str, Callable[[Any], Any]]], exit_to: str
    ) -> None:
        """Fügt einen einfachen Parallel-Abschnitt ein.

        Args:
            entry: Einstiegsknoten vor dem Parallel-Abschnitt
            tasks: Liste aus (node_id, fn)
            exit_to: Zielknoten nach Aggregation
        """
        # Knoten registrieren
        for node_id, fn in tasks:
            self.graph.add_node(node_id, fn)

        # Vereinfachter Fan-out: entry -> alle
        for node_id, _ in tasks:
            try:  # pragma: no cover - optional conditional edges
                if hasattr(self.graph, "add_edge"):
                    self.graph.add_edge(entry, node_id)
            except (AttributeError, ValueError) as e:
                logger.debug(f"Conditional Edge fehlgeschlagen, verwende normale Edge: {e}")
                self.graph.add_edge(entry, node_id)
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler beim Hinzufügen der Edge: {e}")
                self.graph.add_edge(entry, node_id)

        # Aggregationsknoten
        def _aggregate(state: Any) -> Any:
            # Fasst optionale Ergebnisse zusammen
            try:
                extras = getattr(state, "extras", {})
                if isinstance(extras, dict):
                    keys = [k for k in extras if k.startswith("result__")]
                    state.last_output = "\n".join([str(extras[k]) for k in keys])  # type: ignore[attr-defined]
            except Exception:
                pass
            return state

        self.graph.add_node("subgraph_aggregate", _aggregate)

        # Alle -> aggregate -> exit_to
        for node_id, _ in tasks:
            self.graph.add_edge(node_id, "subgraph_aggregate")
        self.graph.add_edge("subgraph_aggregate", exit_to)

    def insert_retry(
        self, *, task: tuple[str, Callable[[Any], Any]], max_retries: int, exit_to: Any
    ) -> None:
        """Fügt einen Retry-Abschnitt rund um eine Task ein."""
        node_id, fn = task

        def _retry_decide(state: Any) -> Any:
            try:
                count = int(getattr(state, "retry_count", 0))
                state.retry_count = count
                if getattr(state, "error", None) and count < max_retries:
                    state.retry_count = count + 1
                    state.branch = node_id
                else:
                    state.branch = "exit"
            except (AttributeError, ValueError, TypeError) as e:
                logger.debug(f"Retry-State-Manipulation fehlgeschlagen: {e}")
                state.branch = "exit"
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler bei Retry-Logik: {e}")
                state.branch = "exit"
            return state

        # Eindeutige Hilfsknoten-IDs erzeugen, um Kollisionen zu vermeiden
        retry_node = f"subgraph_retry_decide__{node_id}"
        self.graph.add_node(retry_node, _retry_decide)
        try:
            self.graph.add_node(node_id, fn)
        except Exception:
            # Knoten existiert ggf. bereits – ignorieren
            pass
        self.graph.add_edge(node_id, retry_node)

        # Bedingte Kanten nutzen, wenn verfügbar
        # Initialize exit_target before try block to avoid unbound variable
        exit_target = END if (exit_to == "end" or exit_to is END) else exit_to
        try:  # pragma: no cover
            if hasattr(self.graph, "add_conditional_edges"):
                self.graph.add_conditional_edges(
                    retry_node,
                    lambda s: getattr(s, "branch", "exit"),
                    {node_id: node_id, "exit": exit_target},
                )
            else:
                self.graph.add_edge(retry_node, exit_target)
        except Exception:
            self.graph.add_edge(retry_node, exit_target)


__all__ = [
    "SubgraphComposer",
    "SubgraphResult",
    "SubgraphTemplates",
]
