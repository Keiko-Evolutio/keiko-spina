"""Visualisierung und Export von LangGraph-Workflows in DOT, Mermaid und HTML.

Dieses Modul stellt eine produktionsreife Visualisierung für LangGraph
Workflows bereit. Es akzeptiert eine generische Graph-Repräsentation mit Nodes
und Edges und erzeugt:
- Graphviz DOT
- Mermaid flowchart
- Interaktive HTML-Vorschau (Mermaid-initialisiert)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from observability import trace_span

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Hilfstypen
# ---------------------------------------------------------------------------

NodeId = str


@dataclass
class VisualNode:
    """Repräsentation eines Knotens für die Visualisierung.

    Attributes:
        id: Eindeutige ID des Knotens
        label: Menschlich lesbarer Titel
        type: Logischer Node-Typ (z. B. task, decision, human, parallel, end)
    """

    id: NodeId
    label: str
    type: str = "task"


@dataclass
class VisualEdge:
    """Repräsentation einer Kante für die Visualisierung.

    Attributes:
        source: Quellknoten-ID
        target: Zielknoten-ID
        condition: Optionales Label für Bedingungen
        edge_type: Optionaler Edge-Typ (z. B. default, success, error)
    """

    source: NodeId
    target: NodeId
    condition: str | None = None
    edge_type: str = "default"


class WorkflowVisualizer:
    """Erzeugt Visualisierungen aus einer Workflow-Graph-Repräsentation.

    Erwartete Graph-Struktur (flexibel):
    - nodes: Liste von Strings (IDs) oder Dicts mit Feldern {id, label, type}
    - edges: Liste von 2er-Tupeln (source, target) oder Dicts {source, target, condition?, type?}
    """

    def __init__(self, graph: dict[str, Any]) -> None:
        """Initialisiert den Visualizer mit generischem Graph.

        Args:
            graph: Serialisierbare Graph-Struktur
        """
        # Intern normalisieren wir Nodes und Edges in klare Klassen
        self.nodes: list[VisualNode] = self._normalize_nodes(graph.get("nodes", []))
        self.edges: list[VisualEdge] = self._normalize_edges(graph.get("edges", []))

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def export_to_dot(self) -> str:
        """Erzeugt Graphviz DOT-Text für den Graphen."""
        with trace_span(
            "workflow.visualize.dot", {"nodes": len(self.nodes), "edges": len(self.edges)}
        ):
            lines: list[str] = [
                "digraph Workflow {",
                "  rankdir=LR;",
                "  node [fontname=Helvetica];",
            ]

            for node in self.nodes:
                shape, color = _shape_and_color_for_type(node.type)
                safe_label = node.label.replace('"', "'")
                lines.append(
                    f'  "{node.id}" [label="{safe_label}", shape={shape}, style=filled, fillcolor="{color}"];'
                )

            for edge in self.edges:
                label = f' [label="{edge.condition}"]' if edge.condition else ""
                lines.append(f'  "{edge.source}" -> "{edge.target}"{label};')

            lines.append("}")
            return "\n".join(lines)

    def export_to_mermaid(self) -> str:
        """Erzeugt Mermaid flowchart-Text für den Graphen."""
        with trace_span(
            "workflow.visualize.mermaid", {"nodes": len(self.nodes), "edges": len(self.edges)}
        ):
            lines: list[str] = ["flowchart LR"]

            for node in self.nodes:
                mermaid_label = node.label.replace("[", "(").replace("]", ")")
                node_repr = _mermaid_node_repr(node, mermaid_label)
                lines.append(f"  {node.id}{node_repr}")

            for edge in self.edges:
                label = f' |"{edge.condition}"|' if edge.condition else ""
                lines.append(f"  {edge.source} --{label}--> {edge.target}")

            return "\n".join(lines)

    def generate_html_preview(self, *, mermaid_theme: str = "default") -> str:
        """Erzeugt eine eigenständige HTML-Vorschau mit Mermaid.

        Args:
            mermaid_theme: Mermaid-Theme (z. B. default, dark, neutral)

        Returns:
            Vollständige HTML-Seite als String
        """
        diagram = self.export_to_mermaid()
        with trace_span("workflow.visualize.html", {"theme": mermaid_theme}):
            return f"""
<!doctype html>
<html lang=\"de\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Keiko Workflow Preview</title>
  <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>
  <style>
    body {{ margin: 0; padding: 16px; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .mermaid {{ background: #fff; border-radius: 8px; padding: 16px; }}
  </style>
  <script>
    mermaid.initialize({{ startOnLoad: true, theme: '{mermaid_theme}' }});
  </script>
  </head>
  <body>
    <div class=\"container\">
      <h1>Keiko Workflow Preview</h1>
      <div class=\"mermaid\">{diagram}</div>
    </div>
  </body>
</html>
""".strip()

    # ------------------------------------------------------------------
    # Normalisierung
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_nodes(raw_nodes: Iterable[str | dict[str, Any]]) -> list[VisualNode]:
        """Normalisiert eine gemischte Node-Liste in `VisualNode`-Objekte."""
        normalized: list[VisualNode] = []
        for item in raw_nodes:
            if isinstance(item, str):
                normalized.append(VisualNode(id=item, label=item, type=_infer_type(item)))
                continue
            if isinstance(item, dict):
                node_id: str = str(item.get("id") or item.get("name"))
                label: str = str(item.get("label") or node_id)
                node_type: str = str(item.get("type") or _infer_type(node_id))
                normalized.append(VisualNode(id=node_id, label=label, type=node_type))
        return normalized

    @staticmethod
    def _normalize_edges(
        raw_edges: Iterable[tuple[str, str] | dict[str, Any]],
    ) -> list[VisualEdge]:
        """Normalisiert eine gemischte Edge-Liste in `VisualEdge`-Objekte."""
        edges: list[VisualEdge] = []
        for item in raw_edges:
            if isinstance(item, tuple) and len(item) == 2:
                edges.append(VisualEdge(source=str(item[0]), target=str(item[1])))
                continue
            if isinstance(item, dict):
                source: str = str(item.get("source") or item.get("from"))
                target: str = str(item.get("target") or item.get("to"))
                condition: str | None = item.get("condition")
                edge_type: str = str(item.get("type") or "default")
                edges.append(
                    VisualEdge(
                        source=source, target=target, condition=condition, edge_type=edge_type
                    )
                )
        return edges


# ---------------------------------------------------------------------------
# Kompatible Top-Level-Exports (Backward Compatibility)
# ---------------------------------------------------------------------------


def export_to_dot(graph: dict[str, Any]) -> str:
    """Exportiert Graph in DOT (kompatibler Top-Level-Helper)."""
    return WorkflowVisualizer(graph).export_to_dot()


def export_to_mermaid(graph: dict[str, Any]) -> str:
    """Exportiert Graph in Mermaid (kompatibler Top-Level-Helper)."""
    return WorkflowVisualizer(graph).export_to_mermaid()


# ---------------------------------------------------------------------------
# Interne Utilities
# ---------------------------------------------------------------------------


def _shape_and_color_for_type(node_type: str) -> tuple[str, str]:
    """Liefert Shape und Farbe für DOT basierend auf Node-Typ."""
    mapping = {
        "start": ("circle", "#E3F2FD"),
        "end": ("doublecircle", "#E8F5E9"),
        "task": ("box", "#FFFDE7"),
        "decision": ("diamond", "#F3E5F5"),
        "parallel": ("hexagon", "#E0F7FA"),
        "human": ("oval", "#FFECB3"),
        "retry": ("box", "#FFEBEE"),
        "loop": ("box", "#EDE7F6"),
        "router": ("parallelogram", "#E1F5FE"),
    }
    return mapping.get(node_type, ("box", "#FFFFFF"))


def _infer_type(node_id: str) -> str:
    """Leitet Node-Typ heuristisch aus der ID ab (Fallback für einfache Graphen)."""
    lower = node_id.lower()
    if lower in {"start", "entry"}:
        return "start"
    if lower in {"end", "finish", "exit"}:
        return "end"
    if "route" in lower or "router" in lower:
        return "router"
    if "human" in lower or "review" in lower:
        return "human"
    if "retry" in lower or "error" in lower:
        return "retry"
    if "loop" in lower or "iterate" in lower:
        return "loop"
    if "cond" in lower or "branch" in lower or "decision" in lower:
        return "decision"
    if "parallel" in lower or "fanout" in lower:
        return "parallel"
    return "task"


def _mermaid_node_repr(node: VisualNode, label: str) -> str:
    """Erzeugt Mermaid-Repräsentation für einen einzelnen Node."""
    t = node.type
    if t == "start":
        return f'(("{label}"))'
    if t == "end":
        return f"(({label}))"
    if t == "decision":
        return f"{{{label}}}"
    if t == "parallel":
        return f"[[{label}]]"
    if t == "human":
        return f"([{label}])"
    return f"[{label}]"


__all__ = [
    "WorkflowVisualizer",
    "export_to_dot",
    "export_to_mermaid",
]
