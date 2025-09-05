"""Basis-Klassen für Workflow-Module.

Dieses Modul definiert abstrakte Basis-Klassen und gemeinsame Konfigurationen
für alle Workflow-Komponenten, um Code-Duplikation zu vermeiden und
einheitliche Interfaces zu gewährleisten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .workflows_constants import DEFAULT_MAX_DYNAMIC_NODES, DEFAULT_MAX_RETRIES


@dataclass
class BaseWorkflowConfig:
    """Basis-Konfiguration für Workflows.

    Diese Klasse definiert gemeinsame Konfigurationsparameter, die von
    verschiedenen Workflow-Komponenten verwendet werden.

    Attributes:
        max_retries: Maximale Anzahl von Wiederholungsversuchen
        max_dynamic_nodes: Maximale Anzahl dynamischer Nodes
    """
    max_retries: int = DEFAULT_MAX_RETRIES
    max_dynamic_nodes: int = DEFAULT_MAX_DYNAMIC_NODES


class BaseWorkflowBuilder(ABC):
    """Abstrakte Basis-Klasse für Workflow-Builder.

    Diese Klasse definiert das gemeinsame Interface für alle Workflow-Builder
    und stellt sicher, dass alle Implementierungen die notwendigen Methoden
    bereitstellen.
    """

    def __init__(self, config: BaseWorkflowConfig | None = None) -> None:
        """Initialisiert den Workflow-Builder.

        Args:
            config: Optionale Konfiguration (Standard: BaseWorkflowConfig())
        """
        self.config = config or BaseWorkflowConfig()

    @abstractmethod
    def build(self) -> dict[str, Any]:
        """Baut die Workflow-Repräsentation.

        Returns:
            Serialisierbare Workflow-Struktur mit Nodes und Edges
        """

    @abstractmethod
    def export_to_dot(self) -> str:
        """Exportiert den Workflow in DOT-Format.

        Returns:
            DOT-formatierter String für Graphviz
        """

    @abstractmethod
    def export_to_mermaid(self) -> str:
        """Exportiert den Workflow in Mermaid-Format.

        Returns:
            Mermaid-formatierter String für Diagramme
        """


class BaseNodeFactory(ABC):
    """Abstrakte Basis-Klasse für Node-Factories.

    Diese Klasse definiert das Interface für Factories, die dynamisch
    Workflow-Nodes erstellen.
    """

    def __init__(self, config: BaseWorkflowConfig | None = None) -> None:
        """Initialisiert die Node-Factory.

        Args:
            config: Optionale Konfiguration
        """
        self.config = config or BaseWorkflowConfig()

    @abstractmethod
    async def create_nodes(self, **kwargs: Any) -> Any:
        """Erstellt dynamische Nodes.

        Args:
            **kwargs: Factory-spezifische Parameter

        Returns:
            Erstellte Nodes und Edges
        """


class BaseWorkflowVisualizer(ABC):
    """Abstrakte Basis-Klasse für Workflow-Visualizer.

    Diese Klasse definiert das Interface für Komponenten, die Workflows
    in verschiedene Visualisierungsformate exportieren.
    """

    def __init__(self, graph: dict[str, Any]) -> None:
        """Initialisiert den Visualizer.

        Args:
            graph: Workflow-Graph-Struktur
        """
        self.graph = graph

    @abstractmethod
    def export_to_dot(self) -> str:
        """Exportiert in DOT-Format."""

    @abstractmethod
    def export_to_mermaid(self) -> str:
        """Exportiert in Mermaid-Format."""

    @abstractmethod
    def generate_html_preview(self, **kwargs: Any) -> str:
        """Generiert HTML-Vorschau."""


class BaseBridge(ABC):
    """Abstrakte Basis-Klasse für Workflow-Bridges.

    Diese Klasse definiert das Interface für Bridges, die Workflows
    mit externen Systemen verbinden.
    """

    def __init__(self, config: Any | None = None) -> None:
        """Initialisiert die Bridge.

        Args:
            config: Bridge-spezifische Konfiguration
        """
        self.config = config

    @abstractmethod
    async def trigger(self, **kwargs: Any) -> Any:
        """Triggert eine externe Operation.

        Args:
            **kwargs: Bridge-spezifische Parameter

        Returns:
            Ergebnis der Operation
        """


@dataclass
class WorkflowMetrics:
    """Metriken für Workflow-Performance und -Qualität.

    Diese Klasse sammelt wichtige Metriken über Workflow-Ausführungen
    für Monitoring und Optimierung.

    Attributes:
        execution_time: Ausführungszeit in Sekunden
        node_count: Anzahl der Nodes im Workflow
        edge_count: Anzahl der Edges im Workflow
        error_count: Anzahl der aufgetretenen Fehler
        retry_count: Anzahl der Wiederholungsversuche
    """
    execution_time: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    error_count: int = 0
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert die Metriken in ein Dictionary.

        Returns:
            Dictionary mit allen Metrik-Werten
        """
        return {
            "execution_time": self.execution_time,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
        }


class WorkflowValidator:
    """Validator für Workflow-Strukturen.

    Diese Klasse stellt Methoden zur Validierung von Workflow-Definitionen
    bereit, um Fehler frühzeitig zu erkennen.
    """

    @staticmethod
    def validate_graph_structure(graph: dict[str, Any]) -> list[str]:
        """Validiert die Grundstruktur eines Workflow-Graphs.

        Args:
            graph: Zu validierender Graph

        Returns:
            Liste von Validierungsfehlern (leer wenn gültig)
        """
        errors: list[str] = []

        # Validiere Basis-Struktur
        errors.extend(WorkflowValidator._validate_basic_structure(graph))
        if errors:
            return errors

        # Validiere Nodes und Edges
        nodes = graph["nodes"]
        edges = graph["edges"]
        errors.extend(WorkflowValidator._validate_nodes_and_edges_types(nodes, edges))

        # Validiere Edge-Referenzen
        node_ids = WorkflowValidator._extract_node_ids(nodes)
        errors.extend(WorkflowValidator._validate_edge_references(edges, node_ids))

        return errors

    @staticmethod
    def _validate_basic_structure(graph: dict[str, Any]) -> list[str]:
        """Validiert die Basis-Struktur des Graphs."""
        errors: list[str] = []

        if "nodes" not in graph:
            errors.append("Graph muss 'nodes' enthalten")
        if "edges" not in graph:
            errors.append("Graph muss 'edges' enthalten")

        return errors

    @staticmethod
    def _validate_nodes_and_edges_types(nodes: Any, edges: Any) -> list[str]:
        """Validiert die Typen von Nodes und Edges."""
        errors: list[str] = []

        if not isinstance(nodes, list):
            errors.append("'nodes' muss eine Liste sein")
        if not isinstance(edges, list):
            errors.append("'edges' muss eine Liste sein")

        return errors

    @staticmethod
    def _extract_node_ids(nodes: list[Any]) -> set[str]:
        """Extrahiert alle Node-IDs aus der Nodes-Liste."""
        node_ids = set()
        for node in nodes:
            if isinstance(node, str):
                node_ids.add(node)
            elif isinstance(node, dict) and "id" in node:
                node_ids.add(node["id"])
        return node_ids

    @staticmethod
    def _validate_edge_references(edges: list[Any], node_ids: set[str]) -> list[str]:
        """Validiert, dass alle Edge-Referenzen auf existierende Nodes zeigen."""
        errors: list[str] = []

        for edge in edges:
            if isinstance(edge, tuple | list) and len(edge) >= 2:
                source, target = edge[0], edge[1]
                errors.extend(WorkflowValidator._check_edge_endpoints(source, target, node_ids))
            elif isinstance(edge, dict):
                source = edge.get("source") or edge.get("from")
                target = edge.get("target") or edge.get("to")
                if source and target:
                    errors.extend(WorkflowValidator._check_edge_endpoints(source, target, node_ids))

        return errors

    @staticmethod
    def _check_edge_endpoints(source: str, target: str, node_ids: set[str]) -> list[str]:
        """Prüft, ob Edge-Endpunkte in den verfügbaren Nodes existieren."""
        errors: list[str] = []

        if source not in node_ids:
            errors.append(f"Edge-Source '{source}' nicht in Nodes gefunden")
        if target not in node_ids:
            errors.append(f"Edge-Target '{target}' nicht in Nodes gefunden")

        return errors

    @staticmethod
    def validate_node_types(nodes: list[Any]) -> list[str]:
        """Validiert Node-Typen.

        Args:
            nodes: Liste der Nodes

        Returns:
            Liste von Validierungsfehlern
        """
        errors: list[str] = []

        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                if "id" not in node:
                    errors.append(f"Node {i} fehlt 'id' Feld")
                if "type" in node:
                    node_type = node["type"]
                    from .workflows_constants import NODE_TYPES
                    valid_types = list(NODE_TYPES.values())
                    if node_type not in valid_types:
                        errors.append(f"Node {i} hat ungültigen Typ '{node_type}'")

        return errors


__all__ = [
    "BaseBridge",
    "BaseNodeFactory",
    "BaseWorkflowBuilder",
    "BaseWorkflowConfig",
    "BaseWorkflowVisualizer",
    "WorkflowMetrics",
    "WorkflowValidator",
]
