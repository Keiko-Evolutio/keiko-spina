# backend/services/enhanced_dependency_resolution/dependency_graph_engine.py
"""Enhanced Dependency Graph Engine.

Implementiert Intelligent Dependency Graph Analysis mit automatischer
Dependency-Detection und Enterprise-Grade Graph-Management.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from kei_logging import get_logger

from .data_models import (
    CircularResolutionStrategy,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    DependencyRelation,
    DependencyType,
    ResolutionStrategy,
)

logger = get_logger(__name__)


class EnhancedDependencyGraphEngine:
    """Enhanced Dependency Graph Engine für Intelligent Graph Analysis."""

    def __init__(self):
        """Initialisiert Enhanced Dependency Graph Engine."""
        # Graph-Storage
        self._graphs: dict[str, DependencyGraph] = {}

        # Graph-Analysis-Konfiguration
        self.enable_automatic_detection = True
        self.enable_circular_detection = True
        self.enable_graph_optimization = True
        self.max_graph_depth = 50

        # Performance-Tracking
        self._graph_analysis_count = 0
        self._total_graph_analysis_time_ms = 0.0
        self._circular_detection_count = 0
        self._total_circular_detection_time_ms = 0.0

        # Graph-Cache
        self._analysis_cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl_seconds = 300  # 5 Minuten
        self._cache_timestamps: dict[str, float] = {}

        logger.info("Enhanced Dependency Graph Engine initialisiert")

    async def create_dependency_graph(
        self,
        graph_id: str,
        name: str,
        description: str,
        resolution_strategy: ResolutionStrategy = ResolutionStrategy.EAGER,
        circular_resolution_strategy: CircularResolutionStrategy = CircularResolutionStrategy.BREAK_WEAKEST
    ) -> DependencyGraph:
        """Erstellt neuen Dependency-Graph.

        Args:
            graph_id: Graph-ID
            name: Graph-Name
            description: Graph-Beschreibung
            resolution_strategy: Resolution-Strategie
            circular_resolution_strategy: Circular-Resolution-Strategie

        Returns:
            Dependency-Graph
        """
        try:
            graph = DependencyGraph(
                graph_id=graph_id,
                name=name,
                description=description,
                resolution_strategy=resolution_strategy,
                circular_resolution_strategy=circular_resolution_strategy
            )

            self._graphs[graph_id] = graph

            logger.info({
                "event": "dependency_graph_created",
                "graph_id": graph_id,
                "name": name,
                "resolution_strategy": resolution_strategy.value
            })

            return graph

        except Exception as e:
            logger.error(f"Dependency graph creation fehlgeschlagen: {e}")
            raise

    async def add_dependency_node(
        self,
        graph_id: str,
        node_id: str,
        node_type: DependencyType,
        name: str,
        description: str,
        version: str | None = None,
        priority: int = 0,
        metadata: dict[str, Any] | None = None
    ) -> DependencyNode:
        """Fügt Dependency-Node zu Graph hinzu.

        Args:
            graph_id: Graph-ID
            node_id: Node-ID
            node_type: Node-Type
            name: Node-Name
            description: Node-Beschreibung
            version: Node-Version
            priority: Node-Priorität
            metadata: Node-Metadata

        Returns:
            Dependency-Node
        """
        try:
            graph = self._graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph {graph_id} nicht gefunden")

            node = DependencyNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                description=description,
                version=version,
                priority=priority,
                metadata=metadata or {}
            )

            graph.nodes[node_id] = node
            graph.updated_at = datetime.utcnow()

            # Invalidiere Cache
            self._invalidate_graph_cache(graph_id)

            logger.debug({
                "event": "dependency_node_added",
                "graph_id": graph_id,
                "node_id": node_id,
                "node_type": node_type.value
            })

            return node

        except Exception as e:
            logger.error(f"Dependency node addition fehlgeschlagen: {e}")
            raise

    async def add_dependency_edge(
        self,
        graph_id: str,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        relation: DependencyRelation,
        dependency_type: DependencyType,
        weight: float = 1.0,
        version_constraint: str | None = None,
        condition: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> DependencyEdge:
        """Fügt Dependency-Edge zu Graph hinzu.

        Args:
            graph_id: Graph-ID
            edge_id: Edge-ID
            source_node_id: Source-Node-ID
            target_node_id: Target-Node-ID
            relation: Dependency-Relation
            dependency_type: Dependency-Type
            weight: Edge-Weight
            version_constraint: Version-Constraint
            condition: Condition
            metadata: Edge-Metadata

        Returns:
            Dependency-Edge
        """
        try:
            graph = self._graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph {graph_id} nicht gefunden")

            # Prüfe ob Nodes existieren
            if source_node_id not in graph.nodes:
                raise ValueError(f"Source node {source_node_id} nicht gefunden")
            if target_node_id not in graph.nodes:
                raise ValueError(f"Target node {target_node_id} nicht gefunden")

            edge = DependencyEdge(
                edge_id=edge_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relation=relation,
                dependency_type=dependency_type,
                weight=weight,
                version_constraint=version_constraint,
                condition=condition,
                metadata=metadata or {}
            )

            graph.edges[edge_id] = edge

            # Aktualisiere Node-Dependencies
            source_node = graph.nodes[source_node_id]
            target_node = graph.nodes[target_node_id]

            if relation == DependencyRelation.REQUIRES:
                source_node.dependencies.add(target_node_id)
                target_node.dependents.add(source_node_id)
            elif relation == DependencyRelation.PROVIDES:
                target_node.dependencies.add(source_node_id)
                source_node.dependents.add(target_node_id)

            graph.updated_at = datetime.utcnow()

            # Invalidiere Cache
            self._invalidate_graph_cache(graph_id)

            # Prüfe auf Circular Dependencies
            if self.enable_circular_detection:
                await self._check_circular_dependencies(graph)

            logger.debug({
                "event": "dependency_edge_added",
                "graph_id": graph_id,
                "edge_id": edge_id,
                "source": source_node_id,
                "target": target_node_id,
                "relation": relation.value
            })

            return edge

        except Exception as e:
            logger.error(f"Dependency edge addition fehlgeschlagen: {e}")
            raise

    async def analyze_dependency_graph(
        self,
        graph_id: str,
        include_circular_detection: bool = True,
        include_optimization: bool = True
    ) -> dict[str, Any]:
        """Analysiert Dependency-Graph.

        Args:
            graph_id: Graph-ID
            include_circular_detection: Circular-Detection einschließen
            include_optimization: Graph-Optimierung einschließen

        Returns:
            Graph-Analysis-Result
        """
        start_time = time.time()

        try:
            graph = self._graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph {graph_id} nicht gefunden")

            logger.debug({
                "event": "dependency_graph_analysis_started",
                "graph_id": graph_id,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges)
            })

            # Cache-Check
            cache_key = f"{graph_id}_{include_circular_detection}_{include_optimization}"
            cached_result = self._get_cached_analysis(cache_key)
            if cached_result:
                return cached_result

            analysis_result = {
                "graph_id": graph_id,
                "nodes_count": len(graph.nodes),
                "edges_count": len(graph.edges),
                "is_acyclic": True,
                "circular_dependencies": [],
                "topological_order": [],
                "graph_depth": 0,
                "strongly_connected_components": [],
                "optimization_suggestions": []
            }

            # 1. Topological Sort
            topological_order = await self._topological_sort(graph)
            analysis_result["topological_order"] = topological_order

            # 2. Graph-Depth-Analysis
            graph_depth = await self._calculate_graph_depth(graph)
            analysis_result["graph_depth"] = graph_depth

            # 3. Circular-Dependency-Detection
            if include_circular_detection:
                circular_deps = await self._detect_circular_dependencies(graph)
                analysis_result["circular_dependencies"] = circular_deps
                analysis_result["is_acyclic"] = len(circular_deps) == 0

                # Aktualisiere Graph-Status
                graph.has_circular_dependencies = len(circular_deps) > 0
                graph.circular_cycles = [[node.node_id for node in cycle] for cycle in circular_deps]
                graph.is_acyclic = len(circular_deps) == 0

            # 4. Strongly Connected Components
            scc = await self._find_strongly_connected_components(graph)
            analysis_result["strongly_connected_components"] = scc

            # 5. Graph-Optimierung
            if include_optimization:
                optimization_suggestions = await self._generate_optimization_suggestions(graph)
                analysis_result["optimization_suggestions"] = optimization_suggestions

            # Cache Result
            self._cache_analysis_result(cache_key, analysis_result)

            # Performance-Tracking
            analysis_time_ms = (time.time() - start_time) * 1000
            self._update_graph_analysis_performance_stats(analysis_time_ms)

            logger.debug({
                "event": "dependency_graph_analysis_completed",
                "graph_id": graph_id,
                "is_acyclic": analysis_result["is_acyclic"],
                "circular_dependencies": len(analysis_result["circular_dependencies"]),
                "analysis_time_ms": analysis_time_ms
            })

            return analysis_result

        except Exception as e:
            logger.error(f"Dependency graph analysis fehlgeschlagen: {e}")
            raise

    async def detect_automatic_dependencies(
        self,
        graph_id: str,
        context: dict[str, Any]
    ) -> list[DependencyEdge]:
        """Erkennt automatisch Dependencies basierend auf Kontext.

        Args:
            graph_id: Graph-ID
            context: Kontext für Dependency-Detection

        Returns:
            Liste von automatisch erkannten Dependencies
        """
        try:
            if not self.enable_automatic_detection:
                return []

            graph = self._graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph {graph_id} nicht gefunden")

            detected_dependencies = []

            # 1. Task-Dependencies basierend auf Task-Decomposition
            if "task_context" in context:
                task_deps = await self._detect_task_dependencies(graph, context["task_context"])
                detected_dependencies.extend(task_deps)

            # 2. Resource-Dependencies basierend auf Resource-Requirements
            if "resource_context" in context:
                resource_deps = await self._detect_resource_dependencies(graph, context["resource_context"])
                detected_dependencies.extend(resource_deps)

            # 3. Agent-Dependencies basierend auf Capability-Requirements
            if "agent_context" in context:
                agent_deps = await self._detect_agent_dependencies(graph, context["agent_context"])
                detected_dependencies.extend(agent_deps)

            # 4. Service-Dependencies basierend auf API-Dependencies
            if "service_context" in context:
                service_deps = await self._detect_service_dependencies(graph, context["service_context"])
                detected_dependencies.extend(service_deps)

            logger.info({
                "event": "automatic_dependencies_detected",
                "graph_id": graph_id,
                "detected_count": len(detected_dependencies)
            })

            return detected_dependencies

        except Exception as e:
            logger.error(f"Automatic dependency detection fehlgeschlagen: {e}")
            return []

    async def _topological_sort(self, graph: DependencyGraph) -> list[str]:
        """Führt Topological Sort durch."""
        try:
            # Kahn's Algorithm
            in_degree = defaultdict(int)

            # Berechne In-Degrees
            for edge in graph.edges.values():
                if edge.relation == DependencyRelation.REQUIRES:
                    in_degree[edge.source_node_id] += 1

            # Initialisiere Queue mit Nodes ohne Dependencies
            queue = deque([node_id for node_id in graph.nodes.keys() if in_degree[node_id] == 0])
            result = []

            while queue:
                node_id = queue.popleft()
                result.append(node_id)

                # Reduziere In-Degree für abhängige Nodes
                node = graph.nodes[node_id]
                for dependent_id in node.dependents:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)

            # Prüfe auf Circular Dependencies
            if len(result) != len(graph.nodes):
                logger.warning("Topological sort incomplete - possible circular dependencies")

            return result

        except Exception as e:
            logger.error(f"Topological sort fehlgeschlagen: {e}")
            return []

    async def _calculate_graph_depth(self, graph: DependencyGraph) -> int:
        """Berechnet Graph-Depth."""
        try:
            if not graph.nodes:
                return 0

            # DFS für maximale Depth
            visited = set()
            max_depth = 0

            async def dfs(node_id: str, current_depth: int) -> int:
                if node_id in visited:
                    return current_depth

                visited.add(node_id)
                node = graph.nodes[node_id]

                max_child_depth = current_depth
                for dep_id in node.dependencies:
                    if dep_id in graph.nodes:
                        child_depth = await dfs(dep_id, current_depth + 1)
                        max_child_depth = max(max_child_depth, child_depth)

                return max_child_depth

            # Finde Root-Nodes (ohne Dependencies)
            root_nodes = [
                node_id for node_id, node in graph.nodes.items()
                if not node.dependencies
            ]

            if not root_nodes:
                # Fallback: Alle Nodes als potenzielle Roots
                root_nodes = list(graph.nodes.keys())

            for root_id in root_nodes:
                depth = await dfs(root_id, 0)
                max_depth = max(max_depth, depth)

            return max_depth

        except Exception as e:
            logger.error(f"Graph depth calculation fehlgeschlagen: {e}")
            return 0

    async def _detect_circular_dependencies(self, graph: DependencyGraph) -> list[list[DependencyNode]]:
        """Erkennt Circular Dependencies."""
        start_time = time.time()

        try:
            circular_cycles = []
            visited = set()
            rec_stack = set()

            async def dfs(node_id: str, path: list[str]) -> None:
                if node_id in rec_stack:
                    # Circular Dependency gefunden
                    cycle_start = path.index(node_id)
                    cycle = path[cycle_start:] + [node_id]
                    cycle_nodes = [graph.nodes[nid] for nid in cycle if nid in graph.nodes]
                    circular_cycles.append(cycle_nodes)
                    return

                if node_id in visited:
                    return

                visited.add(node_id)
                rec_stack.add(node_id)

                node = graph.nodes.get(node_id)
                if node:
                    for dep_id in node.dependencies:
                        if dep_id in graph.nodes:
                            await dfs(dep_id, path + [node_id])

                rec_stack.remove(node_id)

            # Prüfe alle Nodes
            for node_id in graph.nodes.keys():
                if node_id not in visited:
                    await dfs(node_id, [])

            # Performance-Tracking
            detection_time_ms = (time.time() - start_time) * 1000
            self._update_circular_detection_performance_stats(detection_time_ms)

            if circular_cycles:
                logger.warning({
                    "event": "circular_dependencies_detected",
                    "graph_id": graph.graph_id,
                    "cycles_count": len(circular_cycles),
                    "detection_time_ms": detection_time_ms
                })

            return circular_cycles

        except Exception as e:
            logger.error(f"Circular dependency detection fehlgeschlagen: {e}")
            return []

    async def _check_circular_dependencies(self, graph: DependencyGraph) -> None:
        """Prüft Graph auf Circular Dependencies."""
        try:
            circular_deps = await self._detect_circular_dependencies(graph)

            if circular_deps:
                graph.has_circular_dependencies = True
                graph.circular_cycles = [[node.node_id for node in cycle] for cycle in circular_deps]
                graph.is_acyclic = False

                logger.warning({
                    "event": "circular_dependencies_found",
                    "graph_id": graph.graph_id,
                    "cycles": len(circular_deps)
                })
            else:
                graph.has_circular_dependencies = False
                graph.circular_cycles = []
                graph.is_acyclic = True

        except Exception as e:
            logger.error(f"Circular dependency check fehlgeschlagen: {e}")

    async def _find_strongly_connected_components(self, graph: DependencyGraph) -> list[list[str]]:
        """Findet Strongly Connected Components (Tarjan's Algorithm)."""
        try:
            index_counter = [0]
            stack = []
            lowlinks = {}
            index = {}
            on_stack = {}
            components = []

            def strongconnect(node_id: str):
                index[node_id] = index_counter[0]
                lowlinks[node_id] = index_counter[0]
                index_counter[0] += 1
                stack.append(node_id)
                on_stack[node_id] = True

                node = graph.nodes.get(node_id)
                if node:
                    for dep_id in node.dependencies:
                        if dep_id in graph.nodes:
                            if dep_id not in index:
                                strongconnect(dep_id)
                                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[dep_id])
                            elif on_stack.get(dep_id, False):
                                lowlinks[node_id] = min(lowlinks[node_id], index[dep_id])

                if lowlinks[node_id] == index[node_id]:
                    component = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        component.append(w)
                        if w == node_id:
                            break
                    components.append(component)

            for node_id in graph.nodes.keys():
                if node_id not in index:
                    strongconnect(node_id)

            return components

        except Exception as e:
            logger.error(f"Strongly connected components detection fehlgeschlagen: {e}")
            return []

    async def _generate_optimization_suggestions(self, graph: DependencyGraph) -> list[str]:
        """Generiert Graph-Optimierung-Suggestions."""
        try:
            suggestions = []

            # 1. Redundante Dependencies
            redundant_deps = await self._find_redundant_dependencies(graph)
            if redundant_deps:
                suggestions.append(f"Entferne {len(redundant_deps)} redundante Dependencies")

            # 2. Lange Dependency-Chains
            long_chains = await self._find_long_dependency_chains(graph)
            if long_chains:
                suggestions.append(f"Optimiere {len(long_chains)} lange Dependency-Chains")

            # 3. High-Degree Nodes
            high_degree_nodes = await self._find_high_degree_nodes(graph)
            if high_degree_nodes:
                suggestions.append(f"Refaktoriere {len(high_degree_nodes)} Nodes mit vielen Dependencies")

            return suggestions

        except Exception as e:
            logger.error(f"Optimization suggestions generation fehlgeschlagen: {e}")
            return []

    async def _detect_task_dependencies(self, _graph: DependencyGraph, _task_context: dict[str, Any]) -> list[DependencyEdge]:
        """Erkennt Task-Dependencies."""
        # TODO: Implementiere Task-Dependency-Detection - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/118
        return []

    async def _detect_resource_dependencies(self, _graph: DependencyGraph, _resource_context: dict[str, Any]) -> list[DependencyEdge]:
        """Erkennt Resource-Dependencies."""
        # TODO: Implementiere Resource-Dependency-Detection - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/118
        return []

    async def _detect_agent_dependencies(self, _graph: DependencyGraph, _agent_context: dict[str, Any]) -> list[DependencyEdge]:
        """Erkennt Agent-Dependencies."""
        # TODO: Implementiere Agent-Dependency-Detection - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/118
        return []

    async def _detect_service_dependencies(self, _graph: DependencyGraph, _service_context: dict[str, Any]) -> list[DependencyEdge]:
        """Erkennt Service-Dependencies."""
        # TODO: Implementiere Service-Dependency-Detection - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/118
        return []

    async def _find_redundant_dependencies(self, _graph: DependencyGraph) -> list[str]:
        """Findet redundante Dependencies."""
        # TODO: Implementiere Redundant-Dependency-Detection - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/118
        return []

    async def _find_long_dependency_chains(self, _graph: DependencyGraph) -> list[list[str]]:
        """Findet lange Dependency-Chains."""
        # TODO: Implementiere Long-Chain-Detection - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/118
        return []

    async def _find_high_degree_nodes(self, graph: DependencyGraph) -> list[str]:
        """Findet Nodes mit vielen Dependencies."""
        try:
            high_degree_nodes = []
            threshold = 10  # Mehr als 10 Dependencies

            for node_id, node in graph.nodes.items():
                total_degree = len(node.dependencies) + len(node.dependents)
                if total_degree > threshold:
                    high_degree_nodes.append(node_id)

            return high_degree_nodes

        except Exception as e:
            logger.error(f"High degree nodes detection fehlgeschlagen: {e}")
            return []

    def _get_cached_analysis(self, cache_key: str) -> dict[str, Any] | None:
        """Holt Cached Analysis-Result."""
        try:
            if cache_key not in self._analysis_cache:
                return None

            # Prüfe TTL
            if cache_key in self._cache_timestamps:
                age = time.time() - self._cache_timestamps[cache_key]
                if age > self._cache_ttl_seconds:
                    del self._analysis_cache[cache_key]
                    del self._cache_timestamps[cache_key]
                    return None

            return self._analysis_cache[cache_key]

        except Exception as e:
            logger.error(f"Cache lookup fehlgeschlagen: {e}")
            return None

    def _cache_analysis_result(self, cache_key: str, result: dict[str, Any]) -> None:
        """Cached Analysis-Result."""
        try:
            self._analysis_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

            # Memory-Limit prüfen
            if len(self._analysis_cache) > 1000:
                # Entferne älteste Einträge
                oldest_keys = sorted(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k]
                )[:100]

                for key in oldest_keys:
                    del self._analysis_cache[key]
                    del self._cache_timestamps[key]

        except Exception as e:
            logger.error(f"Cache storage fehlgeschlagen: {e}")

    def _invalidate_graph_cache(self, graph_id: str) -> None:
        """Invalidiert Cache für Graph."""
        try:
            keys_to_remove = [
                key for key in self._analysis_cache.keys()
                if key.startswith(graph_id)
            ]

            for key in keys_to_remove:
                del self._analysis_cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]

        except Exception as e:
            logger.error(f"Cache invalidation fehlgeschlagen: {e}")

    def _update_graph_analysis_performance_stats(self, analysis_time_ms: float) -> None:
        """Aktualisiert Graph-Analysis-Performance-Statistiken."""
        self._graph_analysis_count += 1
        self._total_graph_analysis_time_ms += analysis_time_ms

    def _update_circular_detection_performance_stats(self, detection_time_ms: float) -> None:
        """Aktualisiert Circular-Detection-Performance-Statistiken."""
        self._circular_detection_count += 1
        self._total_circular_detection_time_ms += detection_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_analysis_time = (
            self._total_graph_analysis_time_ms / self._graph_analysis_count
            if self._graph_analysis_count > 0 else 0.0
        )

        avg_circular_detection_time = (
            self._total_circular_detection_time_ms / self._circular_detection_count
            if self._circular_detection_count > 0 else 0.0
        )

        return {
            "total_graph_analyses": self._graph_analysis_count,
            "avg_graph_analysis_time_ms": avg_analysis_time,
            "total_circular_detections": self._circular_detection_count,
            "avg_circular_detection_time_ms": avg_circular_detection_time,
            "managed_graphs": len(self._graphs),
            "cache_size": len(self._analysis_cache),
            "automatic_detection_enabled": self.enable_automatic_detection,
            "circular_detection_enabled": self.enable_circular_detection,
            "graph_optimization_enabled": self.enable_graph_optimization
        }
