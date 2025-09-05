# backend/services/enhanced_dependency_resolution/circular_dependency_engine.py
"""Circular Dependency Detection und Resolution Engine.

Implementiert Advanced Circular Dependency Detection mit verschiedenen
Resolution-Strategien und Performance-Optimierung.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from kei_logging import get_logger

from .data_models import (
    CircularDependency,
    CircularResolutionStrategy,
    DependencyGraph,
    DependencyNode,
    DependencyRelation,
    DependencyStatus,
)

logger = get_logger(__name__)


class CircularDependencyEngine:
    """Circular Dependency Detection und Resolution Engine."""

    def __init__(self):
        """Initialisiert Circular Dependency Engine."""
        # Detection-Konfiguration
        self.enable_advanced_detection = True
        self.enable_weak_cycle_detection = True
        self.enable_resolution_suggestions = True
        self.max_cycle_length = 20

        # Detection-Algorithmen
        self.detection_algorithms = [
            "dfs_based",
            "tarjan_scc",
            "johnson_cycles",
            "floyd_warshall"
        ]
        self.default_algorithm = "dfs_based"

        # Resolution-Strategien
        self.resolution_strategies = {
            CircularResolutionStrategy.BREAK_WEAKEST: self._break_weakest_strategy,
            CircularResolutionStrategy.BREAK_OPTIONAL: self._break_optional_strategy,
            CircularResolutionStrategy.MERGE_CYCLES: self._merge_cycles_strategy,
            CircularResolutionStrategy.FAIL_FAST: self._fail_fast_strategy,
            CircularResolutionStrategy.IGNORE: self._ignore_strategy
        }

        # Performance-Tracking
        self._detection_count = 0
        self._total_detection_time_ms = 0.0
        self._cycles_detected = 0
        self._cycles_resolved = 0

        # Detection-Cache
        self._detection_cache: dict[str, list[CircularDependency]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl_seconds = 300  # 5 Minuten

        logger.info("Circular Dependency Engine initialisiert")

    async def detect_circular_dependencies(
        self,
        graph: DependencyGraph,
        algorithm: str = "dfs_based",
        use_cache: bool = True
    ) -> list[CircularDependency]:
        """Erkennt Circular Dependencies im Graph.

        Args:
            graph: Dependency-Graph
            algorithm: Detection-Algorithmus
            use_cache: Cache verwenden

        Returns:
            Liste von Circular Dependencies
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "circular_dependency_detection_started",
                "graph_id": graph.graph_id,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "algorithm": algorithm
            })

            # Cache-Check
            if use_cache:
                cached_result = self._get_cached_detection(graph.graph_id)
                if cached_result is not None:
                    return cached_result

            # Wähle Detection-Algorithmus
            if algorithm == "dfs_based":
                cycles = await self._detect_cycles_dfs(graph)
            elif algorithm == "tarjan_scc":
                cycles = await self._detect_cycles_tarjan(graph)
            elif algorithm == "johnson_cycles":
                cycles = await self._detect_cycles_johnson(graph)
            elif algorithm == "floyd_warshall":
                cycles = await self._detect_cycles_floyd_warshall(graph)
            else:
                logger.warning(f"Unbekannter Algorithmus {algorithm}, verwende DFS")
                cycles = await self._detect_cycles_dfs(graph)

            # Konvertiere zu CircularDependency-Objekten
            circular_dependencies = []
            for i, cycle in enumerate(cycles):
                circular_dep = await self._create_circular_dependency(
                    graph, cycle, f"cycle_{i}_{graph.graph_id}"
                )
                circular_dependencies.append(circular_dep)

            # Cache Result
            if use_cache:
                self._cache_detection_result(graph.graph_id, circular_dependencies)

            # Performance-Tracking
            detection_time_ms = (time.time() - start_time) * 1000
            self._update_detection_performance_stats(detection_time_ms, len(circular_dependencies))

            logger.debug({
                "event": "circular_dependency_detection_completed",
                "graph_id": graph.graph_id,
                "cycles_detected": len(circular_dependencies),
                "detection_time_ms": detection_time_ms,
                "algorithm": algorithm
            })

            return circular_dependencies

        except Exception as e:
            logger.error(f"Circular dependency detection fehlgeschlagen: {e}")
            return []

    async def resolve_circular_dependencies(
        self,
        graph: DependencyGraph,
        circular_dependencies: list[CircularDependency],
        strategy: CircularResolutionStrategy = CircularResolutionStrategy.BREAK_WEAKEST
    ) -> dict[str, Any]:
        """Löst Circular Dependencies auf.

        Args:
            graph: Dependency-Graph
            circular_dependencies: Circular Dependencies
            strategy: Resolution-Strategie

        Returns:
            Resolution-Result
        """
        try:
            logger.debug({
                "event": "circular_dependency_resolution_started",
                "graph_id": graph.graph_id,
                "cycles": len(circular_dependencies),
                "strategy": strategy.value
            })

            resolution_handler = self.resolution_strategies.get(strategy)
            if not resolution_handler:
                raise ValueError(f"Unbekannte Resolution-Strategie: {strategy}")

            # Führe Resolution durch
            resolution_result = await resolution_handler(graph, circular_dependencies)

            # Aktualisiere Performance-Stats
            self._cycles_resolved += len(circular_dependencies)

            logger.debug({
                "event": "circular_dependency_resolution_completed",
                "graph_id": graph.graph_id,
                "strategy": strategy.value,
                "success": resolution_result["success"],
                "broken_edges": len(resolution_result.get("broken_edges", []))
            })

            return resolution_result

        except Exception as e:
            logger.error(f"Circular dependency resolution fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e),
                "broken_edges": [],
                "modified_nodes": []
            }

    async def _detect_cycles_dfs(self, graph: DependencyGraph) -> list[list[str]]:
        """DFS-basierte Cycle-Detection."""
        try:
            cycles = []
            visited = set()
            rec_stack = set()
            path = []

            async def dfs(node_id: str) -> None:
                if node_id in rec_stack:
                    # Cycle gefunden
                    cycle_start = path.index(node_id)
                    cycle = path[cycle_start:] + [node_id]

                    # Prüfe Cycle-Länge
                    if len(cycle) <= self.max_cycle_length:
                        cycles.append(cycle[:-1])  # Entferne Duplikat am Ende
                    return

                if node_id in visited:
                    return

                visited.add(node_id)
                rec_stack.add(node_id)
                path.append(node_id)

                # Besuche Dependencies
                node = graph.nodes.get(node_id)
                if node:
                    for dep_id in node.dependencies:
                        if dep_id in graph.nodes:
                            await dfs(dep_id)

                rec_stack.remove(node_id)
                path.pop()

            # Starte DFS für alle Nodes
            for node_id in graph.nodes.keys():
                if node_id not in visited:
                    await dfs(node_id)

            return cycles

        except Exception as e:
            logger.error(f"DFS cycle detection fehlgeschlagen: {e}")
            return []

    async def _detect_cycles_tarjan(self, graph: DependencyGraph) -> list[list[str]]:
        """Tarjan's Strongly Connected Components Algorithmus."""
        try:
            index_counter = [0]
            stack = []
            lowlinks = {}
            index = {}
            on_stack = {}
            sccs = []

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

                    # Nur SCCs mit mehr als einem Node sind Cycles
                    if len(component) > 1:
                        sccs.append(component)

            for node_id in graph.nodes.keys():
                if node_id not in index:
                    strongconnect(node_id)

            return sccs

        except Exception as e:
            logger.error(f"Tarjan cycle detection fehlgeschlagen: {e}")
            return []

    async def _detect_cycles_johnson(self, graph: DependencyGraph) -> list[list[str]]:
        """Johnson's Algorithmus für alle elementaren Cycles."""
        try:
            # Vereinfachte Implementierung von Johnson's Algorithmus
            cycles = []

            # Erstelle Adjacency-Liste
            adj_list = defaultdict(list)
            for node_id, node in graph.nodes.items():
                for dep_id in node.dependencies:
                    if dep_id in graph.nodes:
                        adj_list[node_id].append(dep_id)

            # Finde alle elementaren Cycles
            path = []

            async def find_cycles(start_node: str, current_node: str) -> None:
                if current_node == start_node and len(path) > 1:
                    # Cycle gefunden
                    if len(path) <= self.max_cycle_length:
                        cycles.append(path.copy())
                    return

                if current_node in path:
                    return

                path.append(current_node)

                for neighbor in adj_list[current_node]:
                    await find_cycles(start_node, neighbor)

                path.pop()

            # Starte für jeden Node
            for start_node in graph.nodes.keys():
                path = []
                await find_cycles(start_node, start_node)

            # Entferne Duplikate
            unique_cycles = []
            for cycle in cycles:
                normalized_cycle = self._normalize_cycle(cycle)
                if normalized_cycle not in unique_cycles:
                    unique_cycles.append(normalized_cycle)

            return unique_cycles

        except Exception as e:
            logger.error(f"Johnson cycle detection fehlgeschlagen: {e}")
            return []

    async def _detect_cycles_floyd_warshall(self, graph: DependencyGraph) -> list[list[str]]:
        """Floyd-Warshall-basierte Cycle-Detection."""
        try:
            nodes = list(graph.nodes.keys())
            n = len(nodes)
            node_to_index = {node: i for i, node in enumerate(nodes)}

            # Initialisiere Distanz-Matrix
            dist = [[float("inf")] * n for _ in range(n)]
            next_node = [[None] * n for _ in range(n)]

            # Setze direkte Edges
            for node_id, node in graph.nodes.items():
                i = node_to_index[node_id]
                dist[i][i] = 0

                for dep_id in node.dependencies:
                    if dep_id in graph.nodes:
                        j = node_to_index[dep_id]
                        dist[i][j] = 1
                        next_node[i][j] = j

            # Floyd-Warshall
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                            next_node[i][j] = next_node[i][k]

            # Finde Cycles (negative Cycles auf Diagonal)
            cycles = []
            for i in range(n):
                if dist[i][i] < 0:
                    # Rekonstruiere Cycle
                    cycle = []
                    current = i

                    while True:
                        cycle.append(nodes[current])
                        current = next_node[current][i]

                        if current == i or len(cycle) > self.max_cycle_length:
                            break

                    if len(cycle) > 1 and len(cycle) <= self.max_cycle_length:
                        cycles.append(cycle)

            return cycles

        except Exception as e:
            logger.error(f"Floyd-Warshall cycle detection fehlgeschlagen: {e}")
            return []

    async def _create_circular_dependency(
        self,
        graph: DependencyGraph,
        cycle: list[str],
        cycle_id: str
    ) -> CircularDependency:
        """Erstellt CircularDependency-Objekt."""
        try:
            # Finde Edges im Cycle
            cycle_edges = []
            cycle_weight = 0.0
            weakest_edge_id = None
            min_weight = float("inf")

            for i in range(len(cycle)):
                source_id = cycle[i]
                target_id = cycle[(i + 1) % len(cycle)]

                # Finde Edge zwischen source und target
                for edge_id, edge in graph.edges.items():
                    if edge.source_node_id == source_id and edge.target_node_id == target_id:
                        cycle_edges.append(edge_id)
                        cycle_weight += edge.weight

                        if edge.weight < min_weight:
                            min_weight = edge.weight
                            weakest_edge_id = edge_id
                        break

            # Prüfe ob Cycle gebrochen werden kann
            can_be_broken = weakest_edge_id is not None
            break_cost = min_weight if can_be_broken else float("inf")

            # Prüfe ob Strong Cycle (alle Edges sind REQUIRES)
            is_strong_cycle = True
            for edge_id in cycle_edges:
                edge = graph.edges.get(edge_id)
                if edge and edge.relation != DependencyRelation.REQUIRES:
                    is_strong_cycle = False
                    break

            circular_dependency = CircularDependency(
                cycle_id=cycle_id,
                graph_id=graph.graph_id,
                cycle_nodes=cycle,
                cycle_edges=cycle_edges,
                cycle_length=len(cycle),
                is_strong_cycle=is_strong_cycle,
                cycle_weight=cycle_weight,
                weakest_edge_id=weakest_edge_id,
                can_be_broken=can_be_broken,
                break_cost=break_cost
            )

            return circular_dependency

        except Exception as e:
            logger.error(f"Circular dependency creation fehlgeschlagen: {e}")
            raise

    async def _break_weakest_strategy(
        self,
        graph: DependencyGraph,
        circular_dependencies: list[CircularDependency]
    ) -> dict[str, Any]:
        """Break-Weakest Resolution-Strategie."""
        try:
            broken_edges = []
            modified_nodes = []

            for circular_dep in circular_dependencies:
                if circular_dep.can_be_broken and circular_dep.weakest_edge_id:
                    # Breche schwächste Edge
                    edge = graph.edges.get(circular_dep.weakest_edge_id)
                    if edge:
                        # Markiere Edge als gebrochen
                        edge.status = DependencyStatus.FAILED
                        edge.metadata["broken_by_circular_resolution"] = True
                        edge.metadata["break_strategy"] = "weakest"
                        edge.updated_at = datetime.utcnow()

                        broken_edges.append(circular_dep.weakest_edge_id)

                        # Aktualisiere betroffene Nodes
                        source_node = graph.nodes.get(edge.source_node_id)
                        target_node = graph.nodes.get(edge.target_node_id)

                        if source_node:
                            source_node.dependencies.discard(edge.target_node_id)
                            source_node.updated_at = datetime.utcnow()
                            modified_nodes.append(edge.source_node_id)

                        if target_node:
                            target_node.dependents.discard(edge.source_node_id)
                            target_node.updated_at = datetime.utcnow()
                            modified_nodes.append(edge.target_node_id)

            return {
                "success": True,
                "strategy": "break_weakest",
                "broken_edges": broken_edges,
                "modified_nodes": list(set(modified_nodes)),
                "cycles_resolved": len(circular_dependencies)
            }

        except Exception as e:
            logger.error(f"Break weakest strategy fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    async def _break_optional_strategy(
        self,
        graph: DependencyGraph,
        circular_dependencies: list[CircularDependency]
    ) -> dict[str, Any]:
        """Break-Optional Resolution-Strategie."""
        try:
            broken_edges = []
            modified_nodes = []

            for circular_dep in circular_dependencies:
                # Finde optionale Edges im Cycle
                for edge_id in circular_dep.cycle_edges:
                    edge = graph.edges.get(edge_id)
                    if edge and edge.relation == DependencyRelation.OPTIONAL:
                        # Breche optionale Edge
                        edge.status = DependencyStatus.FAILED
                        edge.metadata["broken_by_circular_resolution"] = True
                        edge.metadata["break_strategy"] = "optional"
                        edge.updated_at = datetime.utcnow()

                        broken_edges.append(edge_id)

                        # Aktualisiere betroffene Nodes
                        source_node = graph.nodes.get(edge.source_node_id)
                        target_node = graph.nodes.get(edge.target_node_id)

                        if source_node:
                            source_node.dependencies.discard(edge.target_node_id)
                            source_node.updated_at = datetime.utcnow()
                            modified_nodes.append(edge.source_node_id)

                        if target_node:
                            target_node.dependents.discard(edge.source_node_id)
                            target_node.updated_at = datetime.utcnow()
                            modified_nodes.append(edge.target_node_id)

                        break  # Nur eine optionale Edge pro Cycle brechen

            return {
                "success": True,
                "strategy": "break_optional",
                "broken_edges": broken_edges,
                "modified_nodes": list(set(modified_nodes)),
                "cycles_resolved": len([cd for cd in circular_dependencies if any(
                    graph.edges.get(edge_id, {}).get("relation") == DependencyRelation.OPTIONAL
                    for edge_id in cd.cycle_edges
                )])
            }

        except Exception as e:
            logger.error(f"Break optional strategy fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    async def _merge_cycles_strategy(
        self,
        graph: DependencyGraph,
        circular_dependencies: list[CircularDependency]
    ) -> dict[str, Any]:
        """Merge-Cycles Resolution-Strategie."""
        try:
            # Vereinfachte Implementierung - erstelle Mega-Node für jeden Cycle
            merged_nodes = []
            modified_nodes = []

            for i, circular_dep in enumerate(circular_dependencies):
                # Erstelle Mega-Node-ID
                mega_node_id = f"merged_cycle_{i}_{graph.graph_id}"

                # Erstelle Mega-Node
                from .data_models import DependencyType

                mega_node = DependencyNode(
                    node_id=mega_node_id,
                    node_type=DependencyType.SERVICE,  # Generischer Type
                    name=f"Merged Cycle {i}",
                    description=f"Merged node for circular dependency cycle {i}",
                    metadata={
                        "is_merged_cycle": True,
                        "original_nodes": circular_dep.cycle_nodes,
                        "merge_strategy": "cycles"
                    }
                )

                # Füge Mega-Node zum Graph hinzu
                graph.nodes[mega_node_id] = mega_node
                merged_nodes.append(mega_node_id)

                # Markiere Original-Nodes als merged
                for node_id in circular_dep.cycle_nodes:
                    node = graph.nodes.get(node_id)
                    if node:
                        node.status = DependencyStatus.RESOLVED
                        node.metadata["merged_into"] = mega_node_id
                        node.updated_at = datetime.utcnow()
                        modified_nodes.append(node_id)

            return {
                "success": True,
                "strategy": "merge_cycles",
                "merged_nodes": merged_nodes,
                "modified_nodes": modified_nodes,
                "cycles_resolved": len(circular_dependencies)
            }

        except Exception as e:
            logger.error(f"Merge cycles strategy fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    async def _fail_fast_strategy(
        self,
        graph: DependencyGraph,
        circular_dependencies: list[CircularDependency]
    ) -> dict[str, Any]:
        """Fail-Fast Resolution-Strategie."""
        try:
            if circular_dependencies:
                cycle_info = [
                    f"Cycle {i}: {' -> '.join(cd.cycle_nodes)}"
                    for i, cd in enumerate(circular_dependencies)
                ]

                error_message = f"Circular dependencies detected in graph {graph.graph_id}: {'; '.join(cycle_info)}"

                return {
                    "success": False,
                    "strategy": "fail_fast",
                    "error": error_message,
                    "cycles_detected": len(circular_dependencies)
                }

            return {
                "success": True,
                "strategy": "fail_fast",
                "cycles_detected": 0
            }

        except Exception as e:
            logger.error(f"Fail fast strategy fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    async def _ignore_strategy(
        self,
        graph: DependencyGraph,
        circular_dependencies: list[CircularDependency]
    ) -> dict[str, Any]:
        """Ignore Resolution-Strategie."""
        try:
            # Markiere Cycles als ignoriert
            for circular_dep in circular_dependencies:
                for node_id in circular_dep.cycle_nodes:
                    node = graph.nodes.get(node_id)
                    if node:
                        node.metadata["circular_dependency_ignored"] = True
                        node.updated_at = datetime.utcnow()

            return {
                "success": True,
                "strategy": "ignore",
                "cycles_ignored": len(circular_dependencies),
                "warning": "Circular dependencies were ignored - this may cause resolution issues"
            }

        except Exception as e:
            logger.error(f"Ignore strategy fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    def _normalize_cycle(self, cycle: list[str]) -> list[str]:
        """Normalisiert Cycle für Duplikat-Erkennung."""
        try:
            if not cycle:
                return cycle

            # Finde minimales Element als Start
            min_index = cycle.index(min(cycle))

            # Rotiere Cycle so dass minimales Element am Anfang steht
            normalized = cycle[min_index:] + cycle[:min_index]

            return normalized

        except Exception as e:
            logger.error(f"Cycle normalization fehlgeschlagen: {e}")
            return cycle

    def _get_cached_detection(self, graph_id: str) -> list[CircularDependency] | None:
        """Holt Cached Detection-Result."""
        try:
            if graph_id not in self._detection_cache:
                return None

            # Prüfe TTL
            if graph_id in self._cache_timestamps:
                age = time.time() - self._cache_timestamps[graph_id]
                if age > self._cache_ttl_seconds:
                    del self._detection_cache[graph_id]
                    del self._cache_timestamps[graph_id]
                    return None

            return self._detection_cache[graph_id]

        except Exception as e:
            logger.error(f"Cache lookup fehlgeschlagen: {e}")
            return None

    def _cache_detection_result(self, graph_id: str, result: list[CircularDependency]) -> None:
        """Cached Detection-Result."""
        try:
            self._detection_cache[graph_id] = result
            self._cache_timestamps[graph_id] = time.time()

            # Memory-Limit prüfen
            if len(self._detection_cache) > 100:
                # Entferne älteste Einträge
                oldest_keys = sorted(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k]
                )[:10]

                for key in oldest_keys:
                    del self._detection_cache[key]
                    del self._cache_timestamps[key]

        except Exception as e:
            logger.error(f"Cache storage fehlgeschlagen: {e}")

    def _update_detection_performance_stats(self, detection_time_ms: float, cycles_count: int) -> None:
        """Aktualisiert Detection-Performance-Statistiken."""
        self._detection_count += 1
        self._total_detection_time_ms += detection_time_ms
        self._cycles_detected += cycles_count

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_detection_time = (
            self._total_detection_time_ms / self._detection_count
            if self._detection_count > 0 else 0.0
        )

        return {
            "total_detections": self._detection_count,
            "avg_detection_time_ms": avg_detection_time,
            "cycles_detected": self._cycles_detected,
            "cycles_resolved": self._cycles_resolved,
            "cache_size": len(self._detection_cache),
            "available_algorithms": self.detection_algorithms,
            "default_algorithm": self.default_algorithm,
            "advanced_detection_enabled": self.enable_advanced_detection,
            "weak_cycle_detection_enabled": self.enable_weak_cycle_detection,
            "resolution_suggestions_enabled": self.enable_resolution_suggestions,
            "max_cycle_length": self.max_cycle_length
        }
