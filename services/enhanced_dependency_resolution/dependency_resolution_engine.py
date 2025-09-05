# backend/services/enhanced_dependency_resolution/dependency_resolution_engine.py
"""Enhanced Dependency Resolution Engine.

Implementiert Dynamic Dependency Resolution mit Real-time Tracking
und Integration mit bestehenden Task- und Resource-Management-Systemen.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_quotas_limits_management import (
    EnhancedQuotaManagementEngine,
    QuotaScope,
    ResourceType,
)
from services.enhanced_security_integration import SecurityContext

from .data_models import (
    CircularResolutionStrategy,
    DependencyGraph,
    DependencyResolutionRequest,
    DependencyResolutionResult,
    DependencyStatus,
    ResolutionStrategy,
    ResourceDependencyContext,
    TaskDependencyContext,
)
from .dependency_graph_engine import EnhancedDependencyGraphEngine

logger = get_logger(__name__)


class EnhancedDependencyResolutionEngine:
    """Enhanced Dependency Resolution Engine für Dynamic Resolution."""

    def __init__(
        self,
        graph_engine: EnhancedDependencyGraphEngine,
        quota_management_engine: EnhancedQuotaManagementEngine | None = None
    ):
        """Initialisiert Enhanced Dependency Resolution Engine.

        Args:
            graph_engine: Dependency Graph Engine
            quota_management_engine: Quota Management Engine
        """
        self.graph_engine = graph_engine
        self.quota_management_engine = quota_management_engine

        # Resolution-Konfiguration
        self.enable_real_time_tracking = True
        self.enable_quota_integration = True
        self.enable_parallel_resolution = True
        self.default_timeout_seconds = 300
        self.max_resolution_depth = 50

        # Resolution-Storage
        self._active_resolutions: dict[str, DependencyResolutionRequest] = {}
        self._resolution_results: dict[str, DependencyResolutionResult] = {}
        self._resolution_cache: dict[str, DependencyResolutionResult] = {}

        # Performance-Tracking
        self._resolution_count = 0
        self._total_resolution_time_ms = 0.0
        self._successful_resolutions = 0
        self._failed_resolutions = 0
        self._circular_resolutions = 0

        # Real-time Tracking
        self._resolution_status: dict[str, dict[str, Any]] = {}
        self._resolution_callbacks: dict[str, list[callable]] = defaultdict(list)

        # Background-Tasks
        self._tracking_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Enhanced Dependency Resolution Engine initialisiert")

    async def start(self) -> None:
        """Startet Dependency Resolution Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._tracking_task = asyncio.create_task(self._real_time_tracking_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Enhanced Dependency Resolution Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Dependency Resolution Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._tracking_task:
            self._tracking_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        await asyncio.gather(
            self._tracking_task,
            self._cleanup_task,
            return_exceptions=True
        )

        logger.info("Enhanced Dependency Resolution Engine gestoppt")

    async def resolve_dependencies(
        self,
        request: DependencyResolutionRequest,
        security_context: SecurityContext | None = None
    ) -> DependencyResolutionResult:
        """Löst Dependencies auf.

        Args:
            request: Dependency-Resolution-Request
            security_context: Security-Context

        Returns:
            Dependency-Resolution-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "dependency_resolution_started",
                "request_id": request.request_id,
                "graph_id": request.graph_id,
                "target_nodes": len(request.target_nodes),
                "strategy": request.resolution_strategy.value
            })

            # Registriere aktive Resolution
            self._active_resolutions[request.request_id] = request

            # Hole Graph
            graph = self.graph_engine._graphs.get(request.graph_id)
            if not graph:
                raise ValueError(f"Graph {request.graph_id} nicht gefunden")

            # Cache-Check
            if request.use_cache:
                cached_result = await self._get_cached_resolution(request)
                if cached_result:
                    cached_result.resolution_time_ms = (time.time() - start_time) * 1000
                    return cached_result

            # Quota-Check (falls aktiviert)
            if self.enable_quota_integration and self.quota_management_engine and security_context:
                quota_check = await self._check_resolution_quotas(request, security_context)
                if not quota_check["allowed"]:
                    return self._create_failed_result(
                        request,
                        ["Quota exceeded for dependency resolution"],
                        start_time
                    )

            # Führe Resolution durch
            if request.resolution_strategy == ResolutionStrategy.PARALLEL:
                result = await self._resolve_parallel(graph, request)
            elif request.resolution_strategy == ResolutionStrategy.SEQUENTIAL:
                result = await self._resolve_sequential(graph, request)
            elif request.resolution_strategy == ResolutionStrategy.LAZY:
                result = await self._resolve_lazy(graph, request)
            else:  # EAGER
                result = await self._resolve_eager(graph, request)

            # Cache Result
            if request.use_cache and result.success:
                await self._cache_resolution_result(request, result)

            # Performance-Tracking
            resolution_time_ms = (time.time() - start_time) * 1000
            result.resolution_time_ms = resolution_time_ms
            self._update_resolution_performance_stats(resolution_time_ms, result.success)

            # Speichere Result
            self._resolution_results[request.request_id] = result

            # Cleanup aktive Resolution
            if request.request_id in self._active_resolutions:
                del self._active_resolutions[request.request_id]

            logger.debug({
                "event": "dependency_resolution_completed",
                "request_id": request.request_id,
                "success": result.success,
                "resolved_nodes": len(result.resolved_nodes),
                "failed_nodes": len(result.failed_nodes),
                "circular_dependencies": len(result.circular_dependencies),
                "resolution_time_ms": resolution_time_ms
            })

            return result

        except Exception as e:
            logger.error(f"Dependency resolution fehlgeschlagen: {e}")

            # Cleanup
            if request.request_id in self._active_resolutions:
                del self._active_resolutions[request.request_id]

            return self._create_failed_result(request, [str(e)], start_time)

    async def resolve_task_dependencies(
        self,
        task_context: TaskDependencyContext,
        security_context: SecurityContext | None = None
    ) -> DependencyResolutionResult:
        """Löst Task-Dependencies auf.

        Args:
            task_context: Task-Dependency-Context
            security_context: Security-Context

        Returns:
            Dependency-Resolution-Result
        """
        try:
            # Erstelle Graph für Task-Dependencies
            graph_id = f"task_deps_{task_context.task_id}"

            graph = await self.graph_engine.create_dependency_graph(
                graph_id=graph_id,
                name=f"Task Dependencies for {task_context.task_id}",
                description=f"Dependency graph for task {task_context.task_id}"
            )

            # Füge Task-Dependencies hinzu
            await self._build_task_dependency_graph(graph, task_context)

            # Erstelle Resolution-Request
            import uuid
            request = DependencyResolutionRequest(
                request_id=str(uuid.uuid4()),
                graph_id=graph_id,
                target_nodes=[task_context.task_id],
                resolution_strategy=ResolutionStrategy.EAGER,
                max_depth=task_context.max_resolution_time_ms or 100,
                timeout_seconds=300,
                security_level=task_context.security_level,
                user_id=security_context.user_id if security_context else None,
                tenant_id=security_context.tenant_id if security_context else None
            )

            # Löse Dependencies auf
            return await self.resolve_dependencies(request, security_context)

        except Exception as e:
            logger.error(f"Task dependency resolution fehlgeschlagen: {e}")
            raise

    async def resolve_resource_dependencies(
        self,
        resource_context: ResourceDependencyContext,
        security_context: SecurityContext | None = None
    ) -> DependencyResolutionResult:
        """Löst Resource-Dependencies auf.

        Args:
            resource_context: Resource-Dependency-Context
            security_context: Security-Context

        Returns:
            Dependency-Resolution-Result
        """
        try:
            # Erstelle Graph für Resource-Dependencies
            graph_id = f"resource_deps_{resource_context.resource_id}"

            graph = await self.graph_engine.create_dependency_graph(
                graph_id=graph_id,
                name=f"Resource Dependencies for {resource_context.resource_id}",
                description=f"Dependency graph for resource {resource_context.resource_id}"
            )

            # Füge Resource-Dependencies hinzu
            await self._build_resource_dependency_graph(graph, resource_context)

            # Erstelle Resolution-Request
            import uuid
            request = DependencyResolutionRequest(
                request_id=str(uuid.uuid4()),
                graph_id=graph_id,
                target_nodes=[resource_context.resource_id],
                resolution_strategy=ResolutionStrategy.PARALLEL,
                timeout_seconds=300,
                user_id=security_context.user_id if security_context else None,
                tenant_id=security_context.tenant_id if security_context else None
            )

            # Löse Dependencies auf
            return await self.resolve_dependencies(request, security_context)

        except Exception as e:
            logger.error(f"Resource dependency resolution fehlgeschlagen: {e}")
            raise

    async def _resolve_eager(
        self,
        graph: DependencyGraph,
        request: DependencyResolutionRequest
    ) -> DependencyResolutionResult:
        """Führt Eager Resolution durch."""
        try:
            import uuid
            result = DependencyResolutionResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                success=True
            )

            # Topological Sort für Resolution-Order
            analysis = await self.graph_engine.analyze_dependency_graph(
                request.graph_id,
                include_circular_detection=True
            )

            if not analysis["is_acyclic"]:
                # Handle Circular Dependencies
                circular_result = await self._handle_circular_dependencies(
                    graph, request, analysis["circular_dependencies"]
                )
                result.circular_dependencies = circular_result["circular_dependencies"]
                result.broken_dependencies = circular_result["broken_dependencies"]

            # Löse Dependencies in topologischer Reihenfolge auf
            resolution_order = analysis["topological_order"]

            for node_id in resolution_order:
                if node_id in request.target_nodes or node_id in result.resolved_nodes:
                    # Löse Node-Dependencies auf
                    node_result = await self._resolve_single_node(graph, node_id, request)

                    if node_result["success"]:
                        result.resolved_nodes.add(node_id)
                        result.resolution_order.append(node_id)
                    else:
                        result.failed_nodes.add(node_id)
                        result.errors.extend(node_result["errors"])

            # Prüfe ob alle Target-Nodes aufgelöst wurden
            unresolved_targets = set(request.target_nodes) - result.resolved_nodes
            if unresolved_targets:
                result.success = False
                result.errors.append(f"Failed to resolve target nodes: {list(unresolved_targets)}")

            return result

        except Exception as e:
            logger.error(f"Eager resolution fehlgeschlagen: {e}")
            raise

    async def _resolve_parallel(
        self,
        graph: DependencyGraph,
        request: DependencyResolutionRequest
    ) -> DependencyResolutionResult:
        """Führt Parallel Resolution durch."""
        try:
            import uuid
            result = DependencyResolutionResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                success=True
            )

            # Gruppiere Nodes nach Dependency-Level
            dependency_levels = await self._calculate_dependency_levels(graph, request.target_nodes)

            # Löse Level für Level parallel auf
            for level, nodes in dependency_levels.items():
                if not nodes:
                    continue

                # Parallel Resolution für alle Nodes im Level
                tasks = []
                for node_id in nodes:
                    task = asyncio.create_task(
                        self._resolve_single_node(graph, node_id, request)
                    )
                    tasks.append((node_id, task))

                # Warte auf alle Tasks im Level
                for node_id, task in tasks:
                    try:
                        node_result = await task

                        if node_result["success"]:
                            result.resolved_nodes.add(node_id)
                            result.resolution_order.append(node_id)
                        else:
                            result.failed_nodes.add(node_id)
                            result.errors.extend(node_result["errors"])

                    except Exception as e:
                        result.failed_nodes.add(node_id)
                        result.errors.append(f"Node {node_id} resolution failed: {e}")

            # Prüfe Erfolg
            unresolved_targets = set(request.target_nodes) - result.resolved_nodes
            if unresolved_targets:
                result.success = False
                result.errors.append(f"Failed to resolve target nodes: {list(unresolved_targets)}")

            return result

        except Exception as e:
            logger.error(f"Parallel resolution fehlgeschlagen: {e}")
            raise

    async def _resolve_sequential(
        self,
        graph: DependencyGraph,
        request: DependencyResolutionRequest
    ) -> DependencyResolutionResult:
        """Führt Sequential Resolution durch."""
        try:
            import uuid
            result = DependencyResolutionResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                success=True
            )

            # Sequenzielle Resolution der Target-Nodes
            for target_node_id in request.target_nodes:
                node_result = await self._resolve_node_with_dependencies(
                    graph, target_node_id, request, result.resolved_nodes
                )

                if node_result["success"]:
                    result.resolved_nodes.update(node_result["resolved_nodes"])
                    result.resolution_order.extend(node_result["resolution_order"])
                else:
                    result.failed_nodes.add(target_node_id)
                    result.errors.extend(node_result["errors"])
                    result.success = False

            return result

        except Exception as e:
            logger.error(f"Sequential resolution fehlgeschlagen: {e}")
            raise

    async def _resolve_lazy(
        self,
        graph: DependencyGraph,
        request: DependencyResolutionRequest
    ) -> DependencyResolutionResult:
        """Führt Lazy Resolution durch."""
        try:
            import uuid
            result = DependencyResolutionResult(
                result_id=str(uuid.uuid4()),
                request_id=request.request_id,
                success=True
            )

            # Lazy Resolution - nur Target-Nodes und direkte Dependencies
            for target_node_id in request.target_nodes:
                if target_node_id in graph.nodes:
                    node = graph.nodes[target_node_id]

                    # Löse nur direkte Dependencies auf
                    for dep_id in node.dependencies:
                        if dep_id in graph.nodes:
                            dep_result = await self._resolve_single_node(graph, dep_id, request)

                            if dep_result["success"]:
                                result.resolved_nodes.add(dep_id)
                                result.resolution_order.append(dep_id)
                            else:
                                result.failed_nodes.add(dep_id)
                                result.errors.extend(dep_result["errors"])

                    # Löse Target-Node auf
                    target_result = await self._resolve_single_node(graph, target_node_id, request)

                    if target_result["success"]:
                        result.resolved_nodes.add(target_node_id)
                        result.resolution_order.append(target_node_id)
                    else:
                        result.failed_nodes.add(target_node_id)
                        result.errors.extend(target_result["errors"])
                        result.success = False

            return result

        except Exception as e:
            logger.error(f"Lazy resolution fehlgeschlagen: {e}")
            raise

    async def _resolve_single_node(
        self,
        graph: DependencyGraph,
        node_id: str,
        _request: DependencyResolutionRequest
    ) -> dict[str, Any]:
        """Löst einzelnen Node auf."""
        try:
            node = graph.nodes.get(node_id)
            if not node:
                return {
                    "success": False,
                    "errors": [f"Node {node_id} nicht gefunden"]
                }

            # Simuliere Node-Resolution (in Realität würde hier die echte Resolution stattfinden)
            await asyncio.sleep(0.001)  # Simuliere Arbeit

            # Aktualisiere Node-Status
            node.status = DependencyStatus.RESOLVED
            node.updated_at = datetime.utcnow()

            return {
                "success": True,
                "errors": []
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Node {node_id} resolution failed: {e}"]
            }

    async def _resolve_node_with_dependencies(
        self,
        graph: DependencyGraph,
        node_id: str,
        request: DependencyResolutionRequest,
        already_resolved: set[str]
    ) -> dict[str, Any]:
        """Löst Node mit allen Dependencies auf."""
        try:
            resolved_nodes = set()
            resolution_order = []
            errors = []

            # DFS für Dependencies
            visited = set()

            async def resolve_recursive(current_node_id: str) -> bool:
                if current_node_id in visited or current_node_id in already_resolved:
                    return True

                visited.add(current_node_id)
                node = graph.nodes.get(current_node_id)

                if not node:
                    errors.append(f"Node {current_node_id} nicht gefunden")
                    return False

                # Löse Dependencies zuerst auf
                for dep_id in node.dependencies:
                    if not await resolve_recursive(dep_id):
                        return False

                # Löse aktuellen Node auf
                node_result = await self._resolve_single_node(graph, current_node_id, request)

                if node_result["success"]:
                    resolved_nodes.add(current_node_id)
                    resolution_order.append(current_node_id)
                    return True
                errors.extend(node_result["errors"])
                return False

            success = await resolve_recursive(node_id)

            return {
                "success": success,
                "resolved_nodes": resolved_nodes,
                "resolution_order": resolution_order,
                "errors": errors
            }

        except Exception as e:
            return {
                "success": False,
                "resolved_nodes": set(),
                "resolution_order": [],
                "errors": [f"Recursive resolution failed: {e}"]
            }

    async def _calculate_dependency_levels(
        self,
        graph: DependencyGraph,
        _target_nodes: list[str]
    ) -> dict[int, list[str]]:
        """Berechnet Dependency-Levels für parallele Resolution."""
        try:
            levels = defaultdict(list)
            node_levels = {}

            # BFS für Level-Calculation
            queue = deque()

            # Starte mit Nodes ohne Dependencies
            for node_id in graph.nodes.keys():
                node = graph.nodes[node_id]
                if not node.dependencies:
                    queue.append((node_id, 0))
                    node_levels[node_id] = 0

            while queue:
                node_id, level = queue.popleft()
                levels[level].append(node_id)

                # Füge Dependents zum nächsten Level hinzu
                node = graph.nodes[node_id]
                for dependent_id in node.dependents:
                    if dependent_id not in node_levels:
                        # Prüfe ob alle Dependencies des Dependents aufgelöst sind
                        dependent = graph.nodes[dependent_id]
                        all_deps_resolved = all(
                            dep_id in node_levels for dep_id in dependent.dependencies
                        )

                        if all_deps_resolved:
                            max_dep_level = max(
                                node_levels[dep_id] for dep_id in dependent.dependencies
                            ) if dependent.dependencies else -1

                            dependent_level = max_dep_level + 1
                            queue.append((dependent_id, dependent_level))
                            node_levels[dependent_id] = dependent_level

            return dict(levels)

        except Exception as e:
            logger.error(f"Dependency levels calculation fehlgeschlagen: {e}")
            return {}

    async def _handle_circular_dependencies(
        self,
        graph: DependencyGraph,
        _request: DependencyResolutionRequest,
        circular_dependencies: list[list[str]]
    ) -> dict[str, Any]:
        """Behandelt Circular Dependencies."""
        try:
            broken_dependencies = []

            for cycle in circular_dependencies:
                if graph.circular_resolution_strategy == CircularResolutionStrategy.BREAK_WEAKEST:
                    # Finde schwächste Dependency im Cycle
                    weakest_edge = await self._find_weakest_edge_in_cycle(graph, cycle)
                    if weakest_edge:
                        broken_dependencies.append(weakest_edge)
                        # Entferne Edge temporär
                        await self._temporarily_break_edge(graph, weakest_edge)

                elif graph.circular_resolution_strategy == CircularResolutionStrategy.BREAK_OPTIONAL:
                    # Breche optionale Dependencies
                    optional_edges = await self._find_optional_edges_in_cycle(graph, cycle)
                    broken_dependencies.extend(optional_edges)
                    for edge_id in optional_edges:
                        await self._temporarily_break_edge(graph, edge_id)

                elif graph.circular_resolution_strategy == CircularResolutionStrategy.FAIL_FAST:
                    raise ValueError(f"Circular dependency detected: {cycle}")

            return {
                "circular_dependencies": circular_dependencies,
                "broken_dependencies": broken_dependencies
            }

        except Exception as e:
            logger.error(f"Circular dependency handling fehlgeschlagen: {e}")
            raise

    async def _find_weakest_edge_in_cycle(self, graph: DependencyGraph, cycle: list[str]) -> str | None:
        """Findet schwächste Edge im Cycle."""
        try:
            weakest_edge_id = None
            min_weight = float("inf")

            for i in range(len(cycle)):
                source_id = cycle[i]
                target_id = cycle[(i + 1) % len(cycle)]

                # Finde Edge zwischen source und target
                for edge_id, edge in graph.edges.items():
                    if edge.source_node_id == source_id and edge.target_node_id == target_id:
                        if edge.weight < min_weight:
                            min_weight = edge.weight
                            weakest_edge_id = edge_id
                        break

            return weakest_edge_id

        except Exception as e:
            logger.error(f"Weakest edge finding fehlgeschlagen: {e}")
            return None

    async def _find_optional_edges_in_cycle(self, graph: DependencyGraph, cycle: list[str]) -> list[str]:
        """Findet optionale Edges im Cycle."""
        try:
            optional_edges = []

            for i in range(len(cycle)):
                source_id = cycle[i]
                target_id = cycle[(i + 1) % len(cycle)]

                # Finde Edge zwischen source und target
                for edge_id, edge in graph.edges.items():
                    if (edge.source_node_id == source_id and
                        edge.target_node_id == target_id and
                        edge.relation.value == "optional"):
                        optional_edges.append(edge_id)
                        break

            return optional_edges

        except Exception as e:
            logger.error(f"Optional edges finding fehlgeschlagen: {e}")
            return []

    async def _temporarily_break_edge(self, graph: DependencyGraph, edge_id: str) -> None:
        """Bricht Edge temporär."""
        try:
            edge = graph.edges.get(edge_id)
            if edge:
                # Markiere Edge als gebrochen
                edge.status = DependencyStatus.FAILED
                edge.metadata["temporarily_broken"] = True
                edge.updated_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Edge breaking fehlgeschlagen: {e}")

    async def _build_task_dependency_graph(
        self,
        graph: DependencyGraph,
        task_context: TaskDependencyContext
    ) -> None:
        """Baut Task-Dependency-Graph auf."""
        try:
            from .data_models import DependencyRelation, DependencyType

            # Füge Task-Node hinzu
            await self.graph_engine.add_dependency_node(
                graph_id=graph.graph_id,
                node_id=task_context.task_id,
                node_type=DependencyType.TASK,
                name=f"Task {task_context.task_id}",
                description=f"Main task node for {task_context.task_id}",
                metadata=task_context.metadata
            )

            # Füge Required-Tasks hinzu
            for i, required_task_id in enumerate(task_context.required_tasks):
                await self.graph_engine.add_dependency_node(
                    graph_id=graph.graph_id,
                    node_id=required_task_id,
                    node_type=DependencyType.TASK,
                    name=f"Required Task {required_task_id}",
                    description="Required task dependency"
                )

                await self.graph_engine.add_dependency_edge(
                    graph_id=graph.graph_id,
                    edge_id=f"task_dep_{i}",
                    source_node_id=task_context.task_id,
                    target_node_id=required_task_id,
                    relation=DependencyRelation.REQUIRES,
                    dependency_type=DependencyType.TASK
                )

            # Füge Required-Agents hinzu
            for i, required_agent_id in enumerate(task_context.required_agents):
                await self.graph_engine.add_dependency_node(
                    graph_id=graph.graph_id,
                    node_id=required_agent_id,
                    node_type=DependencyType.AGENT,
                    name=f"Required Agent {required_agent_id}",
                    description="Required agent dependency"
                )

                await self.graph_engine.add_dependency_edge(
                    graph_id=graph.graph_id,
                    edge_id=f"agent_dep_{i}",
                    source_node_id=task_context.task_id,
                    target_node_id=required_agent_id,
                    relation=DependencyRelation.REQUIRES,
                    dependency_type=DependencyType.AGENT
                )

        except Exception as e:
            logger.error(f"Task dependency graph building fehlgeschlagen: {e}")
            raise

    async def _build_resource_dependency_graph(
        self,
        graph: DependencyGraph,
        resource_context: ResourceDependencyContext
    ) -> None:
        """Baut Resource-Dependency-Graph auf."""
        try:
            from .data_models import DependencyRelation, DependencyType

            # Füge Resource-Node hinzu
            await self.graph_engine.add_dependency_node(
                graph_id=graph.graph_id,
                node_id=resource_context.resource_id,
                node_type=DependencyType.RESOURCE,
                name=f"Resource {resource_context.resource_id}",
                description=f"Main resource node for {resource_context.resource_id}",
                metadata=resource_context.metadata
            )

            # Füge Required-Resources hinzu
            for i, required_resource_id in enumerate(resource_context.required_resources):
                await self.graph_engine.add_dependency_node(
                    graph_id=graph.graph_id,
                    node_id=required_resource_id,
                    node_type=DependencyType.RESOURCE,
                    name=f"Required Resource {required_resource_id}",
                    description="Required resource dependency"
                )

                await self.graph_engine.add_dependency_edge(
                    graph_id=graph.graph_id,
                    edge_id=f"resource_dep_{i}",
                    source_node_id=resource_context.resource_id,
                    target_node_id=required_resource_id,
                    relation=DependencyRelation.REQUIRES,
                    dependency_type=DependencyType.RESOURCE
                )

        except Exception as e:
            logger.error(f"Resource dependency graph building fehlgeschlagen: {e}")
            raise

    async def _check_resolution_quotas(
        self,
        request: DependencyResolutionRequest,
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Prüft Resolution-Quotas."""
        try:
            if not self.quota_management_engine:
                return {"allowed": True}

            # Prüfe API-Call-Quota für Dependency-Resolution
            quota_check = await self.quota_management_engine.check_quota_with_security(
                resource_type=ResourceType.API_CALL,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                amount=len(request.target_nodes),
                security_context=security_context
            )

            return {
                "allowed": quota_check.allowed,
                "quota_result": quota_check
            }

        except Exception as e:
            logger.error(f"Resolution quota check fehlgeschlagen: {e}")
            return {"allowed": True}  # Fail-open

    async def _get_cached_resolution(self, request: DependencyResolutionRequest) -> DependencyResolutionResult | None:
        """Holt Cached Resolution-Result."""
        try:
            cache_key = self._create_cache_key(request)
            return self._resolution_cache.get(cache_key)

        except Exception as e:
            logger.error(f"Cache lookup fehlgeschlagen: {e}")
            return None

    async def _cache_resolution_result(
        self,
        request: DependencyResolutionRequest,
        result: DependencyResolutionResult
    ) -> None:
        """Cached Resolution-Result."""
        try:
            cache_key = self._create_cache_key(request)
            self._resolution_cache[cache_key] = result

            # Memory-Limit prüfen
            if len(self._resolution_cache) > 1000:
                # Entferne älteste Einträge
                oldest_keys = list(self._resolution_cache.keys())[:100]
                for key in oldest_keys:
                    del self._resolution_cache[key]

        except Exception as e:
            logger.error(f"Cache storage fehlgeschlagen: {e}")

    def _create_cache_key(self, request: DependencyResolutionRequest) -> str:
        """Erstellt Cache-Key für Request."""
        return f"{request.graph_id}_{hash(tuple(sorted(request.target_nodes)))}_{request.resolution_strategy.value}"

    def _create_failed_result(
        self,
        request: DependencyResolutionRequest,
        errors: list[str],
        start_time: float
    ) -> DependencyResolutionResult:
        """Erstellt Failed Resolution-Result."""
        import uuid

        return DependencyResolutionResult(
            result_id=str(uuid.uuid4()),
            request_id=request.request_id,
            success=False,
            errors=errors,
            resolution_time_ms=(time.time() - start_time) * 1000
        )

    async def _real_time_tracking_loop(self) -> None:
        """Background-Loop für Real-time Tracking."""
        while self._is_running:
            try:
                await asyncio.sleep(1)  # Update jede Sekunde

                if self._is_running:
                    await self._update_resolution_status()

            except Exception as e:
                logger.error(f"Real-time tracking loop fehlgeschlagen: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Cleanup alle 5 Minuten

                if self._is_running:
                    await self._cleanup_old_results()

            except Exception as e:
                logger.error(f"Cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _update_resolution_status(self) -> None:
        """Aktualisiert Resolution-Status."""
        try:
            for request_id, request in self._active_resolutions.items():
                self._resolution_status[request_id] = {
                    "request_id": request_id,
                    "graph_id": request.graph_id,
                    "target_nodes": request.target_nodes,
                    "status": "resolving",
                    "start_time": request.created_at.isoformat()
                }

        except Exception as e:
            logger.error(f"Resolution status update fehlgeschlagen: {e}")

    async def _cleanup_old_results(self) -> None:
        """Bereinigt alte Resolution-Results."""
        try:
            from datetime import timedelta

            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            # Cleanup Results
            old_result_ids = [
                result_id for result_id, result in self._resolution_results.items()
                if result.resolved_at < cutoff_time
            ]

            for result_id in old_result_ids:
                del self._resolution_results[result_id]

            if old_result_ids:
                logger.debug(f"Resolution cleanup: {len(old_result_ids)} alte Results entfernt")

        except Exception as e:
            logger.error(f"Resolution cleanup fehlgeschlagen: {e}")

    def _update_resolution_performance_stats(self, resolution_time_ms: float, success: bool) -> None:
        """Aktualisiert Resolution-Performance-Statistiken."""
        self._resolution_count += 1
        self._total_resolution_time_ms += resolution_time_ms

        if success:
            self._successful_resolutions += 1
        else:
            self._failed_resolutions += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_resolution_time = (
            self._total_resolution_time_ms / self._resolution_count
            if self._resolution_count > 0 else 0.0
        )

        success_rate = (
            self._successful_resolutions / self._resolution_count
            if self._resolution_count > 0 else 0.0
        )

        return {
            "total_resolutions": self._resolution_count,
            "avg_resolution_time_ms": avg_resolution_time,
            "successful_resolutions": self._successful_resolutions,
            "failed_resolutions": self._failed_resolutions,
            "circular_resolutions": self._circular_resolutions,
            "success_rate": success_rate,
            "active_resolutions": len(self._active_resolutions),
            "cached_results": len(self._resolution_cache),
            "meets_resolution_sla": avg_resolution_time < 100.0,
            "real_time_tracking_enabled": self.enable_real_time_tracking,
            "quota_integration_enabled": self.enable_quota_integration,
            "parallel_resolution_enabled": self.enable_parallel_resolution
        }
