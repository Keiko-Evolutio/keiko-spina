"""Parallel Discovery Implementation für Voice Performance Optimization.
Implementiert gleichzeitige Discovery-Operationen für Agents, Tools und Services.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .interfaces import (
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryType,
    IParallelDiscovery,
    VoicePerformanceSettings,
)

logger = get_logger(__name__)


class ParallelDiscoveryEngine(IParallelDiscovery):
    """Parallel Discovery Engine Implementation.
    Führt Discovery-Operationen parallel aus für optimale Performance.
    """

    def __init__(self, settings: VoicePerformanceSettings):
        self.settings = settings

        # Discovery Registries
        self._agent_registries: list[Any] = []
        self._tool_registries: list[Any] = []
        self._service_registries: list[Any] = []

        # Performance Tracking
        self._discovery_stats = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_time_ms": 0.0,
            "cache_hits": 0
        })

        # Concurrent Execution Limits
        self._agent_semaphore = asyncio.Semaphore(settings.max_concurrent_discoveries)
        self._tool_semaphore = asyncio.Semaphore(settings.max_concurrent_tools)
        self._service_semaphore = asyncio.Semaphore(settings.max_concurrent_agents)

        # Cache für Discovery Results
        self._discovery_cache: dict[str, tuple[DiscoveryResult, datetime]] = {}

        # Registries initialisieren
        self._initialize_registries()

        logger.info("Parallel discovery engine initialized")

    def _initialize_registries(self) -> None:
        """Initialisiert Mock Registries für Testing."""
        # Mock Agent Registries
        self._agent_registries = [
            type("MockRegistry", (), {"name": "primary_registry"})(),
            type("MockRegistry", (), {"name": "secondary_registry"})(),
            type("MockRegistry", (), {"name": "fallback_registry"})()
        ]

        # Mock Tool Registries
        self._tool_registries = [
            type("MockToolRegistry", (), {"name": "tool_registry_1"})(),
            type("MockToolRegistry", (), {"name": "tool_registry_2"})()
        ]

        # Mock Service Registries
        self._service_registries = [
            type("MockServiceRegistry", (), {"name": "service_registry"})()
        ]

    async def discover_agents(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Führt parallele Agent Discovery durch."""
        start_time = time.time()
        discovery_type = DiscoveryType.AGENT_DISCOVERY

        try:
            # Cache Check
            cache_key = self._generate_cache_key(request)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self._update_stats(discovery_type, time.time() - start_time, True, cache_hit=True)
                return cached_result

            # Parallel Agent Discovery
            async with self._agent_semaphore:
                if request.parallel_execution and len(self._agent_registries) > 1:
                    results = await self._parallel_agent_discovery(request)
                else:
                    results = await self._sequential_agent_discovery(request)

            # Merge und Score Results
            merged_result = await self._merge_agent_results(results, request)

            # Cache Result
            await self._cache_result(cache_key, merged_result)

            discovery_time_ms = (time.time() - start_time) * 1000
            merged_result.discovery_time_ms = discovery_time_ms

            self._update_stats(discovery_type, discovery_time_ms, True)
            return merged_result

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            discovery_time_ms = (time.time() - start_time) * 1000
            self._update_stats(discovery_type, discovery_time_ms, False)

            return DiscoveryResult(
                discovery_type=discovery_type,
                query=request.query,
                items=[],
                total_found=0,
                confidence_scores=[],
                discovery_time_ms=discovery_time_ms
            )

    async def _parallel_agent_discovery(self, request: DiscoveryRequest) -> list[DiscoveryResult]:
        """Führt parallele Agent Discovery über mehrere Registries durch."""
        tasks = []

        for registry in self._agent_registries:
            task = asyncio.create_task(
                self._discover_from_agent_registry(registry, request)
            )
            tasks.append(task)

        # Warte auf alle Tasks mit Timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=request.timeout_seconds
            )

            # Filtere erfolgreiche Results
            successful_results = []
            for result in results:
                if isinstance(result, DiscoveryResult):
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Agent registry discovery failed: {result}")

            return successful_results

        except TimeoutError:
            logger.warning(f"Agent discovery timeout after {request.timeout_seconds}s")
            # Cancelle noch laufende Tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            return []

    async def _discover_from_agent_registry(
        self,
        registry: Any,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Führt Discovery von einem Agent Registry durch."""
        try:
            # Simuliere Agent Registry Discovery
            # In echter Implementation würde hier der Registry-Call stehen
            await asyncio.sleep(0.1)  # Simuliere Registry-Latenz

            # Mock Agent Discovery Results
            mock_agents = [
                {
                    "agent_id": f"agent_{i}",
                    "name": f"Agent {i}",
                    "capabilities": ["text_processing", "voice_interaction"],
                    "confidence": 0.8 + (i * 0.05),
                    "latency_ms": 100 + (i * 20),
                    "registry": getattr(registry, "name", "unknown")
                }
                for i in range(min(request.max_results, 5))
            ]

            # Filtere nach Required Capabilities
            if request.required_capabilities:
                filtered_agents = []
                for agent in mock_agents:
                    agent_caps = set(agent.get("capabilities", []))
                    required_caps = set(request.required_capabilities)
                    if required_caps.issubset(agent_caps):
                        filtered_agents.append(agent)
                mock_agents = filtered_agents

            # Filtere nach Excluded Agents
            if request.excluded_agents:
                mock_agents = [
                    agent for agent in mock_agents
                    if agent["agent_id"] not in request.excluded_agents
                ]

            confidence_scores = [agent["confidence"] for agent in mock_agents]

            return DiscoveryResult(
                discovery_type=DiscoveryType.AGENT_DISCOVERY,
                query=request.query,
                items=mock_agents,
                total_found=len(mock_agents),
                confidence_scores=confidence_scores,
                discovery_time_ms=100.0,  # Mock time
                average_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                best_match_confidence=max(confidence_scores) if confidence_scores else 0.0
            )

        except Exception as e:
            logger.error(f"Agent registry discovery error: {e}")
            return DiscoveryResult(
                discovery_type=DiscoveryType.AGENT_DISCOVERY,
                query=request.query,
                items=[],
                total_found=0,
                confidence_scores=[],
                discovery_time_ms=0.0
            )

    async def discover_tools(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Führt parallele Tool Discovery durch."""
        start_time = time.time()
        discovery_type = DiscoveryType.TOOL_DISCOVERY

        try:
            # Cache Check
            cache_key = self._generate_cache_key(request)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self._update_stats(discovery_type, time.time() - start_time, True, cache_hit=True)
                return cached_result

            # Parallel Tool Discovery
            async with self._tool_semaphore:
                if request.parallel_execution:
                    results = await self._parallel_tool_discovery(request)
                else:
                    results = await self._sequential_tool_discovery(request)

            # Merge Results
            merged_result = await self._merge_tool_results(results, request)

            # Cache Result
            await self._cache_result(cache_key, merged_result)

            discovery_time_ms = (time.time() - start_time) * 1000
            merged_result.discovery_time_ms = discovery_time_ms

            self._update_stats(discovery_type, discovery_time_ms, True)
            return merged_result

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            discovery_time_ms = (time.time() - start_time) * 1000
            self._update_stats(discovery_type, discovery_time_ms, False)

            return DiscoveryResult(
                discovery_type=discovery_type,
                query=request.query,
                items=[],
                total_found=0,
                confidence_scores=[],
                discovery_time_ms=discovery_time_ms
            )

    async def _parallel_tool_discovery(self, request: DiscoveryRequest) -> list[DiscoveryResult]:
        """Führt parallele Tool Discovery durch."""
        # Mock Tool Discovery - in echter Implementation würde hier
        # parallele Abfrage verschiedener Tool-Registries stehen

        mock_tools = [
            {
                "tool_id": f"tool_{i}",
                "name": f"Tool {i}",
                "description": f"Tool for {request.query}",
                "capabilities": ["api_call", "data_processing"],
                "confidence": 0.7 + (i * 0.1),
                "latency_ms": 50 + (i * 10),
                "schema": {"type": "function", "parameters": {}}
            }
            for i in range(min(request.max_results, 3))
        ]

        confidence_scores = [tool["confidence"] for tool in mock_tools]

        return [DiscoveryResult(
            discovery_type=DiscoveryType.TOOL_DISCOVERY,
            query=request.query,
            items=mock_tools,
            total_found=len(mock_tools),
            confidence_scores=confidence_scores,
            discovery_time_ms=75.0,
            average_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            best_match_confidence=max(confidence_scores) if confidence_scores else 0.0
        )]

    async def discover_services(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Führt parallele Service Discovery durch."""
        start_time = time.time()
        discovery_type = DiscoveryType.SERVICE_DISCOVERY

        # Versuche Service Discovery Module zu importieren
        ServiceDiscoveryEngine = None
        ServiceDiscoveryQuery = None

        try:
            from agents.registry.service_discovery import DiscoveryQuery as ServiceDiscoveryQuery
            from agents.registry.service_discovery import ServiceDiscoveryEngine
        except ImportError:
            ServiceDiscoveryEngine = None
            ServiceDiscoveryQuery = None
            logger.debug("Service discovery not available, using mock implementation")

        try:
            # Verwende bestehende Service Discovery wenn verfügbar
            if ServiceDiscoveryEngine is not None and ServiceDiscoveryQuery is not None:
                # Konvertiere Request zu Service Discovery Query
                service_query = ServiceDiscoveryQuery(
                    service_name=request.query,
                    capabilities=request.required_capabilities,
                    region_preference=request.region_preference,
                    max_latency_ms=request.max_latency_ms,
                    max_results=request.max_results
                )

                # Führe Service Discovery durch
                service_engine = ServiceDiscoveryEngine()
                service_result = await service_engine.discover_services(service_query)

                # Konvertiere zu unserem Format
                confidence_scores = [0.8] * len(service_result.instances)

                result = DiscoveryResult(
                    discovery_type=discovery_type,
                    query=request.query,
                    items=[
                        {
                            "service_id": instance.instance_id,
                            "name": instance.service_name,
                            "endpoint": instance.endpoint,
                            "capabilities": instance.capabilities,
                            "confidence": 0.8,
                            "latency_ms": instance.latency_ms,
                            "health_status": instance.health_status
                        }
                        for instance in service_result.instances
                    ],
                    total_found=service_result.total_found,
                    confidence_scores=confidence_scores,
                    discovery_time_ms=service_result.query_time * 1000,
                    average_confidence=0.8,
                    best_match_confidence=0.8
                )

                self._update_stats(discovery_type, result.discovery_time_ms, True)
                return result

            # Fallback zu Mock Implementation
            mock_services = [
                {
                    "service_id": f"service_{i}",
                    "name": f"Service {i}",
                    "endpoint": f"http://service-{i}.local:8080",
                    "capabilities": ["api", "processing"],
                    "confidence": 0.75,
                    "latency_ms": 200,
                    "health_status": "healthy"
                }
                for i in range(min(request.max_results, 2))
            ]

            discovery_time_ms = (time.time() - start_time) * 1000
            confidence_scores = [service["confidence"] for service in mock_services]

            result = DiscoveryResult(
                discovery_type=discovery_type,
                query=request.query,
                items=mock_services,
                total_found=len(mock_services),
                confidence_scores=confidence_scores,
                discovery_time_ms=discovery_time_ms,
                average_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                best_match_confidence=max(confidence_scores) if confidence_scores else 0.0
            )

            self._update_stats(discovery_type, discovery_time_ms, True)
            return result

        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            discovery_time_ms = (time.time() - start_time) * 1000
            self._update_stats(discovery_type, discovery_time_ms, False)

            return DiscoveryResult(
                discovery_type=discovery_type,
                query=request.query,
                items=[],
                total_found=0,
                confidence_scores=[],
                discovery_time_ms=discovery_time_ms
            )

    async def discover_capabilities(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Führt parallele Capability Discovery durch."""
        start_time = time.time()
        discovery_type = DiscoveryType.CAPABILITY_DISCOVERY

        try:
            # Mock Capability Discovery
            mock_capabilities = [
                {
                    "capability_id": f"cap_{i}",
                    "name": f"Capability {i}",
                    "description": f"Capability for {request.query}",
                    "providers": [f"agent_{j}" for j in range(2)],
                    "confidence": 0.9,
                    "complexity": "medium"
                }
                for i in range(min(request.max_results, 4))
            ]

            discovery_time_ms = (time.time() - start_time) * 1000
            confidence_scores = [cap["confidence"] for cap in mock_capabilities]

            result = DiscoveryResult(
                discovery_type=discovery_type,
                query=request.query,
                items=mock_capabilities,
                total_found=len(mock_capabilities),
                confidence_scores=confidence_scores,
                discovery_time_ms=discovery_time_ms,
                average_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                best_match_confidence=max(confidence_scores) if confidence_scores else 0.0
            )

            self._update_stats(discovery_type, discovery_time_ms, True)
            return result

        except Exception as e:
            logger.error(f"Capability discovery failed: {e}")
            discovery_time_ms = (time.time() - start_time) * 1000
            self._update_stats(discovery_type, discovery_time_ms, False)

            return DiscoveryResult(
                discovery_type=discovery_type,
                query=request.query,
                items=[],
                total_found=0,
                confidence_scores=[],
                discovery_time_ms=discovery_time_ms
            )

    async def discover_all(self, requests: list[DiscoveryRequest]) -> list[DiscoveryResult]:
        """Führt mehrere Discovery-Operationen parallel durch."""
        start_time = time.time()

        try:
            # Erstelle Tasks für alle Discovery-Requests
            tasks = []
            for request in requests:
                if request.discovery_type == DiscoveryType.AGENT_DISCOVERY:
                    task = asyncio.create_task(self.discover_agents(request))
                elif request.discovery_type == DiscoveryType.TOOL_DISCOVERY:
                    task = asyncio.create_task(self.discover_tools(request))
                elif request.discovery_type == DiscoveryType.SERVICE_DISCOVERY:
                    task = asyncio.create_task(self.discover_services(request))
                elif request.discovery_type == DiscoveryType.CAPABILITY_DISCOVERY:
                    task = asyncio.create_task(self.discover_capabilities(request))
                else:
                    logger.warning(f"Unknown discovery type: {request.discovery_type}")
                    continue

                tasks.append(task)

            # Führe alle Tasks parallel aus
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filtere erfolgreiche Results
            successful_results = []
            for result in results:
                if isinstance(result, DiscoveryResult):
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Discovery task failed: {result}")

            total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Parallel discovery completed: {len(successful_results)}/{len(requests)} successful in {total_time_ms:.1f}ms")

            return successful_results

        except Exception as e:
            logger.error(f"Parallel discovery failed: {e}")
            return []

    # Helper Methods

    def _generate_cache_key(self, request: DiscoveryRequest) -> str:
        """Generiert Cache-Key für Discovery Request."""
        key_parts = [
            request.discovery_type.value,
            request.query,
            str(request.max_results),
            ",".join(sorted(request.required_capabilities)),
            ",".join(sorted(request.excluded_agents)),
            request.region_preference or "any"
        ]
        return ":".join(key_parts)

    async def _get_cached_result(self, cache_key: str) -> DiscoveryResult | None:
        """Holt gecachtes Discovery Result."""
        if not self.settings.cache_enabled:
            return None

        if cache_key in self._discovery_cache:
            result, cached_at = self._discovery_cache[cache_key]

            # Prüfe TTL
            if datetime.utcnow() - cached_at < timedelta(seconds=self.settings.cache_ttl_seconds):
                result.cache_hit = True
                return result
            # Abgelaufen - entferne aus Cache
            del self._discovery_cache[cache_key]

        return None

    async def _cache_result(self, cache_key: str, result: DiscoveryResult) -> None:
        """Cached Discovery Result."""
        if not self.settings.cache_enabled:
            return

        self._discovery_cache[cache_key] = (result, datetime.utcnow())

        # Cache-Größe begrenzen (einfache LRU)
        if len(self._discovery_cache) > self.settings.cache_max_size:
            # Entferne ältesten Eintrag
            oldest_key = min(self._discovery_cache.keys(),
                           key=lambda k: self._discovery_cache[k][1])
            del self._discovery_cache[oldest_key]

    def _update_stats(
        self,
        discovery_type: DiscoveryType,
        time_ms: float,
        success: bool,
        cache_hit: bool = False
    ) -> None:
        """Aktualisiert Discovery-Statistiken."""
        stats = self._discovery_stats[discovery_type.value]

        stats["total_requests"] += 1
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1

        if cache_hit:
            stats["cache_hits"] += 1

        # Update average time (exponential moving average)
        alpha = 0.1
        stats["average_time_ms"] = (
            alpha * time_ms + (1 - alpha) * stats["average_time_ms"]
        )

    async def _sequential_agent_discovery(self, request: DiscoveryRequest) -> list[DiscoveryResult]:
        """Fallback zu sequenzieller Agent Discovery."""
        results = []
        for registry in self._agent_registries[:1]:  # Nur erste Registry
            result = await self._discover_from_agent_registry(registry, request)
            results.append(result)
        return results

    async def _sequential_tool_discovery(self, request: DiscoveryRequest) -> list[DiscoveryResult]:
        """Fallback zu sequenzieller Tool Discovery."""
        return await self._parallel_tool_discovery(request)  # Gleiche Implementation

    async def _merge_agent_results(
        self,
        results: list[DiscoveryResult],
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Merged Agent Discovery Results."""
        all_items = []
        all_confidence_scores = []
        total_time = 0.0

        for result in results:
            all_items.extend(result.items)
            all_confidence_scores.extend(result.confidence_scores)
            total_time += result.discovery_time_ms

        # Dedupliziere und sortiere nach Confidence
        unique_items = {}
        for item in all_items:
            agent_id = item.get("agent_id", item.get("name", "unknown"))
            if agent_id not in unique_items or item.get("confidence", 0) > unique_items[agent_id].get("confidence", 0):
                unique_items[agent_id] = item

        final_items = list(unique_items.values())
        final_items.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Begrenze auf max_results
        final_items = final_items[:request.max_results]
        final_confidence_scores = [item.get("confidence", 0) for item in final_items]

        return DiscoveryResult(
            discovery_type=DiscoveryType.AGENT_DISCOVERY,
            query=request.query,
            items=final_items,
            total_found=len(final_items),
            confidence_scores=final_confidence_scores,
            discovery_time_ms=total_time / len(results) if results else 0.0,
            parallel_executions=len(results),
            average_confidence=sum(final_confidence_scores) / len(final_confidence_scores) if final_confidence_scores else 0.0,
            best_match_confidence=max(final_confidence_scores) if final_confidence_scores else 0.0
        )

    async def _merge_tool_results(
        self,
        results: list[DiscoveryResult],
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Merged Tool Discovery Results."""
        # Ähnlich zu _merge_agent_results aber für Tools
        all_items = []
        all_confidence_scores = []

        for result in results:
            all_items.extend(result.items)
            all_confidence_scores.extend(result.confidence_scores)

        # Sortiere nach Confidence
        all_items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        final_items = all_items[:request.max_results]
        final_confidence_scores = [item.get("confidence", 0) for item in final_items]

        return DiscoveryResult(
            discovery_type=DiscoveryType.TOOL_DISCOVERY,
            query=request.query,
            items=final_items,
            total_found=len(final_items),
            confidence_scores=final_confidence_scores,
            discovery_time_ms=results[0].discovery_time_ms if results else 0.0,
            parallel_executions=len(results),
            average_confidence=sum(final_confidence_scores) / len(final_confidence_scores) if final_confidence_scores else 0.0,
            best_match_confidence=max(final_confidence_scores) if final_confidence_scores else 0.0
        )

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Discovery-Statistiken zurück."""
        return {
            "discovery_stats": dict(self._discovery_stats),
            "cache_stats": {
                "cache_size": len(self._discovery_cache),
                "cache_enabled": self.settings.cache_enabled,
                "cache_max_size": self.settings.cache_max_size,
                "cache_ttl_seconds": self.settings.cache_ttl_seconds
            },
            "concurrency_limits": {
                "max_concurrent_discoveries": self.settings.max_concurrent_discoveries,
                "max_concurrent_agents": self.settings.max_concurrent_agents,
                "max_concurrent_tools": self.settings.max_concurrent_tools
            }
        }
