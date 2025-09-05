# backend/services/enhanced_dependency_resolution/dependency_cache_engine.py
"""Dependency Cache Engine für Performance-Optimierung.

Implementiert intelligentes Caching für Dependency-Resolution mit
Invalidation-Strategien und Performance-Optimierung.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .data_models import (
    DependencyCache,
    DependencyGraph,
    DependencyResolutionRequest,
    DependencyResolutionResult,
)

logger = get_logger(__name__)


class DependencyCacheEngine:
    """Dependency Cache Engine für Performance-Optimierung."""

    def __init__(self):
        """Initialisiert Dependency Cache Engine."""
        # Cache-Konfiguration
        self.enable_caching = True
        self.enable_intelligent_invalidation = True
        self.enable_cache_warming = True
        self.default_ttl_seconds = 3600  # 1 Stunde
        self.max_cache_size = 10000

        # Cache-Storage
        self._resolution_cache: dict[str, DependencyCache] = {}
        self._graph_cache: dict[str, dict[str, Any]] = {}
        self._analysis_cache: dict[str, dict[str, Any]] = {}

        # Cache-Statistiken
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_invalidations = 0
        self._cache_evictions = 0

        # Invalidation-Tracking
        self._dependency_tracking: dict[str, set[str]] = defaultdict(set)
        self._graph_versions: dict[str, str] = {}

        # Background-Tasks
        self._cleanup_task: asyncio.Task | None = None
        self._warming_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Dependency Cache Engine initialisiert")

    async def start(self) -> None:
        """Startet Dependency Cache Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if self.enable_cache_warming:
            self._warming_task = asyncio.create_task(self._cache_warming_loop())

        logger.info("Dependency Cache Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Dependency Cache Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._warming_task:
            self._warming_task.cancel()

        await asyncio.gather(
            self._cleanup_task,
            self._warming_task,
            return_exceptions=True
        )

        logger.info("Dependency Cache Engine gestoppt")

    async def get_cached_resolution(
        self,
        request: DependencyResolutionRequest
    ) -> DependencyResolutionResult | None:
        """Holt Cached Resolution-Result.

        Args:
            request: Dependency-Resolution-Request

        Returns:
            Cached Resolution-Result oder None
        """
        try:
            if not self.enable_caching:
                return None

            cache_key = self._create_resolution_cache_key(request)
            cache_entry = self._resolution_cache.get(cache_key)

            if not cache_entry:
                self._cache_misses += 1
                return None

            # Prüfe TTL
            if datetime.utcnow() > cache_entry.expires_at:
                # Cache abgelaufen
                del self._resolution_cache[cache_key]
                self._cache_misses += 1
                return None

            # Prüfe Graph-Version
            if not self._is_cache_valid(cache_entry):
                # Cache invalidiert
                del self._resolution_cache[cache_key]
                self._cache_invalidations += 1
                self._cache_misses += 1
                return None

            # Cache-Hit
            cache_entry.access_count += 1
            cache_entry.last_accessed = datetime.utcnow()
            self._cache_hits += 1

            logger.debug({
                "event": "cache_hit",
                "cache_key": cache_key,
                "access_count": cache_entry.access_count
            })

            return cache_entry.resolution_result

        except Exception as e:
            logger.error(f"Cache lookup fehlgeschlagen: {e}")
            self._cache_misses += 1
            return None

    async def cache_resolution_result(
        self,
        request: DependencyResolutionRequest,
        result: DependencyResolutionResult,
        ttl_seconds: int | None = None
    ) -> None:
        """Cached Resolution-Result.

        Args:
            request: Dependency-Resolution-Request
            result: Dependency-Resolution-Result
            ttl_seconds: Time-to-Live in Sekunden
        """
        try:
            if not self.enable_caching or not result.success:
                return

            cache_key = self._create_resolution_cache_key(request)
            ttl = ttl_seconds or self.default_ttl_seconds

            # Erstelle Cache-Entry
            cache_entry = DependencyCache(
                cache_key=cache_key,
                graph_id=request.graph_id,
                resolution_result=result,
                ttl_seconds=ttl,
                cached_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl)
            )

            # Bestimme Invalidation-Triggers
            invalidation_triggers = self._determine_invalidation_triggers(request)
            cache_entry.invalidation_triggers = invalidation_triggers

            # Speichere Cache-Entry
            self._resolution_cache[cache_key] = cache_entry

            # Tracke Dependencies für Invalidation
            for trigger in invalidation_triggers:
                self._dependency_tracking[trigger].add(cache_key)

            # Prüfe Cache-Größe
            await self._enforce_cache_size_limit()

            logger.debug({
                "event": "cache_stored",
                "cache_key": cache_key,
                "ttl_seconds": ttl,
                "invalidation_triggers": len(invalidation_triggers)
            })

        except Exception as e:
            logger.error(f"Cache storage fehlgeschlagen: {e}")

    async def invalidate_cache(
        self,
        graph_id: str | None = None,
        node_id: str | None = None,
        edge_id: str | None = None,
        trigger: str | None = None
    ) -> int:
        """Invalidiert Cache-Einträge.

        Args:
            graph_id: Graph-ID für Invalidation
            node_id: Node-ID für Invalidation
            edge_id: Edge-ID für Invalidation
            trigger: Spezifischer Trigger

        Returns:
            Anzahl invalidierter Cache-Einträge
        """
        try:
            invalidated_count = 0

            # Bestimme Invalidation-Triggers
            triggers_to_invalidate = set()

            if trigger:
                triggers_to_invalidate.add(trigger)

            if graph_id:
                triggers_to_invalidate.add(f"graph:{graph_id}")

                # Aktualisiere Graph-Version
                self._graph_versions[graph_id] = self._generate_version_hash()

            if node_id:
                triggers_to_invalidate.add(f"node:{node_id}")

            if edge_id:
                triggers_to_invalidate.add(f"edge:{edge_id}")

            # Invalidiere betroffene Cache-Einträge
            cache_keys_to_remove = set()

            for trigger_key in triggers_to_invalidate:
                if trigger_key in self._dependency_tracking:
                    cache_keys_to_remove.update(self._dependency_tracking[trigger_key])
                    del self._dependency_tracking[trigger_key]

            # Entferne Cache-Einträge
            for cache_key in cache_keys_to_remove:
                if cache_key in self._resolution_cache:
                    del self._resolution_cache[cache_key]
                    invalidated_count += 1

            self._cache_invalidations += invalidated_count

            if invalidated_count > 0:
                logger.debug({
                    "event": "cache_invalidated",
                    "triggers": list(triggers_to_invalidate),
                    "invalidated_count": invalidated_count
                })

            return invalidated_count

        except Exception as e:
            logger.error(f"Cache invalidation fehlgeschlagen: {e}")
            return 0

    async def warm_cache(
        self,
        graph: DependencyGraph,
        common_requests: list[DependencyResolutionRequest]
    ) -> int:
        """Wärmt Cache mit häufigen Requests vor.

        Args:
            graph: Dependency-Graph
            common_requests: Häufige Requests

        Returns:
            Anzahl vorgewärmter Cache-Einträge
        """
        try:
            if not self.enable_cache_warming:
                return 0

            warmed_count = 0

            for request in common_requests:
                # Prüfe ob bereits gecacht
                cache_key = self._create_resolution_cache_key(request)
                if cache_key in self._resolution_cache:
                    continue

                # Simuliere Resolution für Cache-Warming
                # In Realität würde hier die echte Resolution aufgerufen
                mock_result = DependencyResolutionResult(
                    result_id=f"warmed_{cache_key}",
                    request_id=request.request_id,
                    success=True,
                    resolution_order=request.target_nodes,
                    resolved_nodes=set(request.target_nodes)
                )

                await self.cache_resolution_result(request, mock_result)
                warmed_count += 1

            logger.debug({
                "event": "cache_warmed",
                "graph_id": graph.graph_id,
                "warmed_count": warmed_count
            })

            return warmed_count

        except Exception as e:
            logger.error(f"Cache warming fehlgeschlagen: {e}")
            return 0

    async def get_cache_statistics(self) -> dict[str, Any]:
        """Holt Cache-Statistiken.

        Returns:
            Cache-Statistiken
        """
        try:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
            miss_rate = self._cache_misses / total_requests if total_requests > 0 else 0.0

            # Cache-Größen
            resolution_cache_size = len(self._resolution_cache)
            graph_cache_size = len(self._graph_cache)
            analysis_cache_size = len(self._analysis_cache)

            # Memory-Usage (approximiert)
            estimated_memory_mb = (
                resolution_cache_size * 0.01 +  # ~10KB pro Resolution-Cache-Entry
                graph_cache_size * 0.005 +      # ~5KB pro Graph-Cache-Entry
                analysis_cache_size * 0.002     # ~2KB pro Analysis-Cache-Entry
            )

            # Cache-Effizienz
            cache_efficiency = hit_rate * 100 if hit_rate > 0 else 0.0

            return {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "cache_efficiency": cache_efficiency,
                "cache_invalidations": self._cache_invalidations,
                "cache_evictions": self._cache_evictions,
                "resolution_cache_size": resolution_cache_size,
                "graph_cache_size": graph_cache_size,
                "analysis_cache_size": analysis_cache_size,
                "total_cache_entries": resolution_cache_size + graph_cache_size + analysis_cache_size,
                "estimated_memory_mb": estimated_memory_mb,
                "max_cache_size": self.max_cache_size,
                "cache_utilization": (resolution_cache_size / self.max_cache_size) * 100,
                "caching_enabled": self.enable_caching,
                "intelligent_invalidation_enabled": self.enable_intelligent_invalidation,
                "cache_warming_enabled": self.enable_cache_warming,
                "default_ttl_seconds": self.default_ttl_seconds
            }

        except Exception as e:
            logger.error(f"Cache statistics generation fehlgeschlagen: {e}")
            return {}

    def _create_resolution_cache_key(self, request: DependencyResolutionRequest) -> str:
        """Erstellt Cache-Key für Resolution-Request."""
        try:
            # Erstelle deterministischen Hash aus Request-Parametern
            key_components = [
                request.graph_id,
                "|".join(sorted(request.target_nodes)),
                request.resolution_strategy.value,
                str(request.max_depth),
                str(request.include_optional),
                str(request.break_circular),
                request.security_level.value if request.security_level else "none",
                request.user_id or "anonymous",
                request.tenant_id or "no_tenant"
            ]

            key_string = "|".join(key_components)
            cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]

            return f"resolution:{cache_key}"

        except Exception as e:
            logger.error(f"Cache key creation fehlgeschlagen: {e}")
            return f"resolution:error_{time.time()}"

    def _determine_invalidation_triggers(self, request: DependencyResolutionRequest) -> set[str]:
        """Bestimmt Invalidation-Triggers für Request."""
        try:
            triggers = set()

            # Graph-Level Trigger
            triggers.add(f"graph:{request.graph_id}")

            # Node-Level Triggers
            for node_id in request.target_nodes:
                triggers.add(f"node:{node_id}")

            # User/Tenant-Level Triggers (für Security-Context)
            if request.user_id:
                triggers.add(f"user:{request.user_id}")

            if request.tenant_id:
                triggers.add(f"tenant:{request.tenant_id}")

            return triggers

        except Exception as e:
            logger.error(f"Invalidation triggers determination fehlgeschlagen: {e}")
            return set()

    def _is_cache_valid(self, cache_entry: DependencyCache) -> bool:
        """Prüft ob Cache-Entry noch gültig ist."""
        try:
            # Prüfe Graph-Version
            graph_id = cache_entry.graph_id
            current_version = self._graph_versions.get(graph_id)
            cached_version = cache_entry.metadata.get("graph_version")

            if current_version and cached_version and current_version != cached_version:
                return False

            # Prüfe Invalidation-Triggers
            for trigger in cache_entry.invalidation_triggers:
                if trigger not in self._dependency_tracking:
                    # Trigger wurde invalidiert
                    return False

            return True

        except Exception as e:
            logger.error(f"Cache validity check fehlgeschlagen: {e}")
            return False

    def _generate_version_hash(self) -> str:
        """Generiert Version-Hash."""
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]

    async def _enforce_cache_size_limit(self) -> None:
        """Erzwingt Cache-Größen-Limit."""
        try:
            if len(self._resolution_cache) <= self.max_cache_size:
                return

            # Sortiere Cache-Einträge nach LRU (Least Recently Used)
            cache_items = list(self._resolution_cache.items())
            cache_items.sort(key=lambda x: x[1].last_accessed)

            # Entferne älteste Einträge
            entries_to_remove = len(cache_items) - self.max_cache_size + 100  # Puffer

            for i in range(entries_to_remove):
                cache_key, cache_entry = cache_items[i]

                # Entferne aus Cache
                del self._resolution_cache[cache_key]

                # Entferne aus Dependency-Tracking
                for trigger in cache_entry.invalidation_triggers:
                    if trigger in self._dependency_tracking:
                        self._dependency_tracking[trigger].discard(cache_key)

                self._cache_evictions += 1

            logger.debug({
                "event": "cache_size_enforced",
                "evicted_entries": entries_to_remove,
                "current_size": len(self._resolution_cache)
            })

        except Exception as e:
            logger.error(f"Cache size enforcement fehlgeschlagen: {e}")

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cache-Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Cleanup alle 5 Minuten

                if self._is_running:
                    await self._cleanup_expired_entries()

            except Exception as e:
                logger.error(f"Cache cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _cache_warming_loop(self) -> None:
        """Background-Loop für Cache-Warming."""
        while self._is_running:
            try:
                await asyncio.sleep(1800)  # Warming alle 30 Minuten

                if self._is_running:
                    await self._perform_intelligent_warming()

            except Exception as e:
                logger.error(f"Cache warming loop fehlgeschlagen: {e}")
                await asyncio.sleep(1800)

    async def _cleanup_expired_entries(self) -> None:
        """Bereinigt abgelaufene Cache-Einträge."""
        try:
            current_time = datetime.utcnow()
            expired_keys = []

            for cache_key, cache_entry in self._resolution_cache.items():
                if current_time > cache_entry.expires_at:
                    expired_keys.append(cache_key)

            # Entferne abgelaufene Einträge
            for cache_key in expired_keys:
                cache_entry = self._resolution_cache[cache_key]
                del self._resolution_cache[cache_key]

                # Entferne aus Dependency-Tracking
                for trigger in cache_entry.invalidation_triggers:
                    if trigger in self._dependency_tracking:
                        self._dependency_tracking[trigger].discard(cache_key)

            if expired_keys:
                logger.debug(f"Cache cleanup: {len(expired_keys)} abgelaufene Einträge entfernt")

        except Exception as e:
            logger.error(f"Expired entries cleanup fehlgeschlagen: {e}")

    async def _perform_intelligent_warming(self) -> None:
        """Führt intelligentes Cache-Warming durch."""
        try:
            # Analysiere Cache-Access-Patterns
            access_patterns = defaultdict(int)

            for cache_entry in self._resolution_cache.values():
                # Tracke häufig verwendete Patterns
                pattern_key = f"{cache_entry.graph_id}:{len(cache_entry.resolution_result.resolved_nodes)}"
                access_patterns[pattern_key] += cache_entry.access_count

            # Identifiziere Top-Patterns für Warming
            top_patterns = sorted(
                access_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            logger.debug({
                "event": "intelligent_cache_warming",
                "top_patterns": len(top_patterns),
                "total_patterns": len(access_patterns)
            })

        except Exception as e:
            logger.error(f"Intelligent cache warming fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_invalidations": self._cache_invalidations,
            "cache_evictions": self._cache_evictions,
            "cache_size": len(self._resolution_cache),
            "cache_efficiency": hit_rate * 100,
            "caching_enabled": self.enable_caching,
            "intelligent_invalidation_enabled": self.enable_intelligent_invalidation,
            "cache_warming_enabled": self.enable_cache_warming
        }
