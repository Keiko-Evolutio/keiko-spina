"""Performance Cache Implementation für Voice Workflow Optimization.
Implementiert intelligentes Caching mit Predictive Loading und Warm-up.
"""

import asyncio
import hashlib
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .interfaces import (
    CacheEntry,
    CacheStrategy,
    IPerformanceCache,
    VoicePerformanceSettings,
    VoiceWorkflowContext,
)

logger = get_logger(__name__)


class PerformanceCacheEngine(IPerformanceCache):
    """Performance Cache Engine Implementation.
    Intelligentes Caching mit Predictive Loading und Pattern Learning.
    """

    def __init__(self, settings: VoicePerformanceSettings):
        self.settings = settings

        # Cache Storage (Thread-safe)
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = asyncio.Lock()

        # LRU Tracking
        self._access_order: OrderedDict[str, datetime] = OrderedDict()

        # Pattern Learning
        self._usage_patterns: dict[str, list[datetime]] = defaultdict(list)
        self._prediction_cache: dict[str, set[str]] = defaultdict(set)

        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_puts": 0,
            "cache_evictions": 0,
            "warm_up_operations": 0,
            "predictive_loads": 0,
            "pattern_predictions": 0
        }

        # Background Tasks
        self._cleanup_task: asyncio.Task | None = None
        self._pattern_learning_task: asyncio.Task | None = None
        self._running = False

        logger.info("Performance cache engine initialized")

    async def start(self) -> None:
        """Startet Background-Tasks."""
        if self._running:
            return

        self._running = True

        # Starte Cleanup-Task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Starte Pattern Learning Task
        if self.settings.pattern_learning_enabled:
            self._pattern_learning_task = asyncio.create_task(self._pattern_learning_loop())

        logger.info("Performance cache background tasks started")

    async def stop(self) -> None:
        """Stoppt Background-Tasks."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._pattern_learning_task:
            self._pattern_learning_task.cancel()
            try:
                await self._pattern_learning_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance cache background tasks stopped")

    async def get(self, key: str) -> CacheEntry | None:
        """Holt Cache-Eintrag."""
        if not self.settings.cache_enabled:
            return None

        async with self._cache_lock:
            if key not in self._cache:
                self._stats["cache_misses"] += 1
                return None

            entry = self._cache[key]

            # Prüfe Expiration
            if datetime.utcnow() > entry.expires_at:
                # Abgelaufen - entferne
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                self._stats["cache_misses"] += 1
                return None

            # Update Access-Tracking
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._access_order[key] = entry.last_accessed
            self._access_order.move_to_end(key)

            # Pattern Learning
            if self.settings.pattern_learning_enabled:
                self._record_access_pattern(key)

            self._stats["cache_hits"] += 1
            return entry

    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300,
        strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE
    ) -> None:
        """Speichert Cache-Eintrag."""
        if not self.settings.cache_enabled:
            return

        async with self._cache_lock:
            # Prüfe Cache-Größe und evict falls nötig
            await self._evict_if_needed()

            # Erstelle Cache-Eintrag
            now = datetime.utcnow()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
                ttl_seconds=ttl_seconds,
                strategy=strategy
            )

            # Speichere im Cache
            self._cache[key] = entry
            self._access_order[key] = now
            self._access_order.move_to_end(key)

            self._stats["cache_puts"] += 1

            logger.debug(f"Cached entry: {key} (TTL: {ttl_seconds}s, Strategy: {strategy.value})")

    async def invalidate(self, key: str) -> None:
        """Invalidiert Cache-Eintrag."""
        async with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_order:
                del self._access_order[key]

            logger.debug(f"Invalidated cache entry: {key}")

    async def warm_up(self, context: VoiceWorkflowContext) -> None:
        """Führt Cache Warm-up durch."""
        if not self.settings.warm_up_enabled:
            return

        start_time = time.time()

        try:
            # Warm-up basierend auf User-Kontext
            warm_up_keys = await self._generate_warm_up_keys(context)

            # Parallel Warm-up
            tasks = []
            for key in warm_up_keys:
                task = asyncio.create_task(self._warm_up_entry(key, context))
                tasks.append(task)

            # Warte auf alle Warm-up Tasks
            await asyncio.gather(*tasks, return_exceptions=True)

            warm_up_time = (time.time() - start_time) * 1000
            self._stats["warm_up_operations"] += 1

            logger.info(f"Cache warm-up completed: {len(warm_up_keys)} entries in {warm_up_time:.1f}ms")

        except Exception as e:
            logger.error(f"Cache warm-up failed: {e}")

    async def predict_and_preload(self, context: VoiceWorkflowContext) -> None:
        """Führt Predictive Preloading durch."""
        if not self.settings.predictive_loading_enabled:
            return

        start_time = time.time()

        try:
            # Generiere Predictions basierend auf Patterns
            predicted_keys = await self._predict_needed_keys(context)

            # Preload nur Keys die nicht bereits im Cache sind
            keys_to_load = []
            async with self._cache_lock:
                for key in predicted_keys:
                    if key not in self._cache:
                        keys_to_load.append(key)

            if not keys_to_load:
                return

            # Parallel Preloading
            tasks = []
            for key in keys_to_load:
                task = asyncio.create_task(self._preload_entry(key, context))
                tasks.append(task)

            # Warte auf alle Preload Tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_loads = sum(1 for result in results if not isinstance(result, Exception))
            preload_time = (time.time() - start_time) * 1000
            self._stats["predictive_loads"] += successful_loads

            logger.info(f"Predictive preloading completed: {successful_loads}/{len(keys_to_load)} successful in {preload_time:.1f}ms")

        except Exception as e:
            logger.error(f"Predictive preloading failed: {e}")

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        async with self._cache_lock:
            cache_size = len(self._cache)

            # Berechne Hit Rate
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
            hit_rate = self._stats["cache_hits"] / total_requests if total_requests > 0 else 0.0

            # Memory Usage (grobe Schätzung)
            memory_usage_mb = cache_size * 0.001  # Grobe Schätzung: 1KB pro Entry

            return {
                "cache_size": cache_size,
                "max_cache_size": self.settings.cache_max_size,
                "usage_percentage": (cache_size / self.settings.cache_max_size) * 100,
                "hit_rate": hit_rate,
                "memory_usage_mb": memory_usage_mb,
                "statistics": self._stats.copy(),
                "settings": {
                    "cache_enabled": self.settings.cache_enabled,
                    "cache_strategy": self.settings.cache_strategy.value,
                    "cache_ttl_seconds": self.settings.cache_ttl_seconds,
                    "predictive_loading_enabled": self.settings.predictive_loading_enabled,
                    "warm_up_enabled": self.settings.warm_up_enabled,
                    "pattern_learning_enabled": self.settings.pattern_learning_enabled
                }
            }

    # Private Methods

    async def _evict_if_needed(self) -> None:
        """Evict Cache-Einträge falls nötig."""
        if len(self._cache) < self.settings.cache_max_size:
            return

        # LRU Eviction
        entries_to_evict = len(self._cache) - self.settings.cache_max_size + 1

        for _ in range(entries_to_evict):
            if not self._access_order:
                break

            # Entferne ältesten Eintrag
            oldest_key, _ = self._access_order.popitem(last=False)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                self._stats["cache_evictions"] += 1

    async def _cleanup_loop(self) -> None:
        """Periodische Cleanup-Schleife."""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(300)  # Cleanup alle 5 Minuten (reduziert CPU-Last)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_entries(self) -> None:
        """Entfernt abgelaufene Cache-Einträge."""
        now = datetime.utcnow()
        expired_keys = []

        async with self._cache_lock:
            for key, entry in self._cache.items():
                if now > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _pattern_learning_loop(self) -> None:
        """Pattern Learning Background-Task."""
        while self._running:
            try:
                await self._analyze_usage_patterns()
                await asyncio.sleep(600)  # Analyse alle 10 Minuten (reduziert CPU-Last)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pattern learning error: {e}")
                await asyncio.sleep(300)

    def _record_access_pattern(self, key: str) -> None:
        """Zeichnet Access-Pattern für Key auf."""
        now = datetime.utcnow()
        self._usage_patterns[key].append(now)

        # Begrenze Pattern-History
        max_history = 100
        if len(self._usage_patterns[key]) > max_history:
            self._usage_patterns[key] = self._usage_patterns[key][-max_history:]

    async def _analyze_usage_patterns(self) -> None:
        """Analysiert Usage-Patterns für Predictions."""
        try:
            # Analysiere Patterns und generiere Predictions
            for key, access_times in self._usage_patterns.items():
                if len(access_times) < 3:
                    continue

                # Einfache Pattern-Analyse: Häufig zusammen verwendete Keys
                # In echter Implementation würde hier ML-basierte Analyse stehen
                related_keys = self._find_related_keys(key, access_times)
                self._prediction_cache[key] = related_keys

            self._stats["pattern_predictions"] += 1

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")

    def _find_related_keys(self, key: str, access_times: list[datetime]) -> set[str]:
        """Findet verwandte Keys basierend auf Access-Patterns."""
        related_keys = set()

        # Einfache Heuristik: Keys die oft in zeitlicher Nähe verwendet werden
        time_window = timedelta(minutes=5)

        for access_time in access_times[-10:]:  # Nur letzte 10 Zugriffe
            for other_key, other_access_times in self._usage_patterns.items():
                if other_key == key:
                    continue

                # Prüfe ob other_key in zeitlicher Nähe verwendet wurde
                for other_access_time in other_access_times:
                    if abs(access_time - other_access_time) <= time_window:
                        related_keys.add(other_key)
                        break

        return related_keys

    async def _generate_warm_up_keys(self, context: VoiceWorkflowContext) -> list[str]:
        """Generiert Keys für Cache Warm-up."""
        warm_up_keys = []

        # User-spezifische Keys
        warm_up_keys.extend([
            f"user_preferences:{context.user_id}",
            f"user_agents:{context.user_id}",
            f"user_tools:{context.user_id}",
            f"session_state:{context.session_id}"
        ])

        # Häufig verwendete Keys
        warm_up_keys.extend([
            "common_agents:voice",
            "common_tools:text_processing",
            "service_endpoints:primary",
            "capability_schemas:standard"
        ])

        return warm_up_keys

    async def _warm_up_entry(self, key: str, context: VoiceWorkflowContext) -> None:
        """Führt Warm-up für einzelnen Key durch."""
        try:
            # Mock Warm-up - in echter Implementation würde hier
            # der entsprechende Service/Registry abgefragt werden
            mock_value = {
                "warmed_up": True,
                "key": key,
                "context": context.workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.put(key, mock_value, ttl_seconds=600)  # Längere TTL für Warm-up

        except Exception as e:
            logger.debug(f"Warm-up failed for key {key}: {e}")

    async def _predict_needed_keys(self, context: VoiceWorkflowContext) -> list[str]:
        """Predicts benötigte Keys basierend auf Kontext."""
        predicted_keys = []

        # Basis-Predictions basierend auf User-Kontext
        base_key = f"user:{context.user_id}"
        if base_key in self._prediction_cache:
            predicted_keys.extend(self._prediction_cache[base_key])

        # Session-basierte Predictions
        session_key = f"session:{context.session_id}"
        if session_key in self._prediction_cache:
            predicted_keys.extend(self._prediction_cache[session_key])

        # Text-basierte Predictions
        if context.text_input:
            text_hash = hashlib.md5(context.text_input.encode()).hexdigest()[:8]
            text_key = f"text_pattern:{text_hash}"
            if text_key in self._prediction_cache:
                predicted_keys.extend(self._prediction_cache[text_key])

        return list(set(predicted_keys))  # Dedupliziere

    async def _preload_entry(self, key: str, context: VoiceWorkflowContext) -> None:
        """Führt Preloading für einzelnen Key durch."""
        try:
            # Mock Preloading
            mock_value = {
                "preloaded": True,
                "key": key,
                "predicted_for": context.workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.put(key, mock_value, ttl_seconds=300)

        except Exception as e:
            logger.debug(f"Preloading failed for key {key}: {e}")


def create_performance_cache(settings: VoicePerformanceSettings) -> PerformanceCacheEngine:
    """Factory-Funktion für Performance Cache."""
    return PerformanceCacheEngine(settings)
