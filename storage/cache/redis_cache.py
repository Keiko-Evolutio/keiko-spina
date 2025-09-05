# backend/storage/cache/redis_cache.py
"""Redis-Cache für Cosmos DB Performance-Optimierung."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

try:
    import redis.asyncio as aioredis
    from azure.cosmos.aio import ContainerProxy
    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    aioredis = Any  # type: ignore
    ContainerProxy = Any  # type: ignore
    _REDIS_AVAILABLE = False

from kei_logging import get_logger
from monitoring.metrics_definitions import (
    KEIKO_CACHE_OP_LATENCY,
    record_cache_hit,
    record_cache_miss,
)

from ..client_factory import NoOpCache, client_factory
from ..utils import (
    build_cache_key,
    calculate_cache_ttl,
    deserialize_value,
    handle_storage_errors,
    serialize_value,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


async def get_cache_client() -> aioredis.Redis:
    """Gibt Cache-Client zurück (Redis oder Fallback)."""
    return await client_factory.get_redis_client()


async def close_redis_connection() -> None:
    """Schließt Redis-Verbindung."""
    await client_factory.close_all_clients()


# =====================================================================
# Cache Metrics - Vereinfacht
# =====================================================================

class CacheMetrics:
    """Cache-Metriken - Vereinfacht."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0

    def record_hit(self, latency: float = 0.0) -> None:
        self.hits += 1

    def record_miss(self, latency: float = 0.0) -> None:
        self.misses += 1

    def record_set(self) -> None:
        self.sets += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "hit_rate": self.hit_rate,
        }


# Globale Metriken-Instanz
cache_metrics = CacheMetrics()


# =====================================================================
# Cached Cosmos Client - Vereinfacht
# =====================================================================

class CachedCosmosClient:
    """Redis-cached Cosmos DB Client."""

    def __init__(self, container: ContainerProxy):
        self.container = container
        self._redis: aioredis.Redis | None = None

    async def _ensure_redis(self) -> None:
        """Stellt Redis-Client sicher."""
        if self._redis is None:
            self._redis = await get_cache_client()

    @handle_storage_errors("get_item")
    async def get_item(
        self,
        item_id: str,
        partition_key: str,
        *,
        cache_type: str = "default",
        ttl: int | None = None,
        bypass_cache: bool = False
    ) -> dict[str, Any] | None:
        """Lädt Item mit Cache-First-Strategie."""
        if bypass_cache:
            return await self._load_from_cosmos(item_id, partition_key)

        cache_key = build_cache_key(cache_type, f"{partition_key}:{item_id}")

        # Cache-Lookup versuchen
        cached_item = await self._try_cache_lookup(cache_key, cache_type)
        if cached_item is not None:
            return cached_item

        # Von Cosmos DB laden und cachen
        item = await self._load_from_cosmos(item_id, partition_key)
        if item:
            await self._cache_item(cache_key, item, cache_type, ttl)

        return item

    async def _try_cache_lookup(self, cache_key: str, cache_type: str) -> dict[str, Any] | None:
        """Versucht Item aus Cache zu laden."""
        await self._ensure_redis()

        if isinstance(self._redis, NoOpCache):
            cache_metrics.record_miss()
            record_cache_miss(cache_type, tenant_id="-")
            return None

        try:
            with KEIKO_CACHE_OP_LATENCY.labels(cache_type=cache_type, operation="get").time():
                cached = await self._redis.get(cache_key)

            if cached:
                cache_metrics.record_hit()
                record_cache_hit(cache_type, tenant_id="-")
                return deserialize_value(cached)
        except Exception as e:
            logger.warning(f"Cache-Get fehlgeschlagen: {e}")

        cache_metrics.record_miss()
        record_cache_miss(cache_type, tenant_id="-")
        return None

    @handle_storage_errors("get_items_batch")
    async def get_items_batch(
        self,
        item_requests: list[dict[str, str]],
        *,
        cache_type: str = "default",
        ttl: int | None = None
    ) -> dict[str, dict[str, Any] | None]:
        """Lädt mehrere Items mit Batch-Caching."""
        cache_keys = [
            build_cache_key(cache_type, f"{req['partition_key']}:{req['item_id']}")
            for req in item_requests
        ]

        # Batch-Cache-Lookup
        results, missing_requests = await self._batch_cache_lookup(
            cache_keys, item_requests, cache_type
        )

        # Fehlende Items aus Cosmos DB laden
        if missing_requests:
            await self._load_missing_items(missing_requests, results, cache_type, ttl)

        return results

    async def _batch_cache_lookup(
        self,
        cache_keys: list[str],
        item_requests: list[dict[str, str]],
        cache_type: str
    ) -> tuple[dict[str, dict[str, Any] | None], list[tuple[str, dict[str, str]]]]:
        """Führt Batch-Cache-Lookup durch."""
        await self._ensure_redis()
        results = {}
        missing_requests = []

        if isinstance(self._redis, NoOpCache):
            missing_requests = [(cache_keys[i], req) for i, req in enumerate(item_requests)]
            for cache_key in cache_keys:
                results[cache_key] = None
                cache_metrics.record_miss()
            return results, missing_requests

        try:
            with KEIKO_CACHE_OP_LATENCY.labels(cache_type=cache_type, operation="mget").time():
                cached_values = await self._redis.mget(cache_keys)

            for i, (cache_key, cached_value) in enumerate(zip(cache_keys, cached_values, strict=False)):
                if cached_value:
                    results[cache_key] = deserialize_value(cached_value)
                    cache_metrics.record_hit()
                else:
                    results[cache_key] = None
                    missing_requests.append((cache_key, item_requests[i]))
                    cache_metrics.record_miss()
        except Exception as e:
            logger.warning(f"Batch-Cache-Lookup fehlgeschlagen: {e}")
            missing_requests = [(cache_keys[i], req) for i, req in enumerate(item_requests)]
            for cache_key in cache_keys:
                results[cache_key] = None

        return results, missing_requests

    async def _load_missing_items(
        self,
        missing_requests: list[tuple[str, dict[str, str]]],
        results: dict[str, dict[str, Any] | None],
        cache_type: str,
        ttl: int | None
    ) -> None:
        """Lädt fehlende Items aus Cosmos DB und aktualisiert Cache."""
        cosmos_items = await self._load_from_cosmos_batch([req for _, req in missing_requests])

        for (cache_key, _), item in zip(missing_requests, cosmos_items, strict=False):
            results[cache_key] = item
            if item:
                await self._cache_item(cache_key, item, cache_type, ttl)

    async def _load_from_cosmos(self, item_id: str, partition_key: str) -> dict[str, Any] | None:
        """Lädt Item direkt aus Cosmos DB mit ETag/Optimistic Locking Support."""
        try:
            return await self.container.read_item(item=item_id, partition_key=partition_key)
        except Exception as e:
            if "NotFound" in str(e):
                return None
            from ..constants import ErrorMessages
            logger.exception(f"{ErrorMessages.COSMOS_READ_FAILED}: {e}")
            raise

    async def _load_from_cosmos_batch(self, item_requests: list[dict[str, str]]) -> list[dict[str, Any] | None]:
        """Lädt mehrere Items parallel aus Cosmos DB."""
        tasks = [
            self._load_from_cosmos(req["item_id"], req["partition_key"])
            for req in item_requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _cache_item(
        self,
        cache_key: str,
        item: dict[str, Any],
        cache_type: str,
        ttl: int | None
    ) -> None:
        """Speichert Item im Cache."""
        if isinstance(self._redis, NoOpCache):
            return

        try:
            effective_ttl = calculate_cache_ttl(cache_type, ttl)
            with KEIKO_CACHE_OP_LATENCY.labels(cache_type=cache_type, operation="setex").time():
                await self._redis.setex(cache_key, effective_ttl, serialize_value(item))
            cache_metrics.record_set()
        except Exception as e:
            from ..constants import ErrorMessages
            logger.exception(f"{ErrorMessages.CACHE_OPERATION_FAILED}: {e}")

    @handle_storage_errors("upsert_item")
    async def upsert_item(
        self,
        item: dict[str, Any],
        *,
        partition_key: str,
        if_match_etag: str | None = None,
        cache_type: str = "default",
        ttl: int | None = None,
    ) -> dict[str, Any]:
        """Schreibt Item mit Optimistic Locking und aktualisiert Cache."""
        # Cosmos DB Operation
        result = await self._perform_cosmos_write(item, partition_key, if_match_etag)

        # Cache aktualisieren
        cache_key = build_cache_key(cache_type, f"{partition_key}:{item.get('id')}")
        await self._cache_item(cache_key, result, cache_type, ttl)

        return result

    async def _perform_cosmos_write(
        self,
        item: dict[str, Any],
        partition_key: str,
        if_match_etag: str | None
    ) -> dict[str, Any]:
        """Führt Cosmos DB Write-Operation durch."""
        try:
            if if_match_etag:
                headers = {"If-Match": if_match_etag}
                return await self.container.replace_item(
                    item=item, body=item, headers=headers  # type: ignore
                )
            return await self.container.upsert_item(
                body=item, partition_key=partition_key  # type: ignore
            )
        except Exception as e:
            # ETag Mismatch propagieren
            if "PreconditionFailed" in str(e) or "ETag" in str(e):
                raise
            logger.exception(f"Cosmos upsert/replace fehlgeschlagen: {e}")
            raise


# =====================================================================
# Public API Functions
# =====================================================================

@asynccontextmanager
async def get_cached_cosmos_container() -> AsyncIterator[CachedCosmosClient | None]:
    """Context Manager für cached Cosmos Container."""
    from voice.common.common import get_cosmos_container

    async with get_cosmos_container() as container:
        if container:
            yield CachedCosmosClient(container)
        else:
            yield None
