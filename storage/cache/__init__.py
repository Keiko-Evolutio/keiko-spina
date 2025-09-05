# backend/storage/cache/__init__.py
"""Redis-Cache für Cosmos DB Performance."""

from .redis_cache import (
    CachedCosmosClient,
    CacheMetrics,
    cache_metrics,
    get_cache_client,
    get_cached_cosmos_container,
)

__all__ = [
    "CacheMetrics",
    "CachedCosmosClient",
    "cache_metrics",
    "get_cache_client",
    "get_cached_cosmos_container",
]
