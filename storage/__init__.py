# backend/storage/__init__.py
"""Storage-System mit Azure Blob Storage, Redis-Cache und Query-Optimierung."""

from __future__ import annotations

from kei_logging import get_logger

# Azure Blob Storage
from .azure_blob_storage import close_storage_clients, get_storage_client

# Redis Cache
from .cache import (
    CachedCosmosClient,
    CacheMetrics,
    cache_metrics,
    get_cache_client,
    get_cached_cosmos_container,
)

# Client Factory (für erweiterte Nutzung)
from .client_factory import client_factory

# Constants (für Konfiguration)
from .constants import CacheConfig, StorageConstants

# Query Optimization
from .query import (
    AdvancedQueryBuilder,
    BatchQueryExecutor,
    OptimizationStrategy,
    QueryBatch,
    QueryConstants,
    QueryPerformanceAnalyzer,
    QueryRequest,
    QueryResult,
    QueryType,
)

# Utilities (für erweiterte Nutzung)
from .utils import (
    build_cache_key,
    deserialize_value,
    serialize_value,
    validate_blob_name,
    validate_container_name,
)

logger = get_logger(__name__)

__version__ = "0.1.0"
__author__ = "Keiko Development Team"

# =====================================================================
# Exports
# =====================================================================

__all__ = [
    "AdvancedQueryBuilder",
    "BatchQueryExecutor",
    "CacheConfig",
    "CacheMetrics",
    # Cache
    "CachedCosmosClient",
    "OptimizationStrategy",
    "QueryBatch",
    "QueryConstants",
    "QueryPerformanceAnalyzer",
    "QueryRequest",
    "QueryResult",
    # Query Optimization
    "QueryType",
    "StorageConstants",
    "build_cache_key",
    "cache_metrics",
    # Extended API
    "client_factory",
    "close_storage_clients",
    "deserialize_value",
    "get_cache_client",
    "get_cached_cosmos_container",
    # Core Storage
    "get_storage_client",
    "serialize_value",
    "validate_blob_name",
    "validate_container_name",
]

logger.debug("Storage-Modul geladen v0.1.0")
