# backend/services/pools/__init__.py
"""Resource Pools Paket für Keiko Personal Assistant.

Bietet wiederverwendbare Resource-Pool-Implementierungen für
HTTP-Clients und andere Ressourcen mit automatischem Management,
Health-Monitoring und Konfiguration.
"""

from .azure_pools import (
    BaseResourcePool,
    HTTPClientPool,
    PoolConfig,
    PoolHealth,
    PoolManager,
    PoolManagerConfig,
    PoolMetrics,
    cleanup_pools,
    get_health_status,
    get_http_client,
    initialize_pools,
)
from .worker_patterns import (
    BaseWorker,
    BaseWorkerPool,
    WorkerConfig,
    WorkerMetrics,
    WorkerPoolConfig,
    WorkerState,
)

__all__ = [
    # Resource Pools
    "BaseResourcePool",
    "BaseWorker",
    "BaseWorkerPool",
    "HTTPClientPool",
    "PoolConfig",
    "PoolHealth",
    "PoolManager",
    "PoolManagerConfig",
    "PoolMetrics",
    "WorkerConfig",
    "WorkerMetrics",
    "WorkerPoolConfig",
    # Worker Patterns
    "WorkerState",
    "cleanup_pools",
    "get_health_status",
    "get_http_client",
    "initialize_pools",
]
