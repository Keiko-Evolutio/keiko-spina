# backend/agents/resilience/__init__.py
"""Resilienz- und Performance-Features für Personal Assistant

Implementiert Resilienz-Patterns mit Capability-spezifischen Circuit Breakern,
Retry-Mechanismen und Request-Budget-Management.
"""

from .async_circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from .circuit_breaker import (
    CapabilityCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerMetrics,
    CircuitBreakerState,
)

# Kompatibilitäts-Aliase
CircuitBreaker = CapabilityCircuitBreaker
CircuitBreakerError = CircuitBreakerOpenError

from .retry_manager import (
    AdaptiveRetryManager,
    RetryConfig,
    RetryExhaustedError,
    RetryManager,
    RetryStrategy,
    UpstreamRetryPolicy,
)

# Kompatibilitäts-Aliase
RetryPolicy = UpstreamRetryPolicy
RetryError = RetryExhaustedError

from .deadline_manager import (
    DeadlineConfig,
    DeadlineManager,
    DeadlineType,
    DeadlineViolationError,
    RequestDeadline,
    TimeoutHandler,
    TimeoutStrategy,
)
from .performance_monitor import (
    AlertManager,
    CapabilityMetrics,
    PerformanceMonitor,
    ResilienceMetrics,
    UpstreamMetrics,
)
from .request_budgets import (
    BudgetConfig,
    BudgetExhaustedError,
    BudgetExhaustionHandler,
    BudgetManager,
    DeadlineExceededError,
    RequestBudget,
    ResourceTracker,
)
from .resilience_coordinator import ResilienceConfig, ResilienceCoordinator, ResiliencePolicy

# Versions-Information
__version__ = "1.0.0"

# Öffentliche APIs
__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CapabilityCircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerManager",
    "CircuitBreakerMetrics",
    # Async Circuit Breaker
    "AsyncCircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    # Retry Manager
    "RetryManager",
    "RetryStrategy",
    "RetryConfig",
    "RetryPolicy",
    "RetryError",
    "UpstreamRetryPolicy",
    "AdaptiveRetryManager",
    "RetryExhaustedError",
    # Request Budgets
    "RequestBudget",
    "BudgetManager",
    "BudgetConfig",
    "BudgetExhaustionHandler",
    "ResourceTracker",
    "BudgetExhaustedError",
    "DeadlineExceededError",
    # Performance Monitor
    "PerformanceMonitor",
    "CapabilityMetrics",
    "UpstreamMetrics",
    "ResilienceMetrics",
    "AlertManager",
    # Deadline Manager
    "DeadlineManager",
    "RequestDeadline",
    "DeadlineConfig",
    "TimeoutHandler",
    "DeadlineType",
    "TimeoutStrategy",
    "DeadlineViolationError",
    # Resilience Coordinator
    "ResilienceCoordinator",
    "ResiliencePolicy",
    "ResilienceConfig",
    # Version
    "__version__",
]

# Logging-Konfiguration
from kei_logging import get_logger

logger = get_logger(__name__)
logger.info(f"Resilience Framework v{__version__} initialisiert")
