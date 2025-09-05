"""Agent Circuit Breaker Package.
Konsolidierte Circuit Breaker-Implementierung für alle Agent-Typen.
"""

from ..constants import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    DEFAULT_SUCCESS_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS,
    ERROR_CIRCUIT_BREAKER_OPEN,
    FALLBACK_AGENTS,
)
from .circuit_breaker import AgentCircuitBreaker, CircuitBreakerOpenError
from .interfaces import (
    AgentCallContext,
    AgentCircuitBreakerSettings,
    AgentExecutionResult,
    AgentType,
    CircuitBreakerConfig,
    CircuitBreakerState,
    FailureType,
    IAgentCircuitBreakerManager,
    IAgentCircuitBreakerService,
    ICacheStrategy,
    ICircuitBreaker,
    IFallbackStrategy,
    RecoveryStrategy,
)
from .manager import AgentCircuitBreakerManager
from .service import AgentCircuitBreakerService, create_agent_circuit_breaker_service
from .strategies import MemoryCacheStrategy, SimpleFallbackStrategy

__all__ = [
    # Core Classes
    "AgentCircuitBreaker",
    "AgentCircuitBreakerManager",
    "AgentCircuitBreakerService",
    "create_agent_circuit_breaker_service",

    # Interfaces
    "ICircuitBreaker",
    "IAgentCircuitBreakerManager",
    "IAgentCircuitBreakerService",
    "ICacheStrategy",
    "IFallbackStrategy",

    # Data Classes
    "CircuitBreakerConfig",
    "AgentCallContext",
    "AgentExecutionResult",
    "AgentCircuitBreakerSettings",

    # Enums
    "CircuitBreakerState",
    "AgentType",
    "FailureType",
    "RecoveryStrategy",

    # Strategies
    "SimpleFallbackStrategy",
    "MemoryCacheStrategy",

    # Exceptions
    "CircuitBreakerOpenError",

    # Constants
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RECOVERY_TIMEOUT_SECONDS",
    "DEFAULT_SUCCESS_THRESHOLD",
    "DEFAULT_TIMEOUT_SECONDS",
    "FALLBACK_AGENTS",
    "ERROR_CIRCUIT_BREAKER_OPEN"
]
