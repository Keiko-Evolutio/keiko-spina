"""Agent Circuit Breaker Interfaces für Keiko Personal Assistant.
Definiert abstrakte Interfaces für Agent-spezifische Circuit Breaker.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class CircuitBreakerState(Enum):
    """Circuit Breaker States."""
    CLOSED = "closed"           # Normal operation, requests allowed
    OPEN = "open"              # Failing state, requests blocked
    HALF_OPEN = "half_open"    # Testing state, limited requests for recovery


class AgentType(Enum):
    """Agent-Typen für spezifische Circuit Breaker."""
    VOICE_AGENT = "voice_agent"
    TOOL_AGENT = "tool_agent"
    WORKFLOW_AGENT = "workflow_agent"
    ORCHESTRATOR_AGENT = "orchestrator_agent"
    CUSTOM_AGENT = "custom_agent"


class FailureType(Enum):
    """Kategorien von Agent-Failures."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    BUSINESS_LOGIC_FAILURE = "business_logic_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    AUTHENTICATION_FAILURE = "authentication_failure"
    VALIDATION_ERROR = "validation_error"


class RecoveryStrategy(Enum):
    """Recovery-Strategien für Circuit Breaker."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    ADAPTIVE = "adaptive"


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker Konfiguration."""
    failure_threshold: int = 5              # Anzahl Failures bis OPEN
    recovery_timeout_seconds: int = 60      # Zeit bis HALF_OPEN Versuch
    success_threshold: int = 3              # Erfolge für CLOSED Recovery
    timeout_seconds: float = 30.0           # Request Timeout

    # Agent-spezifische Konfiguration
    agent_type: AgentType | None = None
    agent_id: str | None = None

    # Recovery-Konfiguration
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF
    max_recovery_attempts: int = 3
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 300

    # Monitoring-Konfiguration
    monitoring_enabled: bool = True
    alert_on_open: bool = True
    metrics_window_seconds: int = 300       # 5 minutes

    # Fallback-Konfiguration
    fallback_enabled: bool = True
    fallback_agent_ids: list[str] = None
    cached_response_enabled: bool = False
    cache_ttl_seconds: int = 300


@dataclass
class AgentCallContext:
    """Kontext für Agent-Calls."""
    agent_id: str
    agent_type: AgentType
    framework: str
    task: str

    # Voice-Workflow-Kontext
    voice_workflow_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    # Request-Metadaten
    request_id: str | None = None
    timeout_seconds: float | None = None
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical

    # Tool-Call-Kontext
    tool_name: str | None = None
    tool_parameters: dict[str, Any] | None = None

    # Workflow-Step-Kontext
    workflow_step: str | None = None
    step_index: int | None = None


@dataclass
class CircuitBreakerResult:
    """Ergebnis einer Circuit Breaker Prüfung."""
    allowed: bool
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: datetime | None = None
    next_attempt_time: datetime | None = None

    # Fallback-Informationen
    fallback_used: bool = False
    fallback_agent_id: str | None = None
    cached_response_used: bool = False

    # Monitoring-Daten
    response_time_ms: float | None = None
    error_message: str | None = None
    failure_type: FailureType | None = None


@dataclass
class AgentExecutionResult:
    """Ergebnis einer Agent-Execution."""
    success: bool
    result: Any = None
    error: str | None = None
    failure_type: FailureType | None = None

    # Performance-Metriken
    execution_time_ms: float = 0.0
    tokens_used: int | None = None

    # Quality-Metriken
    confidence_score: float | None = None
    output_quality: str | None = None

    # Fallback-Informationen
    fallback_used: bool = False
    fallback_agent_id: str | None = None


# =============================================================================
# CIRCUIT BREAKER INTERFACES
# =============================================================================

@runtime_checkable
class ICircuitBreaker(Protocol):
    """Interface für einzelnen Circuit Breaker."""

    @property
    def state(self) -> CircuitBreakerState:
        """Aktueller Circuit Breaker State."""
        ...

    @property
    def config(self) -> CircuitBreakerConfig:
        """Circuit Breaker Konfiguration."""
        ...

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Führt Funktion mit Circuit Breaker Protection aus."""
        ...

    async def is_call_allowed(self) -> bool:
        """Prüft ob Call erlaubt ist."""
        ...

    async def record_success(self, execution_time_ms: float = 0.0) -> None:
        """Registriert erfolgreichen Call."""
        ...

    async def record_failure(
        self,
        failure_type: FailureType,
        error_message: str | None = None
    ) -> None:
        """Registriert fehlgeschlagenen Call."""
        ...

    async def force_open(self, reason: str = "") -> None:
        """Erzwingt OPEN State (Admin-Funktion)."""
        ...

    async def force_close(self, reason: str = "") -> None:
        """Erzwingt CLOSED State (Admin-Funktion)."""
        ...

    async def reset(self) -> None:
        """Setzt Circuit Breaker zurück."""
        ...

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Circuit Breaker Statistiken zurück."""
        ...


@runtime_checkable
class IAgentCircuitBreakerManager(Protocol):
    """Interface für Agent Circuit Breaker Manager."""

    async def get_circuit_breaker(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.CUSTOM_AGENT
    ) -> ICircuitBreaker:
        """Holt oder erstellt Circuit Breaker für Agent."""
        ...

    async def execute_agent_call(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Führt Agent-Call mit Circuit Breaker Protection aus."""
        ...

    async def get_fallback_agent(
        self,
        original_agent_id: str,
        agent_type: AgentType
    ) -> str | None:
        """Bestimmt Fallback-Agent für fehlgeschlagenen Agent."""
        ...

    async def get_cached_response(
        self,
        context: AgentCallContext
    ) -> Any | None:
        """Holt gecachte Response für Agent-Call."""
        ...

    async def cache_response(
        self,
        context: AgentCallContext,
        response: Any,
        ttl_seconds: int = 300
    ) -> None:
        """Cached Response für Agent-Call."""
        ...

    async def get_all_circuit_breakers(self) -> dict[str, ICircuitBreaker]:
        """Gibt alle Circuit Breaker zurück."""
        ...

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Manager-Statistiken zurück."""
        ...


@runtime_checkable
class IAgentCircuitBreakerService(Protocol):
    """Haupt-Interface für Agent Circuit Breaker Service."""

    @property
    def manager(self) -> IAgentCircuitBreakerManager:
        """Circuit Breaker Manager."""
        ...

    async def initialize(self) -> None:
        """Initialisiert Circuit Breaker Service."""
        ...

    async def shutdown(self) -> None:
        """Fährt Circuit Breaker Service herunter."""
        ...

    async def execute_agent_with_protection(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Führt Agent-Execution mit vollständiger Protection aus."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Führt Health Check für Circuit Breaker Service durch."""
        ...

    async def get_service_statistics(self) -> dict[str, Any]:
        """Gibt Service-Statistiken zurück."""
        ...

    async def get_circuit_breaker_status(self, agent_id: str, agent_type: str = "custom_agent") -> dict[str, Any]:
        """Gibt Status eines spezifischen Circuit Breakers zurück."""
        ...

    async def get_all_circuit_breakers_status(self) -> dict[str, Any]:
        """Gibt Status aller Circuit Breaker zurück."""
        ...

    async def force_circuit_breaker_state(
        self,
        agent_id: str,
        agent_type: str,
        state: str,
        reason: str = ""
    ) -> dict[str, Any]:
        """Erzwingt Circuit Breaker State (Admin-Funktion)."""
        ...


# =============================================================================
# FALLBACK STRATEGY INTERFACES
# =============================================================================

@runtime_checkable
class IFallbackStrategy(Protocol):
    """Interface für Fallback-Strategien."""

    async def get_fallback_agent(
        self,
        original_agent_id: str,
        agent_type: AgentType,
        context: AgentCallContext
    ) -> str | None:
        """Bestimmt Fallback-Agent."""
        ...

    async def execute_fallback(
        self,
        context: AgentCallContext,
        original_error: Exception
    ) -> AgentExecutionResult:
        """Führt Fallback-Execution aus."""
        ...


@runtime_checkable
class ICacheStrategy(Protocol):
    """Interface für Cache-Strategien."""

    async def get_cache_key(self, context: AgentCallContext) -> str:
        """Generiert Cache-Key für Kontext."""
        ...

    async def get_cached_response(self, cache_key: str) -> Any | None:
        """Holt gecachte Response."""
        ...

    async def cache_response(
        self,
        cache_key: str,
        response: Any,
        ttl_seconds: int = 300
    ) -> None:
        """Cached Response."""
        ...

    async def invalidate_cache(self, cache_key: str) -> None:
        """Invalidiert Cache-Eintrag."""
        ...


# =============================================================================
# CONFIGURATION INTERFACES
# =============================================================================

@dataclass
class AgentCircuitBreakerSettings:
    """Agent Circuit Breaker Konfiguration."""

    # Basis-Konfiguration
    enabled: bool = True
    monitoring_enabled: bool = True
    fallback_enabled: bool = True
    caching_enabled: bool = True

    # Default Circuit Breaker Settings
    default_failure_threshold: int = 5
    default_recovery_timeout_seconds: int = 60
    default_success_threshold: int = 3
    default_timeout_seconds: float = 30.0

    # Agent-Type-spezifische Settings
    voice_agent_failure_threshold: int = 3
    voice_agent_timeout_seconds: float = 10.0
    tool_agent_failure_threshold: int = 5
    tool_agent_timeout_seconds: float = 30.0
    workflow_agent_failure_threshold: int = 2
    workflow_agent_timeout_seconds: float = 60.0
    orchestrator_agent_failure_threshold: int = 2
    orchestrator_agent_timeout_seconds: float = 15.0

    # Recovery-Konfiguration
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF
    max_recovery_attempts: int = 3
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 300

    # Fallback-Konfiguration
    fallback_timeout_seconds: float = 15.0
    max_fallback_attempts: int = 2
    fallback_cache_enabled: bool = True

    # Cache-Konfiguration
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000
    cache_cleanup_interval_seconds: int = 600

    # Monitoring-Konfiguration
    metrics_window_seconds: int = 300
    alert_threshold: float = 0.8  # Alert bei 80% Failure Rate
    statistics_retention_hours: int = 24
