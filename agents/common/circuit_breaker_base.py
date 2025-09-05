# backend/agents/common/circuit_breaker_base.py
"""Gemeinsame Circuit Breaker-Basis-Implementierung für Keiko Personal Assistant

Konsolidiert alle Circuit Breaker-Implementierungen in eine einheitliche,
wiederverwendbare Basis-Klasse mit Enterprise-Grade Features.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

# Fallback für kei_logging und monitoring
try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

try:
    from monitoring.custom_metrics import MetricsCollector
except ImportError:
    class MetricsCollector:
        def increment_counter(self, name: str, tags: dict = None): pass
        def record_histogram(self, name: str, value: float, tags: dict = None): pass
        def record_gauge(self, name: str, value: float, tags: dict = None): pass

logger = get_logger(__name__)

T = TypeVar("T")

# Konstanten für bessere Wartbarkeit
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RECOVERY_TIMEOUT = 60.0
DEFAULT_SUCCESS_THRESHOLD = 3
DEFAULT_TIMEOUT = 30.0
DEFAULT_SLIDING_WINDOW_SIZE = 100
DEFAULT_SLIDING_WINDOW_DURATION = 300.0
DEFAULT_FAILURE_RATE_THRESHOLD = 0.5
DEFAULT_MINIMUM_THROUGHPUT = 10


class CircuitBreakerState(str, Enum):
    """Einheitliche Circuit Breaker-Zustände."""

    CLOSED = "closed"      # Normaler Betrieb
    OPEN = "open"          # Circuit offen, Calls werden abgelehnt
    HALF_OPEN = "half_open"  # Test-Phase für Recovery


class CircuitBreakerOpenError(Exception):
    """Exception für offene Circuit Breaker."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        circuit_name: str = "",
        state: CircuitBreakerState = CircuitBreakerState.OPEN,
        metrics: dict[str, Any] | None = None
    ):
        """Initialisiert Circuit Breaker-Exception.

        Args:
            message: Fehlermeldung
            circuit_name: Name des Circuit Breakers
            state: Aktueller Zustand
            metrics: Zusätzliche Metriken
        """
        super().__init__(message)
        self.message = message
        self.circuit_name = circuit_name
        self.state = state
        self.metrics = metrics or {}


@dataclass
class CircuitBreakerConfig:
    """Einheitliche Circuit Breaker-Konfiguration."""

    # Basis-Konfiguration
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT
    success_threshold: int = DEFAULT_SUCCESS_THRESHOLD
    timeout: float = DEFAULT_TIMEOUT

    # Erweiterte Konfiguration
    failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD
    minimum_throughput: int = DEFAULT_MINIMUM_THROUGHPUT
    sliding_window_size: int = DEFAULT_SLIDING_WINDOW_SIZE
    sliding_window_duration: float = DEFAULT_SLIDING_WINDOW_DURATION

    # Capability-spezifische Einstellungen
    capability_specific_thresholds: dict[str, int] = field(default_factory=dict)
    capability_specific_timeouts: dict[str, float] = field(default_factory=dict)

    # Callbacks
    on_state_change: Callable[[CircuitBreakerState, CircuitBreakerState], Awaitable[None]] | None = None
    on_call_rejected: Callable[[str, str], Awaitable[None]] | None = None
    on_call_success: Callable[[str, str, float], Awaitable[None]] | None = None
    on_call_failure: Callable[[str, str, Exception], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        """Validiert Konfiguration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold muss positiv sein")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout muss positiv sein")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold muss positiv sein")
        if self.timeout <= 0:
            raise ValueError("timeout muss positiv sein")
        if not 0 < self.failure_rate_threshold <= 1:
            raise ValueError("failure_rate_threshold muss zwischen 0 und 1 liegen")


@dataclass
class CircuitBreakerMetrics:
    """Einheitliche Circuit Breaker-Metriken."""

    # Basis-Metriken
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0

    # Timing-Metriken
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change_time: float = 0.0

    # Response-Time-Metriken
    total_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0

    # Zustand
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Berechnet Fehlerrate."""
        return 1.0 - self.success_rate

    @property
    def avg_response_time(self) -> float:
        """Berechnet durchschnittliche Response-Zeit."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_response_time / self.successful_calls


class CircuitBreakerStateHandler(ABC):
    """Abstrakte Basis-Klasse für State-Handler."""

    @abstractmethod
    async def handle_call(
        self,
        circuit_breaker: "BaseCircuitBreaker",
        func: Callable[[], Awaitable[T]]
    ) -> T:
        """Behandelt Function-Call im spezifischen State.

        Args:
            circuit_breaker: Circuit Breaker-Instanz
            func: Auszuführende Funktion

        Returns:
            Function-Result

        Raises:
            CircuitBreakerOpenError: Wenn Call abgelehnt wird
        """


class ClosedStateHandler(CircuitBreakerStateHandler):
    """Handler für CLOSED-State."""

    async def handle_call(
        self,
        circuit_breaker: "BaseCircuitBreaker",
        func: Callable[[], Awaitable[T]]
    ) -> T:
        """Führt Function-Call im CLOSED-State aus."""
        try:
            start_time = time.time()
            result = await func()
            response_time = time.time() - start_time

            await circuit_breaker.handle_success(response_time)
            return result

        except Exception as e:
            await circuit_breaker.handle_failure(e)
            raise


class OpenStateHandler(CircuitBreakerStateHandler):
    """Handler für OPEN-State."""

    async def handle_call(
        self,
        circuit_breaker: "BaseCircuitBreaker",
        func: Callable[[], Awaitable[T]]
    ) -> T:
        """Lehnt Function-Call im OPEN-State ab."""
        # Prüfe ob Recovery-Timeout erreicht
        if circuit_breaker.should_attempt_reset():
            await circuit_breaker.transition_to_half_open()
            return await circuit_breaker.state_handler.handle_call(circuit_breaker, func)

        # Call ablehnen
        circuit_breaker.metrics.rejected_calls += 1
        raise CircuitBreakerOpenError(
            f"Circuit breaker '{circuit_breaker.name}' ist offen. "
            f"Nächster Versuch in {circuit_breaker.time_until_retry():.1f}s",
            circuit_breaker.name,
            CircuitBreakerState.OPEN,
            circuit_breaker.metrics.__dict__
        )


class HalfOpenStateHandler(CircuitBreakerStateHandler):
    """Handler für HALF_OPEN-State."""

    async def handle_call(
        self,
        circuit_breaker: "BaseCircuitBreaker",
        func: Callable[[], Awaitable[T]]
    ) -> T:
        """Führt Test-Call im HALF_OPEN-State aus."""
        try:
            start_time = time.time()
            result = await func()
            response_time = time.time() - start_time

            await circuit_breaker.handle_success(response_time)

            # Prüfe ob genug erfolgreiche Calls für CLOSED
            if circuit_breaker.metrics.consecutive_successes >= circuit_breaker.config.success_threshold:
                await circuit_breaker.transition_to_closed()

            return result

        except Exception as e:
            await circuit_breaker.handle_failure(e)
            await circuit_breaker.transition_to_open()
            raise


class BaseCircuitBreaker(Generic[T]):
    """Basis-Klasse für alle Circuit Breaker-Implementierungen."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialisiert Circuit Breaker.

        Args:
            name: Circuit Breaker-Name
            config: Konfiguration
        """
        self.name = name
        self.config = config
        self.metrics = CircuitBreakerMetrics()

        # State-Management
        self._state = CircuitBreakerState.CLOSED
        self._state_handlers = {
            CircuitBreakerState.CLOSED: ClosedStateHandler(),
            CircuitBreakerState.OPEN: OpenStateHandler(),
            CircuitBreakerState.HALF_OPEN: HalfOpenStateHandler(),
        }
        self._state_handler = self._state_handlers[CircuitBreakerState.CLOSED]

        # Sliding Window für Failure-Rate-Tracking
        self._failure_window: deque[tuple[float, bool]] = deque(maxlen=config.sliding_window_size)

        # Metrics
        self._metrics_collector = MetricsCollector()

        # Thread-Safety
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Aktueller Circuit-State."""
        return self._state

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Führt Function-Call mit Circuit-Breaker-Schutz aus.

        Args:
            func: Auszuführende Funktion

        Returns:
            Function-Result

        Raises:
            CircuitBreakerOpenError: Wenn Circuit offen
            Exception: Original Function-Exception
        """
        async with self._lock:
            self.metrics.total_calls += 1

            # Timeout-Wrapper
            try:
                return await asyncio.wait_for(
                    self._state_handler.handle_call(self, func),
                    timeout=self.config.timeout
                )
            except TimeoutError:
                await self._on_failure(TimeoutError(f"Timeout nach {self.config.timeout}s"))
                raise CircuitBreakerOpenError(
                    f"Function-Call Timeout nach {self.config.timeout}s",
                    self.name,
                    self._state,
                    self.metrics.__dict__
                )

    async def _on_success(self, response_time: float) -> None:
        """Behandelt erfolgreichen Function-Call."""
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = time.time()
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0

        # Response-Time-Tracking
        self.metrics.total_response_time += response_time
        self.metrics.min_response_time = min(self.metrics.min_response_time, response_time)
        self.metrics.max_response_time = max(self.metrics.max_response_time, response_time)

        # Sliding Window aktualisieren
        self._failure_window.append((time.time(), False))

        # Metrics
        self._metrics_collector.record_histogram(
            "circuit_breaker.response_time",
            response_time,
            tags={"circuit_name": self.name, "state": self._state.value}
        )

        # Callback
        if self.config.on_call_success:
            await self.config.on_call_success(self.name, "", response_time)

    async def _on_failure(self, exception: Exception) -> None:
        """Behandelt fehlgeschlagenen Function-Call."""
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0

        # Sliding Window aktualisieren
        self._failure_window.append((time.time(), True))

        # Metrics
        self._metrics_collector.increment_counter(
            "circuit_breaker.failures",
            tags={"circuit_name": self.name, "state": self._state.value}
        )

        # Callback
        if self.config.on_call_failure:
            await self.config.on_call_failure(self.name, "", exception)

        # Prüfe State-Transition
        if self._state == CircuitBreakerState.CLOSED and self._should_open():
            await self._transition_to_open()

    def _should_open(self) -> bool:
        """Prüft ob Circuit Breaker öffnen sollte."""
        # Threshold-basierte Prüfung
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True

        # Failure-Rate-basierte Prüfung
        if len(self._failure_window) >= self.config.minimum_throughput:
            current_time = time.time()
            window_start = current_time - self.config.sliding_window_duration

            # Filtere relevante Calls im Zeitfenster
            recent_calls = [
                (timestamp, is_failure)
                for timestamp, is_failure in self._failure_window
                if timestamp >= window_start
            ]

            if len(recent_calls) >= self.config.minimum_throughput:
                failure_count = sum(1 for _, is_failure in recent_calls if is_failure)
                failure_rate = failure_count / len(recent_calls)

                return failure_rate >= self.config.failure_rate_threshold

        return False

    # Public APIs für State Handler
    async def handle_success(self, response_time: float) -> None:
        """Public API für erfolgreiche Calls.

        Args:
            response_time: Response-Zeit in Sekunden
        """
        await self._on_success(response_time)

    async def handle_failure(self, exception: Exception) -> None:
        """Public API für fehlgeschlagene Calls.

        Args:
            exception: Aufgetretene Exception
        """
        await self._on_failure(exception)

    def should_attempt_reset(self) -> bool:
        """Public API für Reset-Prüfung.

        Returns:
            True wenn Reset-Versuch möglich ist
        """
        return self._should_attempt_reset()

    def time_until_retry(self) -> float:
        """Public API für Retry-Zeit-Berechnung.

        Returns:
            Zeit bis zum nächsten Retry-Versuch in Sekunden
        """
        return self._time_until_retry()

    async def transition_to_open(self) -> None:
        """Public API für Übergang zu OPEN-State."""
        await self._transition_to_open()

    async def transition_to_half_open(self) -> None:
        """Public API für Übergang zu HALF_OPEN-State."""
        await self._transition_to_half_open()

    async def transition_to_closed(self) -> None:
        """Public API für Übergang zu CLOSED-State."""
        await self._transition_to_closed()

    @property
    def state_handler(self) -> CircuitBreakerStateHandler:
        """Public API für aktuellen State Handler.

        Returns:
            Aktueller State Handler
        """
        return self._state_handler

    def _should_attempt_reset(self) -> bool:
        """Prüft ob Reset-Versuch möglich ist."""
        return (time.time() - self.metrics.last_failure_time) >= self.config.recovery_timeout

    def _time_until_retry(self) -> float:
        """Berechnet Zeit bis zum nächsten Retry-Versuch."""
        elapsed = time.time() - self.metrics.last_failure_time
        return max(0.0, float(self.config.recovery_timeout) - elapsed)

    async def _transition_to_open(self) -> None:
        """Übergang zu OPEN-State."""
        await self._change_state(CircuitBreakerState.OPEN)

    async def _transition_to_half_open(self) -> None:
        """Übergang zu HALF_OPEN-State."""
        await self._change_state(CircuitBreakerState.HALF_OPEN)

    async def _transition_to_closed(self) -> None:
        """Übergang zu CLOSED-State."""
        await self._change_state(CircuitBreakerState.CLOSED)

    async def _change_state(self, new_state: CircuitBreakerState) -> None:
        """Ändert Circuit Breaker-State."""
        old_state = self._state
        self._state = new_state
        self.metrics.state = new_state
        self.metrics.last_state_change_time = time.time()
        self._state_handler = self._state_handlers[new_state]

        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")

        # Metrics
        self._metrics_collector.increment_counter(
            "circuit_breaker.state_changes",
            tags={
                "circuit_name": self.name,
                "from_state": old_state.value,
                "to_state": new_state.value
            }
        )

        # Callback
        if self.config.on_state_change:
            await self.config.on_state_change(old_state, new_state)
