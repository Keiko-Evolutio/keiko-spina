"""Agent Circuit Breaker Implementation.
Konsolidierte, saubere Circuit Breaker-Implementierung für alle Agent-Typen.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from kei_logging import get_logger

from ..constants import (
    ERROR_CIRCUIT_BREAKER_OPEN,
    HIGH_FAILURE_RATE_MULTIPLIER,
    HIGH_FAILURE_RATE_THRESHOLD,
    LOG_CIRCUIT_BREAKER_CLOSED,
    LOG_CIRCUIT_BREAKER_OPENED,
    LOG_FAILURE_RECORDED,
    LOG_SUCCESS_RECORDED,
    MEDIUM_FAILURE_RATE_MULTIPLIER,
    MEDIUM_FAILURE_RATE_THRESHOLD,
    RECENT_CALLS_WINDOW_MINUTES,
)
from .interfaces import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    FailureType,
    ICircuitBreaker,
    RecoveryStrategy,
)

logger = get_logger(__name__)


class AgentCircuitBreaker(ICircuitBreaker):
    """Circuit Breaker Implementation für Agent-Calls.
    Thread-safe Implementation mit State Management und Recovery-Strategien.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialisiert Circuit Breaker mit gegebener Konfiguration."""
        self.name = name
        self._config = config
        self._state = CircuitBreakerState.CLOSED

        # Failure/Success Tracking
        self._failure_count = 0
        self._success_count = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0

        # Timing
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._state_changed_time = datetime.now(UTC)
        self._next_attempt_time: datetime | None = None

        # Recovery Tracking
        self._recovery_attempts = 0
        self._current_backoff_seconds = config.recovery_timeout_seconds

        # Metrics (Sliding Window) - Konstante aus constants.py
        from ..constants import DEFAULT_SLIDING_WINDOW_SIZE
        self._recent_calls = deque(maxlen=DEFAULT_SLIDING_WINDOW_SIZE)
        self._response_times = deque(maxlen=DEFAULT_SLIDING_WINDOW_SIZE)

        # Thread Safety
        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' erstellt mit Konfiguration: "
                    f"failure_threshold={config.failure_threshold}, "
                    f"timeout={config.timeout_seconds}s")

    @property
    def state(self) -> CircuitBreakerState:
        """Aktueller Circuit Breaker State."""
        return self._state

    @property
    def config(self) -> CircuitBreakerConfig:
        """Circuit Breaker Konfiguration."""
        return self._config

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Führt Funktion mit Circuit Breaker Protection aus."""
        await self._validate_call_allowed()
        return await self._execute_with_protection(func, *args, **kwargs)

    async def _validate_call_allowed(self) -> None:
        """Prüft ob Call erlaubt ist und wirft Exception falls nicht."""
        async with self._lock:
            if not await self._is_call_allowed_internal():
                raise CircuitBreakerOpenError(
                    ERROR_CIRCUIT_BREAKER_OPEN.format(
                        name=self.name,
                        state=self._state.value
                    ),
                    circuit_breaker_name=self.name
                )

    async def _execute_with_protection(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Führt Funktion mit Timeout und Error-Handling aus."""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self._config.timeout_seconds
            )

            execution_time_ms = (time.time() - start_time) * 1000
            await self.record_success(execution_time_ms)
            return result

        except TimeoutError:
            await self.record_failure(FailureType.TIMEOUT, "Request timeout")
            raise
        except Exception as e:
            failure_type = AgentCircuitBreaker._categorize_error(e)
            await self.record_failure(failure_type, str(e))
            raise

    async def is_call_allowed(self) -> bool:
        """Prüft ob Call erlaubt ist."""
        async with self._lock:
            return await self._is_call_allowed_internal()

    async def _is_call_allowed_internal(self) -> bool:
        """Interne Call-Erlaubnis-Prüfung (ohne Lock)."""
        current_time = datetime.now(UTC)

        if self._state == CircuitBreakerState.CLOSED:
            return True

        if self._state == CircuitBreakerState.OPEN:
            # Prüfe ob Recovery-Zeit erreicht ist
            if (self._next_attempt_time and
                current_time >= self._next_attempt_time):
                # Wechsle zu HALF_OPEN
                await self._transition_to_half_open()
                return True
            return False

        if self._state == CircuitBreakerState.HALF_OPEN:
            # In HALF_OPEN nur begrenzte Anzahl Calls erlauben
            return True

        # Fallback für unbekannte States
        return False

    async def record_success(self, execution_time_ms: float = 0.0) -> None:
        """Registriert erfolgreichen Call."""
        async with self._lock:
            current_time = datetime.now(UTC)

            self._success_count += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._last_success_time = current_time

            # Metrics aktualisieren
            self._recent_calls.append(("success", current_time))
            self._response_times.append(execution_time_ms)

            # State-Transition prüfen
            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._consecutive_successes >= self._config.success_threshold:
                    await self._transition_to_closed()

            logger.debug(
                LOG_SUCCESS_RECORDED.format(
                    name=self.name,
                    time=execution_time_ms,
                    consecutive=self._consecutive_successes
                )
            )

    async def record_failure(
        self,
        failure_type: FailureType,
        error_message: str | None = None
    ) -> None:
        """Registriert fehlgeschlagenen Call."""
        async with self._lock:
            current_time = datetime.now(UTC)

            self._failure_count += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = current_time

            # Metrics aktualisieren
            self._recent_calls.append(("failure", current_time))

            # State-Transition prüfen
            if self._state == CircuitBreakerState.CLOSED:
                if self._consecutive_failures >= self._config.failure_threshold:
                    await self._transition_to_open()

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Bei Failure in HALF_OPEN zurück zu OPEN
                await self._transition_to_open()

            logger.warning(
                LOG_FAILURE_RECORDED.format(
                    name=self.name,
                    type=failure_type.value,
                    consecutive=self._consecutive_failures,
                    error=error_message
                )
            )

    async def _transition_to_open(self) -> None:
        """Wechselt zu OPEN State."""
        self._state = CircuitBreakerState.OPEN
        self._state_changed_time = datetime.now(UTC)
        self._recovery_attempts += 1

        # Berechne nächste Attempt-Zeit mit Backoff
        backoff_seconds = self._calculate_backoff()
        self._next_attempt_time = self._state_changed_time + timedelta(seconds=backoff_seconds)

        logger.warning(
            LOG_CIRCUIT_BREAKER_OPENED.format(
                name=self.name,
                failures=self._consecutive_failures,
                next_attempt=self._next_attempt_time
            )
        )

    async def _transition_to_half_open(self) -> None:
        """Wechselt zu HALF_OPEN State.

        Im HALF_OPEN State werden begrenzte Test-Calls erlaubt,
        um zu prüfen, ob der Service wieder verfügbar ist.
        """
        self._state = CircuitBreakerState.HALF_OPEN
        self._state_changed_time = datetime.now(UTC)
        self._consecutive_successes = 0

        from ..constants import LOG_CIRCUIT_BREAKER_HALF_OPEN
        logger.info(LOG_CIRCUIT_BREAKER_HALF_OPEN.format(name=self.name))

    async def _transition_to_closed(self) -> None:
        """Wechselt zu CLOSED State."""
        self._state = CircuitBreakerState.CLOSED
        self._state_changed_time = datetime.now(UTC)
        self._consecutive_failures = 0
        self._recovery_attempts = 0
        self._current_backoff_seconds = self._config.recovery_timeout_seconds
        self._next_attempt_time = None

        logger.info(
            LOG_CIRCUIT_BREAKER_CLOSED.format(
                name=self.name,
                successes=self._consecutive_successes
            )
        )

    def _calculate_backoff(self) -> float:
        """Berechnet Backoff-Zeit basierend auf Recovery-Strategie."""
        if self._config.recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            backoff = min(
                self._current_backoff_seconds * (self._config.backoff_multiplier ** (self._recovery_attempts - 1)),
                self._config.max_backoff_seconds
            )
            self._current_backoff_seconds = backoff
            return backoff

        if self._config.recovery_strategy == RecoveryStrategy.LINEAR_BACKOFF:
            return min(
                self._config.recovery_timeout_seconds * self._recovery_attempts,
                self._config.max_backoff_seconds
            )

        if self._config.recovery_strategy == RecoveryStrategy.FIXED_INTERVAL:
            return self._config.recovery_timeout_seconds

        if self._config.recovery_strategy == RecoveryStrategy.ADAPTIVE:
            # Adaptive basierend auf historischer Performance
            recent_failure_rate = self._get_recent_failure_rate()
            base_backoff = self._config.recovery_timeout_seconds

            if recent_failure_rate > HIGH_FAILURE_RATE_THRESHOLD:
                return min(base_backoff * HIGH_FAILURE_RATE_MULTIPLIER, self._config.max_backoff_seconds)
            if recent_failure_rate > MEDIUM_FAILURE_RATE_THRESHOLD:
                return min(base_backoff * MEDIUM_FAILURE_RATE_MULTIPLIER, self._config.max_backoff_seconds)
            return base_backoff

        # Fallback für den Fall, dass keine recent calls vorhanden sind
        return self._config.recovery_timeout_seconds

    def _get_recent_failure_rate(self) -> float:
        """Berechnet aktuelle Failure Rate basierend auf recent calls.

        Returns:
            Failure Rate als Float zwischen 0.0 und 1.0
        """
        if not self._recent_calls:
            return 0.0

        try:
            # Nur Calls der letzten X Minuten betrachten (aus Konstanten)
            cutoff_time = datetime.now(UTC) - timedelta(minutes=RECENT_CALLS_WINDOW_MINUTES)
            recent_calls = [
                call for call in self._recent_calls
                if call[1] >= cutoff_time
            ]

            if not recent_calls:
                return 0.0

            failures = sum(1 for call in recent_calls if call[0] == "failure")
            return failures / len(recent_calls)
        except Exception as e:
            logger.warning(f"Fehler beim Berechnen der Failure Rate für {self.name}: {e}")
            return 0.0

    @staticmethod
    def _categorize_error(error: Exception) -> FailureType:
        """Kategorisiert Fehler-Typ.

        Args:
            error: Zu kategorisierender Fehler

        Returns:
            Kategorisierter Fehler-Typ
        """
        error_str = str(error).lower()

        if isinstance(error, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        if "rate limit" in error_str or "too many requests" in error_str:
            return FailureType.RATE_LIMIT_EXCEEDED
        if "authentication" in error_str or "unauthorized" in error_str:
            return FailureType.AUTHENTICATION_FAILURE
        if "validation" in error_str or "invalid" in error_str:
            return FailureType.VALIDATION_ERROR
        if "resource" in error_str or "unavailable" in error_str:
            return FailureType.RESOURCE_UNAVAILABLE
        return FailureType.EXCEPTION

    async def force_open(self, reason: str = "") -> None:
        """Erzwingt OPEN State (Admin-Funktion)."""
        async with self._lock:
            self._state = CircuitBreakerState.OPEN
            self._state_changed_time = datetime.now(UTC)
            self._next_attempt_time = self._state_changed_time + timedelta(
                seconds=self._config.recovery_timeout_seconds
            )

            logger.warning(f"Circuit breaker '{self.name}' manuell geöffnet: {reason or 'Kein Grund angegeben'}")

    async def force_close(self, reason: str = "") -> None:
        """Erzwingt CLOSED State (Admin-Funktion)."""
        async with self._lock:
            await self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manuell geschlossen: {reason or 'Kein Grund angegeben'}")

    async def reset(self) -> None:
        """Setzt Circuit Breaker zurück."""
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._last_failure_time = None
            self._last_success_time = None
            self._state_changed_time = datetime.now(UTC)
            self._next_attempt_time = None
            self._recovery_attempts = 0
            self._current_backoff_seconds = self._config.recovery_timeout_seconds
            self._recent_calls.clear()
            self._response_times.clear()

            logger.info(f"Circuit breaker '{self.name}' zurückgesetzt - alle Statistiken gelöscht")

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Circuit Breaker Statistiken zurück."""
        async with self._lock:
            current_time = datetime.now(UTC)
            uptime_seconds = (current_time - self._state_changed_time).total_seconds()

            # Response Time Statistiken
            avg_response_time = 0.0
            if self._response_times:
                avg_response_time = sum(self._response_times) / len(self._response_times)

            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
                "state_changed_time": self._state_changed_time.isoformat(),
                "next_attempt_time": self._next_attempt_time.isoformat() if self._next_attempt_time else None,
                "recovery_attempts": self._recovery_attempts,
                "current_backoff_seconds": self._current_backoff_seconds,
                "uptime_seconds": uptime_seconds,
                "recent_failure_rate": self._get_recent_failure_rate(),
                "avg_response_time_ms": avg_response_time,
                "total_calls": len(self._recent_calls),
                "config": {
                    "failure_threshold": self._config.failure_threshold,
                    "recovery_timeout_seconds": self._config.recovery_timeout_seconds,
                    "success_threshold": self._config.success_threshold,
                    "timeout_seconds": self._config.timeout_seconds,
                    "recovery_strategy": self._config.recovery_strategy.value
                }
            }


class CircuitBreakerOpenError(Exception):
    """Exception für geöffneten Circuit Breaker.

    Wird geworfen, wenn ein Call versucht wird, während der Circuit Breaker
    im OPEN State ist und noch nicht für Recovery-Tests bereit ist.
    """

    def __init__(self, message: str, circuit_breaker_name: str | None = None):
        """Initialisiert CircuitBreakerOpenError.

        Args:
            message: Fehlermeldung
            circuit_breaker_name: Name des betroffenen Circuit Breakers
        """
        super().__init__(message)
        self.circuit_breaker_name = circuit_breaker_name
