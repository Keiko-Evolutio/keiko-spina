"""Konsolidierter Circuit Breaker für Agent-Operationen."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar

from kei_logging import get_logger
from observability.tracing import trace_function

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Zustände des Circuit Breakers."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreakerConfig:
    """Konfiguration für Circuit Breaker.

    Konsolidiert alle Konfigurationsoptionen aus verschiedenen Implementierungen.
    """

    # Failure-Handling
    failure_threshold: int = 5
    success_threshold: int = 3  # Für HALF_OPEN -> CLOSED Übergang

    # Timeouts
    open_timeout_seconds: float = 20.0
    request_timeout_seconds: float = 30.0

    # Concurrency
    half_open_max_concurrent: int = 1

    # Backoff-Strategie
    recovery_backoff_base: float = 1.5
    recovery_backoff_max_seconds: float = 300.0

    # Monitoring
    enable_detailed_stats: bool = True
    stats_window_seconds: float = 60.0


@dataclass
class CircuitBreakerStats:
    """Detaillierte Statistiken für Circuit Breaker Monitoring."""

    state: CircuitState = CircuitState.CLOSED
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_changed_at: float = field(default_factory=time.time)
    window_start: float = field(default_factory=time.time)

    @property
    def failure_rate(self) -> float:
        """Berechnet aktuelle Fehlerrate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Berechnet aktuelle Erfolgsrate."""
        return 1.0 - self.failure_rate


class CircuitBreakerException(Exception):
    """Exception für Circuit Breaker Fehler."""

    def __init__(self, message: str, state: CircuitState, stats: CircuitBreakerStats):
        super().__init__(message)
        self.state = state
        self.stats = stats


class UnifiedCircuitBreaker(Generic[T]):
    """Konsolidierter Circuit Breaker für alle Agent-Operationen.

    Vereint Features aus allen vorhandenen Circuit Breaker Implementierungen:
    - Einfache Konfiguration und Verwendung
    - Detailliertes Monitoring und Statistiken
    - Exponentielles Backoff
    - Timeout-Handling
    - Thread-safe Operationen
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        """Initialisiert Circuit Breaker.

        Args:
            name: Eindeutiger Name für Logging und Monitoring
            config: Konfiguration, verwendet Defaults wenn None
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._half_open_sem = asyncio.Semaphore(self.config.half_open_max_concurrent)

    @property
    def state(self) -> CircuitState:
        """Gibt aktuellen Zustand zurück."""
        return self.stats.state

    def _reset_stats_window(self) -> None:
        """Setzt Statistik-Fenster zurück wenn nötig."""
        current_time = time.time()
        if current_time - self.stats.window_start > self.config.stats_window_seconds:
            self.stats.window_start = current_time
            self.stats.total_requests = 0
            self.stats.successful_requests = 0
            self.stats.failed_requests = 0

    def _transition_to_open(self) -> None:
        """Übergang zu OPEN-Zustand bei Überschreitung der Fehlerschwelle."""
        self.stats.state = CircuitState.OPEN
        self.stats.state_changed_at = time.time()
        logger.warning(
            f"Circuit Breaker '{self.name}' -> OPEN "
            f"(failures={self.stats.consecutive_failures})"
        )

    def _transition_to_half_open(self) -> None:
        """Übergang zu HALF_OPEN-Zustand für Recovery-Versuche."""
        self.stats.state = CircuitState.HALF_OPEN
        self.stats.state_changed_at = time.time()
        logger.info(f"Circuit Breaker '{self.name}' -> HALF_OPEN")

    def _transition_to_closed(self) -> None:
        """Übergang zu CLOSED-Zustand nach erfolgreichem Recovery."""
        self.stats.state = CircuitState.CLOSED
        self.stats.state_changed_at = time.time()
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0
        logger.info(f"Circuit Breaker '{self.name}' -> CLOSED")

    def _can_attempt_recovery(self) -> bool:
        """Prüft ob Recovery-Versuch möglich ist basierend auf Backoff-Strategie."""
        if self.stats.state != CircuitState.OPEN:
            return True

        time_since_open = time.time() - self.stats.state_changed_at

        # Einfaches Timeout für erste Recovery-Versuche
        # Exponentielles Backoff nur bei wiederholten Fehlern
        if self.stats.consecutive_failures <= self.config.failure_threshold:
            backoff_time = self.config.open_timeout_seconds
        else:
            # Exponentielles Backoff für wiederholte Fehler
            extra_failures = self.stats.consecutive_failures - self.config.failure_threshold
            backoff_time = min(
                self.config.open_timeout_seconds * (
                    self.config.recovery_backoff_base ** extra_failures
                ),
                self.config.recovery_backoff_max_seconds
            )

        return time_since_open >= backoff_time

    def _update_state(self) -> None:
        """Aktualisiert Circuit Breaker Zustand basierend auf aktueller Situation."""
        if self.config.enable_detailed_stats:
            self._reset_stats_window()

        if self.stats.state == CircuitState.OPEN and self._can_attempt_recovery():
            self._transition_to_half_open()

    async def _on_success(self) -> None:
        """Behandelt erfolgreiche Ausführung."""
        async with self._lock:
            self.stats.successful_requests += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = time.time()

            # Übergang von HALF_OPEN zu CLOSED nach genügend Erfolgen
            if (self.stats.state == CircuitState.HALF_OPEN and
                self.stats.consecutive_successes >= self.config.success_threshold):
                self._transition_to_closed()

    async def _on_failure(self) -> None:
        """Behandelt fehlgeschlagene Ausführung."""
        async with self._lock:
            self.stats.failed_requests += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()

            # Übergang zu OPEN bei Überschreitung der Fehlerschwelle
            if (self.stats.state == CircuitState.CLOSED and
                self.stats.consecutive_failures >= self.config.failure_threshold):
                self._transition_to_open()
            elif self.stats.state == CircuitState.HALF_OPEN:
                # Bei Fehler in HALF_OPEN zurück zu OPEN
                self._transition_to_open()

    @trace_function("circuit_breaker.call")
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Führt Funktion mit Circuit Breaker Schutz aus.

        Args:
            func: Auszuführende async Funktion
            *args: Positionale Argumente
            **kwargs: Keyword Argumente

        Returns:
            Ergebnis der Funktion

        Raises:
            CircuitBreakerException: Wenn Circuit offen ist
            Exception: Ursprüngliche Exception bei Fehlern
        """
        async with self._lock:
            self._update_state()
            self.stats.total_requests += 1

            # Prüfe Circuit Zustand
            if self.stats.state == CircuitState.OPEN:
                logger.warning(f"Circuit Breaker '{self.name}' ist OPEN - Request blockiert")
                raise CircuitBreakerException(
                    f"Circuit Breaker '{self.name}' ist offen",
                    self.stats.state,
                    self.stats
                )

        # Für HALF_OPEN: Begrenze gleichzeitige Aufrufe
        if self.stats.state == CircuitState.HALF_OPEN:
            async with self._half_open_sem:
                return await self._execute_with_timeout(func, *args, **kwargs)
        else:
            return await self._execute_with_timeout(func, *args, **kwargs)

    async def _execute_with_timeout(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Führt Funktion mit Timeout aus."""
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.request_timeout_seconds
            )
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise


# Globaler Circuit Breaker Registry für Agent-Operationen
_circuit_breakers: dict[str, UnifiedCircuitBreaker] = {}


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> UnifiedCircuitBreaker:
    """Holt oder erstellt Circuit Breaker für gegebenen Namen.

    Args:
        name: Eindeutiger Name für Circuit Breaker
        config: Optionale Konfiguration

    Returns:
        Circuit Breaker Instanz
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = UnifiedCircuitBreaker(name, config)
    return _circuit_breakers[name]


def clear_circuit_breakers() -> None:
    """Leert Circuit Breaker Registry (hauptsächlich für Tests)."""
    global _circuit_breakers
    _circuit_breakers.clear()
