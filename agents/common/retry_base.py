# backend/agents/common/retry_base.py
"""Gemeinsame Retry-Basis-Implementierung für Keiko Personal Assistant

Konsolidiert alle Retry-Mechanismen in eine einheitliche, wiederverwendbare
Basis-Klasse mit verschiedenen Retry-Strategien und adaptiven Features.
"""

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)

T = TypeVar("T")

# Konstanten für bessere Wartbarkeit
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_EXPONENTIAL_BASE = 2.0
DEFAULT_JITTER_RANGE = 0.1
DEFAULT_PERFORMANCE_WINDOW_SIZE = 100
DEFAULT_PERFORMANCE_THRESHOLD_MS = 1000.0


class RetryStrategy(str, Enum):
    """Einheitliche Retry-Strategien."""

    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"


class RetryExhaustedError(Exception):
    """Exception für erschöpfte Retry-Versuche."""

    def __init__(
        self,
        message: str = "Retry attempts exhausted",
        attempts: int = 0,
        last_exception: Exception | None = None,
        total_duration: float = 0.0
    ):
        """Initialisiert Retry-Exhausted-Exception.

        Args:
            message: Fehlermeldung
            attempts: Anzahl der Versuche
            last_exception: Letzte Exception
            total_duration: Gesamtdauer der Versuche
        """
        super().__init__(message)
        self.message = message
        self.attempts = attempts
        self.last_exception = last_exception
        self.total_duration = total_duration


@dataclass
class RetryConfig:
    """Einheitliche Retry-Konfiguration."""

    # Basis-Konfiguration
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Backoff-Parameter
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE
    jitter: bool = True
    jitter_range: float = DEFAULT_JITTER_RANGE

    # Retry-Bedingungen
    retry_on_exceptions: list[type] = field(default_factory=list)
    retry_on_status_codes: list[int] = field(default_factory=lambda: [429, 502, 503, 504])

    # Custom Retry-Funktionen
    custom_retry_condition: Callable[[Exception], bool] | None = None
    custom_delay_function: Callable[[int, float], float] | None = None

    # Adaptive Retry-Einstellungen
    adaptive_enabled: bool = False
    performance_window_size: int = DEFAULT_PERFORMANCE_WINDOW_SIZE
    performance_threshold_ms: float = DEFAULT_PERFORMANCE_THRESHOLD_MS

    # Callbacks
    on_retry: Callable[[int, Exception, float], Awaitable[None]] | None = None
    on_retry_exhausted: Callable[[int, Exception], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        """Validiert Konfiguration."""
        if self.max_attempts <= 0:
            raise ValueError("max_attempts muss positiv sein")
        if self.base_delay < 0:
            raise ValueError("base_delay muss nicht-negativ sein")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay muss >= base_delay sein")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base muss > 1 sein")
        if not 0 <= self.jitter_range <= 1:
            raise ValueError("jitter_range muss zwischen 0 und 1 liegen")


@dataclass
class RetryMetrics:
    """Retry-Metriken für Monitoring."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retry_time: float = 0.0

    # Performance-Tracking
    response_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_success_time: float = 0.0
    last_failure_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.total_attempts == 0:
            return 1.0
        return self.successful_attempts / self.total_attempts

    @property
    def avg_response_time(self) -> float:
        """Berechnet durchschnittliche Response-Zeit."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class RetryDelayCalculator:
    """Berechnet Retry-Delays basierend auf verschiedenen Strategien."""

    @staticmethod
    def calculate_delay(
        strategy: RetryStrategy,
        attempt: int,
        base_delay: float,
        max_delay: float,
        exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
        jitter: bool = True,
        jitter_range: float = DEFAULT_JITTER_RANGE,
        custom_delay_function: Callable[[int, float], float] | None = None
    ) -> float:
        """Berechnet Delay für Retry-Versuch.

        Args:
            strategy: Retry-Strategie
            attempt: Versuch-Nummer (0-basiert)
            base_delay: Basis-Delay
            max_delay: Maximaler Delay
            exponential_base: Basis für exponential backoff
            jitter: Ob Jitter angewendet werden soll
            jitter_range: Jitter-Bereich
            custom_delay_function: Custom Delay-Funktion

        Returns:
            Berechneter Delay in Sekunden
        """
        if custom_delay_function:
            delay = custom_delay_function(attempt, base_delay)
        elif strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * (attempt + 1)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (exponential_base ** attempt)
        elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = base_delay * RetryDelayCalculator._fibonacci(attempt + 1)
        else:  # ADAPTIVE wird extern behandelt
            delay = base_delay

        # Max-Delay anwenden
        delay = min(delay, max_delay)

        # Jitter anwenden
        if jitter and jitter_range > 0:
            jitter_amount = delay * jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)  # Negative Delays vermeiden

        return delay

    @staticmethod
    def _fibonacci(n: int) -> int:
        """Berechnet n-te Fibonacci-Zahl."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class BaseRetryManager:
    """Basis-Klasse für alle Retry-Manager-Implementierungen."""

    def __init__(self, config: RetryConfig):
        """Initialisiert Retry-Manager.

        Args:
            config: Retry-Konfiguration
        """
        self.config = config
        self.metrics = RetryMetrics()
        self._metrics_collector = MetricsCollector()

        # Adaptive Retry-Tracking
        self._performance_history: deque[float] = deque(maxlen=config.performance_window_size)
        self._adaptive_multiplier = 1.0

    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        operation_name: str = "unknown",
        _context: dict[str, Any] | None = None
    ) -> T:
        """Führt Funktion mit Retry-Mechanismus aus.

        Args:
            func: Auszuführende Funktion
            operation_name: Name der Operation für Logging
            _context: Zusätzlicher Kontext

        Returns:
            Function-Result

        Raises:
            RetryExhaustedError: Wenn alle Retry-Versuche erschöpft sind
        """
        last_exception: Exception | None = None
        start_time = time.time()

        for attempt in range(self.config.max_attempts):
            try:
                # Führe Funktion aus
                attempt_start = time.time()
                result = await func()
                response_time = time.time() - attempt_start

                # Erfolg verzeichnen
                await self._on_success(attempt, response_time, operation_name)
                return result

            except Exception as e:
                last_exception = e

                # Prüfe ob Retry sinnvoll ist
                if not self._should_retry(e, attempt):
                    break

                # Letzter Versuch?
                if attempt == self.config.max_attempts - 1:
                    break

                # Berechne Delay
                delay = self._calculate_retry_delay(attempt)

                # Retry verzeichnen
                await self._on_retry(attempt, e, delay, operation_name)

                # Warte vor nächstem Versuch
                if delay > 0:
                    await asyncio.sleep(delay)

        # Alle Versuche erschöpft
        total_duration = time.time() - start_time
        await self._on_retry_exhausted(self.config.max_attempts, last_exception, operation_name)

        raise RetryExhaustedError(
            f"Retry-Versuche für '{operation_name}' nach {self.config.max_attempts} Versuchen erschöpft",
            self.config.max_attempts,
            last_exception,
            total_duration
        )

    def _should_retry(self, exception: Exception, _attempt: int) -> bool:
        """Prüft ob Retry für Exception sinnvoll ist.

        Args:
            exception: Aufgetretene Exception
            _attempt: Aktueller Versuch (0-basiert, ungenutzt in Basis-Implementation)

        Returns:
            True wenn Retry sinnvoll ist
        """
        # Custom Retry-Condition
        if self.config.custom_retry_condition:
            return self.config.custom_retry_condition(exception)

        # Exception-Type-basierte Prüfung
        if self.config.retry_on_exceptions:
            return any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions)

        # Status-Code-basierte Prüfung (für HTTP-Exceptions)
        if hasattr(exception, "status_code"):
            return exception.status_code in self.config.retry_on_status_codes

        # Default: Retry für alle Exceptions außer bestimmten
        non_retryable_exceptions = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            SyntaxError,
            TypeError,
            ValueError
        )

        return not isinstance(exception, non_retryable_exceptions)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Berechnet Delay für Retry-Versuch."""
        if self.config.adaptive_enabled:
            return self._calculate_adaptive_delay(attempt)

        return RetryDelayCalculator.calculate_delay(
            strategy=self.config.strategy,
            attempt=attempt,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
            jitter_range=self.config.jitter_range,
            custom_delay_function=self.config.custom_delay_function
        )

    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Berechnet adaptiven Delay basierend auf Performance-Historie."""
        base_delay = RetryDelayCalculator.calculate_delay(
            strategy=self.config.strategy,
            attempt=attempt,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
            jitter_range=self.config.jitter_range
        )

        # Adaptive Anpassung basierend auf Performance
        if self.metrics.avg_response_time > self.config.performance_threshold_ms / 1000:
            self._adaptive_multiplier = min(self._adaptive_multiplier * 1.5, 3.0)
        else:
            self._adaptive_multiplier = max(self._adaptive_multiplier * 0.9, 0.5)

        return base_delay * self._adaptive_multiplier

    async def _on_success(self, attempt: int, response_time: float, operation_name: str) -> None:
        """Behandelt erfolgreichen Function-Call."""
        self.metrics.total_attempts += 1
        self.metrics.successful_attempts += 1
        self.metrics.last_success_time = time.time()
        self.metrics.response_times.append(response_time)

        # Performance-Historie aktualisieren
        if self.config.adaptive_enabled:
            self._performance_history.append(response_time)

        # Metrics
        self._metrics_collector.record_histogram(
            "retry.response_time",
            response_time,
            tags={"operation": operation_name, "attempt": str(attempt)}
        )

        logger.debug(f"Retry-Success für '{operation_name}' nach {attempt + 1} Versuchen")

    async def _on_retry(self, attempt: int, exception: Exception, delay: float, operation_name: str) -> None:
        """Behandelt Retry-Versuch."""
        self.metrics.total_attempts += 1
        self.metrics.failed_attempts += 1
        self.metrics.last_failure_time = time.time()
        self.metrics.total_retry_time += delay

        # Metrics
        self._metrics_collector.increment_counter(
            "retry.attempts",
            tags={
                "operation": operation_name,
                "attempt": str(attempt),
                "exception_type": type(exception).__name__
            }
        )

        # Callback
        if self.config.on_retry:
            await self.config.on_retry(attempt, exception, delay)

        logger.debug(
            f"Retry {attempt + 1}/{self.config.max_attempts} für '{operation_name}' "
            f"nach {delay:.2f}s (Fehler: {type(exception).__name__})"
        )

    async def _on_retry_exhausted(self, attempts: int, last_exception: Exception | None, operation_name: str) -> None:
        """Behandelt erschöpfte Retry-Versuche."""
        # Metrics
        self._metrics_collector.increment_counter(
            "retry.exhausted",
            tags={
                "operation": operation_name,
                "attempts": str(attempts),
                "last_exception_type": type(last_exception).__name__ if last_exception else "unknown"
            }
        )

        # Callback
        if self.config.on_retry_exhausted is not None and last_exception:
            await self.config.on_retry_exhausted(attempts, last_exception)

        logger.warning(
            f"Retry-Versuche für '{operation_name}' nach {attempts} Versuchen erschöpft. "
            f"Letzter Fehler: {type(last_exception).__name__ if last_exception else 'unknown'}"
        )
