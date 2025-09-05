# backend/agents/resilience/retry_manager.py
"""Upstream-spezifische Retry-Mechanismen für Personal Assistant

Implementiert intelligente Retry-Strategien pro Upstream-Service/Agent
mit verschiedenen Retry-Patterns und adaptiven Delays.
"""

import asyncio
import random
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)


class RetryStrategy(str, Enum):
    """Retry-Strategien."""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Konfiguration für Retry-Mechanismen."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1

    retry_on_exceptions: list[type] = field(default_factory=list)
    retry_on_status_codes: list[int] = field(default_factory=lambda: [429, 502, 503, 504])

    custom_retry_condition: Callable[[Exception], bool] | None = None
    custom_delay_function: Callable[[int, float], float] | None = None
    adaptive_enabled: bool = False
    performance_window_size: int = 100
    performance_threshold_ms: float = 1000.0

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Prüft ob Retry durchgeführt werden soll.

        Args:
            exception: Aufgetretene Exception
            attempt: Aktueller Versuch (0-basiert)

        Returns:
            True wenn Retry durchgeführt werden soll
        """
        if attempt >= self.max_attempts:
            return False

        if self.custom_retry_condition:
            return self.custom_retry_condition(exception)

        if self.retry_on_exceptions:
            return any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions)

        if hasattr(exception, "status_code"):
            return exception.status_code in self.retry_on_status_codes
        return self._is_temporary_error(exception)

    def _is_temporary_error(self, exception: Exception) -> bool:
        """Prüft ob Exception temporär ist."""
        error_str = str(exception).lower()
        temporary_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "overloaded",
            "rate limit",
        ]
        return any(indicator in error_str for indicator in temporary_indicators)


@dataclass
class UpstreamRetryPolicy:
    """Retry-Policy für spezifischen Upstream-Service."""

    upstream_id: str
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    config: RetryConfig = field(default_factory=RetryConfig)

    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_success_rate: float = 1.0
    current_base_delay: float = 1.0
    last_adjustment_time: float = 0.0

    def calculate_delay(self, attempt: int) -> float:
        """Berechnet Delay für Retry-Versuch.

        Args:
            attempt: Aktueller Versuch (0-basiert)

        Returns:
            Delay in Sekunden
        """
        if self.config.custom_delay_function:
            delay = self.config.custom_delay_function(attempt, self.current_base_delay)
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.current_base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.current_base_delay * (self.config.exponential_base**attempt)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.current_base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.current_base_delay * self._fibonacci(attempt + 1)
        elif self.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt)
        else:
            delay = self.current_base_delay

        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            jitter_range = delay * self.config.jitter_range
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter

        return max(0.0, delay)

    def _fibonacci(self, n: int) -> int:
        """Berechnet Fibonacci-Zahl."""
        if n <= 1:
            return n

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b

        return b

    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Berechnet adaptiven Delay basierend auf Performance."""
        if not self.recent_response_times:
            return self.current_base_delay

        avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times)

        if avg_response_time > self.config.performance_threshold_ms:
            multiplier = min(2.0, avg_response_time / self.config.performance_threshold_ms)
            self.current_base_delay = min(
                self.config.max_delay / 4, self.config.base_delay * multiplier
            )
        else:
            self.current_base_delay = max(
                self.config.base_delay * 0.5,
                self.current_base_delay * 0.9,
            )

        return self.current_base_delay * (self.config.exponential_base**attempt)

    def record_response_time(self, response_time_ms: float):
        """Zeichnet Response-Zeit auf."""
        self.recent_response_times.append(response_time_ms)

    def update_success_rate(self, success: bool):
        """Aktualisiert Success-Rate."""
        alpha = 0.1
        new_value = 1.0 if success else 0.0
        self.recent_success_rate = (alpha * new_value) + ((1 - alpha) * self.recent_success_rate)


class AdaptiveRetryManager:
    """Adaptiver Retry-Manager für Performance-basierte Anpassungen."""

    def __init__(self):
        """Initialisiert Adaptive Retry-Manager."""
        self._upstream_policies: dict[str, UpstreamRetryPolicy] = {}
        self._performance_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()


        self._metrics_collector = MetricsCollector()

    def get_policy(self, upstream_id: str) -> UpstreamRetryPolicy:
        """Holt oder erstellt Retry-Policy für Upstream.

        Args:
            upstream_id: Upstream-Service-ID

        Returns:
            Retry-Policy für Upstream
        """
        with self._lock:
            if upstream_id not in self._upstream_policies:
                self._upstream_policies[upstream_id] = UpstreamRetryPolicy(
                    upstream_id=upstream_id,
                    strategy=RetryStrategy.ADAPTIVE,
                    config=RetryConfig(adaptive_enabled=True),
                )

            return self._upstream_policies[upstream_id]

    def record_call_result(
        self,
        upstream_id: str,
        success: bool,
        response_time_ms: float,
        exception: Exception | None = None,
    ):
        """Zeichnet Call-Ergebnis für Adaptive Retry auf.

        Args:
            upstream_id: Upstream-Service-ID
            success: Ob Call erfolgreich war
            response_time_ms: Response-Zeit in Millisekunden
            exception: Optional aufgetretene Exception
        """
        policy = self.get_policy(upstream_id)

        with self._lock:
            policy.record_response_time(response_time_ms)
            policy.update_success_rate(success)
            self._performance_history[upstream_id].append(
                {
                    "timestamp": time.time(),
                    "success": success,
                    "response_time_ms": response_time_ms,
                    "exception_type": type(exception).__name__ if exception else None,
                }
            )


        self._metrics_collector.record_histogram(
            "retry_manager.response_time",
            response_time_ms / 1000.0,  # Convert to seconds
            tags={"upstream_id": upstream_id, "success": str(success)},
        )

        self._metrics_collector.record_gauge(
            "retry_manager.success_rate",
            policy.recent_success_rate,
            tags={"upstream_id": upstream_id},
        )

    def get_performance_metrics(self, upstream_id: str) -> dict[str, Any]:
        """Holt Performance-Metriken für Upstream.

        Args:
            upstream_id: Upstream-Service-ID

        Returns:
            Performance-Metriken
        """
        policy = self.get_policy(upstream_id)
        history = self._performance_history[upstream_id]

        if not history:
            return {
                "upstream_id": upstream_id,
                "avg_response_time_ms": 0.0,
                "success_rate": 1.0,
                "total_calls": 0,
                "current_base_delay": policy.current_base_delay,
            }


        total_calls = len(history)
        successful_calls = sum(1 for call in history if call["success"])
        avg_response_time = sum(call["response_time_ms"] for call in history) / total_calls

        return {
            "upstream_id": upstream_id,
            "avg_response_time_ms": avg_response_time,
            "success_rate": successful_calls / total_calls,
            "total_calls": total_calls,
            "current_base_delay": policy.current_base_delay,
            "recent_success_rate": policy.recent_success_rate,
        }


class RetryManager:
    """Haupt-Retry-Manager für alle Upstream-Services."""

    def __init__(self, default_config: RetryConfig | None = None):
        """Initialisiert Retry-Manager.

        Args:
            default_config: Standard-Retry-Konfiguration
        """
        self.default_config = default_config or RetryConfig()
        self._upstream_policies: dict[str, UpstreamRetryPolicy] = {}
        self._adaptive_manager = AdaptiveRetryManager()
        self._lock = threading.RLock()


        self._metrics_collector = MetricsCollector()

    def configure_upstream(
        self,
        upstream_id: str,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        config: RetryConfig | None = None,
    ) -> UpstreamRetryPolicy:
        """Konfiguriert Retry-Policy für Upstream-Service.

        Args:
            upstream_id: Upstream-Service-ID
            strategy: Retry-Strategie
            config: Optional spezifische Konfiguration

        Returns:
            Konfigurierte Retry-Policy
        """
        with self._lock:
            retry_config = config or self.default_config

            self._upstream_policies[upstream_id] = UpstreamRetryPolicy(
                upstream_id=upstream_id, strategy=strategy, config=retry_config
            )

            return self._upstream_policies[upstream_id]

    def get_policy(self, upstream_id: str) -> UpstreamRetryPolicy:
        """Holt Retry-Policy für Upstream-Service.

        Args:
            upstream_id: Upstream-Service-ID

        Returns:
            Retry-Policy für Upstream
        """
        with self._lock:
            if upstream_id not in self._upstream_policies:

                self._upstream_policies[upstream_id] = UpstreamRetryPolicy(
                    upstream_id=upstream_id, config=self.default_config
                )

            return self._upstream_policies[upstream_id]

    async def execute_with_retry(
        self, upstream_id: str, func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """Führt Funktion mit Retry-Mechanismus aus.

        Args:
            upstream_id: Upstream-Service-ID
            func: Auszuführende Funktion
            *args: Funktions-Argumente
            **kwargs: Funktions-Keyword-Argumente

        Returns:
            Funktions-Ergebnis

        Raises:
            RetryExhaustedError: Bei erschöpften Retry-Versuchen
        """
        policy = self.get_policy(upstream_id)
        last_exception = None

        for attempt in range(policy.config.max_attempts):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                response_time_ms = (time.time() - start_time) * 1000

                if policy.config.adaptive_enabled:
                    self._adaptive_manager.record_call_result(upstream_id, True, response_time_ms)
                self._metrics_collector.increment_counter(
                    "retry_manager.calls_successful",
                    tags={"upstream_id": upstream_id, "attempt": str(attempt + 1)},
                )

                return result

            except Exception as e:
                last_exception = e
                response_time_ms = (time.time() - start_time) * 1000

                if policy.config.adaptive_enabled:
                    self._adaptive_manager.record_call_result(
                        upstream_id, False, response_time_ms, e
                    )

                if not policy.config.should_retry(e, attempt):
                    break

                if attempt >= policy.config.max_attempts - 1:
                    break
                delay = policy.calculate_delay(attempt)

                logger.warning(
                    f"Retry {attempt + 1}/{policy.config.max_attempts} für {upstream_id} "
                    f"nach {delay:.2f}s. Fehler: {e}"
                )

                self._metrics_collector.increment_counter(
                    "retry_manager.calls_retried",
                    tags={
                        "upstream_id": upstream_id,
                        "attempt": str(attempt + 1),
                        "exception_type": type(e).__name__,
                    },
                )
                await asyncio.sleep(delay)

        # Alle Retry-Versuche erschöpft
        self._metrics_collector.increment_counter(
            "retry_manager.calls_exhausted",
            tags={
                "upstream_id": upstream_id,
                "exception_type": type(last_exception).__name__ if last_exception else "unknown",
            },
        )

        raise RetryExhaustedError(
            f"Retry für {upstream_id} nach {policy.config.max_attempts} Versuchen erschöpft",
            last_exception=last_exception,
        )

    def get_all_policies(self) -> dict[str, UpstreamRetryPolicy]:
        """Holt alle Retry-Policies.

        Returns:
            Dictionary aller Retry-Policies
        """
        with self._lock:
            return self._upstream_policies.copy()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt Zusammenfassung aller Retry-Metriken.

        Returns:
            Retry-Metriken-Zusammenfassung
        """
        with self._lock:
            summary = {
                "total_upstreams": len(self._upstream_policies),
                "strategies": defaultdict(int),
                "upstreams": {},
            }

            for upstream_id, policy in self._upstream_policies.items():
                summary["strategies"][policy.strategy.value] += 1


                if policy.config.adaptive_enabled:
                    perf_metrics = self._adaptive_manager.get_performance_metrics(upstream_id)
                    summary["upstreams"][upstream_id] = {
                        "strategy": policy.strategy.value,
                        "current_base_delay": policy.current_base_delay,
                        **perf_metrics,
                    }
                else:
                    summary["upstreams"][upstream_id] = {
                        "strategy": policy.strategy.value,
                        "current_base_delay": policy.current_base_delay,
                    }

            return summary



class RetryExhaustedError(Exception):
    """Exception wenn alle Retry-Versuche erschöpft sind."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception
