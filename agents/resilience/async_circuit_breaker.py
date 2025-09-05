# backend/agents/resilience/async_circuit_breaker.py
"""Async Circuit-Breaker für Personal Assistant

Verwendet die gemeinsame Circuit Breaker-Basis-Implementierung.
Behält Kompatibilität für bestehende APIs bei.
"""
import logging
import time
from typing import Any, TypeVar

# Fallback für Logging
try:
    from kei_logging import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)

from ..common.circuit_breaker_base import (
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)
from ..common.circuit_breaker_base import (
    CircuitBreakerState as CircuitState,
)
from ..common.constants import CircuitBreakerDefaults

logger = get_logger(__name__)

T = TypeVar("T")

# Kompatibilitäts-Aliase
CircuitBreakerError = CircuitBreakerOpenError
class AsyncCircuitBreaker(BaseCircuitBreaker[T]):
    """Async Circuit-Breaker-Implementierung basierend auf gemeinsamer Basis.

    Erweitert die BaseCircuitBreaker-Implementierung um spezifische
    Async-Features und behält Kompatibilität bei.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        """Initialisiert Async Circuit-Breaker.

        Args:
            name: Circuit-Breaker-Name
            config: Konfiguration (optional, verwendet Defaults)
        """
        # Verwende Standard-Werte falls keine Konfiguration gegeben
        if config is None:
            config = CircuitBreakerConfig(
                failure_threshold=CircuitBreakerDefaults.FAILURE_THRESHOLD,
                recovery_timeout=CircuitBreakerDefaults.RECOVERY_TIMEOUT,
                success_threshold=CircuitBreakerDefaults.SUCCESS_THRESHOLD,
                timeout=CircuitBreakerDefaults.TIMEOUT
            )

        super().__init__(name, config)

        logger.debug(f"AsyncCircuitBreaker '{name}' initialisiert mit Config: {config}")

    # Kompatibilitäts-Properties für bestehende Tests
    @property
    def _consecutive_failures(self) -> int:
        """Kompatibilität für consecutive failures."""
        return self.metrics.consecutive_failures

    @property
    def _consecutive_successes(self) -> int:
        """Kompatibilität für consecutive successes."""
        return self.metrics.consecutive_successes

    @property
    def _last_failure_time(self) -> float:
        """Kompatibilität für last failure time."""
        return self.metrics.last_failure_time

    @property
    def _state(self) -> CircuitState:
        """Kompatibilität für state access."""
        return self.state

    def _should_attempt_reset(self) -> bool:
        """Kompatibilität für reset check."""
        return super()._should_attempt_reset()

    def _time_until_retry(self) -> float:
        """Kompatibilität für retry time calculation."""
        return super()._time_until_retry()

    async def _transition_to_open(self) -> None:
        """Kompatibilität für state transition."""
        await super()._transition_to_open()

    async def _transition_to_half_open(self) -> None:
        """Kompatibilität für state transition."""
        await super()._transition_to_half_open()

    async def _transition_to_closed(self) -> None:
        """Kompatibilität für state transition."""
        await super()._transition_to_closed()


    def get_metrics(self) -> dict[str, Any]:
        """Holt Circuit-Breaker-Metriken für Kompatibilität."""
        # Konvertiere zu altem Format für Kompatibilität
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "rejected_calls": self.metrics.rejected_calls,
            "success_rate": self.metrics.success_rate,
            "failure_rate": self.metrics.failure_rate,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "state_changes": getattr(self.metrics, "state_changes", 0),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    async def force_open(self) -> None:
        """Forciert Circuit Breaker in OPEN-Zustand (für Tests)."""
        async with self._lock:
            await self._transition_to_open()
            # Setze last_failure_time für normale Recovery-Tests
            self.metrics.last_failure_time = time.time()
            logger.info(f"Circuit-Breaker {self.name}: forciert geöffnet")

    async def force_open_permanently(self) -> None:
        """Forciert Circuit Breaker dauerhaft in OPEN-Zustand (für Tests)."""
        async with self._lock:
            await self._transition_to_open()
            # Setze last_failure_time weit in die Zukunft
            self.metrics.last_failure_time = time.time() + 3600  # 1 Stunde
            logger.info(f"Circuit-Breaker {self.name}: forciert dauerhaft geöffnet")
