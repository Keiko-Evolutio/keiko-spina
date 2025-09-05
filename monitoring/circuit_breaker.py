"""Circuit Breaker Implementation.
Implementiert Circuit Breaker Pattern für externe Services.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger

from .interfaces import ICircuitBreaker, IMetricsCollector

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit Breaker Zustände."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker Konfiguration."""
    failure_threshold: int = 5          # Anzahl Fehler bis Circuit öffnet
    recovery_timeout_seconds: int = 60  # Zeit bis Half-Open-Test
    success_threshold: int = 3          # Erfolge bis Circuit schließt
    timeout_seconds: float = 30.0       # Request-Timeout

    # Erweiterte Konfiguration
    failure_rate_threshold: float = 0.5  # 50% Fehlerrate
    minimum_requests: int = 10           # Mindest-Requests für Statistik
    sliding_window_seconds: int = 60     # Zeitfenster für Statistik


class CircuitBreaker(ICircuitBreaker):
    """Circuit Breaker Implementation für externe Services.
    Verhindert Cascade-Failures durch automatisches Blockieren fehlerhafter Services.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig, metrics_collector: IMetricsCollector):
        self.name = name
        self.config = config
        self.metrics_collector = metrics_collector

        # Circuit State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None

        # Sliding Window für Statistiken
        self._request_history: list = []  # (timestamp, success: bool)

        # Locks für Thread-Safety
        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' initialized with config: {config}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Führt Funktion mit Circuit Breaker aus."""
        async with self._lock:
            # State-Check vor Ausführung
            if not await self._can_execute():
                self._record_blocked_request()
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")

        start_time = time.time()

        try:
            # Funktion mit Timeout ausführen
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=self.config.timeout_seconds
                )

            # Erfolg verarbeiten
            await self._record_success()

            # Response-Zeit tracken
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics_collector.observe_histogram(
                "circuit_breaker_request_duration_seconds",
                response_time_ms / 1000.0,
                labels={"circuit": self.name, "result": "success"}
            )

            return result

        except TimeoutError:
            await self._record_failure("timeout")
            raise CircuitBreakerTimeoutError(f"Circuit breaker '{self.name}' timeout after {self.config.timeout_seconds}s")

        except Exception as e:
            await self._record_failure(type(e).__name__)
            raise

    async def _can_execute(self) -> bool:
        """Prüft ob Request ausgeführt werden kann."""
        current_time = datetime.utcnow()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Prüfe ob Recovery-Timeout erreicht
            if (self._last_failure_time and
                current_time - self._last_failure_time >= timedelta(seconds=self.config.recovery_timeout_seconds)):

                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

                self.metrics_collector.set_gauge(
                    "circuit_breaker_state",
                    2,  # HALF_OPEN = 2
                    labels={"circuit": self.name}
                )

                return True

            return False

        if self._state == CircuitState.HALF_OPEN:
            return True

        return False

    async def _record_success(self) -> None:
        """Verarbeitet erfolgreichen Request."""
        current_time = datetime.utcnow()

        self._last_success_time = current_time
        self._request_history.append((time.time(), True))

        # Cleanup alte Einträge
        self._cleanup_history()

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1

            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0

                logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")

                self.metrics_collector.set_gauge(
                    "circuit_breaker_state",
                    0,  # CLOSED = 0
                    labels={"circuit": self.name}
                )

        # Metriken
        self.metrics_collector.increment_counter(
            "circuit_breaker_requests_total",
            labels={"circuit": self.name, "result": "success"}
        )

    async def _record_failure(self, error_type: str) -> None:
        """Verarbeitet fehlgeschlagenen Request."""
        current_time = datetime.utcnow()

        self._last_failure_time = current_time
        self._failure_count += 1
        self._request_history.append((time.time(), False))

        # Cleanup alte Einträge
        self._cleanup_history()

        # State-Transition prüfen
        if self._state == CircuitState.CLOSED:
            # Prüfe Failure-Threshold
            if (self._failure_count >= self.config.failure_threshold or
                self._calculate_failure_rate() >= self.config.failure_rate_threshold):

                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN due to failures")

                self.metrics_collector.set_gauge(
                    "circuit_breaker_state",
                    1,  # OPEN = 1
                    labels={"circuit": self.name}
                )

                self.metrics_collector.increment_counter(
                    "circuit_breaker_opened_total",
                    labels={"circuit": self.name}
                )

        elif self._state == CircuitState.HALF_OPEN:
            # Bei Fehler in Half-Open zurück zu Open
            self._state = CircuitState.OPEN
            self._success_count = 0

            logger.warning(f"Circuit breaker '{self.name}' transitioned back to OPEN")

            self.metrics_collector.set_gauge(
                "circuit_breaker_state",
                1,  # OPEN = 1
                labels={"circuit": self.name}
            )

        # Metriken
        self.metrics_collector.increment_counter(
            "circuit_breaker_requests_total",
            labels={"circuit": self.name, "result": "failure", "error_type": error_type}
        )

    def _record_blocked_request(self) -> None:
        """Verarbeitet blockierten Request."""
        self.metrics_collector.increment_counter(
            "circuit_breaker_requests_total",
            labels={"circuit": self.name, "result": "blocked"}
        )

    def _cleanup_history(self) -> None:
        """Entfernt alte Einträge aus der Request-History."""
        current_time = time.time()
        cutoff_time = current_time - self.config.sliding_window_seconds

        self._request_history = [
            entry for entry in self._request_history
            if entry[0] >= cutoff_time
        ]

    def _calculate_failure_rate(self) -> float:
        """Berechnet aktuelle Fehlerrate."""
        if len(self._request_history) < self.config.minimum_requests:
            return 0.0

        failures = sum(1 for _, success in self._request_history if not success)
        total = len(self._request_history)

        return failures / total if total > 0 else 0.0

    def get_state(self) -> str:
        """Gibt aktuellen Circuit Breaker State zurück."""
        return self._state.value

    def get_failure_rate(self) -> float:
        """Gibt aktuelle Failure-Rate zurück."""
        return self._calculate_failure_rate()

    def get_statistics(self) -> dict[str, Any]:
        """Gibt detaillierte Statistiken zurück."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_rate": self._calculate_failure_rate(),
            "total_requests": len(self._request_history),
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds
            }
        }

    async def reset(self) -> None:
        """Setzt Circuit Breaker zurück (für Tests/Admin)."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._last_success_time = None
            self._request_history.clear()

            logger.info(f"Circuit breaker '{self.name}' reset")

            self.metrics_collector.set_gauge(
                "circuit_breaker_state",
                0,  # CLOSED = 0
                labels={"circuit": self.name}
            )


# Custom Exceptions
class CircuitBreakerError(Exception):
    """Basis-Exception für Circuit Breaker Fehler."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Exception wenn Circuit Breaker offen ist."""


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Exception bei Circuit Breaker Timeout."""


# Circuit Breaker Manager
class CircuitBreakerManager:
    """Verwaltet mehrere Circuit Breaker."""

    def __init__(self, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector
        self._breakers: dict[str, CircuitBreaker] = {}

    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Erstellt neuen Circuit Breaker."""
        if config is None:
            config = CircuitBreakerConfig()

        breaker = CircuitBreaker(name, config, self.metrics_collector)
        self._breakers[name] = breaker

        return breaker

    def get_circuit_breaker(self, name: str) -> CircuitBreaker | None:
        """Gibt Circuit Breaker zurück."""
        return self._breakers.get(name)

    def get_all_statistics(self) -> dict[str, Any]:
        """Gibt Statistiken aller Circuit Breaker zurück."""
        return {
            name: breaker.get_statistics()
            for name, breaker in self._breakers.items()
        }
