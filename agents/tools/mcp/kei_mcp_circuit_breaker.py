"""Circuit Breaker Pattern für externe MCP Server.

Implementiert Circuit Breaker mit automatischer Wiederherstellung für robuste
Fehlerbehandlung bei externen Service-Aufrufen.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from kei_logging import get_logger
from observability import trace_function

from .core.constants import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_MONITOR_WINDOW_SECONDS,
    DEFAULT_RECOVERY_TIMEOUT,
    DEFAULT_SUCCESS_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS,
)

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit Breaker Zustände."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Konfiguration für Circuit Breaker.

    Attributes:
        failure_threshold: Anzahl Fehler bevor Circuit öffnet
        recovery_timeout: Zeit in Sekunden bevor Half-Open versucht wird
        success_threshold: Anzahl erfolgreiche Requests für Schließung
        timeout_seconds: Timeout für einzelne Requests
        monitor_window_seconds: Zeitfenster für Fehler-Monitoring
    """

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT
    success_threshold: int = DEFAULT_SUCCESS_THRESHOLD
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    monitor_window_seconds: float = DEFAULT_MONITOR_WINDOW_SECONDS


@dataclass
class CircuitBreakerStats:
    """Statistiken für Circuit Breaker.

    Attributes:
        state: Aktueller Zustand
        failure_count: Anzahl Fehler im aktuellen Fenster
        success_count: Anzahl Erfolge im Half-Open Zustand
        last_failure_time: Zeitstempel des letzten Fehlers
        last_success_time: Zeitstempel des letzten Erfolgs
        total_requests: Gesamtanzahl Requests
        total_failures: Gesamtanzahl Fehler
        state_changes: Anzahl Zustandsänderungen
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_requests: int = 0
    total_failures: int = 0
    state_changes: int = 0
    window_start: float = field(default_factory=time.time)


class CircuitBreakerException(Exception):
    """Exception für Circuit Breaker Fehler."""

    def __init__(self, message: str, state: CircuitState, stats: CircuitBreakerStats):
        super().__init__(message)
        self.state = state
        self.stats = stats


class CircuitBreaker(Generic[T]):
    """Circuit Breaker für externe Service-Aufrufe.

    Implementiert das Circuit Breaker Pattern mit automatischer Wiederherstellung
    und detailliertem Monitoring.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        """Initialisiert Circuit Breaker.

        Args:
            name: Name des Circuit Breakers für Logging
            config: Konfiguration, verwendet Defaults wenn None
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    @trace_function("circuit_breaker.call")
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Führt Funktion mit Circuit Breaker Schutz aus.

        Args:
            func: Auszuführende Funktion
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

            # Prüfe Circuit Zustand
            if self.stats.state == CircuitState.OPEN:
                logger.warning(f"Circuit Breaker {self.name} ist OPEN - Request blockiert")
                raise CircuitBreakerException(
                    f"Circuit Breaker {self.name} ist offen",
                    self.stats.state,
                    self.stats
                )

            self.stats.total_requests += 1

        # Führe Funktion aus
        start_time = time.time()
        try:
            # Timeout für einzelne Requests
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )

            # Erfolg verarbeiten
            await self._record_success()

            execution_time = time.time() - start_time
            logger.debug(f"Circuit Breaker {self.name}: Erfolg in {execution_time:.2f}s")

            return result

        except TimeoutError as exc:
            await self._record_failure(exc)
            logger.warning(f"Circuit Breaker {self.name}: Timeout nach {self.config.timeout_seconds}s")
            raise

        except Exception as exc:
            await self._record_failure(exc)
            execution_time = time.time() - start_time
            logger.warning(f"Circuit Breaker {self.name}: Fehler nach {execution_time:.2f}s - {exc}")
            raise

    async def _record_success(self):
        """Verzeichnet erfolgreichen Request."""
        async with self._lock:
            self.stats.last_success_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                self.stats.success_count += 1

                # Prüfe ob genug Erfolge für Schließung
                if self.stats.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()

            # Reset Failure Count bei Erfolg
            self.stats.failure_count = 0

    async def _record_failure(self, _exception: Exception):
        """Verzeichnet fehlgeschlagenen Request.

        Args:
            _exception: Aufgetretene Exception
        """
        async with self._lock:
            self.stats.last_failure_time = time.time()
            self.stats.total_failures += 1
            self.stats.failure_count += 1

            # Prüfe ob Failure Threshold erreicht
            if (self.stats.state == CircuitState.CLOSED and
                self.stats.failure_count >= self.config.failure_threshold):
                await self._transition_to_open()

            elif self.stats.state == CircuitState.HALF_OPEN:
                # Bei Fehler im Half-Open zurück zu Open
                await self._transition_to_open()

    def _update_state(self):
        """Aktualisiert Circuit Breaker Zustand basierend auf Zeit."""
        current_time = time.time()

        # Reset Monitoring Window
        if current_time - self.stats.window_start >= self.config.monitor_window_seconds:
            self.stats.failure_count = 0
            self.stats.window_start = current_time

        # Transition von Open zu Half-Open nach Recovery Timeout
        if (self.stats.state == CircuitState.OPEN and
            self.stats.last_failure_time and
            current_time - self.stats.last_failure_time >= self.config.recovery_timeout):
            asyncio.create_task(self._transition_to_half_open())

    async def _transition_to_open(self):
        """Übergang zu OPEN Zustand."""
        if self.stats.state != CircuitState.OPEN:
            old_state = self.stats.state
            self.stats.state = CircuitState.OPEN
            self.stats.state_changes += 1
            self.stats.success_count = 0

            logger.error(f"Circuit Breaker {self.name}: {old_state.value} -> OPEN "
                        f"(Fehler: {self.stats.failure_count}/{self.config.failure_threshold})")

    async def _transition_to_half_open(self):
        """Übergang zu HALF_OPEN Zustand."""
        if self.stats.state == CircuitState.OPEN:
            self.stats.state = CircuitState.HALF_OPEN
            self.stats.state_changes += 1
            self.stats.success_count = 0

            logger.info(f"Circuit Breaker {self.name}: OPEN -> HALF_OPEN "
                       f"(Recovery Timeout erreicht)")

    async def _transition_to_closed(self):
        """Übergang zu CLOSED Zustand."""
        if self.stats.state != CircuitState.CLOSED:
            old_state = self.stats.state
            self.stats.state = CircuitState.CLOSED
            self.stats.state_changes += 1
            self.stats.failure_count = 0
            self.stats.success_count = 0

            logger.info(f"Circuit Breaker {self.name}: {old_state.value} -> CLOSED "
                       f"(Erfolge: {self.stats.success_count}/{self.config.success_threshold})")

    def get_stats(self) -> dict[str, Any]:
        """Gibt aktuelle Statistiken zurück.

        Returns:
            Dictionary mit Circuit Breaker Statistiken
        """
        current_time = time.time()

        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "state_changes": self.stats.state_changes,
            "last_failure_time": self.stats.last_failure_time,
            "last_success_time": self.stats.last_success_time,
            "uptime_seconds": current_time - self.stats.window_start,
            "error_rate": (
                self.stats.total_failures / self.stats.total_requests
                if self.stats.total_requests > 0 else 0.0
            ),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds
            }
        }

    async def reset(self):
        """Setzt Circuit Breaker zurück."""
        async with self._lock:
            self.stats = CircuitBreakerStats()
            logger.info(f"Circuit Breaker {self.name} zurückgesetzt")

    async def force_open(self):
        """Erzwingt OPEN Zustand (für Testing/Maintenance)."""
        async with self._lock:
            await self._transition_to_open()
            logger.warning(f"Circuit Breaker {self.name} manuell geöffnet")

    async def force_close(self):
        """Erzwingt CLOSED Zustand (für Testing/Recovery)."""
        async with self._lock:
            await self._transition_to_closed()
            logger.info(f"Circuit Breaker {self.name} manuell geschlossen")


class CircuitBreakerRegistry:
    """Registry für Circuit Breaker Management."""

    def __init__(self):
        """Initialisiert Registry."""
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
        """Gibt existierenden Circuit Breaker zurück oder erstellt neuen.

        Args:
            name: Name des Circuit Breakers
            config: Konfiguration für neuen Circuit Breaker

        Returns:
            Circuit Breaker Instanz
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)

        return self._breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Gibt Statistiken aller Circuit Breaker zurück.

        Returns:
            Dictionary mit Statistiken aller Circuit Breaker
        """
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    async def reset_all(self):
        """Setzt alle Circuit Breaker zurück."""
        for breaker in self._breakers.values():
            await breaker.reset()

        logger.info(f"Alle {len(self._breakers)} Circuit Breaker zurückgesetzt")


# Globale Registry
circuit_breaker_registry = CircuitBreakerRegistry()


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerException",
    "CircuitBreakerRegistry",
    "CircuitBreakerStats",
    "CircuitState",
    "circuit_breaker_registry"
]
