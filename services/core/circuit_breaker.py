"""Einfacher Circuit Breaker für ausgehende HTTP/gRPC-Aufrufe.

Dieser Circuit Breaker ist generisch und kann mit beliebigen Callables
verwendet werden. Er unterstützt die Zustände CLOSED, OPEN, HALF_OPEN
und nutzt exponentielles Backoff für Probeaufrufe im HALF_OPEN Zustand.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)


class CircuitState(Enum):
    """Zustände des Circuit Breakers."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Konstanten für Circuit Breaker Konfiguration
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_OPEN_TIMEOUT_SECONDS = 20.0
DEFAULT_HALF_OPEN_MAX_CONCURRENT = 1
DEFAULT_RECOVERY_BACKOFF_BASE = 1.5
DEFAULT_RECOVERY_BACKOFF_MAX_SECONDS = 30.0


@dataclass(slots=True)
class CircuitPolicy:
    """Konfiguration für Circuit Breaker.

    Definiert das Verhalten des Circuit Breakers bei Fehlern und Recovery.
    """

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    open_timeout_seconds: float = DEFAULT_OPEN_TIMEOUT_SECONDS
    half_open_max_concurrent: int = DEFAULT_HALF_OPEN_MAX_CONCURRENT
    recovery_backoff_base: float = DEFAULT_RECOVERY_BACKOFF_BASE
    recovery_backoff_max_seconds: float = DEFAULT_RECOVERY_BACKOFF_MAX_SECONDS


class CircuitBreaker:
    """Einfacher asynchroner Circuit Breaker.

    Implementiert das Circuit Breaker Pattern für Fehlerbehandlung bei
    externen Service-Aufrufen. Unterstützt die Zustände CLOSED, OPEN
    und HALF_OPEN mit konfigurierbarem Backoff-Verhalten.

    Args:
        name: Eindeutiger Name für Logging und Monitoring
        policy: Konfiguration für Circuit Breaker Verhalten
    """

    def __init__(self, name: str, policy: CircuitPolicy | None = None) -> None:
        self.name = name
        self.policy = policy or CircuitPolicy()
        self._state: CircuitState = CircuitState.CLOSED
        self._failures: int = 0
        self._opened_at: float = 0.0
        self._half_open_sem = asyncio.Semaphore(self.policy.half_open_max_concurrent)

    @property
    def state(self) -> CircuitState:
        """Gibt aktuellen Zustand zurück."""
        return self._state

    def _transition_to_open(self) -> None:
        """Übergang zu OPEN-Zustand bei Überschreitung der Fehlerschwelle."""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        logger.warning(f"Circuit '{self.name}' -> OPEN (failures={self._failures})")

    def _transition_to_half_open(self) -> None:
        """Übergang zu HALF_OPEN-Zustand für Recovery-Versuche."""
        self._state = CircuitState.HALF_OPEN
        logger.info(f"Circuit '{self.name}' -> HALF_OPEN")

    def _transition_to_closed(self) -> None:
        """Übergang zu CLOSED-Zustand nach erfolgreichem Recovery."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        logger.info(f"Circuit '{self.name}' -> CLOSED")

    def _can_attempt_recovery(self) -> bool:
        """Prüft ob Recovery-Versuch möglich ist basierend auf Backoff-Strategie.

        Returns:
            True wenn genügend Zeit vergangen ist für Recovery-Versuch
        """
        if self._state != CircuitState.OPEN:
            return False

        elapsed_time = time.time() - self._opened_at

        # Exponentielles Backoff berechnen
        excess_failures = max(0, self._failures - self.policy.failure_threshold)
        exponential_backoff = self.policy.open_timeout_seconds * (
            self.policy.recovery_backoff_base ** excess_failures
        )

        # Backoff auf Maximum begrenzen
        actual_backoff = min(exponential_backoff, self.policy.recovery_backoff_max_seconds)
        required_wait_time = max(self.policy.open_timeout_seconds, actual_backoff)

        return elapsed_time >= required_wait_time

    async def _execute_in_half_open(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Führt Aufruf im HALF_OPEN Zustand aus.

        Args:
            func: Asynchrones Callable
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente

        Returns:
            Ergebnis des Callables

        Raises:
            Exception: Bei Fehlern wird Circuit wieder geöffnet
        """
        async with self._half_open_sem:
            try:
                result = await func(*args, **kwargs)
                self._transition_to_closed()
                return result
            except Exception:
                self._failures += 1
                self._transition_to_open()
                raise

    async def _execute_in_closed(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Führt Aufruf im CLOSED Zustand aus.

        Args:
            func: Asynchrones Callable
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente

        Returns:
            Ergebnis des Callables

        Raises:
            Exception: Bei zu vielen Fehlern wird Circuit geöffnet
        """
        try:
            result = await func(*args, **kwargs)
            self._failures = 0  # Reset bei Erfolg
            return result
        except Exception:
            self._failures += 1
            if self._failures >= self.policy.failure_threshold:
                self._transition_to_open()
            raise

    async def call(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Führt Aufruf unter Circuit Breaker Kontrolle aus.

        Args:
            func: Asynchrones Callable
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente

        Returns:
            Ergebnis des Callables

        Raises:
            RuntimeError: Wenn Circuit OPEN ist und kein Recovery möglich
            Exception: Fehler des ursprünglichen Callables
        """
        # Prüfe OPEN Zustand und Recovery-Möglichkeit
        if self._state == CircuitState.OPEN:
            if not self._can_attempt_recovery():
                raise RuntimeError(f"Circuit '{self.name}' is OPEN")
            self._transition_to_half_open()

        # Führe Aufruf je nach Zustand aus
        if self._state == CircuitState.HALF_OPEN:
            return await self._execute_in_half_open(func, *args, **kwargs)
        # CLOSED
        return await self._execute_in_closed(func, *args, **kwargs)


__all__ = ["CircuitBreaker", "CircuitPolicy", "CircuitState"]


