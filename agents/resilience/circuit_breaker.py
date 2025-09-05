# backend/agents/resilience/circuit_breaker.py
"""Capability-spezifische Circuit Breaker für Personal Assistant

Erweitert die bestehende Circuit Breaker-Struktur um individuelle Circuit Breaker
pro Agent-Capability mit separatem Failure-Tracking und Recovery-Mechanismen.
"""

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


class CircuitBreakerState(str, Enum):
    """Circuit Breaker-Zustände."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Konfiguration für Capability-spezifische Circuit Breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    failure_rate_threshold: float = 0.5
    minimum_throughput: int = 10

    sliding_window_size: int = 100
    sliding_window_duration: float = 300.0

    capability_specific_thresholds: dict[str, int] = field(default_factory=dict)
    capability_specific_timeouts: dict[str, float] = field(default_factory=dict)
    on_state_change: Callable[[str, CircuitBreakerState, CircuitBreakerState], Awaitable[None]] | None = None
    on_call_rejected: Callable[[str, str], Awaitable[None]] | None = None


@dataclass
class CircuitBreakerMetrics:
    """Metriken für Circuit Breaker."""

    capability: str
    agent_id: str
    state: CircuitBreakerState

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0

    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_changed_at: float = 0.0

    recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))
    half_open_calls: int = 0
    half_open_successes: int = 0

    def get_failure_rate(self) -> float:
        """Berechnet aktuelle Fehlerrate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def get_recent_failure_rate(self) -> float:
        """Berechnet Fehlerrate im Sliding Window."""
        if not self.recent_calls:
            return 0.0

        failures = sum(1 for call in self.recent_calls if not call["success"])
        return failures / len(self.recent_calls)

    def add_call_result(self, success: bool, duration: float):
        """Fügt Call-Ergebnis zum Sliding Window hinzu."""
        call_data = {"success": success, "timestamp": time.time(), "duration": duration}
        self.recent_calls.append(call_data)

        self.total_calls += 1
        if success:
            self.successful_calls += 1
            self.last_success_time = time.time()
        else:
            self.failed_calls += 1
            self.last_failure_time = time.time()


class CapabilityCircuitBreaker:
    """Circuit Breaker für spezifische Agent-Capability."""

    def __init__(self, capability: str, agent_id: str, config: CircuitBreakerConfig):
        """Initialisiert Capability-spezifischen Circuit Breaker.

        Args:
            capability: Name der Capability
            agent_id: Agent-ID
            config: Circuit Breaker-Konfiguration
        """
        self.capability = capability
        self.agent_id = agent_id
        self.config = config

        self.failure_threshold = config.capability_specific_thresholds.get(
            capability, config.failure_threshold
        )
        self.recovery_timeout = config.capability_specific_timeouts.get(
            capability, config.recovery_timeout
        )

        self.metrics = CircuitBreakerMetrics(
            capability=capability, agent_id=agent_id, state=CircuitBreakerState.CLOSED
        )

        self._lock = threading.RLock()
        self._metrics_collector = MetricsCollector()

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Führt Funktion mit Circuit Breaker-Schutz aus.

        Args:
            func: Auszuführende Funktion
            *args: Funktions-Argumente
            **kwargs: Funktions-Keyword-Argumente

        Returns:
            Funktions-Ergebnis

        Raises:
            CircuitBreakerOpenError: Wenn Circuit Breaker offen ist
        """
        if not await self._can_execute():
            self.metrics.rejected_calls += 1

            if self.config.on_call_rejected:
                await self.config.on_call_rejected(self.agent_id, self.capability)
            self._metrics_collector.increment_counter(
                "circuit_breaker.calls_rejected",
                tags={
                    "agent_id": self.agent_id,
                    "capability": self.capability,
                    "state": self.metrics.state.value,
                },
            )

            raise CircuitBreakerOpenError(
                f"Circuit Breaker für {self.agent_id}.{self.capability} ist offen. "
                f"Nächster Versuch in {self._time_until_retry():.1f}s"
            )

        start_time = time.time()

        try:
            result = await func(*args, **kwargs)

            duration = time.time() - start_time
            await self._on_success(duration)

            return result

        except Exception:
            duration = time.time() - start_time
            await self._on_failure(duration)
            raise

    async def _can_execute(self) -> bool:
        """Prüft ob Ausführung erlaubt ist.

        Returns:
            True wenn Ausführung erlaubt
        """
        with self._lock:
            if self.metrics.state == CircuitBreakerState.CLOSED:
                return True

            if self.metrics.state == CircuitBreakerState.OPEN:
                if time.time() - self.metrics.state_changed_at >= self.recovery_timeout:
                    await self._transition_to_half_open()
                    return True
                return False

            if self.metrics.state == CircuitBreakerState.HALF_OPEN:
                return self.metrics.half_open_calls < self.config.half_open_max_calls

            return False

    async def _on_success(self, duration: float) -> None:
        """Behandelt erfolgreichen Call."""
        with self._lock:
            self.metrics.add_call_result(True, duration)

            if self.metrics.state == CircuitBreakerState.HALF_OPEN:
                self.metrics.half_open_calls += 1
                self.metrics.half_open_successes += 1

                if self.metrics.half_open_calls >= self.config.half_open_max_calls:
                    await self._transition_to_closed()

            elif self.metrics.state == CircuitBreakerState.CLOSED:
                self.metrics.half_open_calls = 0
                self.metrics.half_open_successes = 0

        # Metrics
        self._metrics_collector.record_histogram(
            "circuit_breaker.call_duration",
            duration,
            tags={"agent_id": self.agent_id, "capability": self.capability, "status": "success"},
        )

    async def _on_failure(self, duration: float) -> None:
        """Behandelt fehlgeschlagenen Call."""
        with self._lock:
            self.metrics.add_call_result(False, duration)

            if self.metrics.state == CircuitBreakerState.HALF_OPEN:
                await self._transition_to_open()

            elif self.metrics.state == CircuitBreakerState.CLOSED:
                if self._should_open():
                    await self._transition_to_open()

        # Metrics
        self._metrics_collector.record_histogram(
            "circuit_breaker.call_duration",
            duration,
            tags={"agent_id": self.agent_id, "capability": self.capability, "status": "failure"},
        )

    def _should_open(self) -> bool:
        """Prüft ob Circuit Breaker geöffnet werden soll.

        Returns:
            True wenn Circuit Breaker geöffnet werden soll
        """
        if self.metrics.failed_calls >= self.failure_threshold:
            return True

        if len(self.metrics.recent_calls) >= self.config.minimum_throughput:
            failure_rate = self.metrics.get_recent_failure_rate()
            return failure_rate >= self.config.failure_rate_threshold

        return False

    async def _transition_to_open(self) -> None:
        """Übergang zu OPEN-Status."""
        old_state = self.metrics.state
        self.metrics.state = CircuitBreakerState.OPEN
        self.metrics.state_changed_at = time.time()

        logger.warning(
            f"Circuit Breaker für {self.agent_id}.{self.capability} "
            f"wechselt von {old_state.value} zu {self.metrics.state.value}"
        )

        if self.config.on_state_change:
            await self.config.on_state_change(
                f"{self.agent_id}.{self.capability}", old_state, self.metrics.state
            )
        self._metrics_collector.increment_counter(
            "circuit_breaker.state_changes",
            tags={
                "agent_id": self.agent_id,
                "capability": self.capability,
                "from_state": old_state.value,
                "to_state": self.metrics.state.value,
            },
        )

    async def _transition_to_half_open(self) -> None:
        """Übergang zu HALF_OPEN-Status."""
        old_state = self.metrics.state
        self.metrics.state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_changed_at = time.time()
        self.metrics.half_open_calls = 0
        self.metrics.half_open_successes = 0

        logger.info(
            f"Circuit Breaker für {self.agent_id}.{self.capability} "
            f"wechselt von {old_state.value} zu {self.metrics.state.value}"
        )

        if self.config.on_state_change:
            await self.config.on_state_change(
                f"{self.agent_id}.{self.capability}", old_state, self.metrics.state
            )
        self._metrics_collector.increment_counter(
            "circuit_breaker.state_changes",
            tags={
                "agent_id": self.agent_id,
                "capability": self.capability,
                "from_state": old_state.value,
                "to_state": self.metrics.state.value,
            },
        )

    async def _transition_to_closed(self) -> None:
        """Übergang zu CLOSED-Status."""
        old_state = self.metrics.state
        self.metrics.state = CircuitBreakerState.CLOSED
        self.metrics.state_changed_at = time.time()
        self.metrics.half_open_calls = 0
        self.metrics.half_open_successes = 0

        logger.info(
            f"Circuit Breaker für {self.agent_id}.{self.capability} "
            f"wechselt von {old_state.value} zu {self.metrics.state.value}"
        )

        if self.config.on_state_change:
            await self.config.on_state_change(
                f"{self.agent_id}.{self.capability}", old_state, self.metrics.state
            )
        self._metrics_collector.increment_counter(
            "circuit_breaker.state_changes",
            tags={
                "agent_id": self.agent_id,
                "capability": self.capability,
                "from_state": old_state.value,
                "to_state": self.metrics.state.value,
            },
        )

    def _time_until_retry(self) -> float:
        """Berechnet Zeit bis zum nächsten Retry-Versuch.

        Returns:
            Zeit in Sekunden
        """
        if self.metrics.state == CircuitBreakerState.OPEN:
            elapsed = time.time() - self.metrics.state_changed_at
            return max(0.0, self.recovery_timeout - elapsed)

        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Holt Circuit Breaker-Metriken.

        Returns:
            Circuit Breaker-Metriken
        """
        with self._lock:
            return {
                "capability": self.capability,
                "agent_id": self.agent_id,
                "state": self.metrics.state.value,
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "rejected_calls": self.metrics.rejected_calls,
                "failure_rate": self.metrics.get_failure_rate(),
                "recent_failure_rate": self.metrics.get_recent_failure_rate(),
                "time_until_retry": self._time_until_retry(),
                "half_open_calls": self.metrics.half_open_calls,
                "half_open_successes": self.metrics.half_open_successes,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "state_changed_at": self.metrics.state_changed_at,
            }


class CircuitBreakerManager:
    """Manager für alle Capability-spezifischen Circuit Breaker."""

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        """Initialisiert Circuit Breaker-Manager.

        Args:
            default_config: Standard-Konfiguration für neue Circuit Breaker
        """
        self.default_config = default_config or CircuitBreakerConfig()
        self._circuit_breakers: dict[str, CapabilityCircuitBreaker] = {}
        self._lock = threading.RLock()


        self._metrics_collector = MetricsCollector()

    def get_circuit_breaker(
        self, agent_id: str, capability: str, config: CircuitBreakerConfig | None = None
    ) -> CapabilityCircuitBreaker:
        """Holt oder erstellt Circuit Breaker für Agent-Capability.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            config: Optional spezifische Konfiguration

        Returns:
            Circuit Breaker für Agent-Capability
        """
        key = f"{agent_id}.{capability}"

        with self._lock:
            if key not in self._circuit_breakers:
                cb_config = config or self.default_config
                self._circuit_breakers[key] = CapabilityCircuitBreaker(
                    capability, agent_id, cb_config
                )

            return self._circuit_breakers[key]

    def get_all_circuit_breakers(self) -> dict[str, CapabilityCircuitBreaker]:
        """Holt alle Circuit Breaker.

        Returns:
            Dictionary aller Circuit Breaker
        """
        with self._lock:
            return self._circuit_breakers.copy()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt Zusammenfassung aller Circuit Breaker-Metriken.

        Returns:
            Metriken-Zusammenfassung
        """
        with self._lock:
            summary = {
                "total_circuit_breakers": len(self._circuit_breakers),
                "states": defaultdict(int),
                "total_calls": 0,
                "total_failures": 0,
                "total_rejections": 0,
                "circuit_breakers": {},
            }

            for key, cb in self._circuit_breakers.items():
                metrics = cb.get_metrics()
                summary["circuit_breakers"][key] = metrics
                summary["states"][metrics["state"]] += 1
                summary["total_calls"] += metrics["total_calls"]
                summary["total_failures"] += metrics["failed_calls"]
                summary["total_rejections"] += metrics["rejected_calls"]

            return summary

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check für alle Circuit Breaker durch.

        Returns:
            Health-Check-Ergebnis
        """
        with self._lock:
            health_status = {
                "healthy": True,
                "total_circuit_breakers": len(self._circuit_breakers),
                "open_circuit_breakers": 0,
                "half_open_circuit_breakers": 0,
                "issues": [],
            }

            for key, cb in self._circuit_breakers.items():
                if cb.metrics.state == CircuitBreakerState.OPEN:
                    health_status["open_circuit_breakers"] += 1
                    health_status["issues"].append(f"Circuit Breaker {key} ist offen")
                    health_status["healthy"] = False

                elif cb.metrics.state == CircuitBreakerState.HALF_OPEN:
                    health_status["half_open_circuit_breakers"] += 1

            return health_status



class CircuitBreakerOpenError(Exception):
    """Exception wenn Circuit Breaker offen ist."""
