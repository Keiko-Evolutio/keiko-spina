"""Circuit Breaker für Webhook‑Targets.

Implementiert ein konfigurierbares Circuit‑Breaker‑Pattern pro Target mit den
Zuständen CLOSED, OPEN und HALF_OPEN. Unterstützt zwei Modi:
- Verhältnis‑basiert (Fehlerrate über Fenstergröße)
- Schwellen‑basiert (aufeinanderfolgende Fehler/Erfolge)

Zustandswechsel werden geloggt und via Prometheus‑Gauge dargestellt.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from kei_logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    """Zustände eines Circuit Breakers."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitConfig:
    """Konfiguration für den Circuit Breaker.

    Attributes:
        window_size: Anzahl der Versuche im Betrachtungsfenster
        failure_ratio_to_open: Fehlerschwelle (Anteil), ab der geöffnet wird
        open_timeout_seconds: Dauer im OPEN Zustand, bevor HALF_OPEN probiert wird
        half_open_timeout_seconds: Dauer des HALF_OPEN Prüfzeitfensters
        half_open_max_calls: Anzahl erlaubter Probe‑Calls in HALF_OPEN
        use_consecutive_failures: Aktiviert Schwellen‑Modus statt Verhältnis‑Modus
        failure_threshold: Anzahl aufeinanderfolgender Fehler zum Öffnen (Schwellen‑Modus)
        recovery_timeout_seconds: Zeit von OPEN zu HALF_OPEN (Schwellen‑Modus)
        success_threshold: Erforderliche Erfolge in HALF_OPEN zum Schließen (Schwellen‑Modus)
    """

    window_size: int = 10
    failure_ratio_to_open: float = 0.5
    open_timeout_seconds: float = 60.0
    half_open_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1
    use_consecutive_failures: bool = False
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    success_threshold: int = 3


@dataclass
class CircuitWindow:
    """Gleitendes Fenster an Ergebnissen (True=Fehler, False=Erfolg)."""

    outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=10))

    def record(self, is_failure: bool) -> None:
        """Fügt ein Ergebnis zum Fenster hinzu."""
        self.outcomes.append(is_failure)

    def failure_ratio(self) -> float:
        """Berechnet Fehlerrate im aktuellen Fenster."""
        if not self.outcomes:
            return 0.0
        return sum(1 for x in self.outcomes if x) / float(len(self.outcomes))


@dataclass
class CircuitEntry:
    """Zustandsdaten je Target."""

    state: CircuitState = CircuitState.CLOSED
    window: CircuitWindow = field(default_factory=CircuitWindow)
    state_since: float = field(default_factory=time.monotonic)
    half_open_calls: int = 0
    consecutive_failures: int = 0
    half_open_successes: int = 0


class WebhookCircuitBreaker:
    """Circuit Breaker Verwaltung für alle Webhook‑Targets.

    Verwaltet Breaker‑Zustände pro Target und entscheidet, ob eine Anfrage
    derzeit erlaubt ist. Zustandswechsel werden geloggt; Metriken können über
    Prometheus‑Integration gesetzt werden.
    """

    def __init__(self, config: CircuitConfig | None = None, *, now_fn: Callable | None = None) -> None:
        # Konfiguration und Clock‑Injection für Tests
        self.config = config or CircuitConfig()
        self._now = now_fn or time.monotonic
        self._entries: dict[tuple[str, str | None], CircuitEntry] = {}

    def _get(self, target_id: str, tenant_id: str | None) -> CircuitEntry:
        key = (target_id, tenant_id)
        entry = self._entries.get(key)
        if entry is None:
            entry = CircuitEntry()
            # Fenstergröße aus Config anwenden
            entry.window.outcomes = deque(maxlen=self.config.window_size)
            self._entries[key] = entry
        return entry

    def _set_state(self, entry: CircuitEntry, new_state: CircuitState, *, target_id: str, tenant_id: str | None) -> None:
        if entry.state != new_state:
            logger.info(
                f"CircuitBreaker: Zustand {entry.state} -> {new_state} für target={target_id} tenant={tenant_id}"
            )
            entry.state = new_state
            entry.state_since = self._now()
            entry.half_open_calls = 0
            entry.half_open_successes = 0
            try:
                from .prometheus_metrics import set_circuit_state
                set_circuit_state(target_id=target_id, tenant_id=tenant_id, state=new_state.value)
            except Exception:
                pass

    def allow_request(self, *, target_id: str, tenant_id: str | None, policy: CircuitConfig | None = None) -> bool:
        """Prüft, ob eine Anfrage aktuell erlaubt ist.

        - CLOSED: immer erlaubt
        - OPEN: verweigert bis Timeout abgelaufen ist; dann HALF_OPEN
        - HALF_OPEN: erlaubt bis zu `half_open_max_calls` Probe‑Anfragen
        """
        entry = self._get(target_id, tenant_id)
        cfg = policy or self.config
        now = self._now()
        if entry.state == CircuitState.CLOSED:
            return True
        if entry.state == CircuitState.OPEN:
            timeout_s = cfg.recovery_timeout_seconds if cfg.use_consecutive_failures else cfg.open_timeout_seconds
            if now - entry.state_since >= timeout_s:
                self._set_state(entry, CircuitState.HALF_OPEN, target_id=target_id, tenant_id=tenant_id)
                return True
            return False
        # HALF_OPEN
        if entry.half_open_calls < max(1, cfg.half_open_max_calls):
            entry.half_open_calls += 1
            return True
        # Zu viele Probe‑Anfragen – bis Timeout keine weiteren zulassen
        if now - entry.state_since >= cfg.half_open_timeout_seconds:
            entry.half_open_calls = 0
            return True
        return False

    def on_success(self, *, target_id: str, tenant_id: str | None, policy: CircuitConfig | None = None) -> None:
        """Meldet einen erfolgreichen Call und passt Zustand an."""
        entry = self._get(target_id, tenant_id)
        cfg = policy or self.config
        entry.window.record(False)
        if cfg.use_consecutive_failures:
            entry.consecutive_failures = 0
            if entry.state == CircuitState.HALF_OPEN:
                entry.half_open_successes += 1
                if entry.half_open_successes >= max(1, cfg.success_threshold):
                    self._set_state(entry, CircuitState.CLOSED, target_id=target_id, tenant_id=tenant_id)
        elif entry.state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            self._set_state(entry, CircuitState.CLOSED, target_id=target_id, tenant_id=tenant_id)

    def on_failure(self, *, target_id: str, tenant_id: str | None, policy: CircuitConfig | None = None) -> None:
        """Meldet einen fehlgeschlagenen Call und passt Zustand an."""
        entry = self._get(target_id, tenant_id)
        cfg = policy or self.config
        entry.window.record(True)
        if cfg.use_consecutive_failures:
            if entry.state == CircuitState.CLOSED:
                entry.consecutive_failures += 1
                if entry.consecutive_failures >= max(1, cfg.failure_threshold):
                    self._set_state(entry, CircuitState.OPEN, target_id=target_id, tenant_id=tenant_id)
                    return
            elif entry.state == CircuitState.HALF_OPEN:
                entry.half_open_successes = 0
                self._set_state(entry, CircuitState.OPEN, target_id=target_id, tenant_id=tenant_id)
        else:
            frac = entry.window.failure_ratio()
            if entry.state == CircuitState.CLOSED and len(entry.window.outcomes) >= cfg.window_size:
                if frac > cfg.failure_ratio_to_open:
                    self._set_state(entry, CircuitState.OPEN, target_id=target_id, tenant_id=tenant_id)
                    return
            if entry.state == CircuitState.HALF_OPEN:
                self._set_state(entry, CircuitState.OPEN, target_id=target_id, tenant_id=tenant_id)


__all__ = [
    "CircuitConfig",
    "CircuitState",
    "WebhookCircuitBreaker",
]
