# backend/agents/resilience/deadline_manager.py
"""Deadline Manager für Personal Assistant

Deadline-Management mit:
- Request-Deadlines und Timeouts
- Hierarchische Deadline-Vererbung
- Adaptive Timeout-Strategien
- Deadline-Violation-Handling
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class DeadlineType(Enum):
    """Typen von Deadlines."""

    HARD = "hard"  # Absolute Deadline - Verletzung führt zu Abbruch
    SOFT = "soft"  # Weiche Deadline - Verletzung wird protokolliert
    ADAPTIVE = "adaptive"  # Adaptive Deadline - passt sich an Kontext an


class TimeoutStrategy(Enum):
    """Timeout-Strategien."""

    FIXED = "fixed"  # Feste Timeout-Werte
    EXPONENTIAL = "exponential"  # Exponentieller Backoff
    LINEAR = "linear"  # Linearer Anstieg
    ADAPTIVE = "adaptive"  # Adaptive basierend auf Historie


@dataclass
class DeadlineConfig:
    """Konfiguration für Deadline Manager."""

    # Standard-Timeouts
    default_request_timeout: float = 30.0
    default_operation_timeout: float = 60.0
    default_session_timeout: float = 300.0

    # Timeout-Strategien
    timeout_strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE
    max_timeout_multiplier: float = 5.0
    min_timeout_seconds: float = 1.0
    max_timeout_seconds: float = 600.0

    # Adaptive Einstellungen
    enable_adaptive_timeouts: bool = True
    history_window_size: int = 100
    success_rate_threshold: float = 0.8

    # Deadline-Vererbung
    enable_deadline_inheritance: bool = True
    parent_deadline_factor: float = 0.8

    # Violation-Handling
    enable_deadline_violations: bool = True
    violation_retry_attempts: int = 2
    violation_backoff_factor: float = 1.5


@dataclass
class RequestDeadline:
    """Request-Deadline mit Metadaten."""

    deadline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    deadline_type: DeadlineType = DeadlineType.SOFT

    # Zeitstempel
    created_at: float = field(default_factory=time.time)
    deadline_at: float = 0.0
    timeout_seconds: float = 30.0

    # Hierarchie
    parent_deadline_id: str | None = None
    child_deadline_ids: list[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    is_violated: bool = False
    violation_time: float | None = None

    # Kontext
    operation: str = ""
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Callbacks
    violation_callback: Callable | None = None
    warning_callback: Callable | None = None


class DeadlineViolationError(Exception):
    """Exception für Deadline-Verletzungen."""

    def __init__(
        self,
        message: str,
        deadline: RequestDeadline,
        violation_time: float
    ):
        super().__init__(message)
        self.deadline = deadline
        self.violation_time = violation_time


class TimeoutHandler:
    """Handler für Timeout-Ereignisse."""

    def __init__(self, config: DeadlineConfig):
        """Initialisiert Timeout Handler.

        Args:
            config: Deadline-Konfiguration
        """
        self.config = config
        self._timeout_history: dict[str, list[float]] = {}
        self._success_rates: dict[str, float] = {}

    def calculate_timeout(
        self,
        operation: str,
        base_timeout: float | None = None,
        _context: dict[str, Any] | None = None
    ) -> float:
        """Berechnet adaptiven Timeout für Operation.

        Args:
            operation: Operation-Name
            base_timeout: Basis-Timeout
            _context: Zusätzlicher Kontext

        Returns:
            Berechneter Timeout in Sekunden
        """
        if not base_timeout:
            base_timeout = self.config.default_request_timeout

        if not self.config.enable_adaptive_timeouts:
            return base_timeout

        # Historische Daten abrufen
        history = self._timeout_history.get(operation, [])
        if not history:
            return base_timeout

        # Strategie anwenden
        if self.config.timeout_strategy == TimeoutStrategy.ADAPTIVE:
            return self._calculate_adaptive_timeout(operation, base_timeout, history)
        if self.config.timeout_strategy == TimeoutStrategy.EXPONENTIAL:
            return self._calculate_exponential_timeout(base_timeout, len(history))
        if self.config.timeout_strategy == TimeoutStrategy.LINEAR:
            return self._calculate_linear_timeout(base_timeout, len(history))
        return base_timeout

    def _calculate_adaptive_timeout(
        self,
        operation: str,
        base_timeout: float,
        history: list[float]
    ) -> float:
        """Berechnet adaptiven Timeout basierend auf Historie."""
        if len(history) < 5:
            return base_timeout

        # Durchschnittliche Ausführungszeit
        avg_time = sum(history[-20:]) / len(history[-20:])

        # Erfolgsrate berücksichtigen
        success_rate = self._success_rates.get(operation, 1.0)

        # Timeout anpassen
        if success_rate >= self.config.success_rate_threshold:
            # Hohe Erfolgsrate - konservativer Timeout
            timeout = avg_time * 1.5
        else:
            # Niedrige Erfolgsrate - großzügigerer Timeout
            timeout = avg_time * 2.5

        # Grenzen einhalten
        timeout = max(self.config.min_timeout_seconds, timeout)
        timeout = min(self.config.max_timeout_seconds, timeout)
        timeout = min(base_timeout * self.config.max_timeout_multiplier, timeout)

        return timeout

    def _calculate_exponential_timeout(
        self,
        base_timeout: float,
        attempt_count: int
    ) -> float:
        """Berechnet exponentiellen Timeout."""
        multiplier = 2 ** min(attempt_count, 5)  # Max 32x
        timeout = base_timeout * multiplier

        return min(self.config.max_timeout_seconds, timeout)

    def _calculate_linear_timeout(
        self,
        base_timeout: float,
        attempt_count: int
    ) -> float:
        """Berechnet linearen Timeout."""
        multiplier = 1 + (attempt_count * 0.5)  # +50% pro Versuch
        timeout = base_timeout * multiplier

        return min(self.config.max_timeout_seconds, timeout)

    def record_execution_time(
        self,
        operation: str,
        execution_time: float,
        success: bool
    ) -> None:
        """Zeichnet Ausführungszeit auf.

        Args:
            operation: Operation-Name
            execution_time: Ausführungszeit in Sekunden
            success: Ob Operation erfolgreich war
        """
        # Historie aktualisieren
        if operation not in self._timeout_history:
            self._timeout_history[operation] = []

        self._timeout_history[operation].append(execution_time)

        # Nur letzte N Einträge behalten
        if len(self._timeout_history[operation]) > self.config.history_window_size:
            self._timeout_history[operation] = (
                self._timeout_history[operation][-self.config.history_window_size:]
            )

        # Erfolgsrate aktualisieren
        self._update_success_rate(operation, success)

    def _update_success_rate(self, operation: str, success: bool) -> None:
        """Aktualisiert Erfolgsrate für Operation."""
        current_rate = self._success_rates.get(operation, 1.0)

        # Exponentieller gleitender Durchschnitt
        alpha = 0.1  # Lernrate
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate

        self._success_rates[operation] = new_rate


class DeadlineManager:
    """Enterprise Deadline Manager für Keiko Personal Assistant"""

    def __init__(self, config: DeadlineConfig):
        """Initialisiert Deadline Manager.

        Args:
            config: Deadline-Konfiguration
        """
        self.config = config
        self.timeout_handler = TimeoutHandler(config)

        # Deadline-Tracking
        self._active_deadlines: dict[str, RequestDeadline] = {}
        self._deadline_hierarchy: dict[str, list[str]] = {}

        # Monitoring
        self._violation_count = 0
        self._total_deadlines = 0
        self._cleanup_task: asyncio.Task | None = None

        # Cleanup-Task starten
        self._start_cleanup_task()

        logger.info("Deadline Manager initialisiert")

    def _start_cleanup_task(self) -> None:
        """Startet Cleanup-Task für abgelaufene Deadlines."""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_expired_deadlines()
                    await asyncio.sleep(60)  # Alle 60 Sekunden
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Deadline-Cleanup fehlgeschlagen: {e}")
                    await asyncio.sleep(60)

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    @trace_function("deadline.create_deadline")
    def create_deadline(
        self,
        request_id: str,
        timeout_seconds: float | None = None,
        deadline_type: DeadlineType = DeadlineType.SOFT,
        operation: str = "",
        agent_id: str | None = None,
        parent_deadline_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> RequestDeadline:
        """Erstellt neue Request-Deadline.

        Args:
            request_id: Request-ID
            timeout_seconds: Timeout in Sekunden
            deadline_type: Deadline-Typ
            operation: Operation-Name
            agent_id: Agent-ID
            parent_deadline_id: Parent-Deadline-ID
            metadata: Zusätzliche Metadaten

        Returns:
            Erstellte Deadline
        """
        # Timeout berechnen
        if not timeout_seconds:
            timeout_seconds = self.timeout_handler.calculate_timeout(
                operation, _context=metadata
            )

        # Parent-Deadline berücksichtigen
        if parent_deadline_id and self.config.enable_deadline_inheritance:
            parent_deadline = self._active_deadlines.get(parent_deadline_id)
            if parent_deadline:
                # Timeout an Parent-Deadline anpassen
                parent_remaining = parent_deadline.deadline_at - time.time()
                if parent_remaining > 0:
                    timeout_seconds = min(
                        timeout_seconds,
                        parent_remaining * self.config.parent_deadline_factor
                    )

        # Deadline erstellen
        deadline = RequestDeadline(
            request_id=request_id,
            deadline_type=deadline_type,
            deadline_at=time.time() + timeout_seconds,
            timeout_seconds=timeout_seconds,
            parent_deadline_id=parent_deadline_id,
            operation=operation,
            agent_id=agent_id,
            metadata=metadata or {}
        )

        # Deadline registrieren
        self._active_deadlines[deadline.deadline_id] = deadline
        self._total_deadlines += 1

        # Hierarchie aktualisieren
        if parent_deadline_id:
            if parent_deadline_id not in self._deadline_hierarchy:
                self._deadline_hierarchy[parent_deadline_id] = []
            self._deadline_hierarchy[parent_deadline_id].append(deadline.deadline_id)

            # Parent-Deadline aktualisieren
            if parent_deadline_id in self._active_deadlines:
                self._active_deadlines[parent_deadline_id].child_deadline_ids.append(
                    deadline.deadline_id
                )

        logger.debug(
            f"Deadline erstellt: {deadline.deadline_id} "
            f"(Timeout: {timeout_seconds}s, Typ: {deadline_type.value})"
        )

        return deadline

    @trace_function("deadline.check_deadline")
    def check_deadline(self, deadline_id: str) -> bool:
        """Prüft ob Deadline noch aktiv ist.

        Args:
            deadline_id: Deadline-ID

        Returns:
            True wenn Deadline noch aktiv
        """
        deadline = self._active_deadlines.get(deadline_id)
        if not deadline or not deadline.is_active:
            return False

        current_time = time.time()

        if current_time > deadline.deadline_at:
            # Deadline verletzt
            self._handle_deadline_violation(deadline, current_time)
            return False

        return True

    @trace_function("deadline.complete_deadline")
    def complete_deadline(
        self,
        deadline_id: str,
        success: bool = True,
        execution_time: float | None = None
    ) -> None:
        """Markiert Deadline als abgeschlossen.

        Args:
            deadline_id: Deadline-ID
            success: Ob Operation erfolgreich war
            execution_time: Tatsächliche Ausführungszeit
        """
        deadline = self._active_deadlines.get(deadline_id)
        if not deadline:
            return

        # Ausführungszeit berechnen
        if not execution_time:
            execution_time = time.time() - deadline.created_at

        # Historie aktualisieren
        self.timeout_handler.record_execution_time(
            deadline.operation,
            execution_time,
            success
        )

        # Deadline deaktivieren
        deadline.is_active = False

        # Child-Deadlines ebenfalls deaktivieren
        for child_id in deadline.child_deadline_ids:
            if child_id in self._active_deadlines:
                self._active_deadlines[child_id].is_active = False

        logger.debug(
            f"Deadline abgeschlossen: {deadline_id} "
            f"(Erfolg: {success}, Zeit: {execution_time:.2f}s)"
        )

    def _handle_deadline_violation(
        self,
        deadline: RequestDeadline,
        violation_time: float
    ) -> None:
        """Behandelt Deadline-Verletzung."""
        deadline.is_violated = True
        deadline.violation_time = violation_time
        deadline.is_active = False

        self._violation_count += 1

        logger.warning(
            f"Deadline verletzt: {deadline.deadline_id} "
            f"(Typ: {deadline.deadline_type.value}, "
            f"Verspätung: {violation_time - deadline.deadline_at:.2f}s)"
        )

        # Violation-Callback ausführen
        if deadline.violation_callback:
            try:
                if asyncio.iscoroutinefunction(deadline.violation_callback):
                    asyncio.create_task(deadline.violation_callback(deadline))
                else:
                    deadline.violation_callback(deadline)
            except Exception as e:
                logger.error(f"Violation-Callback fehlgeschlagen: {e}")

        # Hard-Deadline-Verletzung
        if deadline.deadline_type == DeadlineType.HARD:
            if self.config.enable_deadline_violations:
                raise DeadlineViolationError(
                    f"Hard deadline violated for {deadline.operation}",
                    deadline,
                    violation_time
                )

    async def _cleanup_expired_deadlines(self) -> None:
        """Bereinigt abgelaufene Deadlines."""
        current_time = time.time()
        expired_ids = []

        for deadline_id, deadline in self._active_deadlines.items():
            # Deadline ist abgelaufen und nicht mehr aktiv
            if (not deadline.is_active and
                current_time - deadline.created_at > 3600):  # 1 Stunde alt
                expired_ids.append(deadline_id)

        # Expired Deadlines entfernen
        for deadline_id in expired_ids:
            del self._active_deadlines[deadline_id]

            # Aus Hierarchie entfernen
            for parent_id, children in self._deadline_hierarchy.items():
                if deadline_id in children:
                    children.remove(deadline_id)

        if expired_ids:
            logger.debug(f"{len(expired_ids)} abgelaufene Deadlines bereinigt")

    @asynccontextmanager
    async def deadline_context(
        self,
        request_id: str,
        timeout_seconds: float | None = None,
        deadline_type: DeadlineType = DeadlineType.SOFT,
        operation: str = "",
        agent_id: str | None = None
    ):
        """Context Manager für Deadline-Management."""
        deadline = self.create_deadline(
            request_id=request_id,
            timeout_seconds=timeout_seconds,
            deadline_type=deadline_type,
            operation=operation,
            agent_id=agent_id
        )

        start_time = time.time()
        success = False

        try:
            yield deadline
            success = True
        except DeadlineViolationError:
            # Deadline-Verletzung weiterleiten
            raise
        except Exception as e:
            logger.error(f"Operation fehlgeschlagen in Deadline-Context: {e}")
            raise
        finally:
            execution_time = time.time() - start_time
            self.complete_deadline(deadline.deadline_id, success, execution_time)

    def get_deadline_statistics(self) -> dict[str, Any]:
        """Gibt Deadline-Statistiken zurück."""
        active_count = len([d for d in self._active_deadlines.values() if d.is_active])
        violated_count = len([d for d in self._active_deadlines.values() if d.is_violated])

        return {
            "total_deadlines": self._total_deadlines,
            "active_deadlines": active_count,
            "violated_deadlines": violated_count,
            "violation_rate": self._violation_count / max(self._total_deadlines, 1),
            "timeout_strategy": self.config.timeout_strategy.value,
            "adaptive_timeouts_enabled": self.config.enable_adaptive_timeouts,
            "deadline_inheritance_enabled": self.config.enable_deadline_inheritance
        }

    def get_active_deadlines(self) -> list[RequestDeadline]:
        """Gibt aktive Deadlines zurück."""
        return [d for d in self._active_deadlines.values() if d.is_active]

    async def shutdown(self) -> None:
        """Fährt Deadline Manager herunter."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Deadline Manager heruntergefahren")
