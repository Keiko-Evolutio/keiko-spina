# backend/agents/registry/mixins/exception_handling_mixin.py
"""Mixin für einheitliche Exception-Behandlung.

Konsolidiert die wiederholten Exception-Handling-Patterns.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

from kei_logging import get_logger

from ..exceptions import (
    AgentNotFoundError,
    RegistrationError,
    RegistryUnavailableError,
)
from ..utils.constants import ErrorConstants

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ExceptionHandlingMixin:
    """Mixin für einheitliche Exception-Behandlung.

    Konsolidiert die wiederholten try-catch-Patterns und
    bietet einheitliche Fehlerbehandlung.
    """

    def __init__(self, *args, **kwargs):
        """Initialisiert das Mixin."""
        super().__init__(*args, **kwargs)

        # Fehler-Tracking
        self._error_history: list[dict[str, Any]] = []
        self._max_error_history = 100

    def handle_registry_error(
        self,
        operation: str,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Behandelt Registry-Fehler einheitlich.

        Args:
            operation: Name der Operation
            error: Aufgetretener Fehler
            context: Zusätzlicher Kontext
        """
        error_info = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Füge zu Fehler-History hinzu
        self._error_history.append(error_info)
        if len(self._error_history) > self._max_error_history:
            self._error_history.pop(0)

        # Setze letzten Fehler (falls verfügbar)
        if hasattr(self, "_set_error"):
            self._set_error(str(error))

        # Logge Fehler
        logger.error(
            f"Registry-Fehler in {operation}: {error}",
            extra={"context": context, "error_type": type(error).__name__}
        )

    def safe_execute(
        self,
        operation: str,
        func: Callable[[], Any],
        default_return: Any = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Führt eine Operation sicher aus.

        Args:
            operation: Name der Operation
            func: Auszuführende Funktion
            default_return: Rückgabewert bei Fehler
            context: Zusätzlicher Kontext

        Returns:
            Ergebnis der Funktion oder default_return
        """
        try:
            return func()
        except Exception as e:
            self.handle_registry_error(operation, e, context)
            return default_return

    async def safe_execute_async(
        self,
        operation: str,
        func: Callable[[], Any],
        default_return: Any = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Führt eine asynchrone Operation sicher aus.

        Args:
            operation: Name der Operation
            func: Auszuführende Funktion
            default_return: Rückgabewert bei Fehler
            context: Zusätzlicher Kontext

        Returns:
            Ergebnis der Funktion oder default_return
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        except Exception as e:
            self.handle_registry_error(operation, e, context)
            return default_return

    def with_retry(
        self,
        max_attempts: int = ErrorConstants.MAX_RETRY_ATTEMPTS,
        delay: float = ErrorConstants.RETRY_DELAY_SECONDS,
        backoff_factor: float = ErrorConstants.EXPONENTIAL_BACKOFF_FACTOR,
    ):
        """Decorator für Retry-Logik.

        Args:
            max_attempts: Maximale Versuche
            delay: Verzögerung zwischen Versuchen
            backoff_factor: Exponential-Backoff-Faktor
        """
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Versuch {attempt + 1}/{max_attempts} fehlgeschlagen: {e}. "
                                f"Wiederholung in {current_delay}s"
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(f"Alle {max_attempts} Versuche fehlgeschlagen")

                # Alle Versuche fehlgeschlagen
                if last_exception:
                    raise last_exception

                # Fallback: Sollte nie erreicht werden, aber für Konsistenz
                # (z.B. wenn max_attempts = 0)
                return None

            return wrapper
        return decorator

    def validate_agent_registration(
        self,
        agent_id: str,
        agent: Any,
        overwrite: bool = False,
    ) -> None:
        """Validiert Agent-Registrierung.

        Args:
            agent_id: Agent-ID
            agent: Agent-Instanz
            overwrite: Überschreiben erlauben

        Raises:
            RegistrationError: Bei Validierungsfehlern
        """
        if not agent_id or not agent_id.strip():
            raise RegistrationError(
                "Agent-ID darf nicht leer sein",
                error_code=ErrorConstants.VALIDATION_ERROR,
            )

        if agent is None:
            raise RegistrationError(
                "Agent-Instanz darf nicht None sein",
                error_code=ErrorConstants.VALIDATION_ERROR,
            )

        # Prüfe auf doppelte Registrierung
        if hasattr(self, "_agents") and not overwrite:
            if agent_id in self._agents:
                raise RegistrationError(
                    f"Agent {agent_id} bereits registriert",
                    error_code=ErrorConstants.DUPLICATE_REGISTRATION,
                    details={"agent_id": agent_id, "overwrite": overwrite},
                )

    def validate_agent_lookup(self, agent_id: str) -> None:
        """Validiert Agent-Lookup.

        Args:
            agent_id: Agent-ID

        Raises:
            AgentNotFoundError: Wenn Agent nicht gefunden
        """
        if not agent_id or not agent_id.strip():
            raise AgentNotFoundError(
                "Agent-ID darf nicht leer sein",
                error_code=ErrorConstants.VALIDATION_ERROR,
            )

        if hasattr(self, "_agents"):
            if agent_id not in self._agents:
                raise AgentNotFoundError(
                    f"Agent {agent_id} nicht gefunden",
                    error_code=ErrorConstants.AGENT_NOT_FOUND,
                    details={"agent_id": agent_id},
                )

    def validate_registry_state(self) -> None:
        """Validiert Registry-Zustand.

        Raises:
            RegistryUnavailableError: Wenn Registry nicht verfügbar
        """
        if hasattr(self, "_initialized") and not self._initialized:
            raise RegistryUnavailableError(
                "Registry ist nicht initialisiert",
                error_code=ErrorConstants.REGISTRY_UNAVAILABLE,
            )

    def get_error_history(self) -> list[dict[str, Any]]:
        """Gibt Fehler-History zurück.

        Returns:
            Liste der Fehler
        """
        return self._error_history.copy()

    def clear_error_history(self) -> None:
        """Löscht Fehler-History."""
        self._error_history.clear()

    def get_recent_errors(self, count: int = 10) -> list[dict[str, Any]]:
        """Gibt die letzten Fehler zurück.

        Args:
            count: Anzahl der Fehler

        Returns:
            Liste der letzten Fehler
        """
        return self._error_history[-count:] if self._error_history else []

    def has_recent_errors(self, minutes: int = 5) -> bool:
        """Prüft ob es kürzliche Fehler gab.

        Args:
            minutes: Zeitfenster in Minuten

        Returns:
            True wenn kürzliche Fehler vorhanden
        """
        if not self._error_history:
            return False

        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(minutes=minutes)

        return any(
            error.get("timestamp", datetime.min) > cutoff
            for error in self._error_history
        )
