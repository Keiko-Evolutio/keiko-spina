"""Abstrakte Basis-Klasse für Agent-Callback-Handler.

Stellt gemeinsame Funktionalitäten für Error-Handling, Lifecycle-Management
und Logging für verschiedene Tracing-Backends bereit.
"""

from __future__ import annotations

import abc
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


class BaseCallbackHandler(abc.ABC):
    """Abstrakte Basis-Klasse für Agent-Callback-Handler.

    Definiert das gemeinsame Interface und Error-Handling für
    verschiedene Tracing-Backend-Implementierungen.
    """

    def __init__(self, agent_id: str) -> None:
        """Initialisiert den Callback-Handler.

        Args:
            agent_id: Agent-ID für Monitoring.
        """
        self.agent_id = agent_id
        self._is_active = False

    @abc.abstractmethod
    async def on_start(
        self,
        instruction: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Startet Monitoring eines Agentenlaufs.

        Args:
            instruction: Eingabeaufforderung/Taskbeschreibung.
            metadata: Optionale Zusatzinformationen.
        """

    @abc.abstractmethod
    async def on_end(
        self,
        response: str | dict[str, Any] | Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """Beendet Monitoring und aktualisiert Metriken.

        Args:
            response: Agentenantwort.
            prompt_tokens: Eingabetoken (sofern verfügbar).
            completion_tokens: Ausgabetoken (sofern verfügbar).
        """

    @abc.abstractmethod
    async def on_error(self, error: Exception) -> None:
        """Meldet Fehlerereignis.

        Args:
            error: Ausgelöste Exception.
        """

    async def _handle_connection_error(
        self,
        exc: Exception,
        operation: str
    ) -> None:
        """Behandelt Verbindungsfehler einheitlich.

        Args:
            exc: Aufgetretene Exception.
            operation: Name der Operation für Logging.
        """
        if isinstance(exc, (ConnectionError, TimeoutError)):
            logger.warning("Verbindungsfehler in %s: %s", operation, exc)
        else:
            logger.debug("%s ignoriert: %s", operation, exc)

    def _cleanup_state(self) -> None:
        """Bereinigt internen Zustand nach Abschluss."""
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """Gibt zurück, ob der Handler aktiv ist."""
        return self._is_active


__all__ = ["BaseCallbackHandler"]
