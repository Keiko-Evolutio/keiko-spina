"""Basis-Klassen für Scheduling-Services.

Stellt gemeinsame AsyncIO-Patterns für start/stop Lifecycle-Management bereit
und eliminiert Code-Duplikation zwischen verschiedenen Scheduler-Implementierungen.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from kei_logging import get_logger

logger = get_logger(__name__)


class AsyncServiceBase(ABC):
    """Basis-Klasse für AsyncIO-Services mit standardisiertem Lifecycle-Management.

    Bietet einheitliche start/stop Patterns mit:
    - Graceful Shutdown-Handling
    - Task-Cancellation mit Timeout
    - Standardisiertes Error Handling
    - Idempotente start/stop Operationen
    """

    def __init__(self, *, service_name: str | None = None) -> None:
        """Initialisiert AsyncService.

        Args:
            service_name: Name des Services für Logging (Default: Klassenname)
        """
        self._service_name = service_name or self.__class__.__name__
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Prüft ob der Service läuft."""
        return self._running

    @property
    def service_name(self) -> str:
        """Name des Services."""
        return self._service_name

    async def start(self) -> None:
        """Startet den Service.

        Idempotent - mehrfache Aufrufe haben keinen Effekt.
        """
        if self._running:
            logger.debug(f"{self._service_name} bereits gestartet")
            return

        try:
            await self._pre_start()
            self._running = True
            self._task = asyncio.create_task(self._run_loop())
            logger.info(f"{self._service_name} gestartet")
        except Exception as e:
            self._running = False
            logger.exception(f"Fehler beim Starten von {self._service_name}: {e}")
            raise

    async def stop(self, *, timeout: float = 30.0) -> None:
        """Stoppt den Service graceful.

        Args:
            timeout: Timeout für Task-Cancellation in Sekunden
        """
        if not self._running:
            logger.debug(f"{self._service_name} bereits gestoppt")
            return

        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.CancelledError:
                # CancelledError ist erwartet beim Stoppen
                pass
            except TimeoutError:
                logger.warning(f"{self._service_name} Stop-Timeout nach {timeout}s")
            except Exception as e:
                logger.warning(f"Fehler beim Stoppen von {self._service_name}: {e}")
            finally:
                self._task = None

        try:
            await self._post_stop()
        except Exception as e:
            logger.warning(f"Fehler in post_stop von {self._service_name}: {e}")

        logger.info(f"{self._service_name} gestoppt")

    @abstractmethod
    async def _run_loop(self) -> None:
        """Haupt-Loop des Services.

        Muss von Subklassen implementiert werden.
        Sollte self._running prüfen für graceful shutdown.
        """

    async def _pre_start(self) -> None:
        """Hook für Initialisierung vor dem Start.

        Kann von Subklassen überschrieben werden.
        """

    async def _post_stop(self) -> None:
        """Hook für Cleanup nach dem Stop.

        Kann von Subklassen überschrieben werden.
        """


class PeriodicServiceBase(AsyncServiceBase):
    """Basis-Klasse für periodische Services mit konfigurierbarem Intervall.

    Erweitert AsyncServiceBase um:
    - Konfigurierbare Intervalle
    - Standardisierte Sleep-Loops mit Unterbrechbarkeit
    - Error Handling mit Retry-Logic
    """

    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        service_name: str | None = None
    ) -> None:
        """Initialisiert PeriodicService.

        Args:
            interval_seconds: Intervall zwischen Ausführungen in Sekunden
            service_name: Name des Services für Logging
        """
        super().__init__(service_name=service_name)
        self.interval_seconds = max(0.1, interval_seconds)

    async def _run_loop(self) -> None:
        """Periodische Ausführung mit Error Handling."""
        while self._running:
            try:
                await self._execute_cycle()
            except Exception as e:
                logger.warning(f"{self._service_name} Cycle-Fehler: {e}")

            # Unterbrechbarer Sleep für graceful shutdown
            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                logger.debug(f"{self._service_name} Sleep unterbrochen")
                break

    @abstractmethod
    async def _execute_cycle(self) -> None:
        """Führt einen Ausführungszyklus durch.

        Muss von Subklassen implementiert werden.
        """


__all__ = ["AsyncServiceBase", "PeriodicServiceBase"]
