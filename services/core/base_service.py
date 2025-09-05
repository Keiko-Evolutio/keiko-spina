"""Basis-Klassen für Services mit konsolidierter Lifecycle-Funktionalität.

Konsolidiert duplizierte Service-Patterns aus allen Services.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from abc import ABC, abstractmethod
from typing import Any

from kei_logging import get_logger
from services.core.constants import (
    HEALTH_CHECK_INTERVAL_SECONDS,
    HEALTH_CHECK_TIMEOUT_SECONDS,
    SERVICE_STATUS_AVAILABLE,
    SERVICE_STATUS_ERROR,
    SERVICE_STATUS_INITIALIZING,
    SERVICE_STATUS_SHUTTING_DOWN,
    SERVICE_STATUS_UNAVAILABLE,
)

logger = get_logger(__name__)


class BaseService(ABC):
    """Abstrakte Basis-Klasse für alle Services.

    Konsolidiert gemeinsame Service-Funktionalität:
    - Lifecycle-Management (start/stop)
    - Health-Check-Funktionalität
    - Status-Management
    - Error-Handling-Patterns
    """

    def __init__(self, service_name: str) -> None:
        """Initialisiert den Basis-Service.

        Args:
            service_name: Name des Services für Logging
        """
        self.service_name = service_name
        self.running = False
        self.last_health_check: float | None = None
        self._task: asyncio.Task | None = None
        self._status = SERVICE_STATUS_UNAVAILABLE

    @property
    def status(self) -> str:
        """Aktueller Service-Status."""
        return self._status

    async def start(self) -> None:
        """Startet den Service."""
        if self.running:
            logger.warning(f"{self.service_name} läuft bereits")
            return

        logger.info(f"Starte {self.service_name}...")
        self._status = SERVICE_STATUS_INITIALIZING

        try:
            await self._initialize()
            self.running = True
            self._status = SERVICE_STATUS_AVAILABLE

            # Starte Background-Task falls implementiert
            if hasattr(self, "_run_background_task"):
                self._task = asyncio.create_task(self._run_background_task())

            logger.info(f"{self.service_name} erfolgreich gestartet")

        except Exception as e:
            self._status = SERVICE_STATUS_ERROR
            logger.exception(f"Fehler beim Starten von {self.service_name}: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt den Service."""
        if not self.running:
            logger.debug(f"{self.service_name} ist bereits gestoppt")
            return

        logger.info(f"Stoppe {self.service_name}...")
        self._status = SERVICE_STATUS_SHUTTING_DOWN

        try:
            # Stoppe Background-Task
            if self._task and not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

            await self._cleanup()
            self.running = False
            self._status = SERVICE_STATUS_UNAVAILABLE
            logger.info(f"{self.service_name} erfolgreich gestoppt")

        except Exception as e:
            self._status = SERVICE_STATUS_ERROR
            logger.exception(f"Fehler beim Stoppen von {self.service_name}: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check durch.

        Returns:
            Health-Status-Dictionary
        """
        try:
            start_time = time.time()
            is_healthy = await self._perform_health_check()
            response_time = (time.time() - start_time) * 1000

            self.last_health_check = time.time()

            return {
                "service": self.service_name,
                "status": SERVICE_STATUS_AVAILABLE if is_healthy else SERVICE_STATUS_ERROR,
                "running": self.running,
                "response_time_ms": response_time,
                "last_check": self.last_health_check,
            }

        except Exception as e:
            logger.warning(f"Health-Check fehlgeschlagen für {self.service_name}: {e}")
            return {
                "service": self.service_name,
                "status": SERVICE_STATUS_ERROR,
                "running": self.running,
                "error": str(e),
                "last_check": time.time(),
            }

    def get_status(self) -> dict[str, Any]:
        """Gibt detaillierten Service-Status zurück.

        Returns:
            Status-Dictionary
        """
        return {
            "service": self.service_name,
            "status": self._status,
            "running": self.running,
            "last_health_check": self.last_health_check,
        }

    @abstractmethod
    async def _initialize(self) -> None:
        """Service-spezifische Initialisierung.

        Muss von abgeleiteten Klassen implementiert werden.
        """

    @abstractmethod
    async def _cleanup(self) -> None:
        """Service-spezifische Bereinigung.

        Muss von abgeleiteten Klassen implementiert werden.
        """

    async def _perform_health_check(self) -> bool:
        """Service-spezifischer Health-Check.

        Standard-Implementierung prüft nur running-Status.
        Kann von abgeleiteten Klassen überschrieben werden.

        Returns:
            True wenn Service gesund ist
        """
        return self.running


class PeriodicService(BaseService):
    """Basis-Klasse für Services mit periodischen Tasks.

    Erweitert BaseService um periodische Ausführung.
    """

    def __init__(
        self,
        service_name: str,
        interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS
    ) -> None:
        """Initialisiert den periodischen Service.

        Args:
            service_name: Name des Services
            interval_seconds: Intervall zwischen periodischen Tasks
        """
        super().__init__(service_name)
        self.interval_seconds = interval_seconds

    async def _run_background_task(self) -> None:
        """Führt periodische Tasks aus."""
        logger.debug(f"Background-Task für {self.service_name} gestartet "
                    f"(Intervall: {self.interval_seconds}s)")

        while self.running:
            try:
                await self._perform_periodic_task()
                await asyncio.sleep(self.interval_seconds)

            except asyncio.CancelledError:
                logger.debug(f"Background-Task für {self.service_name} abgebrochen")
                break

            except Exception as e:
                logger.exception(f"Fehler in periodischem Task von {self.service_name}: {e}")
                await asyncio.sleep(self.interval_seconds)

    @abstractmethod
    async def _perform_periodic_task(self) -> None:
        """Periodischer Task.

        Muss von abgeleiteten Klassen implementiert werden.
        """


class MonitoringService(PeriodicService):
    """Basis-Klasse für Monitoring-Services.

    Erweitert PeriodicService um Monitoring-spezifische Funktionalität.
    """

    def __init__(
        self,
        service_name: str,
        interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS,
        timeout_seconds: float = HEALTH_CHECK_TIMEOUT_SECONDS,
        max_failures: int = 3
    ) -> None:
        """Initialisiert den Monitoring-Service.

        Args:
            service_name: Name des Services
            interval_seconds: Monitoring-Intervall
            timeout_seconds: Timeout für Monitoring-Checks
            max_failures: Maximale Anzahl Fehler vor Eskalation
        """
        super().__init__(service_name, interval_seconds)
        self.timeout_seconds = timeout_seconds
        self.max_failures = max_failures
        self.failure_counts: dict[str, int] = {}

    def _reset_failure_count(self, target: str) -> None:
        """Reset Failure-Counter für Target."""
        if target in self.failure_counts:
            del self.failure_counts[target]

    def _increment_failure_count(self, target: str) -> int:
        """Erhöht Failure-Counter für Target."""
        self.failure_counts[target] = self.failure_counts.get(target, 0) + 1
        return self.failure_counts[target]

    def _should_escalate(self, target: str) -> bool:
        """Prüft, ob Eskalation erforderlich ist."""
        return self.failure_counts.get(target, 0) >= self.max_failures

    async def _handle_monitoring_success(self, target: str) -> None:
        """Behandelt erfolgreiche Monitoring-Checks."""
        self._reset_failure_count(target)
        logger.debug(f"✅ Monitoring OK für {target}")

    async def _handle_monitoring_failure(self, target: str, error: str) -> None:
        """Behandelt fehlgeschlagene Monitoring-Checks."""
        failure_count = self._increment_failure_count(target)

        logger.warning(f"❌ Monitoring fehlgeschlagen für {target} "
                      f"({failure_count}/{self.max_failures}): {error}")

        if self._should_escalate(target):
            await self._escalate_failure(target)

    @abstractmethod
    async def _escalate_failure(self, target: str) -> None:
        """Eskaliert anhaltende Fehler.

        Muss von abgeleiteten Klassen implementiert werden.
        """
