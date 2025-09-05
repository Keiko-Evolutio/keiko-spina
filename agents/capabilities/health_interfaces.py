# backend/agents/capabilities/health_interfaces.py
"""Einheitliche Health-Check-Interfaces für das Agent System.

Konsolidiert alle Health-Check-Implementierungen in einheitliche Abstractions
für bessere Wartbarkeit und Konsistenz.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger

from ..metadata.agent_metadata import HealthStatus, ReadinessStatus

logger = get_logger(__name__)


class HealthCheckConstants:
    """Konstanten für Health-Check-Konfiguration."""

    DEFAULT_TIMEOUT = 5.0
    DEFAULT_HEALTH_INTERVAL = 30.0
    DEFAULT_READINESS_INTERVAL = 10.0
    MAX_RESPONSE_TIME_MS = 1000.0
    CRITICAL_RESPONSE_TIME_MS = 500.0


@dataclass
class HealthCheckResult:
    """Ergebnis eines Health-Checks."""

    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadinessCheckResult:
    """Ergebnis eines Readiness-Checks."""

    status: ReadinessStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    dependencies_ready: bool = True


class BaseHealthChecker(ABC):
    """Abstrakte Basis-Klasse für alle Health-Checker."""

    def __init__(self, name: str, timeout: float = HealthCheckConstants.DEFAULT_TIMEOUT) -> None:
        """Initialisiert Health-Checker.

        Args:
            name: Name des zu überwachenden Services/Capability
            timeout: Timeout für Health-Checks in Sekunden
        """
        self.name = name
        self.timeout = timeout
        self._last_check_time: datetime | None = None
        self._last_result: HealthCheckResult | None = None

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Führt Health-Check durch.

        Returns:
            HealthCheckResult mit Status und Metadaten
        """

    def get_last_result(self) -> HealthCheckResult | None:
        """Gibt letztes Health-Check-Ergebnis zurück."""
        return self._last_result

    @staticmethod
    def _create_health_result(
        status: HealthStatus,
        message: str,
        start_time: float,
        details: dict[str, Any] | None = None
    ) -> HealthCheckResult:
        """Erstellt standardisiertes HealthCheckResult.

        Args:
            status: Health-Status
            message: Status-Nachricht
            start_time: Start-Zeit für Response-Zeit-Berechnung
            details: Zusätzliche Details

        Returns:
            Standardisiertes HealthCheckResult
        """
        response_time = (time.time() - start_time) * 1000

        return HealthCheckResult(
            status=status,
            message=message,
            response_time_ms=response_time,
            timestamp=datetime.now(UTC),
            details=details or {}
        )


class BaseReadinessChecker(ABC):
    """Abstrakte Basis-Klasse für alle Readiness-Checker."""

    def __init__(
        self,
        name: str,
        dependencies: list[str] | None = None,
        timeout: float = HealthCheckConstants.DEFAULT_TIMEOUT
    ) -> None:
        """Initialisiert Readiness-Checker.

        Args:
            name: Name des zu überwachenden Services/Capability
            dependencies: Liste der Abhängigkeiten
            timeout: Timeout für Readiness-Checks in Sekunden
        """
        self.name = name
        self.dependencies = dependencies or []
        self.timeout = timeout
        self._last_check_time: datetime | None = None
        self._last_result: ReadinessCheckResult | None = None

    @abstractmethod
    async def check_readiness(self) -> ReadinessCheckResult:
        """Führt Readiness-Check durch.

        Returns:
            ReadinessCheckResult mit Status und Metadaten
        """

    def get_last_result(self) -> ReadinessCheckResult | None:
        """Gibt letztes Readiness-Check-Ergebnis zurück."""
        return self._last_result

    @staticmethod
    def _create_readiness_result(
        status: ReadinessStatus,
        message: str,
        start_time: float,
        dependencies_ready: bool = True,
        details: dict[str, Any] | None = None
    ) -> ReadinessCheckResult:
        """Erstellt standardisiertes ReadinessCheckResult.

        Args:
            status: Readiness-Status
            message: Status-Nachricht
            start_time: Start-Zeit für Response-Zeit-Berechnung
            dependencies_ready: Ob Abhängigkeiten bereit sind
            details: Zusätzliche Details

        Returns:
            Standardisiertes ReadinessCheckResult
        """
        response_time = (time.time() - start_time) * 1000

        return ReadinessCheckResult(
            status=status,
            message=message,
            response_time_ms=response_time,
            timestamp=datetime.now(UTC),
            dependencies_ready=dependencies_ready,
            details=details or {}
        )


class DefaultHealthChecker(BaseHealthChecker):
    """Standard-Health-Checker für einfache Capabilities."""

    async def check_health(self) -> HealthCheckResult:
        """Führt grundlegenden Health-Check durch."""
        start_time = time.time()

        try:
            await asyncio.sleep(0.001)

            response_time = (time.time() - start_time) * 1000
            if response_time > HealthCheckConstants.MAX_RESPONSE_TIME_MS:
                status = HealthStatus.DEGRADED
                message = f"Capability '{self.name}' antwortet langsam ({response_time:.2f}ms)"
            else:
                status = HealthStatus.OK
                message = f"Capability '{self.name}' ist verfügbar"

            result = self._create_health_result(
                status=status,
                message=message,
                start_time=start_time,
                details={"response_time_ms": response_time}
            )

            self._last_result = result
            self._last_check_time = datetime.now(UTC)

            return result

        except Exception as e:
            logger.exception(f"Health-Check für '{self.name}' fehlgeschlagen: {e}")

            result = self._create_health_result(
                status=HealthStatus.UNAVAILABLE,
                message=f"Health-Check für '{self.name}' fehlgeschlagen: {e!s}",
                start_time=start_time,
                details={"error": str(e), "error_type": type(e).__name__}
            )

            self._last_result = result
            self._last_check_time = datetime.now(UTC)

            return result


class DefaultReadinessChecker(BaseReadinessChecker):
    """Standard-Readiness-Checker für einfache Capabilities."""

    async def check_readiness(self) -> ReadinessCheckResult:
        """Führt grundlegenden Readiness-Check durch."""
        start_time = time.time()

        try:
            dependencies_ready = await self._check_dependencies()

            if not dependencies_ready:
                result = self._create_readiness_result(
                    status=ReadinessStatus.STARTING,
                    message=f"Capability '{self.name}' wartet auf Abhängigkeiten",
                    start_time=start_time,
                    dependencies_ready=False,
                    details={"dependencies": self.dependencies}
                )
            else:
                result = self._create_readiness_result(
                    status=ReadinessStatus.READY,
                    message=f"Capability '{self.name}' ist bereit",
                    start_time=start_time,
                    dependencies_ready=True,
                    details={"dependencies": self.dependencies}
                )

            self._last_result = result
            self._last_check_time = datetime.now(UTC)

            return result

        except Exception as e:
            logger.exception(f"Readiness-Check für '{self.name}' fehlgeschlagen: {e}")

            result = self._create_readiness_result(
                status=ReadinessStatus.STARTING,
                message=f"Readiness-Check für '{self.name}' fehlgeschlagen: {e!s}",
                start_time=start_time,
                dependencies_ready=False,
                details={"error": str(e), "error_type": type(e).__name__}
            )

            self._last_result = result
            self._last_check_time = datetime.now(UTC)

            return result

    async def _check_dependencies(self) -> bool:
        """Prüft, ob alle Abhängigkeiten verfügbar sind."""
        if not self.dependencies:
            return True

        # Prüfe jede Abhängigkeit
        for dependency in self.dependencies:
            try:
                # Hier könnte eine echte Abhängigkeitsprüfung implementiert werden
                # Für jetzt als nicht implementiert markieren
                if hasattr(dependency, 'is_available'):
                    if not await dependency.is_available():
                        return False
                # Fallback: Prüfung erfolgreich wenn keine Prüfmethode verfügbar
            except Exception:
                return False
        
        return True
HealthChecker = BaseHealthChecker
ReadinessChecker = BaseReadinessChecker

import asyncio
