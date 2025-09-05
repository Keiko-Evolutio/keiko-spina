# backend/agents/monitoring/health_monitor.py
"""Health Monitor für das Agent-Framework.

Health-Monitoring mit:
- Kontinuierliche Health-Checks
- Dependency-Monitoring
- Automatische Recovery-Mechanismen
- Health-Status-Aggregation
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..logging_utils import StructuredLogger

logger = StructuredLogger("health_monitor")


class HealthStatus(Enum):
    """Health-Status-Werte."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthConfig:
    """Konfiguration für Health Monitor."""

    # Check-Intervalle
    default_check_interval: float = 30.0
    critical_check_interval: float = 10.0

    # Timeouts
    default_check_timeout: float = 5.0
    critical_check_timeout: float = 2.0

    # Retry-Konfiguration
    max_check_retries: int = 3
    retry_delay: float = 1.0

    # Thresholds
    degraded_threshold: float = 0.8
    unhealthy_threshold: float = 0.5

    # Recovery
    enable_auto_recovery: bool = True
    recovery_check_interval: float = 60.0

    # Alerting
    enable_health_alerts: bool = True
    alert_on_status_change: bool = True


@dataclass
class HealthCheckResult:
    """Ergebnis eines Health-Checks."""

    check_name: str
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)

    # Details
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Performance
    check_duration: float = 0.0

    # Kontext
    component: str | None = None
    agent_id: str | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)


class HealthCheck(ABC):
    """Abstrakte Basis-Klasse für Health-Checks."""

    def __init__(
        self,
        name: str,
        interval: float = 30.0,
        timeout: float = 5.0,
        critical: bool = False
    ):
        """Initialisiert Health-Check.

        Args:
            name: Name des Health-Checks
            interval: Check-Intervall in Sekunden
            timeout: Timeout für Check
            critical: Ob Check kritisch ist
        """
        self.name = name
        self.interval = interval
        self.timeout = timeout
        self.critical = critical

        # Status-Tracking
        self.last_check_time: float | None = None
        self.last_result: HealthCheckResult | None = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0

    @abstractmethod
    async def execute_check(self) -> HealthCheckResult:
        """Führt Health-Check aus.

        Returns:
            Health-Check-Ergebnis
        """

    async def run_check(self) -> HealthCheckResult:
        """Führt Health-Check mit Timeout und Error-Handling aus."""
        start_time = time.time()

        try:
            # Check mit Timeout ausführen
            result = await asyncio.wait_for(
                self.execute_check(),
                timeout=self.timeout
            )

            result.check_duration = time.time() - start_time
            result.timestamp = start_time

            # Erfolgs-Counter aktualisieren
            if result.status == HealthStatus.HEALTHY:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0

            self.last_result = result
            self.last_check_time = time.time()

            return result

        except TimeoutError:
            result = HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health-Check Timeout nach {self.timeout}s",
                check_duration=time.time() - start_time
            )

            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_result = result
            self.last_check_time = time.time()

            return result

        except Exception as e:
            result = HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health-Check Fehler: {e!s}",
                check_duration=time.time() - start_time,
                details={"error_type": type(e).__name__}
            )

            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_result = result
            self.last_check_time = time.time()

            return result


class ComponentHealthCheck(HealthCheck):
    """Health-Check für Framework-Komponenten."""

    def __init__(
        self,
        name: str,
        component: Any,
        health_method: str = "health_check",
        **kwargs
    ):
        """Initialisiert Component Health-Check.

        Args:
            name: Name des Checks
            component: Zu prüfende Komponente
            health_method: Name der Health-Check-Methode
            **kwargs: Weitere HealthCheck-Parameter
        """
        super().__init__(name, **kwargs)
        self.component = component
        self.health_method = health_method

    async def execute_check(self) -> HealthCheckResult:
        """Führt Component Health-Check aus."""
        if not hasattr(self.component, self.health_method):
            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Komponente hat keine {self.health_method}-Methode"
            )

        try:
            health_method = getattr(self.component, self.health_method)

            # Async oder sync Methode aufrufen
            if asyncio.iscoroutinefunction(health_method):
                health_data = await health_method()
            else:
                health_data = health_method()

            # Health-Status interpretieren
            if isinstance(health_data, dict):
                is_healthy = health_data.get("healthy", False)
                message = health_data.get("message", "")
                details = health_data.get("details", {})
            else:
                is_healthy = bool(health_data)
                message = "Component health check"
                details = {}

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

            return HealthCheckResult(
                check_name=self.name,
                status=status,
                message=message,
                details=details,
                component=getattr(self.component, "__class__", {}).get("__name__", "Unknown")
            )

        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Component Health-Check fehlgeschlagen: {e!s}",
                details={"error": str(e), "error_type": type(e).__name__}
            )


class CustomHealthCheck(HealthCheck):
    """Benutzerdefinierter Health-Check."""

    def __init__(
        self,
        name: str,
        check_function: Callable[[], Awaitable[HealthCheckResult]],
        **kwargs
    ):
        """Initialisiert Custom Health-Check.

        Args:
            name: Name des Checks
            check_function: Async Funktion für Health-Check
            **kwargs: Weitere HealthCheck-Parameter
        """
        super().__init__(name, **kwargs)
        self.check_function = check_function

    async def execute_check(self) -> HealthCheckResult:
        """Führt Custom Health-Check aus."""
        return await self.check_function()


class HealthMonitor:
    """Health Monitor für das Agent-Framework."""

    def __init__(self, config: HealthConfig):
        """Initialisiert Health Monitor.

        Args:
            config: Health-Monitor-Konfiguration
        """
        self.config = config

        # Health-Checks
        self._health_checks: dict[str, HealthCheck] = {}
        self._check_tasks: dict[str, asyncio.Task] = {}

        # Status-Tracking
        self._overall_status = HealthStatus.UNKNOWN
        self._status_history: list[dict[str, Any]] = []

        # Recovery-Mechanismen
        self._recovery_handlers: dict[str, Callable] = {}

        # Service-basierte Checks
        self._service_checks: dict[str, dict[str, Any]] = {}
        self._last_results: dict[str, dict[str, Any]] = {}

        # Monitoring-Task
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Metrics-Integration
        self._metrics_collector = None

        logger.info("Health Monitor initialisiert")

    def register_health_check(self, health_check: HealthCheck) -> None:
        """Registriert Health-Check.

        Args:
            health_check: Health-Check-Instanz
        """
        self._health_checks[health_check.name] = health_check
        logger.info(f"Health-Check registriert: {health_check.name}")

    def register_component_health_check(
        self,
        name: str,
        component: Any,
        interval: float = None,
        critical: bool = False
    ) -> None:
        """Registriert Component Health-Check.

        Args:
            name: Name des Checks
            component: Komponente
            interval: Check-Intervall
            critical: Ob Check kritisch ist
        """
        if interval is None:
            interval = self.config.critical_check_interval if critical else self.config.default_check_interval

        health_check = ComponentHealthCheck(
            name=name,
            component=component,
            interval=interval,
            timeout=self.config.critical_check_timeout if critical else self.config.default_check_timeout,
            critical=critical
        )

        self.register_health_check(health_check)

    def register_custom_health_check(
        self,
        name: str,
        check_function: Callable[[], Awaitable[HealthCheckResult]],
        interval: float = None,
        critical: bool = False
    ) -> None:
        """Registriert Custom Health-Check.

        Args:
            name: Name des Checks
            check_function: Check-Funktion
            interval: Check-Intervall
            critical: Ob Check kritisch ist
        """
        if interval is None:
            interval = self.config.critical_check_interval if critical else self.config.default_check_interval

        health_check = CustomHealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=self.config.critical_check_timeout if critical else self.config.default_check_timeout,
            critical=critical
        )

        self.register_health_check(health_check)

    def register_recovery_handler(
        self,
        check_name: str,
        recovery_handler: Callable[[], Awaitable[bool]]
    ) -> None:
        """Registriert Recovery-Handler für Health-Check.

        Args:
            check_name: Name des Health-Checks
            recovery_handler: Recovery-Funktion
        """
        self._recovery_handlers[check_name] = recovery_handler
        logger.info(f"Recovery-Handler registriert für: {check_name}")

    async def start_monitoring(self) -> None:
        """Startet Health-Monitoring."""
        if self._running:
            logger.warning("Health-Monitoring läuft bereits")
            return

        self._running = True

        # Check-Tasks für alle Health-Checks starten
        for check_name, health_check in self._health_checks.items():
            task = asyncio.create_task(self._run_health_check_loop(health_check))
            self._check_tasks[check_name] = task

        # Monitor-Task starten
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Health-Monitoring gestartet")

    async def stop_monitoring(self) -> None:
        """Stoppt Health-Monitoring."""
        if not self._running:
            return

        self._running = False

        # Check-Tasks stoppen
        for task in self._check_tasks.values():
            task.cancel()

        # Auf Task-Beendigung warten
        if self._check_tasks:
            await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)

        # Monitor-Task stoppen
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self._check_tasks.clear()
        logger.info("Health-Monitoring gestoppt")

    async def _run_health_check_loop(self, health_check: HealthCheck) -> None:
        """Führt Health-Check-Loop aus."""
        while self._running:
            try:
                # Health-Check ausführen
                result = await health_check.run_check()

                logger.debug(
                    f"Health-Check ausgeführt: {health_check.name} -> {result.status.value}",
                    extra_data={
                        "check_name": health_check.name,
                        "status": result.status.value,
                        "duration": result.check_duration,
                        "message": result.message
                    }
                )

                # Recovery versuchen bei Fehlern
                if (result.status != HealthStatus.HEALTHY and
                    self.config.enable_auto_recovery and
                    health_check.name in self._recovery_handlers):

                    await self._attempt_recovery(health_check.name)

                # Nächsten Check planen
                await asyncio.sleep(health_check.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Fehler in Health-Check-Loop: {health_check.name}",
                    error=e
                )
                await asyncio.sleep(health_check.interval)

    async def _monitor_loop(self) -> None:
        """Haupt-Monitor-Loop."""
        while self._running:
            try:
                # Overall-Status berechnen
                await self._update_overall_status()

                # Status-Historie aktualisieren
                self._update_status_history()

                # Nächste Iteration
                await asyncio.sleep(60.0)  # Alle 60 Sekunden (reduziert CPU-Last)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Fehler in Monitor-Loop", error=e)
                await asyncio.sleep(10.0)

    async def _update_overall_status(self) -> None:
        """Aktualisiert Overall-Health-Status."""
        if not self._health_checks:
            self._overall_status = HealthStatus.UNKNOWN
            return

        # Status aller Checks sammeln
        statuses = []
        critical_unhealthy = False

        for health_check in self._health_checks.values():
            if health_check.last_result:
                statuses.append(health_check.last_result.status)

                # Kritische Checks prüfen
                if health_check.critical and health_check.last_result.status == HealthStatus.UNHEALTHY:
                    critical_unhealthy = True

        if not statuses:
            self._overall_status = HealthStatus.UNKNOWN
            return

        # Overall-Status bestimmen
        if critical_unhealthy:
            new_status = HealthStatus.UNHEALTHY
        elif HealthStatus.UNHEALTHY in statuses:
            unhealthy_ratio = statuses.count(HealthStatus.UNHEALTHY) / len(statuses)
            if unhealthy_ratio >= self.config.unhealthy_threshold:
                new_status = HealthStatus.UNHEALTHY
            elif unhealthy_ratio >= self.config.degraded_threshold:
                new_status = HealthStatus.DEGRADED
            else:
                new_status = HealthStatus.HEALTHY
        elif HealthStatus.DEGRADED in statuses:
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY

        # Status-Änderung protokollieren
        if new_status != self._overall_status:
            logger.info(
                f"Overall Health-Status geändert: {self._overall_status.value} -> {new_status.value}",
                extra_data={
                    "old_status": self._overall_status.value,
                    "new_status": new_status.value,
                    "check_count": len(statuses),
                    "unhealthy_count": statuses.count(HealthStatus.UNHEALTHY),
                    "degraded_count": statuses.count(HealthStatus.DEGRADED)
                }
            )

        self._overall_status = new_status

    def _update_status_history(self) -> None:
        """Aktualisiert Status-Historie."""
        status_entry = {
            "timestamp": time.time(),
            "overall_status": self._overall_status.value,
            "check_results": {
                name: {
                    "status": check.last_result.status.value if check.last_result else "unknown",
                    "message": check.last_result.message if check.last_result else "",
                    "consecutive_failures": check.consecutive_failures
                }
                for name, check in self._health_checks.items()
            }
        }

        self._status_history.append(status_entry)

        # Nur letzte 1000 Einträge behalten
        if len(self._status_history) > 1000:
            self._status_history = self._status_history[-1000:]

    async def _attempt_recovery(self, check_name: str) -> None:
        """Versucht Recovery für fehlgeschlagenen Check."""
        if check_name not in self._recovery_handlers:
            return

        try:
            logger.info(f"Versuche Recovery für: {check_name}")

            recovery_handler = self._recovery_handlers[check_name]
            success = await recovery_handler()

            if success:
                logger.info(f"Recovery erfolgreich für: {check_name}")
            else:
                logger.warning(f"Recovery fehlgeschlagen für: {check_name}")

        except Exception as e:
            logger.error(f"Recovery-Fehler für {check_name}", error=e)

    def get_health_status(self) -> dict[str, Any]:
        """Gibt aktuellen Health-Status zurück."""
        return {
            "overall_status": self._overall_status.value,
            "timestamp": time.time(),
            "checks": {
                name: {
                    "status": check.last_result.status.value if check.last_result else "unknown",
                    "message": check.last_result.message if check.last_result else "",
                    "last_check": check.last_check_time,
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes,
                    "critical": check.critical
                }
                for name, check in self._health_checks.items()
            },
            "monitoring_active": self._running
        }

    def get_health_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Gibt Health-Status-Historie zurück."""
        return self._status_history[-limit:] if self._status_history else []

    def register_service_check(
        self,
        service_name: str,
        check_func: Callable[[], bool],
        timeout_seconds: float = 30.0,
        interval_seconds: float = 60.0
    ) -> None:
        """Registriert Service-basierten Health-Check.

        Args:
            service_name: Service-Name
            check_func: Check-Funktion
            timeout_seconds: Timeout in Sekunden
            interval_seconds: Check-Intervall in Sekunden
        """
        self._service_checks[service_name] = {
            "check_func": check_func,
            "timeout_seconds": timeout_seconds,
            "interval_seconds": interval_seconds,
            "enabled": True
        }

        # Als Health-Check registrieren
        health_check = ServiceHealthCheck(
            name=service_name,
            check_func=check_func,
            timeout=timeout_seconds,
            interval=interval_seconds
        )

        self.register_health_check(health_check)
        logger.info(f"Service-Check registriert: {service_name}")

    async def check_service_health(self, service_name: str) -> dict[str, Any]:
        """Führt Service-Health-Check aus.

        Args:
            service_name: Service-Name

        Returns:
            Health-Check-Ergebnis
        """
        if service_name not in self._service_checks:
            return {
                "service_name": service_name,
                "is_healthy": False,
                "status": "not_registered",
                "details": {"error": "Service nicht registriert"},
                "response_time_ms": 0.0
            }

        check_config = self._service_checks[service_name]
        start_time = time.time()

        try:
            # Check mit Timeout ausführen
            is_healthy = await asyncio.wait_for(
                asyncio.to_thread(check_config["check_func"]),
                timeout=check_config["timeout_seconds"]
            )

            response_time_ms = (time.time() - start_time) * 1000

            result = {
                "service_name": service_name,
                "is_healthy": is_healthy,
                "status": "healthy" if is_healthy else "unhealthy",
                "details": {"response_time_ms": response_time_ms},
                "response_time_ms": response_time_ms,
                "last_check": time.time()
            }

            # Ergebnis cachen
            self._last_results[service_name] = result

            # Metriken aktualisieren (wenn verfügbar)
            if self._metrics_collector:
                self._metrics_collector.set_gauge(
                    "health_check_status",
                    1.0 if is_healthy else 0.0,
                    {"service": service_name}
                )

                self._metrics_collector.observe_histogram(
                    "health_check_duration_seconds",
                    response_time_ms / 1000.0,
                    {"service": service_name, "status": result["status"]}
                )

            return result

        except TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            result = {
                "service_name": service_name,
                "is_healthy": False,
                "status": "timeout",
                "details": {"error": "Health-Check Timeout", "timeout_seconds": check_config["timeout_seconds"]},
                "response_time_ms": response_time_ms,
                "last_check": time.time()
            }

            self._last_results[service_name] = result
            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            result = {
                "service_name": service_name,
                "is_healthy": False,
                "status": "error",
                "details": {"error": str(e)},
                "response_time_ms": response_time_ms,
                "last_check": time.time()
            }

            self._last_results[service_name] = result
            logger.error(f"Health-Check Fehler für {service_name}", error=e)
            return result

    def set_metrics_collector(self, metrics_collector) -> None:
        """Setzt Metrics Collector für Health-Metriken.

        Args:
            metrics_collector: Metrics Collector Instanz
        """
        self._metrics_collector = metrics_collector
        logger.info("Metrics Collector für Health Monitor gesetzt")

    def get_service_health_summary(self) -> list[dict[str, Any]]:
        """Gibt Service-Health-Zusammenfassung zurück."""
        return list(self._last_results.values())


class ServiceHealthCheck(HealthCheck):
    """Service-basierter Health-Check."""

    def __init__(self, name: str, check_func: Callable[[], bool], timeout: float = 30.0, interval: float = 60.0):
        super().__init__(name, timeout, interval)
        self.check_func = check_func

    async def execute_check(self) -> HealthCheckResult:
        """Führt Service-Check aus."""
        try:
            is_healthy = await asyncio.to_thread(self.check_func)

            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message="Service-Check erfolgreich" if is_healthy else "Service-Check fehlgeschlagen"
            )

        except Exception as e:
            return HealthCheckResult(
                check_name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service-Check Fehler: {e!s}"
            )
