"""Health Checker Implementation.
Implementiert umfassende Health-Checks für alle System-Komponenten.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil

from kei_logging import get_logger

from .interfaces import HealthStatus, IHealthChecker, IMetricsCollector

logger = get_logger(__name__)


@dataclass
class HealthCheckConfig:
    """Konfiguration für einen Health-Check."""
    name: str
    check_func: Callable[[], bool]
    timeout_seconds: float = 5.0
    interval_seconds: float = 30.0
    critical: bool = False
    enabled: bool = True


class HealthChecker(IHealthChecker):
    """Comprehensive Health Checker Implementation.
    Überwacht System-Health und Service-Verfügbarkeit.
    """

    def __init__(self, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector
        self._health_checks: dict[str, HealthCheckConfig] = {}
        self._last_results: dict[str, HealthStatus] = {}
        self._running = False
        self._check_task: asyncio.Task | None = None

        # Standard Health-Checks registrieren
        self._register_default_health_checks()

        logger.info("Health checker initialized")

    def _register_default_health_checks(self) -> None:
        """Registriert Standard-Health-Checks."""
        # System-Health-Checks
        self.register_health_check("system_cpu", self._check_cpu_usage, critical=True)
        self.register_health_check("system_memory", self._check_memory_usage, critical=True)
        self.register_health_check("system_disk", self._check_disk_usage, critical=False)

        # Application-Health-Checks
        self.register_health_check("app_startup", self._check_app_startup, critical=True)
        self.register_health_check("metrics_collector", self._check_metrics_collector, critical=False)

        logger.debug("Default health checks registered")

    def register_health_check(self, service_name: str, check_func: Callable[[], bool],
                            timeout_seconds: float = 5.0, interval_seconds: float = 30.0,
                            critical: bool = False) -> None:
        """Registriert Health-Check-Funktion."""
        config = HealthCheckConfig(
            name=service_name,
            check_func=check_func,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
            critical=critical
        )

        self._health_checks[service_name] = config
        logger.info(f"Registered health check for {service_name} (critical: {critical})")

    async def check_health(self, service_name: str) -> HealthStatus:
        """Führt Health-Check für Service durch."""
        if service_name not in self._health_checks:
            return HealthStatus(
                service_name=service_name,
                is_healthy=False,
                status="unknown",
                details={"error": "Health check not registered"},
                last_check=datetime.utcnow()
            )

        config = self._health_checks[service_name]
        start_time = time.time()

        try:
            # Health-Check mit Timeout ausführen
            is_healthy = await asyncio.wait_for(
                asyncio.to_thread(config.check_func),
                timeout=config.timeout_seconds
            )

            response_time_ms = (time.time() - start_time) * 1000

            status = HealthStatus(
                service_name=service_name,
                is_healthy=is_healthy,
                status="healthy" if is_healthy else "unhealthy",
                details={"response_time_ms": response_time_ms},
                last_check=datetime.utcnow(),
                response_time_ms=response_time_ms
            )

        except TimeoutError:
            response_time_ms = config.timeout_seconds * 1000
            status = HealthStatus(
                service_name=service_name,
                is_healthy=False,
                status="timeout",
                details={"error": f"Health check timed out after {config.timeout_seconds}s"},
                last_check=datetime.utcnow(),
                response_time_ms=response_time_ms
            )

        except (ConnectionError, OSError) as e:
            response_time_ms = (time.time() - start_time) * 1000
            status = HealthStatus(
                service_name=service_name,
                is_healthy=False,
                status="connection_error",
                details={"error": f"Connection error: {e}"},
                last_check=datetime.utcnow(),
                response_time_ms=response_time_ms
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            status = HealthStatus(
                service_name=service_name,
                is_healthy=False,
                status="error",
                details={"error": f"Unexpected error: {e}"},
                last_check=datetime.utcnow(),
                response_time_ms=response_time_ms
            )

        # Ergebnis cachen
        self._last_results[service_name] = status

        # Metriken aktualisieren
        self.metrics_collector.set_gauge(
            "health_check_status",
            1.0 if status.is_healthy else 0.0,
            labels={"service": service_name}
        )

        self.metrics_collector.observe_histogram(
            "health_check_duration_seconds",
            response_time_ms / 1000.0,
            labels={"service": service_name, "status": status.status}
        )

        logger.debug(f"Health check for {service_name}: {status.status} ({response_time_ms:.1f}ms)")

        return status

    async def check_all_services(self) -> list[HealthStatus]:
        """Führt Health-Check für alle Services durch."""
        tasks = []

        for service_name in self._health_checks.keys():
            task = asyncio.create_task(self.check_health(service_name))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_statuses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = list(self._health_checks.keys())[i]
                health_statuses.append(HealthStatus(
                    service_name=service_name,
                    is_healthy=False,
                    status="error",
                    details={"error": str(result)},
                    last_check=datetime.utcnow()
                ))
            else:
                health_statuses.append(result)

        # Gesamtstatus-Metriken
        total_services = len(health_statuses)
        healthy_services = sum(1 for status in health_statuses if status.is_healthy)

        self.metrics_collector.set_gauge("health_check_total_services", total_services)
        self.metrics_collector.set_gauge("health_check_healthy_services", healthy_services)
        self.metrics_collector.set_gauge("health_check_overall_ratio", healthy_services / total_services if total_services > 0 else 0.0)

        return health_statuses

    async def start_periodic_checks(self) -> None:
        """Startet periodische Health-Checks."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._periodic_check_loop())
        logger.info("Started periodic health checks")

    async def stop_periodic_checks(self) -> None:
        """Stoppt periodische Health-Checks."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped periodic health checks")

    async def _periodic_check_loop(self) -> None:
        """Periodische Health-Check-Schleife."""
        while self._running:
            try:
                await self.check_all_services()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    def get_overall_health(self) -> dict[str, Any]:
        """Gibt Gesamtstatus aller Services zurück."""
        if not self._last_results:
            return {"status": "unknown", "services": []}

        all_healthy = all(status.is_healthy for status in self._last_results.values())
        critical_services = [
            name for name, config in self._health_checks.items()
            if config.critical
        ]

        critical_healthy = all(
            self._last_results.get(name, HealthStatus("", False, "", {}, datetime.utcnow())).is_healthy
            for name in critical_services
        )

        overall_status = "healthy" if critical_healthy else "unhealthy"

        return {
            "status": overall_status,
            "all_services_healthy": all_healthy,
            "critical_services_healthy": critical_healthy,
            "total_services": len(self._last_results),
            "healthy_services": sum(1 for status in self._last_results.values() if status.is_healthy),
            "services": [
                {
                    "name": status.service_name,
                    "healthy": status.is_healthy,
                    "status": status.status,
                    "last_check": status.last_check.isoformat(),
                    "response_time_ms": status.response_time_ms
                }
                for status in self._last_results.values()
            ]
        }

    # Standard Health-Check-Funktionen
    def _check_cpu_usage(self) -> bool:
        """Prüft CPU-Nutzung."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90.0  # Threshold: 90%
        except Exception:
            return False

    def _check_memory_usage(self) -> bool:
        """Prüft Speicher-Nutzung."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < 90.0  # Threshold: 90%
        except Exception:
            return False

    def _check_disk_usage(self) -> bool:
        """Prüft Festplatten-Nutzung."""
        try:
            disk = psutil.disk_usage("/")
            return disk.percent < 90.0  # Threshold: 90%
        except Exception:
            return False

    def _check_app_startup(self) -> bool:
        """Prüft ob Anwendung gestartet ist."""
        # Einfacher Check - kann erweitert werden
        return True

    def _check_metrics_collector(self) -> bool:
        """Prüft Metrics Collector."""
        try:
            metrics = self.metrics_collector.get_metrics()
            return len(metrics) > 0  # Collector ist verfügbar und hat Daten
        except Exception:
            return False
