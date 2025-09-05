"""Reporting-Service-Implementation mit Service-Interface-Integration.

Enterprise-grade Service-Implementation die das LifecycleService-Interface
implementiert und in die bestehende Service-Architektur integriert.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import settings
from kei_logging import get_logger
from services.interfaces import FeatureService, ServiceStatus

from .config import ReportingServiceConfig, get_reporting_config
from .exceptions import ReportingServiceError, SchedulerError
from .grafana_client import GrafanaClient
from .scheduler import ReportingScheduler

logger = get_logger(__name__)


class ReportingService(FeatureService):
    """Enterprise-grade Reporting-Service.

    Feature-Service für automatisierte Report-Generierung und -Verteilung.
    Implementiert das LifecycleService-Interface für Integration in die
    Service-Architektur.

    Funktionalitäten:
    - Automatisierte KPI-Report-Generierung
    - Grafana-Dashboard-Exports
    - E-Mail-basierte Report-Verteilung
    - Konfigurierbare Scheduler-Intervalle
    - Health-Monitoring und Status-Management
    """

    def __init__(
        self,
        config: ReportingServiceConfig | None = None,
        grafana_client: GrafanaClient | None = None,
        scheduler: ReportingScheduler | None = None
    ) -> None:
        """Initialisiert den Reporting-Service.

        Args:
            config: Service-Konfiguration (optional)
            grafana_client: Grafana-Client (optional, wird erstellt falls None)
            scheduler: Report-Scheduler (optional, wird erstellt falls None)
        """
        self.config = config or get_reporting_config()
        self.grafana_client = grafana_client or GrafanaClient(config=self.config.grafana)
        self.scheduler = scheduler or ReportingScheduler(
            config=self.config,
            grafana_client=self.grafana_client
        )

        self._status = ServiceStatus.UNINITIALIZED
        self._initialized = False

        logger.info(f"ReportingService erstellt (Version: {self.config.version})")

    async def initialize(self) -> None:
        """Initialisiert den Reporting-Service.

        Startet den Scheduler falls Reporting aktiviert ist und
        setzt den Service-Status auf RUNNING.

        Raises:
            ReportingServiceError: Bei Initialisierungsfehlern
        """
        if self._initialized:
            logger.warning("ReportingService bereits initialisiert")
            return

        try:
            logger.info("Initialisiere ReportingService")
            self._status = ServiceStatus.INITIALIZING

            # Scheduler starten falls Reporting aktiviert
            if settings.reporting_enabled:
                await self.scheduler.start()
                logger.info("Report-Scheduler gestartet")
            else:
                logger.info("Reporting deaktiviert, Scheduler nicht gestartet")

            self._status = ServiceStatus.RUNNING
            self._initialized = True

            logger.info("ReportingService erfolgreich initialisiert")

        except Exception as e:
            self._status = ServiceStatus.ERROR
            raise ReportingServiceError(
                f"ReportingService-Initialisierung fehlgeschlagen: {e!s}"
            ) from e

    async def shutdown(self) -> None:
        """Führt einen geordneten Shutdown des Reporting-Services durch.

        Stoppt den Scheduler und gibt alle Ressourcen frei.
        """
        if not self._initialized:
            logger.warning("ReportingService nicht initialisiert, Shutdown ignoriert")
            return

        try:
            logger.info("Stoppe ReportingService")
            self._status = ServiceStatus.STOPPING

            # Scheduler stoppen
            await self.scheduler.stop()
            logger.info("Report-Scheduler gestoppt")

            self._status = ServiceStatus.STOPPED
            self._initialized = False

            logger.info("ReportingService erfolgreich gestoppt")

        except Exception as e:
            self._status = ServiceStatus.ERROR
            logger.exception(f"Fehler beim ReportingService-Shutdown: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check durch und liefert detaillierte Status-Informationen.

        Returns:
            Dictionary mit Health-Status, Service-Metriken und Konfiguration
        """
        try:
            # Basis Health-Check
            health_data = await super().health_check()

            # Service-spezifische Health-Informationen
            health_data.update({
                "service_name": self.config.service_name,
                "version": self.config.version,
                "status": self._status.value,
                "initialized": self._initialized,
                "reporting_enabled": settings.reporting_enabled,
                "scheduler": {
                    "running": self.scheduler._running,
                    "interval_minutes": self.scheduler.interval_minutes,
                    "has_task": self.scheduler._task is not None
                },
                "grafana": {
                    "base_url": self.grafana_client.base_url,
                    "has_token": bool(self.grafana_client.api_token),
                    "timeout_config": {
                        "panel": self.grafana_client.config.panel_timeout_seconds,
                        "dashboard": self.grafana_client.config.dashboard_timeout_seconds
                    }
                },
                "configuration": {
                    "default_dashboard": self.config.report.default_dashboard_uid,
                    "default_panel": self.config.report.default_panel_id,
                    "log_level": self.config.log_level,
                    "debug_logging": self.config.enable_debug_logging
                }
            })

            # Health-Status basierend auf Service-Zustand
            if self._status == ServiceStatus.RUNNING:
                health_data["status"] = "healthy"
            elif self._status == ServiceStatus.ERROR:
                health_data["status"] = "unhealthy"
            else:
                health_data["status"] = "degraded"

            return health_data

        except Exception as e:
            logger.exception(f"Health-Check fehlgeschlagen: {e}")
            return {
                "status": "unhealthy",
                "service": self.__class__.__name__,
                "error": str(e)
            }

    def get_status(self) -> ServiceStatus:
        """Liefert aktuellen Service-Status.

        Returns:
            Aktueller ServiceStatus
        """
        return self._status

    async def generate_manual_report(self) -> dict[str, Any]:
        """Generiert einen manuellen Report.

        Führt eine einmalige Report-Generierung und -Verteilung durch,
        unabhängig vom Scheduler.

        Returns:
            Dictionary mit Report-Generierungs-Ergebnis

        Raises:
            ReportingServiceError: Bei Report-Generierungsfehlern
        """
        if not self._initialized:
            raise ReportingServiceError("Service nicht initialisiert")

        try:
            logger.info("Starte manuelle Report-Generierung")

            await self.scheduler._generate_and_distribute_reports()

            result = {
                "success": True,
                "message": "Report erfolgreich generiert und verteilt",
                "timestamp": asyncio.get_event_loop().time()
            }

            logger.info("Manuelle Report-Generierung erfolgreich")
            return result

        except Exception as e:
            error_msg = f"Manuelle Report-Generierung fehlgeschlagen: {e!s}"
            logger.exception(error_msg)
            raise ReportingServiceError(error_msg) from e

    async def start_scheduler(self) -> dict[str, Any]:
        """Startet den Report-Scheduler manuell.

        Returns:
            Dictionary mit Start-Ergebnis

        Raises:
            SchedulerError: Bei Scheduler-Start-Fehlern
        """
        try:
            await self.scheduler.start()
            return {
                "success": True,
                "message": "Scheduler erfolgreich gestartet",
                "running": self.scheduler._running
            }
        except Exception as e:
            raise SchedulerError(f"Scheduler-Start fehlgeschlagen: {e!s}") from e

    async def stop_scheduler(self) -> dict[str, Any]:
        """Stoppt den Report-Scheduler manuell.

        Returns:
            Dictionary mit Stop-Ergebnis
        """
        try:
            await self.scheduler.stop()
            return {
                "success": True,
                "message": "Scheduler erfolgreich gestoppt",
                "running": self.scheduler._running
            }
        except Exception as e:
            logger.exception(f"Scheduler-Stop fehlgeschlagen: {e}")
            return {
                "success": False,
                "message": f"Scheduler-Stop fehlgeschlagen: {e!s}",
                "running": self.scheduler._running
            }

    def get_service_info(self) -> dict[str, Any]:
        """Gibt detaillierte Service-Informationen zurück.

        Returns:
            Dictionary mit Service-Metadaten und -Status
        """
        return {
            "name": self.config.service_name,
            "version": self.config.version,
            "type": "FeatureService",
            "status": self._status.value,
            "initialized": self._initialized,
            "capabilities": [
                "report_generation",
                "grafana_integration",
                "email_distribution",
                "scheduled_reports",
                "manual_reports"
            ],
            "configuration": {
                "reporting_enabled": settings.reporting_enabled,
                "interval_minutes": self.scheduler.interval_minutes,
                "grafana_url": self.grafana_client.base_url,
                "dashboard_uid": self.config.report.default_dashboard_uid
            }
        }


__all__ = ["ReportingService"]
