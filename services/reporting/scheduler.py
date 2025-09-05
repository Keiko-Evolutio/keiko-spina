"""Enterprise-grade Reporting-Scheduler für automatisierte KPI-Reports.

Robuster AsyncIO-basierter Scheduler mit:
- Konfigurierbare Intervalle und Parameter
- Graceful Shutdown-Handling
- Umfassendes Error Handling
- Dependency Injection für Testbarkeit
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import settings
from kei_logging import get_logger
from services.webhooks.alerting import get_alert_dispatcher

from .config import ReportingServiceConfig, get_reporting_config
from .exceptions import ReportDistributionError, ReportGenerationError
from .grafana_client import GrafanaClient
from .templates import ReportFormat, ReportTemplateManager

logger = get_logger(__name__)


class ReportingScheduler:
    """Enterprise-grade AsyncIO-basierter Scheduler für KPI-Reports.

    Bietet robuste Funktionalitäten für:
    - Periodische Report-Generierung
    - Konfigurierbare Intervalle
    - Graceful Shutdown
    - Dependency Injection
    - Umfassendes Error Handling
    """

    def __init__(
        self,
        interval_minutes: int | None = None,
        config: ReportingServiceConfig | None = None,
        grafana_client: GrafanaClient | None = None,
        template_manager: ReportTemplateManager | None = None
    ) -> None:
        """Initialisiert den Reporting-Scheduler.

        Args:
            interval_minutes: Intervall in Minuten (optional, verwendet Settings falls None)
            config: Service-Konfiguration (optional, verwendet Standard-Config falls None)
            grafana_client: Grafana-Client (optional, erstellt neuen falls None)
            template_manager: Template-Manager (optional, erstellt neuen falls None)
        """
        self.config = config or get_reporting_config()
        self.interval_minutes = (
            interval_minutes or
            settings.reporting_interval_minutes or
            self.config.scheduler.default_interval_minutes
        )
        self.grafana_client = grafana_client or GrafanaClient()
        self.template_manager = template_manager or ReportTemplateManager()

        self._task: asyncio.Task | None = None
        self._running = False

        logger.info(f"ReportingScheduler initialisiert (Intervall: {self.interval_minutes} min)")

    async def start(self) -> None:
        """Startet den Reporting-Scheduler.

        Raises:
            SchedulerError: Falls der Scheduler bereits läuft
        """
        if self._running:
            logger.warning("Scheduler läuft bereits, Start ignoriert")
            return

        logger.info("Starte Reporting-Scheduler")
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stoppt den Reporting-Scheduler graceful.

        Wartet auf den aktuellen Report-Zyklus und beendet dann den Scheduler.
        """
        if not self._running:
            logger.warning("Scheduler läuft nicht, Stop ignoriert")
            return

        logger.info("Stoppe Reporting-Scheduler")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self.config.scheduler.graceful_shutdown_timeout
                )
            except asyncio.CancelledError:
                logger.debug("Scheduler-Task erfolgreich abgebrochen")
            except TimeoutError:
                logger.warning("Scheduler-Stop Timeout erreicht")
            except Exception as e:
                logger.exception(f"Unerwarteter Fehler beim Scheduler-Stop: {e}")
            finally:
                self._task = None

    async def _run_loop(self) -> None:
        """Haupt-Loop des Schedulers.

        Führt periodisch Report-Generierung aus bis der Scheduler gestoppt wird.
        """
        logger.info("Scheduler-Loop gestartet")

        while self._running and settings.reporting_enabled:
            try:
                await self._generate_and_distribute_reports()
            except (ReportGenerationError, ReportDistributionError) as e:
                logger.exception(f"Report-Fehler: {e.message}", extra={"details": e.details})
            except Exception as e:
                logger.exception(f"Unerwarteter Scheduler-Fehler: {e}")

            # Sleep mit Unterbrechbarkeit für graceful shutdown
            try:
                sleep_seconds = self.interval_minutes * self.config.scheduler.seconds_per_minute
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                logger.debug("Scheduler-Sleep unterbrochen")
                break

        logger.info("Scheduler-Loop beendet")

    def _parse_recipients(self, recipients_string: str | None) -> list[str]:
        """Parst Empfänger-String zu Liste.

        Args:
            recipients_string: Komma-separierte Empfänger-Liste

        Returns:
            Liste der Empfänger-E-Mail-Adressen
        """
        if not recipients_string:
            return []
        return [email.strip() for email in recipients_string.split(",") if email.strip()]

    async def _generate_report(self) -> bytes:
        """Generiert einen Report über Grafana.

        Returns:
            Report-Daten als Bytes

        Raises:
            ReportGenerationError: Bei Generierungsfehlern
        """
        try:
            report_config = self.config.report

            png_data = await self.grafana_client.export_panel_png(
                dashboard_uid=report_config.default_dashboard_uid,
                panel_id=report_config.default_panel_id,
                params=report_config.default_export_params
            )

            logger.debug(f"Report erfolgreich generiert ({len(png_data)} bytes)")
            return png_data

        except Exception as e:
            raise ReportGenerationError(
                f"Report-Generierung fehlgeschlagen: {e!s}",
                report_type="png",
                dashboard_uid=self.config.report.default_dashboard_uid,
                panel_id=self.config.report.default_panel_id
            ) from e

    async def _distribute_report(self, recipients: list[str]) -> None:
        """Verteilt Report via E-Mail.

        Args:
            recipients: Liste der Empfänger

        Raises:
            ReportDistributionError: Bei Verteilungsfehlern
        """
        if not recipients:
            logger.info("Keine Empfänger konfiguriert, überspringe E-Mail-Versand")
            return

        try:
            report_config = self.config.report
            dispatcher = get_alert_dispatcher()

            await dispatcher.send_email(
                subject=report_config.default_subject,
                body=report_config.default_body,
                severity=report_config.default_severity
            )

            logger.info(f"Report erfolgreich an {len(recipients)} Empfänger versendet")

        except Exception as e:
            raise ReportDistributionError(
                f"Report-Verteilung fehlgeschlagen: {e!s}",
                recipients=recipients,
                distribution_method="email"
            ) from e

    async def _generate_and_distribute_reports(self) -> None:
        """Erzeugt Reports und verteilt sie.

        Hauptmethode für den Report-Workflow:
        1. Empfänger parsen
        2. Report generieren
        3. Report verteilen

        Raises:
            ReportGenerationError: Bei Generierungsfehlern
            ReportDistributionError: Bei Verteilungsfehlern
        """
        logger.info("Starte Report-Generierung und -Verteilung")

        # Empfänger aus Settings parsen
        recipients = self._parse_recipients(settings.reporting_default_recipients)

        # Report generieren (auch wenn keine Empfänger vorhanden)
        await self._generate_report()

        # Report verteilen falls Empfänger vorhanden
        await self._distribute_report(recipients)

        logger.info("Report-Workflow erfolgreich abgeschlossen")

    async def generate_template_report(
        self,
        template_id: str,
        recipients: list[str] | None = None
    ) -> dict[str, Any]:
        """Generiert Report basierend auf Template.

        Args:
            template_id: ID des zu verwendenden Templates
            recipients: Empfänger-Liste (optional, überschreibt Template-Standard)

        Returns:
            Dictionary mit Generierungs-Ergebnis

        Raises:
            ReportGenerationError: Bei Template- oder Generierungsfehlern
        """
        template = self.template_manager.get_template(template_id)
        if not template:
            raise ReportGenerationError(
                f"Template '{template_id}' nicht gefunden",
                report_type="template"
            )

        logger.info(f"Generiere Report mit Template '{template_id}'")

        try:
            results = {}

            # Für jedes Panel im Template
            for panel_id in template.panel_ids:
                panel_results = {}

                # Für jedes Format im Template
                for format_type in template.formats:
                    try:
                        if format_type == ReportFormat.PNG:
                            data = await self.grafana_client.export_panel_png(
                                template.dashboard_uid,
                                panel_id,
                                template.export_params
                            )
                            panel_results[format_type.value] = {
                                "size_bytes": len(data),
                                "success": True
                            }
                        elif format_type == ReportFormat.PDF:
                            data = await self.grafana_client.export_dashboard_pdf(
                                template.dashboard_uid,
                                template.export_params
                            )
                            panel_results[format_type.value] = {
                                "size_bytes": len(data),
                                "success": True
                            }
                        elif format_type == ReportFormat.JSON:
                            data = await self.grafana_client.export_panel_json(
                                template.dashboard_uid,
                                panel_id,
                                template.export_params
                            )
                            panel_results[format_type.value] = {
                                "data": data,
                                "success": True
                            }
                        elif format_type == ReportFormat.CSV:
                            data = await self.grafana_client.export_panel_csv(
                                template.dashboard_uid,
                                panel_id,
                                template.export_params
                            )
                            panel_results[format_type.value] = {
                                "data": data,
                                "success": True
                            }
                        else:
                            panel_results[format_type.value] = {
                                "error": f"Format {format_type.value} nicht unterstützt",
                                "success": False
                            }
                    except Exception as e:
                        panel_results[format_type.value] = {
                            "error": str(e),
                            "success": False
                        }

                results[f"panel_{panel_id}"] = panel_results

            # E-Mail-Versand falls konfiguriert
            if template.schedule_enabled:
                email_recipients = recipients or self._parse_recipients(settings.reporting_default_recipients)
                if email_recipients:
                    try:
                        dispatcher = get_alert_dispatcher()
                        await dispatcher.send_email(
                            subject=template.email_subject,
                            body=template.email_body,
                            severity=template.email_severity
                        )
                        results["email_sent"] = True
                    except Exception as e:
                        results["email_error"] = str(e)
                        results["email_sent"] = False

            logger.info(f"Template-Report '{template_id}' erfolgreich generiert")
            return {
                "success": True,
                "template_id": template_id,
                "template_name": template.name,
                "results": results,
                "panels_processed": len(template.panel_ids),
                "formats_processed": len(template.formats)
            }

        except Exception as e:
            raise ReportGenerationError(
                f"Template-Report-Generierung fehlgeschlagen: {e!s}",
                report_type="template",
                dashboard_uid=template.dashboard_uid
            ) from e

    def get_available_templates(self) -> list[dict[str, Any]]:
        """Gibt verfügbare Templates zurück.

        Returns:
            Liste der verfügbaren Templates mit Metadaten
        """
        templates = self.template_manager.list_templates()
        return [
            {
                "template_id": t.template_id,
                "name": t.name,
                "description": t.description,
                "report_type": t.report_type.value,
                "dashboard_uid": t.dashboard_uid,
                "panel_count": len(t.panel_ids),
                "formats": [f.value for f in t.formats],
                "schedule_enabled": t.schedule_enabled,
                "interval_minutes": t.interval_minutes,
                "tags": t.tags
            }
            for t in templates
        ]


__all__ = ["ReportingScheduler"]
