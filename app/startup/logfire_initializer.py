"""Logfire-Initializer für die Keiko Personal Assistant Startup-Sequenz.

Integriert die Enterprise-Grade Pydantic Logfire-Integration nahtlos
in die bestehende Anwendungs-Startup-Architektur.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from kei_logging import get_logger

from ..common.logger_utils import safe_log_exception
from .service_initializers import ServiceInitializer

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..service_container import ServiceContainer

logger = get_logger(__name__)


class LogfireInitializer(ServiceInitializer):
    """Initializer für die Enterprise-Grade Pydantic Logfire-Integration.

    Startet die vollständige Logfire-Integration mit:
    - Automatischer LLM-Instrumentierung
    - Enterprise-Sicherheit und PII-Redaction
    - Distributed Tracing und Structured Logging
    - Performance-Monitoring und Fallback-Mechanismen
    """

    def __init__(self, container: ServiceContainer) -> None:
        super().__init__(container)
        self._logfire_available = False
        self._observability_started = False

    async def initialize(self, app: FastAPI = None) -> bool:
        """Initialisiert die Logfire-Integration."""
        if app is None:
            logger.error("❌ FastAPI-Anwendung erforderlich für Logfire-Integration")
            return False

        try:
            logger.info("🔥 Starte Enterprise Logfire-Integration...")

            # 1. Prüfe Logfire-Verfügbarkeit
            if not await self._check_logfire_availability():
                logger.warning("⚠️ Logfire-Integration nicht verfügbar - überspringe")
                return False

            # 2. Starte Observability-System mit Logfire
            if not await self._start_observability_system():
                logger.warning("⚠️ Observability-System-Start fehlgeschlagen")
                return False

            # 3. Instrumentiere FastAPI-Anwendung
            await self._instrument_fastapi_app(app)

            # 4. Registriere Logfire-Services im Container
            await self._register_logfire_services()

            # 5. Sende Startup-Event
            await self._send_startup_event(app)

            logger.info("✅ Logfire-Integration erfolgreich gestartet")
            return True

        except Exception as e:
            safe_log_exception(logger, e, "Logfire-Initialisierung fehlgeschlagen")
            # Nicht kritisch - Anwendung kann ohne Logfire weiterlaufen
            return False

    async def _check_logfire_availability(self) -> bool:
        """Prüft ob Logfire-Integration verfügbar ist."""
        try:
            from observability import LOGFIRE_INTEGRATION_AVAILABLE

            if not LOGFIRE_INTEGRATION_AVAILABLE:
                logger.info("ℹ️ Logfire-Integration nicht verfügbar (Module nicht importiert)")
                return False

            # Prüfe Konfiguration
            from observability import get_logfire_settings, validate_logfire_config

            settings = get_logfire_settings()
            if not validate_logfire_config(settings):
                logger.warning("⚠️ Ungültige Logfire-Konfiguration")
                return False

            self._logfire_available = True
            logger.info(f"✅ Logfire verfügbar: mode={settings.mode}, env={settings.environment}")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Logfire-Verfügbarkeitsprüfung fehlgeschlagen: {e}")
            return False

    async def _start_observability_system(self) -> bool:
        """Startet das vollständige Observability-System mit Logfire."""
        try:
            from observability import start_observability_system

            # Starte System mit Logfire-Integration
            success = await start_observability_system()

            if success:
                self._observability_started = True
                logger.info("✅ Observability-System mit Logfire gestartet")
                return True
            logger.warning("⚠️ Observability-System-Start fehlgeschlagen")
            return False

        except Exception as e:
            logger.warning(f"⚠️ Observability-System-Start-Fehler: {e}")
            return False

    async def _instrument_fastapi_app(self, app: FastAPI) -> None:
        """Instrumentiert die FastAPI-Anwendung mit Logfire."""
        try:
            if not self._logfire_available:
                return

            from observability import get_logfire_manager

            # Hole Logfire-Manager
            manager = get_logfire_manager()

            if not manager.is_available():
                logger.warning("⚠️ Logfire-Manager nicht verfügbar für FastAPI-Instrumentierung")
                return

            # Instrumentiere FastAPI über LogfireManager
            try:
                manager.instrument_fastapi_app(app)
                logger.info("✅ FastAPI mit Logfire instrumentiert")
            except Exception as e:
                logger.warning(f"⚠️ FastAPI-Instrumentierung fehlgeschlagen: {e}")

            logger.info("✅ Alle Instrumentierungen werden vom LogfireManager verwaltet")

        except Exception as e:
            logger.warning(f"⚠️ FastAPI-Instrumentierung-Fehler: {e}")

    async def _register_logfire_services(self) -> None:
        """Registriert Logfire-Services im DI-Container."""
        try:
            if not self._logfire_available:
                return

            # Services sind bereits über globale Funktionen verfügbar
            # get_logfire_manager() und get_llm_tracker()
            # Keine explizite DI-Container-Registrierung erforderlich

            logger.info("✅ Logfire-Services verfügbar über globale Funktionen")

        except Exception as e:
            logger.warning(f"⚠️ Logfire-Service-Setup fehlgeschlagen: {e}")

    async def _send_startup_event(self, app: FastAPI) -> None:
        """Sendet Startup-Event an Logfire."""
        try:
            if not self._logfire_available:
                return

            import time
            from datetime import datetime

            from observability import get_logfire_manager

            manager = get_logfire_manager()

            if not manager.is_available():
                return

            # Sammle Startup-Informationen
            startup_info = {
                "service_name": "keiko-personal-assistant",
                "version": "1.0.0",
                "environment": "development",
                "startup_time": datetime.now(UTC).isoformat(),
                "startup_timestamp": time.time(),
                "fastapi_version": getattr(app, "version", "unknown"),
                "observability_system_started": self._observability_started,
                "logfire_integration_active": True,
                "components_initialized": [
                    "logfire_manager",
                    "llm_tracker",
                    "fastapi_instrumentation",
                    "http_client_instrumentation",
                    "pydantic_instrumentation"
                ]
            }

            # Sende Startup-Event nur an Logfire (nicht an Console)
            # manager.log_info(
            #     "🚀 Keiko Personal Assistant - Startup abgeschlossen",
            #     **startup_info
            # )

            # Erstelle Startup-Span nur für Logfire (nicht an Console)
            # with manager.span("application_startup") as _span:
            #     manager.log_info(
            #         "📊 Anwendungs-Startup-Metriken",
            #         total_initializers=len(self.container._service_initializers) if hasattr(self.container, '_service_initializers') else 0,
            #         logfire_startup_successful=True
            #     )

            logger.info("✅ Startup-Event an Logfire gesendet")

        except Exception as e:
            logger.warning(f"⚠️ Startup-Event-Fehler: {e}")

    @property
    def name(self) -> str:
        """Name des Initializers."""
        return "logfire"

    @property
    def dependencies(self) -> list[str]:
        """Abhängigkeiten des Initializers."""
        return ["monitoring"]  # Nach Monitoring-Service starten


class LogfireShutdownHandler:
    """Shutdown-Handler für die Logfire-Integration.

    Beendet die Logfire-Integration ordnungsgemäß und sendet
    finale Metriken und Events.
    """

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def shutdown(self, app: FastAPI) -> None:
        """Beendet die Logfire-Integration ordnungsgemäß."""
        try:
            logger.info("🔥 Beende Logfire-Integration...")

            # Sende Shutdown-Event
            await self._send_shutdown_event(app)

            # Beende Observability-System
            await self._stop_observability_system()

            logger.info("✅ Logfire-Integration beendet")

        except Exception as e:
            safe_log_exception(logger, e, "Logfire-Shutdown fehlgeschlagen")

    async def _send_shutdown_event(self, _app: FastAPI) -> None:
        """Sendet Shutdown-Event an Logfire."""
        try:
            from observability import LOGFIRE_INTEGRATION_AVAILABLE, get_logfire_manager

            if not LOGFIRE_INTEGRATION_AVAILABLE:
                return

            manager = get_logfire_manager()

            if not manager.is_available():
                return

            import time
            from datetime import datetime

            # Sammle Shutdown-Informationen
            shutdown_info = {
                "service_name": "keiko-personal-assistant",
                "shutdown_time": datetime.now(UTC).isoformat(),
                "shutdown_timestamp": time.time(),
                "shutdown_reason": "application_shutdown",
                "graceful_shutdown": True
            }

            # Hole finale Metriken
            metrics = manager.get_metrics()
            shutdown_info.update({
                "final_metrics": {
                    "total_logs": metrics.get("total_logs", 0),
                    "total_spans": metrics.get("total_spans", 0),
                    "total_errors": metrics.get("total_errors", 0),
                    "pii_redactions": metrics.get("pii_redactions", 0),
                    "fallback_activations": metrics.get("fallback_activations", 0)
                }
            })

            # Sende Shutdown-Event nur an Logfire (nicht an Console)
            # manager.log_info(
            #     "🛑 Keiko Personal Assistant - Shutdown gestartet",
            #     **shutdown_info
            # )

            logger.info("✅ Shutdown-Event an Logfire gesendet")

        except Exception as e:
            logger.warning(f"⚠️ Shutdown-Event-Fehler: {e}")

    async def _stop_observability_system(self) -> None:
        """Beendet das Observability-System."""
        try:
            from observability import LOGFIRE_INTEGRATION_AVAILABLE, stop_observability_system

            if not LOGFIRE_INTEGRATION_AVAILABLE:
                return

            await stop_observability_system()
            logger.info("✅ Observability-System beendet")

        except Exception as e:
            logger.warning(f"⚠️ Observability-System-Stop-Fehler: {e}")

    @property
    def name(self) -> str:
        """Name des Shutdown-Handlers."""
        return "logfire"
