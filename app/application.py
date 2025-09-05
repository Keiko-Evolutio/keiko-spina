"""KeikoApplication kapselt die FastAPI-App und Lifecycle-Management.

Diese Klasse verwendet ApplicationFactory und befolgt das
Single Responsibility Principle. Alle Kommentare sind auf Deutsch,
während Identifier auf Englisch bleiben. Type Hints folgen PEP 484, Docstrings
PEP 257.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from kei_logging import configure_loki_logging

from .common.logger_utils import get_module_logger
from .factory.application_factory import ApplicationFactory
from .service_container import ServiceContainer
from .startup_manager import StartupManager

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_module_logger(__name__)


class KeikoApplication:
    """Erstellt und konfiguriert die FastAPI‑Anwendung.

    Diese Klasse bindet Middleware, Router und Health‑/Metrics‑Endpoints ein
    und stellt helper Funktionen für Lifecycle‑abhängige Operationen bereit.
    """

    def __init__(self, container: ServiceContainer | None = None) -> None:
        # Container verwaltet zentrale Abhängigkeiten und Factories
        self.container: ServiceContainer = container or ServiceContainer()

        # ApplicationFactory für FastAPI-Setup verwenden
        self.factory = ApplicationFactory(self.container)

        # FastAPI App über Factory erstellen
        self.app: FastAPI = self.factory.create_application()

        logger.info("Keiko Application initialisiert")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Lifecycle-Manager für Startup und Shutdown.

        Koordiniert Validierung, Tracing, Service‑Initialisierung und
        sauberen Shutdown.
        """
        manager = None

        # Startup
        try:
            # Zentrale Logging‑Konfiguration früh aktivieren, damit alle
            # nachfolgenden Komponenten JSON‑Logs für Loki/Promtail ausgeben.
            try:
                configure_loki_logging(service_name="keiko-api")
                logger.info("Loki JSON Logging konfiguriert")
            except Exception as log_exc:  # pragma: no cover
                logger.warning(f"Loki Logging konnte nicht initialisiert werden: {log_exc}")

            # .env und Pflichtvariablen validieren
            from config.settings import validate_environment_or_raise
            validate_environment_or_raise()

            # Startup mittels StartupManager koordinieren
            manager = StartupManager(self.container)
            await manager.on_startup(app)

            yield
        except Exception as exc:
            logger.critical(f"Startup fehlgeschlagen: {exc}")
            raise
        finally:
            # Shutdown sicher durchführen - nur wenn Manager existiert
            if manager is not None:
                try:
                    await manager.on_shutdown(app)
                except Exception as shutdown_exc:
                    logger.exception(f"Shutdown fehlgeschlagen: {shutdown_exc}")
                    # Shutdown-Fehler nicht weiterwerfen, um rekursive Calls zu vermeiden




__all__ = [
    "KeikoApplication",
]
