"""StartupManager koordiniert Startup- und Shutdown-Abläufe.

Diese Klasse verwendet modularisierte Startup-Koordinatoren
um die Komplexität zu reduzieren.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kei_logging import get_logger

from .startup.startup_coordinator import StartupCoordinator

# Import für System Heartbeat Service
try:
    from services.system_heartbeat_service import initialize_system_heartbeat_service
    HEARTBEAT_SERVICE_AVAILABLE = True
except ImportError:
    initialize_system_heartbeat_service = None
    HEARTBEAT_SERVICE_AVAILABLE = False

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .service_container import ServiceContainer

logger = get_logger(__name__)
class StartupManager:
    """Verwaltet Anwendungs-Startup und -Shutdown.

    Diese Klasse verwendet den StartupCoordinator
    um die Komplexität zu reduzieren.
    """

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self.coordinator = StartupCoordinator(container)

    async def on_startup(self, app: FastAPI) -> None:
        """Führt die Startsequenz aus."""
        await self.coordinator.startup(app)

        # System Heartbeat Service starten
        if HEARTBEAT_SERVICE_AVAILABLE:
            try:
                heartbeat_service = initialize_system_heartbeat_service(self.container)
                await heartbeat_service.start()
                logger.info("💓 System Heartbeat Service gestartet")
            except Exception as e:
                logger.warning(f"System Heartbeat Service konnte nicht gestartet werden: {e}")

        logger.info("Startup abgeschlossen")

    async def on_shutdown(self, app: FastAPI) -> None:
        """Führt die Shutdown-Sequenz aus."""
        logger.info("Starte Shutdown...")

        # System Heartbeat Service stoppen
        if HEARTBEAT_SERVICE_AVAILABLE:
            try:
                from services.system_heartbeat_service import get_system_heartbeat_service
                heartbeat_service = get_system_heartbeat_service()
                if heartbeat_service:
                    await heartbeat_service.stop()
                    logger.info("🛑 System Heartbeat Service gestoppt")
            except Exception as e:
                logger.warning(f"Fehler beim Stoppen des System Heartbeat Service: {e}")

        await self.coordinator.shutdown(app)
        logger.info("Shutdown abgeschlossen")


__all__ = [
    "StartupManager",
]
