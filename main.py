"""Keiko Personal Assistant - Minimaler FastAPI Bootstrap."""

from __future__ import annotations

import os


# Initialisiere Deployment-Modus beim Start
def initialize_deployment_mode():
    """Initialisiert und setzt den Deployment-Modus beim Backend-Start."""
    # Importiere KEI-Logging für einheitliche Ausgaben
    from kei_logging import get_logger
    logger = get_logger(__name__)

    try:
        from config.deployment_mode import deployment_mode
        mode = deployment_mode.detect_and_set_mode()
        logger.info(f"🚀 Keiko Backend startet im {mode.value.upper()}-Modus")

        # Setze Environment-Variable für andere Prozesse
        os.environ["KEIKO_DEPLOYMENT_MODE"] = mode.value

        return mode
    except Exception as e:
        logger.warning(f"⚠️  Deployment-Modus-Initialisierung fehlgeschlagen: {e}")
        logger.info("🔄 Verwende Fallback-Konfiguration")
        return None

# Initialisiere Deployment-Modus sofort beim Import
_deployment_mode = initialize_deployment_mode()

from app.application import KeikoApplication

# Für Development-Umgebung deaktivieren wir den Orchestrator
KEIKO_ENV = os.getenv("KEIKO_ENV", "development")

# Initialisiere Logger für Orchestrator-Meldungen
from kei_logging import get_logger

_main_logger = get_logger(__name__)

if KEIKO_ENV == "development":
    ORCHESTRATOR_AVAILABLE = False
    orchestrator = None
    _main_logger.info("ℹ️  Development mode: Startup orchestrator disabled")
else:
    try:
        import sys
        # Add parent directory to path for startup_orchestration module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from services.orchestrator.api import (
            get_orchestrator_service,
            initialize_orchestrator_service,
        )
        orchestrator = get_orchestrator_service()
        ORCHESTRATOR_AVAILABLE = True
    except ImportError:
        import sys  # Definiere sys auch im except Block
        # Definiere die Funktionen als None im except Block
        get_orchestrator_service = None
        initialize_orchestrator_service = None
        orchestrator = None  # Definiere orchestrator auch im except Block
        ORCHESTRATOR_AVAILABLE = False
        _main_logger.warning("⚠️  Startup orchestrator not available - running in simple mode")

# Orchestrator-Konfiguration für erweiterte KeikoApplication
class ExtendedKeikoApplication(KeikoApplication):
    """Erweiterte KeikoApplication mit Orchestrator-Support."""

    def __init__(self, container=None):
        super().__init__(container)

        # Orchestrator-Initialisierung in den bestehenden Lifespan integrieren
        if ORCHESTRATOR_AVAILABLE:
            self._setup_orchestrator_integration()

    def _setup_orchestrator_integration(self):
        """Integriert Orchestrator-Logik in den bestehenden Startup-Prozess."""
        from kei_logging import get_logger
        logger = get_logger(__name__)

        # Erweitere den bestehenden StartupManager
        original_startup = self.factory.startup_manager.on_startup

        async def extended_startup(app):
            """Erweiterte Startup-Logik mit Orchestrator."""
            # Führe zuerst den Standard-Startup durch
            await original_startup(app)

            # Dann Orchestrator-spezifische Initialisierung
            try:
                await initialize_orchestrator_service()
                await orchestrator.start()

                logger.info("Waiting for required infrastructure services...")
                # Vereinfachte Health-Check ohne wait_for_required_services
                health_result = await orchestrator.get_service_health()
                health_status = {
                    "status": "healthy" if health_result.service_healthy else "unhealthy"
                }
                if health_status.get("status") != "healthy":
                    logger.warning(f"Orchestrator health check warning: {health_status}")
                else:
                    logger.info("Orchestrator service is healthy - backend ready to serve requests")
            except Exception as e:
                logger.warning(f"Orchestrator startup failed, continuing in simple mode: {e}")
                logger.info("⚠️  Continuing in simple development mode")

        # Ersetze die Startup-Methode
        self.factory.startup_manager.on_startup = extended_startup

# Minimale App-Initialisierung über erweiterte KeikoApplication
application = ExtendedKeikoApplication()
app = application.app
