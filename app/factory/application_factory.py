"""Application Factory für die Keiko-Anwendung.

Diese Factory kapselt die FastAPI-App-Erstellung und -Konfiguration
um die Komplexität der KeikoApplication zu reduzieren.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from kei_logging import get_logger

from ..common.constants import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from ..startup_manager import StartupManager
from .middleware_factory import MiddlewareFactory
from .router_factory import RouterFactory

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from ..service_container import ServiceContainer

logger = get_logger(__name__)


class ApplicationFactory:
    """Factory für FastAPI-Anwendungserstellung."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self.startup_manager = StartupManager(container)
        self.middleware_factory = MiddlewareFactory(container)
        self.router_factory = RouterFactory()

    def create_application(
        self,
        title: str | None = None,
        description: str | None = None,
        version: str | None = None,
        debug: bool = False
    ) -> FastAPI:
        """Erstellt eine konfigurierte FastAPI-Anwendung.

        Args:
            title: Anwendungstitel (default: APP_TITLE)
            description: Anwendungsbeschreibung (default: APP_DESCRIPTION)
            version: Anwendungsversion (default: APP_VERSION)
            debug: Debug-Modus aktivieren

        Returns:
            Konfigurierte FastAPI-Anwendung
        """
        # Lifespan-Manager erstellen
        lifespan = self._create_lifespan_manager()

        # FastAPI-App erstellen
        app = FastAPI(
            title=title or APP_TITLE,
            description=description or APP_DESCRIPTION,
            version=version or APP_VERSION,
            debug=debug,
            lifespan=lifespan,
            generate_unique_id_function=self._generate_unique_id,
        )

        # Middleware einrichten
        self.middleware_factory.setup_middleware(app)

        # Router registrieren
        self.router_factory.register_all_routers(app)

        logger.info(f"FastAPI-Anwendung erstellt: {app.title} v{app.version}")  # type: ignore[attr-defined]
        return app

    def _create_lifespan_manager(self) -> Callable[[FastAPI], AsyncGenerator[None, None]]:
        """Erstellt den Lifespan-Manager für die Anwendung."""
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """Lifespan-Manager für Startup und Shutdown."""
            # Startup
            logger.info("Starte Anwendung...")
            await self.startup_manager.on_startup(app)

            try:
                yield
            finally:
                # Shutdown
                logger.info("Beende Anwendung...")
                await self.startup_manager.on_shutdown(app)

        return lifespan

    @staticmethod
    def _generate_unique_id(route: Any) -> str:
        """Generiert eindeutige IDs für Routen.

        Args:
            route: Route-Objekt

        Returns:
            Eindeutige Route-ID
        """
        if hasattr(route, "tags") and route.tags:
            return f"{route.tags[0]}-{route.name}"
        return route.name

    def create_test_application(self) -> FastAPI:
        """Erstellt eine Test-Anwendung mit minimaler Konfiguration.

        Returns:
            Test-FastAPI-Anwendung
        """
        app = FastAPI(
            title=f"{APP_TITLE} (Test)",
            description=f"{APP_DESCRIPTION} - Test Environment",
            version=f"{APP_VERSION}-test",
            debug=True,
        )

        # Nur grundlegende Middleware für Tests
        self._setup_test_middleware(app)

        # Nur Health Endpoints für Tests
        self._setup_test_routes(app)

        logger.info("Test-Anwendung erstellt")
        return app

    def _setup_test_middleware(self, app: FastAPI) -> None:
        """Richtet minimale Middleware für Tests ein."""
        try:
            from fastapi.middleware.cors import CORSMiddleware

            from ..common.constants import DEFAULT_CORS_ORIGINS

            app.add_middleware(
                CORSMiddleware,
                allow_origins=DEFAULT_CORS_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.debug("Test-Middleware eingerichtet")
        except ImportError as exc:
            logger.warning(f"Test-Middleware konnte nicht eingerichtet werden: {exc}")

    def _setup_test_routes(self, app: FastAPI) -> None:
        """Richtet minimale Routen für Tests ein."""
        try:
            from ..health_endpoints import register_health_endpoints
            register_health_endpoints(app)
            logger.debug("Test-Routen eingerichtet")
        except ImportError as exc:
            logger.warning(f"Test-Routen konnten nicht eingerichtet werden: {exc}")


class ApplicationConfigurationBuilder:
    """Builder für Anwendungskonfiguration."""

    def __init__(self) -> None:
        self.config = {
            "title": APP_TITLE,
            "description": APP_DESCRIPTION,
            "version": APP_VERSION,
            "debug": False,
            "middleware": {},
            "routers": {},
        }

    def with_title(self, title: str) -> ApplicationConfigurationBuilder:
        """Setzt den Anwendungstitel."""
        self.config["title"] = title
        return self

    def with_description(self, description: str) -> ApplicationConfigurationBuilder:
        """Setzt die Anwendungsbeschreibung."""
        self.config["description"] = description
        return self

    def with_version(self, version: str) -> ApplicationConfigurationBuilder:
        """Setzt die Anwendungsversion."""
        self.config["version"] = version
        return self

    def with_debug(self, debug: bool = True) -> ApplicationConfigurationBuilder:
        """Aktiviert/deaktiviert Debug-Modus."""
        self.config["debug"] = debug
        return self

    def with_cors_origins(self, origins: list[str]) -> ApplicationConfigurationBuilder:
        """Setzt CORS-Origins."""
        self.config["middleware"]["cors_origins"] = origins
        return self

    def with_rate_limiting(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ) -> ApplicationConfigurationBuilder:
        """Konfiguriert Rate Limiting."""
        self.config["middleware"]["rate_limiting"] = {
            "requests_per_minute": requests_per_minute,
            "burst_size": burst_size,
        }
        return self

    def enable_mtls(self, enable: bool = True) -> ApplicationConfigurationBuilder:
        """Aktiviert/deaktiviert mTLS."""
        self.config["middleware"]["mtls_enabled"] = enable
        return self

    def build(self) -> dict[str, Any]:
        """Erstellt die finale Konfiguration."""
        return self.config.copy()


class ApplicationValidator:
    """Validiert Anwendungskonfiguration."""

    @staticmethod
    def validate_application(app: FastAPI) -> dict[str, Any]:
        """Validiert eine FastAPI-Anwendung.

        Args:
            app: FastAPI-Anwendungsinstanz

        Returns:
            Validierungsergebnisse
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
        }

        # Basis-Validierung
        if not app.title:  # type: ignore[attr-defined]
            results["errors"].append("Anwendungstitel fehlt")
            results["valid"] = False

        if not app.version:  # type: ignore[attr-defined]
            results["warnings"].append("Anwendungsversion fehlt")

        # Router-Validierung
        if not app.routes:
            results["warnings"].append("Keine Routen registriert")

        # Middleware-Validierung
        middleware_count = len(app.user_middleware)  # type: ignore[attr-defined]
        if middleware_count == 0:
            results["warnings"].append("Keine Middleware registriert")

        # Informationen sammeln
        results["info"] = {
            "title": app.title,  # type: ignore[attr-defined]
            "version": app.version,  # type: ignore[attr-defined]
            "route_count": len(app.routes),
            "middleware_count": middleware_count,
        }

        return results

    @staticmethod
    def validate_configuration(config: dict[str, Any]) -> dict[str, Any]:
        """Validiert eine Anwendungskonfiguration.

        Args:
            config: Konfigurationsdictionary

        Returns:
            Validierungsergebnisse
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Pflichtfelder prüfen
        required_fields = ["title", "description", "version"]
        for field in required_fields:
            if field not in config or not config[field]:
                results["errors"].append(f"Pflichtfeld '{field}' fehlt")
                results["valid"] = False

        # Version-Format prüfen
        if "version" in config:
            version = config["version"]
            if not isinstance(version, str) or not version.count(".") >= 2:
                results["warnings"].append("Version sollte im Format 'x.y.z' sein")

        return results


__all__ = [
    "ApplicationConfigurationBuilder",
    "ApplicationFactory",
    "ApplicationValidator",
]
