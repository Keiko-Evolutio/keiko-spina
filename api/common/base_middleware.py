"""Basis-Middleware-Abstraktion für alle Keiko-Middleware-Komponenten.

Stellt eine einheitliche Basis-Klasse für alle Middleware-Implementierungen
bereit und eliminiert Code-Duplikation zwischen den verschiedenen Middleware-Typen.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request, Response
    from starlette.types import ASGIApp


class BaseKeikoMiddleware(BaseHTTPMiddleware, ABC):
    """Basis-Klasse für alle Keiko-Middleware-Komponenten.

    Stellt gemeinsame Funktionalität bereit:
    - Standardisiertes Logging
    - Konfigurationsverwaltung
    - Error-Handling-Patterns
    - Performance-Monitoring
    - Request/Response-Hooks
    """

    def __init__(
        self,
        app: ASGIApp,
        config: dict[str, Any] | None = None,
        name: str | None = None
    ) -> None:
        """Initialisiert die Basis-Middleware.

        Args:
            app: ASGI-Anwendung
            config: Middleware-spezifische Konfiguration
            name: Name der Middleware (für Logging)
        """
        super().__init__(app)
        self.config = config or {}
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"middleware.{self.name.lower()}")

        # Performance-Metriken
        self._request_count = 0
        self._total_processing_time = 0.0

        self.logger.debug(f"{self.name} Middleware initialisiert")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Hauptverarbeitungslogik der Middleware.

        Args:
            request: Eingehender HTTP-Request
            call_next: Nächste Middleware/Handler in der Kette

        Returns:
            HTTP-Response
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Pre-Processing Hook
            await self.before_request(request)

            # Hauptverarbeitung
            response = await self.process_request(request, call_next)

            # Post-Processing Hook
            await self.after_request(request, response)

            return response

        except Exception as exc:
            # Error-Handling Hook
            return await self.handle_error(request, exc)

        finally:
            # Performance-Tracking
            processing_time = time.time() - start_time
            self._total_processing_time += processing_time

            if self.config.get("log_performance", False):
                self.logger.debug(
                    f"{self.name} verarbeitet Request in {processing_time:.3f}s"
                )

    @abstractmethod
    async def process_request(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Middleware-spezifische Verarbeitungslogik.

        Diese Methode muss von allen Subklassen implementiert werden.

        Args:
            request: HTTP-Request
            call_next: Nächste Middleware/Handler

        Returns:
            HTTP-Response
        """

    async def before_request(self, request: Request) -> None:
        """Hook der vor der Hauptverarbeitung ausgeführt wird.

        Kann von Subklassen überschrieben werden für:
        - Request-Validierung
        - Header-Extraktion
        - Kontext-Setup

        Args:
            request: HTTP-Request
        """

    async def after_request(self, request: Request, response: Response) -> None:
        """Hook der nach der Hauptverarbeitung ausgeführt wird.

        Kann von Subklassen überschrieben werden für:
        - Response-Modifikation
        - Cleanup-Operationen
        - Metriken-Sammlung

        Args:
            request: HTTP-Request
            response: HTTP-Response
        """

    async def handle_error(self, request: Request, exc: Exception) -> Response:
        """Standardisierte Error-Behandlung.

        Kann von Subklassen überschrieben werden für spezifische Error-Handling.

        Args:
            request: HTTP-Request
            exc: Aufgetretene Exception

        Returns:
            Error-Response
        """
        self.logger.error(
            f"{self.name} Middleware Fehler: {exc}",
            extra={"path": request.url.path, "method": request.method}
        )

        # Re-raise für Standard-Error-Handler
        raise exc

    def is_excluded_path(self, path: str) -> bool:
        """Prüft ob ein Pfad von der Middleware-Verarbeitung ausgenommen ist.

        Args:
            path: URL-Pfad

        Returns:
            True wenn Pfad ausgenommen ist
        """
        excluded_paths = self.config.get("excluded_paths", [])
        return any(path.startswith(excluded) for excluded in excluded_paths)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Holt einen Konfigurationswert mit Fallback.

        Args:
            key: Konfigurationsschlüssel
            default: Fallback-Wert

        Returns:
            Konfigurationswert oder Fallback
        """
        return self.config.get(key, default)

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück.

        Returns:
            Dictionary mit Performance-Metriken
        """
        avg_time = (
            self._total_processing_time / self._request_count
            if self._request_count > 0 else 0.0
        )

        return {
            "middleware_name": self.name,
            "request_count": self._request_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": avg_time
        }


class ConditionalMiddleware(BaseKeikoMiddleware):
    """Middleware die basierend auf Bedingungen aktiviert/deaktiviert werden kann.

    Nützlich für Feature-Flags oder umgebungsabhängige Middleware.
    """

    def __init__(
        self,
        app: ASGIApp,
        condition_func: Callable[[Request], Awaitable[bool]],
        config: dict[str, Any] | None = None,
        name: str | None = None
    ) -> None:
        """Initialisiert bedingte Middleware.

        Args:
            app: ASGI-Anwendung
            condition_func: Async-Funktion die bestimmt ob Middleware aktiv ist
            config: Middleware-Konfiguration
            name: Middleware-Name
        """
        super().__init__(app, config, name)
        self.condition_func = condition_func

    async def process_request(self, request: Request, call_next: Callable) -> Response:
        """Führt Middleware nur aus wenn Bedingung erfüllt ist.

        Args:
            request: HTTP-Request
            call_next: Nächste Middleware/Handler

        Returns:
            HTTP-Response
        """
        if await self.condition_func(request):
            # Bedingung erfüllt - normale Verarbeitung
            return await call_next(request)
        # Middleware überspringen
        return await call_next(request)


class MiddlewareConfig:
    """Konfigurationsklasse für Middleware-Setup.

    Zentralisiert die Konfiguration aller Middleware-Komponenten.
    """

    def __init__(
        self,
        environment: str = "development",
        debug: bool = False,
        log_performance: bool = False
    ) -> None:
        """Initialisiert Middleware-Konfiguration.

        Args:
            environment: Umgebung (development/production)
            debug: Debug-Modus aktiviert
            log_performance: Performance-Logging aktiviert
        """
        self.environment = environment
        self.debug = debug
        self.log_performance = log_performance

    def get_base_config(self) -> dict[str, Any]:
        """Gibt Basis-Konfiguration für alle Middleware zurück."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_performance": self.log_performance
        }
