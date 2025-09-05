"""Konsolidierte Middleware-Base-Klassen für die Keiko-API.

Eliminiert Code-Duplikation durch abstrakte Basis-Implementierungen
für alle Middleware-Komponenten. Implementiert einheitliche Patterns.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger, structured_msg

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from starlette.responses import Response

logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

class MiddlewareConstants:
    """Middleware-Konstanten."""

    # Header-Namen
    HEADER_TRACE_ID = "X-Trace-Id"
    HEADER_CORRELATION_ID = "X-Correlation-Id"
    HEADER_REQUEST_ID = "X-Request-Id"
    HEADER_TENANT_ID = "X-Tenant-Id"
    HEADER_USER_AGENT = "User-Agent"

    # Timing-Konstanten
    DEFAULT_TIMEOUT_SECONDS = 30
    SLOW_REQUEST_THRESHOLD_MS = 1000

    # Status-Konstanten
    STATUS_SUCCESS = "success"
    STATUS_ERROR = "error"
    STATUS_TIMEOUT = "timeout"


# ============================================================================
# BASE MIDDLEWARE CLASSES
# ============================================================================

class BaseKeikoMiddleware(BaseHTTPMiddleware, ABC):
    """Abstrakte Basis-Klasse für alle Keiko-Middleware.

    Konsolidiert gemeinsame Middleware-Funktionalität und eliminiert
    Code-Duplikation zwischen verschiedenen Middleware-Implementierungen.
    """

    def __init__(
        self,
        app,
        middleware_name: str,
        enabled: bool = True,
        excluded_paths: set[str] | None = None
    ):
        """Initialisiert Basis-Middleware.

        Args:
            app: FastAPI/Starlette-Anwendung
            middleware_name: Name der Middleware für Logging
            enabled: Ob Middleware aktiviert ist
            excluded_paths: Pfade die von Middleware ausgenommen sind
        """
        super().__init__(app)
        self.middleware_name = middleware_name
        self.enabled = enabled
        self.excluded_paths = excluded_paths or set()
        self.logger = get_logger(f"middleware.{middleware_name}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Hauptdispatch-Methode mit einheitlichem Pattern.

        Args:
            request: HTTP-Request
            call_next: Nächste Middleware/Handler

        Returns:
            HTTP-Response
        """
        # Middleware-Bypass prüfen
        if not self.enabled or self._should_skip_request(request):
            return await call_next(request)

        # Request-Kontext initialisieren
        start_time = time.time()
        self._initialize_request_context(request, start_time)

        try:
            # Pre-Processing
            await self._pre_process_request(request)

            # Request verarbeiten
            response = await call_next(request)

            # Post-Processing
            await self._post_process_response(request, response, start_time)

            return response

        except Exception as e:
            # Error-Handling
            return await self._handle_middleware_error(request, e, start_time)

    def _should_skip_request(self, request: Request) -> bool:
        """Prüft ob Request von Middleware übersprungen werden soll.

        Args:
            request: HTTP-Request

        Returns:
            True wenn Request übersprungen werden soll
        """
        # OPTIONS-Requests (CORS Preflight) überspringen
        if request.method == "OPTIONS":
            return True

        # Ausgenommene Pfade prüfen
        path = request.url.path
        return any(excluded in path for excluded in self.excluded_paths)

    def _initialize_request_context(self, request: Request, start_time: float) -> None:
        """Initialisiert Request-Kontext mit Standard-Attributen.

        Args:
            request: HTTP-Request
            start_time: Request-Startzeit
        """
        if not hasattr(request.state, "start_time"):
            request.state.start_time = start_time

        if not hasattr(request.state, "middleware_context"):
            request.state.middleware_context = {}

    def _get_request_duration_ms(self, request: Request) -> int:
        """Berechnet Request-Dauer in Millisekunden.

        Args:
            request: HTTP-Request

        Returns:
            Request-Dauer in Millisekunden
        """
        start_time = getattr(request.state, "start_time", time.time())
        return int((time.time() - start_time) * 1000)

    def _extract_tenant_id(self, request: Request) -> str | None:
        """Extrahiert Tenant-ID aus Request-Headers.

        Args:
            request: HTTP-Request

        Returns:
            Tenant-ID oder None
        """
        return (
            request.headers.get(MiddlewareConstants.HEADER_TENANT_ID) or
            request.headers.get(MiddlewareConstants.HEADER_TENANT_ID.lower())
        )

    def _log_request_completion(
        self,
        request: Request,
        status_code: int,
        duration_ms: int,
        status: str = MiddlewareConstants.STATUS_SUCCESS
    ) -> None:
        """Loggt Request-Completion mit strukturierten Daten.

        Args:
            request: HTTP-Request
            status_code: HTTP-Status-Code
            duration_ms: Request-Dauer in Millisekunden
            status: Request-Status
        """
        tenant_id = self._extract_tenant_id(request)
        correlation_id = getattr(request.state, "correlation_id", None)

        self.logger.info(structured_msg(
            f"{self.middleware_name} request completed",
            correlation_id=correlation_id,
            tenant=tenant_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
            status=status,
            middleware=self.middleware_name
        ))

    @abstractmethod
    async def _pre_process_request(self, request: Request) -> None:
        """Pre-Processing-Hook für spezifische Middleware-Logik.

        Args:
            request: HTTP-Request
        """

    @abstractmethod
    async def _post_process_response(
        self,
        request: Request,
        response: Response,
        start_time: float
    ) -> None:
        """Post-Processing-Hook für spezifische Middleware-Logik.

        Args:
            request: HTTP-Request
            response: HTTP-Response
            start_time: Request-Startzeit
        """

    async def _handle_middleware_error(
        self,
        request: Request,
        error: Exception,
        start_time: float
    ) -> Response:
        """Standard-Error-Handling für Middleware-Fehler.

        Args:
            request: HTTP-Request
            error: Aufgetretener Fehler
            start_time: Request-Startzeit

        Returns:
            Error-Response
        """
        duration_ms = self._get_request_duration_ms(request)

        self.logger.error(structured_msg(
            f"{self.middleware_name} middleware error",
            error_type=type(error).__name__,
            error_message=str(error),
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            middleware=self.middleware_name
        ))

        # Re-raise Exception für globale Error-Handler
        raise error


# ============================================================================
# SPECIALIZED BASE CLASSES
# ============================================================================

class TracingBaseMiddleware(BaseKeikoMiddleware):
    """Basis-Klasse für Tracing-Middleware.

    Konsolidiert Tracing-spezifische Funktionalität.
    """

    def __init__(self, app, middleware_name: str, service_name: str = "keiko-api"):
        """Initialisiert Tracing-Middleware.

        Args:
            app: FastAPI/Starlette-Anwendung
            middleware_name: Name der Middleware
            service_name: Name des Services für Tracing
        """
        super().__init__(app, middleware_name)
        self.service_name = service_name

    def _generate_trace_id(self) -> str:
        """Generiert neue Trace-ID.

        Returns:
            Neue Trace-ID
        """
        import uuid
        return str(uuid.uuid4())

    def _set_trace_attributes(
        self,
        request: Request,
        response: Response | None = None
    ) -> None:
        """Setzt Trace-Attribute für OpenTelemetry.

        Args:
            request: HTTP-Request
            response: HTTP-Response (optional)
        """
        try:
            from opentelemetry import trace as otel_trace
            span = otel_trace.get_current_span()

            if span and span.is_recording():
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.target", request.url.path)
                span.set_attribute("service.name", self.service_name)

                if response:
                    span.set_attribute("http.status_code", response.status_code)

                # Tenant-Attribut setzen
                tenant_id = self._extract_tenant_id(request)
                if tenant_id:
                    span.set_attribute("kei.tenant", tenant_id)

                # Cache-Attribute aus Request-State übernehmen
                cache_outcome = getattr(request.state, "cache_outcome", None)
                if cache_outcome:
                    span.set_attribute("kei.cache.outcome", cache_outcome)

        except Exception:
            # Tracing-Fehler sind nicht kritisch
            pass

    async def process_request(self, request: Request, call_next: Callable) -> Response:
        """Standard-Implementierung für Tracing-Middleware.

        Args:
            request: HTTP-Request
            call_next: Nächste Middleware/Handler

        Returns:
            HTTP-Response
        """
        # Trace-Attribute vor Request setzen
        self._set_trace_attributes(request)

        # Request verarbeiten
        response = await call_next(request)

        # Trace-Attribute nach Response setzen
        self._set_trace_attributes(request, response)

        return response


class AuthenticationBaseMiddleware(BaseKeikoMiddleware):
    """Basis-Klasse für Authentication-Middleware.

    Konsolidiert Authentication-spezifische Funktionalität.
    """

    def __init__(
        self,
        app,
        middleware_name: str,
        excluded_paths: set[str] | None = None
    ):
        """Initialisiert Authentication-Middleware.

        Args:
            app: FastAPI/Starlette-Anwendung
            middleware_name: Name der Middleware
            excluded_paths: Pfade die von Authentication ausgenommen sind
        """
        # Standard-Ausnahmen für Authentication
        default_excluded = {
            "/docs", "/redoc", "/openapi.json", "/health", "/metrics",
            "/api/v1/rpc", "/api/v1/mcp/external",
            "/api/v1/webhooks/inbound", "/api/v1/alerts/",
            "/api/v1/health", "/api/v1/specs", "/api/v1/logs/"
        }

        if excluded_paths:
            default_excluded.update(excluded_paths)

        super().__init__(app, middleware_name, excluded_paths=default_excluded)

    def is_excluded_path(self, path: str) -> bool:
        """Prüft ob ein Pfad von der Authentication ausgenommen ist.

        Args:
            path: URL-Pfad

        Returns:
            True wenn Pfad ausgenommen ist
        """
        # Exact match
        if path in self.excluded_paths:
            return True

        # Prefix match für Pfade die mit / enden
        for excluded_path in self.excluded_paths:
            if excluded_path.endswith("/") and path.startswith(excluded_path):
                return True

        # Pattern matching für spezielle Pfade
        return any(excluded in path for excluded in self.excluded_paths)

    def _extract_bearer_token(self, request: Request) -> str | None:
        """Extrahiert Bearer-Token aus Authorization-Header.

        Args:
            request: HTTP-Request

        Returns:
            Bearer-Token oder None
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        return auth_header.split(" ")[1]

    def _has_test_scopes(self, request: Request) -> bool:
        """Prüft ob Request Test-Scopes-Header hat.

        Args:
            request: HTTP-Request

        Returns:
            True wenn Test-Scopes vorhanden
        """
        return bool(
            request.headers.get("X-Scopes") or
            request.headers.get("x-scopes")
        )

    async def process_request(self, request: Request, call_next: Callable) -> Response:
        """Standard-Implementierung für Authentication-Middleware.

        Args:
            request: HTTP-Request
            call_next: Nächste Middleware/Handler

        Returns:
            HTTP-Response
        """
        # Prüfen ob Pfad von Authentication ausgenommen ist
        if self.is_excluded_path(request.url.path):
            return await call_next(request)

        # Bearer-Token extrahieren
        token = self._extract_bearer_token(request)
        if not token:
            # Keine Authentifizierung - Request trotzdem weiterleiten
            # (konkrete Implementierungen können hier stricter sein)
            pass

        # Request verarbeiten
        return await call_next(request)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_middleware_config(
    middleware_name: str,
    enabled: bool = True,
    excluded_paths: set[str] | None = None,
    **kwargs
) -> dict[str, Any]:
    """Erstellt Middleware-Konfiguration.

    Args:
        middleware_name: Name der Middleware
        enabled: Ob Middleware aktiviert ist
        excluded_paths: Ausgenommene Pfade
        **kwargs: Zusätzliche Konfigurationsparameter

    Returns:
        Middleware-Konfiguration
    """
    config = {
        "middleware_name": middleware_name,
        "enabled": enabled,
        "excluded_paths": excluded_paths or set()
    }
    config.update(kwargs)
    return config
