"""Monitoring Middleware für automatisches Performance-Tracking.
Integriert Monitoring in alle HTTP-Requests und Voice-Workflows.
"""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.container import get_container
from kei_logging import get_logger
from monitoring.interfaces import IMonitoringService

logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Monitoring Middleware für automatisches Performance-Tracking.
    Trackt alle HTTP-Requests und integriert sich in Voice-Workflows.
    """

    def __init__(self, app, monitoring_service: IMonitoringService | None = None):
        super().__init__(app)
        self._monitoring_service = monitoring_service
        self._enabled = True

        # Versuche Monitoring Service aus DI Container zu holen falls nicht übergeben
        if self._monitoring_service is None:
            try:
                container = get_container()
                self._monitoring_service = container.resolve(IMonitoringService)
            except Exception as e:
                logger.warning(f"Could not resolve monitoring service: {e}")
                self._enabled = False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet HTTP-Request mit Monitoring."""
        if not self._enabled or not self._monitoring_service:
            return await call_next(request)

        # Request-Tracking starten
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Request-Metadaten sammeln
        endpoint = self._get_endpoint_name(request)
        method = request.method

        # Request-Kontext für Voice-Workflows
        if self._is_voice_request(request):
            await self._start_voice_workflow_tracking(request, request_id)

        try:
            # Request verarbeiten
            response = await call_next(request)

            # Response-Zeit tracken
            duration_ms = (time.time() - start_time) * 1000

            # Performance-Metriken aktualisieren
            self._monitoring_service.performance_monitor.track_response_time(endpoint, duration_ms)
            self._monitoring_service.performance_monitor.track_throughput(endpoint)

            # Status-Code-spezifische Metriken
            status_code = response.status_code
            self._monitoring_service.metrics_collector.increment_counter(
                "http_requests_total",
                labels={
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": str(status_code)
                }
            )

            # Error-Tracking
            if status_code >= 400:
                self._monitoring_service.metrics_collector.increment_counter(
                    "http_errors_total",
                    labels={
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": str(status_code)
                    }
                )

            # Voice-Workflow-spezifisches Tracking
            if self._is_voice_request(request):
                await self._track_voice_response(request, response, duration_ms)

            # Response-Header für Monitoring hinzufügen
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = str(round(duration_ms, 2))

            return response

        except Exception as e:
            # Exception-Tracking
            duration_ms = (time.time() - start_time) * 1000

            self._monitoring_service.metrics_collector.increment_counter(
                "http_exceptions_total",
                labels={
                    "method": method,
                    "endpoint": endpoint,
                    "exception_type": type(e).__name__
                }
            )

            # Voice-Workflow-Fehler tracken
            if self._is_voice_request(request):
                await self._track_voice_error(request, e)

            logger.error(f"Request {request_id} failed after {duration_ms:.1f}ms: {e}")
            raise

    def _get_endpoint_name(self, request: Request) -> str:
        """Extrahiert Endpoint-Namen aus Request."""
        path = request.url.path

        # Normalisiere Pfad für bessere Gruppierung
        if path.startswith("/api/v1/voice"):
            return "voice_api"
        if path.startswith("/api/v1/chat"):
            return "chat_api"
        if path.startswith("/api/v1/agents"):
            return "agents_api"
        if path.startswith("/monitoring"):
            return "monitoring_api"
        if path.startswith("/health"):
            return "health_check"
        # Entferne IDs und Parameter für bessere Gruppierung
        normalized_path = self._normalize_path(path)
        return normalized_path.replace("/", "_").strip("_") or "root"

    def _normalize_path(self, path: str) -> str:
        """Normalisiert Pfad durch Entfernung von IDs."""
        import re

        # Entferne UUIDs
        path = re.sub(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "/{id}", path)

        # Entferne numerische IDs
        path = re.sub(r"/\d+", "/{id}", path)

        return path

    def _is_voice_request(self, request: Request) -> bool:
        """Prüft ob Request ein Voice-Workflow ist."""
        path = request.url.path
        return (
            path.startswith("/api/v1/voice") or
            "voice" in path.lower() or
            request.headers.get("X-Voice-Workflow") == "true"
        )

    async def _start_voice_workflow_tracking(self, request: Request, request_id: str) -> None:
        """Startet Voice-Workflow-Tracking."""
        try:
            # Extrahiere User-ID und Session-ID aus Request
            user_id = self._extract_user_id(request)
            session_id = self._extract_session_id(request) or request_id

            # Voice-Workflow-Tracking starten
            await self._monitoring_service.voice_monitor.start_workflow_tracking(
                workflow_id=request_id,
                user_id=user_id,
                session_id=session_id
            )

            # Request-Kontext für spätere Verwendung speichern
            request.state.voice_workflow_id = request_id
            request.state.voice_user_id = user_id
            request.state.voice_session_id = session_id

        except Exception as e:
            logger.warning(f"Failed to start voice workflow tracking: {e}")

    async def _track_voice_response(self, request: Request, response: Response, _: float) -> None:
        """Trackt Voice-Response."""
        try:
            workflow_id = getattr(request.state, "voice_workflow_id", None)
            if not workflow_id:
                return

            # Erfolg basierend auf Status-Code bestimmen
            success = response.status_code < 400

            # Voice-Workflow abschließen
            await self._monitoring_service.voice_monitor.complete_workflow(
                workflow_id=workflow_id,
                success=success
            )

        except Exception as e:
            logger.warning(f"Failed to track voice response: {e}")

    async def _track_voice_error(self, request: Request, error: Exception) -> None:
        """Trackt Voice-Workflow-Fehler."""
        try:
            workflow_id = getattr(request.state, "voice_workflow_id", None)
            if not workflow_id:
                return

            # Voice-Workflow mit Fehler abschließen
            await self._monitoring_service.voice_monitor.complete_workflow(
                workflow_id=workflow_id,
                success=False,
                error=str(error)
            )

        except Exception as e:
            logger.warning(f"Failed to track voice error: {e}")

    def _extract_user_id(self, request: Request) -> str:
        """Extrahiert User-ID aus Request."""
        # Versuche verschiedene Quellen für User-ID

        # 1. Authorization Header (JWT)
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                # Vereinfachte JWT-Dekodierung (ohne Verifikation für Monitoring)
                import base64
                import json

                token = auth_header.replace("Bearer ", "")
                # JWT hat 3 Teile: header.payload.signature
                parts = token.split(".")
                if len(parts) >= 2:
                    # Payload dekodieren
                    payload = parts[1]
                    # Padding hinzufügen falls nötig
                    payload += "=" * (4 - len(payload) % 4)
                    decoded = base64.b64decode(payload)
                    jwt_data = json.loads(decoded)

                    # User-ID aus verschiedenen Claims extrahieren
                    user_id = (
                        jwt_data.get("sub") or
                        jwt_data.get("user_id") or
                        jwt_data.get("uid")
                    )
                    if user_id:
                        return str(user_id)
            except Exception:
                pass

        # 2. X-User-ID Header
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id

        # 3. Query Parameter
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id

        # 4. Request State (falls von anderem Middleware gesetzt)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return str(user_id)

        # 5. Fallback: Anonymous User
        return "anonymous"

    def _extract_session_id(self, request: Request) -> str | None:
        """Extrahiert Session-ID aus Request."""
        # Versuche verschiedene Quellen für Session-ID

        # 1. X-Session-ID Header
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return session_id

        # 2. Query Parameter
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id

        # 3. Request State
        session_id = getattr(request.state, "session_id", None)
        if session_id:
            return str(session_id)

        return None


def create_monitoring_middleware():
    """Factory-Funktion für Monitoring Middleware mit Dependency Injection.

    Returns:
        Middleware-Factory-Funktion
    """
    def middleware_factory(app):
        # Versuche Monitoring Service aus DI Container zu holen
        monitoring_service = None
        try:
            container = get_container()
            monitoring_service = container.resolve(IMonitoringService)
        except Exception as e:
            logger.warning(f"Could not resolve monitoring service for middleware: {e}")

        return MonitoringMiddleware(app, monitoring_service)

    return middleware_factory
