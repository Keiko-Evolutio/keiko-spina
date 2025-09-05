"""Request Tracing Middleware - Refactored mit konsolidierten Utilities.

Implementiert Request-Tracing mit Performance-Logging unter Verwendung
der konsolidierten Middleware-Base-Klasse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kei_logging import get_logger, structured_msg

from ..common.api_constants import ConfigDefaults, HeaderNames

# Konsolidierte Utilities
from ..common.middleware_base import TracingBaseMiddleware

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.responses import Response

logger = get_logger(__name__)


class TracingMiddleware(TracingBaseMiddleware):
    """Request Tracing mit Performance-Logging.

    Refactored um konsolidierte Middleware-Base zu verwenden und
    Funktionskomplexität zu reduzieren.
    """

    def __init__(self, app, service_name: str = ConfigDefaults.SERVICE_NAME):
        """Initialisiert Tracing-Middleware.

        Args:
            app: FastAPI/Starlette-Anwendung
            service_name: Name des Services für Tracing
        """
        super().__init__(app, "tracing", service_name)

    async def _pre_process_request(self, request: Request) -> None:
        """Pre-Processing: Trace-ID generieren und Request-Kontext setzen.

        Args:
            request: HTTP-Request
        """
        trace_id = self._generate_trace_id()
        request.state.trace_id = trace_id

        # Trace-Attribute für OpenTelemetry setzen
        self._set_trace_attributes(request)

    async def _post_process_response(
        self,
        request: Request,
        response: Response,
        start_time: float
    ) -> None:
        """Post-Processing: Trace-Attribute setzen und Logging.

        Args:
            request: HTTP-Request
            response: HTTP-Response
            start_time: Request-Startzeit
        """
        # Trace-Attribute für Response setzen
        self._set_trace_attributes(request, response)

        # Trace-ID zu Response-Headers hinzufügen
        trace_id = getattr(request.state, "trace_id", "unknown")
        response.headers[HeaderNames.X_TRACE_ID] = trace_id

        # Performance-Logging
        duration_ms = self._get_request_duration_ms(request)
        self._log_request_completion(request, response.status_code, duration_ms)

    def _log_request_completion(
        self,
        request: Request,
        status_code: int,
        duration_ms: int,
        status: str = "success"
    ) -> None:
        """Loggt Request-Completion mit strukturierten Daten.

        Args:
            request: HTTP-Request
            status_code: HTTP-Status-Code
            duration_ms: Request-Dauer in Millisekunden
            status: Request-Status (default: "success")
        """
        trace_id = getattr(request.state, "trace_id", "unknown")
        tenant_id = self._extract_tenant_id(request)
        correlation_id = getattr(request.state, "correlation_id", trace_id)

        try:
            # Standard-Logging
            self.logger.info(structured_msg(
                "REST request completed",
                correlation_id=correlation_id,
                tenant=tenant_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                trace_id=trace_id
            ))

            # Logfire-Logging (falls verfügbar)
            self._log_to_logfire(request, status_code, duration_ms, correlation_id, tenant_id, trace_id)

        except Exception:
            # Fallback-Logging
            self.logger.info(
                f"{request.method} {request.url.path} [{status_code}] "
                f"{duration_ms}ms (trace: {trace_id})"
            )

    def _log_to_logfire(
        self,
        request,
        status_code: int,
        duration_ms: int,
        correlation_id: str,
        tenant_id: str,
        trace_id: str
    ) -> None:
        """Loggt Request an Logfire falls verfügbar."""
        try:
            from app.common.logfire_logger import log_api_request

            log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                tenant_id=tenant_id,
                trace_id=trace_id,
                user_agent=request.headers.get("user-agent", "unknown"),
                content_length=request.headers.get("content-length", 0)
            )
        except Exception:
            # Logfire-Fehler sollen Standard-Logging nicht beeinträchtigen
            pass
