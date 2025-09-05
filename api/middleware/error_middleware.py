"""Error Handling und Security Headers Middleware."""

from collections.abc import Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import settings
from core.error_handler import GlobalErrorHandler
from kei_logging import get_logger

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Globale Exception-Behandlung."""

    def __init__(self, app, include_details: bool = False):
        super().__init__(app)
        self.include_details = include_details
        # Einheitlicher GlobalErrorHandler
        self._handler = GlobalErrorHandler(include_details=include_details)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except Exception as e:
            # Delegiere einheitlich an GlobalErrorHandler
            return await self._handler.handle_request_exception(request, e)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security Headers für alle Responses.

    Diese Middleware setzt standardisierte Security‑Header und – abhängig von
    der Umgebung – HSTS (HTTP Strict Transport Security). HSTS wird per
    Default in Production aktiviert und kann in Development über die Settings
    optional zugeschaltet werden.
    """

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'"
        ),
        # PRODUCTION SECURITY: Strict-Transport-Security für OWASP-Compliance
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload"
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        # HSTS nur setzen, wenn Umgebung dies erlaubt und HTTPS erkannt wird
        try:
            # Aktivierung: Production immer, sonst nur wenn explizit aktiviert
            hsts_allowed = settings.is_production or settings.hsts_enabled
            if hsts_allowed:
                # Direkt über Scheme prüfen
                try:
                    is_https = request.url.scheme.lower() == "https"
                except (AttributeError, ValueError):
                    is_https = False
                # Hinter Proxy: X-Forwarded-Proto berücksichtigen
                xf_proto = request.headers.get("x-forwarded-proto") or request.headers.get("X-Forwarded-Proto")
                if xf_proto and xf_proto.lower().strip() == "https":
                    is_https = True

                if (not settings.hsts_https_only) or is_https:
                    directives = [f"max-age={int(settings.hsts_max_age)}"]
                    if settings.hsts_include_subdomains:
                        directives.append("includeSubDomains")
                    if settings.hsts_preload:
                        directives.append("preload")
                    response.headers["Strict-Transport-Security"] = "; ".join(directives)
        except Exception as exc:
            # Sicherheitsheader sollen die Response nicht verhindern
            logger.debug(f"HSTS-Header konnte nicht gesetzt werden: {exc}")
        return response
