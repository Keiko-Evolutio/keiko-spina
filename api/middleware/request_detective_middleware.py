"""Request Detective Middleware - Temporäres Debugging-Tool
Analysiert eingehende Requests um herauszufinden, woher Alertmanager-Requests kommen.
"""

import logging
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestDetectiveMiddleware(BaseHTTPMiddleware):
    """Temporäres Middleware zum Debuggen von Request-Quellen."""

    def __init__(self, app, target_paths: list[str] = None):
        super().__init__(app)
        self.target_paths = target_paths or ["/api/v1/alerts/alertmanager"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Analysiert Requests und loggt Details für Ziel-Pfade."""
        # Prüfe ob Request zu unseren Ziel-Pfaden gehört
        if any(request.url.path.startswith(path) for path in self.target_paths):
            await self._log_request_details(request)

        # Normale Request-Verarbeitung
        response = await call_next(request)
        return response

    async def _log_request_details(self, request: Request) -> None:
        """Loggt detaillierte Request-Informationen."""
        # Basis-Request-Info
        logger.warning("🔍 REQUEST DETECTIVE - Unbekannter Request erkannt:")
        logger.warning(f"  📍 URL: {request.url}")
        logger.warning(f"  🔧 Method: {request.method}")
        logger.warning(f"  🌐 Client IP: {request.client.host if request.client else 'unknown'}")

        # Headers analysieren
        logger.warning("  📋 Headers:")
        for name, value in request.headers.items():
            # Wichtige Headers hervorheben
            if name.lower() in ["user-agent", "x-forwarded-for", "x-real-ip", "referer", "origin", "host"]:
                logger.warning(f"    🔑 {name}: {value}")
            else:
                logger.warning(f"    📄 {name}: {value}")

        # User-Agent analysieren
        user_agent = request.headers.get("user-agent", "")
        if user_agent:
            logger.warning("  🤖 User-Agent Analysis:")
            if "prometheus" in user_agent.lower():
                logger.warning("    ⚠️  PROMETHEUS detected in User-Agent!")
            elif "alertmanager" in user_agent.lower():
                logger.warning("    ⚠️  ALERTMANAGER detected in User-Agent!")
            elif "curl" in user_agent.lower():
                logger.warning("    🔧 CURL detected - likely manual/script request")
            elif "python" in user_agent.lower():
                logger.warning("    🐍 PYTHON detected - likely script/automation")
            elif "docker" in user_agent.lower():
                logger.warning("    🐳 DOCKER detected - likely container health check")
            elif "kubernetes" in user_agent.lower():
                logger.warning("    ☸️  KUBERNETES detected - likely k8s health check")
            else:
                logger.warning(f"    ❓ Unknown User-Agent: {user_agent}")

        # Forwarded Headers prüfen (für Proxy/Load Balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            logger.warning(f"  🔄 Request via Proxy/Load Balancer: {forwarded_for}")

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            logger.warning(f"  🎯 Real IP: {real_ip}")

        # Request Body (falls vorhanden)
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                # Body lesen (vorsichtig, da es nur einmal gelesen werden kann)
                body = await request.body()
                if body:
                    body_preview = body[:200].decode("utf-8", errors="ignore")
                    logger.warning(f"  📦 Body Preview: {body_preview}...")
                else:
                    logger.warning("  📦 Body: (empty)")
        except Exception as e:
            logger.warning(f"  📦 Body: (could not read: {e})")

        # Timing
        logger.warning(f"  ⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Mögliche Quellen identifizieren
        logger.warning("  🕵️ Possible Sources:")
        if "127.0.0.1" in str(request.client.host if request.client else ""):
            logger.warning("    🏠 LOCAL REQUEST - from same machine")

        if "prometheus" in user_agent.lower() or "alertmanager" in user_agent.lower():
            logger.warning("    📊 MONITORING SYSTEM - Prometheus/Alertmanager")

        if request.headers.get("host", "").startswith("localhost"):
            logger.warning("    🔧 DEVELOPMENT TOOL - localhost request")

        logger.warning("  " + "="*50)


def create_request_detective_middleware(target_paths: list[str] = None):
    """Factory-Funktion für das Request Detective Middleware."""
    def middleware_factory(app):
        return RequestDetectiveMiddleware(app, target_paths)
    return middleware_factory
