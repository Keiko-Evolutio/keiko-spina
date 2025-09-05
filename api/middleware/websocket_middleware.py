"""WebSocket Middleware - Konfigurierbare Authentifizierung für Standard-WebSocket-Verbindungen.

Unterstützt sowohl Auth-Bypass (Development) als auch JWT/mTLS-Authentifizierung (Production).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from kei_logging import get_logger

logger = get_logger(__name__)


class WebSocketMiddleware(BaseHTTPMiddleware):
    """Konfigurierbare Authentifizierung für Standard-WebSocket-Verbindungen.

    Unterstützt sowohl Auth-Bypass (Development) als auch JWT/mTLS-Authentifizierung (Production).
    KEI-Stream-WebSocket-Pfade (/stream/ws/) bleiben unverändert mit eigener Auth/mTLS.
    """

    # Standard-WebSocket-Pfade, die konfigurierbare Auth verwenden
    STANDARD_WEBSOCKET_PATHS = {"/ws/", "/websocket/", "/api/v1/voice/", "/api/voice/"}

    # KEI-Stream-WebSocket-Pfade, die KEINE konfigurierbare Auth verwenden (eigene Auth/mTLS)
    KEI_STREAM_WEBSOCKET_PATHS = {"/stream/ws/"}

    async def dispatch(self, request, call_next) -> Response:
        """Enterprise-grade WebSocket-Authentifizierung mit konfigurierbaren Sicherheitsrichtlinien."""
        import os

        # Sichere Pfad-Extraktion mit Null-Schutz
        path = getattr(request.url, "path", None) if hasattr(request, "url") else None

        if path is None:
            # Kein gültiger Pfad - keine Auth-Behandlung
            return await call_next(request)

        # Prüfe, ob es sich um eine WebSocket-Verbindung handelt
        is_websocket = (
            request.headers.get("upgrade", "").lower() == "websocket" and
            request.headers.get("connection", "").lower() == "upgrade"
        )

        if not is_websocket:
            # Keine WebSocket-Verbindung - normale Middleware-Verarbeitung
            return await call_next(request)

        # Prüfe, ob Pfad zu Standard-WebSocket-Pfaden gehört
        is_standard_websocket = any(path.startswith(ws_path) for ws_path in self.STANDARD_WEBSOCKET_PATHS)
        is_kei_stream = any(path.startswith(stream_path) for stream_path in self.KEI_STREAM_WEBSOCKET_PATHS)

        if is_kei_stream:
            # KEI-Stream-WebSocket-Pfade haben eigene Auth/mTLS - nicht überschreiben
            logger.debug(f"🔌 KEI-Stream WebSocket-Verbindung (eigene Auth): {path}")
            # Keine State-Änderungen für KEI-Stream-Pfade
            return await call_next(request)

        if is_standard_websocket:
            # Standard-WebSocket-Pfade: Konfigurierbare Authentifizierung
            environment = os.getenv("ENVIRONMENT", "development").lower()
            websocket_auth_enabled = os.getenv("WEBSOCKET_AUTH_ENABLED", "false").lower() == "true"

            if environment == "production" or websocket_auth_enabled:
                # Production oder explizit aktiviert: Authentifizierung erforderlich
                logger.debug(f"🔐 WebSocket-Authentifizierung erforderlich für: {path}")

                try:
                    # Use unified authentication system for WebSocket authentication
                    from auth.unified_enterprise_auth import unified_auth

                    # Extract token from request headers
                    token = None
                    auth_header = request.headers.get("authorization")
                    if auth_header and auth_header.lower().startswith("bearer "):
                        token = auth_header[7:]

                    if token:
                        from fastapi.security import HTTPAuthorizationCredentials
                        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
                        auth_result = await unified_auth.authenticate_http(request, credentials)

                        if auth_result.success:
                            logger.info(f"🔐 WebSocket-Authentifizierung erfolgreich: {path}")
                            request.state.websocket_authenticated = True
                            request.state.websocket_auth_method = "unified_auth"
                            request.state.websocket_user_id = auth_result.context.subject
                        else:
                            logger.warning(f"❌ WebSocket-Authentifizierung fehlgeschlagen: {path} - {auth_result.error}")
                            # Für WebSocket-Verbindungen: Verbindung mit 403 ablehnen
                            from fastapi.responses import Response
                            return Response(
                                status_code=401,
                                content="WebSocket Authentication Failed",
                                headers={"Connection": "close"}
                            )
                    else:
                        logger.warning(f"❌ WebSocket-Authentifizierung fehlgeschlagen: {path} - Kein Token")
                        # Für WebSocket-Verbindungen: Verbindung mit 401 ablehnen
                        from fastapi.responses import Response
                        return Response(
                            status_code=401,
                            content="WebSocket Authentication Required",
                            headers={"Connection": "close"}
                        )


                except Exception as e:
                    logger.error(f"❌ WebSocket-Authentifizierung Fehler für {path}: {e}")
                    # Für WebSocket-Verbindungen: Verbindung mit 500 ablehnen
                    from fastapi.responses import Response
                    return Response(
                        status_code=500,
                        content="WebSocket Authentication Error",
                        headers={"Connection": "close"}
                    )
            else:
                # Development Mode: Authentifizierung optional
                logger.debug(f"🔓 WebSocket-Verbindung (Development Mode): {path}")
                request.state.websocket_development_mode = True
                request.state.websocket_authenticated = False

        return await call_next(request)
