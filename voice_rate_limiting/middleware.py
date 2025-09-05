"""Voice Rate Limiting Middleware.
Integriert Rate Limiting in HTTP Requests und WebSocket-Verbindungen.
"""

import base64
import json
import time
from collections.abc import Callable

from fastapi import Request, Response, WebSocket
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger

from .interfaces import (
    IVoiceRateLimitMiddleware,
    IVoiceRateLimitService,
    RateLimitResult,
    UserTier,
    VoiceOperation,
    VoiceRateLimitContext,
)

logger = get_logger(__name__)


class VoiceRateLimitMiddleware(BaseHTTPMiddleware, IVoiceRateLimitMiddleware):
    """Voice Rate Limiting Middleware für HTTP und WebSocket.
    Automatische Integration von Rate Limiting in Voice-Endpoints.
    """

    def __init__(self, app, rate_limit_service: IVoiceRateLimitService):
        super().__init__(app)
        self.rate_limit_service = rate_limit_service
        self._enabled = True

        # WebSocket Connection Tracking
        self._websocket_slots: dict[str, str] = {}  # connection_id -> slot_id

        logger.info("Voice rate limiting middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet HTTP-Request mit Voice Rate Limiting."""
        if not self._enabled or not self.rate_limit_service.is_running:
            return await call_next(request)

        # Bestimme ob es sich um einen Voice-Request handelt
        operation = self._detect_voice_operation(request)
        if not operation:
            # Kein Voice-Request, normale Verarbeitung
            return await call_next(request)

        # Rate Limiting für Voice-Request
        try:
            result = await self.check_http_rate_limit(request, operation)

            if not result.allowed:
                return await self.handle_rate_limit_exceeded(result)

            # Request verarbeiten
            start_time = time.time()
            response = await call_next(request)

            # Rate Limit Headers hinzufügen
            for header_name, header_value in result.headers.items():
                response.headers[header_name] = header_value

            # Performance-Tracking
            duration_ms = (time.time() - start_time) * 1000
            response.headers["X-Processing-Time-MS"] = str(round(duration_ms, 2))

            return response

        except Exception as e:
            logger.error(f"Error in voice rate limiting middleware: {e}")
            # Bei Fehlern im Rate Limiting, Request trotzdem verarbeiten
            return await call_next(request)

    @staticmethod
    def _detect_voice_operation(request: Request) -> VoiceOperation | None:
        """Erkennt Voice-Operation basierend auf Request-Path."""
        path = request.url.path.lower()
        method = request.method.upper()

        # Voice API Endpoints
        if "/voice" in path:
            if "/speech-to-text" in path or "/stt" in path:
                return VoiceOperation.SPEECH_TO_TEXT
            if "/synthesis" in path or "/tts" in path:
                return VoiceOperation.VOICE_SYNTHESIS
            if "/stream" in path:
                return VoiceOperation.REALTIME_STREAMING
            if "/upload" in path and method == "POST":
                return VoiceOperation.AUDIO_UPLOAD
            if "/text" in path and method == "POST":
                return VoiceOperation.TEXT_INPUT
            if "/workflow" in path and method == "POST":
                return VoiceOperation.WORKFLOW_START
            # Generischer Voice-Request
            return VoiceOperation.SPEECH_TO_TEXT

        # Agent-Execution in Voice-Kontext
        if "/agents" in path and request.headers.get("X-Voice-Context") == "true":
            return VoiceOperation.AGENT_EXECUTION

        # Tool-Calls in Voice-Kontext
        if "/tools" in path and request.headers.get("X-Voice-Context") == "true":
            return VoiceOperation.TOOL_CALL

        return None

    async def check_http_rate_limit(self, request: Request, operation: VoiceOperation) -> RateLimitResult:
        """Prüft Rate Limit für HTTP Request."""
        context = self._create_rate_limit_context(request)

        # Rate Limit konsumieren (nicht nur prüfen)
        result = await self.rate_limit_service.rate_limiter.consume_rate_limit(
            operation=operation,
            context=context
        )

        logger.debug(
            f"Rate limit check for {operation.value}: "
            f"allowed={result.allowed}, remaining={result.remaining}, user={context.user_id}"
        )

        return result

    async def check_websocket_rate_limit(self, websocket: WebSocket, user_id: str) -> RateLimitResult:
        """Prüft Rate Limit für WebSocket Connection."""
        context = self._create_websocket_rate_limit_context(websocket, user_id)

        # Concurrent Connection Limit prüfen
        result = await self.rate_limit_service.rate_limiter.check_concurrent_limit(
            operation=VoiceOperation.WEBSOCKET_CONNECTION,
            context=context
        )

        logger.debug(
            f"WebSocket rate limit check: "
            f"allowed={result.allowed}, remaining={result.remaining}, user={user_id}"
        )

        return result

    async def handle_rate_limit_exceeded(self, result: RateLimitResult) -> JSONResponse:
        """Behandelt Rate Limit Überschreitung."""
        error_response = {
            "error": "Rate limit exceeded",
            "message": f"Too many requests for {result.operation.value if result.operation else 'this operation'}",
            "details": {
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "retry_after_seconds": result.retry_after_seconds
            }
        }

        # User-friendly Message für verschiedene Operationen
        if result.operation == VoiceOperation.SPEECH_TO_TEXT:
            error_response["user_message"] = "You've reached the limit for speech-to-text requests. Please wait before trying again."
        elif result.operation == VoiceOperation.VOICE_SYNTHESIS:
            error_response["user_message"] = "You've reached the limit for voice synthesis requests. Please wait before trying again."
        elif result.operation == VoiceOperation.WEBSOCKET_CONNECTION:
            error_response["user_message"] = "Too many active voice connections. Please close some connections before opening new ones."
        else:
            error_response["user_message"] = "You've reached the rate limit for this voice operation. Please wait before trying again."

        return JSONResponse(
            status_code=429,
            content=error_response,
            headers=result.headers
        )

    def _create_rate_limit_context(self, request: Request) -> VoiceRateLimitContext:
        """Erstellt Rate Limit Context aus HTTP Request."""
        # User-ID extrahieren
        user_id = self._extract_user_id(request)

        # Session-ID extrahieren
        session_id = self._extract_session_id(request)

        # IP-Adresse extrahieren
        ip_address = self._extract_ip_address(request)

        # User-Tier bestimmen
        user_tier = self._determine_user_tier(request, user_id)

        # Request-Metadaten
        request_size = self._estimate_request_size(request)

        return self._build_rate_limit_context(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_tier=user_tier,
            endpoint=request.url.path,
            request_size_bytes=request_size
        )

    def _create_websocket_rate_limit_context(self, websocket: WebSocket, user_id: str) -> VoiceRateLimitContext:
        """Erstellt Rate Limit Context für WebSocket."""
        # IP-Adresse aus WebSocket extrahieren
        ip_address = None
        if hasattr(websocket, "client") and websocket.client:
            ip_address = websocket.client.host

        # User-Tier bestimmen (vereinfacht)
        user_tier = UserTier.STANDARD  # Default, könnte aus DB geholt werden

        return self._build_rate_limit_context(
            user_id=user_id,
            ip_address=ip_address,
            user_tier=user_tier,
            endpoint="/websocket/voice"
        )

    def _build_rate_limit_context(
        self,
        user_id: str,
        ip_address: str | None = None,
        user_tier: UserTier = UserTier.STANDARD,
        endpoint: str = "/unknown",
        session_id: str | None = None,
        request_size_bytes: int | None = None
    ) -> VoiceRateLimitContext:
        """Erstellt VoiceRateLimitContext mit gemeinsamer Logik."""
        # System-Kontext
        current_load = None  # Könnte aus Monitoring-Service geholt werden
        peak_hours = self._is_peak_hours()

        return VoiceRateLimitContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_tier=user_tier,
            endpoint=endpoint,
            request_size_bytes=request_size_bytes,
            current_load=current_load,
            peak_hours=peak_hours
        )

    def _extract_user_id(self, request: Request) -> str:
        """Extrahiert User-ID aus Request."""
        # 1. Authorization Header (JWT)
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                # Vereinfachte JWT-Dekodierung



                token = auth_header.replace("Bearer ", "")
                parts = token.split(".")
                if len(parts) >= 2:
                    payload = parts[1]
                    payload += "=" * (4 - len(payload) % 4)
                    decoded = base64.b64decode(payload)
                    jwt_data = json.loads(decoded)

                    user_id = (
                        jwt_data.get("sub") or
                        jwt_data.get("user_id") or
                        jwt_data.get("uid")
                    )
                    if user_id:
                        return str(user_id)
            except (ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
                # JWT parsing failed, continue with other methods
                pass

        # 2. X-User-ID Header
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id

        # 3. Query Parameter
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id

        # 4. Request State
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return str(user_id)

        # 5. Fallback: IP-basierte ID
        ip_address = self._extract_ip_address(request)
        return f"ip:{ip_address}"

    @staticmethod
    def _extract_session_id(request: Request) -> str | None:
        """Extrahiert Session-ID aus Request."""
        # X-Session-ID Header
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return session_id

        # Query Parameter
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id

        # Request State
        session_id = getattr(request.state, "session_id", None)
        if session_id:
            return str(session_id)

        return None

    @staticmethod
    def _extract_ip_address(request: Request) -> str:
        """Extrahiert IP-Adresse aus Request."""
        # X-Forwarded-For Header (für Proxy/Load Balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # X-Real-IP Header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    @staticmethod
    def _determine_user_tier(request: Request, user_id: str) -> UserTier:
        """Bestimmt User-Tier."""
        # X-User-Tier Header
        tier_header = request.headers.get("X-User-Tier")
        if tier_header:
            try:
                return UserTier(tier_header.lower())
            except ValueError:
                pass

        # Fallback-Logik basierend auf User-ID
        if user_id.startswith("ip:"):
            return UserTier.ANONYMOUS
        if "enterprise" in user_id.lower():
            return UserTier.ENTERPRISE
        if "premium" in user_id.lower():
            return UserTier.PREMIUM
        return UserTier.STANDARD

    @staticmethod
    def _estimate_request_size(request: Request) -> int | None:
        """Schätzt Request-Größe."""
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass

        return None

    @staticmethod
    def _is_peak_hours() -> bool:
        """Prüft ob aktuell Peak Hours sind."""
        from datetime import datetime

        current_hour = datetime.now().hour
        # Peak Hours: 9 AM - 5 PM
        return 9 <= current_hour <= 17

    async def acquire_websocket_slot(self, websocket: WebSocket, user_id: str) -> tuple[bool, str | None]:
        """Erwirbt WebSocket-Slot für Concurrent Connection Limiting."""
        context = self._create_websocket_rate_limit_context(websocket, user_id)

        success, slot_id = await self.rate_limit_service.rate_limiter.acquire_concurrent_slot(
            operation=VoiceOperation.WEBSOCKET_CONNECTION,
            context=context
        )

        if success and slot_id:
            # Connection-ID für Tracking generieren
            connection_id = f"{user_id}:{id(websocket)}"
            self._websocket_slots[connection_id] = slot_id

            logger.debug(f"WebSocket slot acquired for user {user_id}: {slot_id}")

        return success, slot_id

    async def release_websocket_slot(self, websocket: WebSocket, user_id: str) -> None:
        """Gibt WebSocket-Slot frei."""
        connection_id = f"{user_id}:{id(websocket)}"

        if connection_id in self._websocket_slots:
            slot_id = self._websocket_slots[connection_id]
            context = self._create_websocket_rate_limit_context(websocket, user_id)

            await self.rate_limit_service.rate_limiter.release_concurrent_slot(
                operation=VoiceOperation.WEBSOCKET_CONNECTION,
                context=context,
                slot_id=slot_id
            )

            del self._websocket_slots[connection_id]
            logger.debug(f"WebSocket slot released for user {user_id}: {slot_id}")


def create_voice_rate_limit_middleware(rate_limit_service: IVoiceRateLimitService):
    """Factory-Funktion für Voice Rate Limiting Middleware.

    Args:
        rate_limit_service: Voice Rate Limiting Service

    Returns:
        Middleware-Factory-Funktion
    """
    def middleware_factory(app):
        return VoiceRateLimitMiddleware(app, rate_limit_service)

    return middleware_factory
