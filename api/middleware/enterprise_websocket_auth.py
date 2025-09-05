"""Enterprise WebSocket Authentication Middleware.

State-of-the-Art WebSocket-Authentifizierung mit umfassender Sicherheit,
Rate Limiting, Logging und OWASP-Compliance.
"""

import time
from collections import defaultdict
from datetime import UTC, datetime

from fastapi import WebSocket
from fastapi.security import HTTPBearer

from auth.enterprise_auth import auth
from config.websocket_auth_config import WEBSOCKET_AUTH_CONFIG
from kei_logging import get_logger

logger = get_logger(__name__)


class WebSocketRateLimiter:
    """Enterprise-grade Rate Limiter für WebSocket-Authentifizierung.

    Unterstützt Redis-basierten, verteilten Betrieb mit speicherbasiertem Fallback
    für Entwicklungsumgebungen. Verwendet die vereinheitlichte Rate-Limiting-
    Infrastruktur.
    """

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.attempts: dict[str, list] = defaultdict(list)
        self.blocked_ips: dict[str, float] = {}

        # Redis-basierte Rate Limiting Infrastruktur vorbereiten (lazy)
        self._redis_rl = None
        self._policy = None

    async def _ensure_backend(self):
        """Initialisiert Redis-Rate-Limiter und Policy falls konfiguriert."""
        if self._redis_rl is not None and self._policy is not None:
            return
        try:
            from config.unified_rate_limiting import (
                EndpointType,
                IdentificationStrategy,
                RateLimitAlgorithm,
                RateLimitPolicy,
                get_unified_rate_limit_config,
            )
            from services.redis_rate_limiter import RedisRateLimiter
            cfg = get_unified_rate_limit_config()
            if cfg.backend in (cfg.backend.REDIS, cfg.backend.HYBRID):
                # Erzeuge Redis Rate Limiter und eine dedizierte Policy für WS-Auth-Attempts
                self._redis_rl = RedisRateLimiter(cfg)
                # Sliding window für Auth-Versuche pro IP pro Fenster
                self._policy = RateLimitPolicy(
                    requests_per_minute=max(1, self.max_attempts),
                    window_size_seconds=self.window_seconds,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    endpoint_type=EndpointType.WEBSOCKET,
                    identification_strategy=IdentificationStrategy.IP_ADDRESS,
                )
        except Exception as e:
            logger.debug(f"Redis Rate Limiter für WS-Auth nicht initialisierbar, fallback auf Memory: {e}")
            self._redis_rl = None
            self._policy = None

    async def is_rate_limited(self, client_ip: str) -> bool:
        """Prüft ob Client rate-limited ist (Redis bevorzugt)."""
        await self._ensure_backend()
        current_time = time.time()

        # Redis-gestützte Prüfung
        if self._redis_rl and self._policy:
            key = f"ws_auth:{client_ip}"
            try:
                result = await self._redis_rl.check_rate_limit(key, self._policy, current_time)
                return not result.allowed
            except Exception as e:
                logger.warning(f"Fehler bei Redis Rate Limit Check, fallback auf Memory: {e}")
                # Fallback auf Memory unten

        # Memory-Variante
        if client_ip in self.blocked_ips:
            if current_time - self.blocked_ips[client_ip] < self.window_seconds:
                return True
            del self.blocked_ips[client_ip]

        self.attempts[client_ip] = [
            t for t in self.attempts[client_ip]
            if current_time - t < self.window_seconds
        ]
        if len(self.attempts[client_ip]) >= self.max_attempts:
            self.blocked_ips[client_ip] = current_time
            logger.warning(f"🚫 Rate limit exceeded for IP: {client_ip}")
            return True
        return False

    async def record_attempt(self, client_ip: str):
        """Registriert einen Authentifizierungsversuch (auch in Redis)."""
        await self._ensure_backend()
        current_time = time.time()

        if self._redis_rl and self._policy:
            key = f"ws_auth:{client_ip}"
            try:
                # Konsumiere einen Slot im Sliding Window, ohne zu blockieren
                await self._redis_rl._sliding_window_check(key, self._policy, current_time)
            except Exception as e:
                logger.debug(f"Redis record_attempt Fallback: {e}")
        # Immer auch Memory pflegen für lokale Statistiken
        self.attempts[client_ip].append(current_time)


class EnterpriseWebSocketAuthenticator:
    """Enterprise-grade WebSocket-Authentifizierung."""

    def __init__(self):
        self.rate_limiter = WebSocketRateLimiter()
        self.security = HTTPBearer(auto_error=False)
        self.failed_attempts: dict[str, int] = defaultdict(int)
        self.suspicious_ips: set[str] = set()

    def _get_client_ip(self, websocket: WebSocket) -> str:
        """Extrahiert Client-IP-Adresse."""
        # Prüfe X-Forwarded-For Header (Proxy-Support)
        forwarded_for = websocket.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Prüfe X-Real-IP Header
        real_ip = websocket.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback auf Client-IP
        if hasattr(websocket, "client") and websocket.client:
            return websocket.client.host

        return "unknown"

    def _extract_token(self, websocket: WebSocket) -> str | None:
        """Extrahiert Token aus WebSocket-Request."""
        # 1. Authorization Header prüfen
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # 2. Query Parameter prüfen
        if hasattr(websocket, "query_params"):
            token = websocket.query_params.get("token")
            if token:
                return token

            # Alternative Query Parameter
            access_token = websocket.query_params.get("access_token")
            if access_token:
                return access_token

        return None

    async def authenticate_websocket(self, websocket: WebSocket, endpoint: str) -> dict | None:
        """Enterprise-grade WebSocket-Authentifizierung.

        Args:
            websocket: WebSocket-Verbindung
            endpoint: WebSocket-Endpoint-Pfad

        Returns:
            User context dict if authenticated, None otherwise
        """
        client_ip = self._get_client_ip(websocket)

        # Rate Limiting prüfen
        if await self.rate_limiter.is_rate_limited(client_ip):
            # Versuche weiterhin zählen, um Security-Statistiken konsistent zu halten
            self.failed_attempts[client_ip] += 1
            logger.warning(f"🚫 Rate limit exceeded for WebSocket auth: {client_ip} -> {endpoint}")
            self._log_security_event("websocket_auth_rate_limited", {
                "client_ip": client_ip,
                "endpoint": endpoint,
                "failed_attempts": self.failed_attempts[client_ip]
            })
            return None

        # Token extrahieren
        token = self._extract_token(websocket)
        if not token:
            await self.rate_limiter.record_attempt(client_ip)
            logger.warning(f"🔑 WebSocket authentication failed - no token: {client_ip} -> {endpoint}")
            return None

        try:
            # Enterprise Auth System für Token-Validierung
            mock_request = type("MockRequest", (), {})()
            result = await auth.validator.validate(token, mock_request)

            if result.success and result.context:
                # Erfolgreiche Authentifizierung
                logger.info(f"✅ WebSocket authentication successful: {result.context.subject} -> {endpoint}")

                # Security Event Logging
                self._log_security_event("websocket_auth_success", {
                    "client_ip": client_ip,
                    "endpoint": endpoint,
                    "user_id": result.context.subject,
                    "scopes": [scope.value for scope in result.context.scopes],
                    "privilege_level": result.context.privilege.value
                })

                return {
                    "user_id": result.context.subject,
                    "scopes": [scope.value for scope in result.context.scopes],
                    "privilege_level": result.context.privilege.value,
                    "authenticated_at": time.time(),
                    "client_ip": client_ip,
                    "endpoint": endpoint
                }
            # Fehlgeschlagene Authentifizierung
            await self.rate_limiter.record_attempt(client_ip)
            self.failed_attempts[client_ip] += 1

            logger.warning(f"❌ WebSocket authentication failed: {client_ip} -> {endpoint} - {result.error}")

            # Security Event Logging
            self._log_security_event("websocket_auth_failure", {
                "client_ip": client_ip,
                "endpoint": endpoint,
                "error": result.error,
                "failed_attempts": self.failed_attempts[client_ip]
            })

            # Markiere verdächtige IPs
            if self.failed_attempts[client_ip] >= 3:
                self.suspicious_ips.add(client_ip)
                logger.error(f"🚨 Suspicious activity detected: {client_ip} - {self.failed_attempts[client_ip]} failed attempts")

            return None

        except Exception as e:
            await self.rate_limiter.record_attempt(client_ip)
            logger.error(f"💥 WebSocket authentication error: {client_ip} -> {endpoint} - {e}")

            # Security Event Logging
            self._log_security_event("websocket_auth_error", {
                "client_ip": client_ip,
                "endpoint": endpoint,
                "error": str(e)
            })

            return None

    def _log_security_event(self, event_type: str, details: dict):
        """Protokolliert Sicherheitsereignisse für SIEM-Integration."""
        security_event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "component": "websocket_auth",
            "severity": "HIGH" if "failure" in event_type or "error" in event_type else "INFO",
            "details": details
        }

        # Strukturiertes Logging für SIEM-Systeme
        logger.info(f"SECURITY_EVENT: {security_event}")

    def is_endpoint_protected(self, path: str) -> bool:
        """Prüft ob Endpoint Authentifizierung erfordert."""
        # System-Client-Pfade prüfen
        system_patterns = WEBSOCKET_AUTH_CONFIG.system_clients.path_patterns
        for pattern in system_patterns:
            if path.startswith(pattern.replace("*", "")):
                return not WEBSOCKET_AUTH_CONFIG.system_clients.bypass_auth

        # Standard-WebSocket-Pfade prüfen
        protected_paths = WEBSOCKET_AUTH_CONFIG.apply_to_paths
        for protected_path in protected_paths:
            if path.startswith(protected_path):
                return WEBSOCKET_AUTH_CONFIG.enabled

        return False

    def get_security_stats(self) -> dict:
        """Gibt Sicherheitsstatistiken zurück."""
        return {
            "total_failed_attempts": sum(self.failed_attempts.values()),
            "suspicious_ips": len(self.suspicious_ips),
            "blocked_ips": len(self.rate_limiter.blocked_ips),
            "rate_limited_ips": list(self.rate_limiter.blocked_ips.keys()),
            "suspicious_ip_list": list(self.suspicious_ips)
        }


# Globale Instanz für Enterprise WebSocket Authentication
enterprise_websocket_auth = EnterpriseWebSocketAuthenticator()


__all__ = [
    "EnterpriseWebSocketAuthenticator",
    "WebSocketRateLimiter",
    "enterprise_websocket_auth"
]
