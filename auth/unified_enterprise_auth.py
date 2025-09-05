"""Unified Enterprise Authentication System.

Konsolidiert alle Authentication-Systeme in eine einheitliche, enterprise-grade Lösung.
Integriert JWT, mTLS, OIDC, Static Tokens und WebSocket-Authentifizierung.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials

from kei_logging import get_logger

from .enterprise_auth import (
    AuthContext,
    AuthResult,
    EnterpriseAuthenticator,
    PrivilegeLevel,
    Scope,
    TokenType,
)

logger = get_logger(__name__)


class AuthenticationMode(Enum):
    """Authentifizierungsmodi für verschiedene Umgebungen."""
    DISABLED = "disabled"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    STRICT = "strict"


@dataclass
class UnifiedAuthConfig:
    """Konfiguration für das Unified Authentication System."""

    # Grundkonfiguration
    mode: AuthenticationMode = AuthenticationMode.DEVELOPMENT
    enabled: bool = True

    # WebSocket-spezifische Konfiguration
    websocket_auth_enabled: bool = False
    websocket_auth_required: bool = False

    # Fallback-Verhalten
    allow_development_tokens: bool = True
    allow_api_tokens: bool = True
    fallback_to_bypass: bool = True

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600

    # Logging
    log_auth_attempts: bool = True
    log_auth_failures: bool = True
    debug_mode: bool = False


class UnifiedEnterpriseAuth:
    """Unified Enterprise Authentication System.

    Konsolidiert alle Authentication-Funktionalitäten in einem System:
    - JWT/OIDC Authentication
    - mTLS Certificate Authentication
    - Static Token Authentication
    - WebSocket Authentication
    - Rate Limiting
    - Comprehensive Audit Logging
    """

    def __init__(self, config: UnifiedAuthConfig | None = None):
        """Initialisiert das Unified Authentication System."""
        self.config = config or self._load_config_from_environment()
        self.enterprise_auth = EnterpriseAuthenticator()

        # Rate Limiting Storage
        self._rate_limit_storage: dict[str, dict[str, float]] = {}

        logger.info(f"🔐 Unified Enterprise Auth initialisiert (Mode: {self.config.mode.value})")

    def _load_config_from_environment(self) -> UnifiedAuthConfig:
        """Lädt Konfiguration aus Umgebungsvariablen."""
        environment = os.getenv("ENVIRONMENT", "development").lower()

        # Bestimme Authentication Mode basierend auf Environment
        if environment == "production":
            mode = AuthenticationMode.PRODUCTION
            websocket_auth_enabled = True
            websocket_auth_required = True
            allow_development_tokens = False
            fallback_to_bypass = False
        elif environment == "strict":
            mode = AuthenticationMode.STRICT
            websocket_auth_enabled = True
            websocket_auth_required = True
            allow_development_tokens = False
            fallback_to_bypass = False
        else:
            mode = AuthenticationMode.DEVELOPMENT
            websocket_auth_enabled = os.getenv("WEBSOCKET_AUTH_ENABLED", "false").lower() == "true"
            websocket_auth_required = os.getenv("WEBSOCKET_AUTH_STRICT_MODE", "false").lower() == "true"
            allow_development_tokens = True
            fallback_to_bypass = os.getenv("WEBSOCKET_AUTH_FALLBACK_TO_BYPASS", "true").lower() == "true"

        return UnifiedAuthConfig(
            mode=mode,
            enabled=os.getenv("AUTH_ENABLED", "true").lower() == "true",
            websocket_auth_enabled=websocket_auth_enabled,
            websocket_auth_required=websocket_auth_required,
            allow_development_tokens=allow_development_tokens,
            allow_api_tokens=True,
            fallback_to_bypass=fallback_to_bypass,
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests=int(os.getenv("KEI_DEFAULT_RATE_LIMIT", "1000")),
            rate_limit_window=int(os.getenv("KEI_RATE_LIMIT_WINDOW", "3600")),
            log_auth_attempts=os.getenv("LOG_AUTH_ATTEMPTS", "true").lower() == "true",
            log_auth_failures=os.getenv("LOG_AUTH_FAILURES", "true").lower() == "true",
            debug_mode=os.getenv("AUTH_DEBUG_MODE", "false").lower() == "true"
        )

    async def authenticate_http(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None
    ) -> AuthResult:
        """Authentifiziert HTTP-Requests.

        Args:
            request: HTTP-Request
            credentials: Authorization Credentials

        Returns:
            AuthResult mit Authentifizierungsergebnis
        """
        if not self.config.enabled:
            return self._create_bypass_result("authentication_disabled")

        # Rate Limiting prüfen
        if self.config.rate_limit_enabled:
            await self._check_rate_limit(request)

        # Credentials prüfen
        if not credentials or not credentials.credentials:
            if self.config.fallback_to_bypass and self.config.mode == AuthenticationMode.DEVELOPMENT:
                return self._create_bypass_result("no_credentials_development")
            return AuthResult(success=False, error="Missing authentication credentials")

        token = credentials.credentials

        # Multi-Method Authentication
        auth_methods = [
            self._authenticate_enterprise_token,
            self._authenticate_development_token,
            self._authenticate_api_token
        ]

        for auth_method in auth_methods:
            try:
                result = await auth_method(token, request)
                if result.success:
                    if self.config.log_auth_attempts:
                        logger.info(f"🔐 HTTP Authentication erfolgreich: {result.context.subject}")
                    return result
            except Exception as e:
                if self.config.debug_mode:
                    logger.debug(f"Auth method {auth_method.__name__} failed: {e}")
                continue

        # Alle Authentifizierungsmethoden fehlgeschlagen
        if self.config.log_auth_failures:
            logger.warning(f"❌ HTTP Authentication fehlgeschlagen für: {request.url.path}")

        if self.config.fallback_to_bypass and self.config.mode == AuthenticationMode.DEVELOPMENT:
            return self._create_bypass_result("authentication_failed_fallback")

        return AuthResult(success=False, error="Authentication failed")

    async def authenticate_websocket(
        self,
        websocket: WebSocket,
        user_id: str | None = None
    ) -> AuthResult:
        """Authentifiziert WebSocket-Verbindungen.

        Args:
            websocket: WebSocket-Verbindung
            user_id: Optional User ID aus URL-Parameter

        Returns:
            AuthResult mit Authentifizierungsergebnis
        """
        if not self.config.websocket_auth_enabled:
            return self._create_bypass_result("websocket_auth_disabled")

        # Token aus WebSocket extrahieren
        token = await self._extract_websocket_token(websocket)

        if not token:
            if self.config.websocket_auth_required:
                return AuthResult(success=False, error="WebSocket authentication token required")
            if self.config.fallback_to_bypass:
                return self._create_bypass_result("websocket_no_token_fallback")
            return AuthResult(success=False, error="WebSocket authentication token missing")

        # Multi-Method Authentication für WebSocket
        auth_methods = [
            self._authenticate_enterprise_token,
            self._authenticate_development_token,
            self._authenticate_api_token
        ]

        for auth_method in auth_methods:
            try:
                # Create mock request for token validation
                mock_request = type("MockRequest", (), {})()
                result = await auth_method(token, mock_request)
                if result.success:
                    # Zusätzliche WebSocket-spezifische Validierung
                    if user_id and not self._validate_websocket_user_access(result.context, user_id):
                        return AuthResult(success=False, error="Insufficient privileges for requested user")

                    if self.config.log_auth_attempts:
                        logger.info(f"🔐 WebSocket Authentication erfolgreich: {result.context.subject}")
                    return result
            except Exception as e:
                if self.config.debug_mode:
                    logger.debug(f"WebSocket auth method {auth_method.__name__} failed: {e}")
                continue

        # Alle Authentifizierungsmethoden fehlgeschlagen
        if self.config.log_auth_failures:
            logger.warning("❌ WebSocket Authentication fehlgeschlagen")

        if self.config.fallback_to_bypass and not self.config.websocket_auth_required:
            return self._create_bypass_result("websocket_authentication_failed_fallback")

        return AuthResult(success=False, error="WebSocket authentication failed")

    # Helper Methods

    async def _authenticate_enterprise_token(self, token: str, request: Any) -> AuthResult:
        """Authentifiziert Token über Enterprise Auth System."""
        return await self.enterprise_auth.validator.validate(token, request)

    async def _authenticate_development_token(self, token: str, request: Any) -> AuthResult:
        """Authentifiziert Development Token."""
        if not self.config.allow_development_tokens:
            return AuthResult(success=False, error="Development tokens not allowed")

        # Support multiple dev token environment variables and hardcoded dev token
        dev_tokens = [
            os.getenv("KEIKO_DEV_TOKEN"),
            os.getenv("KEI_MCP_DEV_TOKEN"),
            "dev-token-12345"  # Hardcoded dev token for development
        ]
        dev_tokens = [t for t in dev_tokens if t]  # Remove None values

        if token in dev_tokens:
            context = AuthContext(
                subject="dev-user",
                scopes=[Scope.SYSTEM_READ, Scope.SYSTEM_WRITE, Scope.AGENTS_READ, Scope.AGENTS_WRITE],  # Limited scopes for dev
                privilege=PrivilegeLevel.ADMIN,
                token_type=TokenType.STATIC,
                issued_at=datetime.now(UTC),
                expires_at=None,
                token_id=f"dev-{int(time.time())}",
                metadata={"type": "development", "environment": "development"}
            )
            return AuthResult(success=True, context=context)

        return AuthResult(success=False, error="Invalid development token")

    async def _authenticate_api_token(self, token: str, request: Any) -> AuthResult:
        """Authentifiziert API Token."""
        if not self.config.allow_api_tokens:
            return AuthResult(success=False, error="API tokens not allowed")

        api_tokens = os.getenv("KEIKO_API_TOKENS", "").split(",")
        api_tokens = [t.strip() for t in api_tokens if t.strip()]

        if token in api_tokens:
            context = AuthContext(
                subject="api-user",
                scopes=[Scope.AGENTS_READ, Scope.AGENTS_WRITE],  # Standard API scopes
                privilege=PrivilegeLevel.USER,
                token_type=TokenType.STATIC,
                issued_at=datetime.now(UTC),
                expires_at=None,
                token_id=f"api-{int(time.time())}",
                metadata={"type": "api", "token_index": api_tokens.index(token)}
            )
            return AuthResult(success=True, context=context)

        return AuthResult(success=False, error="Invalid API token")

    async def _extract_websocket_token(self, websocket: WebSocket) -> str | None:
        """Extrahiert Token aus WebSocket Query-Parametern oder Headers."""
        try:
            # Query-Parameter prüfen
            if hasattr(websocket, "query_params") and websocket.query_params:
                token = websocket.query_params.get("token") or websocket.query_params.get("access_token")
                if token:
                    return token

            # Headers prüfen (falls verfügbar)
            if hasattr(websocket, "headers"):
                auth_header = websocket.headers.get("authorization")
                if auth_header and auth_header.lower().startswith("bearer "):
                    return auth_header[7:]  # Remove "Bearer " prefix

            return None
        except Exception as e:
            if self.config.debug_mode:
                logger.debug(f"Token extraction from WebSocket failed: {e}")
            return None

    def _validate_websocket_user_access(self, context: AuthContext, requested_user_id: str) -> bool:
        """Validiert ob User Zugriff auf angeforderte User ID hat."""
        # Admin/System-Benutzer können auf alle User IDs zugreifen
        admin_scopes = {Scope.ADMIN, Scope.SYSTEM}
        if admin_scopes.intersection(set(context.scopes)):
            return True

        # Normale Benutzer können nur auf ihre eigene User ID zugreifen
        return context.subject == requested_user_id

    def _create_bypass_result(self, reason: str) -> AuthResult:
        """Erstellt AuthResult für Authentication Bypass."""
        context = AuthContext(
            subject="bypass-user",
            scopes=[Scope.AGENTS_READ],  # Minimale Berechtigung für Bypass
            privilege=PrivilegeLevel.USER,
            token_type=TokenType.STATIC,
            issued_at=datetime.now(UTC),
            expires_at=None,
            token_id=f"bypass-{int(time.time())}",
            metadata={"type": "bypass", "reason": reason}
        )
        return AuthResult(success=True, context=context)

    async def _check_rate_limit(self, request: Request) -> None:
        """Prüft Rate Limiting für Request."""
        if not self.config.rate_limit_enabled:
            return

        # Client IP extrahieren
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Rate Limit Window prüfen
        if client_ip not in self._rate_limit_storage:
            self._rate_limit_storage[client_ip] = {}

        client_data = self._rate_limit_storage[client_ip]
        window_start = current_time - self.config.rate_limit_window

        # Alte Einträge entfernen (Fix: Konvertiere Timestamp-Keys zu float für Vergleich)
        client_data = {
            timestamp: count for timestamp, count in client_data.items()
            if float(timestamp) > window_start
        }
        self._rate_limit_storage[client_ip] = client_data

        # Aktuelle Requests zählen
        current_requests = sum(client_data.values())

        if current_requests >= self.config.rate_limit_requests:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip}: {current_requests}/{self.config.rate_limit_requests}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate Limit Exceeded",
                    "message": f"Too many requests. Limit: {self.config.rate_limit_requests} per {self.config.rate_limit_window} seconds",
                    "retry_after": self.config.rate_limit_window
                }
            )

        # Request zählen (Timestamp als String speichern für Konsistenz)
        timestamp_key = str(int(current_time))
        client_data[timestamp_key] = client_data.get(timestamp_key, 0) + 1

    def _get_client_ip(self, request: Request) -> str:
        """Extrahiert Client IP aus Request."""
        # Prüfe X-Forwarded-For Header (für Proxy/Load Balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Prüfe X-Real-IP Header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback auf Client Host
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"


# Globale Instanz des Unified Authentication Systems
unified_auth = UnifiedEnterpriseAuth()


# FastAPI Dependencies
async def require_unified_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = None
) -> AuthContext:
    """FastAPI Dependency für Unified Authentication.

    Robust: Extrahiert fehlende Credentials aus Header und respektiert bereits
    durch Middleware gesetzten Auth‑Kontext in request.state.auth_context.
    """
    # Fallback: Wenn keine Credentials injiziert wurden, versuche Header zu lesen
    if credentials is None:
        try:
            auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                from fastapi.security import HTTPAuthorizationCredentials
                credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        except Exception:
            credentials = None

    # Wenn Middleware bereits erfolgreich authentifiziert hat, verwende deren Kontext
    state_ctx = getattr(request.state, "auth_context", None)
    if state_ctx and isinstance(state_ctx, AuthContext):
        return state_ctx

    result = await unified_auth.authenticate_http(request, credentials)

    if not result.success:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authentication Required",
                "message": result.error,
                "type": "unified_authentication_error"
            }
        )

    return result.context


async def require_websocket_auth(
    websocket: WebSocket,
    user_id: str | None = None
) -> AuthContext:
    """FastAPI Dependency für WebSocket Authentication."""
    result = await unified_auth.authenticate_websocket(websocket, user_id)

    if not result.success:
        await websocket.close(code=4001, reason=result.error or "Authentication required")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "WebSocket Authentication Required",
                "message": result.error,
                "type": "websocket_authentication_error"
            }
        )

    return result.context
