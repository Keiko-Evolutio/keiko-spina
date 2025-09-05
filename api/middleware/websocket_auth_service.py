"""WebSocket-Authentifizierungs-Service.

Zentrale Authentifizierungslogik für Standard-WebSocket-Endpoints
mit Unterstützung für JWT, mTLS und Hybrid-Authentifizierung.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

from config.websocket_auth_config import (
    WEBSOCKET_AUTH_CONFIG,
    WebSocketAuthConfig,
    WebSocketAuthMethod,
    WebSocketAuthMode,
)
from kei_logging import get_logger
from services.streaming.ws_mtls import (
    extract_client_certificate_from_ws_headers,
    validate_client_certificate,
)

if TYPE_CHECKING:
    from fastapi import Request

logger = get_logger(__name__)


class WebSocketAuthResult:
    """Ergebnis einer WebSocket-Authentifizierung."""

    def __init__(
        self,
        authenticated: bool,
        method: str | None = None,
        user_info: dict[str, Any] | None = None,
        error: str | None = None,
        should_bypass: bool = False
    ):
        self.authenticated = authenticated
        self.method = method  # "jwt", "mtls", "bypass"
        self.user_info = user_info or {}
        self.error = error
        self.should_bypass = should_bypass


class WebSocketAuthenticator:
    """WebSocket-Authentifizierungs-Service mit konfigurierbaren Methoden."""

    def __init__(self, config: WebSocketAuthConfig = None):
        self.config = config or WEBSOCKET_AUTH_CONFIG
        self.logger = logger

    def should_authenticate_path(self, path: str) -> bool:
        """Prüft, ob ein Pfad authentifiziert werden soll."""
        # Prüfe System-Client-Patterns (State-of-the-Art)
        if self._is_system_client_path(path):
            return False

        # Prüfe Ausschluss-Pfade (Legacy-Support)
        if any(path.startswith(exclude_path) for exclude_path in self.config.exclude_paths):
            return False

        # Prüfe Anwendungs-Pfade
        return any(path.startswith(apply_path) for apply_path in self.config.apply_to_paths)

    def should_authenticate_path_non_system(self, path: str) -> bool:
        """Prüft, ob ein Pfad authentifiziert werden soll (ohne System-Client-Ausschluss)."""
        # Prüfe Ausschluss-Pfade (Legacy-Support)
        if any(path.startswith(exclude_path) for exclude_path in self.config.exclude_paths):
            return False

        # Prüfe Anwendungs-Pfade
        return any(path.startswith(apply_path) for apply_path in self.config.apply_to_paths)

    def _is_system_client_path(self, path: str) -> bool:
        """Prüft, ob ein Pfad einem System-Client-Pattern entspricht."""
        # Extrahiere Client-ID aus dem Pfad
        client_id = self._extract_client_id_from_path(path)

        # Prüfe Client-ID-Patterns
        if client_id and self._matches_system_client_patterns(client_id):
            return True

        # Prüfe Pfad-Patterns
        return self._matches_system_path_patterns(path)

    def _extract_client_id_from_path(self, path: str) -> str | None:
        """Extrahiert Client-ID aus WebSocket-Pfad."""
        # Standard-Patterns: /ws/client/{client_id}, /ws/agent/{agent_id}
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "ws" and parts[1] in ("client", "agent"):
            return parts[2]
        return None

    def _matches_system_client_patterns(self, client_id: str) -> bool:
        """Prüft, ob Client-ID einem System-Client-Pattern entspricht."""
        for pattern in self.config.system_clients.client_id_patterns:
            if fnmatch.fnmatch(client_id, pattern):
                return True
        return False

    def _matches_system_path_patterns(self, path: str) -> bool:
        """Prüft, ob Pfad einem System-Pfad-Pattern entspricht."""
        for pattern in self.config.system_clients.path_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    async def authenticate_websocket_request(self, request: Request) -> WebSocketAuthResult:
        """Authentifiziert eine WebSocket-Anfrage basierend auf der Konfiguration.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            WebSocketAuthResult mit Authentifizierungsergebnis
        """
        path = getattr(request.url, "path", "")

        # System-Client-Behandlung (State-of-the-Art) - ZUERST prüfen
        if self._is_system_client_path(path):
            if self.config.system_clients.log_system_client_access:
                self.logger.info(f"🔧 System-Client erkannt und bypassed: {path}")
            return WebSocketAuthResult(
                authenticated=True,
                method="system_client_bypass",
                should_bypass=True,
                user_info={"client_type": "system", "path": path}
            )

        # Prüfe, ob Authentifizierung für diesen Pfad erforderlich ist
        if not self.should_authenticate_path_non_system(path):
            if self.config.log_auth_attempts:
                self.logger.debug(f"WebSocket-Auth übersprungen für Pfad: {path}")
            return WebSocketAuthResult(authenticated=True, method="bypass", should_bypass=True)

        # Wenn Auth deaktiviert ist, Bypass verwenden
        if not self.config.enabled or self.config.mode == WebSocketAuthMode.DISABLED:
            if self.config.log_auth_attempts:
                self.logger.debug(f"WebSocket-Auth deaktiviert, Bypass für: {path}")
            return WebSocketAuthResult(authenticated=True, method="bypass", should_bypass=True)

        # Authentifizierungsversuche durchführen
        auth_results = []

        # JWT-Authentifizierung versuchen
        if self.config.jwt.enabled and WebSocketAuthMethod.JWT in self.config.methods:
            jwt_result = await self._authenticate_jwt(request)
            auth_results.append(jwt_result)

            if jwt_result.authenticated:
                if self.config.log_auth_attempts:
                    self.logger.info(f"WebSocket-JWT-Auth erfolgreich für: {path}")
                return jwt_result

        # mTLS-Authentifizierung versuchen
        if self.config.mtls.enabled and WebSocketAuthMethod.MTLS in self.config.methods:
            mtls_result = await self._authenticate_mtls(request)
            auth_results.append(mtls_result)

            if mtls_result.authenticated:
                if self.config.log_auth_attempts:
                    self.logger.info(f"WebSocket-mTLS-Auth erfolgreich für: {path}")
                return mtls_result

        # Hybrid-Authentifizierung (JWT oder mTLS)
        if WebSocketAuthMethod.HYBRID in self.config.methods:
            for result in auth_results:
                if result.authenticated:
                    if self.config.log_auth_attempts:
                        self.logger.info(f"WebSocket-Hybrid-Auth erfolgreich ({result.method}) für: {path}")
                    return result

        # Alle Authentifizierungsversuche fehlgeschlagen
        if self.config.log_auth_failures:
            errors = [r.error for r in auth_results if r.error]
            self.logger.warning(f"WebSocket-Auth fehlgeschlagen für {path}: {', '.join(errors)}")

        # Fallback-Verhalten prüfen
        if self.config.mode == WebSocketAuthMode.OPTIONAL or self.config.fallback_to_bypass:
            if self.config.log_auth_attempts:
                self.logger.info(f"WebSocket-Auth fehlgeschlagen, Fallback zu Bypass für: {path}")
            return WebSocketAuthResult(authenticated=True, method="bypass", should_bypass=True)

        # Strenger Modus: Authentifizierung erforderlich
        error_msg = "WebSocket-Authentifizierung fehlgeschlagen"
        if auth_results:
            error_msg = f"WebSocket-Authentifizierung fehlgeschlagen: {auth_results[0].error}"

        return WebSocketAuthResult(authenticated=False, error=error_msg)

    async def _authenticate_jwt(self, request: Request) -> WebSocketAuthResult:
        """JWT-Authentifizierung für WebSocket-Anfrage."""
        try:
            # Token aus verschiedenen Quellen extrahieren
            token = self._extract_jwt_token(request)

            if not token:
                if self.config.jwt.required:
                    return WebSocketAuthResult(authenticated=False, error="JWT-Token fehlt")
                return WebSocketAuthResult(authenticated=False, error="JWT-Token optional aber nicht vorhanden")

            # Token validieren
            payload = self._validate_jwt_token(token)

            user_info = {
                "sub": payload.get("sub"),
                "exp": payload.get("exp"),
                "iat": payload.get("iat"),
                "scopes": payload.get("scopes", []),
                "roles": payload.get("roles", []),
                "token_type": "jwt"
            }

            return WebSocketAuthResult(
                authenticated=True,
                method="jwt",
                user_info=user_info
            )

        except Exception as e:
            error_msg = f"JWT-Validierung fehlgeschlagen: {e!s}"
            return WebSocketAuthResult(authenticated=False, error=error_msg)

    async def _authenticate_mtls(self, request: Request) -> WebSocketAuthResult:
        """mTLS-Authentifizierung für WebSocket-Anfrage."""
        try:
            # Client-Zertifikat aus Headers extrahieren
            headers = request.headers
            cert_info = extract_client_certificate_from_ws_headers(headers)

            if not cert_info:
                if self.config.mtls.required:
                    return WebSocketAuthResult(authenticated=False, error="Client-Zertifikat fehlt")
                return WebSocketAuthResult(authenticated=False, error="Client-Zertifikat optional aber nicht vorhanden")

            # Zertifikat validieren
            validation_result = validate_client_certificate(cert_info)

            if not validation_result.get("valid", False):
                error_msg = validation_result.get("error", "mTLS-Validierung fehlgeschlagen")
                return WebSocketAuthResult(authenticated=False, error=error_msg)

            user_info = {
                "sub": validation_result.get("subject"),
                "issuer": validation_result.get("issuer"),
                "serial_number": validation_result.get("serial_number"),
                "fingerprint": validation_result.get("fingerprint"),
                "token_type": "mtls"
            }

            return WebSocketAuthResult(
                authenticated=True,
                method="mtls",
                user_info=user_info
            )

        except Exception as e:
            error_msg = f"mTLS-Validierung fehlgeschlagen: {e!s}"
            return WebSocketAuthResult(authenticated=False, error=error_msg)

    def _extract_jwt_token(self, request: Request) -> str | None:
        """Extrahiert JWT-Token aus WebSocket-Request."""
        # 1. Authorization Header prüfen
        auth_header = request.headers.get(self.config.jwt.auth_header_name)
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ", 1)[1]

        # 2. Query-Parameter prüfen (falls erlaubt)
        if self.config.jwt.allow_query_token:
            query_params = parse_qs(str(request.url.query))
            token_values = query_params.get(self.config.jwt.token_query_param, [])
            if token_values:
                return token_values[0]

        # 3. Alternative Header-Namen prüfen
        for header_name in ["X-Auth-Token", "X-JWT-Token", "Authorization"]:
            header_value = request.headers.get(header_name)
            if header_value:
                # Bearer-Präfix entfernen falls vorhanden
                if header_value.startswith("Bearer "):
                    return header_value.split(" ", 1)[1]
                # Direkter Token-Wert
                if not header_value.startswith("Basic "):
                    return header_value

        return None

    def _validate_jwt_token(self, token: str) -> dict[str, Any]:
        """Validiert JWT-Token."""
        if not self.config.jwt.secret_key:
            raise ValueError("JWT-Secret-Key nicht konfiguriert")

        # JWT-Optionen konfigurieren
        options = {
            "verify_signature": True,
            "verify_exp": self.config.jwt.verify_exp,
            "verify_aud": self.config.jwt.verify_aud,
            "verify_iss": self.config.jwt.verify_iss,
        }

        # Token dekodieren und validieren
        try:
            return jwt.decode(
                token,
                self.config.jwt.secret_key,
                algorithms=[self.config.jwt.algorithm],
                audience=self.config.jwt.audience if self.config.jwt.verify_aud else None,
                issuer=self.config.jwt.issuer if self.config.jwt.verify_iss else None,
                options=options,
                leeway=self.config.jwt.leeway
            )


        except ExpiredSignatureError:
            raise ValueError("JWT-Token abgelaufen")
        except JWTError as e:
            raise ValueError(f"Ungültiger JWT-Token: {e!s}")


# Globale Authenticator-Instanz
websocket_authenticator = WebSocketAuthenticator()


__all__ = [
    "WebSocketAuthResult",
    "WebSocketAuthenticator",
    "websocket_authenticator",
]
