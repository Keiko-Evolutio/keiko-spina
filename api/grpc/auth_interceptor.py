"""Authentication Interceptor für KEI-RPC gRPC Server.

Implementiert Bearer Token Authentication mit optionaler OIDC/JWT-Validierung
und Scope-basierter Autorisierung. Nutzt BaseInterceptor für Code-Deduplizierung.
"""

from __future__ import annotations

from typing import Any

import grpc

from kei_logging import get_logger

from .base_interceptor import BaseInterceptor, ServicerContext, UnaryUnaryHandler
from .constants import AuthConfig, ErrorCodes, ErrorMessages, MetadataKeys

logger = get_logger(__name__)

# Optional OIDC/JWT Dependencies
try:
    import jwt
    from jwt import PyJWKClient

    JWT_AVAILABLE = True
except ImportError:
    jwt = None
    PyJWKClient = None
    JWT_AVAILABLE = False


class AuthInterceptor(BaseInterceptor):
    """Authentication Interceptor mit Bearer Token und optionaler OIDC/JWT-Validierung.

    Features:
    - Bearer Token Validation (statisch oder JWT)
    - OIDC/JWT mit JWKS-Validierung
    - Scope-basierte Autorisierung
    - Tenant-ID Extraktion
    """

    def __init__(self) -> None:
        """Initialisiert Authentication Interceptor."""
        super().__init__("Auth")

        # OIDC/JWT Client Setup
        self._jwks_client: Any | None = None
        self._setup_oidc_client()

    def _setup_oidc_client(self) -> None:
        """Initialisiert OIDC/JWT Client falls konfiguriert."""
        if not JWT_AVAILABLE or not AuthConfig.OIDC_ISSUER:
            self.logger.info("OIDC/JWT nicht konfiguriert - verwende statische Tokens")
            return

        try:
            if AuthConfig.OIDC_JWKS_URI:
                self._jwks_client = PyJWKClient(AuthConfig.OIDC_JWKS_URI)
                self.logger.info("OIDC/JWT Client initialisiert")
            else:
                self.logger.warning("OIDC_JWKS_URI nicht konfiguriert")
        except Exception as e:
            self.logger.exception(f"OIDC/JWT Client Setup fehlgeschlagen: {e}")

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeitet Unary-Unary Request mit Authentication.

        Args:
            request: gRPC Request
            context: gRPC Service Context
            behavior: Original Handler
            method_name: Name der gRPC-Methode

        Returns:
            Response vom Original Handler

        Raises:
            grpc.RpcError: Bei Authentication-Fehlern
        """
        # 1. Token extrahieren und validieren
        token = self._extract_bearer_token(context)
        if not token:
            self._abort_with_auth_error(context, ErrorCodes.AUTH_TOKEN_MISSING)

        # 2. Token validieren (statisch oder JWT)
        token_data = await self._validate_token(token)
        if not token_data:
            self._abort_with_auth_error(context, ErrorCodes.AUTH_TOKEN_INVALID)

        # 3. Scope-Autorisierung prüfen
        required_scope = self._get_required_scope(method_name)
        if required_scope and not self._has_required_scope(token_data, required_scope):
            self._abort_with_auth_error(context, ErrorCodes.AUTH_SCOPE_INSUFFICIENT)

        # 4. Metadata für nachgelagerte Services setzen
        self._set_auth_metadata(context, token_data)

        # 5. Original Handler ausführen
        return await behavior(request, context)

    def _extract_bearer_token(self, context: ServicerContext) -> str | None:
        """Extrahiert Bearer Token aus Authorization Header.

        Args:
            context: gRPC Service Context

        Returns:
            Token-String oder None
        """
        metadata = context.invocation_metadata() or []

        for key, value in metadata:
            if key.lower() == MetadataKeys.AUTHORIZATION.lower():
                if value.startswith(MetadataKeys.BEARER_PREFIX):
                    return value[len(MetadataKeys.BEARER_PREFIX) :].strip()

        return None

    async def _validate_token(self, token: str) -> dict | None:
        """Validiert Token (statisch oder JWT).

        Args:
            token: Token-String

        Returns:
            Token-Daten oder None bei ungültigem Token
        """
        # 1. Statische Token prüfen
        if token in AuthConfig.STATIC_TOKENS:
            return {
                "sub": "static-user",
                "scope": "kei.rpc.admin",  # Admin-Scope für statische Tokens
                "tenant_id": "default",
                "token_type": "static",
            }

        # 2. JWT-Validierung falls verfügbar
        if self._jwks_client and JWT_AVAILABLE:
            return await self._validate_jwt_token(token)

        return None

    async def _validate_jwt_token(self, token: str) -> dict | None:
        """Validiert JWT Token mit JWKS.

        Args:
            token: JWT Token-String

        Returns:
            JWT Claims oder None
        """
        try:
            # JWT Header dekodieren für Key-ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                self.logger.warning("JWT Token ohne Key-ID")
                return None

            # Signing Key von JWKS abrufen
            signing_key = self._jwks_client.get_signing_key(kid)

            # JWT validieren
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=AuthConfig.OIDC_ISSUER,
                audience=AuthConfig.OIDC_AUDIENCE,
                options={"verify_exp": True},
            )

            # Claims normalisieren
            return {
                "sub": claims.get("sub"),
                "scope": claims.get("scope", claims.get("scp", "")),
                "tenant_id": claims.get("tenant_id", "default"),
                "roles": claims.get("roles", []),
                "token_type": "jwt",
            }

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT Token abgelaufen")
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Ungültiger JWT Token: {e}")
        except Exception as e:
            self.logger.exception(f"JWT Validierung fehlgeschlagen: {e}")

        return None

    def _get_required_scope(self, method_name: str) -> str | None:
        """Ermittelt erforderlichen Scope für Methode.

        Args:
            method_name: Name der gRPC-Methode

        Returns:
            Erforderlicher Scope oder None
        """
        # Extrahiere Methoden-Namen aus vollständigem Pfad
        if "/" in method_name:
            method_name = method_name.split("/")[-1]

        return AuthConfig.REQUIRED_SCOPES.get(method_name)

    def _has_required_scope(self, token_data: dict, required_scope: str) -> bool:
        """Prüft ob Token erforderlichen Scope hat.

        Args:
            token_data: Token-Daten mit Scopes
            required_scope: Erforderlicher Scope

        Returns:
            True wenn Scope vorhanden
        """
        token_scopes = self._extract_scopes(token_data)

        # Admin-Scope hat Zugriff auf alles
        if "kei.rpc.admin" in token_scopes:
            return True

        # Spezifischen Scope prüfen
        return required_scope in token_scopes

    def _extract_scopes(self, token_data: dict) -> set[str]:
        """Extrahiert Scopes aus Token-Daten.

        Args:
            token_data: Token-Daten

        Returns:
            Set von Scopes
        """
        scopes = set()

        # Scope-String aufteilen
        scope_string = token_data.get("scope", "")
        if scope_string:
            scopes.update(scope_string.split())

        # Roles als Scopes behandeln
        roles = token_data.get("roles", [])
        if isinstance(roles, list):
            scopes.update(roles)

        return scopes

    def _set_auth_metadata(self, context: ServicerContext, token_data: dict) -> None:
        """Setzt Authentication-Metadata für nachgelagerte Services.

        Args:
            context: gRPC Service Context
            token_data: Validierte Token-Daten
        """
        try:
            # Scopes als Metadata setzen
            scopes = " ".join(self._extract_scopes(token_data))

            # Trailing Metadata für nachgelagerte Services
            metadata = [
                (MetadataKeys.SCOPES, scopes),
                (MetadataKeys.USER_ID, token_data.get("sub", "unknown")),
                (MetadataKeys.TENANT_ID, token_data.get("tenant_id", "default")),
            ]

            context.set_trailing_metadata(metadata)

        except Exception as e:
            self.logger.warning(f"Fehler beim Setzen der Auth-Metadata: {e}")

    def _abort_with_auth_error(self, context: ServicerContext, error_code: str) -> None:
        """Bricht Request mit Authentication-Fehler ab.

        Args:
            context: gRPC Service Context
            error_code: Error-Code
        """
        error_message = getattr(ErrorMessages, error_code, "Authentication Fehler")

        # Error-Metadata setzen
        context.set_trailing_metadata(
            [
                (MetadataKeys.ERROR_CODE, error_code),
                (MetadataKeys.ERROR_SEVERITY, "ERROR"),
            ]
        )

        # Request abbrechen
        if error_code in (ErrorCodes.AUTH_TOKEN_MISSING, ErrorCodes.AUTH_TOKEN_INVALID):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, error_message)
        elif error_code == ErrorCodes.AUTH_SCOPE_INSUFFICIENT:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, error_message)
        else:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, error_message)


__all__ = ["AuthInterceptor"]
