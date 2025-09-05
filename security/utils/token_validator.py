# backend/security/utils/token_validator.py
"""Zentrale Token-Validierung für Keiko Personal Assistant

Konsolidiert JWT/Token-Validierung aus verschiedenen Security-Modulen
und bietet einheitliche Validierungs-API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import jwt
from jwt import PyJWKClient, PyJWKClientError

from kei_logging import get_logger
from observability import trace_function

from ..constants import CryptoConstants, SecurityErrorMessages, SecurityTimeouts

logger = get_logger(__name__)


class TokenType(str, Enum):
    """Typen von Tokens."""
    JWT_BEARER = "jwt_bearer"
    STATIC_API_TOKEN = "static_api_token"
    SERVICE_ACCOUNT = "service_account"
    REFRESH_TOKEN = "refresh_token"


class TokenValidationError(Exception):
    """Fehler bei Token-Validierung."""

    def __init__(self, message: str, error_code: str = "token_invalid"):
        """Initialisiert Token-Validierung-Fehler.

        Args:
            message: Fehlermeldung
            error_code: Fehler-Code
        """
        super().__init__(message)
        self.error_code = error_code


@dataclass
class TokenValidationResult:
    """Ergebnis einer Token-Validierung."""
    valid: bool
    token_type: TokenType
    subject: str | None = None
    issuer: str | None = None
    audience: str | None = None
    scopes: set[str] = None
    claims: dict[str, Any] = None
    expires_at: datetime | None = None
    error_message: str | None = None
    error_code: str | None = None

    def __post_init__(self):
        """Post-Initialisierung."""
        if self.scopes is None:
            self.scopes = set()
        if self.claims is None:
            self.claims = {}


class TokenValidator:
    """Zentrale Token-Validierung."""

    def __init__(self):
        """Initialisiert Token Validator."""
        self._jwks_clients: dict[str, PyJWKClient] = {}
        self._static_tokens: set[str] = set()
        self._validation_cache: dict[str, TokenValidationResult] = {}
        self._cache_ttl = SecurityTimeouts.TOKEN_VALIDATION_CACHE_TTL

    def add_jwks_client(self, issuer: str, jwks_uri: str) -> None:
        """Fügt JWKS-Client für Issuer hinzu.

        Args:
            issuer: Token-Issuer
            jwks_uri: JWKS-URI
        """
        self._jwks_clients[issuer] = PyJWKClient(jwks_uri)
        logger.info(f"JWKS-Client für {issuer} hinzugefügt")

    def add_static_token(self, token: str) -> None:
        """Fügt statisches Token hinzu.

        Args:
            token: Statisches Token
        """
        self._static_tokens.add(token)

    def remove_static_token(self, token: str) -> None:
        """Entfernt statisches Token.

        Args:
            token: Zu entfernendes Token
        """
        self._static_tokens.discard(token)

    @trace_function("token_validator.validate")
    async def validate_token(self, token: str, expected_audience: str | None = None) -> TokenValidationResult:
        """Validiert Token.

        Args:
            token: Zu validierendes Token
            expected_audience: Erwartete Audience

        Returns:
            Token-Validierung-Ergebnis
        """
        # Cache-Check
        cache_key = f"{token[:20]}:{expected_audience or 'none'}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # Versuche JWT-Validierung
            result = await self._validate_jwt_token(token, expected_audience)
            if result.valid:
                self._cache_result(cache_key, result)
                return result

            # Fallback: Statisches Token
            result = self.validate_static_token(token)
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            logger.exception(f"Token-Validierung fehlgeschlagen: {e}")
            return TokenValidationResult(
                valid=False,
                token_type=TokenType.JWT_BEARER,
                error_message=str(e),
                error_code="validation_error"
            )

    async def _validate_jwt_token(self, token: str, expected_audience: str | None = None) -> TokenValidationResult:
        """Validiert JWT-Token.

        Args:
            token: JWT-Token
            expected_audience: Erwartete Audience

        Returns:
            Validierung-Ergebnis
        """
        try:
            # Decode Header ohne Verifikation für Issuer
            unverified_header = jwt.get_unverified_header(token)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})

            issuer = unverified_payload.get("iss")
            if not issuer:
                raise TokenValidationError("Token enthält keinen Issuer", "missing_issuer")

            # JWKS-Client für Issuer finden
            jwks_client = self._jwks_clients.get(issuer)
            if not jwks_client:
                raise TokenValidationError(f"Kein JWKS-Client für Issuer {issuer}", "unknown_issuer")

            # Signing Key abrufen
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Token verifizieren
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[unverified_header.get("alg", CryptoConstants.JWT_ALGORITHM_RS256)],
                audience=expected_audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": bool(expected_audience),
                    "verify_iss": True
                }
            )

            # Scopes extrahieren
            scopes = set()
            if "scope" in payload:
                scopes = set(payload["scope"].split())
            elif "scopes" in payload:
                scopes = set(payload["scopes"])

            # Expires-At berechnen
            expires_at = None
            if "exp" in payload:
                expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)

            return TokenValidationResult(
                valid=True,
                token_type=TokenType.JWT_BEARER,
                subject=payload.get("sub"),
                issuer=payload.get("iss"),
                audience=payload.get("aud"),
                scopes=scopes,
                claims=payload,
                expires_at=expires_at
            )

        except jwt.ExpiredSignatureError:
            return TokenValidationResult(
                valid=False,
                token_type=TokenType.JWT_BEARER,
                error_message=SecurityErrorMessages.AUTH_TOKEN_EXPIRED,
                error_code="token_expired"
            )
        except jwt.InvalidAudienceError:
            return TokenValidationResult(
                valid=False,
                token_type=TokenType.JWT_BEARER,
                error_message="Invalid token audience",
                error_code="invalid_audience"
            )
        except jwt.InvalidTokenError as e:
            return TokenValidationResult(
                valid=False,
                token_type=TokenType.JWT_BEARER,
                error_message=f"Invalid JWT token: {e!s}",
                error_code="invalid_jwt"
            )
        except PyJWKClientError as e:
            return TokenValidationResult(
                valid=False,
                token_type=TokenType.JWT_BEARER,
                error_message=f"JWKS error: {e!s}",
                error_code="jwks_error"
            )

    def validate_static_token(self, token: str) -> TokenValidationResult:
        """Validiert statisches Token.

        Args:
            token: Statisches Token

        Returns:
            Validierung-Ergebnis
        """
        if token in self._static_tokens:
            return TokenValidationResult(
                valid=True,
                token_type=TokenType.STATIC_API_TOKEN,
                subject="static_token_user",
                scopes={"api:access"}
            )

        return TokenValidationResult(
            valid=False,
            token_type=TokenType.STATIC_API_TOKEN,
            error_message=SecurityErrorMessages.AUTH_INVALID_TOKEN,
            error_code="invalid_static_token"
        )

    def _get_cached_result(self, cache_key: str) -> TokenValidationResult | None:
        """Gibt gecachtes Validierung-Ergebnis zurück.

        Args:
            cache_key: Cache-Schlüssel

        Returns:
            Gecachtes Ergebnis oder None
        """
        # Einfacher Cache ohne Expiry-Check für diese Version
        return self._validation_cache.get(cache_key)

    def _cache_result(self, cache_key: str, result: TokenValidationResult) -> None:
        """Cached Validierung-Ergebnis.

        Args:
            cache_key: Cache-Schlüssel
            result: Validierung-Ergebnis
        """
        # Einfacher Cache - in Produktion würde hier TTL implementiert
        if len(self._validation_cache) > 1000:  # Einfache Cache-Größen-Begrenzung
            keys_to_remove = list(self._validation_cache.keys())[:100]
            for key in keys_to_remove:
                del self._validation_cache[key]

        self._validation_cache[cache_key] = result

    def clear_cache(self) -> None:
        """Leert Validierung-Cache."""
        self._validation_cache.clear()
        logger.info("Token-Validierung-Cache geleert")

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        return {
            "cache_size": len(self._validation_cache),
            "jwks_clients": len(self._jwks_clients),
            "static_tokens": len(self._static_tokens)
        }


__all__ = [
    "TokenType",
    "TokenValidationError",
    "TokenValidationResult",
    "TokenValidator",
]
