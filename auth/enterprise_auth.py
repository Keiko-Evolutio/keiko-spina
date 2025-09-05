"""Enterprise-Grade Unified Authentication System.

Konsolidiert alle Authentication-Funktionalitäten in einem einzigen,
enterprise-grade Modul mit Clean Code Prinzipien.

Features:
- JWT und Static Token-Validierung
- Rate Limiting mit konfigurierbaren Limits
- Scope-basierte Authorization
- Umfassende Audit-Logging
- Type-Safe Interfaces
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kei_logging import get_logger

logger = get_logger(__name__)

# Konfiguration aus Environment Variables
JWT_SECRET = os.getenv("KEI_JWT_SECRET")
JWT_ALGORITHM = os.getenv("KEI_JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("KEI_JWT_ISSUER", "keiko-platform")
JWT_AUDIENCE = os.getenv("KEI_JWT_AUDIENCE", "keiko-api")
DEFAULT_RATE_LIMIT = int(os.getenv("KEI_DEFAULT_RATE_LIMIT", "1000"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("KEI_RATE_LIMIT_WINDOW", "3600"))
STATIC_TOKENS_FILE = os.getenv("KEI_STATIC_TOKENS_FILE", "config/static_tokens.json")


class TokenType(Enum):
    """Token-Typen für verschiedene Authentication-Methoden."""
    JWT = "jwt"
    STATIC = "static"
    OIDC = "oidc"


class PrivilegeLevel(Enum):
    """Privilege-Level für hierarchische Authorization."""
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"


class Scope(Enum):
    """Verfügbare Scopes für granulare Permissions.

    Hinweis: Zusätzlich zu domänenspezifischen Scopes (z. B. "agents:read")
    werden auch generische Scopes "read"/"write" unterstützt, um
    Kompatibilität mit generischen Tests/Clients zu gewährleisten.
    """
    # Domänenspezifische Scopes
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    AGENTS_ADMIN = "agents:admin"
    VOICE_READ = "voice:read"
    VOICE_WRITE = "voice:write"
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"

    # Generische Scopes (Kompatibilität)
    READ = "read"
    WRITE = "write"

    # Privileg-/Rollen-Scoping
    SYSTEM = "system"  # Hinzugefügt für System-Level-Zugriff
    ADMIN = "admin"    # Hinzugefügt für Admin-Level-Zugriff


@dataclass
class AuthContext:
    """Authentication-Kontext mit allen relevanten Informationen."""
    token_type: TokenType
    subject: str
    privilege: PrivilegeLevel
    scopes: list[Scope]
    token_id: str
    expires_at: datetime | None = None
    issued_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AuthResult:
    """Ergebnis einer Authentication-Operation."""
    success: bool
    context: AuthContext | None = None
    error: str | None = None
    rate_limit_remaining: int | None = None


class TokenValidator:
    """Unified Token-Validierung für alle Token-Typen."""

    def __init__(self):
        """Initialisiert den Token-Validator mit Konfiguration."""
        self.rate_limits: dict[str, list[float]] = {}
        self.static_tokens = self._load_static_tokens()
        logger.info("Token-Validator initialisiert mit %d statischen Tokens", len(self.static_tokens))

    def _load_static_tokens(self) -> dict[str, dict[str, Any]]:
        """Lädt statische Tokens aus Konfigurationsdatei."""
        try:
            if os.path.exists(STATIC_TOKENS_FILE):
                with open(STATIC_TOKENS_FILE, encoding="utf-8") as f:
                    tokens_config = json.load(f)
                    return {
                        token: {
                            "privilege": PrivilegeLevel(info["privilege"]),
                            "scopes": [Scope(s) for s in info["scopes"]],
                            "rate_limit": info.get("rate_limit", DEFAULT_RATE_LIMIT),
                            "description": info.get("description", ""),
                            "created_at": datetime.fromisoformat(info.get("created_at", "2024-01-01T00:00:00"))
                        }
                        for token, info in tokens_config.items()
                    }
        except Exception as e:
            logger.warning("Fehler beim Laden der statischen Tokens: %s", e)
        return {}

    def _check_rate_limit(self, token_id: str, limit: int = DEFAULT_RATE_LIMIT) -> bool:
        """Prüft Rate Limit für Token."""
        current_time = time.time()
        window_start = current_time - RATE_LIMIT_WINDOW_SECONDS

        # Bereinige alte Einträge
        if token_id in self.rate_limits:
            self.rate_limits[token_id] = [
                timestamp for timestamp in self.rate_limits[token_id]
                if timestamp > window_start
            ]
        else:
            self.rate_limits[token_id] = []

        # Prüfe Limit
        if len(self.rate_limits[token_id]) >= limit:
            return False

        # Füge aktuellen Request hinzu
        self.rate_limits[token_id].append(current_time)
        return True

    def _is_jwt_format(self, token: str) -> bool:
        """Prüft ob Token JWT-Format hat."""
        return token.count(".") == 2

    async def validate_jwt(self, token: str) -> AuthResult:
        """Validiert JWT-Token."""
        if not JWT_SECRET:
            return AuthResult(success=False, error="JWT nicht konfiguriert")

        try:
            payload = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                issuer=JWT_ISSUER,
                audience=JWT_AUDIENCE
            )

            context = AuthContext(
                token_type=TokenType.JWT,
                subject=payload.get("sub", "unknown"),
                privilege=PrivilegeLevel(payload.get("privilege", "user")),
                scopes=[Scope(s) for s in payload.get("scopes", ["agents:read"])],
                token_id=payload.get("jti", token[:16]),
                expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC) if "exp" in payload else None,
                issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC) if "iat" in payload else None
            )

            # Rate Limiting prüfen
            if not self._check_rate_limit(context.token_id):
                return AuthResult(success=False, error="Rate Limit überschritten")

            return AuthResult(success=True, context=context)

        except jwt.ExpiredSignatureError:
            return AuthResult(success=False, error="Token abgelaufen")
        except jwt.InvalidTokenError as e:
            return AuthResult(success=False, error=f"Ungültiger JWT: {e}")
        except Exception as e:
            logger.exception("JWT-Validierung fehlgeschlagen: %s", e)
            return AuthResult(success=False, error="JWT-Validierung fehlgeschlagen")

    def validate_static_token(self, token: str) -> AuthResult:
        """Validiert statischen Token."""
        token_info = self.static_tokens.get(token)
        if not token_info:
            return AuthResult(success=False, error="Ungültiger Token")

        token_id = hashlib.sha256(token.encode()).hexdigest()[:16]

        # Rate Limiting prüfen
        if not self._check_rate_limit(token_id, token_info["rate_limit"]):
            return AuthResult(success=False, error="Rate Limit überschritten")

        context = AuthContext(
            token_type=TokenType.STATIC,
            subject=f"static-{token[:8]}",
            privilege=token_info["privilege"],
            scopes=token_info["scopes"],
            token_id=token_id,
            issued_at=token_info["created_at"],
            metadata={"description": token_info["description"]}
        )

        return AuthResult(success=True, context=context)

    async def validate(self, token: str, request: Request = None) -> AuthResult:
        """Unified Token-Validierung für alle Token-Typen."""
        try:
            # JWT-Validierung versuchen
            if self._is_jwt_format(token):
                result = await self.validate_jwt(token)
                if result.success:
                    return result

            # Static Token-Validierung versuchen
            return self.validate_static_token(token)

        except Exception as e:
            logger.exception("Token-Validierung fehlgeschlagen: %s", e)
            return AuthResult(success=False, error="Validierung fehlgeschlagen")


class EnterpriseAuthenticator:
    """Enterprise-Grade Authentication-System."""

    def __init__(self):
        """Initialisiert den Enterprise-Authenticator."""
        self.validator = TokenValidator()
        self.bearer_security = HTTPBearer(auto_error=False)
        logger.info("Enterprise-Authenticator initialisiert")

    async def authenticate(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None
    ) -> AuthResult:
        """Authentifiziert einen Request."""
        if not credentials or not credentials.credentials:
            return AuthResult(success=False, error="Fehlende Authentifizierung")

        return await self.validator.validate(credentials.credentials, request)

    def check_scope(self, context: AuthContext, required_scopes: list[Scope]) -> bool:
        """Prüft ob Kontext erforderliche Scopes hat."""
        return all(scope in context.scopes for scope in required_scopes)

    def check_privilege(self, context: AuthContext, required_privilege: PrivilegeLevel) -> bool:
        """Prüft ob Kontext erforderliches Privilege-Level hat."""
        privilege_hierarchy = {
            PrivilegeLevel.USER: 1,
            PrivilegeLevel.ADMIN: 2,
            PrivilegeLevel.SYSTEM: 3
        }
        return privilege_hierarchy[context.privilege] >= privilege_hierarchy[required_privilege]


# Globale Authenticator-Instanz
auth = EnterpriseAuthenticator()


# FastAPI Dependencies
async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(auth.bearer_security)
) -> AuthContext:
    """FastAPI Dependency für Authentication."""
    result = await auth.authenticate(request, credentials)

    if not result.success:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authentication fehlgeschlagen",
                "message": result.error,
                "type": "authentication_error"
            }
        )

    return result.context


def require_scope(required_scopes: list[Scope]):
    """Erstellt Scope-Requirement Dependency."""
    async def scope_check(context: AuthContext = Depends(require_auth)) -> AuthContext:
        if not auth.check_scope(context, required_scopes):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Unzureichende Berechtigung",
                    "required": [s.value for s in required_scopes],
                    "current": [s.value for s in context.scopes]
                }
            )
        return context
    return scope_check


def require_privilege(required_privilege: PrivilegeLevel):
    """Erstellt Privilege-Requirement Dependency."""
    async def privilege_check(context: AuthContext = Depends(require_auth)) -> AuthContext:
        if not auth.check_privilege(context, required_privilege):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Unzureichendes Privilege-Level",
                    "required": required_privilege.value,
                    "current": context.privilege.value
                }
            )
        return context
    return privilege_check


__all__ = [
    "AuthContext",
    "AuthResult",
    "EnterpriseAuthenticator",
    "PrivilegeLevel",
    "Scope",
    "TokenType",
    "TokenValidator",
    "auth",
    "require_auth",
    "require_privilege",
    "require_scope"
]
