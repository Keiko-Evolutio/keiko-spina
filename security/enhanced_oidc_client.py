# backend/security/enhanced_oidc_client.py
"""Erweiterte OIDC/OAuth2-Integration für Keiko Personal Assistant

Implementiert Service Account-basierte Authentifizierung, Token-Refresh,
automatische Token-Rotation und vollständige OIDC Discovery-Integration.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from urllib.parse import urljoin

import httpx
import jwt
from jwt import PyJWKClient

from kei_logging import get_logger
from observability import trace_function, trace_span

logger = get_logger(__name__)


class TokenType(str, Enum):
    """Token-Typen für verschiedene Authentifizierungsszenarien."""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    SERVICE_ACCOUNT = "service_account"
    CLIENT_CREDENTIALS = "client_credentials"


class GrantType(str, Enum):
    """OAuth2 Grant Types."""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
    SERVICE_ACCOUNT = "urn:ietf:params:oauth:grant-type:service-account"


@dataclass
class TokenInfo:
    """Informationen über einen Token."""
    token: str
    token_type: TokenType
    expires_at: datetime | None = None
    scopes: set[str] = field(default_factory=set)
    subject: str | None = None
    issuer: str | None = None
    audience: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Token abgelaufen ist."""
        if not self.expires_at:
            return False
        # 30 Sekunden Puffer für Token-Refresh
        return datetime.now(UTC) + timedelta(seconds=30) >= self.expires_at

    @property
    def expires_in_seconds(self) -> int | None:
        """Gibt verbleibende Gültigkeitsdauer in Sekunden zurück."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.now(UTC)
        return max(0, int(delta.total_seconds()))


@dataclass
class ServiceAccountConfig:
    """Konfiguration für Service Account-Authentifizierung."""
    client_id: str
    client_secret: str | None = None
    private_key: str | None = None
    private_key_id: str | None = None
    token_endpoint: str = ""
    scopes: set[str] = field(default_factory=set)
    subject: str | None = None  # Für Impersonation
    audience: str | None = None

    def validate(self) -> bool:
        """Validiert Service Account-Konfiguration."""
        if not self.client_id or not self.token_endpoint:
            return False

        # Entweder client_secret oder private_key muss vorhanden sein
        return bool(self.client_secret or self.private_key)


@dataclass
class OIDCDiscoveryDocument:
    """OIDC Discovery-Dokument."""
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str | None = None
    jwks_uri: str | None = None
    revocation_endpoint: str | None = None
    introspection_endpoint: str | None = None
    scopes_supported: list[str] = field(default_factory=list)
    response_types_supported: list[str] = field(default_factory=list)
    grant_types_supported: list[str] = field(default_factory=list)
    token_endpoint_auth_methods_supported: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OIDCDiscoveryDocument:
        """Erstellt Discovery-Dokument aus Dictionary."""
        return cls(
            issuer=data.get("issuer", ""),
            authorization_endpoint=data.get("authorization_endpoint", ""),
            token_endpoint=data.get("token_endpoint", ""),
            userinfo_endpoint=data.get("userinfo_endpoint"),
            jwks_uri=data.get("jwks_uri"),
            revocation_endpoint=data.get("revocation_endpoint"),
            introspection_endpoint=data.get("introspection_endpoint"),
            scopes_supported=data.get("scopes_supported", []),
            response_types_supported=data.get("response_types_supported", []),
            grant_types_supported=data.get("grant_types_supported", []),
            token_endpoint_auth_methods_supported=data.get("token_endpoint_auth_methods_supported", [])
        )


class EnhancedOIDCClient:
    """Erweiterte OIDC/OAuth2-Client-Implementierung."""

    def __init__(
        self,
        issuer_url: str,
        client_id: str,
        client_secret: str | None = None,
        timeout_seconds: int = 30
    ) -> None:
        """Initialisiert Enhanced OIDC Client.

        Args:
            issuer_url: OIDC Issuer URL
            client_id: OAuth2 Client ID
            client_secret: OAuth2 Client Secret (optional für public clients)
            timeout_seconds: HTTP-Timeout
        """
        self.issuer_url = issuer_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_seconds = timeout_seconds

        # Discovery und JWKS
        self._discovery_document: OIDCDiscoveryDocument | None = None
        self._jwks_client: PyJWKClient | None = None
        self._discovery_cache_ttl = 3600  # 1 Stunde
        self._discovery_cached_at: datetime | None = None

        # Token-Management
        self._token_cache: dict[str, TokenInfo] = {}
        self._refresh_tokens: dict[str, str] = {}
        self._token_refresh_tasks: dict[str, asyncio.Task] = {}

        # Service Accounts
        self._service_accounts: dict[str, ServiceAccountConfig] = {}

        # HTTP-Client
        self._http_client = httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self):
        """Async Context Manager Entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self.close()

    async def close(self) -> None:
        """Schließt HTTP-Client und stoppt Background-Tasks."""
        # Stoppe alle Token-Refresh-Tasks
        for task in self._token_refresh_tasks.values():
            task.cancel()

        if self._token_refresh_tasks:
            await asyncio.gather(*self._token_refresh_tasks.values(), return_exceptions=True)

        await self._http_client.aclose()

    @trace_function("oidc.discover")
    async def discover_endpoints(self, force_refresh: bool = False) -> OIDCDiscoveryDocument:
        """Führt OIDC Discovery durch.

        Args:
            force_refresh: Erzwingt Neuladen des Discovery-Dokuments

        Returns:
            OIDC Discovery-Dokument
        """
        # Prüfe Cache
        if (not force_refresh and
            self._discovery_document and
            self._discovery_cached_at and
            (datetime.now(UTC) - self._discovery_cached_at).total_seconds() < self._discovery_cache_ttl):
            return self._discovery_document

        discovery_url = urljoin(self.issuer_url + "/", ".well-known/openid_configuration")

        try:
            with trace_span("oidc.discovery_request", {"url": discovery_url}):
                response = await self._http_client.get(discovery_url)
                response.raise_for_status()

                discovery_data = response.json()
                self._discovery_document = OIDCDiscoveryDocument.from_dict(discovery_data)
                self._discovery_cached_at = datetime.now(UTC)

                # JWKS-Client initialisieren
                if self._discovery_document.jwks_uri:
                    self._jwks_client = PyJWKClient(
                        self._discovery_document.jwks_uri,
                        cache_ttl=3600
                    )

                logger.info(f"OIDC Discovery erfolgreich für {self.issuer_url}")
                return self._discovery_document

        except Exception as e:
            logger.exception(f"OIDC Discovery fehlgeschlagen für {self.issuer_url}: {e}")
            raise

    @trace_function("oidc.validate_token")
    async def validate_token(self, token: str, expected_audience: str | None = None) -> TokenInfo:
        """Validiert JWT-Token.

        Args:
            token: JWT-Token
            expected_audience: Erwartete Audience

        Returns:
            Token-Informationen
        """
        try:
            # Discovery sicherstellen
            if not self._discovery_document:
                await self.discover_endpoints()

            if not self._jwks_client:
                raise ValueError("JWKS-Client nicht verfügbar")

            # JWT-Header dekodieren
            jwt.get_unverified_header(token)

            # Signing-Key holen
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)

            # Token validieren
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                issuer=self.issuer_url,
                audience=expected_audience,
                options={"verify_exp": True, "verify_aud": expected_audience is not None}
            )

            # Token-Info erstellen
            expires_at = None
            if "exp" in payload:
                expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)

            scopes = set()
            if "scope" in payload:
                scopes = set(payload["scope"].split())
            elif "scp" in payload:
                scopes = set(payload["scp"]) if isinstance(payload["scp"], list) else {payload["scp"]}

            return TokenInfo(
                token=token,
                token_type=TokenType.ACCESS_TOKEN,
                expires_at=expires_at,
                scopes=scopes,
                subject=payload.get("sub"),
                issuer=payload.get("iss"),
                audience=payload.get("aud"),
                claims=payload
            )

        except Exception as e:
            logger.exception(f"Token-Validierung fehlgeschlagen: {e}")
            raise

    def register_service_account(self, name: str, config: ServiceAccountConfig) -> None:
        """Registriert Service Account-Konfiguration.

        Args:
            name: Service Account-Name
            config: Service Account-Konfiguration
        """
        if not config.validate():
            raise ValueError(f"Ungültige Service Account-Konfiguration für {name}")

        self._service_accounts[name] = config
        logger.info(f"Service Account {name} registriert")

    @trace_function("oidc.service_account_token")
    async def get_service_account_token(
        self,
        service_account_name: str,
        force_refresh: bool = False
    ) -> TokenInfo:
        """Holt Token für Service Account.

        Args:
            service_account_name: Name des Service Accounts
            force_refresh: Erzwingt Token-Refresh

        Returns:
            Service Account Token
        """
        config = self._service_accounts.get(service_account_name)
        if not config:
            raise ValueError(f"Service Account {service_account_name} nicht registriert")

        cache_key = f"sa_{service_account_name}"

        # Prüfe Cache
        if not force_refresh and cache_key in self._token_cache:
            token_info = self._token_cache[cache_key]
            if not token_info.is_expired:
                return token_info

        # Token anfordern
        if config.private_key:
            token_info = await self._get_jwt_bearer_token(config)
        else:
            token_info = await self._get_client_credentials_token(config)

        # Cache aktualisieren
        self._token_cache[cache_key] = token_info

        # Automatischen Refresh planen
        if token_info.expires_at:
            await self._schedule_token_refresh(cache_key, token_info)

        return token_info

    async def _get_client_credentials_token(self, config: ServiceAccountConfig) -> TokenInfo:
        """Holt Token mit Client Credentials Grant."""
        if not config.token_endpoint:
            discovery = await self.discover_endpoints()
            config.token_endpoint = discovery.token_endpoint

        data = {
            "grant_type": GrantType.CLIENT_CREDENTIALS.value,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }

        if config.scopes:
            data["scope"] = " ".join(config.scopes)

        if config.audience:
            data["audience"] = config.audience

        try:
            response = await self._http_client.post(
                config.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()

            token_data = response.json()

            expires_at = None
            if "expires_in" in token_data:
                expires_at = datetime.now(UTC) + timedelta(seconds=token_data["expires_in"])

            scopes = set()
            if "scope" in token_data:
                scopes = set(token_data["scope"].split())

            return TokenInfo(
                token=token_data["access_token"],
                token_type=TokenType.CLIENT_CREDENTIALS,
                expires_at=expires_at,
                scopes=scopes,
                subject=config.subject or config.client_id,
                issuer=self.issuer_url
            )

        except Exception as e:
            logger.exception(f"Client Credentials Token-Anfrage fehlgeschlagen: {e}")
            raise

    async def _get_jwt_bearer_token(self, config: ServiceAccountConfig) -> TokenInfo:
        """Holt Token mit JWT Bearer Grant."""
        # JWT-Assertion erstellen
        now = datetime.now(UTC)

        jwt_payload = {
            "iss": config.client_id,
            "sub": config.subject or config.client_id,
            "aud": config.token_endpoint or self.issuer_url,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            "jti": f"{config.client_id}_{int(time.time())}"
        }

        if config.audience:
            jwt_payload["aud"] = config.audience

        # JWT signieren
        jwt_headers = {"alg": "RS256"}
        if config.private_key_id:
            jwt_headers["kid"] = config.private_key_id

        assertion = jwt.encode(
            jwt_payload,
            config.private_key,
            algorithm="RS256",
            headers=jwt_headers
        )

        # Token-Request
        if not config.token_endpoint:
            discovery = await self.discover_endpoints()
            config.token_endpoint = discovery.token_endpoint

        data = {
            "grant_type": GrantType.JWT_BEARER.value,
            "assertion": assertion
        }

        if config.scopes:
            data["scope"] = " ".join(config.scopes)

        try:
            response = await self._http_client.post(
                config.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()

            token_data = response.json()

            expires_at = None
            if "expires_in" in token_data:
                expires_at = datetime.now(UTC) + timedelta(seconds=token_data["expires_in"])

            scopes = set()
            if "scope" in token_data:
                scopes = set(token_data["scope"].split())

            return TokenInfo(
                token=token_data["access_token"],
                token_type=TokenType.ACCESS_TOKEN,
                expires_at=expires_at,
                scopes=scopes,
                subject=config.subject or config.client_id,
                issuer=self.issuer_url
            )

        except Exception as e:
            logger.exception(f"JWT Bearer Token-Anfrage fehlgeschlagen: {e}")
            raise

    async def _schedule_token_refresh(self, cache_key: str, token_info: TokenInfo) -> None:
        """Plant automatischen Token-Refresh."""
        if not token_info.expires_at:
            return

        # Refresh 5 Minuten vor Ablauf planen
        refresh_at = token_info.expires_at - timedelta(minutes=5)
        delay = (refresh_at - datetime.now(UTC)).total_seconds()

        if delay > 0:
            # Stoppe vorherigen Refresh-Task
            if cache_key in self._token_refresh_tasks:
                self._token_refresh_tasks[cache_key].cancel()

            # Neuen Refresh-Task starten
            task = asyncio.create_task(self._refresh_token_after_delay(cache_key, delay))
            self._token_refresh_tasks[cache_key] = task

    async def _refresh_token_after_delay(self, cache_key: str, delay: float) -> None:
        """Führt Token-Refresh nach Verzögerung durch."""
        try:
            await asyncio.sleep(delay)

            # Service Account-Token refreshen
            if cache_key.startswith("sa_"):
                service_account_name = cache_key[3:]
                await self.get_service_account_token(service_account_name, force_refresh=True)
                logger.info(f"Token für Service Account {service_account_name} automatisch refreshed")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Automatischer Token-Refresh fehlgeschlagen für {cache_key}: {e}")
        finally:
            # Task aus Dictionary entfernen
            if cache_key in self._token_refresh_tasks:
                del self._token_refresh_tasks[cache_key]

    def get_cached_token(self, cache_key: str) -> TokenInfo | None:
        """Gibt gecachten Token zurück."""
        token_info = self._token_cache.get(cache_key)
        if token_info and not token_info.is_expired:
            return token_info
        return None

    def clear_token_cache(self, cache_key: str | None = None) -> None:
        """Leert Token-Cache."""
        if cache_key:
            self._token_cache.pop(cache_key, None)
            if cache_key in self._token_refresh_tasks:
                self._token_refresh_tasks[cache_key].cancel()
                del self._token_refresh_tasks[cache_key]
        else:
            self._token_cache.clear()
            for task in self._token_refresh_tasks.values():
                task.cancel()
            self._token_refresh_tasks.clear()

    def get_token_cache_stats(self) -> dict[str, Any]:
        """Gibt Token-Cache-Statistiken zurück."""
        total_tokens = len(self._token_cache)
        expired_tokens = sum(1 for token in self._token_cache.values() if token.is_expired)
        active_refresh_tasks = len(self._token_refresh_tasks)

        return {
            "total_tokens": total_tokens,
            "valid_tokens": total_tokens - expired_tokens,
            "expired_tokens": expired_tokens,
            "active_refresh_tasks": active_refresh_tasks,
            "service_accounts": len(self._service_accounts)
        }
