"""Header Extraction Utilities für Middleware.

Zentrale Utility-Klasse für einheitliche Header-Extraktion mit Validation
und Normalisierung für alle Middleware-Komponenten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from kei_logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

logger = get_logger(__name__)


@dataclass
class ExtractedHeaders:
    """Container für extrahierte Header-Werte."""

    # Authentifizierung
    jwt_token: str | None = None
    auth_header: str | None = None

    # Tenant-Informationen
    tenant_id: str | None = None

    # Scopes und Rollen
    scopes: list[str] = None
    roles: list[str] = None

    # Benutzer-Informationen
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None

    # Cache-relevante Header
    if_none_match: str | None = None
    cache_control: str | None = None

    # Sonstige
    content_type: str | None = None
    user_agent: str | None = None

    def __post_init__(self):
        """Initialisiert leere Listen."""
        if self.scopes is None:
            self.scopes = []
        if self.roles is None:
            self.roles = []


class HeaderExtractor:
    """Zentrale Header-Extraktion für Middleware.

    Bietet einheitliche Header-Extraktion mit Validation, Normalisierung
    und Fallback-Mechanismen für alle Middleware-Komponenten.
    """

    # Standard Header-Namen
    TENANT_HEADERS = ["X-Tenant-Id", "x-tenant-id", "x-tenant", "tenant"]
    SCOPE_HEADERS = ["X-Scopes", "x-scopes", "scopes"]
    AUTH_HEADERS = ["Authorization", "X-Auth-Token", "X-JWT-Token"]
    USER_ID_HEADERS = ["X-User-ID", "x-user-id", "user-id"]
    AGENT_ID_HEADERS = ["X-Agent-ID", "x-agent-id", "agent-id"]
    SESSION_ID_HEADERS = ["X-Session-ID", "x-session-id", "session-id"]

    def __init__(self, component_name: str = "middleware"):
        """Initialisiert Header Extractor.

        Args:
            component_name: Name der Komponente für Logging
        """
        self.component_name = component_name

    def extract_all_headers(self, request: Request) -> ExtractedHeaders:
        """Extrahiert alle relevanten Header aus einem Request.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            ExtractedHeaders mit allen extrahierten Werten
        """
        headers = ExtractedHeaders()

        # Authentifizierung
        headers.jwt_token = self.extract_jwt_token(request)
        headers.auth_header = self.extract_auth_header(request)

        # Tenant-Informationen
        headers.tenant_id = self.extract_tenant_id(request)

        # Scopes und Rollen
        headers.scopes = self.extract_scopes(request)
        headers.roles = self.extract_roles(request)

        # Benutzer-Informationen
        headers.user_id = self.extract_user_id(request)
        headers.agent_id = self.extract_agent_id(request)
        headers.session_id = self.extract_session_id(request)

        # Cache-relevante Header
        headers.if_none_match = self.extract_if_none_match(request)
        headers.cache_control = self.extract_cache_control(request)

        # Sonstige
        headers.content_type = self.extract_content_type(request)
        headers.user_agent = self.extract_user_agent(request)

        return headers

    def extract_jwt_token(self, request: Request, allow_query_param: bool = False) -> str | None:
        """Extrahiert JWT-Token aus verschiedenen Quellen.

        Args:
            request: FastAPI Request-Objekt
            allow_query_param: Ob Query-Parameter erlaubt sind

        Returns:
            JWT-Token oder None
        """
        # 1. Authorization Header prüfen (Standard)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token:
                return token

        # 2. Alternative Auth-Header prüfen
        for header_name in self.AUTH_HEADERS:
            header_value = request.headers.get(header_name)
            if header_value:
                # Bearer-Präfix entfernen falls vorhanden
                if header_value.startswith("Bearer "):
                    token = header_value.split(" ", 1)[1].strip()
                    if token:
                        return token
                # Direkter Token-Wert (aber nicht Basic Auth)
                elif not header_value.startswith("Basic "):
                    return header_value.strip()

        # 3. Query-Parameter prüfen (falls erlaubt)
        if allow_query_param:
            query_params = parse_qs(str(request.url.query))
            for param_name in ["token", "access_token", "jwt"]:
                token_values = query_params.get(param_name, [])
                if token_values and token_values[0].strip():
                    return token_values[0].strip()

        return None

    def extract_auth_header(self, request: Request) -> str | None:
        """Extrahiert Authorization Header.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Authorization Header-Wert oder None
        """
        return request.headers.get("Authorization")

    def extract_tenant_id(self, request: Request, default: str | None = None) -> str | None:
        """Extrahiert Tenant-ID aus verschiedenen Header-Quellen.

        Args:
            request: FastAPI Request-Objekt
            default: Standard-Wert falls kein Header gefunden

        Returns:
            Tenant-ID oder default-Wert
        """
        # Prüfe Request-State zuerst (von anderen Middleware gesetzt)
        forced_tenant = getattr(request.state, "forced_tenant", None)
        if forced_tenant:
            return forced_tenant

        # Prüfe verschiedene Header-Namen
        for header_name in self.TENANT_HEADERS:
            tenant_id = request.headers.get(header_name)
            if tenant_id and tenant_id.strip():
                return tenant_id.strip()

        return default

    def extract_scopes(self, request: Request) -> list[str]:
        """Extrahiert Scopes aus verschiedenen Quellen.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Liste von Scopes
        """
        scopes = []

        # 1. Aus Request-State (von Auth-Middleware gesetzt)
        user_payload = getattr(request.state, "user", {}) or {}
        state_scopes = user_payload.get("scopes") or user_payload.get("scope") or []

        if isinstance(state_scopes, str):
            scopes.extend([s.strip() for s in state_scopes.split() if s.strip()])
        elif isinstance(state_scopes, list):
            scopes.extend([str(s).strip() for s in state_scopes if str(s).strip()])

        # 2. Aus Header-Fallback
        if not scopes:
            for header_name in self.SCOPE_HEADERS:
                header_value = request.headers.get(header_name)
                if header_value:
                    scopes.extend([s.strip() for s in header_value.split() if s.strip()])
                    break

        return list(set(scopes))  # Duplikate entfernen

    def extract_roles(self, request: Request) -> list[str]:
        """Extrahiert Rollen aus Request-State.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Liste von Rollen
        """
        user_payload = getattr(request.state, "user", {}) or {}
        roles = user_payload.get("roles") or []

        if isinstance(roles, str):
            return [s.strip() for s in roles.split(",") if s.strip()]
        if isinstance(roles, list):
            return [str(r).strip() for r in roles if str(r).strip()]

        return []

    def extract_user_id(self, request: Request) -> str | None:
        """Extrahiert User-ID aus verschiedenen Quellen.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            User-ID oder None
        """
        # 1. Aus Request-State
        user_payload = getattr(request.state, "user", {}) or {}
        user_id = user_payload.get("sub") or user_payload.get("user_id")
        if user_id:
            return str(user_id).strip()

        # 2. Aus Headern
        for header_name in self.USER_ID_HEADERS:
            user_id = request.headers.get(header_name)
            if user_id and user_id.strip():
                return user_id.strip()

        return None

    def extract_agent_id(self, request: Request) -> str | None:
        """Extrahiert Agent-ID aus Headern.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Agent-ID oder None
        """
        for header_name in self.AGENT_ID_HEADERS:
            agent_id = request.headers.get(header_name)
            if agent_id and agent_id.strip():
                return agent_id.strip()

        return None

    def extract_session_id(self, request: Request) -> str | None:
        """Extrahiert Session-ID aus Headern.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Session-ID oder None
        """
        for header_name in self.SESSION_ID_HEADERS:
            session_id = request.headers.get(header_name)
            if session_id and session_id.strip():
                return session_id.strip()

        return None

    def extract_if_none_match(self, request: Request) -> str | None:
        """Extrahiert If-None-Match Header für Cache-Validierung.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            If-None-Match Header-Wert oder None
        """
        return request.headers.get("If-None-Match")

    def extract_cache_control(self, request: Request) -> str | None:
        """Extrahiert Cache-Control Header.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Cache-Control Header-Wert oder None
        """
        return request.headers.get("Cache-Control")

    def extract_content_type(self, request: Request) -> str | None:
        """Extrahiert Content-Type Header.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            Content-Type Header-Wert oder None
        """
        return request.headers.get("Content-Type")

    def extract_user_agent(self, request: Request) -> str | None:
        """Extrahiert User-Agent Header.

        Args:
            request: FastAPI Request-Objekt

        Returns:
            User-Agent Header-Wert oder None
        """
        return request.headers.get("User-Agent")


__all__ = ["ExtractedHeaders", "HeaderExtractor"]
