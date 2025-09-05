"""OIDC-Konfiguration für KEI-MCP API.

Diese Datei enthält die Konfiguration für OpenID Connect (OIDC) basierte
Authentifizierung in der KEI-MCP API.
"""

import os
from dataclasses import dataclass

from kei_logging import get_logger

logger = get_logger(__name__)


@dataclass
class OIDCSettings:
    """OIDC-Einstellungen für die KEI-MCP API."""

    # Authentifizierungsmodus
    auth_mode: str = "hybrid"  # oidc, static, hybrid

    # OIDC Provider Konfiguration
    issuer_url: str | None = None
    audience: str = "kei-mcp-api"
    required_scopes: list[str] | None = None

    # Cache-Einstellungen
    cache_ttl: int = 3600  # 1 Stunde
    jwks_cache_ttl: int = 3600  # 1 Stunde
    discovery_cache_ttl: int = 86400  # 24 Stunden

    # Statische Token-Einstellungen (Fallback)
    enable_static_tokens: bool = True
    static_tokens: list[str] = None

    # Development-Einstellungen
    enable_dev_tokens: bool = True  # Für lokale Entwicklung aktiviert
    dev_token: str = "dev-token-12345"

    def __post_init__(self):
        """Post-Initialisierung für Validierung."""
        if self.static_tokens is None:
            self.static_tokens = []

        # Validiere auth_mode
        valid_modes = ["oidc", "static", "hybrid"]
        if self.auth_mode not in valid_modes:
            logger.warning(f"Ungültiger auth_mode: {self.auth_mode}, verwende 'hybrid'")
            self.auth_mode = "hybrid"

        # Validiere OIDC-Konfiguration
        if self.auth_mode in ["oidc", "hybrid"] and not self.issuer_url:
            logger.warning("OIDC-Modus aktiviert aber issuer_url nicht konfiguriert")

        # Validiere statische Token-Konfiguration
        if self.auth_mode in ["static", "hybrid"] and not self.static_tokens and not self.enable_dev_tokens:
            logger.warning("Statischer Modus aktiviert aber keine Tokens konfiguriert")


def load_oidc_settings() -> OIDCSettings:
    """Lädt OIDC-Einstellungen aus Umgebungsvariablen.

    Returns:
        OIDC-Einstellungen
    """
    # Authentifizierungsmodus
    auth_mode = os.getenv("KEI_MCP_AUTH_MODE", "hybrid").lower()

    # OIDC Provider Konfiguration
    issuer_url = os.getenv("KEI_MCP_OIDC_ISSUER_URL")
    audience = os.getenv("KEI_MCP_OIDC_AUDIENCE", "kei-mcp-api")

    # Required Scopes
    scopes_str = os.getenv("KEI_MCP_OIDC_REQUIRED_SCOPES", "")
    required_scopes = [s.strip() for s in scopes_str.split(",") if s.strip()] if scopes_str else None

    # Cache-Einstellungen
    cache_ttl = int(os.getenv("KEI_MCP_OIDC_CACHE_TTL", "3600"))
    jwks_cache_ttl = int(os.getenv("KEI_MCP_OIDC_JWKS_CACHE_TTL", "3600"))
    discovery_cache_ttl = int(os.getenv("KEI_MCP_OIDC_DISCOVERY_CACHE_TTL", "86400"))

    # Statische Tokens
    enable_static_tokens = os.getenv("KEI_MCP_ENABLE_STATIC_TOKENS", "true").lower() == "true"

    static_tokens = []

    # Haupt-API-Token
    main_token = os.getenv("KEI_MCP_API_TOKEN") or os.getenv("EXTERNAL_MCP_API_TOKEN")
    if main_token:
        static_tokens.append(main_token)

    # Zusätzliche Tokens
    additional_tokens = os.getenv("KEI_MCP_ADDITIONAL_TOKENS", "").split(",")
    for token in additional_tokens:
        if token.strip():
            static_tokens.append(token.strip())

    # Development-Einstellungen
    environment = os.getenv("ENVIRONMENT", "production").lower()
    enable_dev_tokens = environment in ["development", "dev", "local"]
    dev_token = os.getenv("KEI_MCP_DEV_TOKEN", "dev-token-12345")

    settings = OIDCSettings(
        auth_mode=auth_mode,
        issuer_url=issuer_url,
        audience=audience,
        required_scopes=required_scopes,
        cache_ttl=cache_ttl,
        jwks_cache_ttl=jwks_cache_ttl,
        discovery_cache_ttl=discovery_cache_ttl,
        enable_static_tokens=enable_static_tokens,
        static_tokens=static_tokens,
        enable_dev_tokens=enable_dev_tokens,
        dev_token=dev_token
    )

    logger.info(f"OIDC-Einstellungen geladen: Modus={settings.auth_mode}, "
               f"OIDC={'✓' if settings.issuer_url else '✗'}, "
               f"Statische Tokens={len(settings.static_tokens)}, "
               f"Dev-Tokens={'✓' if settings.enable_dev_tokens else '✗'}")

    return settings


def get_example_oidc_configs() -> dict:
    """Gibt Beispiel-OIDC-Konfigurationen zurück.

    Returns:
        Dictionary mit Beispiel-Konfigurationen
    """
    return {
        "azure_ad": {
            "description": "Azure Active Directory",
            "issuer_url": "https://login.microsoftonline.com/{tenant-id}/v2.0",
            "audience": "api://kei-mcp-api",
            "required_scopes": ["kei-mcp.read", "kei-mcp.write"],
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "oidc",
                "KEI_MCP_OIDC_ISSUER_URL": "https://login.microsoftonline.com/{tenant-id}/v2.0",
                "KEI_MCP_OIDC_AUDIENCE": "api://kei-mcp-api",
                "KEI_MCP_OIDC_REQUIRED_SCOPES": "kei-mcp.read,kei-mcp.write"
            }
        },
        "auth0": {
            "description": "Auth0",
            "issuer_url": "https://{domain}.auth0.com/",
            "audience": "https://api.kei-mcp.example.com",
            "required_scopes": ["read:tools", "write:tools"],
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "oidc",
                "KEI_MCP_OIDC_ISSUER_URL": "https://{domain}.auth0.com/",
                "KEI_MCP_OIDC_AUDIENCE": "https://api.kei-mcp.example.com",
                "KEI_MCP_OIDC_REQUIRED_SCOPES": "read:tools,write:tools"
            }
        },
        "keycloak": {
            "description": "Keycloak",
            "issuer_url": "https://keycloak.example.com/realms/{realm}",
            "audience": "kei-mcp-client",
            "required_scopes": ["kei-mcp"],
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "oidc",
                "KEI_MCP_OIDC_ISSUER_URL": "https://keycloak.example.com/realms/{realm}",
                "KEI_MCP_OIDC_AUDIENCE": "kei-mcp-client",
                "KEI_MCP_OIDC_REQUIRED_SCOPES": "kei-mcp"
            }
        },
        "okta": {
            "description": "Okta",
            "issuer_url": "https://{domain}.okta.com/oauth2/default",
            "audience": "api://kei-mcp",
            "required_scopes": ["kei-mcp.access"],
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "oidc",
                "KEI_MCP_OIDC_ISSUER_URL": "https://{domain}.okta.com/oauth2/default",
                "KEI_MCP_OIDC_AUDIENCE": "api://kei-mcp",
                "KEI_MCP_OIDC_REQUIRED_SCOPES": "kei-mcp.access"
            }
        },
        "google": {
            "description": "Google Identity Platform",
            "issuer_url": "https://accounts.google.com",
            "audience": "{client-id}.apps.googleusercontent.com",
            "required_scopes": None,
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "oidc",
                "KEI_MCP_OIDC_ISSUER_URL": "https://accounts.google.com",
                "KEI_MCP_OIDC_AUDIENCE": "{client-id}.apps.googleusercontent.com"
            }
        },
        "hybrid_mode": {
            "description": "Hybrid Mode (OIDC + Static Tokens)",
            "issuer_url": "https://auth.example.com",
            "audience": "kei-mcp-api",
            "required_scopes": ["kei-mcp.access"],
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "hybrid",
                "KEI_MCP_OIDC_ISSUER_URL": "https://auth.example.com",
                "KEI_MCP_OIDC_AUDIENCE": "kei-mcp-api",
                "KEI_MCP_OIDC_REQUIRED_SCOPES": "kei-mcp.access",
                "KEI_MCP_API_TOKEN": "static-api-token-123",
                "KEI_MCP_ENABLE_STATIC_TOKENS": "true"
            }
        },
        "development": {
            "description": "Development Mode (nur statische Tokens)",
            "issuer_url": None,
            "audience": None,
            "required_scopes": None,
            "env_vars": {
                "KEI_MCP_AUTH_MODE": "static",
                "KEI_MCP_API_TOKEN": "dev-api-token-123",
                "KEI_MCP_ENABLE_STATIC_TOKENS": "true",
                "ENVIRONMENT": "development"
            }
        }
    }


# Globale OIDC-Einstellungen laden
OIDC_SETTINGS = load_oidc_settings()


__all__ = [
    "OIDC_SETTINGS",
    "OIDCSettings",
    "get_example_oidc_configs",
    "load_oidc_settings"
]
