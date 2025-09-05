"""Konfiguration für WebSocket-Authentifizierung.

Umfassende Konfigurationsstruktur für Standard-WebSocket-Endpoints
mit flexiblen Authentifizierungsoptionen (JWT, mTLS, Hybrid).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from kei_logging import get_logger

logger = get_logger(__name__)


class WebSocketAuthMethod(Enum):
    """Unterstützte WebSocket-Authentifizierungsmethoden."""

    JWT = "jwt"
    MTLS = "mtls"
    HYBRID = "hybrid"  # Sowohl JWT als auch mTLS


class WebSocketAuthMode(Enum):
    """WebSocket-Authentifizierungsmodus."""

    DISABLED = "disabled"  # Auth-Bypass (Development)
    OPTIONAL = "optional"  # Auth optional, Fallback erlaubt
    REQUIRED = "required"  # Auth erforderlich (Production)


@dataclass
class WebSocketJWTConfig:
    """JWT-Konfiguration für WebSocket-Authentifizierung."""

    enabled: bool = True
    required: bool = False  # JWT optional oder erforderlich
    secret_key: str | None = None
    algorithm: str = "HS256"

    # Header-Konfiguration für WebSocket
    auth_header_name: str = "Authorization"  # Standard Authorization Header
    token_query_param: str = "token"  # Alternative: Token als Query-Parameter
    allow_query_token: bool = True  # Token über Query-Parameter erlauben

    # Validierungsoptionen
    verify_exp: bool = True  # Token-Ablauf prüfen
    verify_aud: bool = False  # Audience prüfen (optional)
    verify_iss: bool = False  # Issuer prüfen (optional)
    audience: str | None = None
    issuer: str | None = None

    # Leeway für Zeitvalidierung (Sekunden)
    leeway: int = 30


@dataclass
class WebSocketMTLSConfig:
    """mTLS-Konfiguration für WebSocket-Authentifizierung."""

    enabled: bool = False
    required: bool = False  # mTLS optional oder erforderlich

    # Client-Zertifikat-Validierung
    verify_client_certs: bool = True
    client_ca_path: str | None = None

    # Header-Konfiguration (für Proxy-Setup)
    cert_header_name: str = "X-Client-Cert"

    # Subject-Whitelist (optional)
    allowed_client_subjects: list[str] | None = None

    # Zertifikat-Validierungsoptionen
    check_expiry: bool = True
    check_revocation: bool = False  # CRL/OCSP-Prüfung (optional)


@dataclass
class SystemClientConfig:
    """State-of-the-Art Konfiguration für System-Clients."""

    # System Client Patterns (unterstützt Wildcards)
    client_id_patterns: list[str] = field(default_factory=lambda: [
        "system_heartbeat_client",
        "system_monitor_client",
        "system_health_client",
        "internal_service_*"
    ])

    # System Client Pfad-Patterns (unterstützt Wildcards)
    path_patterns: list[str] = field(default_factory=lambda: [
        "/ws/client/system_*",
        "/ws/agent/system_*",
        "/ws/internal/*"
    ])

    # Bypass-Verhalten für System-Clients
    bypass_auth: bool = True
    bypass_authentication: bool = True
    bypass_authorization: bool = True
    bypass_rate_limiting: bool = True

    # Logging für System-Clients
    log_system_client_access: bool = True


@dataclass
class WebSocketAuthConfig:
    """State-of-the-Art WebSocket-Authentifizierungs-Konfiguration."""

    # Hauptkonfiguration
    enabled: bool = False  # Auth-Bypass für Development (Standard)
    mode: WebSocketAuthMode = WebSocketAuthMode.DISABLED
    methods: list[WebSocketAuthMethod] = field(default_factory=lambda: [WebSocketAuthMethod.JWT])

    # Fallback-Verhalten
    fallback_to_bypass: bool = True  # Bei Auth-Fehlern auf Bypass zurückfallen
    strict_mode: bool = False  # Strenger Modus ohne Fallback

    # System Client Konfiguration (State-of-the-Art)
    system_clients: SystemClientConfig = field(default_factory=SystemClientConfig)

    # Spezifische Auth-Konfigurationen
    jwt: WebSocketJWTConfig = field(default_factory=WebSocketJWTConfig)
    mtls: WebSocketMTLSConfig = field(default_factory=WebSocketMTLSConfig)

    # Pfad-spezifische Konfiguration
    apply_to_paths: list[str] = field(default_factory=lambda: ["/ws/", "/websocket/"])
    exclude_paths: list[str] = field(default_factory=list)  # Ersetzt durch system_clients

    # Logging und Debugging
    log_auth_attempts: bool = True
    log_auth_failures: bool = True
    debug_mode: bool = False


def load_websocket_auth_config() -> WebSocketAuthConfig:
    """Lädt WebSocket-Auth-Konfiguration aus Umgebungsvariablen.

    Returns:
        Vollständige WebSocket-Auth-Konfiguration
    """
    # Hauptkonfiguration
    enabled = os.getenv("WEBSOCKET_AUTH_ENABLED", "false").lower() == "true"

    mode_str = os.getenv("WEBSOCKET_AUTH_MODE", "disabled").lower()
    try:
        mode = WebSocketAuthMode(mode_str)
    except ValueError:
        logger.warning(f"Ungültiger WebSocket-Auth-Modus: {mode_str}, verwende 'disabled'")
        mode = WebSocketAuthMode.DISABLED

    # Authentifizierungsmethoden
    methods_str = os.getenv("WEBSOCKET_AUTH_METHODS", "jwt")
    methods = []
    for method_str in methods_str.split(","):
        method_str = method_str.strip().lower()
        try:
            methods.append(WebSocketAuthMethod(method_str))
        except ValueError:
            logger.warning(f"Ungültige WebSocket-Auth-Methode: {method_str}")

    if not methods:
        methods = [WebSocketAuthMethod.JWT]  # Fallback

    # Fallback-Konfiguration
    fallback_to_bypass = os.getenv("WEBSOCKET_AUTH_FALLBACK_TO_BYPASS", "true").lower() == "true"
    strict_mode = os.getenv("WEBSOCKET_AUTH_STRICT_MODE", "false").lower() == "true"

    # JWT-Konfiguration
    jwt_config = WebSocketJWTConfig(
        enabled=WebSocketAuthMethod.JWT in methods or WebSocketAuthMethod.HYBRID in methods,
        required=os.getenv("WEBSOCKET_JWT_REQUIRED", "false").lower() == "true",
        secret_key=os.getenv("WEBSOCKET_JWT_SECRET_KEY") or os.getenv("JWT_SECRET"),
        algorithm=os.getenv("WEBSOCKET_JWT_ALGORITHM", "HS256"),
        auth_header_name=os.getenv("WEBSOCKET_JWT_HEADER_NAME", "Authorization"),
        token_query_param=os.getenv("WEBSOCKET_JWT_QUERY_PARAM", "token"),
        allow_query_token=os.getenv("WEBSOCKET_JWT_ALLOW_QUERY_TOKEN", "true").lower() == "true",
        verify_exp=os.getenv("WEBSOCKET_JWT_VERIFY_EXP", "true").lower() == "true",
        verify_aud=os.getenv("WEBSOCKET_JWT_VERIFY_AUD", "false").lower() == "true",
        verify_iss=os.getenv("WEBSOCKET_JWT_VERIFY_ISS", "false").lower() == "true",
        audience=os.getenv("WEBSOCKET_JWT_AUDIENCE"),
        issuer=os.getenv("WEBSOCKET_JWT_ISSUER"),
        leeway=int(os.getenv("WEBSOCKET_JWT_LEEWAY_SECONDS", "30"))
    )

    # mTLS-Konfiguration
    mtls_config = WebSocketMTLSConfig(
        enabled=WebSocketAuthMethod.MTLS in methods or WebSocketAuthMethod.HYBRID in methods,
        required=os.getenv("WEBSOCKET_MTLS_REQUIRED", "false").lower() == "true",
        verify_client_certs=os.getenv("WEBSOCKET_MTLS_VERIFY_CLIENT_CERTS", "true").lower() == "true",
        client_ca_path=os.getenv("WEBSOCKET_MTLS_CLIENT_CA_PATH"),
        cert_header_name=os.getenv("WEBSOCKET_MTLS_CERT_HEADER_NAME", "X-Client-Cert"),
        check_expiry=os.getenv("WEBSOCKET_MTLS_CHECK_EXPIRY", "true").lower() == "true",
        check_revocation=os.getenv("WEBSOCKET_MTLS_CHECK_REVOCATION", "false").lower() == "true"
    )

    # Subject-Whitelist für mTLS
    allowed_subjects_str = os.getenv("WEBSOCKET_MTLS_ALLOWED_SUBJECTS", "")
    if allowed_subjects_str:
        mtls_config.allowed_client_subjects = [
            subject.strip() for subject in allowed_subjects_str.split(",") if subject.strip()
        ]

    # Pfad-Konfiguration
    apply_to_paths_str = os.getenv("WEBSOCKET_AUTH_APPLY_TO_PATHS", "/ws/,/websocket/,/api/v1/voice/,/api/voice/")
    apply_to_paths = [path.strip() for path in apply_to_paths_str.split(",") if path.strip()]

    exclude_paths_str = os.getenv("WEBSOCKET_AUTH_EXCLUDE_PATHS", "")
    exclude_paths = [path.strip() for path in exclude_paths_str.split(",") if path.strip()]

    # System Client Konfiguration (State-of-the-Art)
    system_client_patterns_str = os.getenv("WEBSOCKET_AUTH_SYSTEM_CLIENT_PATTERNS",
                                          "system_heartbeat_client,system_monitor_client,system_health_client,internal_service_*")
    system_client_patterns = [pattern.strip() for pattern in system_client_patterns_str.split(",") if pattern.strip()]

    system_path_patterns_str = os.getenv("WEBSOCKET_AUTH_SYSTEM_PATH_PATTERNS",
                                        "/ws/client/system_*,/ws/agent/system_*,/ws/internal/*")
    system_path_patterns = [pattern.strip() for pattern in system_path_patterns_str.split(",") if pattern.strip()]

    system_bypass_auth = os.getenv("WEBSOCKET_AUTH_SYSTEM_BYPASS_AUTH", "true").lower() == "true"
    system_bypass_authorization = os.getenv("WEBSOCKET_AUTH_SYSTEM_BYPASS_AUTHORIZATION", "true").lower() == "true"
    system_bypass_rate_limiting = os.getenv("WEBSOCKET_AUTH_SYSTEM_BYPASS_RATE_LIMITING", "true").lower() == "true"
    system_log_access = os.getenv("WEBSOCKET_AUTH_SYSTEM_LOG_ACCESS", "true").lower() == "true"

    # Logging-Konfiguration
    log_auth_attempts = os.getenv("WEBSOCKET_AUTH_LOG_ATTEMPTS", "true").lower() == "true"
    log_auth_failures = os.getenv("WEBSOCKET_AUTH_LOG_FAILURES", "true").lower() == "true"
    debug_mode = os.getenv("WEBSOCKET_AUTH_DEBUG", "false").lower() == "true"

    # System Client Konfiguration erstellen (State-of-the-Art)
    system_client_config = SystemClientConfig(
        client_id_patterns=system_client_patterns,
        path_patterns=system_path_patterns,
        bypass_authentication=system_bypass_auth,
        bypass_authorization=system_bypass_authorization,
        bypass_rate_limiting=system_bypass_rate_limiting,
        log_system_client_access=system_log_access
    )

    # Konfiguration zusammenstellen
    config = WebSocketAuthConfig(
        enabled=enabled,
        mode=mode,
        methods=methods,
        fallback_to_bypass=fallback_to_bypass,
        strict_mode=strict_mode,
        system_clients=system_client_config,
        jwt=jwt_config,
        mtls=mtls_config,
        apply_to_paths=apply_to_paths,
        exclude_paths=exclude_paths,
        log_auth_attempts=log_auth_attempts,
        log_auth_failures=log_auth_failures,
        debug_mode=debug_mode
    )

    # Logging der Konfiguration
    if config.enabled:
        logger.info(f"WebSocket-Authentifizierung aktiviert - Modus: {config.mode.value}, "
                   f"Methoden: {[m.value for m in config.methods]}")

        if config.jwt.enabled:
            logger.info(f"WebSocket-JWT aktiviert - Erforderlich: {config.jwt.required}, "
                       f"Query-Token: {config.jwt.allow_query_token}")

        if config.mtls.enabled:
            logger.info(f"WebSocket-mTLS aktiviert - Erforderlich: {config.mtls.required}, "
                       f"Client-CA: {'✓' if config.mtls.client_ca_path else '✗'}")
    else:
        logger.debug("WebSocket-Authentifizierung deaktiviert (Auth-Bypass aktiv)")

    return config


# Globale Konfigurationsinstanz
WEBSOCKET_AUTH_CONFIG = load_websocket_auth_config()


__all__ = [
    "WEBSOCKET_AUTH_CONFIG",
    "WebSocketAuthConfig",
    "WebSocketAuthMethod",
    "WebSocketAuthMode",
    "WebSocketJWTConfig",
    "WebSocketMTLSConfig",
    "load_websocket_auth_config",
]
