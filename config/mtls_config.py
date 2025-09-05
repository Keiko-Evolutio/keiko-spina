"""mTLS-Konfiguration für KEI-MCP API.

Diese Datei enthält alle Konfigurationsoptionen für mutual TLS (mTLS)
Authentifizierung sowohl für ausgehende Verbindungen zu externen MCP Servern
als auch für eingehende Client-Authentifizierung.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


class MTLSMode(Enum):
    """mTLS-Modi für verschiedene Authentifizierungsszenarien."""

    DISABLED = "disabled"           # Kein mTLS
    OPTIONAL = "optional"           # mTLS optional, Fallback auf Bearer Token
    REQUIRED = "required"           # mTLS erforderlich
    OUTBOUND_ONLY = "outbound_only" # Nur für ausgehende Verbindungen


@dataclass
class MTLSCertificateConfig:
    """Konfiguration für ein mTLS-Zertifikat."""

    cert_path: str
    key_path: str
    ca_bundle_path: str | None = None
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True

    def __post_init__(self):
        """Validiert die Zertifikat-Pfade."""
        cert_file = Path(self.cert_path)
        key_file = Path(self.key_path)

        if not cert_file.exists():
            raise FileNotFoundError(f"Client-Zertifikat nicht gefunden: {self.cert_path}")

        if not key_file.exists():
            raise FileNotFoundError(f"Private Key nicht gefunden: {self.key_path}")

        if self.ca_bundle_path:
            ca_file = Path(self.ca_bundle_path)
            if not ca_file.exists():
                raise FileNotFoundError(f"CA-Bundle nicht gefunden: {self.ca_bundle_path}")

    def to_httpx_cert(self) -> tuple:
        """Konvertiert zu httpx-kompatiblem Zertifikat-Tupel."""
        return self.cert_path, self.key_path

    def get_verify_param(self) -> str | bool:
        """Gibt httpx-kompatiblen verify-Parameter zurück."""
        if self.ca_bundle_path:
            return self.ca_bundle_path
        return self.verify_mode != ssl.CERT_NONE


@dataclass
class MTLSOutboundConfig:
    """Konfiguration für ausgehende mTLS-Verbindungen (KEI → MCP Server)."""

    enabled: bool = False
    default_cert_config: MTLSCertificateConfig | None = None
    per_server_certs: dict[str, MTLSCertificateConfig] = field(default_factory=dict)
    verify_server_certs: bool = True
    ssl_context_options: dict[str, Any] = field(default_factory=dict)

    def get_cert_config(self, server_name: str) -> MTLSCertificateConfig | None:
        """Gibt die Zertifikat-Konfiguration für einen Server zurück."""
        # Zuerst server-spezifische Konfiguration prüfen
        if server_name in self.per_server_certs:
            return self.per_server_certs[server_name]

        # Fallback auf Default-Konfiguration
        return self.default_cert_config


@dataclass
class MTLSInboundConfig:
    """Konfiguration für eingehende mTLS-Verbindungen (Client → KEI API)."""

    enabled: bool = False
    mode: MTLSMode = MTLSMode.OPTIONAL
    client_ca_path: str | None = None
    verify_client_certs: bool = True
    require_client_certs: bool = False
    allowed_client_subjects: list[str] | None = None
    cert_header_name: str = "X-Client-Cert"  # Header für Proxy-übertragene Zerts

    def __post_init__(self):
        """Validiert die Inbound-Konfiguration."""
        if self.enabled and self.verify_client_certs and not self.client_ca_path:
            raise ValueError("client_ca_path erforderlich wenn verify_client_certs aktiviert")

        if self.client_ca_path:
            ca_file = Path(self.client_ca_path)
            if not ca_file.exists():
                raise FileNotFoundError(f"Client-CA nicht gefunden: {self.client_ca_path}")


@dataclass
class MTLSSettings:
    """Umfassende mTLS-Einstellungen für KEI-MCP API."""

    outbound: MTLSOutboundConfig = field(default_factory=MTLSOutboundConfig)
    inbound: MTLSInboundConfig = field(default_factory=MTLSInboundConfig)

    # Globale Einstellungen
    enable_mtls_logging: bool = True
    log_cert_details: bool = False  # Vorsicht: Kann sensitive Daten loggen
    ssl_debug: bool = False

    # Performance-Einstellungen
    ssl_session_cache_size: int = 1000
    ssl_session_timeout: int = 3600  # 1 Stunde


def load_mtls_settings() -> MTLSSettings:
    """Lädt mTLS-Einstellungen aus Umgebungsvariablen.

    Returns:
        Vollständige mTLS-Konfiguration
    """
    # Outbound mTLS-Konfiguration
    outbound_enabled = os.getenv("KEI_MCP_OUTBOUND_MTLS_ENABLED", "false").lower() == "true"

    outbound_config = MTLSOutboundConfig(enabled=outbound_enabled)

    if outbound_enabled:
        # Default Client-Zertifikat
        default_cert_path = os.getenv("KEI_MCP_CLIENT_CERT_PATH")
        default_key_path = os.getenv("KEI_MCP_CLIENT_KEY_PATH")
        default_ca_bundle = os.getenv("KEI_MCP_CA_BUNDLE_PATH")

        if default_cert_path and default_key_path:
            try:
                outbound_config.default_cert_config = MTLSCertificateConfig(
                    cert_path=default_cert_path,
                    key_path=default_key_path,
                    ca_bundle_path=default_ca_bundle,
                    verify_mode=ssl.CERT_REQUIRED if os.getenv("KEI_MCP_VERIFY_SERVER_CERTS", "true").lower() == "true" else ssl.CERT_NONE,
                    check_hostname=os.getenv("KEI_MCP_CHECK_HOSTNAME", "true").lower() == "true"
                )
                logger.info(f"Default Outbound mTLS-Zertifikat konfiguriert: {default_cert_path}")
            except FileNotFoundError as e:
                logger.exception(f"Outbound mTLS-Konfiguration fehlgeschlagen: {e}")
                outbound_config.enabled = False

        # Server-spezifische Zertifikate
        # Format: KEI_MCP_SERVER_{SERVER_NAME}_CERT_PATH, KEI_MCP_SERVER_{SERVER_NAME}_KEY_PATH
        for env_var in os.environ:
            if env_var.startswith("KEI_MCP_SERVER_") and env_var.endswith("_CERT_PATH"):
                server_name = env_var.replace("KEI_MCP_SERVER_", "").replace("_CERT_PATH", "").lower()
                cert_path = os.getenv(env_var)
                key_path = os.getenv(f"KEI_MCP_SERVER_{server_name.upper()}_KEY_PATH")
                ca_bundle = os.getenv(f"KEI_MCP_SERVER_{server_name.upper()}_CA_BUNDLE_PATH")

                if cert_path and key_path:
                    try:
                        outbound_config.per_server_certs[server_name] = MTLSCertificateConfig(
                            cert_path=cert_path,
                            key_path=key_path,
                            ca_bundle_path=ca_bundle
                        )
                        logger.info(f"Server-spezifisches mTLS-Zertifikat konfiguriert für {server_name}")
                    except FileNotFoundError as e:
                        logger.exception(f"Server-spezifische mTLS-Konfiguration fehlgeschlagen für {server_name}: {e}")

    # Inbound mTLS-Konfiguration
    inbound_enabled = os.getenv("KEI_MCP_INBOUND_MTLS_ENABLED", "false").lower() == "true"

    inbound_mode_str = os.getenv("KEI_MCP_INBOUND_MTLS_MODE", "optional").lower()
    try:
        inbound_mode = MTLSMode(inbound_mode_str)
    except ValueError:
        logger.warning(f"Ungültiger mTLS-Modus: {inbound_mode_str}, verwende 'optional'")
        inbound_mode = MTLSMode.OPTIONAL

    inbound_config = MTLSInboundConfig(
        enabled=inbound_enabled,
        mode=inbound_mode,
        client_ca_path=os.getenv("KEI_MCP_CLIENT_CA_PATH"),
        verify_client_certs=os.getenv("KEI_MCP_VERIFY_CLIENT_CERTS", "true").lower() == "true",
        require_client_certs=os.getenv("KEI_MCP_REQUIRE_CLIENT_CERTS", "false").lower() == "true",
        cert_header_name=os.getenv("KEI_MCP_CLIENT_CERT_HEADER", "X-Client-Cert")
    )

    # Allowed Client Subjects
    allowed_subjects_str = os.getenv("KEI_MCP_ALLOWED_CLIENT_SUBJECTS", "")
    if allowed_subjects_str:
        inbound_config.allowed_client_subjects = [s.strip() for s in allowed_subjects_str.split(",") if s.strip()]

    # Globale Einstellungen
    settings = MTLSSettings(
        outbound=outbound_config,
        inbound=inbound_config,
        enable_mtls_logging=os.getenv("KEI_MCP_MTLS_LOGGING", "true").lower() == "true",
        log_cert_details=os.getenv("KEI_MCP_LOG_CERT_DETAILS", "false").lower() == "true",
        ssl_debug=os.getenv("KEI_MCP_SSL_DEBUG", "false").lower() == "true"
    )

    # Logging
    if settings.outbound.enabled:
        logger.info(f"Outbound mTLS aktiviert - Default-Cert: {'✓' if settings.outbound.default_cert_config else '✗'}, "
                   f"Server-spezifische Certs: {len(settings.outbound.per_server_certs)}")

    if settings.inbound.enabled:
        logger.info(f"Inbound mTLS aktiviert - Modus: {settings.inbound.mode.value}, "
                   f"Client-CA: {'✓' if settings.inbound.client_ca_path else '✗'}")

    return settings


def get_example_mtls_configs() -> dict[str, dict[str, Any]]:
    """Gibt Beispiel-mTLS-Konfigurationen zurück.

    Returns:
        Dictionary mit Beispiel-Konfigurationen für verschiedene Szenarien
    """
    return {
        "outbound_only": {
            "description": "Nur ausgehende mTLS-Verbindungen zu externen MCP Servern",
            "env_vars": {
                "KEI_MCP_OUTBOUND_MTLS_ENABLED": "true",
                "KEI_MCP_CLIENT_CERT_PATH": "/etc/ssl/certs/kei-mcp-client.pem",
                "KEI_MCP_CLIENT_KEY_PATH": "/etc/ssl/private/kei-mcp-client.key",
                "KEI_MCP_CA_BUNDLE_PATH": "/etc/ssl/certs/ca-bundle.pem",
                "KEI_MCP_VERIFY_SERVER_CERTS": "true"
            }
        },
        "inbound_optional": {
            "description": "Optionale Client-Zertifikat-Authentifizierung mit Bearer Token Fallback",
            "env_vars": {
                "KEI_MCP_INBOUND_MTLS_ENABLED": "true",
                "KEI_MCP_INBOUND_MTLS_MODE": "optional",
                "KEI_MCP_CLIENT_CA_PATH": "/etc/ssl/certs/client-ca.pem",
                "KEI_MCP_VERIFY_CLIENT_CERTS": "true",
                "KEI_MCP_REQUIRE_CLIENT_CERTS": "false"
            }
        },
        "inbound_required": {
            "description": "Erforderliche Client-Zertifikat-Authentifizierung",
            "env_vars": {
                "KEI_MCP_INBOUND_MTLS_ENABLED": "true",
                "KEI_MCP_INBOUND_MTLS_MODE": "required",
                "KEI_MCP_CLIENT_CA_PATH": "/etc/ssl/certs/client-ca.pem",
                "KEI_MCP_VERIFY_CLIENT_CERTS": "true",
                "KEI_MCP_REQUIRE_CLIENT_CERTS": "true",
                "KEI_MCP_ALLOWED_CLIENT_SUBJECTS": "CN=client1.example.com,CN=client2.example.com"
            }
        },
        "bidirectional": {
            "description": "Vollständige bidirektionale mTLS-Authentifizierung",
            "env_vars": {
                "KEI_MCP_OUTBOUND_MTLS_ENABLED": "true",
                "KEI_MCP_CLIENT_CERT_PATH": "/etc/ssl/certs/kei-mcp-client.pem",
                "KEI_MCP_CLIENT_KEY_PATH": "/etc/ssl/private/kei-mcp-client.key",
                "KEI_MCP_CA_BUNDLE_PATH": "/etc/ssl/certs/ca-bundle.pem",
                "KEI_MCP_INBOUND_MTLS_ENABLED": "true",
                "KEI_MCP_INBOUND_MTLS_MODE": "required",
                "KEI_MCP_CLIENT_CA_PATH": "/etc/ssl/certs/client-ca.pem",
                "KEI_MCP_VERIFY_CLIENT_CERTS": "true",
                "KEI_MCP_REQUIRE_CLIENT_CERTS": "true"
            }
        },
        "per_server_certs": {
            "description": "Server-spezifische Client-Zertifikate für verschiedene MCP Server",
            "env_vars": {
                "KEI_MCP_OUTBOUND_MTLS_ENABLED": "true",
                "KEI_MCP_SERVER_WEATHER_CERT_PATH": "/etc/ssl/certs/weather-client.pem",
                "KEI_MCP_SERVER_WEATHER_KEY_PATH": "/etc/ssl/private/weather-client.key",
                "KEI_MCP_SERVER_FINANCE_CERT_PATH": "/etc/ssl/certs/finance-client.pem",
                "KEI_MCP_SERVER_FINANCE_KEY_PATH": "/etc/ssl/private/finance-client.key",
                "KEI_MCP_CA_BUNDLE_PATH": "/etc/ssl/certs/ca-bundle.pem"
            }
        }
    }


# Globale mTLS-Einstellungen laden
MTLS_SETTINGS = load_mtls_settings()


__all__ = [
    "MTLS_SETTINGS",
    "MTLSCertificateConfig",
    "MTLSInboundConfig",
    "MTLSMode",
    "MTLSOutboundConfig",
    "MTLSSettings",
    "get_example_mtls_configs",
    "load_mtls_settings"
]
