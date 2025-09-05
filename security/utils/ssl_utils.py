# backend/security/utils/ssl_utils.py
"""SSL/TLS Utilities für Keiko Personal Assistant

Konsolidiert SSL/TLS-Kontext-Erstellung und -Management aus verschiedenen
Security-Modulen und bietet einheitliche SSL-API.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from ..constants import EnvVarNames, SecurityErrorMessages

logger = get_logger(__name__)


class SSLContextType(str, Enum):
    """Typen von SSL-Kontexten."""
    CLIENT = "client"
    SERVER = "server"
    MUTUAL_TLS = "mutual_tls"


@dataclass
class SSLConfig:
    """SSL-Konfiguration."""
    # Certificate Paths
    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    server_cert_path: str | None = None
    server_key_path: str | None = None

    # Validation Settings
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True

    # Protocol Settings (deprecated protocol field removed - ssl.create_default_context() handles this automatically)
    ciphers: str | None = None

    def validate(self) -> bool:
        """Validiert SSL-Konfiguration.

        Returns:
            True wenn Konfiguration gültig
        """
        # Prüfe ob erforderliche Dateien existieren
        paths_to_check = []

        if self.ca_cert_path:
            paths_to_check.append(self.ca_cert_path)
        if self.client_cert_path:
            paths_to_check.append(self.client_cert_path)
        if self.client_key_path:
            paths_to_check.append(self.client_key_path)
        if self.server_cert_path:
            paths_to_check.append(self.server_cert_path)
        if self.server_key_path:
            paths_to_check.append(self.server_key_path)

        for path in paths_to_check:
            if not Path(path).exists():
                logger.error(f"SSL-Zertifikat/Schlüssel nicht gefunden: {path}")
                return False

        return True


class SSLContextManager:
    """Manager für SSL-Kontexte."""

    def __init__(self):
        """Initialisiert SSL Context Manager."""
        self._context_cache: dict[str, ssl.SSLContext] = {}
        self._default_config = self._load_default_config()

    def _load_default_config(self) -> SSLConfig:
        """Lädt Standard-SSL-Konfiguration aus Umgebungsvariablen.

        Returns:
            Standard-SSL-Konfiguration
        """
        return SSLConfig(
            ca_cert_path=os.getenv(EnvVarNames.MTLS_CA_CERT_PATH),
            client_cert_path=os.getenv(EnvVarNames.MTLS_CLIENT_CERT_PATH),
            client_key_path=os.getenv(EnvVarNames.MTLS_CLIENT_KEY_PATH),
            server_cert_path=os.getenv("MTLS_SERVER_CERT_PATH"),
            server_key_path=os.getenv("MTLS_SERVER_KEY_PATH")
        )

    @trace_function("ssl_utils.create_context")
    def create_context(
        self,
        context_type: SSLContextType,
        config: SSLConfig | None = None
    ) -> ssl.SSLContext:
        """Erstellt SSL-Kontext.

        Args:
            context_type: Typ des SSL-Kontexts
            config: SSL-Konfiguration (optional)

        Returns:
            SSL-Kontext

        Raises:
            ValueError: Bei ungültiger Konfiguration
        """
        # Verwende Standard-Konfiguration wenn keine angegeben
        if config is None:
            config = self._default_config

        # Cache-Schlüssel erstellen
        cache_key = f"{context_type.value}:{hash(str(config))}"

        # Prüfe Cache
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        # Validiere Konfiguration
        if not config.validate():
            raise ValueError(SecurityErrorMessages.CONFIGURATION_ERROR)

        # Erstelle Kontext basierend auf Typ
        if context_type == SSLContextType.CLIENT:
            context = self._create_client_context(config)
        elif context_type == SSLContextType.SERVER:
            context = self._create_server_context(config)
        elif context_type == SSLContextType.MUTUAL_TLS:
            context = self._create_mutual_tls_context(config)
        else:
            raise ValueError(f"Unbekannter SSL-Kontext-Typ: {context_type}")

        # Cache Kontext
        self._context_cache[cache_key] = context

        logger.info(f"SSL-Kontext erstellt: {context_type}")
        return context

    def _create_client_context(self, config: SSLConfig) -> ssl.SSLContext:
        """Erstellt Client-SSL-Kontext.

        Args:
            config: SSL-Konfiguration

        Returns:
            Client-SSL-Kontext
        """
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # CA-Zertifikat laden
        if config.ca_cert_path:
            context.load_verify_locations(config.ca_cert_path)

        # Client-Zertifikat laden (für mTLS)
        if config.client_cert_path and config.client_key_path:
            context.load_cert_chain(config.client_cert_path, config.client_key_path)

        # Verifikation konfigurieren
        context.verify_mode = config.verify_mode
        context.check_hostname = config.check_hostname

        # Cipher konfigurieren
        if config.ciphers:
            context.set_ciphers(config.ciphers)

        return context

    def _create_server_context(self, config: SSLConfig) -> ssl.SSLContext:
        """Erstellt Server-SSL-Kontext.

        Args:
            config: SSL-Konfiguration

        Returns:
            Server-SSL-Kontext
        """
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # Server-Zertifikat laden
        if config.server_cert_path and config.server_key_path:
            context.load_cert_chain(config.server_cert_path, config.server_key_path)
        else:
            raise ValueError("Server-Zertifikat und -Schlüssel erforderlich für Server-Kontext")

        # CA-Zertifikat für Client-Verifikation
        if config.ca_cert_path:
            context.load_verify_locations(config.ca_cert_path)

        # Verifikation konfigurieren
        context.verify_mode = config.verify_mode

        # Cipher konfigurieren
        if config.ciphers:
            context.set_ciphers(config.ciphers)

        return context

    def _create_mutual_tls_context(self, config: SSLConfig) -> ssl.SSLContext:
        """Erstellt Mutual TLS-Kontext.

        Args:
            config: SSL-Konfiguration

        Returns:
            Mutual TLS-Kontext
        """
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # CA-Zertifikat laden
        if config.ca_cert_path:
            context.load_verify_locations(config.ca_cert_path)
        else:
            raise ValueError("CA-Zertifikat erforderlich für Mutual TLS")

        # Client-Zertifikat laden
        if config.client_cert_path and config.client_key_path:
            context.load_cert_chain(config.client_cert_path, config.client_key_path)
        else:
            raise ValueError("Client-Zertifikat und -Schlüssel erforderlich für Mutual TLS")

        # Strenge Verifikation für mTLS
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = config.check_hostname

        # Cipher konfigurieren
        if config.ciphers:
            context.set_ciphers(config.ciphers)

        return context

    def clear_cache(self) -> None:
        """Leert SSL-Kontext-Cache."""
        self._context_cache.clear()
        logger.info("SSL-Kontext-Cache geleert")

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        return {
            "cached_contexts": len(self._context_cache),
            "context_types": list({
                key.split(":")[0] for key in self._context_cache
            })
        }


# Convenience-Funktionen
def create_ssl_context(
    context_type: SSLContextType,
    ca_cert_path: str | None = None,
    client_cert_path: str | None = None,
    client_key_path: str | None = None,
    server_cert_path: str | None = None,
    server_key_path: str | None = None,
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED,
    check_hostname: bool = True
) -> ssl.SSLContext:
    """Erstellt SSL-Kontext mit gegebenen Parametern.

    Args:
        context_type: Typ des SSL-Kontexts
        ca_cert_path: Pfad zum CA-Zertifikat
        client_cert_path: Pfad zum Client-Zertifikat
        client_key_path: Pfad zum Client-Schlüssel
        server_cert_path: Pfad zum Server-Zertifikat
        server_key_path: Pfad zum Server-Schlüssel
        verify_mode: SSL-Verifikation-Modus
        check_hostname: Hostname-Verifikation aktivieren

    Returns:
        SSL-Kontext
    """
    config = SSLConfig(
        ca_cert_path=ca_cert_path,
        client_cert_path=client_cert_path,
        client_key_path=client_key_path,
        server_cert_path=server_cert_path,
        server_key_path=server_key_path,
        verify_mode=verify_mode,
        check_hostname=check_hostname
    )

    manager = SSLContextManager()
    return manager.create_context(context_type, config)


def create_client_ssl_context(
    ca_cert_path: str | None = None,
    client_cert_path: str | None = None,
    client_key_path: str | None = None,
    verify_hostname: bool = True
) -> ssl.SSLContext:
    """Erstellt Client-SSL-Kontext.

    Args:
        ca_cert_path: Pfad zum CA-Zertifikat
        client_cert_path: Pfad zum Client-Zertifikat
        client_key_path: Pfad zum Client-Schlüssel
        verify_hostname: Hostname-Verifikation aktivieren

    Returns:
        Client-SSL-Kontext
    """
    return create_ssl_context(
        SSLContextType.CLIENT,
        ca_cert_path=ca_cert_path,
        client_cert_path=client_cert_path,
        client_key_path=client_key_path,
        check_hostname=verify_hostname
    )


def create_server_ssl_context(
    server_cert_path: str,
    server_key_path: str,
    ca_cert_path: str | None = None,
    require_client_cert: bool = False
) -> ssl.SSLContext:
    """Erstellt Server-SSL-Kontext.

    Args:
        server_cert_path: Pfad zum Server-Zertifikat
        server_key_path: Pfad zum Server-Schlüssel
        ca_cert_path: Pfad zum CA-Zertifikat
        require_client_cert: Client-Zertifikat erforderlich

    Returns:
        Server-SSL-Kontext
    """
    verify_mode = ssl.CERT_REQUIRED if require_client_cert else ssl.CERT_NONE

    return create_ssl_context(
        SSLContextType.SERVER,
        ca_cert_path=ca_cert_path,
        server_cert_path=server_cert_path,
        server_key_path=server_key_path,
        verify_mode=verify_mode
    )


__all__ = [
    "SSLConfig",
    "SSLContextManager",
    "SSLContextType",
    "create_client_ssl_context",
    "create_server_ssl_context",
    "create_ssl_context",
]
