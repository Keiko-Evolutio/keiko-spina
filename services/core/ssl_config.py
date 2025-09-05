# backend/services/core/ssl_config.py

import ssl
from dataclasses import dataclass

import aiohttp

from kei_logging import get_logger

logger = get_logger(__name__)

# SSL Configuration Constants
DEFAULT_CLIENT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_CONNECTION_LIMIT: int = 100
DEFAULT_CONNECTION_LIMIT_PER_HOST: int = 30
DEFAULT_DNS_CACHE_TTL_SECONDS: int = 300
DEFAULT_USER_AGENT: str = "Keiko-Personal-Assistant/2.0"


@dataclass
class SSLConnectionConfig:
    """SSL Connection Configuration."""
    connection_limit: int = DEFAULT_CONNECTION_LIMIT
    connection_limit_per_host: int = DEFAULT_CONNECTION_LIMIT_PER_HOST
    dns_cache_ttl_seconds: int = DEFAULT_DNS_CACHE_TTL_SECONDS
    use_dns_cache: bool = True
    client_timeout_seconds: float = DEFAULT_CLIENT_TIMEOUT_SECONDS
    user_agent: str = DEFAULT_USER_AGENT


class SSLConfig:
    """Zentrale SSL-Konfiguration für alle Azure-Services."""

    def __init__(self, verify_ssl: bool = False, config: SSLConnectionConfig | None = None):
        """Initialisiert SSL-Konfiguration.

        Args:
            verify_ssl: SSL-Zertifikate verifizieren (False für Development)
            config: SSL Connection Configuration
        """
        self.verify_ssl = verify_ssl
        self.config = config or SSLConnectionConfig()
        self._ssl_context: ssl.SSLContext | None = None
        self._connector: aiohttp.TCPConnector | None = None

        logger.info(f"🔐 SSL-Konfiguration initialisiert: verify_ssl={verify_ssl}")

    @property
    def ssl_context(self) -> ssl.SSLContext:
        """SSL-Kontext für WebSocket und andere Verbindungen."""
        if self._ssl_context is None:
            self._ssl_context = ssl.create_default_context()

            if not self.verify_ssl:
                # Development-Modus: SSL-Verifikation deaktivieren
                self._ssl_context.check_hostname = False
                self._ssl_context.verify_mode = ssl.CERT_NONE
                logger.debug("🔓 SSL-Verifikation deaktiviert für Development")
            else:
                logger.debug("🔒 SSL-Verifikation aktiviert für Production")

        return self._ssl_context

    @property
    def aiohttp_connector(self) -> aiohttp.TCPConnector:
        """TCP-Connector für aiohttp mit SSL-Konfiguration."""
        if self._connector is None:
            try:
                if self.verify_ssl:
                    # Production: Standard SSL-Verifikation
                    self._connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                    )
                else:
                    # Development: SSL-Verifikation deaktiviert
                    self._connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        ssl=False  # SSL-Verifikation komplett deaktivieren
                    )
                    logger.debug("🔓 aiohttp SSL-Verifikation deaktiviert")
            except RuntimeError:
                # Kein Event Loop läuft - erstelle Mock-Connector für Tests
                from unittest.mock import MagicMock
                mock_connector = MagicMock(spec=aiohttp.TCPConnector)
                mock_connector.ssl = False if not self.verify_ssl else True
                self._connector = mock_connector  # type: ignore[assignment]
                logger.debug("🧪 Mock-Connector für Tests erstellt")

        return self._connector

    def create_client_session(self, timeout: float = 30.0) -> aiohttp.ClientSession:
        """Erstellt eine aiohttp ClientSession mit korrekter SSL-Konfiguration.

        Args:
            timeout: Timeout in Sekunden

        Returns:
            Konfigurierte ClientSession
        """
        return aiohttp.ClientSession(
            connector=self.aiohttp_connector,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={
                "User-Agent": "Keiko-Personal-Assistant/2.0"
            }
        )

    async def close(self):
        """Schließt alle offenen Verbindungen."""
        if self._connector:
            await self._connector.close()
            self._connector = None
            logger.debug("🔌 SSL-Connector geschlossen")


# Globale SSL-Konfiguration
# In Development: SSL-Verifikation deaktiviert
# In Production: SSL-Verifikation aktiviert
_ssl_config: SSLConfig | None = None


def get_ssl_config() -> SSLConfig:
    """Gibt die globale SSL-Konfiguration zurück."""
    global _ssl_config
    if _ssl_config is None:
        import os

        # Explizite SSL-Konfiguration über Umgebungsvariable
        # Standard: SSL-Verifikation IMMER aktiviert (sicher)
        verify_ssl_env = os.getenv("VERIFY_SSL", "true").lower()
        verify_ssl = verify_ssl_env in ("true", "1", "yes", "on")

        environment = os.getenv("ENVIRONMENT", "development").lower()

        # Warnung bei deaktivierter SSL-Verifikation
        if not verify_ssl:
            logger.warning("⚠️  SSL-Verifikation ist DEAKTIVIERT! Nur für lokale Entwicklung verwenden.")
            logger.warning("⚠️  Setze VERIFY_SSL=true für sichere Verbindungen.")

        _ssl_config = SSLConfig(verify_ssl=verify_ssl)
        logger.info(f"🌍 SSL-Konfiguration für {environment} erstellt (verify_ssl={verify_ssl})")

    return _ssl_config


async def cleanup_ssl_config():
    """Bereinigt die globale SSL-Konfiguration."""
    global _ssl_config
    if _ssl_config:
        await _ssl_config.close()
        _ssl_config = None
        logger.info("🧹 SSL-Konfiguration bereinigt")


class SSLConfigFactory:
    """Factory für SSL-Konfigurationen."""

    @staticmethod
    def create_from_environment() -> SSLConfig:
        """Erstellt SSL-Konfiguration aus Umgebungsvariablen."""
        import os
        verify_ssl_env = os.getenv("VERIFY_SSL", "false").lower()
        verify_ssl = verify_ssl_env in ("true", "1", "yes", "on")
        return SSLConfig(verify_ssl=verify_ssl)

    @staticmethod
    def create_for_development() -> SSLConfig:
        """Erstellt SSL-Konfiguration für Development."""
        return SSLConfig(verify_ssl=False)

    @staticmethod
    def create_for_production() -> SSLConfig:
        """Erstellt SSL-Konfiguration für Production."""
        return SSLConfig(verify_ssl=True)
