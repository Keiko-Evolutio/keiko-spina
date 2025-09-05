# backend/services/core/tests/test_ssl_config.py
"""Tests für SSL-Konfiguration.

Testet SSL-Konfiguration, Connector-Erstellung und Factory-Pattern.
"""

import ssl
from unittest.mock import MagicMock, patch

import pytest

from services.core.ssl_config import (
    DEFAULT_CLIENT_TIMEOUT_SECONDS,
    DEFAULT_CONNECTION_LIMIT,
    DEFAULT_CONNECTION_LIMIT_PER_HOST,
    DEFAULT_DNS_CACHE_TTL_SECONDS,
    DEFAULT_USER_AGENT,
    SSLConfig,
    SSLConfigFactory,
    SSLConnectionConfig,
)


class TestSSLConnectionConfig:
    """Tests für SSLConnectionConfig."""

    def test_default_values(self):
        """Testet Standard-Konfigurationswerte."""
        config = SSLConnectionConfig()

        assert config.connection_limit == DEFAULT_CONNECTION_LIMIT
        assert config.connection_limit_per_host == DEFAULT_CONNECTION_LIMIT_PER_HOST
        assert config.dns_cache_ttl_seconds == DEFAULT_DNS_CACHE_TTL_SECONDS
        assert config.use_dns_cache is True
        assert config.client_timeout_seconds == DEFAULT_CLIENT_TIMEOUT_SECONDS
        assert config.user_agent == DEFAULT_USER_AGENT

    def test_custom_values(self):
        """Testet benutzerdefinierte Konfigurationswerte."""
        config = SSLConnectionConfig(
            connection_limit=50,
            connection_limit_per_host=10,
            dns_cache_ttl_seconds=600,
            use_dns_cache=False,
            client_timeout_seconds=60.0,
            user_agent="Custom-Agent/1.0"
        )

        assert config.connection_limit == 50
        assert config.connection_limit_per_host == 10
        assert config.dns_cache_ttl_seconds == 600
        assert config.use_dns_cache is False
        assert config.client_timeout_seconds == 60.0
        assert config.user_agent == "Custom-Agent/1.0"

    def test_immutable_config(self):
        """Testet dass Konfiguration unveränderlich ist."""
        config = SSLConnectionConfig()

        with pytest.raises(AttributeError):
            config.connection_limit = 200  # Sollte fehlschlagen (frozen=True)


class TestSSLConfig:
    """Tests für SSLConfig."""

    def test_initialization_with_defaults(self):
        """Testet Initialisierung mit Standard-Werten."""
        ssl_config = SSLConfig()

        assert ssl_config.verify_ssl is False
        assert isinstance(ssl_config.config, SSLConnectionConfig)
        assert ssl_config._ssl_context is None
        assert ssl_config._connector is None

    def test_initialization_with_custom_config(self):
        """Testet Initialisierung mit benutzerdefinierter Konfiguration."""
        custom_config = SSLConnectionConfig(connection_limit=50)
        ssl_config = SSLConfig(verify_ssl=True, config=custom_config)

        assert ssl_config.verify_ssl is True
        assert ssl_config.config.connection_limit == 50

    def test_ssl_context_creation_development(self):
        """Testet SSL-Kontext-Erstellung für Development."""
        ssl_config = SSLConfig(verify_ssl=False)
        context = ssl_config.ssl_context

        assert isinstance(context, ssl.SSLContext)
        assert context.check_hostname is False
        assert context.verify_mode == ssl.CERT_NONE

    def test_ssl_context_creation_production(self):
        """Testet SSL-Kontext-Erstellung für Production."""
        ssl_config = SSLConfig(verify_ssl=True)
        context = ssl_config.ssl_context

        assert isinstance(context, ssl.SSLContext)
        # Production-Kontext sollte Standard-Verifikation haben

    def test_ssl_context_caching(self):
        """Testet dass SSL-Kontext gecacht wird."""
        ssl_config = SSLConfig()
        context1 = ssl_config.ssl_context
        context2 = ssl_config.ssl_context

        assert context1 is context2  # Sollte dasselbe Objekt sein

    @patch("aiohttp.TCPConnector")
    def test_production_connector_creation(self, mock_connector):
        """Testet Connector-Erstellung für Production."""
        SSLConfig(verify_ssl=True)

        mock_connector.assert_called_once_with(
            limit=DEFAULT_CONNECTION_LIMIT,
            limit_per_host=DEFAULT_CONNECTION_LIMIT_PER_HOST,
            ttl_dns_cache=DEFAULT_DNS_CACHE_TTL_SECONDS,
            use_dns_cache=True,
        )

    @patch("aiohttp.TCPConnector")
    def test_development_connector_creation(self, mock_connector):
        """Testet Connector-Erstellung für Development."""
        SSLConfig(verify_ssl=False)

        mock_connector.assert_called_once_with(
            limit=DEFAULT_CONNECTION_LIMIT,
            limit_per_host=DEFAULT_CONNECTION_LIMIT_PER_HOST,
            ttl_dns_cache=DEFAULT_DNS_CACHE_TTL_SECONDS,
            use_dns_cache=True,
            ssl=False
        )

    def test_test_connector_creation_on_runtime_error(self):
        """Testet Mock-Connector-Erstellung bei RuntimeError."""
        ssl_config = SSLConfig(verify_ssl=False)

        # Simuliere RuntimeError (kein Event Loop)
        with patch("aiohttp.TCPConnector", side_effect=RuntimeError("No event loop")):
            connector = ssl_config.aiohttp_connector

            # Sollte Mock-Connector erstellen mit ssl-Attribut
            assert hasattr(connector, "ssl") and connector.ssl is False

    @patch("aiohttp.ClientSession")
    @patch("aiohttp.TCPConnector")
    def test_client_session_creation_default_timeout(self, mock_connector_class, mock_session):
        """Testet Client-Session-Erstellung mit Standard-Timeout."""
        ssl_config = SSLConfig()

        # Mock den Connector
        mock_connector = MagicMock()
        mock_connector_class.return_value = mock_connector

        ssl_config.create_client_session()

        mock_session.assert_called_once()
        call_args = mock_session.call_args

        assert call_args[1]["connector"] == mock_connector
        assert call_args[1]["timeout"].total == DEFAULT_CLIENT_TIMEOUT_SECONDS
        assert call_args[1]["headers"]["User-Agent"] == DEFAULT_USER_AGENT

    @patch("aiohttp.ClientSession")
    @patch("aiohttp.TCPConnector")
    def test_client_session_creation_custom_timeout(self, mock_connector_class, mock_session):
        """Testet Client-Session-Erstellung mit benutzerdefiniertem Timeout."""
        ssl_config = SSLConfig()
        custom_timeout = 60.0

        # Mock den Connector
        mock_connector = MagicMock()
        mock_connector_class.return_value = mock_connector

        ssl_config.create_client_session(timeout=custom_timeout)

        call_args = mock_session.call_args
        assert call_args[1]["timeout"].total == custom_timeout

    @pytest.mark.asyncio
    async def test_close_method_with_async_connector(self):
        """Testet close-Methode mit asynchronem Connector."""
        ssl_config = SSLConfig()

        # Mock-Connector mit asynchroner close-Methode
        mock_connector = MagicMock()

        # Erstelle echte Coroutine für close
        async def mock_close():
            pass

        mock_connector.close.return_value = mock_close()
        ssl_config._connector = mock_connector

        await ssl_config.close()

        mock_connector.close.assert_called_once()
        assert ssl_config._connector is None

    @pytest.mark.asyncio
    async def test_close_method_with_sync_connector(self):
        """Testet close-Methode mit synchronem Connector."""
        ssl_config = SSLConfig()

        # Mock-Connector mit synchroner close-Methode
        mock_connector = MagicMock()
        mock_connector.close.return_value = None  # Synchron
        ssl_config._connector = mock_connector

        await ssl_config.close()

        mock_connector.close.assert_called_once()
        assert ssl_config._connector is None

    @pytest.mark.asyncio
    async def test_close_method_no_connector(self):
        """Testet close-Methode ohne Connector."""
        ssl_config = SSLConfig()

        # Kein Connector gesetzt
        await ssl_config.close()

        # Sollte ohne Fehler durchlaufen


class TestSSLConfigFactory:
    """Tests für SSLConfigFactory."""

    @patch.dict("os.environ", {"ENVIRONMENT": "production"})
    def test_create_from_environment_production(self):
        """Testet Factory für Production-Umgebung."""
        ssl_config = SSLConfigFactory.create_from_environment()

        assert ssl_config.verify_ssl is True

    @patch.dict("os.environ", {"ENVIRONMENT": "development"})
    def test_create_from_environment_development(self):
        """Testet Factory für Development-Umgebung."""
        ssl_config = SSLConfigFactory.create_from_environment()

        assert ssl_config.verify_ssl is False

    @patch.dict("os.environ", {}, clear=True)
    def test_create_from_environment_default(self):
        """Testet Factory mit Standard-Umgebung."""
        ssl_config = SSLConfigFactory.create_from_environment()

        assert ssl_config.verify_ssl is False  # Standard ist Development

    def test_create_for_development(self):
        """Testet explizite Development-Konfiguration."""
        ssl_config = SSLConfigFactory.create_for_development()

        assert ssl_config.verify_ssl is False

    def test_create_for_production(self):
        """Testet explizite Production-Konfiguration."""
        ssl_config = SSLConfigFactory.create_for_production()

        assert ssl_config.verify_ssl is True

    def test_create_with_custom_config(self):
        """Testet Factory mit benutzerdefinierter Konfiguration."""
        custom_config = SSLConnectionConfig(connection_limit=50)
        ssl_config = SSLConfigFactory.create_for_production(config=custom_config)

        assert ssl_config.verify_ssl is True
        assert ssl_config.config.connection_limit == 50


class TestBackwardCompatibility:
    """Tests für Backward Compatibility."""

    def test_get_ssl_config_function(self):
        """Testet deprecated get_ssl_config Funktion."""
        from services.core.ssl_config import get_ssl_config

        ssl_config = get_ssl_config()
        assert isinstance(ssl_config, SSLConfig)

    @pytest.mark.asyncio
    async def test_cleanup_ssl_config_function(self):
        """Testet deprecated cleanup_ssl_config Funktion."""
        from services.core.ssl_config import cleanup_ssl_config, get_ssl_config

        # Erstelle globale Konfiguration
        get_ssl_config()

        # Bereinige
        await cleanup_ssl_config()

        # Sollte erfolgreich sein (keine Exception)
