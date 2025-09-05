# backend/services/clients/tests/test_common_http_config.py
"""Tests für Common HTTP Configuration.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""

from unittest.mock import Mock, patch

import aiohttp
import httpx
import pytest

from services.clients.common.http_config import (
    HTTPClientConfig,
    StandardHTTPClientConfig,
    create_aiohttp_session_config,
    create_azure_headers,
    create_httpx_client_config,
    create_kei_rpc_headers,
    create_openai_headers,
)


class TestHTTPClientConfig:
    """Tests für HTTPClientConfig Dataclass."""

    def test_http_client_config_default_values(self) -> None:
        """Prüft, dass HTTPClientConfig korrekte Standardwerte hat."""
        config = HTTPClientConfig()

        assert config.timeout == 30.0
        assert config.connect_timeout == 5.0
        assert config.connection_limit == 100
        assert config.connection_limit_per_host == 30
        assert config.keepalive_timeout == 30
        assert config.trust_env is True
        assert config.verify_ssl is True
        assert config.headers is None

    def test_http_client_config_custom_values(self) -> None:
        """Prüft, dass HTTPClientConfig benutzerdefinierte Werte akzeptiert."""
        custom_headers = {"Custom-Header": "value"}

        config = HTTPClientConfig(
            timeout=60.0,
            connect_timeout=10.0,
            connection_limit=200,
            connection_limit_per_host=50,
            keepalive_timeout=60,
            trust_env=False,
            verify_ssl=False,
            headers=custom_headers
        )

        assert config.timeout == 60.0
        assert config.connect_timeout == 10.0
        assert config.connection_limit == 200
        assert config.connection_limit_per_host == 50
        assert config.keepalive_timeout == 60
        assert config.trust_env is False
        assert config.verify_ssl is False
        assert config.headers is custom_headers

    def test_http_client_config_slots(self) -> None:
        """Prüft, dass HTTPClientConfig __slots__ verwendet."""
        config = HTTPClientConfig()

        # __slots__ verhindert dynamische Attribute
        with pytest.raises(AttributeError):
            config.dynamic_attribute = "test"  # type: ignore


class TestCreateAiohttpSessionConfig:
    """Tests für create_aiohttp_session_config Funktion."""

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_create_aiohttp_session_config_default(self, mock_ssl_config: Mock) -> None:
        """Prüft, dass Standard aiohttp Session-Konfiguration erstellt wird."""
        mock_ssl_config.return_value = Mock()

        config = create_aiohttp_session_config()

        assert isinstance(config, dict)
        assert "timeout" in config
        assert "connector_config" in config
        assert "trust_env" in config

        # Prüfe Timeout-Konfiguration
        timeout = config["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 30.0
        assert timeout.connect == 5.0

        # Prüfe Connector-Konfiguration
        connector_config = config["connector_config"]
        assert isinstance(connector_config, dict)
        assert connector_config["limit"] == 100
        assert connector_config["limit_per_host"] == 30

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_create_aiohttp_session_config_custom(self, mock_ssl_config: Mock) -> None:
        """Prüft, dass benutzerdefinierte aiohttp Session-Konfiguration erstellt wird."""
        mock_ssl_config.return_value = Mock()

        custom_config = HTTPClientConfig(
            timeout=60.0,
            connect_timeout=10.0,
            headers={"Custom": "header"}
        )

        config = create_aiohttp_session_config(custom_config)

        # Prüfe angepasste Werte
        timeout = config["timeout"]
        assert timeout.total == 60.0
        assert timeout.connect == 10.0
        assert config["headers"] == {"Custom": "header"}

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_create_aiohttp_session_config_overrides(self, mock_ssl_config: Mock) -> None:
        """Prüft, dass Überschreibungen in aiohttp Session-Konfiguration funktionieren."""
        mock_ssl_config.return_value = Mock()

        config = create_aiohttp_session_config(
            trust_env=False, custom_param="value"
        )

        assert config["trust_env"] is False
        assert config["custom_param"] == "value"

    @patch("services.clients.common.http_config.get_ssl_config")
    @patch("services.clients.common.http_config.logger")
    def test_create_aiohttp_session_config_logging(
        self,
        mock_logger: Mock,
        mock_ssl_config: Mock
    ) -> None:
        """Prüft, dass aiohttp Session-Konfiguration geloggt wird."""
        mock_ssl_config.return_value = Mock()

        create_aiohttp_session_config()

        mock_logger.debug.assert_called_once()
        log_data = mock_logger.debug.call_args[0][0]
        assert log_data["event"] == "aiohttp_session_config_created"


class TestCreateHttpxClientConfig:
    """Tests für create_httpx_client_config Funktion."""

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_create_httpx_client_config_default(self, mock_ssl_config: Mock) -> None:
        """Prüft, dass Standard httpx Client-Konfiguration erstellt wird."""
        mock_ssl_config.return_value = Mock(verify_ssl=True)

        config = create_httpx_client_config()

        assert isinstance(config, dict)
        assert "timeout" in config
        assert "verify" in config
        assert "limits" in config
        assert "trust_env" in config

        # Prüfe Limits-Konfiguration
        limits = config["limits"]
        assert isinstance(limits, httpx.Limits)

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_create_httpx_client_config_with_base_url(self, mock_ssl_config: Mock) -> None:
        """Prüft, dass httpx Client-Konfiguration mit Base-URL erstellt wird."""
        mock_ssl_config.return_value = Mock(verify_ssl=True)

        base_url = "https://api.example.com"
        config = create_httpx_client_config(base_url=base_url)

        assert config["base_url"] == base_url

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_create_httpx_client_config_custom(self, mock_ssl_config: Mock) -> None:
        """Prüft, dass benutzerdefinierte httpx Client-Konfiguration erstellt wird."""
        mock_ssl_config.return_value = Mock(verify_ssl=False)

        custom_config = HTTPClientConfig(
            timeout=120.0,
            headers={"Authorization": "Bearer token"}
        )

        config = create_httpx_client_config(custom_config)

        assert config["timeout"] == 120.0
        assert config["headers"] == {"Authorization": "Bearer token"}

    @patch("services.clients.common.http_config.get_ssl_config")
    @patch("services.clients.common.http_config.logger")
    def test_create_httpx_client_config_logging(
        self,
        mock_logger: Mock,
        mock_ssl_config: Mock
    ) -> None:
        """Prüft, dass httpx Client-Konfiguration geloggt wird."""
        mock_ssl_config.return_value = Mock(verify_ssl=True)

        create_httpx_client_config(base_url="https://test.com")

        mock_logger.debug.assert_called_once()
        log_data = mock_logger.debug.call_args[0][0]
        assert log_data["event"] == "httpx_client_config_created"
        assert log_data["base_url"] == "https://test.com"


class TestHeaderCreationFunctions:
    """Tests für Header-Erstellungsfunktionen."""

    def test_create_azure_headers_basic(self) -> None:
        """Prüft, dass Azure Headers korrekt erstellt werden."""
        api_key = "test-api-key"

        headers = create_azure_headers(api_key)

        expected_headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        assert headers == expected_headers

    def test_create_azure_headers_custom_content_type(self) -> None:
        """Prüft, dass Azure Headers mit benutzerdefiniertem Content-Type erstellt werden."""
        api_key = "test-api-key"
        content_type = "application/xml"

        headers = create_azure_headers(api_key, content_type=content_type)

        assert headers["Content-Type"] == content_type

    def test_create_azure_headers_additional_headers(self) -> None:
        """Prüft, dass Azure Headers mit zusätzlichen Headers erstellt werden."""
        api_key = "test-api-key"
        additional = {"X-Custom-Header": "custom-value"}

        headers = create_azure_headers(api_key, additional_headers=additional)

        assert headers["X-Custom-Header"] == "custom-value"
        assert headers["Ocp-Apim-Subscription-Key"] == api_key

    def test_create_openai_headers_basic(self) -> None:
        """Prüft, dass OpenAI Headers korrekt erstellt werden."""
        api_key = "test-openai-key"

        headers = create_openai_headers(api_key)

        expected_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        assert headers == expected_headers

    def test_create_openai_headers_additional_headers(self) -> None:
        """Prüft, dass OpenAI Headers mit zusätzlichen Headers erstellt werden."""
        api_key = "test-openai-key"
        additional = {"X-Request-ID": "request-123"}

        headers = create_openai_headers(api_key, additional_headers=additional)

        assert headers["X-Request-ID"] == "request-123"
        assert headers["Authorization"] == f"Bearer {api_key}"

    def test_create_kei_rpc_headers_basic(self) -> None:
        """Prüft, dass KEI-RPC Headers korrekt erstellt werden."""
        api_token = "test-token"
        tenant_id = "test-tenant"

        headers = create_kei_rpc_headers(api_token, tenant_id)

        expected_headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Tenant-Id": tenant_id,
        }

        assert headers == expected_headers

    def test_create_kei_rpc_headers_additional_headers(self) -> None:
        """Prüft, dass KEI-RPC Headers mit zusätzlichen Headers erstellt werden."""
        api_token = "test-token"
        tenant_id = "test-tenant"
        additional = {"X-Trace-ID": "trace-456"}

        headers = create_kei_rpc_headers(api_token, tenant_id, additional_headers=additional)

        assert headers["X-Trace-ID"] == "trace-456"
        assert headers["X-Tenant-Id"] == tenant_id


class TestStandardHTTPClientConfig:
    """Tests für StandardHTTPClientConfig Factory-Klasse."""

    def test_content_safety_config(self) -> None:
        """Prüft, dass Content Safety Konfiguration korrekt erstellt wird."""
        config = StandardHTTPClientConfig.content_safety()

        assert isinstance(config, HTTPClientConfig)
        assert config.timeout == 10.0
        assert config.connect_timeout == 5.0

    def test_image_generation_config(self) -> None:
        """Prüft, dass Image Generation Konfiguration korrekt erstellt wird."""
        config = StandardHTTPClientConfig.image_generation()

        assert isinstance(config, HTTPClientConfig)
        assert config.timeout == 60.0  # Längere Timeouts für Bildgenerierung
        assert config.connect_timeout == 10.0

    def test_deep_research_config(self) -> None:
        """Prüft, dass Deep Research Konfiguration korrekt erstellt wird."""
        config = StandardHTTPClientConfig.deep_research()

        assert isinstance(config, HTTPClientConfig)
        assert config.timeout == 120.0  # Sehr lange Timeouts für Research
        assert config.connect_timeout == 10.0

    def test_kei_rpc_config(self) -> None:
        """Prüft, dass KEI-RPC Konfiguration korrekt erstellt wird."""
        config = StandardHTTPClientConfig.kei_rpc()

        assert isinstance(config, HTTPClientConfig)
        assert config.timeout == 15.0
        assert config.connect_timeout == 5.0

    def test_all_configs_are_independent(self) -> None:
        """Prüft, dass alle Standard-Konfigurationen unabhängig sind."""
        config1 = StandardHTTPClientConfig.content_safety()
        config2 = StandardHTTPClientConfig.image_generation()
        config3 = StandardHTTPClientConfig.deep_research()
        config4 = StandardHTTPClientConfig.kei_rpc()

        # Verschiedene Instanzen
        configs = [config1, config2, config3, config4]
        for i, config_a in enumerate(configs):
            for j, config_b in enumerate(configs):
                if i != j:
                    assert config_a is not config_b

    def test_configs_have_different_timeouts(self) -> None:
        """Prüft, dass verschiedene Services verschiedene Timeouts haben."""
        content_safety = StandardHTTPClientConfig.content_safety()
        image_generation = StandardHTTPClientConfig.image_generation()
        deep_research = StandardHTTPClientConfig.deep_research()
        kei_rpc = StandardHTTPClientConfig.kei_rpc()

        timeouts = [
            content_safety.timeout,
            image_generation.timeout,
            deep_research.timeout,
            kei_rpc.timeout
        ]

        # Alle Timeouts sollten unterschiedlich sein (für verschiedene Use Cases)
        assert len(set(timeouts)) == len(timeouts)

        # Deep Research sollte die längsten Timeouts haben
        assert deep_research.timeout == max(timeouts)


class TestHTTPConfigIntegration:
    """Integrationstests für HTTP-Konfiguration."""

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_aiohttp_config_with_standard_config(self, mock_ssl_config: Mock) -> None:
        """Prüft Integration zwischen Standard-Konfiguration und aiohttp."""
        mock_ssl_config.return_value = Mock()

        standard_config = StandardHTTPClientConfig.content_safety()
        session_config = create_aiohttp_session_config(standard_config)

        # Prüfe, dass Standard-Werte übernommen wurden
        timeout = session_config["timeout"]
        assert timeout.total == standard_config.timeout
        assert timeout.connect == standard_config.connect_timeout

    @patch("services.clients.common.http_config.get_ssl_config")
    def test_httpx_config_with_standard_config(self, mock_ssl_config: Mock) -> None:
        """Prüft Integration zwischen Standard-Konfiguration und httpx."""
        mock_ssl_config.return_value = Mock(verify_ssl=True)

        standard_config = StandardHTTPClientConfig.image_generation()
        client_config = create_httpx_client_config(standard_config)

        # Prüfe, dass Standard-Werte übernommen wurden
        assert client_config["timeout"] == standard_config.timeout
