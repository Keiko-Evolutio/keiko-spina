# backend/services/clients/tests/test_content_safety.py
"""Tests für Content Safety Client.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.clients.content_safety import (
    ContentSafetyClient,
    ContentSafetyResult,
    create_content_safety_client,
)


class TestContentSafetyResult:
    """Tests für ContentSafetyResult Dataclass."""

    def test_content_safety_result_creation(self) -> None:
        """Prüft, dass ContentSafetyResult korrekt erstellt wird."""
        raw_data = {"test": "data"}
        result = ContentSafetyResult(
            is_safe=True,
            score=0.2,
            category="safe",
            raw=raw_data
        )

        assert result.is_safe is True
        assert result.score == 0.2
        assert result.category == "safe"
        assert result.raw is raw_data

    def test_content_safety_result_slots(self) -> None:
        """Prüft, dass ContentSafetyResult __slots__ verwendet."""
        result = ContentSafetyResult(True, 0.0, "safe", {})

        # __slots__ verhindert dynamische Attribute
        with pytest.raises(AttributeError):
            result.dynamic_attribute = "test"  # type: ignore


class TestContentSafetyClient:
    """Tests für ContentSafetyClient."""

    @patch("services.clients.content_safety.settings")
    def test_content_safety_client_initialization_with_config(self, mock_settings: Mock) -> None:
        """Prüft, dass ContentSafetyClient mit Konfiguration initialisiert wird."""
        mock_settings.azure_content_safety_endpoint = "https://test.cognitiveservices.azure.com"
        mock_key = Mock()
        mock_key.get_secret_value.return_value = "test-api-key"
        mock_settings.azure_content_safety_key = mock_key

        client = ContentSafetyClient()

        assert client.is_available is True
        assert client._endpoint == "https://test.cognitiveservices.azure.com"
        assert client._api_key == "test-api-key"

    @patch("services.clients.content_safety.settings")
    def test_content_safety_client_initialization_without_config(self, mock_settings: Mock) -> None:
        """Prüft, dass ContentSafetyClient ohne Konfiguration initialisiert wird."""
        mock_settings.azure_content_safety_endpoint = None
        mock_settings.azure_content_safety_key = None

        client = ContentSafetyClient()

        assert client.is_available is False

    def test_content_safety_client_custom_config(self) -> None:
        """Prüft, dass ContentSafetyClient mit benutzerdefinierten Werten initialisiert wird."""
        client = ContentSafetyClient(
            endpoint="https://custom.endpoint.com",
            api_key="custom-key"
        )

        assert client.is_available is True
        assert client._endpoint == "https://custom.endpoint.com"
        assert client._api_key == "custom-key"

    def test_content_safety_client_inheritance(self) -> None:
        """Prüft, dass ContentSafetyClient von RetryableClient erbt."""
        from services.clients.common.retry_utils import RetryableClient

        client = ContentSafetyClient("https://test.com", "key")
        assert isinstance(client, RetryableClient)

    def test_create_request_url(self) -> None:
        """Prüft, dass Request-URL korrekt erstellt wird."""
        client = ContentSafetyClient("https://test.cognitiveservices.azure.com", "key")

        url = client._create_request_url()

        expected_url = (
            "https://test.cognitiveservices.azure.com/contentsafety/text:analyze"
            "?api-version=2024-09-01"
        )
        assert url == expected_url

    def test_create_request_payload(self) -> None:
        """Prüft, dass Request-Payload korrekt erstellt wird."""
        client = ContentSafetyClient("https://test.com", "key")

        payload = client._create_request_payload("Test text")

        expected_payload = {
            "text": "Test text",
            "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
            "outputType": "FourSeverityLevels",
        }
        assert payload == expected_payload

    def test_parse_response_data_list_format(self) -> None:
        """Prüft Parsing von Response-Daten im Listen-Format."""
        client = ContentSafetyClient("https://test.com", "key")

        response_data = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Violence", "severity": 2},
                {"category": "Sexual", "severity": 1},
                {"category": "SelfHarm", "severity": 0},
            ]
        }

        result = client._parse_response_data(response_data)

        assert isinstance(result, ContentSafetyResult)
        assert result.is_safe is False  # max severity 2 > threshold 1
        assert result.score == 2.0 / 3.0  # max severity / max level
        assert result.category == "unsafe"
        assert result.raw == response_data

    def test_parse_response_data_dict_format(self) -> None:
        """Prüft Parsing von Response-Daten im Dictionary-Format."""
        client = ContentSafetyClient("https://test.com", "key")

        response_data = {
            "CategoriesAnalysis": {
                "Hate": {"severity": 0},
                "Violence": {"severity": 1},
                "Sexual": {"severity": 0},
                "SelfHarm": {"severity": 0},
            }
        }

        result = client._parse_response_data(response_data)

        assert result.is_safe is True  # max severity 1 <= threshold 1
        assert result.score == 1.0 / 3.0
        assert result.category == "safe"

    def test_parse_response_data_empty_severities(self) -> None:
        """Prüft Parsing bei leeren Severity-Daten."""
        client = ContentSafetyClient("https://test.com", "key")

        response_data = {"categoriesAnalysis": []}

        result = client._parse_response_data(response_data)

        assert result.is_safe is True  # max severity 0 <= threshold 1
        assert result.score == 0.0
        assert result.category == "safe"

    @pytest.mark.asyncio
    async def test_analyze_text_unavailable_service(self) -> None:
        """Prüft analyze_text bei nicht verfügbarem Service."""
        client = ContentSafetyClient()  # Ohne Konfiguration

        result = await client.analyze_text("Test text")

        assert isinstance(result, ContentSafetyResult)
        assert result.is_safe is True
        assert result.score == 0.0
        assert result.category == "unknown"
        assert result.raw["reason"] == "unavailable"

    @pytest.mark.asyncio
    @patch("services.clients.content_safety.create_aiohttp_session_config")
    @patch("services.clients.content_safety.create_azure_headers")
    async def test_analyze_text_successful_request(
        self,
        mock_create_headers: Mock,
        mock_create_session_config: Mock
    ) -> None:
        """Prüft erfolgreiche analyze_text Anfrage."""
        # Setup mocks
        mock_create_headers.return_value = {"Authorization": "Bearer test"}
        mock_create_session_config.return_value = {}

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Violence", "severity": 1},
            ]
        }

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=mock_session):
            client = ContentSafetyClient("https://test.com", "key")

            result = await client.analyze_text("Test text")

        assert isinstance(result, ContentSafetyResult)
        assert result.is_safe is True
        assert result.category == "safe"

    @pytest.mark.asyncio
    @patch("services.clients.content_safety.create_aiohttp_session_config")
    @patch("services.clients.content_safety.create_azure_headers")
    async def test_analyze_text_http_error(
        self,
        mock_create_headers: Mock,
        mock_create_session_config: Mock
    ) -> None:
        """Prüft analyze_text bei HTTP-Fehler."""
        # Setup mocks
        mock_create_headers.return_value = {"Authorization": "Bearer test"}
        mock_create_session_config.return_value = {}

        # Mock aiohttp session mit Fehler
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"error": "Bad request"}

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=mock_session):
            client = ContentSafetyClient("https://test.com", "key")

            result = await client.analyze_text("Test text")

        # Sollte Fallback-Ergebnis zurückgeben
        assert isinstance(result, ContentSafetyResult)
        assert result.is_safe is True
        assert result.category == "unknown"
        assert "error" in result.raw

    @pytest.mark.asyncio
    async def test_analyze_text_with_retry_logic(self) -> None:
        """Prüft, dass analyze_text Retry-Logik verwendet."""
        client = ContentSafetyClient("https://test.com", "key")

        # Mock _execute_with_retry
        mock_result = ContentSafetyResult(True, 0.0, "safe", {})
        client._execute_with_retry = AsyncMock(return_value=mock_result)

        result = await client.analyze_text("Test text")

        assert result is mock_result
        client._execute_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_analysis_request_integration(self) -> None:
        """Integrationstests für _perform_analysis_request."""
        client = ContentSafetyClient("https://test.com", "key")

        # Mock alle HTTP-bezogenen Komponenten
        with patch("services.clients.content_safety.create_aiohttp_session_config") as mock_config, \
             patch("services.clients.content_safety.create_azure_headers") as mock_headers, \
             patch("aiohttp.ClientSession") as mock_session_class:

            mock_config.return_value = {}
            mock_headers.return_value = {"Authorization": "Bearer test"}

            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "categoriesAnalysis": [{"category": "Hate", "severity": 0}]
            }

            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await client._perform_analysis_request("Test text")

            assert isinstance(result, ContentSafetyResult)
            assert result.is_safe is True


class TestContentSafetyClientLogging:
    """Tests für Content Safety Client Logging."""

    @patch("services.clients.content_safety.logger")
    @patch("services.clients.content_safety.settings")
    def test_client_initialization_logging(self, mock_settings: Mock, mock_logger: Mock) -> None:
        """Prüft, dass Client-Initialisierung geloggt wird."""
        mock_settings.azure_content_safety_endpoint = "https://test.com"
        mock_key = Mock()
        mock_key.get_secret_value.return_value = "key"
        mock_settings.azure_content_safety_key = mock_key

        ContentSafetyClient()

        mock_logger.debug.assert_called()
        log_data = mock_logger.debug.call_args[0][0]
        assert log_data["event"] == "content_safety_client_init"
        assert log_data["available"] is True

    @pytest.mark.asyncio
    @patch("services.clients.content_safety.logger")
    async def test_unavailable_service_logging(self, mock_logger: Mock) -> None:
        """Prüft Logging bei nicht verfügbarem Service."""
        client = ContentSafetyClient()  # Ohne Konfiguration

        await client.analyze_text("Test text")

        mock_logger.debug.assert_called()
        log_data = mock_logger.debug.call_args[0][0]
        assert log_data["event"] == "content_safety_unavailable_fallback"

    @pytest.mark.asyncio
    @patch("services.clients.content_safety.logger")
    async def test_analysis_request_logging(self, mock_logger: Mock) -> None:
        """Prüft Logging bei Analyse-Anfragen."""
        client = ContentSafetyClient("https://test.com", "key")

        # Mock HTTP-Komponenten
        with patch("services.clients.content_safety.create_aiohttp_session_config"), \
             patch("services.clients.content_safety.create_azure_headers"), \
             patch("aiohttp.ClientSession") as mock_session_class:

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"categoriesAnalysis": []}

            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session

            await client._perform_analysis_request("Test text")

            # Prüfe Request-Logging
            debug_calls = list(mock_logger.debug.call_args_list)
            request_logs = [call for call in debug_calls
                          if call[0][0].get("event") == "content_safety_request"]
            assert len(request_logs) > 0


class TestCreateContentSafetyClient:
    """Tests für create_content_safety_client Factory-Funktion."""

    def test_create_content_safety_client_factory(self) -> None:
        """Prüft, dass Factory-Funktion ContentSafetyClient erstellt."""
        client = create_content_safety_client()

        assert isinstance(client, ContentSafetyClient)

    @patch("services.clients.content_safety.settings")
    def test_create_content_safety_client_uses_settings(self, mock_settings: Mock) -> None:
        """Prüft, dass Factory-Funktion Settings verwendet."""
        mock_settings.azure_content_safety_endpoint = "https://factory-test.com"
        mock_key = Mock()
        mock_key.get_secret_value.return_value = "factory-key"
        mock_settings.azure_content_safety_key = mock_key

        client = create_content_safety_client()

        assert client._endpoint == "https://factory-test.com"
        assert client._api_key == "factory-key"
