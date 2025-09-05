"""Unit-Tests für tools_constants.py.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""


import pytest

from .tools_constants import (
    API_VERSIONS,
    CACHE_SETTINGS,
    ERROR_MESSAGES,
    FIELD_NAMES,
    HTTP_HEADERS,
    HTTP_TIMEOUTS,
    MCP_BRIDGE_SETTINGS,
    SCORE_WEIGHTS,
    get_api_version,
    get_error_message,
    get_field_name,
    get_retriever_default,
    get_timeout,
)


class TestHTTPConstants:
    """Tests für HTTP-bezogene Constants."""

    def test_http_timeouts_structure(self) -> None:
        """Prüft die Struktur der HTTP_TIMEOUTS."""
        assert isinstance(HTTP_TIMEOUTS, dict)
        assert "default" in HTTP_TIMEOUTS
        assert "short" in HTTP_TIMEOUTS
        assert "long" in HTTP_TIMEOUTS
        assert all(isinstance(v, int | float) for v in HTTP_TIMEOUTS.values())

    def test_http_headers_structure(self) -> None:
        """Prüft die Struktur der HTTP_HEADERS."""
        assert isinstance(HTTP_HEADERS, dict)
        assert "content_type" in HTTP_HEADERS
        assert "api_key" in HTTP_HEADERS
        assert "json" in HTTP_HEADERS
        assert all(isinstance(v, str) for v in HTTP_HEADERS.values())

    def test_get_timeout_default(self) -> None:
        """Prüft get_timeout mit Default-Wert."""
        result = get_timeout()
        assert result == HTTP_TIMEOUTS["default"]

    def test_get_timeout_specific(self) -> None:
        """Prüft get_timeout mit spezifischem Wert."""
        result = get_timeout("short")
        assert result == HTTP_TIMEOUTS["short"]

    def test_get_timeout_unknown(self) -> None:
        """Prüft get_timeout mit unbekanntem Wert."""
        result = get_timeout("unknown")
        assert result == HTTP_TIMEOUTS["default"]


class TestAPIVersions:
    """Tests für API-Versionen."""

    def test_api_versions_structure(self) -> None:
        """Prüft die Struktur der API_VERSIONS."""
        assert isinstance(API_VERSIONS, dict)
        assert "azure_search" in API_VERSIONS
        assert all(isinstance(v, str) for v in API_VERSIONS.values())

    def test_get_api_version_existing(self) -> None:
        """Prüft get_api_version mit existierendem Service."""
        result = get_api_version("azure_search")
        assert result == API_VERSIONS["azure_search"]

    def test_get_api_version_unknown(self) -> None:
        """Prüft get_api_version mit unbekanntem Service."""
        result = get_api_version("unknown_service")
        assert result == "latest"


class TestCacheSettings:
    """Tests für Cache-Einstellungen."""

    def test_cache_settings_structure(self) -> None:
        """Prüft die Struktur der CACHE_SETTINGS."""
        assert isinstance(CACHE_SETTINGS, dict)
        assert "ttl" in CACHE_SETTINGS
        assert "max_size" in CACHE_SETTINGS
        assert isinstance(CACHE_SETTINGS["ttl"], int)
        assert isinstance(CACHE_SETTINGS["max_size"], int)


class TestScoreWeights:
    """Tests für Score-Gewichtungen."""

    def test_score_weights_structure(self) -> None:
        """Prüft die Struktur der SCORE_WEIGHTS."""
        assert isinstance(SCORE_WEIGHTS, dict)
        assert "vector" in SCORE_WEIGHTS
        assert "keyword" in SCORE_WEIGHTS
        assert all(isinstance(v, int | float) for v in SCORE_WEIGHTS.values())

    def test_score_weights_sum_reasonable(self) -> None:
        """Prüft dass Vector + Keyword Gewichtungen sinnvoll sind."""
        vector_weight = SCORE_WEIGHTS["vector"]
        keyword_weight = SCORE_WEIGHTS["keyword"]
        total = vector_weight + keyword_weight
        assert 0.8 <= total <= 1.2  # Sollte ungefähr 1.0 sein


class TestFieldNames:
    """Tests für Standard-Feldnamen."""

    def test_field_names_structure(self) -> None:
        """Prüft die Struktur der FIELD_NAMES."""
        assert isinstance(FIELD_NAMES, dict)
        required_fields = ["content", "text", "embedding", "score", "metadata"]
        for field in required_fields:
            assert field in FIELD_NAMES
            assert isinstance(FIELD_NAMES[field], str)

    def test_get_field_name_existing(self) -> None:
        """Prüft get_field_name mit existierendem Feld."""
        result = get_field_name("content")
        assert result == FIELD_NAMES["content"]

    def test_get_field_name_unknown(self) -> None:
        """Prüft get_field_name mit unbekanntem Feld."""
        result = get_field_name("unknown_field")
        assert result == "unknown_field"


class TestMCPBridgeSettings:
    """Tests für MCP-Bridge-Einstellungen."""

    def test_mcp_bridge_settings_structure(self) -> None:
        """Prüft die Struktur der MCP_BRIDGE_SETTINGS."""
        assert isinstance(MCP_BRIDGE_SETTINGS, dict)
        assert "capability_mappings" in MCP_BRIDGE_SETTINGS
        assert "scoring_weights" in MCP_BRIDGE_SETTINGS

    def test_capability_mappings_structure(self) -> None:
        """Prüft die Struktur der Capability-Mappings."""
        mappings = MCP_BRIDGE_SETTINGS["capability_mappings"]
        assert isinstance(mappings, dict)
        assert "search" in mappings
        assert "weather" in mappings

        # Jedes Mapping sollte ein Set von Strings sein
        for capability, keywords in mappings.items():
            assert isinstance(capability, str)
            assert isinstance(keywords, set)
            assert all(isinstance(keyword, str) for keyword in keywords)

    def test_scoring_weights_structure(self) -> None:
        """Prüft die Struktur der Scoring-Gewichtungen."""
        weights = MCP_BRIDGE_SETTINGS["scoring_weights"]
        assert isinstance(weights, dict)
        assert "name_match" in weights
        assert "description_match" in weights
        assert "capability_match" in weights
        assert all(isinstance(v, int) for v in weights.values())


class TestErrorMessages:
    """Tests für Error-Messages."""

    def test_error_messages_structure(self) -> None:
        """Prüft die Struktur der ERROR_MESSAGES."""
        assert isinstance(ERROR_MESSAGES, dict)
        assert "invalid_tool_id" in ERROR_MESSAGES
        assert "tool_execution_error" in ERROR_MESSAGES
        assert all(isinstance(v, str) for v in ERROR_MESSAGES.values())

    def test_get_error_message_existing(self) -> None:
        """Prüft get_error_message mit existierendem Error-Type."""
        result = get_error_message("tool_execution_error")
        assert result == ERROR_MESSAGES["tool_execution_error"]

    def test_get_error_message_with_params(self) -> None:
        """Prüft get_error_message mit Parametern."""
        result = get_error_message("invalid_tool_id", tool_id="test:tool")
        assert "test:tool" in result

    def test_get_error_message_unknown(self) -> None:
        """Prüft get_error_message mit unbekanntem Error-Type."""
        result = get_error_message("unknown_error")
        assert "unknown_error" in result


class TestRetrieverDefaults:
    """Tests für Retriever-Default-Werte."""

    def test_get_retriever_default_existing(self) -> None:
        """Prüft get_retriever_default mit existierendem Retriever."""
        result = get_retriever_default("azure_search", "top_k")
        assert isinstance(result, int)
        assert result > 0

    def test_get_retriever_default_unknown_retriever(self) -> None:
        """Prüft get_retriever_default mit unbekanntem Retriever."""
        result = get_retriever_default("unknown_retriever", "top_k")
        assert result is None

    def test_get_retriever_default_unknown_setting(self) -> None:
        """Prüft get_retriever_default mit unbekannter Einstellung."""
        result = get_retriever_default("azure_search", "unknown_setting")
        assert result is None


class TestConstantsIntegrity:
    """Tests für die Integrität der Constants."""

    def test_no_empty_values(self) -> None:
        """Prüft dass keine Constants leer sind."""
        constants_to_check = [
            HTTP_TIMEOUTS, HTTP_HEADERS, API_VERSIONS,
            CACHE_SETTINGS, SCORE_WEIGHTS, FIELD_NAMES
        ]

        for constant_dict in constants_to_check:
            assert len(constant_dict) > 0
            for key, value in constant_dict.items():
                assert key  # Key nicht leer
                assert value is not None  # Value nicht None

    def test_timeout_values_positive(self) -> None:
        """Prüft dass alle Timeout-Werte positiv sind."""
        for timeout_value in HTTP_TIMEOUTS.values():
            assert timeout_value > 0

    def test_score_weights_valid_range(self) -> None:
        """Prüft dass Score-Gewichtungen in gültigem Bereich sind."""
        for weight in SCORE_WEIGHTS.values():
            assert 0.0 <= weight <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
