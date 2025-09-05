"""Unit-Tests für retriever_utils.py.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from .retriever_utils import (
    combine_scores,
    cosine_similarity,
    defensive_exception_handler,
    euclidean_distance,
    http_client_with_timeout,
    merge_document_metadata,
    normalize_document_result,
    text_hash_function,
    truncate_text,
)


class TestMathematicalFunctions:
    """Tests für mathematische Utility-Funktionen."""

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Prüft Cosinus-Ähnlichkeit für identische Vektoren."""
        vector = [1.0, 2.0, 3.0]
        result = cosine_similarity(vector, vector)
        assert abs(result - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Prüft Cosinus-Ähnlichkeit für orthogonale Vektoren."""
        vector_a = [1.0, 0.0]
        vector_b = [0.0, 1.0]
        result = cosine_similarity(vector_a, vector_b)
        assert abs(result - 0.0) < 1e-10

    def test_cosine_similarity_empty_vectors(self) -> None:
        """Prüft Cosinus-Ähnlichkeit für leere Vektoren."""
        result = cosine_similarity([], [1.0, 2.0])
        assert result == 0.0

        result = cosine_similarity([1.0, 2.0], [])
        assert result == 0.0

    def test_cosine_similarity_zero_vectors(self) -> None:
        """Prüft Cosinus-Ähnlichkeit für Null-Vektoren."""
        result = cosine_similarity([0.0, 0.0], [1.0, 2.0])
        assert result == 0.0

    def test_euclidean_distance_identical_vectors(self) -> None:
        """Prüft Euklidische Distanz für identische Vektoren."""
        vector = [1.0, 2.0, 3.0]
        result = euclidean_distance(vector, vector)
        assert abs(result - 0.0) < 1e-10

    def test_euclidean_distance_different_lengths(self) -> None:
        """Prüft Euklidische Distanz für Vektoren unterschiedlicher Länge."""
        result = euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])
        assert result == float("inf")

    def test_euclidean_distance_empty_vectors(self) -> None:
        """Prüft Euklidische Distanz für leere Vektoren."""
        result = euclidean_distance([], [1.0, 2.0])
        assert result == float("inf")


class TestHTTPClientUtilities:
    """Tests für HTTP-Client-Utilities."""

    @pytest.mark.asyncio
    async def test_http_client_with_timeout_success(self) -> None:
        """Prüft erfolgreichen HTTP-Request."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await http_client_with_timeout(
                url="https://example.com",
                headers={"Content-Type": "application/json"},
                payload={"test": "data"}
            )

            assert result == {"status": "success"}

    @pytest.mark.asyncio
    async def test_http_client_with_timeout_get_method(self) -> None:
        """Prüft HTTP GET-Request."""
        mock_response = Mock()
        mock_response.json.return_value = {"method": "GET"}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            result = await http_client_with_timeout(
                url="https://example.com",
                headers={},
                payload={"param": "value"},
                method="GET"
            )

            assert result == {"method": "GET"}

    @pytest.mark.asyncio
    async def test_http_client_with_timeout_custom_timeout(self) -> None:
        """Prüft HTTP-Request mit custom Timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {"timeout": "custom"}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await http_client_with_timeout(
                url="https://example.com",
                headers={},
                payload={},
                timeout=30.0
            )

            assert result == {"timeout": "custom"}


class TestExceptionHandling:
    """Tests für Exception-Handling-Utilities."""

    def test_defensive_exception_handler_with_fallback(self) -> None:
        """Prüft Exception-Handler mit Fallback-Wert."""
        exception = ValueError("Test error")
        result = defensive_exception_handler(
            "test_operation",
            exception,
            fallback_value=[]
        )
        assert result == []

    def test_defensive_exception_handler_without_fallback(self) -> None:
        """Prüft Exception-Handler ohne Fallback-Wert."""
        exception = RuntimeError("Test error")
        result = defensive_exception_handler("test_operation", exception)
        assert result is None

    def test_defensive_exception_handler_logging_levels(self) -> None:
        """Prüft verschiedene Log-Level im Exception-Handler."""
        exception = Exception("Test")

        # Test debug level - sollte ohne Fehler ausgeführt werden
        result_debug = defensive_exception_handler("test", exception, log_level="debug")
        assert result_debug is None

        # Test warning level (default) - sollte ohne Fehler ausgeführt werden
        result_warning = defensive_exception_handler("test", exception)
        assert result_warning is None

        # Test mit Fallback-Wert
        result_fallback = defensive_exception_handler("test", exception, fallback_value=[])
        assert result_fallback == []


class TestTextProcessing:
    """Tests für Text-Processing-Utilities."""

    def test_text_hash_function_consistency(self) -> None:
        """Prüft Konsistenz der Text-Hash-Funktion."""
        text = "test string"
        hash1 = text_hash_function(text)
        hash2 = text_hash_function(text)
        assert hash1 == hash2

    def test_text_hash_function_different_algorithms(self) -> None:
        """Prüft verschiedene Hash-Algorithmen."""
        text = "test string"
        sha1_hash = text_hash_function(text, "sha1")
        md5_hash = text_hash_function(text, "md5")
        sha256_hash = text_hash_function(text, "sha256")

        assert sha1_hash != md5_hash
        assert sha1_hash != sha256_hash
        assert md5_hash != sha256_hash

    def test_truncate_text_no_truncation_needed(self) -> None:
        """Prüft Text-Kürzung wenn keine Kürzung nötig ist."""
        text = "short text"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_truncate_text_with_truncation(self) -> None:
        """Prüft Text-Kürzung mit tatsächlicher Kürzung."""
        text = "this is a very long text that needs to be truncated"
        result = truncate_text(text, max_length=20, suffix="...")
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_text_custom_suffix(self) -> None:
        """Prüft Text-Kürzung mit custom Suffix."""
        text = "long text here"
        result = truncate_text(text, max_length=10, suffix=" [...]")
        assert result.endswith(" [...]")


class TestDocumentProcessing:
    """Tests für Document-Processing-Utilities."""

    def test_normalize_document_result_with_content(self) -> None:
        """Prüft Dokument-Normalisierung mit Content-Feld."""
        document = {
            "content": "test content",
            "title": "Test Document",
            "author": "Test Author"
        }
        result = normalize_document_result(document, 0.8)

        assert result["text"] == "test content"
        assert result["score"] == 0.8
        assert result["metadata"]["title"] == "Test Document"
        assert result["metadata"]["author"] == "Test Author"

    def test_normalize_document_result_with_text_field(self) -> None:
        """Prüft Dokument-Normalisierung mit Text-Feld."""
        document = {
            "text": "test text",
            "category": "test"
        }
        result = normalize_document_result(document, 0.5)

        assert result["text"] == "test text"
        assert result["score"] == 0.5
        assert result["metadata"]["category"] == "test"

    def test_normalize_document_result_auto_detect(self) -> None:
        """Prüft Dokument-Normalisierung mit Auto-Detection."""
        document = {
            "description": "test description",
            "id": "123"
        }
        result = normalize_document_result(document, 0.3)

        assert result["text"] == "test description"
        assert result["score"] == 0.3

    def test_merge_document_metadata(self) -> None:
        """Prüft Metadaten-Zusammenführung."""
        doc1 = {
            "text": "content1",
            "score": 0.8,
            "metadata": {"source": "doc1", "type": "text"}
        }
        doc2 = {
            "text": "content2",
            "score": 0.6,
            "metadata": {"author": "test", "type": "other"}
        }

        result = merge_document_metadata(doc1, doc2)

        assert result["text"] == "content1"
        assert result["score"] == 0.8
        assert result["metadata"]["source"] == "doc1"  # doc1 hat Priorität
        assert result["metadata"]["author"] == "test"   # von doc2
        assert result["metadata"]["type"] == "text"     # doc1 hat Priorität


class TestScoringUtilities:
    """Tests für Scoring-Utilities."""

    def test_combine_scores_weighted_average(self) -> None:
        """Prüft Score-Kombination mit gewichtetem Durchschnitt."""
        scores = [0.8, 0.6, 0.4]
        weights = [2.0, 1.0, 1.0]
        result = combine_scores(scores, weights, method="weighted_average")

        expected = (0.8 * 2.0 + 0.6 * 1.0 + 0.4 * 1.0) / (2.0 + 1.0 + 1.0)
        assert abs(result - expected) < 1e-10

    def test_combine_scores_max_method(self) -> None:
        """Prüft Score-Kombination mit Maximum-Methode."""
        scores = [0.3, 0.8, 0.5]
        result = combine_scores(scores, method="max")
        assert result == 0.8

    def test_combine_scores_min_method(self) -> None:
        """Prüft Score-Kombination mit Minimum-Methode."""
        scores = [0.3, 0.8, 0.5]
        result = combine_scores(scores, method="min")
        assert result == 0.3

    def test_combine_scores_empty_list(self) -> None:
        """Prüft Score-Kombination mit leerer Liste."""
        result = combine_scores([])
        assert result == 0.0

    def test_combine_scores_mismatched_weights(self) -> None:
        """Prüft Score-Kombination mit nicht passenden Gewichtungen."""
        scores = [0.8, 0.6, 0.4]
        weights = [2.0]  # Zu wenige Gewichtungen
        result = combine_scores(scores, weights)

        # Sollte automatisch Gewichtungen ergänzen
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
