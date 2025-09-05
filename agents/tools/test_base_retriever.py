"""Unit-Tests für base_retriever.py.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""

from typing import Any
from unittest.mock import patch

import pytest

from .base_retriever import (
    BaseRetriever,
    BaseRetrieverConfig,
    RetrieverRegistry,
    retriever_registry,
)


class TestBaseRetrieverConfig:
    """Tests für BaseRetrieverConfig."""

    def test_default_values(self) -> None:
        """Prüft Default-Werte der Konfiguration."""
        config = BaseRetrieverConfig()
        assert config.top_k == 5
        assert config.timeout > 0
        assert config.enable_fallback is True
        assert config.normalize_results is True

    def test_validation_top_k_too_small(self) -> None:
        """Prüft Validierung für zu kleinen top_k Wert."""
        config = BaseRetrieverConfig(top_k=0)
        assert config.top_k >= 1

    def test_validation_top_k_too_large(self) -> None:
        """Prüft Validierung für zu großen top_k Wert."""
        config = BaseRetrieverConfig(top_k=1000)
        assert config.top_k <= 100

    def test_validation_negative_timeout(self) -> None:
        """Prüft Validierung für negativen Timeout."""
        config = BaseRetrieverConfig(timeout=-1.0)
        assert config.timeout > 0

    def test_custom_values(self) -> None:
        """Prüft custom Konfigurationswerte."""
        config = BaseRetrieverConfig(
            top_k=10,
            timeout=30.0,
            enable_fallback=False,
            normalize_results=False
        )
        assert config.top_k == 10
        assert config.timeout == 30.0
        assert config.enable_fallback is False
        assert config.normalize_results is False


class ConcreteRetriever(BaseRetriever):
    """Konkrete Implementierung für Tests."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        super().__init__(config)
        self.perform_retrieval_called = False
        self.perform_retrieval_args = None

    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Test-Implementierung der Retrieval-Logik."""
        self.perform_retrieval_called = True
        self.perform_retrieval_args = (query, top_k)

        return [
            {"content": f"Result for {query}", "score": 0.8},
            {"content": f"Another result for {query}", "score": 0.6}
        ]


class FailingRetriever(BaseRetriever):
    """Retriever der immer fehlschlägt für Error-Handling Tests."""

    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Wirft immer eine Exception."""
        raise RuntimeError("Test error")


class TestBaseRetriever:
    """Tests für BaseRetriever."""

    def test_initialization(self) -> None:
        """Prüft Initialisierung des BaseRetriever."""
        config = BaseRetrieverConfig(top_k=10)
        retriever = ConcreteRetriever(config)

        assert retriever.config == config
        assert retriever.retriever_name == "ConcreteRetriever"

    @pytest.mark.asyncio
    async def test_aretrieve_success(self) -> None:
        """Prüft erfolgreiche aretrieve-Operation."""
        config = BaseRetrieverConfig(top_k=5)
        retriever = ConcreteRetriever(config)

        results = await retriever.aretrieve("test query", top_k=3)

        assert retriever.perform_retrieval_called
        assert retriever.perform_retrieval_args == ("test query", 3)
        assert len(results) == 2
        assert results[0]["text"] == "Result for test query"

    @pytest.mark.asyncio
    async def test_aretrieve_default_top_k(self) -> None:
        """Prüft aretrieve mit Default top_k."""
        config = BaseRetrieverConfig(top_k=7)
        retriever = ConcreteRetriever(config)

        await retriever.aretrieve("test query")

        assert retriever.perform_retrieval_args[1] == 7

    @pytest.mark.asyncio
    async def test_aretrieve_empty_query(self) -> None:
        """Prüft aretrieve mit leerer Query."""
        config = BaseRetrieverConfig()
        retriever = ConcreteRetriever(config)

        results = await retriever.aretrieve("")

        assert results == []
        assert not retriever.perform_retrieval_called

    @pytest.mark.asyncio
    async def test_aretrieve_whitespace_query(self) -> None:
        """Prüft aretrieve mit Whitespace-Query."""
        config = BaseRetrieverConfig()
        retriever = ConcreteRetriever(config)

        results = await retriever.aretrieve("   ")

        assert results == []
        assert not retriever.perform_retrieval_called

    @pytest.mark.asyncio
    async def test_aretrieve_top_k_validation(self) -> None:
        """Prüft top_k Validierung in aretrieve."""
        config = BaseRetrieverConfig()
        retriever = ConcreteRetriever(config)

        # Zu kleiner Wert
        await retriever.aretrieve("test", top_k=0)
        assert retriever.perform_retrieval_args[1] >= 1

        # Zu großer Wert
        await retriever.aretrieve("test", top_k=1000)
        assert retriever.perform_retrieval_args[1] <= 100

    @pytest.mark.asyncio
    async def test_aretrieve_error_handling_with_fallback(self) -> None:
        """Prüft Error-Handling mit aktiviertem Fallback."""
        config = BaseRetrieverConfig(enable_fallback=True)
        retriever = FailingRetriever(config)

        results = await retriever.aretrieve("test query")

        assert results == []  # Fallback-Wert

    @pytest.mark.asyncio
    async def test_aretrieve_error_handling_without_fallback(self) -> None:
        """Prüft Error-Handling ohne Fallback."""
        config = BaseRetrieverConfig(enable_fallback=False)
        retriever = FailingRetriever(config)

        with pytest.raises(RuntimeError, match="Test error"):
            await retriever.aretrieve("test query")

    @pytest.mark.asyncio
    async def test_normalize_results_enabled(self) -> None:
        """Prüft Ergebnis-Normalisierung wenn aktiviert."""
        config = BaseRetrieverConfig(normalize_results=True)
        retriever = ConcreteRetriever(config)

        results = await retriever.aretrieve("test query")

        # Ergebnisse sollten normalisiert sein
        assert "text" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]

    @pytest.mark.asyncio
    async def test_normalize_results_disabled(self) -> None:
        """Prüft dass Normalisierung übersprungen wird wenn deaktiviert."""
        config = BaseRetrieverConfig(normalize_results=False)
        retriever = ConcreteRetriever(config)

        with patch.object(retriever, "_normalize_results") as mock_normalize:
            await retriever.aretrieve("test query")
            mock_normalize.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Prüft erfolgreichen Gesundheitscheck."""
        config = BaseRetrieverConfig()
        retriever = ConcreteRetriever(config)

        health = await retriever.health_check()

        assert health["status"] == "healthy"
        assert health["retriever_name"] == "ConcreteRetriever"
        assert health["test_query_successful"] is True
        assert health["test_results_count"] == 2

    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        """Prüft Gesundheitscheck bei Fehlern."""
        config = BaseRetrieverConfig(enable_fallback=False)  # Fallback deaktivieren
        retriever = FailingRetriever(config)

        health = await retriever.health_check()

        assert health["status"] == "unhealthy"
        assert health["retriever_name"] == "FailingRetriever"
        assert health["test_query_successful"] is False
        assert "error" in health

    def test_get_stats(self) -> None:
        """Prüft Statistik-Abruf."""
        config = BaseRetrieverConfig(top_k=10, timeout=30.0)
        retriever = ConcreteRetriever(config)

        stats = retriever.get_stats()

        assert stats["retriever_name"] == "ConcreteRetriever"
        assert stats["config"]["top_k"] == 10
        assert stats["config"]["timeout"] == 30.0
        assert "capabilities" in stats

    def test_get_capabilities_default(self) -> None:
        """Prüft Default-Capabilities."""
        config = BaseRetrieverConfig()
        retriever = ConcreteRetriever(config)

        capabilities = retriever._get_capabilities()

        assert "text_retrieval" in capabilities
        assert "async_retrieval" in capabilities


class TestRetrieverRegistry:
    """Tests für RetrieverRegistry."""

    def test_register_retriever(self) -> None:
        """Prüft Registrierung eines Retrievers."""
        registry = RetrieverRegistry()
        registry.register("test_retriever", ConcreteRetriever)

        available = registry.list_available()
        assert "test_retriever" in available

    def test_create_retriever_success(self) -> None:
        """Prüft erfolgreiche Retriever-Erstellung."""
        registry = RetrieverRegistry()
        registry.register("test_retriever", ConcreteRetriever)

        config = BaseRetrieverConfig()
        retriever = registry.create("test_retriever", config)

        assert isinstance(retriever, ConcreteRetriever)
        assert retriever.config == config

    def test_create_retriever_unknown(self) -> None:
        """Prüft Retriever-Erstellung für unbekannten Namen."""
        registry = RetrieverRegistry()

        config = BaseRetrieverConfig()
        retriever = registry.create("unknown_retriever", config)

        assert retriever is None

    def test_list_available_empty(self) -> None:
        """Prüft Liste verfügbarer Retriever wenn leer."""
        registry = RetrieverRegistry()
        available = registry.list_available()
        assert available == []

    def test_global_registry_instance(self) -> None:
        """Prüft dass globale Registry-Instanz existiert."""
        assert retriever_registry is not None
        assert isinstance(retriever_registry, RetrieverRegistry)


if __name__ == "__main__":
    pytest.main([__file__])
