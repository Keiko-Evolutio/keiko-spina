"""Abstrakte Basis-Klasse für alle Retriever.

Stellt eine einheitliche Schnittstelle und gemeinsame Funktionalität für alle
Retriever-Implementierungen bereit, um Code-Duplikation zu eliminieren und
Konsistenz zu gewährleisten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .retriever_utils import (
    defensive_exception_handler,
    normalize_document_result,
)
from .tools_constants import (
    RETRIEVAL_DEFAULTS,
    get_timeout,
)

logger = get_logger(__name__)


# =============================================================================
# Basis-Konfiguration
# =============================================================================

@dataclass
class BaseRetrieverConfig:
    """Basis-Konfiguration für alle Retriever.

    Stellt gemeinsame Konfigurationsoptionen bereit, die von allen
    Retriever-Implementierungen verwendet werden können.

    Attributes:
        top_k: Anzahl der zurückzugebenden Ergebnisse
        timeout: Timeout für Retrieval-Operationen in Sekunden
        enable_fallback: Ob Fallback-Mechanismen aktiviert sind
        normalize_results: Ob Ergebnisse normalisiert werden sollen
    """

    top_k: int = RETRIEVAL_DEFAULTS["top_k"]
    timeout: float = get_timeout("default")
    enable_fallback: bool = True
    normalize_results: bool = True

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte nach Initialisierung."""
        if self.top_k < RETRIEVAL_DEFAULTS["min_top_k"]:
            self.top_k = RETRIEVAL_DEFAULTS["min_top_k"]
        elif self.top_k > RETRIEVAL_DEFAULTS["max_top_k"]:
            self.top_k = RETRIEVAL_DEFAULTS["max_top_k"]

        if self.timeout <= 0:
            self.timeout = get_timeout("default")


# =============================================================================
# Abstrakte Basis-Klasse
# =============================================================================

class BaseRetriever(ABC):
    """Abstrakte Basis-Klasse für alle Retriever.

    Stellt eine einheitliche Schnittstelle und gemeinsame Funktionalität bereit,
    einschließlich standardisiertem Error-Handling, Tracing und Result-Normalisierung.

    Subklassen müssen nur die _perform_retrieval() Methode implementieren.
    """

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialisiert den Basis-Retriever.

        Args:
            config: Retriever-Konfiguration
        """
        self._config = config
        self._retriever_name = self.__class__.__name__

    @property
    def config(self) -> BaseRetrieverConfig:
        """Gibt die aktuelle Konfiguration zurück."""
        return self._config

    @property
    def retriever_name(self) -> str:
        """Gibt den Namen des Retrievers zurück."""
        return self._retriever_name

    @abstractmethod
    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Implementiert die spezifische Retrieval-Logik.

        Diese Methode muss von Subklassen implementiert werden und enthält
        die eigentliche Retrieval-Logik ohne Error-Handling oder Normalisierung.

        Args:
            query: Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Liste von Dokumenten als Dictionaries

        Raises:
            Exception: Bei Retrieval-Fehlern
        """

    @trace_function("retriever.base.aretrieve", {"component": "retriever"})
    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Einheitliche aretrieve-Implementierung mit Error-Handling.

        Diese Methode stellt die öffentliche Schnittstelle dar und kümmert sich um:
        - Parameter-Validierung
        - Error-Handling mit Fallback
        - Result-Normalisierung
        - Tracing und Logging

        Args:
            query: Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse (optional)

        Returns:
            Liste normalisierter Dokumente

        Examples:
            >>> config = BaseRetrieverConfig()
            >>> # Example assuming a concrete retriever implementation exists
            >>> # retriever = ConcreteRetriever(config)
            >>> # results = await retriever.aretrieve("test query", top_k=5)
            >>> # len(results) <= 5
            >>> # True
        """
        # Parameter-Validierung
        if not query or not query.strip():
            logger.warning(f"{self._retriever_name}: Leere Query erhalten")
            return []

        k = top_k or self._config.top_k
        k = max(RETRIEVAL_DEFAULTS["min_top_k"],
                min(k, RETRIEVAL_DEFAULTS["max_top_k"]))

        # Retrieval mit Error-Handling
        try:
            logger.debug(f"{self._retriever_name}: Starte Retrieval für '{query}' (top_k={k})")

            results = await self._perform_retrieval(query, k)

            # Result-Normalisierung
            if self._config.normalize_results:
                results = self._normalize_results(results)

            logger.debug(f"{self._retriever_name}: {len(results)} Ergebnisse gefunden")
            return results

        except Exception as exc:
            if self._config.enable_fallback:
                return defensive_exception_handler(  # type: ignore[no-any-return]
                    f"{self._retriever_name}.aretrieve",
                    exc,
                    []
                )
            raise

    def _normalize_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalisiert Retrieval-Ergebnisse zu einheitlichem Format.

        Args:
            results: Rohe Retrieval-Ergebnisse

        Returns:
            Normalisierte Ergebnisse
        """
        normalized = []
        for result in results:
            try:
                normalized_doc = normalize_document_result(
                    result,
                    result.get("score", 0.0)
                )
                normalized.append(normalized_doc)
            except Exception as exc:
                logger.debug(f"{self._retriever_name}: Normalisierung fehlgeschlagen: {exc}")
                # Fallback: Original-Dokument beibehalten
                normalized.append(result)

        return normalized

    async def health_check(self) -> dict[str, Any]:
        """Führt einen Gesundheitscheck des Retrievers durch.

        Returns:
            Status-Dictionary mit Gesundheitsinformationen
        """
        try:
            # Einfacher Test-Query
            test_results = await self.aretrieve("test", top_k=1)

            return {
                "status": "healthy",
                "retriever_name": self._retriever_name,
                "config": {
                    "top_k": self._config.top_k,
                    "timeout": self._config.timeout,
                },
                "test_query_successful": True,
                "test_results_count": len(test_results),
            }

        except Exception as exc:
            return {
                "status": "unhealthy",
                "retriever_name": self._retriever_name,
                "error": str(exc),
                "test_query_successful": False,
            }

    def get_stats(self) -> dict[str, Any]:
        """Gibt Statistiken über den Retriever zurück.

        Returns:
            Statistik-Dictionary
        """
        return {
            "retriever_name": self._retriever_name,
            "config": {
                "top_k": self._config.top_k,
                "timeout": self._config.timeout,
                "enable_fallback": self._config.enable_fallback,
                "normalize_results": self._config.normalize_results,
            },
            "capabilities": self._get_capabilities(),
        }

    def _get_capabilities(self) -> list[str]:
        """Gibt die Capabilities des Retrievers zurück.

        Kann von Subklassen überschrieben werden.

        Returns:
            Liste der unterstützten Capabilities
        """
        return ["text_retrieval", "async_retrieval"]


# =============================================================================
# Utility-Klassen
# =============================================================================

class RetrieverRegistry:
    """Registry für verfügbare Retriever-Implementierungen.

    Ermöglicht dynamische Registrierung und Erstellung von Retrievern.
    """

    def __init__(self) -> None:
        self._retrievers: dict[str, type[BaseRetriever]] = {}

    def register(self, name: str, retriever_class: type[BaseRetriever]) -> None:
        """Registriert eine Retriever-Klasse.

        Args:
            name: Name des Retrievers
            retriever_class: Retriever-Klasse
        """
        self._retrievers[name] = retriever_class
        logger.debug(f"Retriever '{name}' registriert")

    def create(
        self,
        name: str,
        config: BaseRetrieverConfig
    ) -> BaseRetriever | None:
        """Erstellt eine Retriever-Instanz.

        Args:
            name: Name des Retrievers
            config: Konfiguration

        Returns:
            Retriever-Instanz oder None wenn nicht gefunden
        """
        retriever_class = self._retrievers.get(name)
        if retriever_class:
            return retriever_class(config)
        return None

    def list_available(self) -> list[str]:
        """Gibt Liste verfügbarer Retriever zurück."""
        return list(self._retrievers.keys())


# Globale Registry-Instanz
retriever_registry = RetrieverRegistry()


__all__ = [
    "BaseRetriever",
    "BaseRetrieverConfig",
    "RetrieverRegistry",
    "retriever_registry",
]
