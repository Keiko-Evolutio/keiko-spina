"""Azure Cosmos DB Vector Search Retriever (refactored, enterprise-grade).

Refactored mit BaseRetriever, konsolidierten Protocols und Utilities
zur Eliminierung von Code-Duplikation und Verbesserung der Wartbarkeit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .base_retriever import BaseRetriever, BaseRetrieverConfig
from .retriever_utils import (
    cosine_similarity,
    normalize_document_result,
)
from .tools_constants import (
    RETRIEVAL_DEFAULTS,
    get_field_name,
)

if TYPE_CHECKING:
    from .retriever_protocols import EmbeddingFunction

logger = get_logger(__name__)


@dataclass(slots=True)
class CosmosVectorConfig(BaseRetrieverConfig):
    """Konfiguration für Cosmos Vector Search."""

    collection: str = "documents"
    candidate_multiplier: int = RETRIEVAL_DEFAULTS["default_multiplier"]


class CosmosVectorRetriever(BaseRetriever):
    """Asynchroner Retriever auf Basis von Cosmos DB Vektor-Feldern.

    Refactored mit BaseRetriever für einheitliches Error-Handling,
    konsolidierten Utilities und Protocol-basierte Embedding-Funktion.
    """

    def __init__(
        self,
        config: CosmosVectorConfig,
        *,
        embedder: EmbeddingFunction | None = None
    ) -> None:
        """Initialisiert den Cosmos Vector Retriever."""
        super().__init__(config)
        self._cosmos_config = config
        self._embedder = embedder

    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Implementiert Cosmos DB Vector Search Retrieval-Logik.

        Args:
            query: Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Liste von Dokumenten aus Cosmos DB

        Raises:
            Exception: Bei Cosmos DB-Fehlern
        """
        from storage.cache.redis_cache import get_cached_cosmos_container

        cfg = self._cosmos_config

        async with get_cached_cosmos_container() as cached:
            if not cached:
                return []
            container = cached.container  # type: ignore[attr-defined]

            # Kandidaten-Suche mit Multiplier aus Constants
            candidate_count = top_k * cfg.candidate_multiplier
            query_sql = "SELECT TOP @k c.id, c.content, c.text, c.embedding FROM c WHERE c.category=@cat"
            params = [
                {"name": "@k", "value": candidate_count},
                {"name": "@cat", "value": cfg.collection},
            ]

            candidates: list[dict[str, Any]] = []
            async for item in container.query_items(
                query=query_sql, parameters=params, enable_cross_partition_query=True
            ):
                candidates.append(item)

            if not candidates:
                return []

            # Vector-basierte Suche wenn Embedder verfügbar
            if self._embedder is not None:
                return await self._vector_search(query, candidates, top_k)
            # Fallback: Substring-basierte Suche
            return self._substring_search(query, candidates, top_k)

    async def _vector_search(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int
    ) -> list[dict[str, Any]]:
        """Führt vektor-basierte Ähnlichkeitssuche durch.

        Args:
            query: Suchanfrage
            candidates: Kandidaten-Dokumente
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Nach Ähnlichkeit sortierte Dokumente
        """
        try:
            # Query-Embedding erstellen
            query_vector = (await self._embedder.aembed([query]))[0]

            # Ähnlichkeits-Scores berechnen mit konsolidierter Utility
            scored_candidates: list[tuple[dict[str, Any], float]] = []
            for candidate in candidates:
                embedding = candidate.get(get_field_name("embedding")) or []
                similarity_score = cosine_similarity(query_vector, embedding)
                scored_candidates.append((candidate, similarity_score))

            # Nach Score sortieren und Top-K auswählen
            ranked_candidates = sorted(
                scored_candidates,
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            # Ergebnisse normalisieren
            results = []
            for candidate, score in ranked_candidates:
                normalized_doc = normalize_document_result(candidate, score)
                results.append(normalized_doc)

            return results

        except Exception as exc:
            logger.debug(f"CosmosVectorRetriever Vector-Search Fallback: {exc}")
            # Fallback zu Substring-Suche
            return self._substring_search(query, candidates, top_k)

    def _substring_search(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int
    ) -> list[dict[str, Any]]:
        """Führt substring-basierte Fallback-Suche durch.

        Args:
            query: Suchanfrage
            candidates: Kandidaten-Dokumente
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Nach Substring-Match sortierte Dokumente
        """
        query_lower = query.lower()
        scored_candidates: list[tuple[dict[str, Any], float]] = []

        for candidate in candidates:
            # Text-Inhalt extrahieren
            content = str(candidate.get(get_field_name("content"),
                                       candidate.get(get_field_name("text"), "")))

            # Einfacher Substring-Score
            score = 1.0 if query_lower in content.lower() else 0.0
            scored_candidates.append((candidate, score))

        # Nach Score sortieren
        ranked_candidates = sorted(
            scored_candidates,
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Ergebnisse normalisieren
        results = []
        for candidate, score in ranked_candidates:
            normalized_doc = normalize_document_result(candidate, score)
            results.append(normalized_doc)

        return results


__all__ = [
    "CosmosVectorConfig",
    "CosmosVectorRetriever",
]
