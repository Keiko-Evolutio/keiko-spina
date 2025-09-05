"""Hybrid Retriever (refactored, enterprise-grade).

Refactored mit BaseRetriever, konsolidierten Protocols und Utilities
zur Eliminierung von Code-Duplikation und Verbesserung der Wartbarkeit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .base_retriever import BaseRetriever, BaseRetrieverConfig
from .retriever_utils import (
    combine_scores,
    normalize_document_result,
    text_hash_function,
)
from .tools_constants import (
    SCORE_WEIGHTS,
    get_field_name,
)

if TYPE_CHECKING:
    from .retriever_protocols import AsyncRetriever

logger = get_logger(__name__)


@dataclass(slots=True)
class HybridConfig(BaseRetrieverConfig):
    """Konfiguration für Hybrid-Retriever."""

    weight_vector: float = SCORE_WEIGHTS["vector"]
    weight_keyword: float = SCORE_WEIGHTS["keyword"]


class HybridRetriever(BaseRetriever):
    """Kombiniert zwei Retriever zu einem robusten Ergebnis-Set.

    Refactored mit BaseRetriever für einheitliches Error-Handling
    und konsolidierten Utilities für Score-Kombination.
    """

    def __init__(
        self,
        config: HybridConfig,
        *,
        vector_retriever: AsyncRetriever,
        keyword_retriever: AsyncRetriever,
    ) -> None:
        """Initialisiert den Hybrid Retriever."""
        super().__init__(config)
        self._hybrid_config = config
        self._v = vector_retriever
        self._k = keyword_retriever

    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        k = top_k or self._config.top_k

        async def _safe_call(r: AsyncRetriever) -> list[dict[str, Any]]:
            try:
                return await r.aretrieve(query, top_k=k)
            except Exception as exc:  # pragma: no cover - defensiv
                logger.warning(f"HybridRetriever-Teil-Fallback: {exc}")
                return []

        v_results, k_results = await asyncio.gather(_safe_call(self._v), _safe_call(self._k))

        # Deduplizieren via Text-Hash mit konsolidierter Utility

        combined: dict[str, dict[str, Any]] = {}
        for item in v_results:
            txt = str(item.get("text", ""))
            h = text_hash_function(txt)
            combined[h] = {
                "text": txt,
                "v_score": float(item.get("score", 0.0)),
                "k_score": 0.0,
                "metadata": item.get("metadata", {}),
            }
        for item in k_results:
            txt = str(item.get("text", ""))
            h = text_hash_function(txt)
            if h in combined:
                combined[h]["k_score"] = float(item.get("score", 0.0))
                # Metadaten vorsichtig mergen
                meta = dict(combined[h].get("metadata") or {})
                meta2 = item.get("metadata") or {}
                meta.update({k: v for k, v in meta2.items() if k not in meta})
                combined[h]["metadata"] = meta
            else:
                combined[h] = {
                    "text": txt,
                    "v_score": 0.0,
                    "k_score": float(item.get("score", 0.0)),
                    "metadata": item.get("metadata", {}),
                }

        # Score-Mischung mit konsolidierten Utilities
        cfg = self._hybrid_config
        results: list[dict[str, Any]] = []
        for data in combined.values():
            # Scores mit Utility-Funktion kombinieren
            combined_score = combine_scores(
                [data["v_score"], data["k_score"]],
                [cfg.weight_vector, cfg.weight_keyword],
                method="weighted_average"
            )

            # Ergebnis normalisieren
            normalized_doc = normalize_document_result(
                {"text": data["text"], **data.get("metadata", {})},
                combined_score
            )
            results.append(normalized_doc)

        # Nach Score sortieren und Top-K auswählen
        results.sort(key=lambda x: x.get(get_field_name("score"), 0.0), reverse=True)
        return results[:top_k]


__all__ = [
    "HybridConfig",
    "HybridRetriever",
]
