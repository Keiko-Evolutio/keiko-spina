"""Azure Cognitive Search Retriever (refactored, enterprise-grade).

Refactored mit BaseRetriever, konsolidierten Constants und Utilities
zur Eliminierung von Code-Duplikation und Verbesserung der Wartbarkeit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .base_retriever import BaseRetriever, BaseRetrieverConfig
from .retriever_utils import (
    http_client_with_timeout,
    normalize_document_result,
)
from .tools_constants import (
    HTTP_HEADERS,
    get_api_version,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class AzureSearchConfig(BaseRetrieverConfig):
    """Konfiguration für Azure Cognitive Search Retriever."""

    endpoint: str = ""  # Muss bei Initialisierung gesetzt werden
    index_name: str = ""  # Muss bei Initialisierung gesetzt werden
    api_key: str | None = None
    api_version: str = get_api_version("azure_search")

    def __post_init__(self) -> None:
        """Validiert die Konfiguration nach Initialisierung."""
        super().__post_init__()

        if not self.endpoint:
            raise ValueError("endpoint ist erforderlich für AzureSearchConfig")
        if not self.index_name:
            raise ValueError("index_name ist erforderlich für AzureSearchConfig")


class AzureCognitiveSearchRetriever(BaseRetriever):
    """Asynchroner Retriever für Azure Cognitive Search.

    Refactored mit BaseRetriever für einheitliches Error-Handling,
    Tracing und Result-Normalisierung.
    """

    def __init__(self, config: AzureSearchConfig) -> None:
        """Initialisiert den Azure Search Retriever."""
        super().__init__(config)
        self._azure_config = config

    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Implementiert Azure Cognitive Search Retrieval-Logik.

        Args:
            query: Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Liste von Dokumenten aus Azure Search

        Raises:
            Exception: Bei Azure Search API-Fehlern
        """
        cfg = self._azure_config

        # URL mit API-Version aus Constants
        url = f"{cfg.endpoint}/indexes/{cfg.index_name}/docs/search?api-version={cfg.api_version}"

        # Headers aus Constants
        headers = {
            HTTP_HEADERS["content_type"]: HTTP_HEADERS["json"]
        }
        if cfg.api_key:
            headers[HTTP_HEADERS["api_key"]] = cfg.api_key

        # Payload für Azure Search API
        payload = {
            "search": query,
            "top": top_k
        }

        # HTTP-Request mit konsolidierter Utility
        data = await http_client_with_timeout(
            url=url,
            headers=headers,
            payload=payload,
            timeout=cfg.timeout
        )

        # Ergebnisse verarbeiten
        results: list[dict[str, Any]] = []
        for item in data.get("value", []):
            document = item.get("document", item)
            score = float(item.get("@search.score", 0.0))

            # Normalisierung mit Utility-Funktion
            normalized_doc = normalize_document_result(document, score)
            results.append(normalized_doc)

        return results


__all__ = [
    "AzureCognitiveSearchRetriever",
    "AzureSearchConfig",
]
