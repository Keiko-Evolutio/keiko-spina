"""Multi-Modal Retriever (refactored, enterprise-grade).

Refactored mit konsolidierten Protocols und BaseRetriever
zur Eliminierung von Code-Duplikation und Verbesserung der Wartbarkeit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

from .base_retriever import BaseRetriever, BaseRetrieverConfig

if TYPE_CHECKING:
    from .retriever_protocols import ImageRetrieverBackend, TextRetriever

logger = get_logger(__name__)


@dataclass(slots=True)
class MultiModalConfig(BaseRetrieverConfig):
    """Konfiguration für Multi-Modal Retrieval."""

    text_enabled: bool = True
    image_enabled: bool = False


class MultiModalRetriever(BaseRetriever):
    """Aggregiert Text- und Bildsuche in einer API.

    Refactored mit BaseRetriever für einheitliches Error-Handling.
    """

    def __init__(
        self,
        config: MultiModalConfig,
        *,
        text_retriever: TextRetriever,
        image_backend: ImageRetrieverBackend | None = None,
    ) -> None:
        """Initialisiert den Multi-Modal Retriever."""
        super().__init__(config)
        self._multimodal_config = config
        self._text = text_retriever
        self._image = image_backend

    async def _perform_retrieval(
        self,
        query: str,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Implementiert Multi-Modal Retrieval-Logik.

        Standardmäßig wird nur Text-Retrieval durchgeführt.
        Für kombinierte Ergebnisse verwenden Sie aretrieve_both().
        """
        if self._multimodal_config.text_enabled:
            return await self.aretrieve_text(query, top_k=top_k)
        return []

    @trace_function("retriever.multimodal.aretrieve_text", {"component": "retriever"})
    async def aretrieve_text(
        self, query: str, *, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Führt Text-Retrieval durch."""
        if not self._multimodal_config.text_enabled:
            return []

        k = top_k or self._multimodal_config.top_k
        try:
            return await self._text.aretrieve(query, top_k=k)
        except Exception as exc:  # pragma: no cover - defensiv
            logger.warning(f"MultiModalRetriever Text-Fallback: {exc}")
            return []

    @trace_function("retriever.multimodal.aretrieve_images", {"component": "retriever"})
    async def aretrieve_images(
        self, query: str, *, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Führt Bild-Retrieval durch."""
        if not self._multimodal_config.image_enabled or self._image is None:
            return []

        k = top_k or self._multimodal_config.top_k
        try:
            return await self._image.search_images(query, top_k=k)
        except Exception as exc:  # pragma: no cover - defensiv
            logger.warning(f"MultiModalRetriever Image-Fallback: {exc}")
            return []

    @trace_function("retriever.multimodal.aretrieve_both", {"component": "retriever"})
    async def aretrieve_both(
        self, query: str, *, top_k_text: int | None = None, top_k_images: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Liefert kombinierte Ergebnisse für Text und Bilder."""
        texts = await self.aretrieve_text(query, top_k=top_k_text)
        images = await self.aretrieve_images(query, top_k=top_k_images)
        return {"text": texts, "images": images}


__all__ = [
    "MultiModalConfig",
    "MultiModalRetriever",
]
