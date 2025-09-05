"""Einheitliche Protokoll-Definitionen für das tools-Modul.

Konsolidiert alle Protocol-Definitionen aus verschiedenen Retriever-Modulen
zur Eliminierung von Code-Duplikation und Verbesserung der Type-Safety.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# =============================================================================
# Core Retrieval Protocols
# =============================================================================

@runtime_checkable
class AsyncRetriever(Protocol):
    """Minimaler Retriever-Contract für asynchrone Dokumenten-Suche.

    Konsolidiert AsyncRetriever-Definitionen aus hybrid_retriever.py und
    anderen Modulen zur Standardisierung der Retriever-Schnittstelle.
    """

    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Führt asynchrone Dokumenten-Suche durch.

        Args:
            query: Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Liste von Dokument-Dictionaries mit 'text', optional 'score', 'metadata'
        """
        ...


@runtime_checkable
class TextRetriever(Protocol):
    """Text-Retriever-Contract für reine Text-Suche.

    Konsolidiert TextRetriever-Definition aus multi_modal_retriever.py.
    """

    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Führt Text-basierte Dokumenten-Suche durch.

        Args:
            query: Text-Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse

        Returns:
            Liste von Text-Dokumenten
        """
        ...


# =============================================================================
# Embedding and Vector Protocols
# =============================================================================

@runtime_checkable
class EmbeddingFunction(Protocol):
    """Standardisiertes Embedding-Interface für Text-zu-Vektor-Konvertierung.

    Konsolidiert EmbeddingFunction-Definitionen aus cosmos_vector_retriever.py
    und langchain_retrievers.py zur Eliminierung von Code-Duplikation.
    """

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Erstellt Embeddings für die gegebenen Texte.

        Args:
            texts: Liste von Texten für Embedding-Erstellung

        Returns:
            Liste von Embedding-Vektoren (ein Vektor pro Text)

        Examples:
            >>> # embedder = ConcreteEmbeddingImplementation()
            >>> # vectors = await embedder.aembed(["hello", "world"])
            >>> # len(vectors) == 2
            >>> # True
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Interface für Vektor-Datenbanken und -Speicher."""

    async def search_vectors(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filter_criteria: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Sucht ähnliche Vektoren in der Datenbank.

        Args:
            query_vector: Query-Vektor für Ähnlichkeitssuche
            top_k: Anzahl der gewünschten Ergebnisse
            filter_criteria: Optionale Filter-Kriterien

        Returns:
            Liste ähnlicher Dokumente mit Similarity-Scores
        """
        ...

    async def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]]
    ) -> bool:
        """Fügt Vektoren und zugehörige Dokumente hinzu.

        Args:
            vectors: Liste von Embedding-Vektoren
            documents: Zugehörige Dokument-Metadaten

        Returns:
            True bei Erfolg, False bei Fehlern
        """
        ...


# =============================================================================
# Multi-Modal Protocols
# =============================================================================

@runtime_checkable
class ImageRetrieverBackend(Protocol):
    """Bild-Retriever-Contract für Bild-basierte Suche.

    Konsolidiert ImageRetrieverBackend-Definition aus multi_modal_retriever.py.
    """

    async def search_images(
        self,
        query: str,
        *,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Führt Bild-basierte Suche durch.

        Args:
            query: Suchanfrage (Text oder Bild-Beschreibung)
            top_k: Anzahl der gewünschten Bild-Ergebnisse

        Returns:
            Liste von Bild-Metadaten (z.B. url, caption, score)

        Examples:
            >>> # backend = ConcreteImageBackendImplementation()
            >>> # images = await backend.search_images("cat photos", top_k=3)
            >>> # all("url" in image for image in images)
            >>> # True
        """
        ...


@runtime_checkable
class MultiModalRetriever(Protocol):
    """Interface für Multi-Modal-Retrieval (Text + Bilder)."""

    async def aretrieve_text(
        self,
        query: str,
        *,
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Führt Text-Retrieval durch."""
        ...

    async def aretrieve_images(
        self,
        query: str,
        *,
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Führt Bild-Retrieval durch."""
        ...

    async def aretrieve_both(
        self,
        query: str,
        *,
        top_k_text: int | None = None,
        top_k_images: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Führt kombiniertes Text- und Bild-Retrieval durch."""
        ...


# =============================================================================
# Tool and Bridge Protocols
# =============================================================================

@runtime_checkable
class ToolExecutor(Protocol):
    """Interface für Tool-Ausführung in MCP-Bridge-Kontexten."""

    async def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Führt ein Tool mit gegebenen Parametern aus.

        Args:
            tool_name: Name des auszuführenden Tools
            parameters: Tool-Parameter

        Returns:
            Tool-Ausführungsergebnis
        """
        ...


@runtime_checkable
class CapabilityMatcher(Protocol):
    """Interface für Capability-basierte Tool-Auswahl."""

    def match_capabilities(
        self,
        required_capabilities: list[str],
        available_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Findet Tools basierend auf erforderlichen Capabilities.

        Args:
            required_capabilities: Liste erforderlicher Capabilities
            available_tools: Verfügbare Tools

        Returns:
            Liste passender Tools
        """
        ...


# =============================================================================
# Configuration Protocols
# =============================================================================

@runtime_checkable
class ConfigurableComponent(Protocol):
    """Interface für konfigurierbare Komponenten."""

    def configure(self, config: dict[str, Any]) -> bool:
        """Konfiguriert die Komponente.

        Args:
            config: Konfigurationsdictionary

        Returns:
            True bei erfolgreicher Konfiguration
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Gibt aktuelle Konfiguration zurück.

        Returns:
            Aktuelle Konfiguration
        """
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Interface für Gesundheitschecks."""

    async def health_check(self) -> dict[str, Any]:
        """Führt Gesundheitscheck durch.

        Returns:
            Status-Dictionary mit Gesundheitsinformationen
        """
        ...


# =============================================================================
# Caching Protocols
# =============================================================================

@runtime_checkable
class CacheBackend(Protocol):
    """Interface für Cache-Backends."""

    async def get(self, key: str) -> Any | None:
        """Holt Wert aus Cache."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None
    ) -> bool:
        """Speichert Wert in Cache."""
        ...

    async def delete(self, key: str) -> bool:
        """Löscht Wert aus Cache."""
        ...

    async def clear(self) -> bool:
        """Leert gesamten Cache."""
        ...


# =============================================================================
# Utility Type Aliases
# =============================================================================

# Type Aliases für häufig verwendete Typen
DocumentDict = dict[str, Any]
MetadataDict = dict[str, Any]
ScoreFloat = float
EmbeddingVector = list[float]
QueryString = str


__all__ = [
    # Core Retrieval
    "AsyncRetriever",
    # Caching
    "CacheBackend",
    "CapabilityMatcher",
    # Configuration
    "ConfigurableComponent",
    # Type Aliases
    "DocumentDict",
    # Embedding and Vector
    "EmbeddingFunction",
    "EmbeddingVector",
    "HealthCheckable",
    # Multi-Modal
    "ImageRetrieverBackend",
    "MetadataDict",
    "MultiModalRetriever",
    "QueryString",
    "ScoreFloat",
    "TextRetriever",
    # Tool and Bridge
    "ToolExecutor",
    "VectorStore",
]
