"""Gemeinsame Utility-Funktionen für alle Retriever.

Konsolidiert duplizierte Funktionen aus verschiedenen Retriever-Implementierungen
zur Verbesserung der Code-Wiederverwendung und Wartbarkeit.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .tools_constants import (
    get_error_message,
    get_field_name,
    get_timeout,
)

logger = get_logger(__name__)


# =============================================================================
# Mathematische Utility-Funktionen
# =============================================================================

def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Berechnet die Cosinus-Ähnlichkeit zwischen zwei Vektoren.

    Konsolidierte Implementierung aus cosmos_vector_retriever.py und
    langchain_retrievers.py zur Eliminierung von Code-Duplikation.

    Args:
        vector_a: Erster Vektor
        vector_b: Zweiter Vektor

    Returns:
        Cosinus-Ähnlichkeit zwischen 0.0 und 1.0

    Examples:
        >>> cosine_similarity([1, 0, 0], [1, 0, 0])
        1.0
        >>> cosine_similarity([1, 0], [0, 1])
        0.0
    """
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(x * y for x, y in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in vector_a))
    norm_b = math.sqrt(sum(y * y for y in vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """Berechnet die Euklidische Distanz zwischen zwei Vektoren.

    Args:
        vector_a: Erster Vektor
        vector_b: Zweiter Vektor

    Returns:
        Euklidische Distanz (0.0 = identisch)
    """
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return float("inf")

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vector_a, vector_b, strict=False)))


# =============================================================================
# HTTP-Client-Utilities
# =============================================================================

@trace_function("retriever.utils.http_client")
async def http_client_with_timeout(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
    method: str = "POST"
) -> dict[str, Any]:
    """Standardisierter HTTP-Client mit Timeout und Error-Handling.

    Konsolidiert HTTP-Request-Patterns aus azure_cognitive_search_retriever.py
    und langchain_retrievers.py.

    Args:
        url: Request-URL
        headers: HTTP-Headers
        payload: Request-Payload
        timeout: Timeout in Sekunden (default: aus Constants)
        method: HTTP-Methode

    Returns:
        Response-JSON als Dictionary

    Raises:
        Exception: Bei HTTP-Fehlern oder Timeouts
    """
    import httpx

    timeout_value = timeout or get_timeout("default")

    async with httpx.AsyncClient(timeout=timeout_value) as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers, params=payload)
        else:
            response = await client.post(url, headers=headers, json=payload)

        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


# =============================================================================
# Exception-Handling-Utilities
# =============================================================================

def defensive_exception_handler(
    operation_name: str,
    exception: Exception,
    fallback_value: Any = None,
    *,
    log_level: str = "warning"
) -> Any:
    """Einheitliches Exception-Handling mit Logging.

    Konsolidiert Exception-Handling-Patterns aus allen Retriever-Modulen
    zur Standardisierung der Error-Behandlung.

    Args:
        operation_name: Name der Operation für Logging
        exception: Aufgetretene Exception
        fallback_value: Rückgabewert bei Fehlern (default: None)
        log_level: Log-Level für die Ausgabe

    Returns:
        fallback_value oder None

    Examples:
        >>> result = defensive_exception_handler(
        ...     "test_operation",
        ...     ValueError("test"),
        ...     []
        ... )
        >>> result
        []
    """
    error_msg = get_error_message("retrieval_fallback",
                                  retriever_name=operation_name,
                                  error=str(exception))

    if log_level == "debug":
        logger.debug(error_msg)
    elif log_level == "info":
        logger.info(error_msg)
    elif log_level == "warning":
        logger.warning(error_msg)
    elif log_level == "error":
        logger.error(error_msg)
    else:
        logger.warning(error_msg)

    return fallback_value


# =============================================================================
# Text-Processing-Utilities
# =============================================================================

def text_hash_function(text: str, algorithm: str = "sha1") -> str:
    """Standardisierte Text-Hash-Funktion für Deduplizierung.

    Konsolidiert Hash-Funktionen aus hybrid_retriever.py.

    Args:
        text: Zu hashender Text
        algorithm: Hash-Algorithmus (sha1, md5, sha256)

    Returns:
        Hex-String des Hashes

    Examples:
        >>> text_hash_function("test")
        'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'
    """
    text_bytes = text.encode("utf-8")

    if algorithm == "md5":
        # NOTE: MD5 wird nur für nicht-kryptographische Zwecke verwendet (Checksummen)
        return hashlib.md5(text_bytes).hexdigest()  # noqa: S324
    if algorithm == "sha256":
        return hashlib.sha256(text_bytes).hexdigest()
    # default: sha256 statt sha1 für bessere Sicherheit
    return hashlib.sha256(text_bytes).hexdigest()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Kürzt Text auf maximale Länge mit Suffix.

    Args:
        text: Zu kürzender Text
        max_length: Maximale Länge
        suffix: Suffix für gekürzte Texte

    Returns:
        Gekürzter Text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# =============================================================================
# Document-Processing-Utilities
# =============================================================================

def _extract_text_content(document: dict[str, Any], source_field: str | None = None) -> str:
    """Extrahiert Text-Inhalt aus Dokument mit Auto-Detection.

    Args:
        document: Original-Dokument
        source_field: Spezifisches Feld für Text-Inhalt

    Returns:
        Extrahierter Text-Inhalt
    """
    if source_field and source_field in document:
        return str(document[source_field])

    # Auto-detect: content > text > erstes String-Feld
    for field in [get_field_name("content"), get_field_name("text")]:
        if field in document:
            return str(document[field])

    # Fallback: erstes String-Feld verwenden
    for key, value in document.items():
        if isinstance(value, str) and value.strip():
            return value

    return ""


def _extract_metadata(document: dict[str, Any], source_field: str | None = None) -> dict[str, Any]:
    """Extrahiert Metadaten aus Dokument (alle Felder außer Text-Inhalt).

    Args:
        document: Original-Dokument
        source_field: Feld das als Text-Inhalt verwendet wird

    Returns:
        Metadaten-Dictionary
    """
    excluded_fields = {get_field_name("content"), get_field_name("text"), source_field}
    return {key: value for key, value in document.items() if key not in excluded_fields}


def normalize_document_result(
    document: dict[str, Any],
    score: float = 0.0,
    *,
    source_field: str | None = None
) -> dict[str, Any]:
    """Normalisiert Dokument-Ergebnisse zu einheitlichem Format.

    Konsolidiert Dokument-Normalisierung aus verschiedenen Retriever-Modulen.

    Args:
        document: Original-Dokument
        score: Relevanz-Score
        source_field: Feld für Text-Inhalt (auto-detect wenn None)

    Returns:
        Normalisiertes Dokument mit standardisierten Feldern

    Examples:
        >>> doc = {"content": "test", "title": "Test Doc"}
        >>> normalize_document_result(doc, 0.8)
        {
            "text": "test",
            "score": 0.8,
            "metadata": {"title": "Test Doc"}
        }
    """
    text_content = _extract_text_content(document, source_field)
    metadata = _extract_metadata(document, source_field)

    return {
        get_field_name("text"): text_content,
        get_field_name("score"): float(score),
        get_field_name("metadata"): metadata,
    }


def merge_document_metadata(
    doc1: dict[str, Any],
    doc2: dict[str, Any]
) -> dict[str, Any]:
    """Führt Metadaten von zwei Dokumenten zusammen.

    Args:
        doc1: Erstes Dokument
        doc2: Zweites Dokument

    Returns:
        Dokument mit zusammengeführten Metadaten
    """
    metadata1 = doc1.get(get_field_name("metadata"), {})
    metadata2 = doc2.get(get_field_name("metadata"), {})

    # Metadaten vorsichtig mergen (doc1 hat Priorität)
    merged_metadata = dict(metadata2)
    merged_metadata.update(metadata1)  # doc1 überschreibt doc2

    return {
        get_field_name("text"): doc1.get(get_field_name("text"), ""),
        get_field_name("score"): doc1.get(get_field_name("score"), 0.0),
        get_field_name("metadata"): merged_metadata,
    }


# =============================================================================
# Scoring-Utilities
# =============================================================================

def combine_scores(
    scores: list[float],
    weights: list[float] | None = None,
    method: str = "weighted_average"
) -> float:
    """Kombiniert mehrere Scores zu einem finalen Score.

    Args:
        scores: Liste von Scores
        weights: Gewichtungen (default: gleichmäßig)
        method: Kombinationsmethode (weighted_average, max, min)

    Returns:
        Kombinierter Score
    """
    if not scores:
        return 0.0

    if weights is None:
        weights = [1.0] * len(scores)

    if len(scores) != len(weights):
        weights = weights[:len(scores)] + [1.0] * (len(scores) - len(weights))

    if method == "max":
        return max(scores)
    if method == "min":
        return min(scores)
    # weighted_average
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights, strict=False))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight > 0 else 0.0


__all__ = [
    # Scoring
    "combine_scores",
    # Mathematical Functions
    "cosine_similarity",
    # Exception Handling
    "defensive_exception_handler",
    "euclidean_distance",
    # HTTP Utilities
    "http_client_with_timeout",
    "merge_document_metadata",
    # Document Processing
    "normalize_document_result",
    # Text Processing
    "text_hash_function",
    "truncate_text",
]
