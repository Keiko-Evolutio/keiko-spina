"""LangChain Retriever- und Loader-Utilities (refactored, enterprise-grade).

Refactored mit konsolidierten Protocols, Utilities und Constants
zur Eliminierung von Code-Duplikation und Verbesserung der Wartbarkeit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .retriever_utils import (
    cosine_similarity,
    http_client_with_timeout,
)
from .tools_constants import (
    get_api_version,
    get_timeout,
)

if TYPE_CHECKING:
    from .retriever_protocols import EmbeddingFunction

logger = get_logger(__name__)


# Optional: Document Loader
try:  # pragma: no cover - optional
    import docx  # type: ignore
    DOCX_AVAILABLE = True
except Exception:  # pragma: no cover
    DOCX_AVAILABLE = False

try:  # pragma: no cover - optional
    import fitz  # PyMuPDF  # type: ignore
    PDF_AVAILABLE = True
except Exception:  # pragma: no cover
    PDF_AVAILABLE = False


@dataclass(slots=True)
class SearchConfig:
    """Konfiguration für Azure Cognitive Search Abfragen."""

    endpoint: str
    index_name: str
    api_key: str | None = None
    top_k: int = 5


async def azure_search_query(query: str, *, config: SearchConfig) -> list[dict[str, Any]]:
    """Führt eine Azure Cognitive Search Anfrage aus (refactored).

    Verwendet konsolidierte HTTP-Utilities und Constants.
    """
    try:
        # URL mit API-Version aus Constants
        url = (
            f"{config.endpoint}/indexes/{config.index_name}/docs/search"
            f"?api-version={get_api_version('azure_search')}"
        )

        # Headers aus Constants
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["api-key"] = config.api_key

        payload = {"search": query, "top": config.top_k}

        # HTTP-Request mit konsolidierter Utility
        data = await http_client_with_timeout(
            url=url,
            headers=headers,
            payload=payload,
            timeout=get_timeout("default")
        )

        return [hit.get("document", hit) for hit in data.get("value", [])]

    except (ConnectionError, TimeoutError) as exc:  # pragma: no cover - defensiv
        logger.warning(f"Azure Search Fallback aktiv - Verbindungsproblem: {exc}")
        return []
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
        logger.warning(f"Azure Search Fallback aktiv - Konfigurationsfehler: {exc}")
        return []
    except Exception as exc:  # pragma: no cover - defensiv
        logger.warning(f"Azure Search Fallback aktiv - Unerwarteter Fehler: {exc}")
        return []


def load_text_file(path: str) -> str:
    """Lädt eine Textdatei robust."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as exc:  # pragma: no cover - defensiv
        logger.warning(f"Text-Load fehlgeschlagen - Datei-/Berechtigungsfehler: {exc}")
        return ""
    except (UnicodeDecodeError, ValueError) as exc:  # pragma: no cover - defensiv
        logger.warning(f"Text-Load fehlgeschlagen - Encoding-/Format-Fehler: {exc}")
        return ""
    except Exception as exc:  # pragma: no cover - defensiv
        logger.warning(f"Text-Load fehlgeschlagen - Unerwarteter Fehler: {exc}")
        return ""


def load_pdf_file(path: str) -> str:
    """Extrahiert Text aus PDF (PyMuPDF), Fallback leerer Text."""
    if not PDF_AVAILABLE:
        return ""
    try:  # pragma: no cover - optional
        doc = fitz.open(path)
        texts: list[str] = []
        for page in doc:
            # PyMuPDF page.get_text() method extracts text content
            texts.append(page.get_text())  # type: ignore[attr-defined]
        return "\n".join(texts)
    except (FileNotFoundError, PermissionError) as exc:  # pragma: no cover - defensiv
        logger.debug(f"PDF-Load fehlgeschlagen - Datei-/Berechtigungsfehler: {exc}")
        return ""
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
        logger.debug(f"PDF-Load fehlgeschlagen - PDF-Format-/Parsing-Fehler: {exc}")
        return ""
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"PDF-Load fehlgeschlagen - Unerwarteter Fehler: {exc}")
        return ""


def load_docx_file(path: str) -> str:
    """Extrahiert Text aus DOCX (python-docx), Fallback leerer Text."""
    if not DOCX_AVAILABLE:
        return ""
    try:  # pragma: no cover - optional
        d = docx.Document(path)
        return "\n".join(par.text for par in d.paragraphs)
    except (FileNotFoundError, PermissionError) as exc:  # pragma: no cover - defensiv
        logger.debug(f"DOCX-Load fehlgeschlagen - Datei-/Berechtigungsfehler: {exc}")
        return ""
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
        logger.debug(f"DOCX-Load fehlgeschlagen - DOCX-Format-/Parsing-Fehler: {exc}")
        return ""
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"DOCX-Load fehlgeschlagen - Unerwarteter Fehler: {exc}")
        return ""


# cosine_similarity wurde zu retriever_utils.py verschoben


async def semantic_search(
    query: str, *, documents: list[str], embedder: EmbeddingFunction, top_k: int = 5
) -> list[tuple[int, float]]:
    """Einfacher semantischer Suchalgorithmus auf Basis von Cosinus-Ähnlichkeit.

    Returns:
        Liste aus (Index im Dokument-Array, Score) absteigend sortiert.
    """
    if not documents:
        return []
    try:
        doc_embeddings = await embedder.aembed(documents)
        query_emb = (await embedder.aembed([query]))[0]
        scored = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(doc_embeddings)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    except (ConnectionError, TimeoutError) as exc:  # pragma: no cover - defensiv
        logger.debug(f"semantic_search Fallback - Embedder-Verbindungsproblem: {exc}")
        # Fallback: einfache substring-Score
        scored = [
            (i, (1.0 if query.lower() in (doc.lower()) else 0.0)) for i, doc in enumerate(documents)
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
        logger.debug(f"semantic_search Fallback - Embedder-Parameter-Fehler: {exc}")
        # Fallback: einfache substring-Score
        scored = [
            (i, (1.0 if query.lower() in (doc.lower()) else 0.0)) for i, doc in enumerate(documents)
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"semantic_search Fallback - Unerwarteter Embedder-Fehler: {exc}")
        # Fallback: einfache substring-Score
        scored = [
            (i, (1.0 if query.lower() in (doc.lower()) else 0.0)) for i, doc in enumerate(documents)
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


__all__ = [
    "SearchConfig",
    "azure_search_query",
    "load_docx_file",
    "load_pdf_file",
    "load_text_file",
    "semantic_search",
]
