"""Gemeinsame Infrastruktur für Keiko Chain-Module.

Zentrale Definition von optionalen Abhängigkeiten, Protokollen,
Basis-Konfigurationen und Utility-Funktionen für alle Chain-Implementierungen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from kei_logging import get_logger

from .chains_constants import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEXT_TRUNCATION,
    ECHO_FALLBACK_PREFIX,
    FALLBACK_ANSWER_PREFIX,
    NO_RESULTS_MESSAGE,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

# ============================================================================
# OPTIONALE ABHÄNGIGKEITEN - ZENTRALE VERWALTUNG
# ============================================================================

# LangChain Core Abhängigkeiten
try:  # pragma: no cover - optional
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence

    LANGCHAIN_CORE_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    Runnable = object  # type: ignore[assignment]
    RunnableLambda = object  # type: ignore[assignment]
    RunnableSequence = object  # type: ignore[assignment]
    PromptTemplate = object  # type: ignore[assignment]
    StrOutputParser = object  # type: ignore[assignment]
    LANGCHAIN_CORE_AVAILABLE = False

# LangChain OpenAI Abhängigkeiten
try:  # pragma: no cover - optional
    from langchain_openai import ChatOpenAI

    LANGCHAIN_OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    ChatOpenAI = None  # type: ignore
    LANGCHAIN_OPENAI_AVAILABLE = False


# ============================================================================
# GEMEINSAME PROTOKOLLE
# ============================================================================

class ChatMemory(Protocol):
    """Abstraktes Protokoll für Chat-Speicher.

    Erwartet wird eine minimalistische Schnittstelle, kompatibel zu den
    Operationen aller Chain-Module. Ein Adapter kann auf Cosmos DB oder andere
    Backends setzen und die Historie zwischen Sitzungen persistieren.
    """

    async def load_messages(self, session_id: str) -> list[dict[str, str]]:
        """Lädt alle Nachrichten für die gegebene Session.

        Args:
            session_id: Eindeutige Session-ID.

        Returns:
            Liste von Nachrichten im Format {"role": "user|assistant|system", "content": str}.
        """

    async def append_messages(self, session_id: str, messages: Iterable[dict[str, str]]) -> None:
        """Hängt Nachrichten an eine Session an und persistiert diese.

        Args:
            session_id: Eindeutige Session-ID.
            messages: Iterable von Nachrichten-Dicts.
        """


class AsyncRetriever(Protocol):
    """Protokoll für asynchrone Retriever.

    Erwartetes Rückgabeformat: Liste von Dokumenten-Dicts mit mindestens
    den Schlüsseln `text` (str) und optional `score` (float) und `metadata` (dict).
    """

    async def aretrieve(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Führt eine asynchrone Retrieval-Anfrage aus.

        Args:
            query: Suchanfrage.
            top_k: Maximale Anzahl zurückzugebender Dokumente.

        Returns:
            Liste von Dokumenten-Dicts.
        """


# ============================================================================
# BASIS-KONFIGURATIONEN
# ============================================================================

@dataclass(slots=True)
class BaseChainConfig:
    """Basis-Konfiguration für alle Chain-Typen."""

    session_id: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE


# ============================================================================
# UTILITY-FUNKTIONEN
# ============================================================================

def truncate_text(text: str, max_length: int = DEFAULT_TEXT_TRUNCATION) -> str:
    """Kürzt Text auf maximale Länge.

    Args:
        text: Zu kürzender Text.
        max_length: Maximale Länge.

    Returns:
        Gekürzter Text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length]


def format_history(
    messages: list[dict[str, str]], max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
) -> list[dict[str, str]]:
    """Formatiert und begrenzt Chat-Historie.

    Args:
        messages: Liste von Nachrichten.
        max_messages: Maximale Anzahl von Nachrichten.

    Returns:
        Begrenzte und formatierte Nachrichten-Liste.
    """
    return messages[-max_messages:] if messages else []


def create_echo_fallback(input_text: str, prefix: str = ECHO_FALLBACK_PREFIX) -> str:
    """Erstellt eine Echo-Fallback-Antwort.

    Args:
        input_text: Eingabetext.
        prefix: Präfix für die Fallback-Antwort.

    Returns:
        Formatierte Fallback-Antwort.
    """
    return f"{prefix}: {input_text}"


def create_retrieval_fallback(docs: list[dict[str, Any]], max_docs: int = 3) -> str:
    """Erstellt eine Retrieval-Fallback-Antwort aus Dokumenten.

    Args:
        docs: Liste von Dokumenten.
        max_docs: Maximale Anzahl von Dokumenten für Fallback.

    Returns:
        Formatierte Fallback-Antwort.
    """
    if not docs:
        return f"{FALLBACK_ANSWER_PREFIX}: {NO_RESULTS_MESSAGE}"

    top_snippets = ", ".join(
        truncate_text(d.get("text", "")) for d in docs[:max_docs]
    )
    return f"{FALLBACK_ANSWER_PREFIX}: {top_snippets}"


async def safe_memory_append(
    memory: ChatMemory | None,
    session_id: str,
    user_input: str,
    assistant_output: str,
) -> None:
    """Sicher Nachrichten zu Memory hinzufügen mit Error-Handling.

    Args:
        memory: Optionales Chat-Memory.
        session_id: Session-ID.
        user_input: Benutzereingabe.
        assistant_output: Assistenten-Ausgabe.
    """
    if memory is None:
        return

    try:
        await memory.append_messages(
            session_id,
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output},
            ],
        )
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"Memory-Append fehlgeschlagen: {exc}")


async def safe_memory_load(
    memory: ChatMemory | None, session_id: str, max_messages: int
) -> list[dict[str, str]]:
    """Sicher Historie aus Memory laden mit Error-Handling.

    Args:
        memory: Optionales Chat-Memory.
        session_id: Session-ID.
        max_messages: Maximale Anzahl von Nachrichten.

    Returns:
        Liste von Nachrichten oder leere Liste bei Fehlern.
    """
    if memory is None:
        return []

    try:
        all_messages = await memory.load_messages(session_id)
        return format_history(all_messages, max_messages)
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"Memory-Load fehlgeschlagen: {exc}")
        return []


def create_safe_llm(model: str, temperature: float) -> Any | None:
    """Erstellt sicher ein LLM mit Error-Handling.

    Args:
        model: Modell-Name.
        temperature: Temperatur-Wert.

    Returns:
        LLM-Instanz oder None bei Fehlern.
    """
    if not LANGCHAIN_OPENAI_AVAILABLE:
        return None

    try:
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"LLM-Erstellung fehlgeschlagen: {exc}")
        return None


__all__ = [
    "LANGCHAIN_CORE_AVAILABLE",
    "LANGCHAIN_OPENAI_AVAILABLE",
    "AsyncRetriever",
    "BaseChainConfig",
    "ChatMemory",
    "ChatOpenAI",
    "PromptTemplate",
    "Runnable",
    "RunnableLambda",
    "RunnableSequence",
    "StrOutputParser",
    "create_echo_fallback",
    "create_retrieval_fallback",
    "create_safe_llm",
    "format_history",
    "safe_memory_append",
    "safe_memory_load",
    "truncate_text",
]
