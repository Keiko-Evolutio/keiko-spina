"""Gemeinsame Utility-Funktionen für Keiko Chain-Module.

Konsolidiert duplizierte Code-Patterns und stellt wiederverwendbare
Komponenten für alle Chain-Implementierungen bereit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from kei_logging import get_logger
from observability import trace_function

from .chains_common import (
    LANGCHAIN_CORE_AVAILABLE,
    LANGCHAIN_OPENAI_AVAILABLE,
    PromptTemplate,
    Runnable,
    RunnableLambda,
    StrOutputParser,
    create_safe_llm,
    safe_memory_append,
    safe_memory_load,
)

logger = get_logger(__name__)

# Type Variables für generische Funktionen
F = TypeVar("F", bound=Callable[..., Any])


def create_trace_decorator(trace_name: str, metadata: dict[str, Any] | None = None) -> Callable[[F], F]:
    """Factory für einheitliche Trace-Decorators.

    Args:
        trace_name: Name für das Tracing.
        metadata: Optionale Metadaten für Tracing.

    Returns:
        Decorator-Funktion für Tracing.
    """
    if metadata is None:
        metadata = {"component": "chains"}

    def decorator(func: F) -> F:
        return trace_function(trace_name, metadata)(func)  # type: ignore[return-value]

    return decorator


def create_standard_runnable_pipeline(
    prompt_template: str,
    model: str,
    temperature: float,
    fallback_func: Callable[[dict[str, Any]], str],
) -> Runnable:
    """Erstellt eine Standard-Runnable-Pipeline mit LLM und Fallback.

    Args:
        prompt_template: Template für den Prompt.
        model: LLM-Modell-Name.
        temperature: Temperatur für LLM.
        fallback_func: Fallback-Funktion bei Fehlern.

    Returns:
        Konfigurierte Runnable-Pipeline.
    """
    if not (LANGCHAIN_CORE_AVAILABLE and LANGCHAIN_OPENAI_AVAILABLE):
        return create_fallback_runnable(fallback_func)

    llm = create_safe_llm(model, temperature)
    if llm is None:
        return create_fallback_runnable(fallback_func)

    prompt = PromptTemplate.from_template(prompt_template)
    parser = StrOutputParser()

    return prompt | llm | parser


def create_fallback_runnable(fallback_func: Callable[[dict[str, Any]], str]) -> Runnable:
    """Erstellt eine Fallback-Runnable ohne LLM-Abhängigkeiten.

    Args:
        fallback_func: Funktion für Fallback-Verhalten.

    Returns:
        Fallback-Runnable.
    """
    return RunnableLambda(lambda p: p) | RunnableLambda(fallback_func)


async def load_session_history(
    memory: Any | None,
    session_id: str,
    max_messages: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Lädt Session-Historie und fügt sie zu Parametern hinzu.

    Args:
        memory: Chat-Memory-Instanz.
        session_id: Session-ID.
        max_messages: Maximale Anzahl von Nachrichten.
        params: Bestehende Parameter.

    Returns:
        Parameter mit hinzugefügter Historie.
    """
    history = await safe_memory_load(memory, session_id, max_messages)
    return {**params, "history": history}


class ChainExecutionMixin:
    """Mixin für gemeinsame Chain-Ausführungslogik."""

    def __init__(self) -> None:
        """Initialisiert das Mixin."""
        self._runnable: Runnable | None = None

    async def _ensure_runnable(self) -> Runnable:
        """Stellt sicher, dass eine Runnable-Pipeline existiert (lazy loading).

        Returns:
            Konfigurierte Runnable-Pipeline.
        """
        if self._runnable is None:
            self._runnable = await self._create_runnable()
        return self._runnable

    async def _create_runnable(self) -> Runnable:
        """Erstellt die spezifische Runnable-Pipeline.

        Muss von abgeleiteten Klassen implementiert werden.

        Returns:
            Konfigurierte Runnable-Pipeline.
        """
        raise NotImplementedError("Muss von abgeleiteter Klasse implementiert werden")

    async def _execute_with_fallback(
        self,
        input_data: Any,
        runnable_params: dict[str, Any],
        fallback_func: Callable[[Any], str],
    ) -> str:
        """Führt Chain aus mit automatischem Fallback bei Fehlern.

        Args:
            input_data: Eingabedaten für Fallback.
            runnable_params: Parameter für Runnable-Ausführung.
            fallback_func: Fallback-Funktion bei Fehlern.

        Returns:
            Chain-Ergebnis oder Fallback-Antwort.
        """
        if not LANGCHAIN_CORE_AVAILABLE:
            return fallback_func(input_data)

        try:
            runnable = await self._ensure_runnable()
            return await runnable.ainvoke(runnable_params)  # type: ignore[union-attr]
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
            logger.debug(f"Chain-Ausführung fehlgeschlagen - Validierungsfehler, Fallback aktiv: {exc}")
            return fallback_func(input_data)
        except (ConnectionError, TimeoutError) as exc:  # pragma: no cover - defensiv
            logger.debug(f"Chain-Ausführung fehlgeschlagen - Verbindungsproblem, Fallback aktiv: {exc}")
            return fallback_func(input_data)
        except Exception as exc:  # pragma: no cover - defensiv
            logger.debug(f"Chain-Ausführung fehlgeschlagen - Unerwarteter Fehler, Fallback aktiv: {exc}")
            return fallback_func(input_data)


class TracingMixin:
    """Mixin für einheitliches Tracing in Chains."""

    def _get_trace_name(self) -> str:
        """Gibt den Namen für Tracing zurück.

        Muss von abgeleiteten Klassen implementiert werden.

        Returns:
            Trace-Name für diese Chain.
        """
        raise NotImplementedError("Muss von abgeleiteter Klasse implementiert werden")

    def _create_traced_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Erstellt eine getracte Version einer Methode.

        Args:
            method: Zu tracende Methode.

        Returns:
            Getracte Methode.
        """
        trace_name = self._get_trace_name()
        return create_trace_decorator(trace_name)(method)


class MemoryMixin:
    """Mixin für Memory-Operationen in Chains."""

    def __init__(self) -> None:
        """Initialisiert das Mixin."""
        self._memory: Any | None = None
        self._config: Any | None = None

    async def _persist_conversation(self, user_input: str, assistant_output: str) -> None:
        """Persistiert Konversation in Memory falls verfügbar.

        Args:
            user_input: Benutzereingabe.
            assistant_output: Assistenten-Ausgabe.
        """
        if self._memory is not None and self._config is not None:
            await safe_memory_append(
                self._memory,
                self._config.session_id,
                user_input,
                assistant_output,
            )

    async def _load_history_for_session(
        self,
        session_id: str,
        max_messages: int,
    ) -> list[dict[str, str]]:
        """Lädt Historie für eine Session.

        Args:
            session_id: Session-ID.
            max_messages: Maximale Anzahl von Nachrichten.

        Returns:
            Liste von Nachrichten.
        """
        return await safe_memory_load(self._memory, session_id, max_messages)


def create_simple_chain_factory(
    prompt_template: str,
    fallback_prefix: str,
) -> Callable[[str, float], Runnable]:
    """Factory für einfache Chain-Erstellung.

    Args:
        prompt_template: Template für den Prompt.
        fallback_prefix: Präfix für Fallback-Nachrichten.

    Returns:
        Factory-Funktion für Chain-Erstellung.
    """
    def factory(model: str, temperature: float) -> Runnable:
        def fallback_func(params: dict[str, Any]) -> str:
            from .chains_common import create_echo_fallback
            input_text = params.get("input", params.get("text", ""))
            return create_echo_fallback(input_text, fallback_prefix)

        return create_standard_runnable_pipeline(
            prompt_template,
            model,
            temperature,
            fallback_func,
        )

    return factory


__all__ = [
    "ChainExecutionMixin",
    "MemoryMixin",
    "TracingMixin",
    "create_fallback_runnable",
    "create_simple_chain_factory",
    "create_standard_runnable_pipeline",
    "create_trace_decorator",
    "load_session_history",
]
