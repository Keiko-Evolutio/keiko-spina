"""Konversations-Chains für Multi-Turn-Dialoge.

Produktionsreife Konversationsketten auf Basis von langchain-core mit:

- Mehrschritt-Reasoning (Analyse → Antwort)
- Optionale Tool-Einbindung
- Konversationshistorie via Memory-Adapter
- Tracing-Integration

Die Implementierung ist defensiv ausgelegt und funktioniert auch bei
fehlenden optionalen Abhängigkeiten (Fallback auf Echo-Verhalten).
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .chains_base import BaseKeikoChain
from .chains_common import (
    LANGCHAIN_CORE_AVAILABLE,
    BaseChainConfig,
    ChatMemory,
    PromptTemplate,
    Runnable,
    RunnableLambda,
    RunnableSequence,
    StrOutputParser,
    create_echo_fallback,
    create_safe_llm,
)
from .chains_constants import (
    ANALYSIS_PROMPT_TEMPLATE,
    ANSWER_PROMPT_TEMPLATE,
    DEFAULT_MAX_HISTORY_MESSAGES,
)
from .chains_utils import (
    TracingMixin,
    create_fallback_runnable,
    load_session_history,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass(slots=True)
class ChainConfig(BaseChainConfig):
    """Konfiguration der Keiko-Konversationsketten."""

    enable_tools: bool = True
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES


class KeikoConversationChain(BaseKeikoChain):
    """Mehrschritt-Konversationskette mit optionaler Tool-Unterstützung.

    Diese Kette implementiert zwei Schritte:
    1) Analyse: Verdichtet Nutzerinput und Historie zu einem knappen Gedankengang
    2) Antwort: Formuliert eine hilfreiche, knappe Antwort für den Nutzer

    Die Kette ist rein auf langchain-core Runnables aufgebaut, um Abhängigkeiten
    minimal zu halten. Ein optionales LLM (ChatOpenAI) wird verwendet, wenn
    verfügbar und konfiguriert.
    """

    def __init__(
        self,
        *,
        config: ChainConfig,
        memory: ChatMemory | None = None,
        tools: list[Callable[..., Awaitable[Any]]] | None = None,
    ) -> None:
        """Initialisiert die Konversationskette.

        Args:
            config: Kettenkonfiguration.
            memory: Optionaler Chat-Speicher für Historie.
            tools: Optionale Liste asynchroner Tool-Funktionen (OpenAI Tools kompatibel).
        """
        super().__init__(config=config, memory=memory)
        self._tools = tools or []

    def _get_trace_name(self) -> str:
        """Gibt Trace-Name zurück."""
        return "keiko_chains.conversation.ainvoke"

    def _create_fallback_response(self, input_data: Any) -> str:
        """Erstellt Echo-Fallback-Antwort."""
        return create_echo_fallback(str(input_data))

    async def ainvoke(self, user_input: str) -> str:
        """Führt die Konversationskette einmal aus.

        Args:
            user_input: Nutzertext.

        Returns:
            Modellantwort als String (oder Echo-Fallback).
        """
        # Nutze Tracing-Mixin für einheitliches Tracing
        traced_method = self._create_traced_method(self._execute_conversation)
        return await traced_method(user_input)

    async def _execute_conversation(self, user_input: str) -> str:
        """Führt die Konversations-Logik aus.

        Args:
            user_input: Nutzertext.

        Returns:
            Modellantwort als String (oder Echo-Fallback).
        """
        runnable_params = {
            "session_id": self._config.session_id,
            "input": user_input,
        }

        result = await self._execute_with_fallback_wrapper(user_input, runnable_params)
        await self._persist_conversation(user_input, result)
        return result

    async def _load_history_with_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Lädt Chat-Historie für Session und begrenzt auf konfigurierten Umfang."""
        return await load_session_history(
            self._memory,
            params["session_id"],
            self._config.max_history_messages,
            params,
        )

    @staticmethod
    def _create_prompts() -> tuple[PromptTemplate, PromptTemplate]:
        """Erstellt Analyse- und Antwort-Prompts.

        Returns:
            Tuple aus (Analyse-Prompt, Antwort-Prompt)
        """
        analysis_prompt = PromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
        answer_prompt = PromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)
        return analysis_prompt, answer_prompt

    def _create_llm_with_tools(self) -> Any | None:
        """Erstellt LLM mit optionalen Tools."""
        llm = create_safe_llm(self._config.model, self._config.temperature)
        if llm and self._tools and self._config.enable_tools and hasattr(llm, "bind_tools"):
            try:
                llm = llm.bind_tools(self._tools)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - defensiv
                logger.debug(f"Tool-Binding fehlgeschlagen: {exc}")
        return llm

    async def _create_runnable(self) -> Runnable:
        """Erstellt die Runnable-Pipeline."""
        if not LANGCHAIN_CORE_AVAILABLE:
            return self._create_echo_runnable()

        llm = self._create_llm_with_tools()
        if llm is None:
            return self._create_echo_runnable()

        analysis_prompt, answer_prompt = self._create_prompts()
        parser = StrOutputParser()

        return (
            RunnableLambda(self._load_history_with_session)
            | RunnableSequence(
                analysis_prompt | llm,
                answer_prompt | llm,
            )
            | parser
        )

    @staticmethod
    def _create_echo_runnable() -> Runnable:
        """Erstellt Fallback-Runnable ohne LLM.

        Returns:
            Runnable das Echo-Fallback implementiert
        """
        def _echo(params: dict[str, Any]) -> str:
            return create_echo_fallback(params.get("input", ""))

        return create_fallback_runnable(_echo)


class KeikoSequentialChain(TracingMixin):
    """Einfache sequentielle Kette aus beliebigen Runnable-Schritten.

    Dient als Ersatz für klassische SequentialChain-Implementierungen, basiert aber
    ausschließlich auf langchain-core. Jeder Schritt erhält das Ergebnis des
    vorherigen Schritts als Eingabe.
    """

    def __init__(self, steps: list[Callable[[str], Awaitable[str]]]) -> None:
        """Initialisiert die sequentielle Kette.

        Args:
            steps: Liste asynchroner Ein-Schritt-Funktionen (str → str).
        """
        super().__init__()
        self._steps = steps

    def _get_trace_name(self) -> str:
        """Gibt Trace-Name zurück."""
        return "keiko_chains.sequential.ainvoke"

    async def ainvoke(self, input_text: str) -> str:
        """Führt alle Schritte nacheinander aus und gibt das Endergebnis zurück."""
        # Nutze Tracing-Mixin für einheitliches Tracing
        traced_method = self._create_traced_method(self._execute_sequential_steps)
        return await traced_method(input_text)

    async def _execute_sequential_steps(self, input_text: str) -> str:
        """Führt alle Sequential-Schritte aus.

        Args:
            input_text: Eingabetext.

        Returns:
            Endergebnis nach allen Schritten.
        """
        result = input_text
        for step in self._steps:
            try:
                result = await step(result)
            except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
                logger.warning(f"Sequential-Schritt fehlgeschlagen - Validierungsfehler: {exc}")
                return result
            except (ConnectionError, TimeoutError) as exc:  # pragma: no cover - defensiv
                logger.warning(f"Sequential-Schritt fehlgeschlagen - Verbindungsproblem: {exc}")
                return result
            except Exception as exc:  # pragma: no cover - defensiv
                logger.warning(f"Sequential-Schritt fehlgeschlagen - Unerwarteter Fehler: {exc}")
                return result
        return result


__all__ = [
    "ChainConfig",
    "KeikoConversationChain",
    "KeikoSequentialChain",
]
