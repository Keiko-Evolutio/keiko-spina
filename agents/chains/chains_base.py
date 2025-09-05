"""Basis-Klassen für Chain-Module.

Abstrakte Basis-Implementierungen für einheitliche Chain-Interfaces,
gemeinsame Error-Handling-Strategien und Tracing-Integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from observability import trace_function

from .chains_common import (
    BaseChainConfig,
    ChatMemory,
    Runnable,
    create_echo_fallback,
)
from .chains_utils import (
    ChainExecutionMixin,
    MemoryMixin,
    TracingMixin,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class BaseKeikoChain(ABC, ChainExecutionMixin, TracingMixin, MemoryMixin):
    """Abstrakte Basis-Klasse für alle Keiko-Chains.

    Stellt einheitliche Interfaces und gemeinsame Funktionalitäten
    für alle Chain-Implementierungen bereit. Nutzt Mixins für
    gemeinsame Funktionalitäten.
    """

    def __init__(
        self,
        *,
        config: BaseChainConfig,
        memory: ChatMemory | None = None,
    ) -> None:
        """Initialisiert die Basis-Chain.

        Args:
            config: Chain-Konfiguration.
            memory: Optionaler Chat-Speicher.
        """
        # Initialisiere alle Mixins
        ChainExecutionMixin.__init__(self)
        TracingMixin.__init__(self)
        MemoryMixin.__init__(self)

        self._config = config
        self._memory = memory

    @abstractmethod
    async def _create_runnable(self) -> Runnable:
        """Erstellt die spezifische Runnable-Pipeline.

        Returns:
            Konfigurierte Runnable-Pipeline.
        """

    @abstractmethod
    def _create_fallback_response(self, input_data: Any) -> str:
        """Erstellt eine Fallback-Antwort bei fehlenden Abhängigkeiten.

        Args:
            input_data: Eingabedaten für Fallback.

        Returns:
            Fallback-Antwort als String.
        """

    async def _execute_with_fallback_wrapper(self, input_data: Any, runnable_params: dict[str, Any]) -> str:
        """Wrapper für _execute_with_fallback mit spezifischer Fallback-Funktion.

        Args:
            input_data: Eingabedaten für Fallback.
            runnable_params: Parameter für Runnable-Ausführung.

        Returns:
            Chain-Ergebnis oder Fallback-Antwort.
        """
        return await super()._execute_with_fallback(
            input_data,
            runnable_params,
            self._create_fallback_response,
        )

    @trace_function("keiko_chains.base.astream", {"component": "chains"})
    async def astream(self, input_data: Any) -> AsyncIterator[str]:
        """Streamt die Chain-Ausgabe (Standard-Implementierung).

        Args:
            input_data: Eingabedaten.

        Yields:
            Chain-Ausgabe als String-Chunks.
        """
        result = await self.ainvoke(input_data)
        yield result

    @abstractmethod
    async def ainvoke(self, input_data: Any) -> str:
        """Führt die Chain einmal aus.

        Args:
            input_data: Eingabedaten.

        Returns:
            Chain-Ergebnis als String.
        """


class SimpleChain(BaseKeikoChain):
    """Einfache Chain-Implementierung für grundlegende Use-Cases.

    Kann mit einer benutzerdefinierten Funktion oder einem Standard-Echo-Verhalten
    konfiguriert werden.
    """

    def __init__(
        self,
        *,
        config: BaseChainConfig,
        memory: ChatMemory | None = None,
        transform_function: Callable[[str], str] | None = None,
    ) -> None:
        """Initialisiert die einfache Chain.

        Args:
            config: Chain-Konfiguration.
            memory: Optionaler Chat-Speicher.
            transform_function: Optionale Transformationsfunktion.
        """
        super().__init__(config=config, memory=memory)
        self._transform_function = transform_function

    async def _create_runnable(self) -> Runnable:
        """Erstellt eine einfache Lambda-Runnable."""
        from .chains_common import RunnableLambda

        def _transform(params: dict[str, Any]) -> str:
            input_text = params.get("input", "")
            if self._transform_function:
                return self._transform_function(input_text)
            return create_echo_fallback(input_text)

        return RunnableLambda(_transform)

    def _create_fallback_response(self, input_data: Any) -> str:
        """Erstellt Echo-Fallback-Antwort."""
        input_text = str(input_data) if input_data else ""
        if self._transform_function:
            try:
                return self._transform_function(input_text)
            except Exception:  # pragma: no cover - defensiv
                pass
        return create_echo_fallback(input_text)

    def _get_trace_name(self) -> str:
        """Gibt Trace-Name zurück."""
        return "keiko_chains.simple.ainvoke"

    async def ainvoke(self, input_text: str) -> str:
        """Führt die einfache Chain aus.

        Args:
            input_text: Eingabetext.

        Returns:
            Transformiertes Ergebnis.
        """
        # Nutze Tracing-Mixin für einheitliches Tracing
        traced_method = self._create_traced_method(self._execute_simple_chain)
        return await traced_method(input_text)

    async def _execute_simple_chain(self, input_text: str) -> str:
        """Führt die einfache Chain-Logik aus.

        Args:
            input_text: Eingabetext.

        Returns:
            Transformiertes Ergebnis.
        """
        result = await self._execute_with_fallback_wrapper(input_text, {"input": input_text})
        await self._persist_conversation(input_text, result)
        return result


__all__ = [
    "BaseKeikoChain",
    "SimpleChain",
]
