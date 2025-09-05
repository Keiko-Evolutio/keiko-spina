"""Dokumenten-Zusammenfassungskette mit robustem Fallback.

Erstellt kompakte Zusammenfassungen großer Texte, optional mit Chat-Memory
zur Kontextanreicherung. Fällt bei fehlenden Abhängigkeiten auf
regelorientierte Heuristiken zurück.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .chains_base import BaseKeikoChain
from .chains_common import (
    BaseChainConfig,
    ChatMemory,
    Runnable,
)
from .chains_constants import (
    DEFAULT_TARGET_SUMMARY_POINTS,
    FALLBACK_SUMMARY_PREFIX,
    SUMMARIZATION_PROMPT_TEMPLATE,
)
from .chains_utils import (
    create_simple_chain_factory,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class SummarizationConfig(BaseChainConfig):
    """Konfiguration für Summarization Chain."""

    target_length: int = DEFAULT_TARGET_SUMMARY_POINTS  # Anzahl Stichpunkte


class KeikoSummarizationChain(BaseKeikoChain):
    """Erstellt eine prägnante Zusammenfassung eines oder mehrerer Texte."""

    def __init__(self, *, config: SummarizationConfig, memory: ChatMemory | None = None) -> None:
        """Initialisiert die Zusammenfassungskette."""
        super().__init__(config=config, memory=memory)

    def _get_trace_name(self) -> str:
        """Gibt Trace-Name zurück."""
        return "keiko_chains.summarization.ainvoke"

    def _create_fallback_response(self, input_data: Any) -> str:
        """Erstellt Fallback-Zusammenfassung mit Heuristiken."""
        text = str(input_data) if input_data else ""

        if not text.strip():
            return ""

        sentences = [s.strip() for s in text.split(".") if s.strip()][:self._config.target_length]
        if not sentences:
            return text[:200]

        bullets = "\n".join(f"- {s}." for s in sentences)
        return f"{FALLBACK_SUMMARY_PREFIX}:\n{bullets}"

    async def _create_runnable(self) -> Runnable:
        """Erstellt die Runnable-Pipeline."""
        # Nutze Factory für Standard-Pipeline-Erstellung
        factory = create_simple_chain_factory(
            SUMMARIZATION_PROMPT_TEMPLATE,
            FALLBACK_SUMMARY_PREFIX,
        )
        return factory(self._config.model, self._config.temperature)

    async def ainvoke(self, text: str) -> str:
        """Erzeugt eine Zusammenfassung für den gegebenen Text."""
        if not text.strip():
            return ""

        # Nutze Tracing-Mixin für einheitliches Tracing
        traced_method = self._create_traced_method(self._execute_summarization)
        return await traced_method(text)

    async def _execute_summarization(self, text: str) -> str:
        """Führt die Summarization-Logik aus.

        Args:
            text: Zu zusammenfassender Text.

        Returns:
            Zusammenfassung des Textes.
        """
        runnable_params = {"n": self._config.target_length, "text": text}
        result = await self._execute_with_fallback_wrapper(text, runnable_params)
        await self._persist_conversation(text, result)
        return result


__all__ = [
    "KeikoSummarizationChain",
    "SummarizationConfig",
]
