"""Generische Daten-Transformationskette mit robustem Fallback.

Unterstützt LLM-gestützte Transformationen oder funktionale
Transformationen per Callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .chains_base import BaseKeikoChain
from .chains_common import (
    BaseChainConfig,
    Runnable,
)
from .chains_constants import (
    DEFAULT_TEMPERATURE_DETERMINISTIC,
    FALLBACK_TRANSFORM_PREFIX,
    TRANSFORM_PROMPT_TEMPLATE,
)
from .chains_utils import (
    create_simple_chain_factory,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass(slots=True)
class TransformConfig(BaseChainConfig):
    """Konfiguration für Transformationskette."""

    temperature: float = DEFAULT_TEMPERATURE_DETERMINISTIC
    instruction: str = "Transform the input as specified"


class KeikoTransformChain(BaseKeikoChain):
    """Wendet eine Transformation auf Texte an (LLM oder Callback)."""

    def __init__(
        self, *, config: TransformConfig, function: Callable[[str], str] | None = None
    ) -> None:
        """Initialisiert die Transformationskette."""
        super().__init__(config=config, memory=None)
        self._function = function

    def _get_trace_name(self) -> str:
        """Gibt Trace-Name zurück."""
        return "keiko_chains.transform.ainvoke"

    def _create_fallback_response(self, input_data: Any) -> str:
        """Erstellt Fallback-Transformation."""
        input_text = str(input_data) if input_data else ""

        # Funktionaler Pfad hat Vorrang (rein deterministisch)
        if self._function is not None:
            try:
                return self._function(input_text)
            except Exception as exc:  # pragma: no cover - defensiv
                logger.warning(f"Transformationsfunktion fehlgeschlagen: {exc}")

        return f"[{FALLBACK_TRANSFORM_PREFIX}:{self._config.instruction}] {input_text}"

    async def _create_runnable(self) -> Runnable:
        """Erstellt die Runnable-Pipeline."""
        # Nutze Factory für Standard-Pipeline-Erstellung
        factory = create_simple_chain_factory(
            TRANSFORM_PROMPT_TEMPLATE,
            FALLBACK_TRANSFORM_PREFIX,
        )
        return factory(self._config.model, self._config.temperature)

    async def ainvoke(self, input_text: str) -> str:
        """Transformiert den Eingabetext gemäß Konfiguration."""
        # Funktionaler Pfad hat Vorrang (rein deterministisch)
        if self._function is not None:
            try:
                return self._function(input_text)
            except Exception as exc:  # pragma: no cover - defensiv
                logger.warning(f"Transformationsfunktion fehlgeschlagen: {exc}")

        # Nutze Tracing-Mixin für einheitliches Tracing
        traced_method = self._create_traced_method(self._execute_transform)
        return await traced_method(input_text)

    async def _execute_transform(self, input_text: str) -> str:
        """Führt die Transform-Logik aus.

        Args:
            input_text: Zu transformierender Text.

        Returns:
            Transformierter Text.
        """
        runnable_params = {"instruction": self._config.instruction, "input": input_text}
        return await self._execute_with_fallback_wrapper(input_text, runnable_params)


__all__ = [
    "KeikoTransformChain",
    "TransformConfig",
]
