"""Intelligentes Chain-Routing mit heuristischer Klassifikation.

Ordnet Eingaben passenden Ketten-Typen zu (QA, Summarize, Transform)
und kann optional LLM-basierte Klassifikation verwenden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .chains_common import LANGCHAIN_OPENAI_AVAILABLE, create_safe_llm
from .chains_constants import (
    DEFAULT_MAX_WORD_COUNT_FOR_SUMMARY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE_DETERMINISTIC,
    QA_MARKERS,
    ROUTER_CLASSIFICATION_PROMPT,
    SUMMARY_MARKERS,
)
from .chains_utils import create_trace_decorator

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class RouteType(Enum):
    """Routen-Typen für Chain-Auswahl."""

    QA = auto()
    SUMMARIZE = auto()
    TRANSFORM = auto()


@dataclass(slots=True)
class RouterConfig:
    """Konfiguration für Router-Chain."""

    enable_llm_routing: bool = False
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE_DETERMINISTIC


class KeikoRouterChain:
    """Heuristische Router-Kette mit optionalem LLM-Fallback."""

    def __init__(
        self,
        *,
        config: RouterConfig,
        on_qa: Callable[[str], Any],
        on_summarize: Callable[[str], Any],
        on_transform: Callable[[str], Any],
    ) -> None:
        """Initialisiert Router mit Ziel-Handlern."""
        self._config = config
        self._on_qa = on_qa
        self._on_summarize = on_summarize
        self._on_transform = on_transform

    async def ainvoke(self, user_input: str) -> Any:
        """Routet Eingabe zu passender Kette und liefert Ergebnis."""
        # Direkte Ausführung mit Tracing
        return await self._execute_routing(user_input)

    @create_trace_decorator("keiko_chains.router.execute_routing")
    async def _execute_routing(self, user_input: str) -> Any:
        """Führt die Routing-Logik aus.

        Args:
            user_input: Benutzereingabe.

        Returns:
            Ergebnis der gerouteten Chain.
        """
        route = await self._classify(user_input)
        try:
            if route == RouteType.QA:
                return await self._maybe_await(self._on_qa(user_input))
            if route == RouteType.SUMMARIZE:
                return await self._maybe_await(self._on_summarize(user_input))
            return await self._maybe_await(self._on_transform(user_input))
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensiv
            logger.warning(f"Routing-Ziel fehlgeschlagen - Validierungsfehler: {exc}")
            return await self._maybe_await(self._on_transform(user_input))
        except (ConnectionError, TimeoutError) as exc:  # pragma: no cover - defensiv
            logger.warning(f"Routing-Ziel fehlgeschlagen - Verbindungsproblem: {exc}")
            return await self._maybe_await(self._on_transform(user_input))
        except Exception as exc:  # pragma: no cover - defensiv
            logger.warning(f"Routing-Ziel fehlgeschlagen - Unerwarteter Fehler: {exc}")
            return await self._maybe_await(self._on_transform(user_input))

    @staticmethod
    def _classify_heuristic(text: str) -> RouteType | None:
        """Klassifiziert Text mit Heuristiken (schnell, deterministisch).

        Args:
            text: Zu klassifizierender Text

        Returns:
            RouteType oder None wenn keine Heuristik zutrifft
        """
        lowered = text.lower()

        if any(marker in lowered for marker in QA_MARKERS):
            return RouteType.QA

        if (any(marker in lowered for marker in SUMMARY_MARKERS) or
            len(text.split()) > DEFAULT_MAX_WORD_COUNT_FOR_SUMMARY):
            return RouteType.SUMMARIZE

        return None

    async def _classify_with_llm(self, text: str) -> RouteType | None:
        """Klassifiziert Text mit LLM (optional)."""
        if not (self._config.enable_llm_routing and LANGCHAIN_OPENAI_AVAILABLE):
            return None

        llm = create_safe_llm(self._config.model, self._config.temperature)
        if llm is None:
            return None

        try:
            prompt_content = f"{ROUTER_CLASSIFICATION_PROMPT}\n{text}"
            out = await llm.ainvoke([{"role": "user", "content": prompt_content}])  # type: ignore[union-attr]
            label = str(getattr(out, "content", "")).strip().upper()

            if label == "QA":
                return RouteType.QA
            if label == "SUMMARIZE":
                return RouteType.SUMMARIZE
        except Exception:  # pragma: no cover - defensiv
            pass

        return None

    async def _classify(self, text: str) -> RouteType:
        """Klassifiziert Eingabetext in einen RouteType."""
        # Heuristiken zuerst (schnell, deterministisch)
        heuristic_result = self._classify_heuristic(text)
        if heuristic_result is not None:
            return heuristic_result

        # Optionales LLM-Routing
        llm_result = await self._classify_with_llm(text)
        if llm_result is not None:
            return llm_result

        return RouteType.TRANSFORM

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Hilfsfunktion: awaitet Coroutine oder gibt Wert zurück."""
        if hasattr(value, "__await__"):
            return await value  # type: ignore[misc]
        return value


__all__ = [
    "KeikoRouterChain",
    "RouteType",
    "RouterConfig",
]
