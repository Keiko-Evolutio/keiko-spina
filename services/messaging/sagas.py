"""Einfache Saga-/Kompensations-Hilfen (SDK) für KEI-Bus."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)


class SagaStep:
    """Repräsentiert einen Saga-Schritt mit optionaler Kompensation."""

    def __init__(
        self,
        name: str,
        action: Callable[[], Awaitable[Any]],
        compensate: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self.action = action
        self.compensate = compensate


class Saga:
    """Orchestriert eine Reihe von SagaSteps mit Kompensation bei Fehlern."""

    def __init__(self, steps: list[SagaStep]) -> None:
        self.steps = steps
        self._results: list[tuple[SagaStep, Any]] = []

    async def execute(self) -> list[Any]:
        """Führt alle Schritte aus; kompensiert rückwärts bei Fehlern."""
        try:
            for step in self.steps:
                result = await step.action()
                self._results.append((step, result))
            return [r for _, r in self._results]
        except Exception:
            # Rückwärts kompensieren
            for step, result in reversed(self._results):
                if step.compensate is not None:
                    with contextlib.suppress(Exception):
                        await step.compensate(result)
            raise


__all__ = ["Saga", "SagaStep"]
