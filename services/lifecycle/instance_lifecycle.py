"""Instance Lifecycle – Warmup (Preloading) und Graceful Shutdown.

⚠️  DEPRECATED: Diese Implementierung ist veraltet!
    Verwende stattdessen: from services.lifecycle import lifecycle

Dieses Modul stellt Hooks bereit, die beim Start (Warmup) und beim Stop
(Graceful Shutdown) aufgerufen werden können, z. B. um Models/Clients zu laden
oder laufende Tasks geordnet zu beenden.

Die neue Implementierung bietet:
- Enterprise-Grade Features (Timeout, Retry, Monitoring)
- Bessere Performance und Skalierbarkeit
- Umfangreiches Logging und Error-Handling
- Integration mit dem Agent-Lifecycle-System
"""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class InstanceLifecycle:
    """Verwaltet Warmup/Shutdown‑Callbacks.

    ⚠️  DEPRECATED: Diese Klasse ist veraltet!
        Verwende stattdessen: from services.lifecycle import lifecycle
    """

    def __init__(self) -> None:
        warnings.warn(
            "InstanceLifecycle ist deprecated. Verwende 'from services.lifecycle import lifecycle'",
            DeprecationWarning,
            stacklevel=2
        )
        self._warmup_callbacks: list[Callable[[], asyncio.Future | None]] = []
        self._shutdown_callbacks: list[Callable[[], asyncio.Future | None]] = []

    def on_warmup(self, cb: Callable[[], asyncio.Future | None]) -> None:
        """Registriert einen Warmup‑Callback."""
        self._warmup_callbacks.append(cb)

    def on_shutdown(self, cb: Callable[[], asyncio.Future | None]) -> None:
        """Registriert einen Shutdown‑Callback."""
        self._shutdown_callbacks.append(cb)

    async def run_warmup(self) -> None:
        """Führt Warmup‑Callbacks sequenziell aus (best‑effort)."""
        for cb in self._warmup_callbacks:
            try:
                res = cb()
                if asyncio.isfuture(res) or asyncio.iscoroutine(res):
                    await res  # type: ignore[arg-type]
            except Exception as e:
                logger.debug(f"Warmup‑Callback Fehler: {e}")

    async def run_shutdown(self) -> None:
        """Führt Shutdown‑Callbacks sequenziell aus (best‑effort)."""
        for cb in self._shutdown_callbacks:
            try:
                res = cb()
                if asyncio.isfuture(res) or asyncio.iscoroutine(res):
                    await res  # type: ignore[arg-type]
            except Exception as e:
                logger.debug(f"Shutdown‑Callback Fehler: {e}")


# Globale Instanz
lifecycle = InstanceLifecycle()


__all__ = ["InstanceLifecycle", "lifecycle"]
