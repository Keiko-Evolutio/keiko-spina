"""Backward-Compatibility Adapter für InstanceLifecycle API.

Dieser Adapter ermöglicht eine nahtlose Migration von der alten
InstanceLifecycle API zur neuen AgentLifecycleManager API ohne
Breaking Changes für bestehenden Code.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agents.lifecycle import agent_lifecycle_manager
from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class LifecycleAdapter:
    """Adapter für Backward-Compatibility mit der alten InstanceLifecycle API.

    Konvertiert die alte API-Signatur zur neuen AgentLifecycleManager API
    und bietet dieselbe Funktionalität mit Enterprise-Grade Features.
    """

    def __init__(self) -> None:
        """Initialisiert den Lifecycle-Adapter."""
        self._manager = agent_lifecycle_manager

        # Fallback: Use legacy InstanceLifecycle
        from .instance_lifecycle import InstanceLifecycle
        self._legacy_lifecycle = InstanceLifecycle()
        logger.debug("LifecycleAdapter initialisiert")

    def on_warmup(self, cb: Callable[[], asyncio.Future | None]) -> None:
        """Registriert einen Warmup-Callback (Kompatibilitäts-API).

        Args:
            cb: Callback-Funktion (sync oder async)
        """
        # Konvertiere alte API-Signatur zur neuen API
        async def adapted_callback() -> None:
            try:
                result = cb()
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception(f"Warmup-Callback Fehler: {e}")

        if self._manager:
            self._manager.register_warmup_callback(adapted_callback)
        else:
            # Fallback: Use legacy lifecycle
            self._legacy_lifecycle.on_warmup(cb)
        logger.debug("Warmup-Callback über Adapter registriert")

    def on_shutdown(self, cb: Callable[[], asyncio.Future | None]) -> None:
        """Registriert einen Shutdown-Callback (Kompatibilitäts-API).

        Args:
            cb: Callback-Funktion (sync oder async)
        """
        # Konvertiere alte API-Signatur zur neuen API
        async def adapted_callback() -> None:
            try:
                result = cb()
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception(f"Shutdown-Callback Fehler: {e}")

        if self._manager:
            self._manager.register_shutdown_callback(adapted_callback)
        else:
            # Fallback: Use legacy lifecycle
            self._legacy_lifecycle.on_shutdown(cb)
        logger.debug("Shutdown-Callback über Adapter registriert")

    async def run_warmup(self) -> None:
        """Führt Warmup-Callbacks aus (Kompatibilitäts-API).

        Delegiert an den AgentLifecycleManager mit Enterprise-Grade Features.
        """
        logger.debug("Führe Warmup über Adapter aus")
        if self._manager:
            await self._manager.run_global_warmup()
        else:
            # Fallback: Use legacy lifecycle
            logger.debug("Verwende Legacy-Lifecycle für Warmup")
            await self._legacy_lifecycle.run_warmup()

    async def run_shutdown(self) -> None:
        """Führt Shutdown-Callbacks aus (Kompatibilitäts-API).

        Delegiert an den AgentLifecycleManager mit Enterprise-Grade Features.
        """
        logger.debug("Führe Shutdown über Adapter aus")
        if self._manager:
            await self._manager.run_global_shutdown()
        else:
            # Fallback: Use legacy lifecycle
            logger.debug("Verwende Legacy-Lifecycle für Shutdown")
            await self._legacy_lifecycle.run_shutdown()


class InstanceLifecycle(LifecycleAdapter):
    """Drop-in Replacement für die alte InstanceLifecycle Klasse.

    Bietet 100% API-Kompatibilität mit verbesserter Implementierung.
    """

    def __init__(self) -> None:
        """Initialisiert InstanceLifecycle mit neuer Implementierung."""
        super().__init__()
        logger.info("InstanceLifecycle mit verbesserter Implementierung initialisiert")


# Globale Instanz für Backward-Compatibility
lifecycle = InstanceLifecycle()

__all__ = ["InstanceLifecycle", "LifecycleAdapter", "lifecycle"]
