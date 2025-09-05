"""Adapter-Verwaltung für Multi-Agent-Frameworks."""

from contextlib import asynccontextmanager
from typing import Any

import agents.adapter as adapter_module
from kei_logging import get_logger

logger = get_logger(__name__)


FRAMEWORKS = ["foundry"]
ADAPTER_CONFIG = {
    "foundry": ("create_foundry_adapter", "is_foundry_available"),
}


_adapters: dict[str, Any] = {}


async def initialize_adapters() -> dict[str, Any]:
    """Initialisiert alle verfügbaren Adapter."""
    global _adapters


    _adapters.clear()

    for framework in FRAMEWORKS:
        create_func, check_func = ADAPTER_CONFIG[framework]

        try:

            if check_func and not getattr(adapter_module, check_func, lambda: False)():
                logger.debug(f"{framework} nicht verfügbar")
                continue


            create = getattr(adapter_module, create_func, None)
            if create:

                _adapters[framework] = await create()
                logger.debug(f"{framework} Adapter initialisiert")

        except Exception as e:
            logger.warning(f"{framework} Adapter-Fehler: {e}")

    return {fw: adapter for fw, adapter in _adapters.items() if adapter}


async def get_adapter(framework: str) -> Any | None:
    """Holt Framework-Adapter

    Args:
        framework: Framework-Name

    Returns:
        Adapter-Instanz oder None
    """
    if framework not in _adapters:
        # Lazy Loading für einzelne Adapter
        await _load_single_adapter(framework)

    return _adapters.get(framework)


async def _load_single_adapter(framework: str) -> None:
    """Lädt einzelnen Adapter nach"""
    if framework not in ADAPTER_CONFIG:
        return

    create_func, _ = ADAPTER_CONFIG[framework]

    try:
        create = getattr(adapter_module, create_func, None)
        if create:
            # Adapter holen services intern
            _adapters[framework] = await create()
            logger.debug(f"{framework} Adapter nachgeladen")
    except Exception as e:
        logger.warning(f"Fehler beim Laden von {framework}: {e}")


def clear_adapters() -> None:
    """Leert Adapter-Cache"""
    global _adapters
    _adapters.clear()
    logger.debug("Adapter-Cache geleert")


@asynccontextmanager
async def multi_agent_session():
    """Context Manager für Multi-Agent-Sessions"""
    try:
        await initialize_adapters()
        yield _adapters
    finally:
        clear_adapters()
