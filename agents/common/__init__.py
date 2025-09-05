"""Common Package - Unified Multi-Agent Operations."""

import logging
from typing import Any

# Fallback für kei_logging
try:
    from kei_logging import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

# Fallback für _compat
try:
    from .._compat import (
        unavailable_function_factory as _unavailable_function_factory,
    )
except ImportError:
    def _unavailable_function_factory(name: str):
        def unavailable_func(*_args, **_kwargs):
            raise NotImplementedError(f"Function {name} is not available")
        return unavailable_func


try:
    from agents.common import adapters as _adapters_mod
    from agents.common import operations as _operations_mod

    _UNIFIED_AVAILABLE = True
    logger.debug("Unified API erfolgreich geladen")


    async def initialize_adapters():  # type: ignore[no-redef]
        return await _adapters_mod.initialize_adapters()

    async def get_adapter(framework: str):  # type: ignore[no-redef]
        return await _adapters_mod.get_adapter(framework)

    def clear_adapters() -> None:  # type: ignore[no-redef]
        return _adapters_mod.clear_adapters()

    def multi_agent_session():  # type: ignore[no-redef]

        return _adapters_mod.multi_agent_session()

    async def get_agents(framework: str):  # type: ignore[no-redef]
        return await _operations_mod.get_agents(framework)

    async def create_agent(framework: str, config: dict[str, Any]):  # type: ignore[no-redef]
        return await _operations_mod.create_agent(framework, config)

    async def execute_agent_task(agent_id: str, task: str, framework: str | None = None, **kwargs):  # type: ignore[no-redef]
        return await _operations_mod.execute_agent_task(agent_id, task, framework, **kwargs)

    async def get_foundry_agents():  # type: ignore[no-redef]
        return await _operations_mod.get_foundry_agents()

    async def create_foundry_thread(agent_id: str):  # type: ignore[no-redef]
        return await _operations_mod.create_foundry_thread(agent_id)

    async def execute_foundry_agent(agent_id: str, thread_id: str, message: str):  # type: ignore[no-redef]
        return await _operations_mod.execute_foundry_agent(agent_id, thread_id, message)

    async def create_thread_message(thread_id: str, content: str, role: str):  # type: ignore[no-redef]
        return await _operations_mod.create_thread_message(thread_id, content, role)

    async def get_all_agents():  # type: ignore[no-redef]
        return await _operations_mod.get_all_agents()

    async def find_best_agent_for_task(task: str, preferred_frameworks: list[str] | None = None):  # type: ignore[no-redef]
        return await _operations_mod.find_best_agent_for_task(task, preferred_frameworks)

except ImportError as e:
    logger.error(f"Unified API nicht verfügbar: {e}")
    _UNIFIED_AVAILABLE = False

    # Define fallback variables to prevent NameError
    _adapters_mod = None
    _operations_mod = None

    _unavailable = _unavailable_function_factory("Agent-Operationen")

    initialize_adapters = get_adapter = clear_adapters = _unavailable  # type: ignore[assignment]
    multi_agent_session = _unavailable  # type: ignore[assignment]
    get_agents = create_agent = execute_agent_task = _unavailable  # type: ignore[assignment]
    get_foundry_agents = create_foundry_thread = _unavailable  # type: ignore[assignment]
    execute_foundry_agent = create_thread_message = _unavailable  # type: ignore[assignment]

    get_all_agents = find_best_agent_for_task = _unavailable  # type: ignore[assignment]
    startup_agents = shutdown_agents = _unavailable  # type: ignore[assignment]


try:
    from ..adapter import CircuitBreakerState, ScopedTokenCredential

    _ADAPTER_COMPONENTS = True
except ImportError:
    _ADAPTER_COMPONENTS = False

    _unavailable_adapter_component = _unavailable_function_factory("Adapter-Komponenten")

    class ScopedTokenCredential:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            _unavailable_adapter_component()

    class CircuitBreakerState:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            _unavailable_adapter_component()


def is_unified_available() -> bool:
    """Prüft Unified API Verfügbarkeit."""
    return _UNIFIED_AVAILABLE


def get_common_api_status() -> dict[str, Any]:
    """Gibt Common API Status zurück."""
    return {
        "unified_api": _UNIFIED_AVAILABLE,
        "adapter_components": _ADAPTER_COMPONENTS,
        "healthy": _UNIFIED_AVAILABLE,
    }



async def execute_sk_orchestration(config: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Führt Semantic Kernel Orchestrierung aus."""
    from .operations import execute_sk_orchestration as _exec

    return await _exec(config, instruction)


async def create_sk_agent(config: dict[str, Any]) -> Any | None:
    """Erstellt Semantic Kernel Agent."""
    from .operations import create_sk_agent as _create

    return await _create(config)



async def startup_agents() -> bool:
    """Startet das Agent System."""
    try:
        logger.info("Starte Agent System...")


        await initialize_adapters()


        logger.info("Initialisiere Agent Registry...")
        from agents.registry.dynamic_registry import dynamic_registry
        registry_success = await dynamic_registry.initialize()

        if registry_success:
            logger.info(f"Agent Registry erfolgreich initialisiert - {len(dynamic_registry.agents)} Agents geladen")
        else:
            logger.warning("Agent Registry Initialisierung fehlgeschlagen")

        logger.info("Agent System erfolgreich gestartet")
        return True
    except Exception as startup_error:
        logger.error(f"Fehler beim Starten des Agent Systems: {startup_error}")
        return False


async def shutdown_agents() -> None:
    """Beendet das Agent System."""
    try:
        logger.info("Beende Agent System...")


        clear_adapters()

        logger.info("Agent System erfolgreich beendet")
    except Exception as shutdown_error:
        logger.error(f"Fehler beim Beenden des Agent Systems: {shutdown_error}")


__all__ = [
    # Status
    "is_unified_available",
    "get_common_api_status",
    # System Lifecycle
    "startup_agents",
    "shutdown_agents",
    # Adapter Management
    "initialize_adapters",
    "get_adapter",
    "multi_agent_session",
    "clear_adapters",
    # Unified Operations
    "get_agents",
    "create_agent",
    "execute_agent_task",
    # Foundry Operations
    "get_foundry_agents",
    "create_foundry_thread",
    "execute_foundry_agent",
    "create_thread_message",
    # Cross-Framework
    "get_all_agents",
    "find_best_agent_for_task",
    # Semantic Kernel Operations
    "execute_sk_orchestration",
    "create_sk_agent",
    # Adapter Components
    "ScopedTokenCredential",
    "CircuitBreakerState",
]
