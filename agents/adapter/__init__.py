"""Agents Adapter Package - Einheitliche Multi-Adapter-Schnittstelle."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from kei_logging import get_logger

from .._compat import safe_import as _safe_import
from ..constants import (
    ERROR_ADAPTER_FACTORY_UNAVAILABLE,
    FRAMEWORK_DISPLAY_NAMES,
    FRAMEWORK_FOUNDRY,
    VALIDATION_AVAILABLE,
)

logger = get_logger(__name__)


# Hilfsklassen
class CircuitBreakerState(Enum):
    """Circuit Breaker Status"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ScopedTokenCredential:
    """Token Credential mit Scope-Unterstützung"""

    token: str
    scope: list[str]
    expires_on: int = 0

    def get_token(self):
        """Holt Token für angegebene Scopes"""
        return self


# Adapter Factory
(
    AdapterFactory,
    FoundryConfig,
) = _safe_import(
    "agents.adapter.adapter_factory",
    ["AdapterFactory", "FoundryConfig"],
    label="AdapterFactory",
)

factory = None
try:
    factory = AdapterFactory()  # type: ignore[call-arg]
except Exception as e:
    logger.error(f"AdapterFactory nicht verfügbar: {e}")
    factory = None


# Verfügbarkeits-Funktionen
def get_available_adapters() -> list[str]:
    """Gibt Liste verfügbarer Adapter zurück"""
    return factory.get_available_adapters() if factory else []


def is_foundry_available() -> bool:
    """Prüft, ob der Foundry-Adapter verfügbar ist.

    Returns:
        True wenn Foundry-Adapter verfügbar ist
    """
    return FRAMEWORK_FOUNDRY in get_available_adapters()


def get_adapter_info(adapter_name: str) -> dict[str, Any]:
    """Gibt Informationen über einen Adapter zurück.

    Args:
        adapter_name: Name des Adapters

    Returns:
        Dictionary mit Adapter-Informationen
    """
    available_adapters = get_available_adapters()
    return {
        "available": adapter_name in available_adapters,
        "name": FRAMEWORK_DISPLAY_NAMES.get(adapter_name, "Unknown")
    }


# Factory-Funktionen
async def create_adapter(framework: str, config: dict[str, Any] | None = None) -> Any:
    """Erstellt Adapter für angegebenes Framework.

    Args:
        framework: Name des Frameworks (z.B. "foundry")
        config: Optionale Konfiguration für den Adapter

    Returns:
        Adapter-Instanz für das angegebene Framework

    Raises:
        KeikoServiceError: Wenn AdapterFactory nicht verfügbar ist
    """
    if not factory:
        from core.exceptions import KeikoServiceError

        raise KeikoServiceError(ERROR_ADAPTER_FACTORY_UNAVAILABLE, details={"framework": framework})
    return await factory.create_adapter(framework, config)


# Framework-spezifische Wrapper
async def create_foundry_adapter(config: dict[str, Any] | None = None) -> Any:
    """Erstellt Azure AI Foundry Adapter.

    Args:
        config: Optionale FoundryConfig-Parameter

    Returns:
        FoundryAdapter-Instanz
    """
    return await create_adapter("foundry", config)


async def create_adapter_for_framework(framework: str, **config) -> Any:
    """Erstellt Adapter für Framework mit Keyword-Argumenten.

    Args:
        framework: Name des Frameworks
        **config: Konfigurationsparameter als Keyword-Argumente

    Returns:
        Adapter-Instanz für das angegebene Framework
    """
    return await create_adapter(framework, config)


def validate_adapter_environment(framework: str) -> dict[str, Any]:
    """Validiert Umgebung für angegebenes Framework.

    Args:
        framework: Name des zu validierenden Frameworks

    Returns:
        Dictionary mit Validierungsergebnissen
    """
    available_adapters = get_available_adapters()
    return {
        "framework": framework,
        "valid": framework in available_adapters,
        "available": VALIDATION_AVAILABLE,
        "available_adapters": available_adapters,
    }


def get_adapter_system_status() -> dict[str, Any]:
    """Gibt umfassenden System-Status aller Adapter zurück.

    Returns:
        Dictionary mit System-Status-Informationen
    """
    available_adapters = get_available_adapters()
    return {
        "available_adapters": available_adapters,
        "foundry": FRAMEWORK_FOUNDRY in available_adapters,
        "total_adapters": len(available_adapters),
        "factory_available": factory is not None,
    }


def adapter_factory_session():
    """Context Manager für Factory Session"""

    class AdapterSession:
        """Context Manager für Adapter Factory Session.

        Ermöglicht automatische Cache-Bereinigung nach Session-Ende.
        """

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            if factory:
                factory.clear_cache()

    return AdapterSession()


# Imports
(FoundryAdapter,) = _safe_import("agents.adapter.foundry_adapter", ["FoundryAdapter"], label="FoundryAdapter")

# Exports
__all__ = [
    "AdapterFactory",
    "CircuitBreakerState",
    "FoundryAdapter",
    "FoundryConfig",
    "ScopedTokenCredential",
    "adapter_factory_session",
    "create_adapter",
    "create_adapter_for_framework",
    "create_foundry_adapter",
    "factory",
    "get_adapter_info",
    "get_adapter_system_status",
    "get_available_adapters",
    "validate_adapter_environment",
]

# Metadaten
__version__ = "0.0.1"
