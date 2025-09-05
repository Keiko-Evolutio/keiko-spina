"""Zentrale Adapter Factory für alle Adapter-Typen"""

from dataclasses import dataclass
from typing import Any

from config.settings import settings as config
from kei_logging import get_logger

from ..constants import (
    DEFAULT_API_VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    ERROR_UNKNOWN_FRAMEWORK,
    FRAMEWORK_FOUNDRY,
    FRAMEWORK_MODULE_MAP,
    LOG_ADAPTER_CREATED,
    SUPPORTED_FRAMEWORKS,
)

logger = get_logger(__name__)


@dataclass
class FoundryConfig:
    """Azure AI Foundry Konfiguration."""

    endpoint: str = ""
    api_key: str = ""
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_version: str = DEFAULT_API_VERSION
    project_id: str = ""


async def _create_specific_adapter(framework: str, adapter_config: dict[str, Any]) -> Any | None:
    """Erstellt spezifischen Adapter basierend auf Framework.

    Args:
        framework: Name des Frameworks
        adapter_config: Konfigurationsdictionary für den Adapter

    Returns:
        Adapter-Instanz oder None bei Fehlern
    """
    try:
        if framework == FRAMEWORK_FOUNDRY:
            from .foundry_adapter import FoundryAdapter

            return FoundryAdapter(FoundryConfig(**adapter_config))

        logger.error(f"{ERROR_UNKNOWN_FRAMEWORK}: {framework}")
        return None

    except ImportError as e:
        logger.warning(f"{framework} SDK nicht verfügbar: {e}")
        return None
    except Exception as e:
        logger.error(f"Fehler beim Erstellen von {framework} Adapter: {e}")
        return None


def _get_default_config(framework: str) -> dict[str, Any]:
    """Gibt Standard-Konfiguration für Framework zurück.

    Args:
        framework: Name des Frameworks

    Returns:
        Dictionary mit Standard-Konfigurationswerten
    """
    if framework == FRAMEWORK_FOUNDRY:
        return {
            "endpoint": config.project_keiko_services_endpoint
            or config.project_keiko_openai_endpoint,
            "api_key": config.project_keiko_api_key,
            "model_name": config.project_keiko_model_deployment_name,
            "api_version": getattr(config, "project_keiko_api_version", DEFAULT_API_VERSION),
            "project_id": getattr(config, "project_keiko_project_id", ""),
        }

    return {}


def get_default_config(framework: str) -> dict[str, Any]:
    """Public API für Standard-Konfiguration.

    Args:
        framework: Name des Frameworks

    Returns:
        Dictionary mit Standard-Konfigurationswerten
    """
    return _get_default_config(framework)


def get_available_adapters() -> list[str]:
    """Gibt Liste verfügbarer Adapter zurück.

    Prüft dynamisch, welche Adapter-Module importiert werden können.

    Returns:
        Liste der verfügbaren Framework-Namen
    """
    available = []
    for framework in SUPPORTED_FRAMEWORKS:
        try:
            module_name = FRAMEWORK_MODULE_MAP[framework]
            __import__(f"agents.adapter.{module_name}")
            available.append(framework)
        except ImportError:
            pass
    return available


class AdapterFactory:
    """Zentrale Factory für alle Adapter-Typen"""

    def __init__(self):
        self._adapters = {}

    async def create_adapter(
        self, framework: str, adapter_config: dict[str, Any] | None = None
    ) -> Any | None:
        """Erstellt Adapter für angegebenes Framework"""
        # Cache-Prüfung
        if framework in self._adapters:
            return self._adapters[framework]

        # Konfiguration vorbereiten
        if not adapter_config:
            adapter_config = get_default_config(framework)

        # Adapter erstellen
        adapter = await _create_specific_adapter(framework, adapter_config)

        if adapter:
            self._adapters[framework] = adapter
            logger.info(LOG_ADAPTER_CREATED.format(framework=framework.title()))

        return adapter

    def clear_cache(self):
        """Leert Adapter-Cache"""
        self._adapters.clear()

    @staticmethod
    def get_available_adapters() -> list[str]:
        """Gibt Liste der verfügbaren Adapter zurück.

        Returns:
            Liste der verfügbaren Framework-Namen
        """
        return get_available_adapters()
