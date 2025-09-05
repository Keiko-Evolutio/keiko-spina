# backend/voice/common/config_manager.py
"""Configuration-Management mit Factory-Pattern und Dependency Injection.

Zentrale Configuration-Management-Klasse mit Fallback-Strategien,
Caching und Dependency Injection für bessere Testbarkeit.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Protocol

from data_models import Configuration
from kei_logging import get_logger

from .constants import (
    DEFAULT_PROMPTY_FILENAME,
    ERROR_FALLBACK_CONFIG,
    ERROR_LOAD_DEFAULT_CONFIG,
    PROMPTY_DIR,
    VoiceConfig,
    VoiceFeatureFlags,
)
from .exceptions import (
    ConfigurationError,
    ConfigurationNotFoundError,
    DefaultConfigurationError,
    wrap_exception,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .cosmos_operations import CosmosSettings, VoiceCosmosDBManager
    from .prompty_parser import PromptyParser

logger = get_logger(__name__)


# =============================================================================
# Protocols
# =============================================================================

class ConfigurationSource(Protocol):
    """Protocol für Configuration-Quellen."""

    async def load_default_configuration(self) -> dict[str, Any] | None:
        """Lädt Standard-Konfiguration."""
        ...

    async def load_configuration_by_id(self, config_id: str) -> dict[str, Any] | None:
        """Lädt Konfiguration anhand der ID."""
        ...


class FallbackConfigurationSource(Protocol):
    """Protocol für Fallback-Configuration-Quellen."""

    async def load_fallback_configuration(self) -> dict[str, Any] | None:
        """Lädt Fallback-Konfiguration."""
        ...


# =============================================================================
# Configuration Sources
# =============================================================================

class DatabaseConfigurationSource:
    """Database-basierte Configuration-Quelle."""

    def __init__(self, cosmos_manager: VoiceCosmosDBManager) -> None:
        """Initialisiert Database Configuration Source.

        Args:
            cosmos_manager: Cosmos DB Manager
        """
        self._cosmos_manager = cosmos_manager
        self._logger = logger

    async def load_default_configuration(self) -> dict[str, Any] | None:
        """Lädt Standard-Konfiguration aus Database.

        Returns:
            Standard-Konfiguration oder None
        """
        try:
            return await self._cosmos_manager.load_default_configuration()
        except Exception as e:
            self._logger.exception(ERROR_LOAD_DEFAULT_CONFIG, e)
            return None

    async def load_configuration_by_id(self, config_id: str) -> dict[str, Any] | None:
        """Lädt Konfiguration anhand der ID aus Database.

        Args:
            config_id: Konfigurations-ID

        Returns:
            Konfiguration oder None
        """
        try:
            return await self._cosmos_manager.load_configuration_by_id(config_id)
        except Exception as e:
            self._logger.exception(f"Fehler beim Laden der Konfiguration {config_id}: {e}")
            return None


class FileConfigurationSource:
    """File-basierte Configuration-Quelle."""

    def __init__(
        self,
        prompty_parser: PromptyParser,
        *,
        prompty_dir: Path = PROMPTY_DIR,
        default_filename: str = DEFAULT_PROMPTY_FILENAME,
    ) -> None:
        """Initialisiert File Configuration Source.

        Args:
            prompty_parser: Prompty Parser
            prompty_dir: Prompty-Verzeichnis
            default_filename: Default-Dateiname
        """
        self._prompty_parser = prompty_parser
        self._prompty_dir = prompty_dir
        self._default_filename = default_filename
        self._logger = logger

    async def load_fallback_configuration(self) -> dict[str, Any] | None:
        """Lädt Fallback-Konfiguration aus Datei.

        Returns:
            Fallback-Konfiguration oder None
        """
        try:
            file_path = self._prompty_dir / self._default_filename
            return await self._prompty_parser.load_from_file(file_path, is_default=True)
        except Exception as e:
            self._logger.exception(ERROR_FALLBACK_CONFIG, e)
            return None

    async def load_configuration_from_file(
        self,
        file_path: Path,
        *,
        is_default: bool = False,
    ) -> dict[str, Any] | None:
        """Lädt Konfiguration aus spezifischer Datei.

        Args:
            file_path: Pfad zur Datei
            is_default: Ob es sich um Default-Configuration handelt

        Returns:
            Konfiguration oder None
        """
        try:
            return await self._prompty_parser.load_from_file(file_path, is_default=is_default)
        except Exception as e:
            self._logger.exception(f"Fehler beim Laden der Datei {file_path}: {e}")
            return None


# =============================================================================
# Configuration Manager
# =============================================================================

class VoiceConfigurationManager:
    """Zentrale Configuration-Management-Klasse.

    Kombiniert verschiedene Configuration-Quellen mit Fallback-Strategien,
    Caching und Dependency Injection.
    """

    def __init__(
        self,
        *,
        database_source: ConfigurationSource | None = None,
        fallback_source: FallbackConfigurationSource | None = None,
        enable_caching: bool = False,
    ) -> None:
        """Initialisiert Voice Configuration Manager.

        Args:
            database_source: Database Configuration Source
            fallback_source: Fallback Configuration Source
            enable_caching: Caching aktivieren
        """
        self._database_source = database_source
        self._fallback_source = fallback_source
        self._enable_caching = enable_caching
        self._cache: dict[str, Configuration] = {}
        self._logger = logger

    async def get_default_configuration(self) -> Configuration | None:
        """Lädt Standard-Voice-Konfiguration mit Fallback-Strategie.

        Returns:
            Standard-Konfiguration oder None

        Raises:
            DefaultConfigurationError: Bei kritischen Fehlern
        """
        try:
            # Cache-Check
            if self._enable_caching and "default" in self._cache:
                self._logger.debug("Standard-Konfiguration aus Cache geladen")
                return self._cache["default"]

            # Versuche Database-Source
            if self._database_source and VoiceFeatureFlags.ENABLE_COSMOS_DB:
                config_data = await self._database_source.load_default_configuration()
                if config_data:
                    config = Configuration(**config_data)
                    if self._enable_caching:
                        self._cache["default"] = config
                    self._logger.debug("Standard-Konfiguration aus Database geladen")
                    return config

            # Fallback zu File-Source
            if self._fallback_source and VoiceFeatureFlags.ENABLE_FALLBACK_CONFIG:
                config_data = await self._fallback_source.load_fallback_configuration()
                if config_data:
                    config = Configuration(**config_data)
                    if self._enable_caching:
                        self._cache["default"] = config
                    self._logger.debug("Standard-Konfiguration aus Fallback geladen")
                    return config

            # Keine Konfiguration gefunden
            self._logger.warning("Keine Standard-Konfiguration verfügbar")
            return None

        except Exception as e:
            raise wrap_exception(
                e,
                DefaultConfigurationError,
                f"Fehler beim Laden der Standard-Konfiguration: {e}"
            )

    async def get_default_configuration_data(self) -> dict[str, Any]:
        """Gibt Standard-Konfigurationsdaten als Dict zurück.

        Returns:
            Standard-Konfigurationsdaten oder leeres Dict
        """
        config = await self.get_default_configuration()
        return asdict(config) if config else {}

    async def get_configuration_by_id(self, config_id: str) -> Configuration | None:
        """Lädt Konfiguration anhand der ID.

        Args:
            config_id: Konfigurations-ID

        Returns:
            Konfiguration oder None

        Raises:
            ConfigurationNotFoundError: Wenn Konfiguration nicht gefunden
        """
        try:
            # Cache-Check
            if self._enable_caching and config_id in self._cache:
                self._logger.debug(f"Konfiguration {config_id} aus Cache geladen")
                return self._cache[config_id]

            # Database-Source versuchen
            if self._database_source:
                config_data = await self._database_source.load_configuration_by_id(config_id)
                if config_data:
                    config = Configuration(**config_data)
                    if self._enable_caching:
                        self._cache[config_id] = config
                    self._logger.debug(f"Konfiguration {config_id} aus Database geladen")
                    return config

            # Konfiguration nicht gefunden
            raise ConfigurationNotFoundError(config_id)

        except ConfigurationNotFoundError:
            # Re-raise bekannte Exceptions
            raise
        except Exception as e:
            raise wrap_exception(
                e,
                ConfigurationError,
                f"Fehler beim Laden der Konfiguration {config_id}: {e}",
                config_id=config_id
            )

    async def load_configuration_from_content(
        self,
        content: str,
        *,
        source: str = "inline",
    ) -> Configuration | None:
        """Lädt Konfiguration aus Content-String.

        Args:
            content: Prompty-Content
            source: Quelle des Contents

        Returns:
            Konfiguration oder None
        """
        try:
            # Verwende Fallback-Source falls verfügbar
            if hasattr(self._fallback_source, "_prompty_parser"):
                parser = self._fallback_source._prompty_parser  # type: ignore[attr-defined]
                config_data = parser.extract_config_from_content(content, source=source)
                return Configuration(**config_data)

            # Fallback: Erstelle neuen Parser
            from .prompty_parser import create_prompty_parser
            parser = create_prompty_parser()
            config_data = parser.extract_config_from_content(content, source=source)
            return Configuration(**config_data)

        except Exception as e:
            self._logger.exception(f"Fehler beim Laden der Konfiguration aus Content: {e}")
            return None

    def clear_cache(self) -> None:
        """Leert den Configuration-Cache."""
        self._cache.clear()
        self._logger.debug("Configuration-Cache geleert")

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück.

        Returns:
            Cache-Statistiken
        """
        return {
            "enabled": self._enable_caching,
            "size": len(self._cache),
            "keys": list(self._cache.keys()),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_voice_configuration_manager(
    *,
    cosmos_settings: CosmosSettings | None = None,
    enable_caching: bool = VoiceFeatureFlags.ENABLE_PROMPTY_CACHE,
    prompty_dir: Path | None = None,
) -> VoiceConfigurationManager:
    """Factory für Voice Configuration Manager.

    Args:
        cosmos_settings: Cosmos DB-Settings
        enable_caching: Caching aktivieren
        prompty_dir: Prompty-Verzeichnis

    Returns:
        Konfigurierter VoiceConfigurationManager
    """
    from .cosmos_operations import create_voice_cosmos_manager
    from .prompty_parser import create_prompty_parser

    # Database Source
    database_source = None
    if VoiceFeatureFlags.ENABLE_COSMOS_DB:
        cosmos_manager = create_voice_cosmos_manager(cosmos_settings)
        database_source = DatabaseConfigurationSource(cosmos_manager)

    # Fallback Source
    fallback_source = None
    if VoiceFeatureFlags.ENABLE_FALLBACK_CONFIG:
        prompty_parser = create_prompty_parser()
        fallback_source = FileConfigurationSource(
            prompty_parser,
            prompty_dir=prompty_dir or VoiceConfig.PROMPTY_DIR,
            default_filename=VoiceConfig.DEFAULT_PROMPTY_FILE,
        )

    return VoiceConfigurationManager(
        database_source=database_source,
        fallback_source=fallback_source,
        enable_caching=enable_caching,
    )


# =============================================================================
# Global Instance
# =============================================================================

# Globale Instanz für Legacy-Kompatibilität
_global_config_manager: VoiceConfigurationManager | None = None


def get_global_config_manager() -> VoiceConfigurationManager:
    """Gibt globale Configuration-Manager-Instanz zurück.

    Returns:
        Globale VoiceConfigurationManager-Instanz
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = create_voice_configuration_manager()
    return _global_config_manager


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

async def get_default_configuration() -> Configuration | None:
    """Legacy-Kompatibilität für get_default_configuration().

    Deprecated: Verwende VoiceConfigurationManager.get_default_configuration() stattdessen.
    """
    manager = get_global_config_manager()
    return await manager.get_default_configuration()


async def get_default_configuration_data() -> dict[str, Any]:
    """Legacy-Kompatibilität für get_default_configuration_data().

    Deprecated: Verwende VoiceConfigurationManager.get_default_configuration_data() stattdessen.
    """
    manager = get_global_config_manager()
    return await manager.get_default_configuration_data()


def load_prompty_config(content: str) -> Configuration | None:
    """Legacy-Kompatibilität für load_prompty_config().

    Deprecated: Verwende VoiceConfigurationManager.load_configuration_from_content() stattdessen.
    """
    import asyncio
    manager = get_global_config_manager()
    return asyncio.run(manager.load_configuration_from_content(content))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Protocols
    "ConfigurationSource",
    # Classes
    "DatabaseConfigurationSource",
    "FallbackConfigurationSource",
    "FileConfigurationSource",
    "VoiceConfigurationManager",
    # Factory Functions
    "create_voice_configuration_manager",
    # Legacy Compatibility
    "get_default_configuration",
    "get_default_configuration_data",
    "get_global_config_manager",
    "load_prompty_config",
]
