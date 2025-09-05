# backend/voice/common/common.py
"""Voice-Konfigurationsmanagement für Cosmos DB und Prompty-Dateien.

REFACTORED VERSION 2.0.0 - Enterprise-Ready mit Clean Code Standards.

Diese Datei stellt Legacy-Kompatibilität bereit und delegiert an die
neuen refactored Module. Alle neuen Implementierungen sollten die
spezifischen Module direkt verwenden.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final

from data_models import Configuration
from kei_logging import get_logger

# Import der refactored Module
from .config_manager import get_global_config_manager
from .constants import DEFAULT_PROMPTY_FILENAME, PROMPTY_DIR
from .cosmos_operations import _query_scalar as _cosmos_query_scalar
from .cosmos_operations import get_cosmos_container as _get_cosmos_container
from .prompty_parser import (
    load_prompty_file,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from azure.cosmos.aio import ContainerProxy

logger = get_logger(__name__)

# Legacy-Konstanten für Backward Compatibility
PROMPTY_DIR: Final[Path] = PROMPTY_DIR
DEFAULT_PROMPTY_FILENAME: Final[str] = DEFAULT_PROMPTY_FILENAME

# =============================================================================
# Legacy Database Operations - Delegiert an neue Module
# =============================================================================

@asynccontextmanager
async def get_cosmos_container() -> AsyncIterator[ContainerProxy | None]:
    """Cosmos DB Container-Zugriff mit automatischer Initialisierung.

    DEPRECATED: Verwende VoiceCosmosDBManager.get_container() stattdessen.
    Diese Funktion ist nur für Legacy-Kompatibilität verfügbar.
    """
    async with _get_cosmos_container() as container:
        yield container

async def _query_scalar(
    container: ContainerProxy,
    query: str,
    parameters: list[dict[str, Any]] | None = None,
) -> Any:
    """Führt Cosmos DB Query aus und gibt erstes Ergebnis zurück.

    DEPRECATED: Verwende VoiceCosmosDBManager.query_scalar() stattdessen.
    Diese Funktion ist nur für Legacy-Kompatibilität verfügbar.
    """
    return await _cosmos_query_scalar(container, query, parameters)

# =============================================================================
# Legacy Configuration Management - Delegiert an neue Module
# =============================================================================

async def _load_default_from_db(container: ContainerProxy) -> Configuration | None:
    """Lädt Standard-Konfiguration aus Cosmos DB.

    DEPRECATED: Verwende VoiceCosmosDBManager.load_default_configuration() stattdessen.
    Diese Funktion ist nur für Legacy-Kompatibilität verfügbar.
    """
    try:
        from .constants import COSMOS_QUERY_DEFAULT_CONFIG
        config_data = await _cosmos_query_scalar(container, COSMOS_QUERY_DEFAULT_CONFIG)
        if config_data:
            return Configuration(**config_data)
    except Exception as e:
        logger.exception("Fehler beim Laden der Standard-Konfiguration: %s", e)
    return None

async def _load_fallback_config() -> Configuration | None:
    """Lädt Fallback-Konfiguration aus lokaler Datei.

    DEPRECATED: Verwende FileConfigurationSource.load_fallback_configuration() stattdessen.
    Diese Funktion ist nur für Legacy-Kompatibilität verfügbar.
    """
    try:
        config_data = await load_prompty_file(PROMPTY_DIR / DEFAULT_PROMPTY_FILENAME, default=True)
        return Configuration(**config_data) if config_data else None
    except Exception as e:
        logger.exception("Fallback-Konfiguration konnte nicht geladen werden: %s", e)
        return None

async def get_default_configuration() -> Configuration | None:
    """Lädt Standard-Voice-Konfiguration aus DB oder Fallback-Datei.

    DEPRECATED: Verwende VoiceConfigurationManager.get_default_configuration() stattdessen.
    Diese Funktion delegiert an den neuen Configuration Manager.
    """
    manager = get_global_config_manager()
    return await manager.get_default_configuration()

async def get_default_configuration_data() -> dict[str, Any]:
    """Gibt Standard-Konfigurationsdaten als Dict zurück.

    DEPRECATED: Verwende VoiceConfigurationManager.get_default_configuration_data() stattdessen.
    Diese Funktion delegiert an den neuen Configuration Manager.
    """
    manager = get_global_config_manager()
    return await manager.get_default_configuration_data()

# =============================================================================
# Legacy Prompty File Processing - Delegiert an neue Module
# =============================================================================

# Alle Prompty-Processing-Funktionen sind bereits in den neuen Modulen
# verfügbar und werden über die Imports bereitgestellt.
# Diese Datei stellt nur noch Legacy-Kompatibilität bereit.

# Die folgenden Funktionen sind bereits über Imports verfügbar:
# - _extract_config_from_prompty (aus prompty_parser)
# - load_prompty_file (aus prompty_parser)
# - load_prompty_config (aus prompty_parser)

# =============================================================================
# Module Information
# =============================================================================

__version__ = "2.0.0"
__status__ = "Legacy Compatibility Layer"

# Legacy-Funktionen sind über Imports verfügbar:
# - get_default_configuration()
# - get_default_configuration_data()
# - load_prompty_config()
# - load_prompty_file()
# - get_cosmos_container()
# - _query_scalar()
# - _extract_config_from_prompty()

# Für neue Implementierungen verwende die spezifischen Module:
# from voice.common.config_manager import VoiceConfigurationManager
# from voice.common.cosmos_operations import VoiceCosmosDBManager
# from voice.common.prompty_parser import PromptyParser
