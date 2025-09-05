"""Configuration Management API - Refactored Version.

Dieses Modul stellt CRUD-Operationen für Agent-Konfigurationen bereit.
Refactored für bessere Code-Qualität, Wartbarkeit und Testbarkeit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Path, status

from data_models.api import AgentConfigurationCreateRequest, AgentConfigurationResponse
from kei_logging import get_logger

from ..common import (
    NamedInMemoryStorage,
    duplicate_name,
    generate_config_id,
    get_storage,
    resource_not_found,
)
from ..routes.base import create_router
from .constants import (
    FIELD_CATEGORY,
    FIELD_CREATED_AT,
    FIELD_ID,
    FIELD_IS_DEFAULT,
    FIELD_NAME,
    FIELD_SYSTEM_MESSAGE,
    FIELD_TOOLS,
    FIELD_UPDATED_AT,
    FIELD_VOICE_SETTINGS,
    LOG_CONFIG_CREATED,
    LOG_CONFIG_DELETED,
    LOG_CONFIG_NAME_CONFLICT,
    ROUTER_PREFIX,
    ROUTER_TAGS,
    STORAGE_NAME,
)

logger = get_logger(__name__)

# Router-Konfiguration
router = create_router(ROUTER_PREFIX, ROUTER_TAGS)

# Storage-Instanz für Konfigurationen
_storage: NamedInMemoryStorage[AgentConfigurationResponse] = get_storage(
    STORAGE_NAME,
    NamedInMemoryStorage,
    enable_ttl=False
)


def _validate_configuration_request(request: AgentConfigurationCreateRequest) -> None:
    """Validiert die Konfigurationsanfrage.

    Args:
        request: Zu validierende Konfigurationsanfrage

    Raises:
        HTTPException: Bei Validierungsfehlern
    """
    # Zusätzliche Validierungen können hier hinzugefügt werden
    # Basis-Validierung erfolgt bereits durch Pydantic


async def _check_name_conflict(name: str) -> None:
    """Prüft ob Konfigurationsname bereits existiert.

    Args:
        name: Zu prüfender Name

    Raises:
        HTTPException: Falls Name bereits existiert
    """
    if await _storage.exists_by_name(name):
        logger.warning(LOG_CONFIG_NAME_CONFLICT.format(name=name))
        raise duplicate_name(name, "Konfiguration")


def _build_configuration_dict(
    request: AgentConfigurationCreateRequest,
    config_id: str,
    timestamp: datetime
) -> dict[str, Any]:
    """Erstellt Konfigurations-Dictionary aus Request.

    Args:
        request: Konfigurationsanfrage
        config_id: Generierte Konfigurations-ID
        timestamp: Zeitstempel für created_at/updated_at

    Returns:
        Konfigurations-Dictionary
    """
    return {
        FIELD_ID: config_id,
        FIELD_NAME: request.name,
        FIELD_CATEGORY: request.category.value,
        FIELD_SYSTEM_MESSAGE: request.system_message,
        FIELD_TOOLS: [tool.model_dump() for tool in request.tools],
        FIELD_VOICE_SETTINGS: (
            request.voice_settings.model_dump()
            if request.voice_settings else None
        ),
        FIELD_IS_DEFAULT: request.is_default,
        FIELD_CREATED_AT: timestamp.isoformat(),
        FIELD_UPDATED_AT: timestamp.isoformat(),
    }


def _create_configuration_response(
    config_dict: dict[str, Any]
) -> AgentConfigurationResponse:
    """Erstellt AgentConfigurationResponse aus Dictionary.

    Args:
        config_dict: Konfigurations-Dictionary

    Returns:
        AgentConfigurationResponse-Instanz
    """
    return AgentConfigurationResponse(configuration=config_dict)


@router.post(
    "/",
    response_model=AgentConfigurationResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_configuration(
    request: AgentConfigurationCreateRequest
) -> AgentConfigurationResponse:
    """Erstellt neue Agent-Konfiguration.

    Diese Funktion orchestriert die Erstellung einer neuen Agent-Konfiguration
    durch Validierung, Konfliktprüfung und Persistierung.

    Args:
        request: Konfigurationsanfrage mit allen erforderlichen Daten

    Returns:
        Erstellte Konfiguration mit generierter ID und Zeitstempeln

    Raises:
        HTTPException: Bei Validierungsfehlern oder Name-Konflikten

    Example:
        ```python
        request = AgentConfigurationCreateRequest(
            name="Mein Agent",
            category=AgentConfigurationCategory.GENERAL,
            system_message="Du bist ein hilfreicher Assistent",
            tools=[],
            is_default=False
        )
        response = await create_configuration(request)
        ```
    """
    # Validierung der Anfrage
    _validate_configuration_request(request)

    # Prüfung auf Name-Konflikte
    await _check_name_conflict(request.name)

    # ID-Generierung und Zeitstempel
    config_id = generate_config_id()
    timestamp = datetime.now(UTC)

    # Konfigurations-Dictionary erstellen
    config_dict = _build_configuration_dict(request, config_id, timestamp)

    # Response-Objekt erstellen
    config_response = _create_configuration_response(config_dict)

    # In Storage persistieren
    await _storage.set(config_id, config_response)

    # Logging
    logger.info(LOG_CONFIG_CREATED.format(name=request.name, config_id=config_id))

    return config_response


@router.get(
    "/{configuration_id}",
    response_model=AgentConfigurationResponse
)
async def get_configuration(
    configuration_id: str = Path(..., description="Eindeutige Konfigurations-ID")
) -> AgentConfigurationResponse:
    """Ruft Agent-Konfiguration anhand der ID ab.

    Args:
        configuration_id: Eindeutige ID der Konfiguration

    Returns:
        Gefundene Konfiguration

    Raises:
        HTTPException: Falls Konfiguration nicht gefunden wird

    Example:
        ```python
        config = await get_configuration("cfg_12345678")
        logger.info(config.configuration["name"])
        ```
    """
    config = await _storage.get(configuration_id)

    if config is None:
        raise resource_not_found("Konfiguration", configuration_id)

    return config


@router.delete(
    "/{configuration_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_configuration(
    configuration_id: str = Path(..., description="Eindeutige Konfigurations-ID")
):
    """Löscht Agent-Konfiguration anhand der ID.

    Args:
        configuration_id: Eindeutige ID der zu löschenden Konfiguration

    Raises:
        HTTPException: Falls Konfiguration nicht gefunden wird

    Example:
        ```python
        await delete_configuration("cfg_12345678")
        ```
    """
    # Prüfe ob Konfiguration existiert
    config = await _storage.get(configuration_id)

    if config is None:
        raise resource_not_found("Konfiguration", configuration_id)

    # Lösche aus Storage
    await _storage.delete(configuration_id)

    # Logging
    config_name = config.configuration.get(FIELD_NAME, "Unbekannt")
    logger.info(LOG_CONFIG_DELETED.format(name=config_name, config_id=configuration_id))


# Utility-Funktionen für Tests und Debugging
async def get_all_configurations() -> dict[str, AgentConfigurationResponse]:
    """Ruft alle gespeicherten Konfigurationen ab.

    Diese Funktion ist hauptsächlich für Tests und Debugging gedacht.

    Returns:
        Dictionary mit allen Konfigurations-ID -> Konfiguration Mappings
    """
    return await _storage.list_all()


async def clear_all_configurations() -> None:
    """Löscht alle gespeicherten Konfigurationen.

    Diese Funktion ist hauptsächlich für Tests gedacht.
    """
    await _storage.clear()


async def get_configuration_count() -> int:
    """Gibt die Anzahl der gespeicherten Konfigurationen zurück.

    Returns:
        Anzahl der Konfigurationen
    """
    return await _storage.count()


async def find_configurations_by_category(category: str) -> list[AgentConfigurationResponse]:
    """Findet alle Konfigurationen einer bestimmten Kategorie.

    Args:
        category: Zu suchende Kategorie

    Returns:
        Liste der gefundenen Konfigurationen
    """
    def category_filter(config: AgentConfigurationResponse) -> bool:
        return config.configuration.get(FIELD_CATEGORY) == category

    results = await _storage.find_by_condition(category_filter)
    return [config for _, config in results]
