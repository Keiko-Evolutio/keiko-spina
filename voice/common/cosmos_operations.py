# backend/voice/common/cosmos_operations.py
"""Wiederverwendbare Cosmos DB Operations für das Voice-Modul.

Konsolidierte und abstrahierte Cosmos DB-Operationen mit einheitlicher
Error-Handling-Strategie und Dependency Injection.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

from azure.cosmos import PartitionKey
from azure.cosmos.aio import ContainerProxy, CosmosClient

from kei_logging import get_logger

from .constants import (
    COSMOS_ENABLE_CROSS_PARTITION,
    COSMOS_PARTITION_KEY_PATH,
    COSMOS_QUERY_DEFAULT_CONFIG,
    DEFAULT_QUERY_TIMEOUT,
    ERROR_COSMOSDB_CONNECT,
    ERROR_COSMOSDB_CONNECTION,
    ERROR_COSMOSDB_QUERY,
    VoiceFeatureFlags,
)
from .exceptions import (
    CosmosDBConnectionError,
    CosmosDBQueryError,
    wrap_exception,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


# =============================================================================
# Protocols
# =============================================================================

class CosmosSettings(Protocol):
    """Protocol für Cosmos DB-Settings."""

    @property
    def cosmosdb_connection(self) -> Any:
        """Cosmos DB Connection String."""
        ...

    @property
    def database_name(self) -> str:
        """Database Name."""
        ...

    @property
    def container_name(self) -> str:
        """Container Name."""
        ...


# =============================================================================
# Cosmos DB Base Operations
# =============================================================================

class CosmosDBManager:
    """Wiederverwendbare Cosmos DB-Operations.

    Abstrahiert Cosmos DB-Zugriff mit einheitlicher Error-Handling-Strategie,
    Dependency Injection und Performance-Optimierungen.
    """

    def __init__(
        self,
        settings: CosmosSettings,
        *,
        partition_key_path: str = COSMOS_PARTITION_KEY_PATH,
        enable_cross_partition: bool = COSMOS_ENABLE_CROSS_PARTITION,
        query_timeout: int = DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        """Initialisiert Cosmos DB Manager.

        Args:
            settings: Cosmos DB-Settings
            partition_key_path: Pfad für Partition Key
            enable_cross_partition: Cross-Partition-Queries aktivieren
            query_timeout: Query-Timeout in Sekunden
        """
        self._settings = settings
        self._partition_key_path = partition_key_path
        self._enable_cross_partition = enable_cross_partition
        self._query_timeout = query_timeout
        self._logger = logger

    @asynccontextmanager
    async def get_container(self) -> AsyncIterator[ContainerProxy | None]:
        """Cosmos DB Container-Zugriff mit automatischer Initialisierung.

        Yields:
            Container-Proxy oder None bei Fehlern

        Raises:
            CosmosDBConnectionError: Bei Verbindungsfehlern
        """
        if not VoiceFeatureFlags.ENABLE_COSMOS_DB:
            self._logger.debug("Cosmos DB ist deaktiviert")
            yield None
            return

        if not self._settings.cosmosdb_connection:
            self._logger.warning(ERROR_COSMOSDB_CONNECTION)
            yield None
            return

        try:
            # SecretStr kompatibler Zugriff
            conn_str = self._get_connection_string()

            async with CosmosClient.from_connection_string(conn_str) as client:
                database = await client.create_database_if_not_exists(
                    id=self._settings.database_name
                )
                container = await database.create_container_if_not_exists(
                    id=self._settings.container_name,
                    partition_key=PartitionKey(path=self._partition_key_path),
                )
                yield container

        except Exception as e:
            self._logger.exception(ERROR_COSMOSDB_CONNECT, e)
            raise wrap_exception(
                e,
                CosmosDBConnectionError,
                f"Cosmos DB-Verbindung fehlgeschlagen: {e}",
                details={"database": self._settings.database_name, "container": self._settings.container_name}
            )

    def _get_connection_string(self) -> str:
        """Extrahiert Connection String aus Settings.

        Returns:
            Connection String
        """
        conn = self._settings.cosmosdb_connection
        return (
            conn.get_secret_value()
            if hasattr(conn, "get_secret_value")
            else str(conn)
        )

    async def query_scalar(
        self,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        *,
        operation_name: str = "query_scalar",
    ) -> Any:
        """Führt Cosmos DB Query aus und gibt erstes Ergebnis zurück.

        Args:
            query: SQL-Query
            parameters: Query-Parameter
            operation_name: Name der Operation für Logging

        Returns:
            Erstes Query-Ergebnis oder None

        Raises:
            CosmosDBQueryError: Bei Query-Fehlern
        """
        try:
            async with self.get_container() as container:
                if not container:
                    return None

                items = container.query_items(
                    query=query,
                    parameters=parameters or [],
                    enable_cross_partition_query=self._enable_cross_partition,
                )

                async for item in items:
                    self._logger.debug(f"{operation_name}: Ergebnis gefunden")
                    return item

                self._logger.debug(f"{operation_name}: Kein Ergebnis gefunden")
                return None

        except Exception as e:
            self._logger.exception(ERROR_COSMOSDB_QUERY, e)
            raise wrap_exception(
                e,
                CosmosDBQueryError,
                query,
                f"Query-Fehler: {e}",
                details={"operation": operation_name, "parameters": parameters}
            )

    async def query_items(
        self,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        *,
        max_results: int | None = None,
        operation_name: str = "query_items",
    ) -> list[dict[str, Any]]:
        """Führt Cosmos DB Query aus und gibt alle Ergebnisse zurück.

        Args:
            query: SQL-Query
            parameters: Query-Parameter
            max_results: Maximale Anzahl Ergebnisse
            operation_name: Name der Operation für Logging

        Returns:
            Liste aller Query-Ergebnisse

        Raises:
            CosmosDBQueryError: Bei Query-Fehlern
        """
        try:
            async with self.get_container() as container:
                if not container:
                    return []

                results: list[dict[str, Any]] = []
                items = container.query_items(
                    query=query,
                    parameters=parameters or [],
                    enable_cross_partition_query=self._enable_cross_partition,
                )

                async for item in items:
                    results.append(dict(item))
                    if max_results and len(results) >= max_results:
                        break

                self._logger.debug(f"{operation_name}: {len(results)} Ergebnisse")
                return results

        except Exception as e:
            self._logger.exception(ERROR_COSMOSDB_QUERY, e)
            raise wrap_exception(
                e,
                CosmosDBQueryError,
                query,
                f"Query-Fehler: {e}",
                details={"operation": operation_name, "parameters": parameters}
            )

    async def upsert_item(
        self,
        item: dict[str, Any],
        *,
        operation_name: str = "upsert_item",
    ) -> dict[str, Any]:
        """Fügt Item hinzu oder aktualisiert es.

        Args:
            item: Item-Daten
            operation_name: Name der Operation für Logging

        Returns:
            Upserted Item

        Raises:
            CosmosDBQueryError: Bei Upsert-Fehlern
        """
        try:
            async with self.get_container() as container:
                if not container:
                    raise CosmosDBConnectionError("Container nicht verfügbar")

                result = await container.upsert_item(item)
                self._logger.debug(f"{operation_name}: Item upserted")
                return dict(result)

        except Exception as e:
            self._logger.exception(f"Upsert-Fehler: {e}")
            raise wrap_exception(
                e,
                CosmosDBQueryError,
                "UPSERT",
                f"Upsert-Fehler: {e}",
                details={"operation": operation_name, "item_id": item.get("id")}
            )


# =============================================================================
# Voice-spezifische Cosmos DB Operations
# =============================================================================

class VoiceCosmosDBManager(CosmosDBManager):
    """Voice-spezifische Cosmos DB-Operations.

    Erweitert die Base-Klasse um Voice-spezifische Operationen
    mit vorkonfigurierten Queries und Validierungen.
    """

    async def load_default_configuration(self) -> dict[str, Any] | None:
        """Lädt Standard-Voice-Konfiguration aus Cosmos DB.

        Returns:
            Standard-Konfiguration oder None

        Raises:
            CosmosDBQueryError: Bei Query-Fehlern
        """
        return await self.query_scalar(
            COSMOS_QUERY_DEFAULT_CONFIG,
            operation_name="load_default_configuration"
        )

    async def load_configuration_by_id(self, config_id: str) -> dict[str, Any] | None:
        """Lädt Konfiguration anhand der ID.

        Args:
            config_id: Konfigurations-ID

        Returns:
            Konfiguration oder None

        Raises:
            CosmosDBQueryError: Bei Query-Fehlern
        """
        query = "SELECT * FROM c WHERE c.id = @config_id"
        parameters = [{"name": "@config_id", "value": config_id}]

        return await self.query_scalar(
            query,
            parameters,
            operation_name="load_configuration_by_id"
        )

    async def load_configurations_by_category(
        self,
        category: str,
        *,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Lädt alle Konfigurationen einer Kategorie.

        Args:
            category: Kategorie
            max_results: Maximale Anzahl Ergebnisse

        Returns:
            Liste der Konfigurationen

        Raises:
            CosmosDBQueryError: Bei Query-Fehlern
        """
        query = "SELECT * FROM c WHERE c.category = @category"
        parameters = [{"name": "@category", "value": category}]

        return await self.query_items(
            query,
            parameters,
            max_results=max_results,
            operation_name="load_configurations_by_category"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_voice_cosmos_manager(
    settings: CosmosSettings | None = None,
) -> VoiceCosmosDBManager:
    """Factory für Voice Cosmos DB Manager.

    Args:
        settings: Cosmos DB-Settings (default: globale settings)

    Returns:
        Konfigurierter VoiceCosmosDBManager
    """
    if settings is None:
        raise ValueError("Cosmos DB settings must be provided - no global settings available")

    return VoiceCosmosDBManager(settings)


# =============================================================================
# Legacy Compatibility
# =============================================================================

@asynccontextmanager
async def get_cosmos_container() -> AsyncIterator[ContainerProxy | None]:
    """Legacy-Kompatibilität für get_cosmos_container().

    Deprecated: Verwende VoiceCosmosDBManager.get_container() stattdessen.
    """
    manager = create_voice_cosmos_manager()
    async with manager.get_container() as container:
        yield container


async def _query_scalar(
    container: ContainerProxy,
    query: str,
    parameters: list[dict[str, Any]] | None = None,
) -> Any:
    """Legacy-Kompatibilität für _query_scalar().

    Deprecated: Verwende VoiceCosmosDBManager.query_scalar() stattdessen.
    """
    try:
        items = container.query_items(
            query=query,
            parameters=parameters or [],
            enable_cross_partition_query=COSMOS_ENABLE_CROSS_PARTITION,
        )
        async for item in items:
            return item
        return None
    except Exception as e:
        logger.exception(ERROR_COSMOSDB_QUERY, e)
        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    "CosmosDBManager",
    # Protocols
    "CosmosSettings",
    "VoiceCosmosDBManager",
    "_query_scalar",
    # Factory Functions
    "create_voice_cosmos_manager",
    # Legacy Compatibility
    "get_cosmos_container",
]
