"""Basis-Klasse für Cosmos DB Memory-Operationen."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from storage.cache.redis_cache import CachedCosmosClient, get_cached_cosmos_container

from .memory_constants import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:  # pragma: no cover - optional import
    from azure.cosmos.aio import ContainerProxy
except Exception:  # pragma: no cover - test/runtime ohne cosmos
    # Mock ContainerProxy für Tests/Runtime ohne Cosmos
    class ContainerProxy:  # type: ignore[no-redef]
        """Mock ContainerProxy für Tests ohne Azure Cosmos DB."""

        async def query_items(self, **_):
            """Mock query_items method."""
            return []

        async def upsert_item(self, **kwargs):
            """Mock upsert_item method."""

        async def delete_item(self, **kwargs):
            """Mock delete_item method."""

logger = get_logger(__name__)


class CosmosOperationError(Exception):
    """Fehler bei Cosmos DB Operationen."""

    def __init__(self, operation: str, original_error: Exception) -> None:
        """Initialisiert Cosmos Operation Error.

        Args:
            operation: Name der fehlgeschlagenen Operation
            original_error: Ursprünglicher Fehler
        """
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"Cosmos {operation} fehlgeschlagen: {original_error}")


class BaseCosmosMemory(ABC):
    """Abstrakte Basis-Klasse für Cosmos DB Memory-Implementierungen.

    Stellt gemeinsame Infrastruktur für Connection-Management, Query-Ausführung
    und Error-Handling bereit. Reduziert Code-Duplikation zwischen verschiedenen
    Memory-Implementierungen.
    """

    def __init__(self, container: ContainerProxy | None = None) -> None:
        """Initialisiert Basis-Memory.

        Args:
            container: Optionaler direkter Cosmos Container
        """
        self._container: ContainerProxy | None = container
        self._logger = get_logger(self.__class__.__name__)

    @asynccontextmanager
    async def _get_cosmos_container(self) -> AsyncIterator[CachedCosmosClient | None]:
        """Context Manager für Cosmos Container-Zugriff.

        Yields:
            CachedCosmosClient oder None bei Fehlern
        """
        try:
            if self._container is not None:
                # Direkter Container verfügbar
                cached_client = CachedCosmosClient(self._container)
                yield cached_client
                return

            # Container über Factory beziehen
            async with get_cached_cosmos_container() as cached:
                yield cached

        except (ConnectionError, TimeoutError) as e:
            self._logger.warning(f"Cosmos Container-Zugriff fehlgeschlagen - Verbindungsproblem: {e}")
            yield None
        except Exception as e:
            self._logger.warning(f"Cosmos Container-Zugriff fehlgeschlagen - Unerwarteter Fehler: {e}")
            yield None

    async def _execute_query(
        self,
        query: str,
        parameters: list[dict[str, Any]],
        operation_name: str,
        max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """Führt Cosmos DB Query sicher aus.

        Args:
            query: SQL Query String
            parameters: Query Parameter
            operation_name: Name der Operation für Logging
            max_results: Maximale Anzahl Ergebnisse

        Returns:
            Liste der Query-Ergebnisse

        Raises:
            CosmosOperationError: Bei Query-Fehlern
        """
        try:
            async with self._get_cosmos_container() as cached:
                if not cached:
                    return []

                container: ContainerProxy = cached.container  # type: ignore[attr-defined]
                results: list[dict[str, Any]] = []

                async for item in container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ):
                    results.append(dict(item))
                    if max_results and len(results) >= max_results:
                        break

                self._logger.debug(f"{operation_name}: {len(results)} Ergebnisse")
                return results

        except Exception as e:
            self._logger.warning(f"{operation_name} Query fehlgeschlagen: {e}")
            raise CosmosOperationError(operation_name, e)

    async def _upsert_item(
        self,
        item: dict[str, Any],
        partition_key: str,
        operation_name: str
    ) -> bool:
        """Führt Cosmos DB Upsert sicher aus.

        Args:
            item: Item zum Speichern
            partition_key: Partition Key
            operation_name: Name der Operation für Logging

        Returns:
            True bei Erfolg, False bei Fehlern
        """
        try:
            async with self._get_cosmos_container() as cached:
                if not cached:
                    return False

                container: ContainerProxy = cached.container  # type: ignore[attr-defined]
                await container.upsert_item(body=item, partition_key=partition_key)

                self._logger.debug(f"{operation_name}: Item gespeichert")
                return True

        except Exception as e:
            self._logger.warning(f"{operation_name} Upsert fehlgeschlagen: {e}")
            return False

    async def _delete_item(
        self,
        item_id: str,
        partition_key: str,
        operation_name: str
    ) -> bool:
        """Führt Cosmos DB Delete sicher aus.

        Args:
            item_id: ID des zu löschenden Items
            partition_key: Partition Key
            operation_name: Name der Operation für Logging

        Returns:
            True bei Erfolg, False bei Fehlern
        """
        try:
            async with self._get_cosmos_container() as cached:
                if not cached:
                    return False

                container: ContainerProxy = cached.container  # type: ignore[attr-defined]
                await container.delete_item(item=item_id, partition_key=partition_key)

                self._logger.debug(f"{operation_name}: Item gelöscht")
                return True

        except Exception as e:
            self._logger.debug(f"{operation_name} Delete fehlgeschlagen: {e}")
            return False

    async def _batch_delete_items(
        self,
        item_ids: list[str],
        partition_key: str,
        operation_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> int:
        """Führt Batch-Delete für mehrere Items aus.

        Args:
            item_ids: Liste der zu löschenden Item-IDs
            partition_key: Partition Key
            operation_name: Name der Operation für Logging
            batch_size: Batch-Größe für Verarbeitung

        Returns:
            Anzahl erfolgreich gelöschter Items
        """
        if not item_ids:
            return 0

        # Batch-Größe begrenzen
        effective_batch_size = min(batch_size, MAX_BATCH_SIZE)
        deleted_count = 0

        for i in range(0, len(item_ids), effective_batch_size):
            batch = item_ids[i:i + effective_batch_size]

            for item_id in batch:
                if await self._delete_item(item_id, partition_key, operation_name):
                    deleted_count += 1

        self._logger.debug(f"{operation_name}: {deleted_count}/{len(item_ids)} Items gelöscht")
        return deleted_count

    @staticmethod
    def _validate_input(**kwargs: str | int | float | None) -> None:
        """Validiert Eingabeparameter.

        Args:
            **kwargs: Parameter zur Validierung (str, int, float oder None)

        Raises:
            ValueError: Bei ungültigen Parametern
        """
        for key, value in kwargs.items():
            if value is None:
                raise ValueError(f"Parameter '{key}' darf nicht None sein")

            if isinstance(value, str) and not value.strip():
                raise ValueError(f"Parameter '{key}' darf nicht leer sein")

    @abstractmethod
    async def cleanup(self) -> None:
        """Führt Bereinigung alter/überschüssiger Daten durch.

        Muss von Subklassen implementiert werden.
        """


__all__ = [
    "BaseCosmosMemory",
    "CosmosOperationError",
]
