"""Basis-Klasse für alle Discovery-Engines.

Konsolidiert gemeinsame Discovery-Patterns und eliminiert Code-Duplikation
zwischen tool_discovery.py, resource_discovery.py und prompt_discovery.py.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeVar

from kei_logging import get_logger
from observability import trace_function, trace_span

from ..core.constants import (
    DEFAULT_DISCOVERY_INTERVAL_SECONDS,
    DISCOVERY_CONFIG,
)
from ..core.exceptions import KEIMCPError
from ..core.utils import batch_items, create_cache_key

if TYPE_CHECKING:
    from ..kei_mcp_registry import RegisteredMCPServer

logger = get_logger(__name__)

# Generic Type für Discovery-Ergebnisse
T = TypeVar("T")


@dataclass
class DiscoveryMetrics:
    """Metriken für Discovery-Operationen."""

    total_servers: int = 0
    successful_discoveries: int = 0
    failed_discoveries: int = 0
    total_items_discovered: int = 0
    discovery_duration_seconds: float = 0.0
    last_discovery_time: datetime | None = None
    cache_hits: int = 0
    cache_misses: int = 0

    def reset(self) -> None:
        """Setzt Metriken zurück."""
        self.total_servers = 0
        self.successful_discoveries = 0
        self.failed_discoveries = 0
        self.total_items_discovered = 0
        self.discovery_duration_seconds = 0.0
        self.cache_hits = 0
        self.cache_misses = 0


@dataclass
class CachedDiscoveryResult:
    """Gecachtes Discovery-Ergebnis."""

    items: list[Any]
    cached_at: datetime
    server_name: str
    ttl_seconds: int = 300  # 5 Minuten default

    def is_expired(self) -> bool:
        """Prüft ob Cache-Eintrag abgelaufen ist."""
        return datetime.now(UTC) - self.cached_at > timedelta(seconds=self.ttl_seconds)


class BaseDiscoveryEngine[T](ABC):
    """Basis-Klasse für alle Discovery-Engines.

    Konsolidiert gemeinsame Discovery-Patterns:
    - Server-Iteration
    - Caching-Mechanismen
    - Error-Handling
    - Metriken-Sammlung
    - Batch-Verarbeitung
    """

    def __init__(
        self,
        discovery_interval_seconds: float = DEFAULT_DISCOVERY_INTERVAL_SECONDS,
        cache_ttl_seconds: int = 300,
        max_parallel_workers: int = DISCOVERY_CONFIG["PARALLEL_WORKERS"]
    ):
        """Initialisiert Discovery-Engine.

        Args:
            discovery_interval_seconds: Intervall zwischen Discovery-Läufen
            cache_ttl_seconds: Cache-TTL für Discovery-Ergebnisse
            max_parallel_workers: Maximale parallele Worker
        """
        self._discovery_interval_seconds = discovery_interval_seconds
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_parallel_workers = max_parallel_workers

        # Discovery-State
        self._discovered_items: dict[str, T] = {}
        self._cache: dict[str, CachedDiscoveryResult] = {}
        self._last_discovery_run: datetime | None = None
        self._metrics = DiscoveryMetrics()

        # Concurrency-Control
        self._discovery_semaphore = asyncio.Semaphore(max_parallel_workers)
        self._discovery_lock = asyncio.Lock()

        logger.info(f"{self.__class__.__name__} initialisiert")

    @abstractmethod
    async def _discover_server_items(
        self,
        server_name: str,
        server: RegisteredMCPServer
    ) -> list[T]:
        """Entdeckt Items eines spezifischen Servers.

        Args:
            server_name: Name des Servers
            server: Server-Instanz

        Returns:
            Liste der entdeckten Items
        """

    @abstractmethod
    def _create_item_id(self, item: T, server_name: str) -> str:
        """Erstellt eindeutige ID für Item.

        Args:
            item: Discovery-Item
            server_name: Server-Name

        Returns:
            Eindeutige Item-ID
        """

    @abstractmethod
    def _validate_item(self, item: T) -> bool:
        """Validiert entdecktes Item.

        Args:
            item: Zu validierendes Item

        Returns:
            True wenn Item gültig ist
        """

    @trace_function("mcp.discovery.discover_all")
    async def discover_all_items(
        self,
        servers: dict[str, RegisteredMCPServer],
        force_refresh: bool = False
    ) -> list[T]:
        """Entdeckt alle Items von registrierten Servern.

        Args:
            servers: Dictionary der registrierten Server
            force_refresh: Erzwingt Neuentdeckung

        Returns:
            Liste aller entdeckten Items
        """
        current_time = datetime.now(UTC)

        # Prüfe, ob Discovery erforderlich ist
        if (not force_refresh and
            self._last_discovery_run and
            (current_time - self._last_discovery_run).total_seconds() < self._discovery_interval_seconds):
            logger.debug("Discovery übersprungen - Intervall noch nicht erreicht")
            return list(self._discovered_items.values())

        async with self._discovery_lock:
            start_time = time.time()
            self._metrics.reset()
            self._metrics.total_servers = len(servers)

            logger.info(f"Starte Discovery für {len(servers)} Server")

            try:
                # Parallele Discovery für alle Server
                all_items = await self._discover_from_all_servers(servers, force_refresh)

                # Items validieren und cachen
                valid_items = await self._process_discovered_items(all_items)

                # Discovery-State aktualisieren
                self._update_discovery_state(valid_items)

                # Metriken finalisieren
                self._metrics.discovery_duration_seconds = time.time() - start_time
                self._metrics.last_discovery_time = current_time
                self._metrics.total_items_discovered = len(valid_items)
                self._last_discovery_run = current_time

                logger.info(
                    f"Discovery abgeschlossen: {len(valid_items)} Items entdeckt "
                    f"({self._metrics.successful_discoveries}/{self._metrics.total_servers} Server erfolgreich)"
                )

                return valid_items

            except Exception as e:
                logger.exception(f"Discovery fehlgeschlagen: {e}")
                raise KEIMCPError(f"Discovery-Fehler: {e}", cause=e)

    async def _discover_from_all_servers(
        self,
        servers: dict[str, RegisteredMCPServer],
        force_refresh: bool
    ) -> list[T]:
        """Führt Discovery für alle Server parallel durch."""
        all_items = []

        # Server in Batches aufteilen für bessere Kontrolle
        server_items = list(servers.items())
        batches = batch_items(server_items, DISCOVERY_CONFIG["BATCH_SIZE"])

        for batch in batches:
            # Parallele Discovery für Batch
            batch_tasks = [
                self._discover_from_single_server(server_name, server, force_refresh)
                for server_name, server in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Ergebnisse sammeln
            for result in batch_results:
                if isinstance(result, Exception):
                    self._metrics.failed_discoveries += 1
                    logger.warning(f"Server-Discovery fehlgeschlagen: {result}")
                else:
                    self._metrics.successful_discoveries += 1
                    all_items.extend(result)

        return all_items

    async def _discover_from_single_server(
        self,
        server_name: str,
        server: RegisteredMCPServer,
        force_refresh: bool
    ) -> list[T]:
        """Führt Discovery für einen einzelnen Server durch."""
        async with self._discovery_semaphore:
            # Cache prüfen
            if not force_refresh:
                cached_result = self._get_cached_result(server_name)
                if cached_result:
                    self._metrics.cache_hits += 1
                    return cached_result

            self._metrics.cache_misses += 1

            try:
                with trace_span("mcp.discovery.server", {"server": server_name}):
                    # Server-spezifische Discovery
                    items = await self._discover_server_items(server_name, server)

                    # Ergebnis cachen
                    self._cache_result(server_name, items)

                    logger.debug(f"Server {server_name}: {len(items)} Items entdeckt")
                    return items

            except Exception as e:
                logger.exception(f"Discovery für Server {server_name} fehlgeschlagen: {e}")
                # Leere Liste zurückgeben statt Exception zu propagieren
                return []

    def _get_cached_result(self, server_name: str) -> list[T] | None:
        """Holt gecachtes Discovery-Ergebnis."""
        cache_key = create_cache_key("discovery", self.__class__.__name__, server_name)
        cached = self._cache.get(cache_key)

        if cached and not cached.is_expired():
            return cached.items

        # Abgelaufenen Cache-Eintrag entfernen
        if cached:
            self._cache.pop(cache_key, None)

        return None

    def _cache_result(self, server_name: str, items: list[T]) -> None:
        """Cached Discovery-Ergebnis."""
        cache_key = create_cache_key("discovery", self.__class__.__name__, server_name)

        self._cache[cache_key] = CachedDiscoveryResult(
            items=items,
            cached_at=datetime.now(UTC),
            server_name=server_name,
            ttl_seconds=self._cache_ttl_seconds
        )

    async def _process_discovered_items(self, items: list[T]) -> list[T]:
        """Verarbeitet und validiert entdeckte Items."""
        valid_items = []

        for item in items:
            try:
                if self._validate_item(item):
                    valid_items.append(item)
                else:
                    logger.warning(f"Item-Validierung fehlgeschlagen: {item}")
            except Exception as e:
                logger.exception(f"Fehler bei Item-Validierung: {e}")

        return valid_items

    def _update_discovery_state(self, items: list[T]) -> None:
        """Aktualisiert Discovery-State mit neuen Items."""
        # Bestehende Items löschen
        self._discovered_items.clear()

        # Neue Items hinzufügen
        for item in items:
            try:
                item_id = self._create_item_id(item, getattr(item, "server_name", "unknown"))
                self._discovered_items[item_id] = item
            except Exception as e:
                logger.exception(f"Fehler beim Erstellen der Item-ID: {e}")

    def get_discovered_items(self) -> list[T]:
        """Gibt alle entdeckten Items zurück."""
        return list(self._discovered_items.values())

    def get_item_by_id(self, item_id: str) -> T | None:
        """Gibt Item anhand ID zurück."""
        return self._discovered_items.get(item_id)

    def get_discovery_metrics(self) -> DiscoveryMetrics:
        """Gibt Discovery-Metriken zurück."""
        return self._metrics

    def clear_cache(self) -> None:
        """Löscht Discovery-Cache."""
        self._cache.clear()
        logger.debug("Discovery-Cache geleert")

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        total_entries = len(self._cache)
        expired_entries = sum(1 for cached in self._cache.values() if cached.is_expired())

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "hit_rate": (
                self._metrics.cache_hits / (self._metrics.cache_hits + self._metrics.cache_misses)
                if (self._metrics.cache_hits + self._metrics.cache_misses) > 0 else 0.0
            )
        }
