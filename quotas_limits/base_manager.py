# backend/quotas_limits/base_manager.py
"""Base-Manager-Klasse für das Quotas/Limits System.

Gemeinsame Funktionalitäten für alle Manager-Klassen zur Reduzierung
von Code-Duplikation und Verbesserung der Konsistenz.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .constants import CACHE_TTL_SECONDS, MAX_CACHE_ENTRIES, MONITORING_INTERVAL_SECONDS
from .utils import (
    SimpleCache,
    create_error_context,
    generate_uuid,
    get_current_timestamp,
    measure_execution_time,
)

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)


@dataclass
class ManagerStats:
    """Statistiken für Manager-Klassen."""
    operations_total: int = 0
    operations_successful: int = 0
    operations_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_latency_ms: float = 0.0
    last_operation_timestamp: datetime | None = None

    def get_success_rate(self) -> float:
        """Berechnet Erfolgsrate.

        Returns:
            float: Erfolgsrate (0-1)
        """
        if self.operations_total == 0:
            return 0.0
        return self.operations_successful / self.operations_total

    def get_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate.

        Returns:
            float: Cache-Hit-Rate (0-1)
        """
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations == 0:
            return 0.0
        return self.cache_hits / total_cache_operations


@dataclass
class ManagerConfig:
    """Konfiguration für Manager-Klassen."""
    cache_enabled: bool = True
    cache_ttl_seconds: int = CACHE_TTL_SECONDS
    cache_max_size: int = MAX_CACHE_ENTRIES
    monitoring_enabled: bool = True
    monitoring_interval_seconds: int = MONITORING_INTERVAL_SECONDS
    auto_cleanup_enabled: bool = True
    cleanup_interval_seconds: int = 3600
    performance_tracking_enabled: bool = True
    error_tracking_enabled: bool = True


class BaseManager(ABC):
    """Basis-Klasse für alle Manager-Komponenten."""

    def __init__(self, config: ManagerConfig | None = None):
        """Initialisiert Base Manager.

        Args:
            config: Manager-Konfiguration
        """
        self.config = config or ManagerConfig()
        self.manager_id = generate_uuid()
        self.created_at = get_current_timestamp()

        # Statistiken
        self.stats = ManagerStats()

        # Cache
        if self.config.cache_enabled:
            self.cache = SimpleCache(max_size=self.config.cache_max_size)
        else:
            self.cache = None

        # Locks für Thread-Safety
        self._operation_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()

        # Background-Tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        # Error-Tracking
        self._recent_errors: list[dict[str, Any]] = []
        self._max_error_history = 100

        logger.info(f"{self.__class__.__name__} initialisiert: {self.manager_id}")

    @abstractmethod
    def get_manager_type(self) -> str:
        """Gibt Manager-Typ zurück.

        Returns:
            str: Manager-Typ
        """

    async def start(self) -> None:
        """Startet Manager und Background-Tasks."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Monitoring
        if self.config.monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Starte Cleanup
        if self.config.auto_cleanup_enabled:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(f"{self.get_manager_type()} Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Manager und Background-Tasks."""
        if not self._is_running:
            return

        self._is_running = False

        # Stoppe Background-Tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        logger.info(f"{self.get_manager_type()} Manager gestoppt")

    @measure_execution_time
    async def execute_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        cache_key: str | None = None,
        **kwargs
    ) -> Any:
        """Führt Operation mit Statistik-Tracking aus.

        Args:
            operation_name: Name der Operation
            operation_func: Auszuführende Funktion
            *args: Funktions-Argumente
            cache_key: Optional Cache-Schlüssel
            **kwargs: Funktions-Keyword-Argumente

        Returns:
            Any: Ergebnis der Operation
        """
        async with self._operation_lock:
            start_time = get_current_timestamp()

            try:
                # Prüfe Cache
                if cache_key and self.cache:
                    cached_result = await self.cache.get(cache_key)
                    if cached_result is not None:
                        await self._update_stats(True, True, start_time)
                        return cached_result
                    await self._update_stats(False, True, start_time)

                # Führe Operation aus
                if asyncio.iscoroutinefunction(operation_func):
                    result = await operation_func(*args, **kwargs)
                else:
                    result = operation_func(*args, **kwargs)

                # Cache Ergebnis
                if cache_key and self.cache and result is not None:
                    await self.cache.set(
                        cache_key,
                        result,
                        self.config.cache_ttl_seconds
                    )

                await self._update_stats(True, False, start_time)
                return result

            except Exception as e:
                await self._update_stats(False, False, start_time)
                await self._track_error(operation_name, e, args, kwargs)
                raise

    async def _update_stats(
        self,
        success: bool,
        cache_hit: bool,
        start_time: datetime
    ) -> None:
        """Aktualisiert Statistiken.

        Args:
            success: Operation erfolgreich
            cache_hit: Cache-Hit aufgetreten
            start_time: Start-Zeitpunkt
        """
        async with self._stats_lock:
            self.stats.operations_total += 1

            if success:
                self.stats.operations_successful += 1
            else:
                self.stats.operations_failed += 1

            if cache_hit:
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1

            # Berechne Latenz
            latency_ms = (get_current_timestamp() - start_time).total_seconds() * 1000

            # Aktualisiere durchschnittliche Latenz (exponential moving average)
            if self.stats.average_latency_ms == 0:
                self.stats.average_latency_ms = latency_ms
            else:
                alpha = 0.1  # Smoothing-Faktor
                self.stats.average_latency_ms = (
                    alpha * latency_ms +
                    (1 - alpha) * self.stats.average_latency_ms
                )

            self.stats.last_operation_timestamp = get_current_timestamp()

    async def _track_error(
        self,
        operation_name: str,
        exception: Exception,
        args: tuple,
        kwargs: dict
    ) -> None:
        """Trackt Fehler für Debugging.

        Args:
            operation_name: Name der Operation
            exception: Aufgetretene Exception
            args: Funktions-Argumente
            kwargs: Funktions-Keyword-Argumente
        """
        if not self.config.error_tracking_enabled:
            return

        error_context = create_error_context(
            operation=operation_name,
            details={
                "manager_type": self.get_manager_type(),
                "manager_id": self.manager_id,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            },
            exception=exception
        )

        self._recent_errors.append(error_context)

        # Begrenze Error-History
        if len(self._recent_errors) > self._max_error_history:
            self._recent_errors = self._recent_errors[-self._max_error_history:]

        logger.error(f"Operation {operation_name} fehlgeschlagen: {exception}")

    async def _monitoring_loop(self) -> None:
        """Monitoring-Loop für Performance-Tracking."""
        while self._is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.monitoring_interval_seconds)
            except Exception as e:
                logger.exception(f"Monitoring-Loop-Fehler: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)

    async def _cleanup_loop(self) -> None:
        """Cleanup-Loop für Wartungsaufgaben."""
        while self._is_running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config.cleanup_interval_seconds)
            except Exception as e:
                logger.exception(f"Cleanup-Loop-Fehler: {e}")
                await asyncio.sleep(self.config.cleanup_interval_seconds)

    async def _collect_metrics(self) -> None:
        """Sammelt Performance-Metriken."""
        if not self.config.performance_tracking_enabled:
            return

        # Subklassen können diese Methode überschreiben
        logger.debug(f"{self.get_manager_type()} Metriken gesammelt")

    async def _perform_cleanup(self) -> None:
        """Führt Wartungsaufgaben aus."""
        # Cache-Cleanup
        if self.cache:
            await self.cache._cleanup_expired()

        # Error-History-Cleanup
        if len(self._recent_errors) > self._max_error_history // 2:
            self._recent_errors = self._recent_errors[-self._max_error_history // 2:]

        logger.debug(f"{self.get_manager_type()} Cleanup durchgeführt")

    def get_status(self) -> dict[str, Any]:
        """Gibt Manager-Status zurück.

        Returns:
            Dict[str, Any]: Manager-Status
        """
        return {
            "manager_id": self.manager_id,
            "manager_type": self.get_manager_type(),
            "is_running": self._is_running,
            "created_at": self.created_at.isoformat(),
            "stats": {
                "operations_total": self.stats.operations_total,
                "success_rate": self.stats.get_success_rate(),
                "cache_hit_rate": self.stats.get_cache_hit_rate(),
                "average_latency_ms": self.stats.average_latency_ms,
                "last_operation": (
                    self.stats.last_operation_timestamp.isoformat()
                    if self.stats.last_operation_timestamp else None
                )
            },
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "recent_errors_count": len(self._recent_errors)
        }

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Gibt letzte Fehler zurück.

        Args:
            limit: Maximale Anzahl Fehler

        Returns:
            List[Dict[str, Any]]: Letzte Fehler
        """
        return self._recent_errors[-limit:] if self._recent_errors else []
