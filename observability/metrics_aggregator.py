# backend/observability/metrics_aggregator.py
"""Metriken-Sammlung und -Aggregation für Keiko Personal Assistant

Implementiert Real-time-Metriken-Sammlung, Time-Series-Datensammlung,
Metriken-Aggregation über verschiedene Zeitfenster und Thread-safe Updates.
"""

from __future__ import annotations

import asyncio
import contextlib
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class AggregationWindow(str, Enum):
    """Aggregations-Zeitfenster."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    SIX_HOURS = "6h"
    TWENTY_FOUR_HOURS = "24h"
    ONE_WEEK = "1w"


class AggregationType(str, Enum):
    """Aggregations-Typen."""
    SUM = "sum"
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


@dataclass
class MetricDataPoint:
    """Einzelner Metriken-Datenpunkt."""
    timestamp: float
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags
        }


@dataclass
class AggregatedMetric:
    """Aggregierte Metrik."""
    metric_name: str
    window: AggregationWindow
    aggregation_type: AggregationType
    value: int | float
    timestamp: float
    tags: dict[str, str] = field(default_factory=dict)
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "metric_name": self.metric_name,
            "window": self.window.value,
            "aggregation_type": self.aggregation_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "sample_count": self.sample_count
        }


@dataclass
class MetricsConfig:
    """Konfiguration für Metriken-Aggregation."""
    # Retention-Konfiguration
    max_data_points_per_metric: int = 10000
    retention_hours: int = 168  # 1 Woche

    # Aggregations-Konfiguration
    enable_real_time_aggregation: bool = True
    aggregation_interval_seconds: int = 60

    # Performance-Konfiguration
    max_concurrent_aggregations: int = 10
    batch_size: int = 1000
    enable_compression: bool = True

    # Sampling-Konfiguration
    enable_sampling: bool = False
    sampling_rate: float = 1.0  # 100% = keine Sampling
    high_volume_threshold: int = 1000  # Events/Minute

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "max_data_points_per_metric": self.max_data_points_per_metric,
            "retention_hours": self.retention_hours,
            "enable_real_time_aggregation": self.enable_real_time_aggregation,
            "aggregation_interval_seconds": self.aggregation_interval_seconds,
            "max_concurrent_aggregations": self.max_concurrent_aggregations,
            "batch_size": self.batch_size,
            "enable_compression": self.enable_compression,
            "enable_sampling": self.enable_sampling,
            "sampling_rate": self.sampling_rate,
            "high_volume_threshold": self.high_volume_threshold
        }


class TimeSeriesBuffer:
    """Thread-safe Time-Series-Buffer für Metriken-Datenpunkte."""

    def __init__(self, max_size: int = 10000):
        """Initialisiert Time-Series-Buffer.

        Args:
            max_size: Maximale Anzahl Datenpunkte
        """
        self.max_size = max_size
        self._data_points: deque[MetricDataPoint] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._total_points_added = 0

    def add_data_point(self, data_point: MetricDataPoint) -> None:
        """Fügt Datenpunkt hinzu.

        Args:
            data_point: Metriken-Datenpunkt
        """
        with self._lock:
            self._data_points.append(data_point)
            self._total_points_added += 1

    def get_data_points_in_window(
        self,
        start_time: float,
        end_time: float
    ) -> list[MetricDataPoint]:
        """Holt Datenpunkte in Zeitfenster.

        Args:
            start_time: Start-Timestamp
            end_time: End-Timestamp

        Returns:
            Liste von Datenpunkten
        """
        with self._lock:
            return [
                dp for dp in self._data_points
                if start_time <= dp.timestamp <= end_time
            ]

    def get_latest_data_points(self, count: int) -> list[MetricDataPoint]:
        """Holt neueste Datenpunkte.

        Args:
            count: Anzahl Datenpunkte

        Returns:
            Liste der neuesten Datenpunkte
        """
        with self._lock:
            return list(self._data_points)[-count:]

    def cleanup_old_data(self, cutoff_time: float) -> int:
        """Bereinigt alte Datenpunkte.

        Args:
            cutoff_time: Cutoff-Timestamp

        Returns:
            Anzahl entfernter Datenpunkte
        """
        with self._lock:
            original_size = len(self._data_points)

            # Entferne alte Datenpunkte
            while self._data_points and self._data_points[0].timestamp < cutoff_time:
                self._data_points.popleft()

            return original_size - len(self._data_points)

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Buffer-Statistiken zurück.

        Returns:
            Statistiken-Dictionary
        """
        with self._lock:
            return {
                "current_size": len(self._data_points),
                "max_size": self.max_size,
                "total_points_added": self._total_points_added,
                "oldest_timestamp": self._data_points[0].timestamp if self._data_points else None,
                "newest_timestamp": self._data_points[-1].timestamp if self._data_points else None
            }


class MetricsAggregator:
    """Metriken-Aggregator für verschiedene Zeitfenster."""

    def __init__(self, config: MetricsConfig | None = None):
        """Initialisiert Metrics Aggregator.

        Args:
            config: Metriken-Konfiguration
        """
        self.config = config or MetricsConfig()

        # Time-Series-Buffers pro Metrik
        self._metric_buffers: dict[str, TimeSeriesBuffer] = {}
        self._buffers_lock = threading.RLock()

        # Aggregierte Metriken-Cache
        self._aggregated_metrics: dict[str, dict[AggregationWindow, dict[AggregationType, AggregatedMetric]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._aggregated_lock = threading.RLock()

        # Background-Tasks
        self._aggregation_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        # Sampling-State
        self._sampling_counters: dict[str, int] = defaultdict(int)
        self._sampling_lock = threading.RLock()

        # Statistiken
        self._metrics_collected = 0
        self._metrics_sampled = 0
        self._aggregations_performed = 0
        self._cleanup_operations = 0

    async def start(self) -> None:
        """Startet Metrics Aggregator."""
        if self._is_running:
            return

        self._is_running = True

        if self.config.enable_real_time_aggregation:
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Metrics Aggregator gestartet")

    async def stop(self) -> None:
        """Stoppt Metrics Aggregator."""
        self._is_running = False

        if self._aggregation_task:
            self._aggregation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._aggregation_task

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        logger.info("Metrics Aggregator gestoppt")

    @trace_function("metrics_aggregator.collect_metric")
    def collect_metric(
        self,
        metric_identifier: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        timestamp: float | None = None
    ) -> bool:
        """Sammelt einzelne Metrik.

        Args:
            metric_identifier: Name der Metrik
            value: Wert der Metrik
            tags: Tags für die Metrik
            timestamp: Timestamp (default: aktuell)

        Returns:
            True wenn Metrik gesammelt wurde
        """
        # Sampling-Check
        if not self._should_sample_metric(metric_identifier):
            self._metrics_sampled += 1
            return False

        # Erstelle Datenpunkt
        data_point = MetricDataPoint(
            timestamp=timestamp or time.time(),
            value=value,
            tags=tags or {}
        )

        # Füge zu Buffer hinzu
        with self._buffers_lock:
            if metric_identifier not in self._metric_buffers:
                self._metric_buffers[metric_identifier] = TimeSeriesBuffer(
                    max_size=self.config.max_data_points_per_metric
                )

            self._metric_buffers[metric_identifier].add_data_point(data_point)

        self._metrics_collected += 1

        # Performance-Check: Minimaler Overhead
        if self._metrics_collected % 1000 == 0:
            logger.debug(f"Metrics gesammelt: {self._metrics_collected}")

        return True

    def _should_sample_metric(self, metric_name: str) -> bool:
        """Prüft, ob Metrik gesampelt werden soll.

        Args:
            metric_name: Name der Metrik

        Returns:
            True wenn Metrik gesampelt werden soll
        """
        if not self.config.enable_sampling:
            return True

        with self._sampling_lock:
            self._sampling_counters[metric_name] += 1

            # High-Volume-Check
            if self._sampling_counters[metric_name] > self.config.high_volume_threshold:
                # Verwende Sampling-Rate
                import random
                return random.random() < self.config.sampling_rate

            return True

    @trace_function("metrics_aggregator.aggregate_metrics")
    async def aggregate_metrics(
        self,
        metric_name: str,
        window: AggregationWindow,
        aggregation_types: list[AggregationType]
    ) -> dict[AggregationType, AggregatedMetric]:
        """Aggregiert Metriken für spezifisches Zeitfenster.

        Args:
            metric_name: Name der Metrik
            window: Aggregations-Zeitfenster
            aggregation_types: Liste der Aggregations-Typen

        Returns:
            Dictionary mit aggregierten Metriken
        """
        # Berechne Zeitfenster
        end_time = time.time()
        window_seconds = self._get_window_seconds(window)
        start_time = end_time - window_seconds

        # Hole Datenpunkte
        with self._buffers_lock:
            if metric_name not in self._metric_buffers:
                return {}

            data_points = self._metric_buffers[metric_name].get_data_points_in_window(
                start_time, end_time
            )

        if not data_points:
            return {}

        # Führe Aggregationen durch
        results = {}
        values = [dp.value for dp in data_points]

        for agg_type in aggregation_types:
            aggregated_value = self._calculate_aggregation(values, agg_type, window_seconds)

            aggregated_metric = AggregatedMetric(
                metric_name=metric_name,
                window=window,
                aggregation_type=agg_type,
                value=aggregated_value,
                timestamp=end_time,
                sample_count=len(data_points)
            )

            results[agg_type] = aggregated_metric

        # Cache Ergebnisse
        with self._aggregated_lock:
            for agg_type, metric in results.items():
                self._aggregated_metrics[metric_name][window][agg_type] = metric

        self._aggregations_performed += 1

        return results

    def _calculate_aggregation(
        self,
        values: list[int | float],
        agg_type: AggregationType,
        window_seconds: float
    ) -> int | float:
        """Berechnet Aggregation für Werte.

        Args:
            values: Liste der Werte
            agg_type: Aggregations-Typ
            window_seconds: Zeitfenster in Sekunden

        Returns:
            Aggregierter Wert
        """
        if not values:
            return 0

        if agg_type == AggregationType.SUM:
            return sum(values)
        if agg_type == AggregationType.AVERAGE:
            return statistics.mean(values)
        if agg_type == AggregationType.MIN:
            return min(values)
        if agg_type == AggregationType.MAX:
            return max(values)
        if agg_type == AggregationType.COUNT:
            return len(values)
        if agg_type == AggregationType.RATE:
            return len(values) / window_seconds
        if agg_type == AggregationType.PERCENTILE_50:
            return statistics.median(values)
        if agg_type == AggregationType.PERCENTILE_95:
            return self._calculate_percentile(values, 95)
        if agg_type == AggregationType.PERCENTILE_99:
            return self._calculate_percentile(values, 99)
        return 0

    def _calculate_percentile(self, values: list[int | float], percentile: float) -> float:
        """Berechnet Perzentil.

        Args:
            values: Liste der Werte
            percentile: Perzentil (0-100)

        Returns:
            Perzentil-Wert
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return float(sorted_values[int(index)])
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        weight = index - lower_index

        return float(sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight)

    def _get_window_seconds(self, window: AggregationWindow) -> float:
        """Konvertiert Zeitfenster zu Sekunden.

        Args:
            window: Aggregations-Zeitfenster

        Returns:
            Zeitfenster in Sekunden
        """
        window_mapping = {
            AggregationWindow.ONE_MINUTE: 60,
            AggregationWindow.FIVE_MINUTES: 300,
            AggregationWindow.FIFTEEN_MINUTES: 900,
            AggregationWindow.ONE_HOUR: 3600,
            AggregationWindow.SIX_HOURS: 21600,
            AggregationWindow.TWENTY_FOUR_HOURS: 86400,
            AggregationWindow.ONE_WEEK: 604800
        }

        return float(window_mapping.get(window, 60))

    async def get_aggregated_metrics(
        self,
        metric_name: str | None = None,
        window: AggregationWindow | None = None,
        aggregation_type: AggregationType | None = None
    ) -> dict[str, Any]:
        """Holt aggregierte Metriken.

        Args:
            metric_name: Name der Metrik (optional)
            window: Zeitfenster (optional)
            aggregation_type: Aggregations-Typ (optional)

        Returns:
            Dictionary mit aggregierten Metriken
        """
        with self._aggregated_lock:
            results = {}

            for m_name, windows in self._aggregated_metrics.items():
                if metric_name and m_name != metric_name:
                    continue

                results[m_name] = {}

                for w, aggregations in windows.items():
                    if window and w != window:
                        continue

                    results[m_name][w.value] = {}

                    for agg_type, metric in aggregations.items():
                        if aggregation_type and agg_type != aggregation_type:
                            continue

                        results[m_name][w.value][agg_type.value] = metric.to_dict()

            return results

    async def get_real_time_metrics(
        self,
        metric_name: str,
        last_n_points: int = 100
    ) -> list[dict[str, Any]]:
        """Holt Real-time-Metriken.

        Args:
            metric_name: Name der Metrik
            last_n_points: Anzahl der letzten Datenpunkte

        Returns:
            Liste der Datenpunkte
        """
        with self._buffers_lock:
            if metric_name not in self._metric_buffers:
                return []

            data_points = self._metric_buffers[metric_name].get_latest_data_points(last_n_points)
            return [dp.to_dict() for dp in data_points]

    async def _aggregation_loop(self) -> None:
        """Background-Loop für Aggregation."""
        while self._is_running:
            try:
                await self._perform_scheduled_aggregations()
                await asyncio.sleep(self.config.aggregation_interval_seconds)
            except Exception as e:
                logger.exception(f"Aggregation-Loop-Fehler: {e}")
                await asyncio.sleep(60)

    async def _perform_scheduled_aggregations(self) -> None:
        """Führt geplante Aggregationen durch."""
        # Standard-Aggregationen für alle Metriken
        standard_windows = [
            AggregationWindow.ONE_MINUTE,
            AggregationWindow.FIVE_MINUTES,
            AggregationWindow.ONE_HOUR
        ]

        standard_aggregations = [
            AggregationType.AVERAGE,
            AggregationType.COUNT,
            AggregationType.RATE,
            AggregationType.PERCENTILE_95
        ]

        with self._buffers_lock:
            metric_names = list(self._metric_buffers.keys())

        # Begrenze Concurrent-Aggregationen
        semaphore = asyncio.Semaphore(self.config.max_concurrent_aggregations)

        async def aggregate_metric_window(metric_name: str, time_window: AggregationWindow):
            async with semaphore:
                await self.aggregate_metrics(metric_name, time_window, standard_aggregations)

        # Führe Aggregationen parallel durch
        tasks = []
        for metric_name in metric_names:
            for window in standard_windows:
                task = asyncio.create_task(aggregate_metric_window(metric_name, window))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cleanup."""
        while self._is_running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(3600)  # 1 Stunde
            except Exception as e:
                logger.exception(f"Cleanup-Loop-Fehler: {e}")
                await asyncio.sleep(3600)

    async def _perform_cleanup(self) -> None:
        """Führt Cleanup durch."""
        cutoff_time = time.time() - (self.config.retention_hours * 3600)
        total_removed = 0

        with self._buffers_lock:
            for metric_name, buffer in self._metric_buffers.items():
                removed = buffer.cleanup_old_data(cutoff_time)
                total_removed += removed

        # Cleanup aggregierte Metriken
        with self._aggregated_lock:
            for metric_name in list(self._aggregated_metrics.keys()):
                for window in list(self._aggregated_metrics[metric_name].keys()):
                    for agg_type in list(self._aggregated_metrics[metric_name][window].keys()):
                        metric = self._aggregated_metrics[metric_name][window][agg_type]
                        if metric.timestamp < cutoff_time:
                            del self._aggregated_metrics[metric_name][window][agg_type]

        self._cleanup_operations += 1

        if total_removed > 0:
            logger.info(f"Cleanup: {total_removed} alte Datenpunkte entfernt")

    def get_aggregator_statistics(self) -> dict[str, Any]:
        """Gibt Aggregator-Statistiken zurück.

        Returns:
            Statistiken-Dictionary
        """
        with self._buffers_lock:
            buffer_stats = {
                name: buffer.get_statistics()
                for name, buffer in self._metric_buffers.items()
            }

        with self._aggregated_lock:
            aggregated_count = sum(
                len(windows) for windows in self._aggregated_metrics.values()
            )

        return {
            "config": self.config.to_dict(),
            "is_running": self._is_running,
            "metrics_collected": self._metrics_collected,
            "metrics_sampled": self._metrics_sampled,
            "aggregations_performed": self._aggregations_performed,
            "cleanup_operations": self._cleanup_operations,
            "active_metric_buffers": len(self._metric_buffers),
            "aggregated_metrics_count": aggregated_count,
            "buffer_statistics": buffer_stats
        }


# Globale Metrics Aggregator Instanz
metrics_aggregator = MetricsAggregator()
