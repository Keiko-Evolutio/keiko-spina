# backend/agents/monitoring/metrics_collector.py
"""Metrics Collector für das Agent-Framework.

Metrics Collection mit:
- Custom Metrics
- Metric Aggregation
- Time-Series Data
- Export-Funktionalität
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..logging_utils import StructuredLogger

logger = StructuredLogger("metrics_collector")


class MetricAggregation(Enum):
    """Metric-Aggregations-Typen."""

    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"


@dataclass
class MetricsConfig:
    """Konfiguration für Metrics Collector."""

    # Collection
    enable_collection: bool = True
    collection_interval: float = 10.0

    # Storage
    max_data_points: int = 10000
    retention_hours: int = 24

    # Aggregation
    enable_aggregation: bool = True
    aggregation_intervals: list[int] = field(default_factory=lambda: [60, 300, 3600])


@dataclass
class MetricValue:
    """Einzelner Metric-Wert."""

    value: int | float
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class CustomMetric:
    """Custom Metric Definition."""

    name: str
    description: str
    unit: str = ""
    aggregation: MetricAggregation = MetricAggregation.AVERAGE
    tags: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Metrics Collector für das Agent-Framework."""

    def __init__(self, config: MetricsConfig):
        """Initialisiert Metrics Collector.

        Args:
            config: Metrics-Konfiguration
        """
        self.config = config

        # Metric-Speicher
        self._metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.max_data_points)
        )
        self._custom_metrics: dict[str, CustomMetric] = {}

        # Aggregierte Daten
        self._aggregated_metrics: dict[str, dict[int, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # Legacy-Kompatibilität
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, dict[str, Any]] = {}
        self._summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Metadaten
        self._metric_help: dict[str, str] = {}
        self._default_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

        logger.info("Metrics Collector initialisiert")

    def register_custom_metric(self, metric: CustomMetric) -> None:
        """Registriert Custom Metric.

        Args:
            metric: Custom Metric Definition
        """
        self._custom_metrics[metric.name] = metric
        logger.info(f"Custom Metric registriert: {metric.name}")

    def record_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None
    ) -> None:
        """Zeichnet Metric-Wert auf.

        Args:
            name: Metric-Name
            value: Metric-Wert
            tags: Zusätzliche Tags
        """
        if not self.config.enable_collection:
            return

        metric_value = MetricValue(
            value=value,
            tags=tags or {}
        )

        self._metrics[name].append(metric_value)

        logger.debug(f"Metric aufgezeichnet: {name}={value}")

    def get_metric_values(
        self,
        name: str,
        start_time: float | None = None,
        end_time: float | None = None
    ) -> list[MetricValue]:
        """Gibt Metric-Werte zurück.

        Args:
            name: Metric-Name
            start_time: Start-Zeit
            end_time: End-Zeit

        Returns:
            Liste von Metric-Werten
        """
        if name not in self._metrics:
            return []

        values = list(self._metrics[name])

        # Zeitfilter anwenden
        if start_time or end_time:
            filtered_values = []
            for value in values:
                if start_time and value.timestamp < start_time:
                    continue
                if end_time and value.timestamp > end_time:
                    continue
                filtered_values.append(value)
            values = filtered_values

        return values

    def get_metric_statistics(
        self,
        name: str,
        aggregation: MetricAggregation = MetricAggregation.AVERAGE,
        time_window: float | None = None
    ) -> dict[str, Any]:
        """Gibt Metric-Statistiken zurück.

        Args:
            name: Metric-Name
            aggregation: Aggregations-Typ
            time_window: Zeitfenster in Sekunden

        Returns:
            Statistiken-Dictionary
        """
        values = self.get_metric_values(name)

        # Zeitfenster-Filter
        if time_window:
            cutoff_time = time.time() - time_window
            values = [v for v in values if v.timestamp >= cutoff_time]

        if not values:
            return {}

        numeric_values = [v.value for v in values]

        stats = {
            "count": len(numeric_values),
            "first_timestamp": min(v.timestamp for v in values),
            "last_timestamp": max(v.timestamp for v in values)
        }

        if aggregation == MetricAggregation.SUM:
            stats["value"] = sum(numeric_values)
        elif aggregation == MetricAggregation.AVERAGE:
            stats["value"] = sum(numeric_values) / len(numeric_values)
        elif aggregation == MetricAggregation.MIN:
            stats["value"] = min(numeric_values)
        elif aggregation == MetricAggregation.MAX:
            stats["value"] = max(numeric_values)
        elif aggregation == MetricAggregation.COUNT:
            stats["value"] = len(numeric_values)
        elif aggregation == MetricAggregation.RATE:
            if len(values) > 1:
                time_span = values[-1].timestamp - values[0].timestamp
                stats["value"] = len(numeric_values) / time_span if time_span > 0 else 0
            else:
                stats["value"] = 0

        return stats

    def get_all_metrics(self) -> dict[str, Any]:
        """Gibt alle Metrics zurück."""
        result = {}

        for name in self._metrics.keys():
            result[name] = {
                "count": len(self._metrics[name]),
                "latest_value": self._metrics[name][-1].value if self._metrics[name] else None,
                "latest_timestamp": self._metrics[name][-1].timestamp if self._metrics[name] else None
            }

        return result

    def cleanup_old_metrics(self) -> int:
        """Bereinigt alte Metrics.

        Returns:
            Anzahl der bereinigten Datenpunkte
        """
        if not self.config.retention_hours:
            return 0

        cutoff_time = time.time() - (self.config.retention_hours * 3600)
        cleaned_count = 0

        for name, values in self._metrics.items():
            # Alte Werte entfernen
            while values and values[0].timestamp < cutoff_time:
                values.popleft()
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"{cleaned_count} alte Metric-Datenpunkte bereinigt")

        return cleaned_count

    def increment_counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Inkrementiert Counter-Metrik.

        Args:
            name: Metrik-Name
            value: Inkrement-Wert
            labels: Labels (werden zu Tags konvertiert)
        """
        metric_key = self._create_metric_key(name, labels or {})
        self._counters[metric_key] += value

        # Als Metrik aufzeichnen
        self.record_metric(name, self._counters[metric_key], labels or {})

        logger.debug(f"Counter inkrementiert: {name} += {value}")

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Setzt Gauge-Wert.

        Args:
            name: Metrik-Name
            value: Gauge-Wert
            labels: Labels (werden zu Tags konvertiert)
        """
        metric_key = self._create_metric_key(name, labels or {})
        self._gauges[metric_key] = value

        # Als Metrik aufzeichnen
        self.record_metric(name, value, labels or {})

        logger.debug(f"Gauge gesetzt: {name} = {value}")

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None, buckets: list[float] | None = None) -> None:
        """Fügt Wert zu Histogram hinzu.

        Args:
            name: Metrik-Name
            value: Beobachteter Wert
            labels: Labels
            buckets: Histogram-Buckets
        """
        metric_key = self._create_metric_key(name, labels or {})
        buckets = buckets or self._default_buckets

        if metric_key not in self._histograms:
            self._histograms[metric_key] = {
                "buckets": {str(bucket): 0 for bucket in buckets},
                "count": 0,
                "sum": 0.0
            }

        histogram = self._histograms[metric_key]

        # Wert zu Buckets hinzufügen
        for bucket in buckets:
            if value <= bucket:
                histogram["buckets"][str(bucket)] += 1

        histogram["count"] += 1
        histogram["sum"] += value

        # Auch als Enterprise-Metrik aufzeichnen
        self.record_metric(name, value, labels or {})

        logger.debug(f"Histogram beobachtet: {name} = {value}")

    def record_summary(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Zeichnet Summary-Wert auf.

        Args:
            name: Metrik-Name
            value: Summary-Wert
            labels: Labels
        """
        metric_key = self._create_metric_key(name, labels or {})

        self._summaries[metric_key].append({
            "value": value,
            "timestamp": time.time(),
            "labels": labels or {}
        })

        # Auch als Enterprise-Metrik aufzeichnen
        self.record_metric(name, value, labels or {})

        logger.debug(f"Summary aufgezeichnet: {name} = {value}")

    @staticmethod
    def _create_metric_key(name: str, labels: dict[str, str]) -> str:
        """Erstellt Metrik-Schlüssel aus Name und Labels.

        Args:
            name: Metrik-Name
            labels: Labels

        Returns:
            Metrik-Schlüssel
        """
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_legacy_metrics(self) -> dict[str, Any]:
        """Gibt Legacy-Metriken zurück.

        Returns:
            Legacy-Metriken (Counters, Gauges, Histograms, Summaries)
        """
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": dict(self._histograms),
            "summaries": {k: list(v) for k, v in self._summaries.items()}
        }
