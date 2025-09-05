"""Metrics Collector Implementation.
Sammelt und verwaltet alle System-Metriken für Prometheus Export.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

from kei_logging import get_logger

from .interfaces import IMetricsCollector, MetricType, MetricValue

logger = get_logger(__name__)


@dataclass
class CounterMetric:
    """Counter-Metrik."""
    name: str
    value: float
    labels: dict[str, str]
    help_text: str
    created_at: datetime


@dataclass
class GaugeMetric:
    """Gauge-Metrik."""
    name: str
    value: float
    labels: dict[str, str]
    help_text: str
    updated_at: datetime


@dataclass
class HistogramBucket:
    """Histogram-Bucket."""
    le: float  # Less or equal
    count: int


@dataclass
class HistogramMetric:
    """Histogram-Metrik."""
    name: str
    buckets: list[HistogramBucket]
    count: int
    sum: float
    labels: dict[str, str]
    help_text: str
    updated_at: datetime


class MetricsCollector(IMetricsCollector):
    """Prometheus-kompatible Metrics Collector Implementation.
    Sammelt Counter, Gauge, Histogram und Summary Metriken.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Metrik-Storage
        self._counters: dict[str, CounterMetric] = {}
        self._gauges: dict[str, GaugeMetric] = {}
        self._histograms: dict[str, HistogramMetric] = {}
        self._summaries: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Standard Histogram-Buckets (Prometheus default)
        self._default_buckets = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")]

        # Metrik-Metadaten
        self._metric_help: dict[str, str] = {}

        # Performance-Tracking
        self._collection_start_time = time.time()

        logger.info("Metrics collector initialized")

    def increment_counter(self, name: str, value: float = 1.0, labels: dict[str, str] = None) -> None:
        """Erhöht einen Counter."""
        if labels is None:
            labels = {}

        metric_key = self._create_metric_key(name, labels)

        with self._lock:
            if metric_key in self._counters:
                self._counters[metric_key].value += value
            else:
                self._counters[metric_key] = CounterMetric(
                    name=name,
                    value=value,
                    labels=labels,
                    help_text=self._metric_help.get(name, f"Counter metric {name}"),
                    created_at=datetime.utcnow()
                )

        logger.debug(f"Incremented counter {name} by {value} with labels {labels}")

    def set_gauge(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Setzt einen Gauge-Wert."""
        if labels is None:
            labels = {}

        metric_key = self._create_metric_key(name, labels)

        with self._lock:
            self._gauges[metric_key] = GaugeMetric(
                name=name,
                value=value,
                labels=labels,
                help_text=self._metric_help.get(name, f"Gauge metric {name}"),
                updated_at=datetime.utcnow()
            )

        logger.debug(f"Set gauge {name} to {value} with labels {labels}")

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] = None, buckets: list[float] = None) -> None:
        """Fügt Wert zu Histogram hinzu."""
        if labels is None:
            labels = {}
        if buckets is None:
            buckets = self._default_buckets

        metric_key = self._create_metric_key(name, labels)

        with self._lock:
            if metric_key not in self._histograms:
                # Neue Histogram-Metrik erstellen
                histogram_buckets = [HistogramBucket(le=bucket, count=0) for bucket in buckets]
                self._histograms[metric_key] = HistogramMetric(
                    name=name,
                    buckets=histogram_buckets,
                    count=0,
                    sum=0.0,
                    labels=labels,
                    help_text=self._metric_help.get(name, f"Histogram metric {name}"),
                    updated_at=datetime.utcnow()
                )

            histogram = self._histograms[metric_key]

            # Wert zu Buckets hinzufügen
            for bucket in histogram.buckets:
                if value <= bucket.le:
                    bucket.count += 1

            # Gesamtstatistiken aktualisieren
            histogram.count += 1
            histogram.sum += value
            histogram.updated_at = datetime.utcnow()

        logger.debug(f"Observed histogram {name} value {value} with labels {labels}")

    def record_summary(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Zeichnet Summary-Wert auf."""
        if labels is None:
            labels = {}

        metric_key = self._create_metric_key(name, labels)

        with self._lock:
            self._summaries[metric_key].append({
                "value": value,
                "timestamp": time.time(),
                "labels": labels
            })

        logger.debug(f"Recorded summary {name} value {value} with labels {labels}")

    def get_metrics(self) -> list[MetricValue]:
        """Gibt alle gesammelten Metriken zurück."""
        metrics = []

        with self._lock:
            # Counter-Metriken
            for counter in self._counters.values():
                metrics.append(MetricValue(
                    name=counter.name,
                    value=counter.value,
                    metric_type=MetricType.COUNTER,
                    labels=counter.labels,
                    timestamp=counter.created_at,
                    help_text=counter.help_text
                ))

            # Gauge-Metriken
            for gauge in self._gauges.values():
                metrics.append(MetricValue(
                    name=gauge.name,
                    value=gauge.value,
                    metric_type=MetricType.GAUGE,
                    labels=gauge.labels,
                    timestamp=gauge.updated_at,
                    help_text=gauge.help_text
                ))

            # Histogram-Metriken
            for histogram in self._histograms.values():
                # Histogram-Buckets
                for bucket in histogram.buckets:
                    bucket_labels = histogram.labels.copy()
                    bucket_labels["le"] = str(bucket.le)

                    metrics.append(MetricValue(
                        name=f"{histogram.name}_bucket",
                        value=float(bucket.count),
                        metric_type=MetricType.HISTOGRAM,
                        labels=bucket_labels,
                        timestamp=histogram.updated_at,
                        help_text=f"{histogram.help_text} (bucket)"
                    ))

                # Histogram-Count
                metrics.append(MetricValue(
                    name=f"{histogram.name}_count",
                    value=float(histogram.count),
                    metric_type=MetricType.HISTOGRAM,
                    labels=histogram.labels,
                    timestamp=histogram.updated_at,
                    help_text=f"{histogram.help_text} (count)"
                ))

                # Histogram-Sum
                metrics.append(MetricValue(
                    name=f"{histogram.name}_sum",
                    value=histogram.sum,
                    metric_type=MetricType.HISTOGRAM,
                    labels=histogram.labels,
                    timestamp=histogram.updated_at,
                    help_text=f"{histogram.help_text} (sum)"
                ))

            # Summary-Metriken
            for metric_key, values in self._summaries.items():
                if not values:
                    continue

                # Berechne Quantile
                sorted_values = sorted([v["value"] for v in values])
                count = len(sorted_values)
                sum_values = sum(sorted_values)

                # Labels aus erstem Wert extrahieren
                labels = values[0]["labels"] if values else {}
                name = metric_key.split("|")[0]  # Name vor dem ersten |

                # Summary-Count
                metrics.append(MetricValue(
                    name=f"{name}_count",
                    value=float(count),
                    metric_type=MetricType.SUMMARY,
                    labels=labels,
                    timestamp=datetime.utcnow(),
                    help_text=f"Summary metric {name} (count)"
                ))

                # Summary-Sum
                metrics.append(MetricValue(
                    name=f"{name}_sum",
                    value=sum_values,
                    metric_type=MetricType.SUMMARY,
                    labels=labels,
                    timestamp=datetime.utcnow(),
                    help_text=f"Summary metric {name} (sum)"
                ))

                # Quantile
                for quantile in [0.5, 0.9, 0.95, 0.99]:
                    index = int(quantile * (count - 1))
                    quantile_value = sorted_values[index] if index < count else 0.0

                    quantile_labels = labels.copy()
                    quantile_labels["quantile"] = str(quantile)

                    metrics.append(MetricValue(
                        name=name,
                        value=quantile_value,
                        metric_type=MetricType.SUMMARY,
                        labels=quantile_labels,
                        timestamp=datetime.utcnow(),
                        help_text=f"Summary metric {name} (quantile)"
                    ))

        return metrics

    def register_metric_help(self, name: str, help_text: str) -> None:
        """Registriert Hilfe-Text für Metrik."""
        self._metric_help[name] = help_text

    def get_metric_count(self) -> dict[str, int]:
        """Gibt Anzahl der Metriken pro Typ zurück."""
        with self._lock:
            return {
                "counters": len(self._counters),
                "gauges": len(self._gauges),
                "histograms": len(self._histograms),
                "summaries": len(self._summaries)
            }

    def clear_metrics(self) -> None:
        """Löscht alle Metriken (für Tests)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()

        logger.info("All metrics cleared")

    def _create_metric_key(self, name: str, labels: dict[str, str]) -> str:
        """Erstellt eindeutigen Schlüssel für Metrik."""
        if not labels:
            return name

        # Sortiere Labels für konsistente Schlüssel
        sorted_labels = sorted(labels.items())
        label_string = "|".join([f"{k}={v}" for k, v in sorted_labels])

        return f"{name}|{label_string}"

    def export_prometheus_format(self) -> str:
        """Exportiert Metriken im Prometheus-Format."""
        metrics = self.get_metrics()
        lines = []

        # Gruppiere Metriken nach Namen für HELP und TYPE
        metric_groups = defaultdict(list)
        for metric in metrics:
            base_name = metric.name.split("_")[0] if "_" in metric.name else metric.name
            metric_groups[base_name].append(metric)

        for base_name, group_metrics in metric_groups.items():
            # HELP-Zeile
            help_text = group_metrics[0].help_text or f"Metric {base_name}"
            lines.append(f"# HELP {base_name} {help_text}")

            # TYPE-Zeile
            metric_type = group_metrics[0].metric_type.value
            lines.append(f"# TYPE {base_name} {metric_type}")

            # Metrik-Werte
            for metric in group_metrics:
                if metric.labels:
                    label_string = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                    lines.append(f"{metric.name}{{{label_string}}} {metric.value}")
                else:
                    lines.append(f"{metric.name} {metric.value}")

            lines.append("")  # Leerzeile zwischen Metrik-Gruppen

        return "\n".join(lines)
