"""Gemeinsame Metriken-Utilities für KEI-Stream.

Zentrale Funktionen für Metriken-Aufzeichnung mit konsistenter
Fehlerbehandlung und Logging.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from prometheus_client import Counter, Gauge, Histogram

logger = get_logger(__name__)


class MetricsRecorder:
    """Basis-Klasse für Metriken-Aufzeichnung mit Fehlerbehandlung."""

    def __init__(self, component_name: str):
        """Initialisiert den Metrics Recorder.

        Args:
            component_name: Name der Komponente für Logging
        """
        self.component_name = component_name
        self.logger = logger

    def safe_record(self, metric_func: Callable, *args, **kwargs) -> None:
        """Führt Metriken-Aufzeichnung mit Fehlerbehandlung durch.

        Args:
            metric_func: Metriken-Funktion die aufgerufen werden soll
            *args: Positionale Argumente für die Funktion
            **kwargs: Keyword-Argumente für die Funktion
        """
        try:
            metric_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(
                f"Fehler beim Aufzeichnen von Metriken in {self.component_name}: {e}"
            )

    def record_counter(self, counter: Counter, labels: dict[str, str], value: float = 1.0) -> None:
        """Zeichnet Counter-Metrik auf.

        Args:
            counter: Prometheus Counter
            labels: Label-Dictionary
            value: Wert der hinzugefügt werden soll
        """
        self.safe_record(lambda: counter.labels(**labels).inc(value))

    def record_gauge(self, gauge: Gauge, labels: dict[str, str], value: float) -> None:
        """Zeichnet Gauge-Metrik auf.

        Args:
            gauge: Prometheus Gauge
            labels: Label-Dictionary
            value: Wert der gesetzt werden soll
        """
        self.safe_record(lambda: gauge.labels(**labels).set(value))

    def record_histogram(self, histogram: Histogram, labels: dict[str, str], value: float) -> None:
        """Zeichnet Histogram-Metrik auf.

        Args:
            histogram: Prometheus Histogram
            labels: Label-Dictionary
            value: Wert der beobachtet werden soll
        """
        self.safe_record(lambda: histogram.labels(**labels).observe(value))

    def increment_gauge(self, gauge: Gauge, labels: dict[str, str], value: float = 1.0) -> None:
        """Erhöht Gauge-Wert.

        Args:
            gauge: Prometheus Gauge
            labels: Label-Dictionary
            value: Wert um den erhöht werden soll
        """
        self.safe_record(lambda: gauge.labels(**labels).inc(value))

    def decrement_gauge(self, gauge: Gauge, labels: dict[str, str], value: float = 1.0) -> None:
        """Verringert Gauge-Wert.

        Args:
            gauge: Prometheus Gauge
            labels: Label-Dictionary
            value: Wert um den verringert werden soll
        """
        self.safe_record(lambda: gauge.labels(**labels).dec(value))


class TimingMetrics:
    """Utility-Klasse für Zeit-basierte Metriken."""

    def __init__(self, recorder: MetricsRecorder):
        """Initialisiert TimingMetrics.

        Args:
            recorder: MetricsRecorder-Instanz
        """
        self.recorder = recorder

    @asynccontextmanager
    async def time_async_operation(
        self,
        histogram: Histogram,
        labels: dict[str, str],
        operation_name: str = "operation"
    ):
        """Context Manager für asynchrone Zeit-Messung.

        Args:
            histogram: Prometheus Histogram für Zeitmessung
            labels: Label-Dictionary
            operation_name: Name der Operation für Logging
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.recorder.record_histogram(histogram, labels, duration)
            self.recorder.logger.debug(
                f"{operation_name} dauerte {duration:.3f}s"
            )

    def time_sync_operation(
        self,
        histogram: Histogram,
        labels: dict[str, str],
        operation_name: str = "operation"
    ):
        """Context Manager für synchrone Zeit-Messung.

        Args:
            histogram: Prometheus Histogram für Zeitmessung
            labels: Label-Dictionary
            operation_name: Name der Operation für Logging
        """
        class SyncTimingContext:
            def __init__(self, recorder, hist, lbls, name):
                self.recorder = recorder
                self.histogram = hist
                self.labels = lbls
                self.operation_name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.recorder.record_histogram(self.histogram, self.labels, duration)
                self.recorder.logger.debug(
                    f"{self.operation_name} dauerte {duration:.3f}s"
                )

        return SyncTimingContext(self.recorder, histogram, labels, operation_name)


class ConnectionMetrics:
    """Spezialisierte Metriken für Verbindungsmanagement."""

    def __init__(self, recorder: MetricsRecorder):
        """Initialisiert ConnectionMetrics.

        Args:
            recorder: MetricsRecorder-Instanz
        """
        self.recorder = recorder

    def record_connection_start(
        self,
        active_gauge: Gauge,
        total_counter: Counter,
        endpoint: str,
        session_id: str,
        status: str = "connected"
    ) -> None:
        """Zeichnet Verbindungsstart auf.

        Args:
            active_gauge: Gauge für aktive Verbindungen
            total_counter: Counter für Gesamtverbindungen
            endpoint: Endpoint-Identifier
            session_id: Session-ID
            status: Verbindungsstatus
        """
        labels = {"endpoint": endpoint, "session_id": session_id}
        self.recorder.increment_gauge(active_gauge, labels)

        total_labels = {**labels, "status": status}
        self.recorder.record_counter(total_counter, total_labels)

    def record_connection_end(
        self,
        active_gauge: Gauge,
        duration_histogram: Histogram,
        endpoint: str,
        session_id: str,
        duration_seconds: float,
        disconnect_reason: str = "normal"
    ) -> None:
        """Zeichnet Verbindungsende auf.

        Args:
            active_gauge: Gauge für aktive Verbindungen
            duration_histogram: Histogram für Verbindungsdauer
            endpoint: Endpoint-Identifier
            session_id: Session-ID
            duration_seconds: Verbindungsdauer in Sekunden
            disconnect_reason: Grund für Trennung
        """
        labels = {"endpoint": endpoint, "session_id": session_id}
        self.recorder.decrement_gauge(active_gauge, labels)

        duration_labels = {**labels, "disconnect_reason": disconnect_reason}
        self.recorder.record_histogram(duration_histogram, duration_labels, duration_seconds)


class MessageMetrics:
    """Spezialisierte Metriken für Nachrichten-Verarbeitung."""

    def __init__(self, recorder: MetricsRecorder):
        """Initialisiert MessageMetrics.

        Args:
            recorder: MetricsRecorder-Instanz
        """
        self.recorder = recorder

    def record_message_sent(
        self,
        sent_counter: Counter,
        size_histogram: Histogram,
        endpoint: str,
        session_id: str,
        stream_id: str,
        frame_type: str,
        message_size_bytes: int
    ) -> None:
        """Zeichnet gesendete Nachricht auf.

        Args:
            sent_counter: Counter für gesendete Nachrichten
            size_histogram: Histogram für Nachrichtengröße
            endpoint: Endpoint-Identifier
            session_id: Session-ID
            stream_id: Stream-ID
            frame_type: Frame-Type
            message_size_bytes: Nachrichtengröße in Bytes
        """
        labels = {
            "endpoint": endpoint,
            "session_id": session_id,
            "stream_id": stream_id,
            "frame_type": frame_type
        }

        self.recorder.record_counter(sent_counter, labels)
        self.recorder.record_histogram(size_histogram, labels, message_size_bytes)

    def record_message_error(
        self,
        error_counter: Counter,
        endpoint: str,
        session_id: str,
        error_type: str
    ) -> None:
        """Zeichnet Nachrichten-Fehler auf.

        Args:
            error_counter: Counter für Fehler
            endpoint: Endpoint-Identifier
            session_id: Session-ID
            error_type: Fehler-Type
        """
        labels = {
            "endpoint": endpoint,
            "session_id": session_id,
            "error_type": error_type
        }

        self.recorder.record_counter(error_counter, labels)


def create_metrics_recorder(component_name: str) -> MetricsRecorder:
    """Factory-Funktion für MetricsRecorder.

    Args:
        component_name: Name der Komponente

    Returns:
        Konfigurierte MetricsRecorder-Instanz
    """
    return MetricsRecorder(component_name)


def create_timing_metrics(recorder: MetricsRecorder) -> TimingMetrics:
    """Factory-Funktion für TimingMetrics.

    Args:
        recorder: MetricsRecorder-Instanz

    Returns:
        TimingMetrics-Instanz
    """
    return TimingMetrics(recorder)


def create_connection_metrics(recorder: MetricsRecorder) -> ConnectionMetrics:
    """Factory-Funktion für ConnectionMetrics.

    Args:
        recorder: MetricsRecorder-Instanz

    Returns:
        ConnectionMetrics-Instanz
    """
    return ConnectionMetrics(recorder)


def create_message_metrics(recorder: MetricsRecorder) -> MessageMetrics:
    """Factory-Funktion für MessageMetrics.

    Args:
        recorder: MetricsRecorder-Instanz

    Returns:
        MessageMetrics-Instanz
    """
    return MessageMetrics(recorder)


__all__ = [
    "ConnectionMetrics",
    "MessageMetrics",
    "MetricsRecorder",
    "TimingMetrics",
    "create_connection_metrics",
    "create_message_metrics",
    "create_metrics_recorder",
    "create_timing_metrics",
]
