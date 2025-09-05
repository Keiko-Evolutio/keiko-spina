"""Prometheus-Metriken für KEI-Stream SSE-Transport.

Umfassende Observability-Metriken für Server-Sent Events (SSE) Transport,
einschließlich Verbindungsmanagement, Nachrichtenversand und Fehlerbehandlung.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from kei_logging import get_logger

from .constants import CONNECTION_DURATION_BUCKETS, POLLING_DURATION_BUCKETS, SIZE_BUCKETS_BYTES
from .metrics_utils import (
    create_connection_metrics,
    create_message_metrics,
    create_metrics_recorder,
)

logger = get_logger(__name__)


# ============================================================================
# SSE TRANSPORT PROMETHEUS METRIKEN
# ============================================================================

# Gauge für aktive SSE-Verbindungen
SSE_CONNECTIONS_ACTIVE = Gauge(
    "kei_stream_sse_connections_active",
    "Anzahl aktiver SSE-Verbindungen",
    labelnames=("endpoint", "session_id"),
)

# Counter für gesendete SSE-Nachrichten
SSE_MESSAGES_SENT = Counter(
    "kei_stream_sse_messages_sent_total",
    "Gesamtanzahl gesendeter SSE-Nachrichten",
    labelnames=("endpoint", "session_id", "stream_id", "frame_type"),
)

# Counter für SSE-Fehler
SSE_ERRORS_TOTAL = Counter(
    "kei_stream_sse_errors_total",
    "Gesamtanzahl SSE-Fehler",
    labelnames=("endpoint", "session_id", "error_type"),
)

# Histogram für Verbindungsdauer
SSE_CONNECTION_DURATION_SECONDS = Histogram(
    "kei_stream_sse_connection_duration_seconds",
    "Dauer von SSE-Verbindungen in Sekunden",
    labelnames=("endpoint", "session_id", "disconnect_reason"),
    buckets=CONNECTION_DURATION_BUCKETS,
)

# Histogram für Nachrichtengröße
SSE_MESSAGE_SIZE_BYTES = Histogram(
    "kei_stream_sse_message_size_bytes",
    "Größe von SSE-Nachrichten in Bytes",
    labelnames=("endpoint", "session_id", "stream_id", "frame_type"),
    buckets=SIZE_BUCKETS_BYTES,
)

# Counter für Verbindungsversuche
SSE_CONNECTIONS_TOTAL = Counter(
    "kei_stream_sse_connections_total",
    "Gesamtanzahl SSE-Verbindungsversuche",
    labelnames=("endpoint", "session_id", "status"),
)

# Gauge für Replay-Fenster-Größe
SSE_REPLAY_WINDOW_SIZE = Gauge(
    "kei_stream_sse_replay_window_size",
    "Aktuelle Größe des Replay-Fensters pro Stream",
    labelnames=("endpoint", "session_id", "stream_id"),
)

# Histogram für Polling-Intervall-Performance
SSE_POLLING_DURATION_SECONDS = Histogram(
    "kei_stream_sse_polling_duration_seconds",
    "Dauer einzelner Polling-Zyklen in Sekunden",
    labelnames=("endpoint", "session_id", "stream_id"),
    buckets=POLLING_DURATION_BUCKETS,
)

# Counter für Client-Disconnects
SSE_CLIENT_DISCONNECTS_TOTAL = Counter(
    "kei_stream_sse_client_disconnects_total",
    "Gesamtanzahl Client-Disconnects",
    labelnames=("endpoint", "session_id", "disconnect_type"),
)


# ============================================================================
# METRIKEN-HILFSFUNKTIONEN
# ============================================================================

class SSEMetricsRecorder:
    """Zentrale Klasse für SSE-Metriken-Aufzeichnung mit Fehlerbehandlung."""

    def __init__(self):
        self.base_recorder = create_metrics_recorder("SSE")
        self.connection_metrics = create_connection_metrics(self.base_recorder)
        self.message_metrics = create_message_metrics(self.base_recorder)

    def record_connection_start(
        self,
        endpoint: str,
        session_id: str,
        status: str = "connected"
    ) -> None:
        """Zeichnet Verbindungsstart auf."""
        self.connection_metrics.record_connection_start(
            SSE_CONNECTIONS_ACTIVE,
            SSE_CONNECTIONS_TOTAL,
            endpoint,
            session_id,
            status
        )
        logger.debug(f"SSE-Verbindung gestartet: endpoint={endpoint}, session={session_id}")

    def record_connection_end(
        self,
        endpoint: str,
        session_id: str,
        duration_seconds: float,
        disconnect_reason: str = "normal"
    ) -> None:
        """Zeichnet Verbindungsende auf."""
        self.connection_metrics.record_connection_end(
            SSE_CONNECTIONS_ACTIVE,
            SSE_CONNECTION_DURATION_SECONDS,
            endpoint,
            session_id,
            duration_seconds,
            disconnect_reason
        )
        logger.debug(f"SSE-Verbindung beendet: endpoint={endpoint}, session={session_id}, "
                    f"duration={duration_seconds:.2f}s, reason={disconnect_reason}")

    def record_message_sent(
        self,
        endpoint: str,
        session_id: str,
        stream_id: str,
        frame_type: str,
        message_size_bytes: int
    ) -> None:
        """Zeichnet gesendete Nachricht auf."""
        self.message_metrics.record_message_sent(
            SSE_MESSAGES_SENT,
            SSE_MESSAGE_SIZE_BYTES,
            endpoint,
            session_id,
            stream_id,
            frame_type,
            message_size_bytes
        )

    def record_error(
        self,
        endpoint: str,
        session_id: str,
        error_type: str
    ) -> None:
        """Zeichnet SSE-Fehler auf."""
        self.message_metrics.record_message_error(
            SSE_ERRORS_TOTAL,
            endpoint,
            session_id,
            error_type
        )
        logger.debug(f"SSE-Fehler aufgezeichnet: endpoint={endpoint}, session={session_id}, "
                    f"error_type={error_type}")

    def record_client_disconnect(
        self,
        endpoint: str,
        session_id: str,
        disconnect_type: str = "normal"
    ) -> None:
        """Zeichnet Client-Disconnect auf."""
        labels = {
            "endpoint": endpoint,
            "session_id": session_id,
            "disconnect_type": disconnect_type
        }
        self.base_recorder.record_counter(SSE_CLIENT_DISCONNECTS_TOTAL, labels)

    def record_polling_duration(
        self,
        endpoint: str,
        session_id: str,
        stream_id: str,
        duration_seconds: float
    ) -> None:
        """Zeichnet Polling-Dauer auf."""
        labels = {
            "endpoint": endpoint,
            "session_id": session_id,
            "stream_id": stream_id
        }
        self.base_recorder.record_histogram(SSE_POLLING_DURATION_SECONDS, labels, duration_seconds)

    def update_replay_window_size(
        self,
        endpoint: str,
        session_id: str,
        stream_id: str,
        window_size: int
    ) -> None:
        """Aktualisiert Replay-Fenster-Größe."""
        labels = {
            "endpoint": endpoint,
            "session_id": session_id,
            "stream_id": stream_id
        }
        self.base_recorder.record_gauge(SSE_REPLAY_WINDOW_SIZE, labels, window_size)


# Globale Instanz für einfache Verwendung
sse_metrics = SSEMetricsRecorder()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SSE_CLIENT_DISCONNECTS_TOTAL",
    "SSE_CONNECTIONS_ACTIVE",
    "SSE_CONNECTIONS_TOTAL",
    "SSE_CONNECTION_DURATION_SECONDS",
    "SSE_ERRORS_TOTAL",
    "SSE_MESSAGES_SENT",
    "SSE_MESSAGE_SIZE_BYTES",
    "SSE_POLLING_DURATION_SECONDS",
    "SSE_REPLAY_WINDOW_SIZE",
    "SSEMetricsRecorder",
    "sse_metrics",
]
