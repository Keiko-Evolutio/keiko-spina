"""WebRTC Monitoring für Keiko Personal Assistant.

Dieses Modul implementiert das Monitoring-System für WebRTC-Verbindungen
und Performance-Metriken.
"""

import asyncio
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
from .config import WebRTCConfig
from .types import AudioQualityMetrics, WebRTCMetrics, WebRTCSession

logger = get_logger(__name__)


class MonitoringLevel(str, Enum):
    """Monitoring-Level."""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class ConnectionMetrics:
    """Verbindungsmetriken für eine Session."""
    session_id: str
    connection_setup_time: float = 0.0
    ice_gathering_time: float = 0.0
    dtls_handshake_time: float = 0.0
    first_audio_time: float = 0.0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_packets_sent: int = 0
    total_packets_received: int = 0
    total_packets_lost: int = 0
    latency_samples: list[float] = field(default_factory=list)
    audio_quality_samples: list[AudioQualityMetrics] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AggregatedMetrics:
    """Aggregierte Metriken über alle Sessions."""
    total_sessions: int = 0
    active_sessions: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    average_setup_time: float = 0.0
    average_session_duration: float = 0.0
    total_data_transferred: int = 0
    average_packet_loss: float = 0.0
    average_latency: float = 0.0
    peak_concurrent_sessions: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PerformanceAlert:
    """Performance-Alert."""
    alert_id: str
    alert_type: str
    severity: str
    session_id: str | None
    metric_name: str
    threshold_value: float
    actual_value: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class WebRTCMonitor:
    """Enterprise WebRTC Monitor für umfassendes Performance-Monitoring.

    Implementiert detailliertes Monitoring mit:
    - Real-time Performance-Metriken
    - Connection-Quality-Tracking
    - Automatische Alert-Generierung
    - Aggregierte Statistiken und Trends
    - Configurable Monitoring-Level
    """

    def __init__(self, config: WebRTCConfig | None = None):
        """Initialisiert den WebRTC Monitor.

        Args:
            config: WebRTC-Konfiguration
        """
        self.config = config or WebRTCConfig()

        # Monitoring-Daten
        self._connection_metrics: dict[str, ConnectionMetrics] = {}
        self._aggregated_metrics = AggregatedMetrics()
        self._performance_alerts: list[PerformanceAlert] = []
        self._metrics_lock = asyncio.Lock()

        # Monitoring-Tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Konfiguration
        self.monitoring_level = MonitoringLevel.DETAILED
        self.metrics_interval = timedelta(seconds=self.config.metrics_interval)
        self.enable_performance_logging = self.config.enable_performance_logging
        self.enable_metrics = self.config.enable_metrics

        # Performance-Thresholds
        self.latency_threshold_ms = 150.0
        self.packet_loss_threshold_percent = 5.0
        self.setup_time_threshold_ms = 3000.0
        self.audio_quality_threshold = 0.8

        # Event-Callbacks
        self._metrics_callbacks: list[Callable[[WebRTCMetrics], None]] = []
        self._alert_callbacks: list[Callable[[PerformanceAlert], None]] = []

        logger.info("WebRTC Monitor initialisiert")

    async def start(self) -> None:
        """Startet das Monitoring."""
        if self._running:
            logger.warning("WebRTC Monitor bereits gestartet")
            return

        if not self.enable_metrics:
            logger.info("WebRTC Monitoring deaktiviert")
            return

        self._running = True

        # Monitoring-Tasks starten
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(),
            name="webrtc-monitoring"
        )

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="webrtc-metrics-cleanup"
        )

        logger.info("WebRTC Monitor gestartet")

    async def stop(self) -> None:
        """Stoppt das Monitoring."""
        if not self._running:
            return

        self._running = False

        # Tasks stoppen
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("WebRTC Monitor gestoppt")

    async def track_session_start(self, session: WebRTCSession) -> None:
        """Startet Tracking für eine neue Session.

        Args:
            session: WebRTC-Session
        """
        async with self._metrics_lock:
            if session.session_id not in self._connection_metrics:
                metrics = ConnectionMetrics(session_id=session.session_id)
                self._connection_metrics[session.session_id] = metrics

                # Aggregierte Metriken aktualisieren
                self._aggregated_metrics.total_sessions += 1
                self._aggregated_metrics.active_sessions += 1
                self._aggregated_metrics.peak_concurrent_sessions = max(
                    self._aggregated_metrics.peak_concurrent_sessions,
                    self._aggregated_metrics.active_sessions
                )

                logger.debug(f"Session-Tracking gestartet: {session.session_id}")

    async def track_connection_setup(self, session_id: str, setup_time_ms: float) -> None:
        """Trackt Connection-Setup-Zeit.

        Args:
            session_id: Session-ID
            setup_time_ms: Setup-Zeit in Millisekunden
        """
        async with self._metrics_lock:
            if session_id in self._connection_metrics:
                metrics = self._connection_metrics[session_id]
                metrics.connection_setup_time = setup_time_ms
                metrics.last_updated = datetime.now(UTC)

                # Performance-Alert prüfen
                if setup_time_ms > self.setup_time_threshold_ms:
                    await self._generate_alert(
                        "slow_connection_setup",
                        "warning",
                        session_id,
                        "connection_setup_time",
                        self.setup_time_threshold_ms,
                        setup_time_ms,
                        f"Langsame Verbindungsaufbau: {setup_time_ms:.1f}ms"
                    )

                logger.debug(f"Connection-Setup getrackt: {session_id} - {setup_time_ms:.1f}ms")

    async def track_ice_gathering(self, session_id: str, gathering_time_ms: float) -> None:
        """Trackt ICE-Gathering-Zeit.

        Args:
            session_id: Session-ID
            gathering_time_ms: ICE-Gathering-Zeit in Millisekunden
        """
        async with self._metrics_lock:
            if session_id in self._connection_metrics:
                metrics = self._connection_metrics[session_id]
                metrics.ice_gathering_time = gathering_time_ms
                metrics.last_updated = datetime.now(UTC)

                logger.debug(f"ICE-Gathering getrackt: {session_id} - {gathering_time_ms:.1f}ms")

    async def track_dtls_handshake(self, session_id: str, handshake_time_ms: float) -> None:
        """Trackt DTLS-Handshake-Zeit.

        Args:
            session_id: Session-ID
            handshake_time_ms: DTLS-Handshake-Zeit in Millisekunden
        """
        async with self._metrics_lock:
            if session_id in self._connection_metrics:
                metrics = self._connection_metrics[session_id]
                metrics.dtls_handshake_time = handshake_time_ms
                metrics.last_updated = datetime.now(UTC)

                logger.debug(f"DTLS-Handshake getrackt: {session_id} - {handshake_time_ms:.1f}ms")

    async def track_audio_metrics(
        self,
        session_id: str,
        audio_quality: AudioQualityMetrics,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        packets_sent: int = 0,
        packets_received: int = 0,
        packets_lost: int = 0,
        latency_ms: float = 0.0
    ) -> None:
        """Trackt Audio-Metriken.

        Args:
            session_id: Session-ID
            audio_quality: Audio-Quality-Metriken
            bytes_sent: Gesendete Bytes
            bytes_received: Empfangene Bytes
            packets_sent: Gesendete Pakete
            packets_received: Empfangene Pakete
            packets_lost: Verlorene Pakete
            latency_ms: Latenz in Millisekunden
        """
        async with self._metrics_lock:
            if session_id in self._connection_metrics:
                metrics = self._connection_metrics[session_id]

                # Audio-Quality-Sample hinzufügen
                metrics.audio_quality_samples.append(audio_quality)

                # Latenz-Sample hinzufügen
                if latency_ms > 0:
                    metrics.latency_samples.append(latency_ms)

                # Kumulierte Werte aktualisieren
                metrics.total_bytes_sent += bytes_sent
                metrics.total_bytes_received += bytes_received
                metrics.total_packets_sent += packets_sent
                metrics.total_packets_received += packets_received
                metrics.total_packets_lost += packets_lost
                metrics.last_updated = datetime.now(UTC)

                # Performance-Alerts prüfen
                if latency_ms > self.latency_threshold_ms:
                    await self._generate_alert(
                        "high_latency",
                        "warning",
                        session_id,
                        "latency",
                        self.latency_threshold_ms,
                        latency_ms,
                        f"Hohe Latenz: {latency_ms:.1f}ms"
                    )

                # Packet-Loss prüfen
                if packets_sent > 0:
                    packet_loss_rate = (packets_lost / packets_sent) * 100
                    if packet_loss_rate > self.packet_loss_threshold_percent:
                        await self._generate_alert(
                            "high_packet_loss",
                            "warning",
                            session_id,
                            "packet_loss_rate",
                            self.packet_loss_threshold_percent,
                            packet_loss_rate,
                            f"Hoher Paketverlust: {packet_loss_rate:.1f}%"
                        )

                # WebRTC-Metriken-Event erstellen und senden
                webrtc_metrics = WebRTCMetrics(
                    session_id=session_id,
                    connection_setup_time=metrics.connection_setup_time,
                    ice_gathering_time=metrics.ice_gathering_time,
                    dtls_handshake_time=metrics.dtls_handshake_time,
                    audio_quality=audio_quality,
                    bytes_sent=metrics.total_bytes_sent,
                    bytes_received=metrics.total_bytes_received,
                    packets_sent=metrics.total_packets_sent,
                    packets_received=metrics.total_packets_received,
                    packets_lost=metrics.total_packets_lost,
                    current_latency=latency_ms,
                    average_latency=statistics.mean(metrics.latency_samples) if metrics.latency_samples else 0.0
                )

                # Callbacks aufrufen
                for callback in self._metrics_callbacks:
                    try:
                        callback(webrtc_metrics)
                    except Exception as e:
                        logger.error(f"Fehler in Metrics-Callback: {e}")

    async def track_session_end(self, session_id: str, success: bool = True) -> None:
        """Beendet Tracking für eine Session.

        Args:
            session_id: Session-ID
            success: Ob Session erfolgreich war
        """
        async with self._metrics_lock:
            if session_id in self._connection_metrics:
                metrics = self._connection_metrics[session_id]

                # Session-Dauer berechnen
                session_duration = (datetime.now(UTC) - metrics.created_at).total_seconds()

                # Aggregierte Metriken aktualisieren
                self._aggregated_metrics.active_sessions -= 1

                if success:
                    self._aggregated_metrics.successful_connections += 1
                else:
                    self._aggregated_metrics.failed_connections += 1

                # Durchschnittliche Session-Dauer aktualisieren
                total_sessions = self._aggregated_metrics.successful_connections + self._aggregated_metrics.failed_connections
                if total_sessions > 0:
                    current_total = self._aggregated_metrics.average_session_duration * (total_sessions - 1)
                    self._aggregated_metrics.average_session_duration = (current_total + session_duration) / total_sessions

                # Setup-Zeit-Durchschnitt aktualisieren
                if metrics.connection_setup_time > 0:
                    current_total = self._aggregated_metrics.average_setup_time * (total_sessions - 1)
                    self._aggregated_metrics.average_setup_time = (current_total + metrics.connection_setup_time) / total_sessions

                # Daten-Transfer aktualisieren
                self._aggregated_metrics.total_data_transferred += metrics.total_bytes_sent + metrics.total_bytes_received

                logger.debug(f"Session-Tracking beendet: {session_id} (Erfolg: {success}, Dauer: {session_duration:.1f}s)")

    def add_metrics_callback(self, callback: Callable[[WebRTCMetrics], None]) -> None:
        """Fügt Callback für Metriken-Updates hinzu.

        Args:
            callback: Callback-Funktion
        """
        self._metrics_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Fügt Callback für Performance-Alerts hinzu.

        Args:
            callback: Callback-Funktion
        """
        self._alert_callbacks.append(callback)

    async def get_session_metrics(self, session_id: str) -> ConnectionMetrics | None:
        """Gibt Metriken für eine Session zurück.

        Args:
            session_id: Session-ID

        Returns:
            Connection-Metriken oder None
        """
        async with self._metrics_lock:
            return self._connection_metrics.get(session_id)

    async def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Gibt aggregierte Metriken zurück.

        Returns:
            Aggregierte Metriken
        """
        async with self._metrics_lock:
            # Aktuelle Durchschnittswerte berechnen
            if self._connection_metrics:
                all_latencies = []
                all_packet_losses = []

                for metrics in self._connection_metrics.values():
                    all_latencies.extend(metrics.latency_samples)

                    if metrics.total_packets_sent > 0:
                        packet_loss_rate = (metrics.total_packets_lost / metrics.total_packets_sent) * 100
                        all_packet_losses.append(packet_loss_rate)

                if all_latencies:
                    self._aggregated_metrics.average_latency = statistics.mean(all_latencies)

                if all_packet_losses:
                    self._aggregated_metrics.average_packet_loss = statistics.mean(all_packet_losses)

            self._aggregated_metrics.last_updated = datetime.now(UTC)
            return self._aggregated_metrics

    async def get_recent_alerts(self, limit: int = 50) -> list[PerformanceAlert]:
        """Gibt aktuelle Performance-Alerts zurück.

        Args:
            limit: Maximale Anzahl Alerts

        Returns:
            Liste der Alerts
        """
        async with self._metrics_lock:
            return self._performance_alerts[-limit:]

    async def _generate_alert(
        self,
        alert_type: str,
        severity: str,
        session_id: str | None,
        metric_name: str,
        threshold_value: float,
        actual_value: float,
        message: str
    ) -> None:
        """Generiert Performance-Alert."""
        alert = PerformanceAlert(
            alert_id=f"{alert_type}_{session_id}_{datetime.now(UTC).timestamp()}",
            alert_type=alert_type,
            severity=severity,
            session_id=session_id,
            metric_name=metric_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
            message=message
        )

        self._performance_alerts.append(alert)

        # Callbacks aufrufen
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Fehler in Alert-Callback: {e}")

        if self.enable_performance_logging:
            logger.warning(f"WebRTC Performance-Alert: {message}")

    async def _monitoring_loop(self) -> None:
        """Monitoring-Loop für periodische Metriken-Updates."""
        while self._running:
            try:
                # Aggregierte Metriken aktualisieren
                await self.get_aggregated_metrics()

                # Warten bis zum nächsten Interval
                await asyncio.sleep(self.metrics_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im WebRTC-Monitoring: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self) -> None:
        """Cleanup-Loop für alte Metriken."""
        while self._running:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Stündliche Bereinigung

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Metrics-Cleanup: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_metrics(self) -> None:
        """Bereinigt alte Metriken."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        async with self._metrics_lock:
            # Alte Connection-Metriken entfernen
            old_sessions = [
                session_id for session_id, metrics in self._connection_metrics.items()
                if metrics.last_updated < cutoff_time
            ]

            for session_id in old_sessions:
                del self._connection_metrics[session_id]

            # Alte Alerts entfernen
            self._performance_alerts = [
                alert for alert in self._performance_alerts
                if alert.timestamp > cutoff_time
            ]

            if old_sessions:
                logger.debug(f"Bereinigt {len(old_sessions)} alte Session-Metriken")

    async def get_monitor_status(self) -> dict[str, Any]:
        """Gibt Monitor-Status zurück."""
        async with self._metrics_lock:
            return {
                "running": self._running,
                "monitoring_level": self.monitoring_level.value,
                "enable_metrics": self.enable_metrics,
                "enable_performance_logging": self.enable_performance_logging,
                "tracked_sessions": len(self._connection_metrics),
                "total_alerts": len(self._performance_alerts),
                "metrics_interval_seconds": self.metrics_interval.total_seconds(),
                "thresholds": {
                    "latency_ms": self.latency_threshold_ms,
                    "packet_loss_percent": self.packet_loss_threshold_percent,
                    "setup_time_ms": self.setup_time_threshold_ms,
                    "audio_quality": self.audio_quality_threshold
                }
            }


def create_webrtc_monitor(config: WebRTCConfig | None = None) -> WebRTCMonitor:
    """Factory-Funktion für WebRTC Monitor.

    Args:
        config: WebRTC-Konfiguration

    Returns:
        Neue WebRTCMonitor-Instanz
    """
    return WebRTCMonitor(config)
