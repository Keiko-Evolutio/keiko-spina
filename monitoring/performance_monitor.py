"""Performance Monitor Implementation.
Überwacht System-Performance und Application-Metriken.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import psutil

from kei_logging import get_logger

from .interfaces import IMetricsCollector, IPerformanceMonitor

logger = get_logger(__name__)


@dataclass
class ResponseTimeStats:
    """Response-Zeit-Statistiken."""
    endpoint: str
    count: int
    total_time_ms: float
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    p95_time_ms: float
    p99_time_ms: float


@dataclass
class ThroughputStats:
    """Throughput-Statistiken."""
    endpoint: str
    requests_per_second: float
    total_requests: int
    time_window_seconds: int


class PerformanceMonitor(IPerformanceMonitor):
    """Performance Monitor Implementation.
    Überwacht Response-Zeiten, Throughput und System-Ressourcen.
    """

    def __init__(self, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector

        # Response-Zeit-Tracking
        self._response_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Throughput-Tracking
        self._request_counts: dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))  # 1 hour window

        # Resource-Tracking
        self._resource_history: deque = deque(maxlen=300)  # 5 minutes at 1s intervals

        # Performance-Thresholds
        self.response_time_threshold_ms = 1000.0
        self.cpu_threshold_percent = 80.0
        self.memory_threshold_percent = 80.0

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

        logger.info("Performance monitor initialized")

    def track_response_time(self, endpoint: str, duration_ms: float) -> None:
        """Trackt Response-Zeit für Endpoint."""
        timestamp = time.time()

        # Response-Zeit speichern
        self._response_times[endpoint].append({
            "duration_ms": duration_ms,
            "timestamp": timestamp
        })

        # Metriken aktualisieren
        self.metrics_collector.observe_histogram(
            "http_request_duration_seconds",
            duration_ms / 1000.0,
            labels={"endpoint": endpoint}
        )

        # Slow Request Detection
        if duration_ms > self.response_time_threshold_ms:
            self.metrics_collector.increment_counter(
                "http_slow_requests_total",
                labels={"endpoint": endpoint}
            )

            logger.warning(f"Slow request detected: {endpoint} took {duration_ms:.1f}ms")

        logger.debug(f"Tracked response time for {endpoint}: {duration_ms:.1f}ms")

    def track_throughput(self, endpoint: str, requests_count: int = 1) -> None:
        """Trackt Throughput für Endpoint."""
        timestamp = time.time()

        # Request-Count speichern
        for _ in range(requests_count):
            self._request_counts[endpoint].append(timestamp)

        # Metriken aktualisieren
        self.metrics_collector.increment_counter(
            "http_requests_total",
            value=requests_count,
            labels={"endpoint": endpoint}
        )

        # Aktuelle RPS berechnen
        current_rps = self._calculate_current_rps(endpoint)
        self.metrics_collector.set_gauge(
            "http_requests_per_second",
            current_rps,
            labels={"endpoint": endpoint}
        )

        logger.debug(f"Tracked throughput for {endpoint}: {requests_count} requests, current RPS: {current_rps:.2f}")

    def track_resource_usage(self, cpu_percent: float, memory_mb: float, network_kb: float = None) -> None:
        """Trackt Resource-Nutzung."""
        timestamp = time.time()

        resource_data = {
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "network_kb": network_kb
        }

        self._resource_history.append(resource_data)

        # Metriken aktualisieren
        self.metrics_collector.set_gauge("system_cpu_usage_percent", cpu_percent)
        self.metrics_collector.set_gauge("system_memory_usage_mb", memory_mb)

        if network_kb is not None:
            self.metrics_collector.set_gauge("system_network_usage_kb", network_kb)

        # Threshold-Überwachung
        if cpu_percent > self.cpu_threshold_percent:
            self.metrics_collector.increment_counter("system_cpu_threshold_exceeded_total")
            logger.warning(f"CPU usage threshold exceeded: {cpu_percent:.1f}%")

        if memory_mb > self.memory_threshold_percent * 1024:  # Assuming 1GB threshold
            self.metrics_collector.increment_counter("system_memory_threshold_exceeded_total")
            logger.warning(f"Memory usage threshold exceeded: {memory_mb:.1f}MB")

        logger.debug(f"Tracked resource usage: CPU {cpu_percent:.1f}%, Memory {memory_mb:.1f}MB")

    def track_queue_length(self, queue_name: str, length: int) -> None:
        """Trackt Queue-Länge."""
        self.metrics_collector.set_gauge(
            "queue_length",
            length,
            labels={"queue": queue_name}
        )

        logger.debug(f"Tracked queue length for {queue_name}: {length}")

    async def start_system_monitoring(self) -> None:
        """Startet kontinuierliches System-Monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._system_monitoring_loop())
        logger.info("Started system monitoring")

    async def stop_system_monitoring(self) -> None:
        """Stoppt System-Monitoring."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped system monitoring")

    async def _system_monitoring_loop(self) -> None:
        """System-Monitoring-Schleife."""
        while self._running:
            try:
                # System-Ressourcen sammeln
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # Network I/O (optional)
                try:
                    network = psutil.net_io_counters()
                    network_kb = (network.bytes_sent + network.bytes_recv) / 1024
                except Exception:
                    network_kb = None

                # Resource-Tracking
                self.track_resource_usage(
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / (1024 * 1024),
                    network_kb=network_kb
                )

                # Process-spezifische Metriken
                try:
                    process = psutil.Process()
                    process_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    process_cpu = process.cpu_percent()

                    self.metrics_collector.set_gauge("process_memory_usage_mb", process_memory)
                    self.metrics_collector.set_gauge("process_cpu_usage_percent", process_cpu)

                except Exception as e:
                    logger.debug(f"Could not collect process metrics: {e}")

                await asyncio.sleep(1)  # Monitor every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(5)

    def get_response_time_stats(self, endpoint: str = None) -> list[ResponseTimeStats]:
        """Gibt Response-Zeit-Statistiken zurück."""
        stats = []

        endpoints = [endpoint] if endpoint else self._response_times.keys()

        for ep in endpoints:
            if ep not in self._response_times or not self._response_times[ep]:
                continue

            durations = [entry["duration_ms"] for entry in self._response_times[ep]]

            if not durations:
                continue

            durations.sort()
            count = len(durations)

            stats.append(ResponseTimeStats(
                endpoint=ep,
                count=count,
                total_time_ms=sum(durations),
                min_time_ms=min(durations),
                max_time_ms=max(durations),
                avg_time_ms=sum(durations) / count,
                p95_time_ms=durations[int(0.95 * count)] if count > 0 else 0,
                p99_time_ms=durations[int(0.99 * count)] if count > 0 else 0
            ))

        return stats

    def get_throughput_stats(self, endpoint: str = None, window_seconds: int = 60) -> list[ThroughputStats]:
        """Gibt Throughput-Statistiken zurück."""
        stats = []
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        endpoints = [endpoint] if endpoint else self._request_counts.keys()

        for ep in endpoints:
            if ep not in self._request_counts:
                continue

            # Requests im Zeitfenster zählen
            recent_requests = [
                timestamp for timestamp in self._request_counts[ep]
                if timestamp >= cutoff_time
            ]

            total_requests = len(recent_requests)
            rps = total_requests / window_seconds if window_seconds > 0 else 0

            stats.append(ThroughputStats(
                endpoint=ep,
                requests_per_second=rps,
                total_requests=total_requests,
                time_window_seconds=window_seconds
            ))

        return stats

    def get_resource_stats(self, window_seconds: int = 300) -> dict[str, Any]:
        """Gibt Resource-Statistiken zurück."""
        if not self._resource_history:
            return {}

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Daten im Zeitfenster filtern
        recent_data = [
            entry for entry in self._resource_history
            if entry["timestamp"] >= cutoff_time
        ]

        if not recent_data:
            return {}

        # Statistiken berechnen
        cpu_values = [entry["cpu_percent"] for entry in recent_data]
        memory_values = [entry["memory_mb"] for entry in recent_data]

        return {
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": memory_values[-1] if memory_values else 0
            },
            "data_points": len(recent_data),
            "time_window_seconds": window_seconds
        }

    def _calculate_current_rps(self, endpoint: str, window_seconds: int = 60) -> float:
        """Berechnet aktuelle Requests per Second."""
        if endpoint not in self._request_counts:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        recent_requests = [
            timestamp for timestamp in self._request_counts[endpoint]
            if timestamp >= cutoff_time
        ]

        return len(recent_requests) / window_seconds if window_seconds > 0 else 0.0
