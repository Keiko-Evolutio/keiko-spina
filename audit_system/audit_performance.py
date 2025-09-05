# backend/audit_system/audit_performance.py
"""Performance und Skalierbarkeit für KEI-Agent-Framework Audit System.

Implementiert asynchrone Audit-Logging, Batch-Processing,
horizontale Skalierung und Real-time-Monitoring.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .core_audit_engine import AuditEvent

logger = get_logger(__name__)


class ProcessingMode(str, Enum):
    """Modi für Event-Processing."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"


class ScalingStrategy(str, Enum):
    """Strategien für horizontale Skalierung."""
    ROUND_ROBIN = "round_robin"
    LOAD_BASED = "load_based"
    HASH_BASED = "hash_based"
    PRIORITY_BASED = "priority_based"


@dataclass
class PerformanceMetrics:
    """Performance-Metriken für Audit-System."""
    timestamp: datetime

    # Throughput-Metriken
    events_per_second: float = 0.0
    batches_per_second: float = 0.0

    # Latenz-Metriken
    avg_processing_latency_ms: float = 0.0
    p95_processing_latency_ms: float = 0.0
    p99_processing_latency_ms: float = 0.0

    # Queue-Metriken
    queue_size: int = 0
    queue_utilization: float = 0.0

    # Ressourcen-Metriken
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

    # Error-Metriken
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0

    # Skalierungs-Metriken
    active_workers: int = 0
    worker_utilization: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "events_per_second": self.events_per_second,
            "batches_per_second": self.batches_per_second,
            "avg_processing_latency_ms": self.avg_processing_latency_ms,
            "p95_processing_latency_ms": self.p95_processing_latency_ms,
            "p99_processing_latency_ms": self.p99_processing_latency_ms,
            "queue_size": self.queue_size,
            "queue_utilization": self.queue_utilization,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "error_rate_percent": self.error_rate_percent,
            "timeout_rate_percent": self.timeout_rate_percent,
            "active_workers": self.active_workers,
            "worker_utilization": self.worker_utilization
        }


@dataclass
class BatchConfig:
    """Konfiguration für Batch-Processing."""
    max_batch_size: int = 100
    max_batch_wait_ms: int = 1000
    max_memory_mb: int = 100
    compression_enabled: bool = True
    parallel_processing: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "max_batch_size": self.max_batch_size,
            "max_batch_wait_ms": self.max_batch_wait_ms,
            "max_memory_mb": self.max_memory_mb,
            "compression_enabled": self.compression_enabled,
            "parallel_processing": self.parallel_processing
        }


@dataclass
class WorkerNode:
    """Worker-Node für horizontale Skalierung."""
    node_id: str
    endpoint: str
    capacity: int

    # Status
    is_active: bool = True
    current_load: int = 0
    last_heartbeat: datetime | None = None

    # Performance
    avg_processing_time_ms: float = 0.0
    success_rate: float = 1.0

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Gibt Auslastung zurück."""
        return self.current_load / self.capacity if self.capacity > 0 else 0.0

    @property
    def is_healthy(self) -> bool:
        """Prüft, ob Node gesund ist."""
        if not self.is_active:
            return False

        if self.last_heartbeat:
            time_since_heartbeat = datetime.now(UTC) - self.last_heartbeat
            if time_since_heartbeat > timedelta(minutes=5):
                return False

        return self.success_rate > 0.8  # 80% Erfolgsrate

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "capacity": self.capacity,
            "is_active": self.is_active,
            "current_load": self.current_load,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "success_rate": self.success_rate,
            "utilization": self.utilization,
            "is_healthy": self.is_healthy,
            "metadata": self.metadata
        }


class AsyncAuditProcessor:
    """Asynchroner Audit-Event-Prozessor."""

    def __init__(self, max_concurrent_tasks: int = 100):
        """Initialisiert Async Audit Processor.

        Args:
            max_concurrent_tasks: Maximale gleichzeitige Tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Event-Queue
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Performance-Tracking
        self._processing_times: deque = deque(maxlen=1000)
        self._events_processed = 0
        self._events_failed = 0

        # Callbacks
        self._event_processors: list[Callable[[AuditEvent], Awaitable[bool]]] = []

    def register_processor(self, processor: Callable[[AuditEvent], Awaitable[bool]]) -> None:
        """Registriert Event-Prozessor."""
        self._event_processors.append(processor)

    async def start(self, num_workers: int = 10) -> None:
        """Startet asynchrone Verarbeitung.

        Args:
            num_workers: Anzahl Worker-Tasks
        """
        if self._is_running:
            return

        self._is_running = True

        # Starte Worker-Tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self._processing_tasks.append(task)

        logger.info(f"Async Audit Processor gestartet mit {num_workers} Workern")

    async def stop(self) -> None:
        """Stoppt asynchrone Verarbeitung."""
        self._is_running = False

        # Stoppe alle Worker-Tasks
        for task in self._processing_tasks:
            task.cancel()

        # Warte auf Completion
        await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        logger.info("Async Audit Processor gestoppt")

    @trace_function("audit_performance.submit_event")
    async def submit_event(self, event: AuditEvent) -> None:
        """Reicht Event zur asynchronen Verarbeitung ein.

        Args:
            event: Zu verarbeitendes Event
        """
        await self._event_queue.put(event)

    async def _worker_loop(self, worker_id: str) -> None:
        """Worker-Loop für Event-Verarbeitung."""
        while self._is_running:
            try:
                # Event aus Queue holen
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                # Event verarbeiten
                await self._process_event_async(event, worker_id)

            except TimeoutError:
                # Timeout ist normal
                continue
            except Exception as e:
                logger.exception(f"Worker {worker_id} Fehler: {e}")
                await asyncio.sleep(1.0)

    async def _process_event_async(self, event: AuditEvent, worker_id: str) -> None:
        """Verarbeitet Event asynchron."""
        async with self._semaphore:
            start_time = time.time()

            try:
                # Verarbeite mit allen registrierten Prozessoren
                success_count = 0

                for processor in self._event_processors:
                    try:
                        success = await processor(event)
                        if success:
                            success_count += 1
                    except Exception as e:
                        logger.exception(f"Prozessor-Fehler in {worker_id}: {e}")

                # Tracking
                processing_time = (time.time() - start_time) * 1000
                self._processing_times.append(processing_time)

                if success_count > 0:
                    self._events_processed += 1
                else:
                    self._events_failed += 1

            except Exception as e:
                logger.exception(f"Event-Verarbeitung fehlgeschlagen in {worker_id}: {e}")
                self._events_failed += 1

    def get_performance_metrics(self) -> dict[str, Any]:
        """Gibt Performance-Metriken zurück."""
        if self._processing_times:
            avg_latency = sum(self._processing_times) / len(self._processing_times)
            sorted_times = sorted(self._processing_times)
            p95_latency = sorted_times[int(len(sorted_times) * 0.95)]
            p99_latency = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_latency = p95_latency = p99_latency = 0.0

        total_events = self._events_processed + self._events_failed
        error_rate = (self._events_failed / total_events * 100) if total_events > 0 else 0.0

        return {
            "events_processed": self._events_processed,
            "events_failed": self._events_failed,
            "error_rate_percent": error_rate,
            "avg_processing_latency_ms": avg_latency,
            "p95_processing_latency_ms": p95_latency,
            "p99_processing_latency_ms": p99_latency,
            "queue_size": self._event_queue.qsize(),
            "active_workers": len(self._processing_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks
        }


class BatchProcessor:
    """Batch-Prozessor für High-Volume-Audit-Events."""

    def __init__(self, config: BatchConfig):
        """Initialisiert Batch Processor.

        Args:
            config: Batch-Konfiguration
        """
        self.config = config

        # Batch-Management
        self._current_batch: list[AuditEvent] = []
        self._batch_start_time = time.time()
        self._batch_lock = threading.Lock()

        # Processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._is_running = False
        self._batch_task: asyncio.Task | None = None

        # Callbacks
        self._batch_processors: list[Callable[[list[AuditEvent]], Awaitable[bool]]] = []

        # Statistiken
        self._batches_processed = 0
        self._events_batched = 0
        self._batch_processing_times: deque = deque(maxlen=100)

    def register_batch_processor(self, processor: Callable[[list[AuditEvent]], Awaitable[bool]]) -> None:
        """Registriert Batch-Prozessor."""
        self._batch_processors.append(processor)

    async def start(self) -> None:
        """Startet Batch-Processing."""
        if self._is_running:
            return

        self._is_running = True
        self._batch_task = asyncio.create_task(self._batch_monitoring_loop())

        logger.info("Batch Processor gestartet")

    async def stop(self) -> None:
        """Stoppt Batch-Processing."""
        self._is_running = False

        # Verarbeite verbleibende Events
        if self._current_batch:
            await self._process_current_batch()

        if self._batch_task:
            self._batch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._batch_task

        self._executor.shutdown(wait=True)

        logger.info("Batch Processor gestoppt")

    @trace_function("audit_performance.add_to_batch")
    async def add_event(self, event: AuditEvent) -> None:
        """Fügt Event zum Batch hinzu.

        Args:
            event: Hinzuzufügendes Event
        """
        with self._batch_lock:
            self._current_batch.append(event)

            # Prüfe Batch-Limits
            should_process = (
                len(self._current_batch) >= self.config.max_batch_size or
                self._get_batch_age_ms() >= self.config.max_batch_wait_ms or
                self._estimate_batch_memory_mb() >= self.config.max_memory_mb
            )

        if should_process:
            await self._process_current_batch()

    async def _batch_monitoring_loop(self) -> None:
        """Monitoring-Loop für Batch-Processing."""
        while self._is_running:
            try:
                # Prüfe Batch-Alter
                if self._current_batch and self._get_batch_age_ms() >= self.config.max_batch_wait_ms:
                    await self._process_current_batch()

                # Warte kurz
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"Batch-Monitoring-Fehler: {e}")
                await asyncio.sleep(1.0)

    async def _process_current_batch(self) -> None:
        """Verarbeitet aktuellen Batch."""
        with self._batch_lock:
            if not self._current_batch:
                return

            batch_to_process = self._current_batch.copy()
            self._current_batch.clear()
            self._batch_start_time = time.time()

        start_time = time.time()

        try:
            # Verarbeite Batch
            if self.config.parallel_processing:
                await self._process_batch_parallel(batch_to_process)
            else:
                await self._process_batch_sequential(batch_to_process)

            # Tracking
            processing_time = (time.time() - start_time) * 1000
            self._batch_processing_times.append(processing_time)
            self._batches_processed += 1
            self._events_batched += len(batch_to_process)

            logger.debug(f"Batch verarbeitet: {len(batch_to_process)} Events in {processing_time:.2f}ms")

        except Exception as e:
            logger.exception(f"Batch-Verarbeitung fehlgeschlagen: {e}")

    async def _process_batch_parallel(self, batch: list[AuditEvent]) -> None:
        """Verarbeitet Batch parallel."""
        tasks = []

        for processor in self._batch_processors:
            # Konvertiere Awaitable zu Coroutine für create_task
            coro = processor(batch)
            if hasattr(coro, "__await__"):
                task = asyncio.create_task(coro)
            else:
                # Fallback für nicht-Coroutine Awaitables
                task = asyncio.create_task(asyncio.ensure_future(coro))
            tasks.append(task)

        # Warte auf alle Prozessoren
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Prüfe Ergebnisse
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch-Prozessor {i} fehlgeschlagen: {result}")

    async def _process_batch_sequential(self, batch: list[AuditEvent]) -> None:
        """Verarbeitet Batch sequenziell."""
        for processor in self._batch_processors:
            try:
                await processor(batch)
            except Exception as e:
                logger.exception(f"Batch-Prozessor fehlgeschlagen: {e}")

    def _get_batch_age_ms(self) -> float:
        """Gibt Batch-Alter in Millisekunden zurück."""
        return (time.time() - self._batch_start_time) * 1000

    def _estimate_batch_memory_mb(self) -> float:
        """Schätzt Batch-Memory-Verbrauch."""
        if not self._current_batch:
            return 0.0

        # Vereinfachte Schätzung: ~1KB pro Event
        estimated_bytes = len(self._current_batch) * 1024
        return estimated_bytes / (1024 * 1024)

    def get_batch_statistics(self) -> dict[str, Any]:
        """Gibt Batch-Statistiken zurück."""
        if self._batch_processing_times:
            avg_batch_time = sum(self._batch_processing_times) / len(self._batch_processing_times)
        else:
            avg_batch_time = 0.0

        return {
            "batches_processed": self._batches_processed,
            "events_batched": self._events_batched,
            "avg_batch_processing_time_ms": avg_batch_time,
            "current_batch_size": len(self._current_batch),
            "current_batch_age_ms": self._get_batch_age_ms(),
            "estimated_batch_memory_mb": self._estimate_batch_memory_mb(),
            "config": self.config.to_dict()
        }


class AuditEventStreamer:
    """Event-Streamer für Real-time-Monitoring."""

    def __init__(self):
        """Initialisiert Audit Event Streamer."""
        self._subscribers: dict[str, Callable[[AuditEvent], Awaitable[None]]] = {}
        self._event_filters: dict[str, Callable[[AuditEvent], bool]] = {}

        # Stream-Statistiken
        self._events_streamed = 0
        self._subscribers_notified = 0
        self._stream_errors = 0

    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[AuditEvent], Awaitable[None]],
        event_filter: Callable[[AuditEvent], bool] | None = None
    ) -> None:
        """Abonniert Event-Stream.

        Args:
            subscriber_id: Subscriber-ID
            callback: Callback-Funktion
            event_filter: Optional Event-Filter
        """
        self._subscribers[subscriber_id] = callback

        if event_filter:
            self._event_filters[subscriber_id] = event_filter

        logger.info(f"Event-Stream-Subscriber registriert: {subscriber_id}")

    def unsubscribe(self, subscriber_id: str) -> None:
        """Kündigt Event-Stream-Abonnement.

        Args:
            subscriber_id: Subscriber-ID
        """
        self._subscribers.pop(subscriber_id, None)
        self._event_filters.pop(subscriber_id, None)

        logger.info(f"Event-Stream-Subscriber entfernt: {subscriber_id}")

    @trace_function("audit_performance.stream_event")
    async def stream_event(self, event: AuditEvent) -> None:
        """Streamt Event an alle Subscriber.

        Args:
            event: Zu streamender Event
        """
        self._events_streamed += 1

        # Benachrichtige alle Subscriber
        notification_tasks = []

        for subscriber_id, callback in self._subscribers.items():
            # Prüfe Filter
            event_filter = self._event_filters.get(subscriber_id)
            if event_filter and not event_filter(event):
                continue

            # Erstelle Notification-Task
            task = asyncio.create_task(
                self._notify_subscriber(subscriber_id, callback, event)
            )
            notification_tasks.append(task)

        # Warte auf alle Notifications (mit Timeout)
        if notification_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*notification_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except TimeoutError:
                logger.warning("Event-Stream-Notifications Timeout")

    async def _notify_subscriber(
        self,
        subscriber_id: str,
        callback: Callable[[AuditEvent], Awaitable[None]],
        event: AuditEvent
    ) -> None:
        """Benachrichtigt einzelnen Subscriber."""
        try:
            await callback(event)
            self._subscribers_notified += 1

        except Exception as e:
            logger.exception(f"Event-Stream-Notification fehlgeschlagen für {subscriber_id}: {e}")
            self._stream_errors += 1

    def get_stream_statistics(self) -> dict[str, Any]:
        """Gibt Stream-Statistiken zurück."""
        return {
            "events_streamed": self._events_streamed,
            "subscribers_notified": self._subscribers_notified,
            "stream_errors": self._stream_errors,
            "active_subscribers": len(self._subscribers),
            "filtered_subscribers": len(self._event_filters)
        }


class HorizontalScaler:
    """Horizontaler Scaler für Audit-System."""

    def __init__(self, scaling_strategy: ScalingStrategy = ScalingStrategy.LOAD_BASED):
        """Initialisiert Horizontal Scaler.

        Args:
            scaling_strategy: Skalierungs-Strategie
        """
        self.scaling_strategy = scaling_strategy

        # Worker-Nodes
        self._worker_nodes: dict[str, WorkerNode] = {}
        self._node_selector_index = 0

        # Load-Balancing
        self._load_balancer_lock = threading.Lock()

        # Statistiken
        self._events_distributed = 0
        self._node_failures = 0

    def register_worker_node(self, node: WorkerNode) -> None:
        """Registriert Worker-Node.

        Args:
            node: Worker-Node
        """
        self._worker_nodes[node.node_id] = node
        logger.info(f"Worker-Node registriert: {node.node_id} ({node.endpoint})")

    def unregister_worker_node(self, node_id: str) -> None:
        """Entfernt Worker-Node.

        Args:
            node_id: Node-ID
        """
        self._worker_nodes.pop(node_id, None)
        logger.info(f"Worker-Node entfernt: {node_id}")

    async def distribute_event(self, event: AuditEvent) -> str | None:
        """Verteilt Event an Worker-Node.

        Args:
            event: Zu verteilendes Event

        Returns:
            Node-ID oder None bei Fehler
        """
        # Wähle Worker-Node
        selected_node = self._select_worker_node(event)
        if not selected_node:
            logger.warning("Kein verfügbarer Worker-Node")
            return None

        try:
            # Sende Event an Node (vereinfacht)
            success = await self._send_event_to_node(selected_node, event)

            if success:
                selected_node.current_load += 1
                self._events_distributed += 1
                return selected_node.node_id
            self._node_failures += 1
            return None

        except Exception as e:
            logger.exception(f"Event-Distribution fehlgeschlagen: {e}")
            self._node_failures += 1
            return None

    def _select_worker_node(self, event: AuditEvent) -> WorkerNode | None:
        """Wählt Worker-Node basierend auf Strategie."""
        healthy_nodes = [node for node in self._worker_nodes.values() if node.is_healthy]

        if not healthy_nodes:
            return None

        with self._load_balancer_lock:
            if self.scaling_strategy == ScalingStrategy.ROUND_ROBIN:
                selected_node = healthy_nodes[self._node_selector_index % len(healthy_nodes)]
                self._node_selector_index += 1

            elif self.scaling_strategy == ScalingStrategy.LOAD_BASED:
                # Wähle Node mit niedrigster Auslastung
                selected_node = min(healthy_nodes, key=lambda n: n.utilization)

            elif self.scaling_strategy == ScalingStrategy.HASH_BASED:
                # Hash-basierte Auswahl für Konsistenz
                hash_value = hash(event.event_id) % len(healthy_nodes)
                selected_node = healthy_nodes[hash_value]

            elif self.scaling_strategy == ScalingStrategy.PRIORITY_BASED:
                # Wähle basierend auf Event-Severity
                if event.severity.value in ["critical", "high"]:
                    # Wähle besten verfügbaren Node
                    selected_node = min(healthy_nodes, key=lambda n: n.avg_processing_time_ms)
                else:
                    # Standard Load-Based
                    selected_node = min(healthy_nodes, key=lambda n: n.utilization)

            else:
                # Fallback: Wähle ersten verfügbaren Node
                selected_node = healthy_nodes[0]

        return selected_node

    async def _send_event_to_node(self, node: WorkerNode, event: AuditEvent) -> bool:
        """Sendet Event an Worker-Node."""
        # Vereinfachte Implementierung
        # In Produktion würde hier HTTP/gRPC/Message Queue verwendet

        try:
            # Simuliere Netzwerk-Latenz
            await asyncio.sleep(0.001)  # 1ms

            # Simuliere gelegentliche Fehler
            import random
            if random.random() < 0.05:  # 5% Fehlerrate
                raise Exception("Simulated network error")

            # Aktualisiere Node-Heartbeat
            node.last_heartbeat = datetime.now(UTC)

            return True

        except Exception as e:
            logger.exception(f"Event-Übertragung an {node.node_id} fehlgeschlagen: {e}")
            return False

    def update_node_metrics(self, node_id: str, processing_time_ms: float, success: bool) -> None:
        """Aktualisiert Node-Metriken.

        Args:
            node_id: Node-ID
            processing_time_ms: Verarbeitungszeit
            success: Erfolgreich
        """
        node = self._worker_nodes.get(node_id)
        if not node:
            return

        # Aktualisiere Metriken
        node.avg_processing_time_ms = (
            (node.avg_processing_time_ms * 0.9) + (processing_time_ms * 0.1)
        )

        if success:
            node.success_rate = (node.success_rate * 0.95) + (1.0 * 0.05)
        else:
            node.success_rate = (node.success_rate * 0.95) + (0.0 * 0.05)

        # Reduziere Load
        node.current_load = max(0, node.current_load - 1)

    def get_scaling_statistics(self) -> dict[str, Any]:
        """Gibt Skalierungs-Statistiken zurück."""
        healthy_nodes = sum(1 for node in self._worker_nodes.values() if node.is_healthy)
        total_capacity = sum(node.capacity for node in self._worker_nodes.values())
        total_load = sum(node.current_load for node in self._worker_nodes.values())

        return {
            "scaling_strategy": self.scaling_strategy.value,
            "total_nodes": len(self._worker_nodes),
            "healthy_nodes": healthy_nodes,
            "total_capacity": total_capacity,
            "total_load": total_load,
            "cluster_utilization": (total_load / total_capacity) if total_capacity > 0 else 0.0,
            "events_distributed": self._events_distributed,
            "node_failures": self._node_failures,
            "nodes": [node.to_dict() for node in self._worker_nodes.values()]
        }


class AuditPerformanceManager:
    """Hauptklasse für Audit-Performance-Management."""

    def __init__(self):
        """Initialisiert Audit Performance Manager."""
        # Performance-Komponenten
        self.async_processor = AsyncAuditProcessor()
        self.batch_processor = BatchProcessor(BatchConfig())
        self.event_streamer = AuditEventStreamer()
        self.horizontal_scaler = HorizontalScaler()

        # Performance-Monitoring
        self._metrics_history: deque = deque(maxlen=1000)
        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False

        # Konfiguration
        self.processing_mode = ProcessingMode.ASYNCHRONOUS
        self.enable_streaming = True
        self.enable_scaling = False

    async def start(self) -> None:
        """Startet Performance-Manager."""
        # Starte Komponenten
        await self.async_processor.start()
        await self.batch_processor.start()

        # Starte Monitoring
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._performance_monitoring_loop())

        logger.info("Audit Performance Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Performance-Manager."""
        self._is_monitoring = False

        # Stoppe Monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        # Stoppe Komponenten
        await self.async_processor.stop()
        await self.batch_processor.stop()

        logger.info("Audit Performance Manager gestoppt")

    @trace_function("audit_performance.process_event")
    async def process_event(self, event: AuditEvent) -> None:
        """Verarbeitet Event basierend auf Konfiguration.

        Args:
            event: Zu verarbeitendes Event
        """
        # Event-Streaming
        if self.enable_streaming:
            await self.event_streamer.stream_event(event)

        # Event-Processing
        if self.processing_mode == ProcessingMode.ASYNCHRONOUS:
            await self.async_processor.submit_event(event)
        elif self.processing_mode == ProcessingMode.BATCH:
            await self.batch_processor.add_event(event)
        elif self.processing_mode == ProcessingMode.STREAMING:
            # Direkte Verarbeitung für Streaming
            await self.async_processor.submit_event(event)

        # Horizontale Skalierung
        if self.enable_scaling:
            await self.horizontal_scaler.distribute_event(event)

    async def _performance_monitoring_loop(self) -> None:
        """Performance-Monitoring-Loop."""
        while self._is_monitoring:
            try:
                # Sammle Metriken
                metrics = await self._collect_performance_metrics()
                self._metrics_history.append(metrics)

                # Warte 10 Sekunden
                await asyncio.sleep(10.0)

            except Exception as e:
                logger.exception(f"Performance-Monitoring-Fehler: {e}")
                await asyncio.sleep(10.0)

    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Sammelt aktuelle Performance-Metriken."""
        # Hole Metriken von Komponenten
        async_metrics = self.async_processor.get_performance_metrics()
        batch_metrics = self.batch_processor.get_batch_statistics()
        self.event_streamer.get_stream_statistics()
        scaling_metrics = self.horizontal_scaler.get_scaling_statistics()

        # Berechne aggregierte Metriken
        events_per_second = async_metrics.get("events_processed", 0) / 10.0  # 10s Fenster

        return PerformanceMetrics(
            timestamp=datetime.now(UTC),
            events_per_second=events_per_second,
            batches_per_second=batch_metrics.get("batches_processed", 0) / 10.0,
            avg_processing_latency_ms=async_metrics.get("avg_processing_latency_ms", 0.0),
            p95_processing_latency_ms=async_metrics.get("p95_processing_latency_ms", 0.0),
            p99_processing_latency_ms=async_metrics.get("p99_processing_latency_ms", 0.0),
            queue_size=async_metrics.get("queue_size", 0),
            error_rate_percent=async_metrics.get("error_rate_percent", 0.0),
            active_workers=scaling_metrics.get("healthy_nodes", 0),
            worker_utilization=scaling_metrics.get("cluster_utilization", 0.0)
        )

    def get_performance_summary(self) -> dict[str, Any]:
        """Gibt Performance-Zusammenfassung zurück."""
        return {
            "processing_mode": self.processing_mode.value,
            "enable_streaming": self.enable_streaming,
            "enable_scaling": self.enable_scaling,
            "async_processor": self.async_processor.get_performance_metrics(),
            "batch_processor": self.batch_processor.get_batch_statistics(),
            "event_streamer": self.event_streamer.get_stream_statistics(),
            "horizontal_scaler": self.horizontal_scaler.get_scaling_statistics(),
            "metrics_history_size": len(self._metrics_history),
            "is_monitoring": self._is_monitoring
        }


# Globale Audit Performance Manager Instanz
audit_performance_manager = AuditPerformanceManager()
