# backend/agents/lifecycle/task_event_handler.py
"""Task- und Event-Handling für Agent-Lifecycle-Management.

Implementiert asynchrone Task-Queue, Event-Subscription, Retry-Logik
und Backpressure-Handling für Agent-Operationen.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger

from .models import (
    AgentEvent,
    AgentTask,
    BackpressureStrategy,
    EventSubscription,
    EventType,
    PriorityTask,
    TaskExecutionResult,
    TaskPriority,
    TaskQueueConfig,
    TaskStatus,
)

logger = get_logger(__name__)

# Konstanten
MILLISECONDS_PER_SECOND = 1000
TASK_LOOP_SLEEP_SECONDS = 0.1
FAST_TASK_LOOP_SLEEP_SECONDS = 0.01
ERROR_RECOVERY_SLEEP_SECONDS = 1.0
DEFAULT_EVENT_HISTORY_SIZE = 10000

# Zusätzliche Konstanten
DEFAULT_RETRY_DELAY_SECONDS = 5.0
MAX_EVENT_HISTORY_SIZE = 1000


class TaskQueue:
    """Asynchrone Task-Queue mit Prioritätsverwaltung."""

    def __init__(self, config: TaskQueueConfig) -> None:
        """Initialisiert Task-Queue."""
        self.config = config
        self._priority_queue: list[PriorityTask] = []
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._task_results: dict[str, TaskExecutionResult] = {}
        self._queue_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self._shutdown_event = asyncio.Event()

        # Metriken
        self._total_tasks_processed = 0
        self._total_tasks_failed = 0
        self._total_tasks_timeout = 0
        self._queue_size_peak = 0

    async def enqueue_task(self, task: AgentTask) -> bool:
        """Fügt Task zur Queue hinzu.

        Args:
            task: Hinzuzufügende Task

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        async with self._queue_lock:
            # Priorität berechnen
            priority = self._calculate_priority(task.priority)
            priority_task = PriorityTask(priority=priority, timestamp=time.time(), task=task)

            # Prüfe Queue-Größe und Backpressure
            if len(self._priority_queue) >= self.config.max_queue_size:
                if not await self._handle_backpressure(task):
                    logger.warning(
                        f"Task {task.task_id} abgelehnt aufgrund von Backpressure",
                        extra={
                            "task_id": task.task_id,
                            "queue_size": len(self._priority_queue),
                            "max_queue_size": self.config.max_queue_size,
                            "backpressure_strategy": self.config.backpressure_strategy.value,
                            "operation": "backpressure_rejection"
                        }
                    )
                    return False
                # Nach erfolgreichem Backpressure-Handling ist Platz frei

            # Task zur Priority-Queue hinzufügen
            heapq.heappush(self._priority_queue, priority_task)

            # Metriken aktualisieren
            self._queue_size_peak = max(self._queue_size_peak, len(self._priority_queue))

            logger.debug(
                f"Task {task.task_id} zur Queue hinzugefügt (Priorität: {priority})",
                extra={
                    "task_id": task.task_id,
                    "priority": priority,
                    "queue_size": len(self._priority_queue),
                    "agent_id": getattr(task, "agent_id", None),
                    "operation": "task_enqueue"
                }
            )
            return True

    async def _handle_backpressure(self, _new_task: AgentTask) -> bool:
        """Behandelt Backpressure-Situationen.

        Args:
            _new_task: Neue Task (aktuell nicht verwendet)

        Returns:
            True wenn Task hinzugefügt werden kann
        """
        strategy = self.config.backpressure_strategy

        if strategy.value == BackpressureStrategy.REJECT_NEW.value:
            return False

        if strategy.value == BackpressureStrategy.DROP_NEWEST.value:
            # Entferne neueste Task (höchste Timestamp)
            if self._priority_queue:
                removed = max(self._priority_queue, key=lambda pt: pt.timestamp)
                self._priority_queue.remove(removed)
                heapq.heapify(self._priority_queue)
                logger.info(
                    f"Task {removed.task.task_id} aufgrund Backpressure entfernt (DROP_NEWEST)"
                )
            return True

        if strategy.value == BackpressureStrategy.DROP_OLDEST.value:
            # Entferne älteste Task (niedrigste Timestamp)
            if self._priority_queue:
                removed = min(self._priority_queue, key=lambda pt: pt.timestamp)
                self._priority_queue.remove(removed)
                heapq.heapify(self._priority_queue)
                logger.info(
                    f"Task {removed.task.task_id} aufgrund Backpressure entfernt (DROP_OLDEST)"
                )
            return True

        if strategy == BackpressureStrategy.QUEUE_UNLIMITED:
            return True

        return False

    def _calculate_priority(self, task_priority: TaskPriority) -> int:
        """Berechnet numerische Priorität für Task.

        Args:
            task_priority: Task-Priorität

        Returns:
            Numerische Priorität (niedrigere Werte = höhere Priorität)
        """
        priority_mapping = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 10,
            TaskPriority.NORMAL: 20,
            TaskPriority.LOW: 30,
        }
        return priority_mapping.get(task_priority, 20)

    async def get_next_task(self) -> AgentTask | None:
        """Holt nächste Task aus der Queue.

        Returns:
            Nächste Task oder None wenn Queue leer
        """
        async with self._queue_lock:
            if not self._priority_queue:
                return None

            priority_task = heapq.heappop(self._priority_queue)
            return priority_task.task

    async def start_processing(
        self, task_handler: Callable[[AgentTask], Awaitable[TaskExecutionResult]]
    ) -> None:
        """Startet Task-Processing-Loop.

        Args:
            task_handler: Handler-Funktion für Tasks
        """
        logger.info("Starte Task-Processing-Loop")

        while not self._shutdown_event.is_set():
            try:
                # Warte auf verfügbaren Slot
                await self._semaphore.acquire()

                # Hole nächste Task
                task = await self.get_next_task()
                if not task:
                    self._semaphore.release()
                    await asyncio.sleep(TASK_LOOP_SLEEP_SECONDS)
                    continue

                # Starte Task-Ausführung
                execution_task = asyncio.create_task(
                    self._execute_task_with_timeout(task, task_handler)
                )
                self._running_tasks[task.task_id] = execution_task

                # Cleanup nach Completion
                execution_task.add_done_callback(
                    lambda t, task_id=task.task_id: self._cleanup_completed_task(task_id)
                )

            except asyncio.CancelledError:
                break
            except Exception as processing_error:
                logger.error(
                    f"Task-Processing-Loop-Fehler: {processing_error}",
                    extra={
                        "error_type": type(processing_error).__name__,
                        "operation": "task_processing_loop",
                        "queue_size": len(self._priority_queue),
                        "running_tasks": len(self._running_tasks)
                    }
                )
                self._semaphore.release()
                await asyncio.sleep(ERROR_RECOVERY_SLEEP_SECONDS)

    async def _execute_task_with_timeout(
        self, task: AgentTask, task_handler: Callable[[AgentTask], Awaitable[TaskExecutionResult]]
    ) -> TaskExecutionResult:
        """Führt Task mit Timeout aus.

        Args:
            task: Auszuführende Task
            task_handler: Handler-Funktion

        Returns:
            Task-Ausführungsergebnis
        """
        start_time = time.time()
        task.started_at = datetime.now(UTC)

        try:
            _result = await asyncio.wait_for(task_handler(task), timeout=task.timeout_seconds)
            return self._handle_successful_task(task, _result, start_time)

        except TimeoutError:
            return self._handle_timeout_task(task, start_time)

        except Exception as execution_error:
            return await self._handle_failed_task(task, execution_error, start_time)

        finally:
            self._semaphore.release()

    def _handle_successful_task(
        self, task: AgentTask, result: TaskExecutionResult, start_time: float
    ) -> TaskExecutionResult:
        """Behandelt erfolgreich ausgeführte Task."""
        execution_time_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
        result.execution_time = execution_time_ms / MILLISECONDS_PER_SECOND

        task.completed_at = datetime.now(UTC)
        self._total_tasks_processed += 1

        logger.debug(f"Task {task.task_id} erfolgreich ausgeführt in {execution_time_ms:.1f}ms")
        return result

    def _handle_timeout_task(self, task: AgentTask, start_time: float) -> TaskExecutionResult:
        """Behandelt Task-Timeout."""
        execution_time_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
        task.completed_at = datetime.now(UTC)
        task.error = f"Task timeout after {task.timeout_seconds}s"

        self._total_tasks_timeout += 1

        logger.warning(f"Task {task.task_id} Timeout nach {task.timeout_seconds}s")

        return TaskExecutionResult(
            task_id=task.task_id,
            status=TaskStatus.TIMEOUT,
            error=task.error,
            execution_time=execution_time_ms / MILLISECONDS_PER_SECOND,
        )

    async def _handle_failed_task(
        self, task: AgentTask, error: Exception, start_time: float
    ) -> TaskExecutionResult:
        """Behandelt fehlgeschlagene Task."""
        execution_time_ms = (time.time() - start_time) * MILLISECONDS_PER_SECOND
        task.completed_at = datetime.now(UTC)
        task.error = str(error)

        self._total_tasks_failed += 1

        logger.error(
            f"Task {task.task_id} fehlgeschlagen: {error}",
            extra={
                "task_id": task.task_id,
                "agent_id": getattr(task, "agent_id", None),
                "error_type": type(error).__name__,
                "execution_time_ms": execution_time_ms,
                "retry_count": task.retry_count,
                "can_retry": task.can_retry,
                "operation": "task_execution"
            }
        )

        # Retry-Logik
        if task.can_retry:
            await self._schedule_retry(task)

        return TaskExecutionResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=str(error),
            execution_time=execution_time_ms / MILLISECONDS_PER_SECOND,
        )

    async def _schedule_retry(self, task: AgentTask) -> None:
        """Plant Task-Retry.

        Args:
            task: Zu wiederholende Task
        """
        task.retry_count += 1

        # Exponential Backoff für Retry-Delay
        delay = min(
            self.config.retry_delay_seconds * (2 ** (task.retry_count - 1)),
            self.config.max_retry_delay_seconds,
        )

        logger.info(
            f"Plane Retry für Task {task.task_id} in {delay}s (Versuch {task.retry_count}/{task.max_retries})",
            extra={
                "task_id": task.task_id,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "delay_seconds": delay,
                "agent_id": getattr(task, "agent_id", None),
                "operation": "retry_scheduling"
            }
        )

        # Schedule Retry
        asyncio.create_task(self._delayed_retry(task, delay))

    async def _delayed_retry(self, task: AgentTask, delay: float) -> None:
        """Führt verzögerten Retry durch.

        Args:
            task: Task für Retry
            delay: Verzögerung in Sekunden
        """
        await asyncio.sleep(delay)

        # Reset Task-State für Retry
        task.started_at = None
        task.completed_at = None
        task.error = None

        # Task wieder zur Queue hinzufügen
        await self.enqueue_task(task)

    def _cleanup_completed_task(self, task_id: str) -> None:
        """Bereinigt abgeschlossene Task.

        Args:
            task_id: Task-ID
        """
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]

    async def stop(self) -> None:
        """Stoppt Task-Queue."""
        logger.info("Stoppe Task-Queue")

        self._shutdown_event.set()

        # Warte auf laufende Tasks
        if self._running_tasks:
            logger.info(f"Warte auf {len(self._running_tasks)} laufende Tasks")
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)

    def get_queue_stats(self) -> dict[str, Any]:
        """Gibt Queue-Statistiken zurück.

        Returns:
            Dictionary mit Statistiken
        """
        return {
            "queue_size": len(self._priority_queue),
            "running_tasks": len(self._running_tasks),
            "total_processed": self._total_tasks_processed,
            "total_tasks_processed": self._total_tasks_processed,  # Alias für Test-Kompatibilität
            "total_failed": self._total_tasks_failed,
            "total_tasks_failed": self._total_tasks_failed,  # Alias für Test-Kompatibilität
            "total_timeout": self._total_tasks_timeout,
            "queue_size_peak": self._queue_size_peak,
            "success_rate": (
                (self._total_tasks_processed - self._total_tasks_failed)
                / max(self._total_tasks_processed, 1)
            ),
        }

    async def add_task(self, task: AgentTask) -> bool:
        """Alias für enqueue_task."""
        return await self.enqueue_task(task)



    async def stop_processing(self) -> None:
        """Stoppt Task-Processing."""
        self._shutdown_event.set()

    def get_metrics(self) -> dict[str, Any]:
        """Alias für get_queue_stats."""
        return self.get_queue_stats()

    def get_queue_lock(self) -> asyncio.Lock:
        """Public API für Queue-Lock Zugriff.

        Returns:
            Queue-Lock für externe Synchronisation
        """
        return self._queue_lock

    def get_running_tasks_info(self) -> dict[str, Any]:
        """Public API für laufende Tasks Informationen.

        Returns:
            Dictionary mit Informationen über laufende Tasks
        """
        return {
            "running_task_ids": list(self._running_tasks.keys()),
            "running_task_count": len(self._running_tasks)
        }

    def remove_tasks_by_agent(self, agent_id: str) -> int:
        """Public API für Agent-spezifische Task-Entfernung.

        Args:
            agent_id: Agent-ID für die Tasks entfernt werden sollen

        Returns:
            Anzahl entfernter Tasks
        """
        removed_count = 0

        # Entferne aus Priority Queue
        original_queue = self._priority_queue[:]
        self._priority_queue = [
            pt for pt in self._priority_queue
            if not (hasattr(pt.task, "agent_id") and pt.task.agent_id == agent_id)
        ]
        removed_count += len(original_queue) - len(self._priority_queue)

        # Re-heapify nach Änderung
        import heapq
        heapq.heapify(self._priority_queue)

        return removed_count

    def get_running_tasks_by_agent(self, agent_id: str) -> list[str]:
        """Public API für Agent-spezifische laufende Tasks.

        Args:
            agent_id: Agent-ID

        Returns:
            Liste der Task-IDs für laufende Tasks des Agents
        """
        # Da _running_tasks asyncio.Task Objekte enthält, können wir nur Task-IDs zurückgeben
        # Die Agent-ID-Filterung müsste über eine separate Datenstruktur erfolgen
        return [
            task_id for task_id in self._running_tasks.keys()
            if agent_id in task_id  # Einfache Heuristik, falls Task-ID Agent-ID enthält
        ]


class EventBus:
    """Event-Bus für Agent-Event-Handling."""

    def __init__(self) -> None:
        """Initialisiert Event-Bus."""
        self._subscriptions: dict[str, EventSubscription] = {}
        self._event_history: list[AgentEvent] = []
        self._max_history_size = DEFAULT_EVENT_HISTORY_SIZE
        self._publish_lock = asyncio.Lock()

    def subscribe(
        self,
        subscription_id: str,
        event_types: set[EventType],
        callback: Callable[[AgentEvent], Awaitable[None]],
        agent_id_filter: str | None = None,
    ) -> bool:
        """Abonniert Events.

        Args:
            subscription_id: Eindeutige Subscription-ID
            event_types: Set von Event-Typen
            callback: Callback-Funktion
            agent_id_filter: Optional Agent-ID-Filter

        Returns:
            True wenn erfolgreich abonniert
        """
        if subscription_id in self._subscriptions:
            logger.warning(f"Subscription {subscription_id} bereits vorhanden")
            return False

        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_types=event_types,
            agent_id_filter=agent_id_filter,
            callback=callback,
        )

        self._subscriptions[subscription_id] = subscription
        logger.info(
            f"Event-Subscription {subscription_id} erstellt für {len(event_types)} Event-Typen"
        )
        return True

    def unsubscribe(self, subscription_id: str) -> bool:
        """Beendet Event-Subscription.

        Args:
            subscription_id: Subscription-ID

        Returns:
            True wenn erfolgreich beendet
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"Event-Subscription {subscription_id} beendet")
            return True
        return False

    async def publish(self, event: AgentEvent) -> None:
        """Publiziert Event an Subscribers.

        Args:
            event: Zu publizierendes Event
        """
        async with self._publish_lock:
            # Event zur Historie hinzufügen
            self._add_to_history(event)

            # Event an passende Subscriptions weiterleiten
            for subscription in self._subscriptions.values():
                if not subscription.active:
                    continue

                # Prüfe Event-Typ-Filter
                if event.event_type not in subscription.event_types:
                    continue

                # Prüfe Agent-ID-Filter
                if subscription.agent_id_filter and subscription.agent_id_filter != event.agent_id:
                    continue

                # Rufe Callback auf
                try:
                    if subscription.callback is not None:
                        await subscription.callback(event)
                except Exception as callback_error:
                    logger.error(
                        f"Event-Callback-Fehler für Subscription {subscription.subscription_id}: {callback_error}",
                        extra={
                            "subscription_id": subscription.subscription_id,
                            "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                            "agent_id": event.agent_id,
                            "error_type": type(callback_error).__name__,
                            "operation": "event_callback"
                        }
                    )

    def get_event_history(
        self,
        agent_id: str | None = None,
        event_types: set[EventType] | None = None,
        limit: int = 100,
    ) -> list[AgentEvent]:
        """Gibt Event-Historie zurück.

        Args:
            agent_id: Optional Agent-ID-Filter
            event_types: Optional Event-Typ-Filter
            limit: Maximale Anzahl Events

        Returns:
            Liste von Events
        """
        filtered_events = []

        for event in reversed(self._event_history):
            # Anwenden der Filter
            if agent_id and event.agent_id != agent_id:
                continue

            if event_types and event.event_type not in event_types:
                continue

            filtered_events.append(event)

            if len(filtered_events) >= limit:
                break

        return filtered_events

    def get_subscription_stats(self) -> dict[str, Any]:
        """Gibt Subscription-Statistiken zurück.

        Returns:
            Dictionary mit Statistiken
        """
        active_subscriptions = sum(1 for s in self._subscriptions.values() if s.active)

        return {
            "total_subscriptions": len(self._subscriptions),
            "active_subscriptions": active_subscriptions,
            "event_history_size": len(self._event_history),
            "max_history_size": self._max_history_size,
        }

    def _add_to_history(self, event: AgentEvent) -> None:
        """Fügt Event zur Historie hinzu."""
        self._event_history.insert(0, event)  # Neueste Events zuerst

        # Begrenze History-Größe
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[:self._max_history_size]

        logger.debug(
            f"Event zur Historie hinzugefügt: {event.event_type} für Agent {event.agent_id}",
            extra={
                "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                "agent_id": event.agent_id,
                "event_id": getattr(event, "event_id", None),
                "history_size": len(self._event_history),
                "max_history_size": self._max_history_size,
                "operation": "event_history_add"
            }
        )
