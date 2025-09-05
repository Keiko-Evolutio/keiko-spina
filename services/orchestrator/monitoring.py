# backend/services/orchestrator/monitoring.py
"""Real-time Monitoring für Orchestrator Service.

Implementiert Real-time Task-Monitoring, Progress-Tracking,
Event-Streaming und Performance-Metriken.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from services.messaging import BusService, get_bus_service

from .data_models import OrchestrationEvent, OrchestrationProgress, SubtaskExecution

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class OrchestrationMonitor:
    """Real-time Monitoring für Orchestrations."""

    def __init__(self, bus_service: BusService | None = None):
        """Initialisiert Orchestration Monitor.

        Args:
            bus_service: Message Bus für Event-Streaming
        """
        self.bus_service = bus_service or get_bus_service()

        # Event-Subscribers
        self.event_subscribers: dict[str, list[Callable]] = {}
        self.progress_subscribers: dict[str, list[Callable]] = {}

        # Event-Storage für Replay
        self.event_history: dict[str, list[OrchestrationEvent]] = {}
        self.max_events_per_orchestration = 1000

        # Performance-Metriken
        self.orchestration_metrics: dict[str, dict[str, Any]] = {}

        # Real-time Connections
        self.websocket_connections: set[Any] = set()

        # Background-Tasks
        self._monitoring_tasks: list[asyncio.Task] = []
        self._is_running = False

        logger.info("Orchestration Monitor initialisiert")

    async def start(self) -> None:
        """Startet Monitoring-System."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._metrics_aggregation_loop()),
            asyncio.create_task(self._event_cleanup_loop())
        ]

        logger.info("Orchestration Monitor gestartet")

    async def stop(self) -> None:
        """Stoppt Monitoring-System."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        self._monitoring_tasks.clear()

        logger.info("Orchestration Monitor gestoppt")

    async def track_orchestration_started(
        self,
        orchestration_id: str,
        request_data: dict[str, Any]
    ) -> None:
        """Trackt Start einer Orchestration."""
        event = OrchestrationEvent(
            event_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            event_type="orchestration_started",
            event_data={
                "task_id": request_data.get("task_id"),
                "task_type": request_data.get("task_type"),
                "execution_mode": request_data.get("execution_mode"),
                "max_parallel_tasks": request_data.get("max_parallel_tasks")
            },
            user_id=request_data.get("user_id"),
            session_id=request_data.get("session_id"),
            correlation_id=request_data.get("correlation_id")
        )

        await self._emit_event(event)

        # Initialisiere Metriken
        self.orchestration_metrics[orchestration_id] = {
            "start_time": datetime.utcnow(),
            "events_count": 1,
            "subtasks_total": 0,
            "subtasks_completed": 0,
            "subtasks_failed": 0,
            "agents_used": set(),
            "performance_data": []
        }

    async def track_subtask_started(
        self,
        orchestration_id: str,
        subtask: SubtaskExecution
    ) -> None:
        """Trackt Start eines Subtasks."""
        event = OrchestrationEvent(
            event_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            event_type="subtask_started",
            subtask_id=subtask.subtask_id,
            agent_id=subtask.assigned_agent_id,
            event_data={
                "subtask_name": subtask.name,
                "task_type": subtask.task_type.value,
                "priority": subtask.priority.value,
                "predicted_duration_ms": subtask.predicted_execution_time_ms,
                "prediction_confidence": subtask.prediction_confidence
            }
        )

        await self._emit_event(event)

        # Update Metriken
        if orchestration_id in self.orchestration_metrics:
            metrics = self.orchestration_metrics[orchestration_id]
            metrics["subtasks_total"] += 1
            if subtask.assigned_agent_id:
                metrics["agents_used"].add(subtask.assigned_agent_id)

    async def track_subtask_completed(
        self,
        orchestration_id: str,
        subtask: SubtaskExecution
    ) -> None:
        """Trackt Completion eines Subtasks."""
        event = OrchestrationEvent(
            event_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            event_type="subtask_completed",
            subtask_id=subtask.subtask_id,
            agent_id=subtask.assigned_agent_id,
            event_data={
                "execution_time_ms": subtask.execution_time_ms,
                "predicted_time_ms": subtask.predicted_execution_time_ms,
                "performance_variance": subtask.performance_variance,
                "retry_count": subtask.retry_count,
                "result_size": len(str(subtask.result)) if subtask.result else 0
            }
        )

        await self._emit_event(event)

        # Update Metriken
        if orchestration_id in self.orchestration_metrics:
            metrics = self.orchestration_metrics[orchestration_id]
            metrics["subtasks_completed"] += 1

            # Performance-Daten sammeln
            if subtask.execution_time_ms and subtask.predicted_execution_time_ms:
                variance = abs(subtask.execution_time_ms - subtask.predicted_execution_time_ms) / subtask.predicted_execution_time_ms
                metrics["performance_data"].append({
                    "subtask_id": subtask.subtask_id,
                    "actual_time": subtask.execution_time_ms,
                    "predicted_time": subtask.predicted_execution_time_ms,
                    "variance": variance,
                    "agent_id": subtask.assigned_agent_id
                })

    async def track_subtask_failed(
        self,
        orchestration_id: str,
        subtask: SubtaskExecution,
        error_message: str
    ) -> None:
        """Trackt Failure eines Subtasks."""
        event = OrchestrationEvent(
            event_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            event_type="subtask_failed",
            subtask_id=subtask.subtask_id,
            agent_id=subtask.assigned_agent_id,
            event_data={
                "error_message": error_message,
                "retry_count": subtask.retry_count,
                "max_retries": subtask.max_retries,
                "execution_time_ms": subtask.execution_time_ms,
                "will_retry": subtask.retry_count < subtask.max_retries
            }
        )

        await self._emit_event(event)

        # Update Metriken
        if orchestration_id in self.orchestration_metrics:
            self.orchestration_metrics[orchestration_id]["subtasks_failed"] += 1

    async def track_orchestration_completed(
        self,
        orchestration_id: str,
        success: bool,
        total_execution_time_ms: float,
        orchestration_overhead_ms: float
    ) -> None:
        """Trackt Completion einer Orchestration."""
        event = OrchestrationEvent(
            event_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            event_type="orchestration_completed",
            event_data={
                "success": success,
                "total_execution_time_ms": total_execution_time_ms,
                "orchestration_overhead_ms": orchestration_overhead_ms,
                "overhead_percentage": (orchestration_overhead_ms / total_execution_time_ms) * 100 if total_execution_time_ms > 0 else 0
            }
        )

        await self._emit_event(event)

        # Finalisiere Metriken
        if orchestration_id in self.orchestration_metrics:
            metrics = self.orchestration_metrics[orchestration_id]
            metrics["end_time"] = datetime.utcnow()
            metrics["total_duration_ms"] = total_execution_time_ms
            metrics["overhead_ms"] = orchestration_overhead_ms
            metrics["success"] = success

    async def track_progress_update(
        self,
        orchestration_id: str,
        progress: OrchestrationProgress
    ) -> None:
        """Trackt Progress-Update."""
        event = OrchestrationEvent(
            event_id=str(uuid.uuid4()),
            orchestration_id=orchestration_id,
            event_type="progress_update",
            event_data={
                "state": progress.state.value,
                "completion_percentage": progress.completion_percentage,
                "total_subtasks": progress.total_subtasks,
                "completed_subtasks": progress.completed_subtasks,
                "failed_subtasks": progress.failed_subtasks,
                "running_subtasks": progress.running_subtasks,
                "pending_subtasks": progress.pending_subtasks,
                "execution_efficiency": progress.execution_efficiency,
                "resource_utilization": progress.resource_utilization
            }
        )

        await self._emit_event(event)

        # Notify Progress-Subscribers
        await self._notify_progress_subscribers(orchestration_id, progress)

    async def subscribe_to_events(
        self,
        orchestration_id: str,
        callback: Callable[[OrchestrationEvent], None]
    ) -> None:
        """Abonniert Events für Orchestration."""
        if orchestration_id not in self.event_subscribers:
            self.event_subscribers[orchestration_id] = []

        self.event_subscribers[orchestration_id].append(callback)

    async def subscribe_to_progress(
        self,
        orchestration_id: str,
        callback: Callable[[OrchestrationProgress], None]
    ) -> None:
        """Abonniert Progress-Updates für Orchestration."""
        if orchestration_id not in self.progress_subscribers:
            self.progress_subscribers[orchestration_id] = []

        self.progress_subscribers[orchestration_id].append(callback)

    async def get_orchestration_events(
        self,
        orchestration_id: str,
        event_types: list[str] | None = None,
        limit: int | None = None
    ) -> list[OrchestrationEvent]:
        """Holt Events für Orchestration."""
        events = self.event_history.get(orchestration_id, [])

        # Filter nach Event-Types
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        # Limitiere Anzahl
        if limit:
            events = events[-limit:]

        return events

    async def get_orchestration_metrics(self, orchestration_id: str) -> dict[str, Any] | None:
        """Holt Metriken für Orchestration."""
        return self.orchestration_metrics.get(orchestration_id)

    async def get_real_time_stats(self) -> dict[str, Any]:
        """Holt Real-time Statistiken."""
        active_orchestrations = len([
            m for m in self.orchestration_metrics.values()
            if "end_time" not in m
        ])

        completed_orchestrations = len([
            m for m in self.orchestration_metrics.values()
            if "end_time" in m
        ])

        # Durchschnittliche Performance
        completed_metrics = [
            m for m in self.orchestration_metrics.values()
            if "total_duration_ms" in m
        ]

        avg_duration = (
            sum(m["total_duration_ms"] for m in completed_metrics) / len(completed_metrics)
            if completed_metrics else 0.0
        )

        avg_overhead = (
            sum(m["overhead_ms"] for m in completed_metrics) / len(completed_metrics)
            if completed_metrics else 0.0
        )

        success_rate = (
            sum(1 for m in completed_metrics if m.get("success", False)) / len(completed_metrics)
            if completed_metrics else 0.0
        )

        return {
            "active_orchestrations": active_orchestrations,
            "completed_orchestrations": completed_orchestrations,
            "avg_duration_ms": avg_duration,
            "avg_overhead_ms": avg_overhead,
            "success_rate": success_rate,
            "total_events": sum(len(events) for events in self.event_history.values()),
            "websocket_connections": len(self.websocket_connections)
        }

    async def _emit_event(self, event: OrchestrationEvent) -> None:
        """Emittiert Event an alle Subscriber."""
        # Speichere in History
        if event.orchestration_id not in self.event_history:
            self.event_history[event.orchestration_id] = []

        self.event_history[event.orchestration_id].append(event)

        # Limitiere History-Größe
        if len(self.event_history[event.orchestration_id]) > self.max_events_per_orchestration:
            self.event_history[event.orchestration_id] = self.event_history[event.orchestration_id][-self.max_events_per_orchestration:]

        # Notify Event-Subscribers
        await self._notify_event_subscribers(event)

        # Publish über Message Bus
        await self._publish_event_to_bus(event)

        # Notify WebSocket-Connections
        await self._notify_websocket_connections(event)

    async def _notify_event_subscribers(self, event: OrchestrationEvent) -> None:
        """Benachrichtigt Event-Subscribers."""
        subscribers = self.event_subscribers.get(event.orchestration_id, [])

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.exception(f"Event-Subscriber-Fehler: {e}")

    async def _notify_progress_subscribers(
        self,
        orchestration_id: str,
        progress: OrchestrationProgress
    ) -> None:
        """Benachrichtigt Progress-Subscribers."""
        subscribers = self.progress_subscribers.get(orchestration_id, [])

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.exception(f"Progress-Subscriber-Fehler: {e}")

    async def _publish_event_to_bus(self, event: OrchestrationEvent) -> None:
        """Publiziert Event über Message Bus."""
        try:
            if self.bus_service:
                subject = f"kei.orchestrator.events.{event.event_type}.v1"

                await self.bus_service.publish(
                    subject=subject,
                    data={
                        "event_id": event.event_id,
                        "orchestration_id": event.orchestration_id,
                        "event_type": event.event_type,
                        "subtask_id": event.subtask_id,
                        "agent_id": event.agent_id,
                        "event_data": event.event_data,
                        "timestamp": event.timestamp.isoformat(),
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "correlation_id": event.correlation_id
                    }
                )
        except Exception as e:
            logger.exception(f"Event-Publishing fehlgeschlagen: {e}")

    async def _notify_websocket_connections(self, event: OrchestrationEvent) -> None:
        """Benachrichtigt WebSocket-Connections."""
        if not self.websocket_connections:
            return

        message = {
            "type": "orchestration_event",
            "event": {
                "event_id": event.event_id,
                "orchestration_id": event.orchestration_id,
                "event_type": event.event_type,
                "subtask_id": event.subtask_id,
                "agent_id": event.agent_id,
                "event_data": event.event_data,
                "timestamp": event.timestamp.isoformat()
            }
        }

        # Sende an alle aktiven Connections
        disconnected_connections = set()

        for connection in self.websocket_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                # Connection ist disconnected
                disconnected_connections.add(connection)

        # Entferne disconnected Connections
        self.websocket_connections -= disconnected_connections

    async def add_websocket_connection(self, websocket) -> None:
        """Fügt WebSocket-Connection hinzu."""
        self.websocket_connections.add(websocket)

    async def remove_websocket_connection(self, websocket) -> None:
        """Entfernt WebSocket-Connection."""
        self.websocket_connections.discard(websocket)

    async def _metrics_aggregation_loop(self) -> None:
        """Background-Loop für Metriken-Aggregation."""
        while self._is_running:
            try:
                await self._aggregate_performance_metrics()
                await asyncio.sleep(60)  # Aggregiere alle 60 Sekunden
            except Exception as e:
                logger.exception(f"Metriken-Aggregation-Fehler: {e}")
                await asyncio.sleep(60)

    async def _aggregate_performance_metrics(self) -> None:
        """Aggregiert Performance-Metriken."""
        # TODO: Implementiere erweiterte Metriken-Aggregation - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/117
        # - Agent-Performance-Trends
        # - Prediction-Accuracy-Tracking
        # - Resource-Utilization-Patterns

    async def _event_cleanup_loop(self) -> None:
        """Background-Loop für Event-Cleanup."""
        while self._is_running:
            try:
                # Entferne alte Events (älter als 24 Stunden)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                for orchestration_id in list(self.event_history.keys()):
                    events = self.event_history[orchestration_id]

                    # Filtere Events
                    recent_events = [
                        e for e in events
                        if e.timestamp > cutoff_time
                    ]

                    if recent_events:
                        self.event_history[orchestration_id] = recent_events
                    else:
                        # Keine Events mehr - entferne Orchestration
                        del self.event_history[orchestration_id]
                        self.orchestration_metrics.pop(orchestration_id, None)

                await asyncio.sleep(3600)  # Cleanup alle Stunde

            except Exception as e:
                logger.exception(f"Event-Cleanup-Fehler: {e}")
                await asyncio.sleep(3600)
