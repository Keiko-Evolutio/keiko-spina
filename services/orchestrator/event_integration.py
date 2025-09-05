# backend/services/orchestrator/event_integration.py
"""Event Bus Integration für Orchestrator Service.

Implementiert asynchrone Event-driven Kommunikation für Orchestrator
mit Platform Event Bus und NATS JetStream Integration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kei_logging import get_logger
from messaging import PlatformEvent, PlatformEventBus
from messaging.platform_event_bus import PlatformEventBusConfig
from messaging.platform_nats_client import PlatformNATSConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..enhanced_real_time_monitoring.data_models import SagaTransaction
    from ..task_decomposition.data_models import DecompositionPlan
    from .data_models import OrchestrationRequest, OrchestrationResult

logger = get_logger(__name__)


class OrchestratorEventTypes:
    """Event-Typen für Orchestrator Service."""

    # Orchestration Lifecycle Events
    ORCHESTRATION_STARTED = "orchestrator.orchestration.started"
    ORCHESTRATION_COMPLETED = "orchestrator.orchestration.completed"
    ORCHESTRATION_FAILED = "orchestrator.orchestration.failed"
    ORCHESTRATION_CANCELLED = "orchestrator.orchestration.cancelled"

    # Plan Events
    PLAN_CREATED = "orchestrator.plan.created"
    PLAN_VALIDATED = "orchestrator.plan.validated"
    PLAN_PERSISTED = "orchestrator.plan.persisted"
    PLAN_RECOVERED = "orchestrator.plan.recovered"

    # Subtask Events
    SUBTASK_ASSIGNED = "orchestrator.subtask.assigned"
    SUBTASK_STARTED = "orchestrator.subtask.started"
    SUBTASK_COMPLETED = "orchestrator.subtask.completed"
    SUBTASK_FAILED = "orchestrator.subtask.failed"

    # Saga Events
    SAGA_STARTED = "orchestrator.saga.started"
    SAGA_STEP_COMPLETED = "orchestrator.saga.step.completed"
    SAGA_COMPENSATION_STARTED = "orchestrator.saga.compensation.started"
    SAGA_COMPLETED = "orchestrator.saga.completed"

    # Agent Events
    AGENT_SELECTED = "orchestrator.agent.selected"
    AGENT_LOAD_UPDATED = "orchestrator.agent.load.updated"
    AGENT_PERFORMANCE_UPDATED = "orchestrator.agent.performance.updated"

    # System Events
    SERVICE_STARTED = "orchestrator.service.started"
    SERVICE_STOPPED = "orchestrator.service.stopped"
    HEALTH_CHECK_COMPLETED = "orchestrator.health.check.completed"


class OrchestratorEventIntegration:
    """Event Bus Integration für Orchestrator Service.

    Features:
    - Asynchrone Event-Publikation für Orchestration-Lifecycle
    - Event-Subscription für externe System-Integration
    - Real-time Progress-Updates über Events
    - Saga-Event-Coordination
    - Agent-Performance-Events
    """

    def __init__(
        self,
        event_bus: PlatformEventBus | None = None,
        service_name: str = "orchestrator-service"
    ):
        """Initialisiert Event Integration.

        Args:
            event_bus: Platform Event Bus Instance
            service_name: Name des Services für Event-Metadaten
        """
        self.event_bus = event_bus
        self.service_name = service_name

        # Event Handlers
        self.event_handlers: dict[str, list[Callable]] = {}

        # Metriken
        self.events_published = 0
        self.events_consumed = 0
        self.events_failed = 0

        # Status
        self.started = False

        logger.info("Orchestrator Event Integration initialisiert")

    async def start(self) -> None:
        """Startet Event Integration."""
        if self.started:
            return

        try:
            # Event Bus erstellen falls nicht vorhanden
            if not self.event_bus:
                self.event_bus = await self._create_default_event_bus()

            # Event Bus starten
            if not self.event_bus.started:
                await self.event_bus.start()

            # Standard Event-Subscriptions registrieren
            await self._register_default_subscriptions()

            self.started = True
            logger.info("Orchestrator Event Integration gestartet")

        except Exception as e:
            logger.exception(f"Event Integration Start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Event Integration."""
        if not self.started:
            return

        try:
            if self.event_bus:
                await self.event_bus.stop()

            self.started = False
            logger.info("Orchestrator Event Integration gestoppt")

        except Exception as e:
            logger.exception(f"Event Integration Stop fehlgeschlagen: {e}")

    # =========================================================================
    # Event Publishing
    # =========================================================================

    async def publish_orchestration_started(
        self,
        orchestration_id: str,
        request: OrchestrationRequest
    ) -> bool:
        """Publiziert Orchestration-Started Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.ORCHESTRATION_STARTED,
            data={
                "orchestration_id": orchestration_id,
                "task_id": request.task_id,
                "task_type": request.task_type.value,
                "execution_mode": request.execution_mode.value,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "correlation_id": request.correlation_id,
                "tenant_id": request.tenant_id,
                "max_parallel_tasks": request.max_parallel_tasks,
                "started_at": datetime.utcnow().isoformat()
            },
            correlation_id=request.correlation_id
        )

    async def publish_orchestration_completed(
        self,
        orchestration_id: str,
        result: OrchestrationResult
    ) -> bool:
        """Publiziert Orchestration-Completed Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.ORCHESTRATION_COMPLETED,
            data={
                "orchestration_id": orchestration_id,
                "success": result.success,
                "subtasks_count": len(result.subtask_results),
                "total_execution_time_ms": result.total_execution_time_ms,
                "orchestration_overhead_ms": result.orchestration_overhead_ms,
                "parallelization_achieved": result.parallelization_achieved,
                "completed_at": datetime.utcnow().isoformat()
            },
            correlation_id=orchestration_id
        )

    async def publish_plan_created(
        self,
        plan: DecompositionPlan,
        orchestration_id: str
    ) -> bool:
        """Publiziert Plan-Created Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.PLAN_CREATED,
            data={
                "plan_id": plan.plan_id,
                "orchestration_id": orchestration_id,
                "subtasks_count": len(plan.subtasks),
                "execution_strategy": plan.execution_strategy.value,
                "estimated_total_duration_minutes": plan.estimated_total_duration_minutes,
                "estimated_parallel_duration_minutes": plan.estimated_parallel_duration_minutes,
                "parallelization_efficiency": plan.parallelization_efficiency,
                "plan_confidence": plan.plan_confidence,
                "created_at": plan.created_at.isoformat()
            },
            correlation_id=orchestration_id
        )

    async def publish_subtask_assigned(
        self,
        subtask_id: str,
        agent_id: str,
        orchestration_id: str,
        estimated_duration_minutes: float
    ) -> bool:
        """Publiziert Subtask-Assigned Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.SUBTASK_ASSIGNED,
            data={
                "subtask_id": subtask_id,
                "agent_id": agent_id,
                "orchestration_id": orchestration_id,
                "estimated_duration_minutes": estimated_duration_minutes,
                "assigned_at": datetime.utcnow().isoformat()
            },
            correlation_id=orchestration_id
        )

    async def publish_saga_started(
        self,
        saga: SagaTransaction,
        orchestration_id: str
    ) -> bool:
        """Publiziert Saga-Started Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.SAGA_STARTED,
            data={
                "saga_id": saga.saga_id,
                "orchestration_id": orchestration_id,
                "steps_count": len(saga.steps),
                "compensation_strategy": saga.compensation_strategy,
                "started_at": saga.started_at.isoformat() if saga.started_at else None
            },
            correlation_id=orchestration_id
        )

    async def publish_agent_performance_updated(
        self,
        agent_id: str,
        performance_metrics: dict[str, Any]
    ) -> bool:
        """Publiziert Agent-Performance-Updated Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.AGENT_PERFORMANCE_UPDATED,
            data={
                "agent_id": agent_id,
                "performance_metrics": performance_metrics,
                "updated_at": datetime.utcnow().isoformat()
            }
        )

    async def publish_service_health_check(
        self,
        health_status: dict[str, Any]
    ) -> bool:
        """Publiziert Service-Health-Check Event."""
        return await self._publish_event(
            event_type=OrchestratorEventTypes.HEALTH_CHECK_COMPLETED,
            data={
                "service_name": self.service_name,
                "health_status": health_status,
                "checked_at": datetime.utcnow().isoformat()
            }
        )

    # =========================================================================
    # Event Subscription
    # =========================================================================

    async def subscribe_to_agent_events(
        self,
        handler: Callable[[PlatformEvent], None]
    ) -> str | None:
        """Abonniert Agent-Events."""
        return await self._subscribe_event(
            event_type="agents.*",
            handler=handler,
            consumer_group="orchestrator-agent-events"
        )

    async def subscribe_to_task_events(
        self,
        handler: Callable[[PlatformEvent], None]
    ) -> str | None:
        """Abonniert Task-Events."""
        return await self._subscribe_event(
            event_type="tasks.*",
            handler=handler,
            consumer_group="orchestrator-task-events"
        )

    async def subscribe_to_system_events(
        self,
        handler: Callable[[PlatformEvent], None]
    ) -> str | None:
        """Abonniert System-Events."""
        return await self._subscribe_event(
            event_type="system.*",
            handler=handler,
            consumer_group="orchestrator-system-events"
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _publish_event(
        self,
        event_type: str,
        data: dict[str, Any],
        correlation_id: str | None = None
    ) -> bool:
        """Publiziert Event über Platform Event Bus."""
        if not self.event_bus or not self.started:
            logger.warning("Event Bus nicht verfügbar")
            return False

        try:
            event = PlatformEvent(
                event_id=str(uuid4()),
                event_type=event_type,
                source_service=self.service_name,
                data=data,
                correlation_id=correlation_id or str(uuid4()),
                timestamp=datetime.now(UTC),
                version="1.0"
            )

            success = await self.event_bus.publish(event)

            if success:
                self.events_published += 1
                logger.debug(f"Event publiziert: {event_type}")
            else:
                self.events_failed += 1
                logger.error(f"Event-Publikation fehlgeschlagen: {event_type}")

            return success

        except Exception as e:
            self.events_failed += 1
            logger.exception(f"Fehler beim Publizieren des Events: {e}")
            return False

    async def _subscribe_event(
        self,
        event_type: str,
        handler: Callable[[PlatformEvent], None],
        consumer_group: str | None = None
    ) -> str | None:
        """Abonniert Event über Platform Event Bus."""
        if not self.event_bus or not self.started:
            logger.warning("Event Bus nicht verfügbar")
            return None

        try:
            subscription_id = await self.event_bus.subscribe(
                event_type=event_type,
                handler=handler,
                consumer_group=consumer_group
            )

            if subscription_id:
                logger.info(f"Event-Subscription erstellt: {event_type}")
            else:
                logger.error(f"Event-Subscription fehlgeschlagen: {event_type}")

            return subscription_id

        except Exception as e:
            logger.exception(f"Fehler beim Erstellen der Event-Subscription: {e}")
            return None

    async def _create_default_event_bus(self) -> PlatformEventBus:
        """Erstellt Standard Event Bus."""
        nats_config = PlatformNATSConfig(
            servers=["nats://localhost:4222"],
            cluster_name="orchestrator-cluster",
            jetstream_enabled=True
        )

        config = PlatformEventBusConfig(
            nats_config=nats_config,
            enable_schema_validation=True,
            enable_dead_letter_queue=True
        )

        return PlatformEventBus(config)

    async def _register_default_subscriptions(self) -> None:
        """Registriert Standard Event-Subscriptions."""
        try:
            # Agent-Events für Performance-Tracking
            await self.subscribe_to_agent_events(self._handle_agent_event)

            # Task-Events für Coordination
            await self.subscribe_to_task_events(self._handle_task_event)

            # System-Events für Health-Monitoring
            await self.subscribe_to_system_events(self._handle_system_event)

            logger.info("Standard Event-Subscriptions registriert")

        except Exception as e:
            logger.exception(f"Fehler beim Registrieren der Standard-Subscriptions: {e}")

    async def _handle_agent_event(self, event: PlatformEvent) -> None:
        """Behandelt Agent-Events."""
        try:
            self.events_consumed += 1
            logger.debug(f"Agent-Event empfangen: {event.event_type}")

            # Hier würde die spezifische Agent-Event-Behandlung implementiert werden

        except Exception as e:
            logger.exception(f"Fehler beim Behandeln des Agent-Events: {e}")

    async def _handle_task_event(self, event: PlatformEvent) -> None:
        """Behandelt Task-Events."""
        try:
            self.events_consumed += 1
            logger.debug(f"Task-Event empfangen: {event.event_type}")

            # Hier würde die spezifische Task-Event-Behandlung implementiert werden

        except Exception as e:
            logger.exception(f"Fehler beim Behandeln des Task-Events: {e}")

    async def _handle_system_event(self, event: PlatformEvent) -> None:
        """Behandelt System-Events."""
        try:
            self.events_consumed += 1
            logger.debug(f"System-Event empfangen: {event.event_type}")

            # Hier würde die spezifische System-Event-Behandlung implementiert werden

        except Exception as e:
            logger.exception(f"Fehler beim Behandeln des System-Events: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Holt Event-Integration-Metriken."""
        return {
            "events_published": self.events_published,
            "events_consumed": self.events_consumed,
            "events_failed": self.events_failed,
            "started": self.started,
            "event_bus_connected": self.event_bus.started if self.event_bus else False
        }
