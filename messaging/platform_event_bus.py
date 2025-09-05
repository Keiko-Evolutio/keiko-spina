#!/usr/bin/env python3
"""Platform Event Bus für Issue #56 Messaging-first Architecture
Implementiert Event-driven Architecture mit NATS JetStream für Platform-interne Kommunikation

ARCHITEKTUR-COMPLIANCE:
- Ausschließlich für Platform-interne Events
- Keine SDK-Dependencies oder -Exports
- API-basierte Kommunikation für externe Systeme
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger

from .platform_nats_client import PlatformMessage, PlatformNATSClient, PlatformNATSConfig
from .platform_schema_registry import PlatformEventSchema, PlatformSchemaRegistry

logger = get_logger(__name__)

class PlatformEventType(Enum):
    """Platform-interne Event-Typen"""
    # Agent Events
    AGENT_CREATED = "platform.agent.created"
    AGENT_UPDATED = "platform.agent.updated"
    AGENT_STATUS_CHANGED = "platform.agent.status_changed"
    AGENT_DELETED = "platform.agent.deleted"

    # Task Events
    TASK_CREATED = "platform.task.created"
    TASK_ASSIGNED = "platform.task.assigned"
    TASK_STARTED = "platform.task.started"
    TASK_COMPLETED = "platform.task.completed"
    TASK_FAILED = "platform.task.failed"

    # Workflow Events
    WORKFLOW_STARTED = "platform.workflow.started"
    WORKFLOW_STEP_COMPLETED = "platform.workflow.step_completed"
    WORKFLOW_COMPLETED = "platform.workflow.completed"
    WORKFLOW_FAILED = "platform.workflow.failed"

    # System Events
    SYSTEM_HEALTH_CHECK = "platform.system.health_check"
    SYSTEM_MAINTENANCE_START = "platform.system.maintenance_start"
    SYSTEM_MAINTENANCE_END = "platform.system.maintenance_end"

@dataclass
class PlatformEvent:
    """Platform-interne Event-Struktur"""
    event_type: str
    event_id: str
    source_service: str
    data: dict[str, Any]
    correlation_id: str | None = None
    timestamp: datetime | None = None
    version: str = "1.0"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        if self.correlation_id is None:
            self.correlation_id = self.event_id

@dataclass
class PlatformEventBusConfig:
    """Konfiguration für Platform Event Bus"""
    nats_config: PlatformNATSConfig
    enable_schema_validation: bool = True
    enable_dead_letter_queue: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_metrics: bool = True

class PlatformEventBus:
    """Platform Event Bus für interne Event-driven Architecture"""

    def __init__(self, config: PlatformEventBusConfig):
        self.config = config
        self.nats_client = PlatformNATSClient(config.nats_config)
        self.schema_registry = PlatformSchemaRegistry() if config.enable_schema_validation else None

        # Event Handlers
        self.event_handlers: dict[str, list[Callable]] = {}
        self.middleware: list[Callable] = []

        # Metriken
        self.events_published = 0
        self.events_consumed = 0
        self.events_failed = 0
        self.handler_errors = 0

        # Status
        self.started = False

    async def start(self) -> bool:
        """Startet den Platform Event Bus"""
        try:
            logger.info("Starte Platform Event Bus...")

            # NATS Client verbinden
            if not await self.nats_client.connect():
                logger.error("Fehler beim Verbinden des NATS Clients")
                return False

            # Schema Registry initialisieren
            if self.schema_registry:
                await self.schema_registry.initialize()
                await self._register_platform_schemas()

            self.started = True
            logger.info("Platform Event Bus erfolgreich gestartet")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Starten des Platform Event Bus: {e}")
            return False

    async def stop(self):
        """Stoppt den Platform Event Bus"""
        try:
            logger.info("Stoppe Platform Event Bus...")

            await self.nats_client.disconnect()
            self.started = False

            logger.info("Platform Event Bus gestoppt")

        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Platform Event Bus: {e}")

    async def publish(self, event: PlatformEvent) -> bool:
        """Publiziert Platform-interne Events"""
        if not self.started:
            logger.error("Platform Event Bus nicht gestartet")
            return False

        try:
            # Schema-Validierung
            if self.schema_registry:
                if not await self.schema_registry.validate_event(event):
                    logger.error(f"Schema-Validierung fehlgeschlagen für Event: {event.event_type}")
                    return False

            # Middleware ausführen
            for middleware in self.middleware:
                event = await middleware(event)
                if event is None:
                    logger.warning("Event von Middleware blockiert")
                    return False

            # NATS Message erstellen
            message = PlatformMessage(
                subject=event.event_type,
                data=asdict(event),
                headers={
                    "event-type": event.event_type,
                    "source-service": event.source_service,
                    "version": event.version
                },
                message_id=event.event_id,
                correlation_id=event.correlation_id,
                timestamp=event.timestamp
            )

            # Message publizieren
            success = await self.nats_client.publish(message)

            if success:
                self.events_published += 1
                logger.debug(f"Platform Event publiziert: {event.event_type}")
            else:
                self.events_failed += 1
                logger.error(f"Fehler beim Publizieren des Platform Events: {event.event_type}")

            return success

        except Exception as e:
            self.events_failed += 1
            logger.error(f"Fehler beim Publizieren des Platform Events: {e}")
            return False

    async def subscribe(self,
                       event_type: str,
                       handler: Callable[[PlatformEvent], None],
                       consumer_group: str | None = None) -> str | None:
        """Abonniert Platform-interne Events"""
        if not self.started:
            logger.error("Platform Event Bus nicht gestartet")
            return None

        try:
            # Handler registrieren
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(handler)

            # NATS Subscription erstellen
            consumer_name = consumer_group or f"platform-consumer-{event_type.replace('.', '-')}"

            subscription_id = await self.nats_client.subscribe(
                subject=event_type,
                callback=self._create_event_handler(event_type),
                consumer_name=consumer_name,
                durable=True
            )

            logger.info(f"Platform Event Subscription erstellt: {event_type}")
            return subscription_id

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Platform Event Subscription: {e}")
            return None

    def _create_event_handler(self, event_type: str):
        """Erstellt Event-Handler für NATS Subscription"""
        async def handler(message: PlatformMessage):
            try:
                # Platform Event aus Message erstellen
                event_data = message.data
                event = PlatformEvent(
                    event_type=event_data["event_type"],
                    event_id=event_data["event_id"],
                    source_service=event_data["source_service"],
                    data=event_data["data"],
                    correlation_id=event_data.get("correlation_id"),
                    timestamp=datetime.fromisoformat(event_data["timestamp"]) if event_data.get("timestamp") else None,
                    version=event_data.get("version", "1.0")
                )

                # Alle Handler für Event-Typ ausführen
                handlers = self.event_handlers.get(event_type, [])
                for event_handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(event_handler):
                            await event_handler(event)
                        else:
                            event_handler(event)
                    except Exception as e:
                        self.handler_errors += 1
                        logger.error(f"Fehler in Platform Event Handler: {e}")

                self.events_consumed += 1

            except Exception as e:
                self.handler_errors += 1
                logger.error(f"Fehler beim Verarbeiten des Platform Events: {e}")

        return handler

    async def add_middleware(self, middleware: Callable[[PlatformEvent], PlatformEvent]):
        """Fügt Middleware zum Event Bus hinzu"""
        self.middleware.append(middleware)
        logger.debug("Platform Event Bus Middleware hinzugefügt")

    async def _register_platform_schemas(self):
        """Registriert Platform-interne Event-Schemas"""
        try:
            # Agent Event Schemas
            agent_created_schema = PlatformEventSchema(
                event_type=PlatformEventType.AGENT_CREATED.value,
                version="1.0",
                schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "agent_type": {"type": "string"},
                        "capabilities": {"type": "array", "items": {"type": "string"}},
                        "configuration": {"type": "object"},
                        "created_by": {"type": "string"}
                    },
                    "required": ["agent_id", "agent_type", "capabilities"]
                }
            )

            agent_status_changed_schema = PlatformEventSchema(
                event_type=PlatformEventType.AGENT_STATUS_CHANGED.value,
                version="1.0",
                schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "previous_status": {"type": "string"},
                        "new_status": {"type": "string"},
                        "reason": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    },
                    "required": ["agent_id", "previous_status", "new_status"]
                }
            )

            # Task Event Schemas
            task_created_schema = PlatformEventSchema(
                event_type=PlatformEventType.TASK_CREATED.value,
                version="1.0",
                schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "task_type": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "data": {"type": "object"},
                        "created_by": {"type": "string"}
                    },
                    "required": ["task_id", "task_type", "priority"]
                }
            )

            # Workflow Event Schemas
            workflow_started_schema = PlatformEventSchema(
                event_type=PlatformEventType.WORKFLOW_STARTED.value,
                version="1.0",
                schema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "plan_id": {"type": "string"},
                        "context": {"type": "object"},
                        "steps": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["workflow_id", "plan_id"]
                }
            )

            # System Event Schemas
            system_health_schema = PlatformEventSchema(
                event_type=PlatformEventType.SYSTEM_HEALTH_CHECK.value,
                version="1.0",
                schema={
                    "type": "object",
                    "properties": {
                        "component": {"type": "string"},
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                        "metrics": {"type": "object"},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "errors": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["component", "status"]
                }
            )

            # Schemas registrieren
            schemas = [
                agent_created_schema,
                agent_status_changed_schema,
                task_created_schema,
                workflow_started_schema,
                system_health_schema
            ]

            for schema in schemas:
                await self.schema_registry.register_schema(schema)

            logger.info(f"Platform Event Schemas registriert: {len(schemas)}")

        except Exception as e:
            logger.error(f"Fehler beim Registrieren der Platform Event Schemas: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Event Bus Metriken zurück"""
        nats_metrics = self.nats_client.get_metrics()

        return {
            "started": self.started,
            "events_published": self.events_published,
            "events_consumed": self.events_consumed,
            "events_failed": self.events_failed,
            "handler_errors": self.handler_errors,
            "registered_handlers": len(self.event_handlers),
            "middleware_count": len(self.middleware),
            "nats_metrics": nats_metrics
        }

    @asynccontextmanager
    async def transaction(self):
        """Context Manager für transaktionale Event-Publikation"""
        events_to_publish = []

        class TransactionContext:
            def add_event(self, event: PlatformEvent):
                events_to_publish.append(event)

        context = TransactionContext()

        try:
            yield context

            # Alle Events in Transaktion publizieren
            for event in events_to_publish:
                await self.publish(event)

        except Exception as e:
            logger.error(f"Fehler in Platform Event Bus Transaktion: {e}")
            raise
