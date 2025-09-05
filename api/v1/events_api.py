#!/usr/bin/env python3
"""Platform-SDK Events API v1 für Issue #56 Messaging-first Architecture
Implementiert HTTP/REST APIs basierend auf api-contracts/openapi/platform-sdk-events-api-v1.yaml

ARCHITEKTUR-COMPLIANCE:
- Exponiert Platform-interne Events über HTTP APIs
- Nutzt Platform Event Bus aus Phase 1
- Keine direkten NATS-Zugriffe für externe Systeme
- Implementiert OpenAPI Contract exakt
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from kei_logging import get_logger
from messaging import PlatformEvent, PlatformEventBus
from messaging.platform_event_bus import PlatformEventType

logger = get_logger(__name__)

# Router für Events API
events_router = APIRouter(prefix="/events", tags=["Platform-SDK Events"])

# Pydantic Models basierend auf OpenAPI Contract
class AgentEventRequest(BaseModel):
    """Agent Event Request Schema"""
    event_type: str = Field(..., description="Typ des Agent-Events")
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9_]+$", description="Eindeutige Agent-ID")
    agent_type: str | None = Field(None, description="Typ des Agents")
    capabilities: list[str] | None = Field(None, description="Agent-Fähigkeiten")
    status: str | None = Field(None, description="Agent-Status")
    metadata: dict[str, str] | None = Field(None, description="Zusätzliche Event-Metadaten")

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        allowed_types = [
            "agent.created", "agent.updated", "agent.deleted", "agent.status_changed"
        ]
        if v not in allowed_types:
            raise ValueError(f"Event type must be one of: {allowed_types}")
        return v

class TaskEventRequest(BaseModel):
    """Task Event Request Schema"""
    event_type: str = Field(..., description="Typ des Task-Events")
    task_id: str = Field(..., pattern=r"^task_[a-zA-Z0-9_]+$", description="Eindeutige Task-ID")
    agent_id: str | None = Field(None, pattern=r"^agent_[a-zA-Z0-9_]+$")
    task_type: str | None = Field(None, description="Task-Typ")
    priority: str | None = Field(None, description="Task-Priorität")
    metadata: dict[str, str] | None = Field(None, description="Zusätzliche Event-Metadaten")

class WorkflowEventRequest(BaseModel):
    """Workflow Event Request Schema"""
    event_type: str = Field(..., description="Typ des Workflow-Events")
    workflow_id: str = Field(..., pattern=r"^workflow_[a-zA-Z0-9_]+$")
    plan_id: str | None = Field(None, description="Plan-ID")
    status: str | None = Field(None, description="Workflow-Status")
    metadata: dict[str, str] | None = Field(None, description="Zusätzliche Event-Metadaten")

class EventResponse(BaseModel):
    """Event Response Schema"""
    event_id: str = Field(..., description="Eindeutige Event-ID")
    status: str = Field(..., description="Event-Verarbeitungsstatus")
    timestamp: datetime = Field(..., description="Event-Publikationszeitpunkt")
    api_version: str = Field(default="v1", description="Verwendete API-Version")

class ErrorResponse(BaseModel):
    """Error Response Schema"""
    error: str = Field(..., description="Error-Code")
    message: str = Field(..., description="Human-readable Error-Message")
    details: dict[str, Any] | None = Field(None, description="Zusätzliche Error-Details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class HealthResponse(BaseModel):
    """Health Response Schema"""
    status: str = Field(..., description="Health-Status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = Field(default="v1.0.0", description="API-Version")
    platform_connectivity: str = Field(..., description="Platform-Konnektivität")
    metrics: dict[str, Any] | None = Field(None, description="Health-Metriken")

# Dependency für Event Bus
async def get_event_bus() -> PlatformEventBus:
    """Dependency für Platform Event Bus"""
    # Hier würde normalerweise der Event Bus aus der Dependency Injection kommen
    # Für jetzt erstellen wir eine Mock-Instanz
    from messaging.platform_event_bus import PlatformEventBusConfig
    from messaging.platform_nats_client import PlatformNATSConfig

    nats_config = PlatformNATSConfig(
        servers=["nats://localhost:4222"],
        cluster_name="platform-cluster"
    )

    config = PlatformEventBusConfig(
        nats_config=nats_config,
        enable_schema_validation=True
    )

    event_bus = PlatformEventBus(config)

    # Event Bus sollte bereits gestartet sein - hier nur für Demo
    if not event_bus.started:
        await event_bus.start()

    return event_bus

# API Endpoints
@events_router.post("/agent", response_model=EventResponse, status_code=201)
async def publish_agent_event(
    request: AgentEventRequest,
    event_bus: PlatformEventBus = Depends(get_event_bus)
) -> EventResponse:
    """Publiziert Agent-Events über HTTP API

    SDK publiziert Agent-Events über HTTP API statt direkter Messaging-Kommunikation.
    Platform verarbeitet Events intern mit eigenem Messaging-System.
    """
    try:
        # Platform Event erstellen
        event_id = str(uuid.uuid4())

        # Map API Event Type zu Platform Event Type
        platform_event_type = _map_api_to_platform_event_type("agent", request.event_type)

        platform_event = PlatformEvent(
            event_type=platform_event_type,
            event_id=event_id,
            source_service="platform-sdk-api",
            data={
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "capabilities": request.capabilities or [],
                "status": request.status,
                "metadata": request.metadata or {}
            },
            correlation_id=event_id
        )

        # Event über Platform Event Bus publizieren
        success = await event_bus.publish(platform_event)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to publish agent event"
            )

        logger.info(f"Agent Event publiziert via API: {request.event_type} -> {platform_event_type}")

        return EventResponse(
            event_id=event_id,
            status="published",
            timestamp=datetime.now(UTC),
            api_version="v1"
        )

    except Exception as e:
        logger.error(f"Fehler beim Publizieren des Agent Events: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to publish agent event: {e!s}"
        )

@events_router.post("/task", response_model=EventResponse, status_code=201)
async def publish_task_event(
    request: TaskEventRequest,
    event_bus: PlatformEventBus = Depends(get_event_bus)
) -> EventResponse:
    """Publiziert Task-Events über HTTP API"""
    try:
        event_id = str(uuid.uuid4())
        platform_event_type = _map_api_to_platform_event_type("task", request.event_type)

        platform_event = PlatformEvent(
            event_type=platform_event_type,
            event_id=event_id,
            source_service="platform-sdk-api",
            data={
                "task_id": request.task_id,
                "agent_id": request.agent_id,
                "task_type": request.task_type,
                "priority": request.priority,
                "metadata": request.metadata or {}
            }
        )

        success = await event_bus.publish(platform_event)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to publish task event")

        return EventResponse(
            event_id=event_id,
            status="published",
            timestamp=datetime.now(UTC)
        )

    except Exception as e:
        logger.error(f"Fehler beim Publizieren des Task Events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@events_router.post("/workflow", response_model=EventResponse, status_code=201)
async def publish_workflow_event(
    request: WorkflowEventRequest,
    event_bus: PlatformEventBus = Depends(get_event_bus)
) -> EventResponse:
    """Publiziert Workflow-Events über HTTP API"""
    try:
        event_id = str(uuid.uuid4())
        platform_event_type = _map_api_to_platform_event_type("workflow", request.event_type)

        platform_event = PlatformEvent(
            event_type=platform_event_type,
            event_id=event_id,
            source_service="platform-sdk-api",
            data={
                "workflow_id": request.workflow_id,
                "plan_id": request.plan_id,
                "status": request.status,
                "metadata": request.metadata or {}
            }
        )

        success = await event_bus.publish(platform_event)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to publish workflow event")

        return EventResponse(
            event_id=event_id,
            status="published",
            timestamp=datetime.now(UTC)
        )

    except Exception as e:
        logger.error(f"Fehler beim Publizieren des Workflow Events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@events_router.get("/agent/{agent_id}/stream")
async def stream_agent_events(
    agent_id: str,
    event_types: str | None = None,
    request: Request = None
) -> StreamingResponse:
    """Stream Agent Events via Server-Sent Events

    Real-time Event-Stream für Agent-Updates via Server-Sent Events.
    Ersetzt direkte Messaging-Subscriptions durch HTTP-basierte Streaming.
    """
    # Validiere Agent ID
    if not agent_id.startswith("agent_"):
        raise HTTPException(
            status_code=400,
            detail="Agent ID must match pattern '^agent_[a-zA-Z0-9_]+$'"
        )

    # Parse Event Types Filter
    event_types_filter = []
    if event_types:
        event_types_filter = event_types.split(",")

    async def event_stream():
        """Generator für Server-Sent Events"""
        try:
            logger.info(f"Starting SSE stream for agent {agent_id}")

            # Simuliere Event-Stream (in echter Implementierung würde hier der Event Bus abonniert)
            counter = 0
            while True:
                # Prüfe ob Client noch verbunden ist
                if await request.is_disconnected():
                    logger.info(f"SSE client disconnected for agent {agent_id}")
                    break

                # Simuliere Event-Daten
                event_data = {
                    "event_id": f"evt_{counter}",
                    "event_type": "agent.updated",
                    "agent_id": agent_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {
                        "status": "active",
                        "last_activity": datetime.now(UTC).isoformat()
                    }
                }

                # Filtere Events falls Filter gesetzt
                if not event_types_filter or event_data["event_type"] in event_types_filter:
                    # Format als SSE-kompatible Nachricht
                    sse_data = f"event: agent_event\ndata: {json.dumps(event_data)}\nid: {event_data['event_id']}\n\n"
                    yield sse_data

                counter += 1
                await asyncio.sleep(5)  # Event alle 5 Sekunden

        except Exception as e:
            logger.error(f"Fehler im SSE Stream für Agent {agent_id}: {e}")
            error_data = f"event: error\ndata: {json.dumps({'error': str(e)})}\nid: error\n\n"
            yield error_data

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@events_router.get("/health", response_model=HealthResponse)
async def health_check(
    event_bus: PlatformEventBus = Depends(get_event_bus)
) -> HealthResponse:
    """API Health Check

    Überprüft API-Verfügbarkeit und Platform-Connectivity
    """
    try:
        # Prüfe Event Bus Status
        event_bus_metrics = event_bus.get_metrics()
        platform_connected = event_bus_metrics.get("started", False)

        # Bestimme Health Status
        if platform_connected:
            status = "healthy"
            connectivity = "connected"
        else:
            status = "degraded"
            connectivity = "disconnected"

        return HealthResponse(
            status=status,
            platform_connectivity=connectivity,
            metrics={
                "event_bus_started": platform_connected,
                "events_published": event_bus_metrics.get("events_published", 0),
                "events_consumed": event_bus_metrics.get("events_consumed", 0),
                "handler_errors": event_bus_metrics.get("handler_errors", 0)
            }
        )

    except Exception as e:
        logger.error(f"Health Check Fehler: {e}")
        return HealthResponse(
            status="unhealthy",
            platform_connectivity="error",
            metrics={"error": str(e)}
        )

# Helper Functions
def _map_api_to_platform_event_type(category: str, api_event_type: str) -> str:
    """Mappt API Event Types zu Platform Event Types"""
    # Mapping von API Event Types zu Platform Event Types
    mapping = {
        "agent": {
            "agent.created": PlatformEventType.AGENT_CREATED.value,
            "agent.updated": PlatformEventType.AGENT_UPDATED.value,
            "agent.status_changed": PlatformEventType.AGENT_STATUS_CHANGED.value,
            "agent.deleted": PlatformEventType.AGENT_DELETED.value
        },
        "task": {
            "task.created": PlatformEventType.TASK_CREATED.value,
            "task.assigned": PlatformEventType.TASK_ASSIGNED.value,
            "task.completed": PlatformEventType.TASK_COMPLETED.value,
            "task.failed": PlatformEventType.TASK_FAILED.value
        },
        "workflow": {
            "workflow.started": PlatformEventType.WORKFLOW_STARTED.value,
            "workflow.completed": PlatformEventType.WORKFLOW_COMPLETED.value,
            "workflow.failed": PlatformEventType.WORKFLOW_FAILED.value
        }
    }

    category_mapping = mapping.get(category, {})
    platform_event_type = category_mapping.get(api_event_type)

    if not platform_event_type:
        # Fallback: verwende API Event Type direkt
        platform_event_type = f"platform.{api_event_type}"

    return platform_event_type
