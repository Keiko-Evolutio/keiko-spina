#!/usr/bin/env python3
"""SDK-Platform gRPC Service Implementation für Issue #56 Messaging-first Architecture
Implementiert gRPC Services basierend auf api-contracts/protobuf/sdk/v1/sdk_communication.proto

ARCHITEKTUR-COMPLIANCE:
- Implementiert SDKPlatformCommunicationService
- Nutzt Platform Event Bus für interne Kommunikation
- Keine direkten NATS-Zugriffe für externe Systeme
- High-performance bidirektionale Streaming
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

# Protocol Buffers Imports (würden normalerweise aus generierten Files kommen)
# Für Demo verwenden wir Mock-Klassen
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import grpc

from kei_logging import get_logger
from messaging import PlatformEvent, PlatformEventBus
from messaging.platform_event_bus import PlatformEventType

logger = get_logger(__name__)

# Mock Protocol Buffers Messages (in echter Implementierung aus .proto generiert)
class SDKAgentType(Enum):
    SDK_AGENT_TYPE_UNSPECIFIED = 0
    SDK_AGENT_TYPE_CODE_ASSISTANT = 1
    SDK_AGENT_TYPE_DATA_ANALYST = 2
    SDK_AGENT_TYPE_WORKFLOW_MANAGER = 3
    SDK_AGENT_TYPE_MONITORING_AGENT = 4
    SDK_AGENT_TYPE_CUSTOM = 5

class SDKAgentStatus(Enum):
    SDK_AGENT_STATUS_UNSPECIFIED = 0
    SDK_AGENT_STATUS_ACTIVE = 1
    SDK_AGENT_STATUS_INACTIVE = 2
    SDK_AGENT_STATUS_ERROR = 3
    SDK_AGENT_STATUS_MAINTENANCE = 4

class SDKFunctionCallStatus(Enum):
    SDK_FUNCTION_CALL_STATUS_UNSPECIFIED = 0
    SDK_FUNCTION_CALL_STATUS_PENDING = 1
    SDK_FUNCTION_CALL_STATUS_RUNNING = 2
    SDK_FUNCTION_CALL_STATUS_SUCCESS = 3
    SDK_FUNCTION_CALL_STATUS_ERROR = 4
    SDK_FUNCTION_CALL_STATUS_TIMEOUT = 5
    SDK_FUNCTION_CALL_STATUS_CANCELLED = 6

@dataclass
class RegisterAgentRequest:
    agent_id: str
    agent_type: SDKAgentType
    capabilities: list[str]
    metadata: dict[str, Any]
    configuration: dict[str, Any]
    sdk_version: str

@dataclass
class RegisterAgentResponse:
    agent_id: str
    registered: bool
    registered_at: datetime
    platform_agent_id: str
    assigned_configuration: dict[str, Any]

@dataclass
class UpdateAgentRequest:
    agent_id: str
    status: SDKAgentStatus
    capabilities: list[str]
    metadata: dict[str, Any]

@dataclass
class UpdateAgentResponse:
    agent_id: str
    updated: bool
    updated_at: datetime

@dataclass
class PublishEventRequest:
    agent_id: str
    event: dict[str, Any]
    wait_for_confirmation: bool

@dataclass
class PublishEventResponse:
    event_id: str
    published: bool
    published_at: datetime

@dataclass
class SubscribeEventsRequest:
    agent_id: str
    event_types: list[str]
    source_agents: list[str]
    include_platform_events: bool

@dataclass
class EventMessage:
    event_id: str
    event_type: str
    timestamp: datetime
    source_agent_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any]

@dataclass
class FunctionCallRequest:
    function_id: str
    agent_id: str
    function_name: str
    parameters: dict[str, Any]
    timeout_seconds: int
    correlation_id: str

@dataclass
class FunctionCallResponse:
    call_id: str
    function_id: str
    status: SDKFunctionCallStatus
    result: dict[str, Any] | None
    error_message: str | None
    error_code: str | None
    execution_time_ms: int
    completed_at: datetime

@dataclass
class HeartbeatRequest:
    agent_id: str
    status: SDKAgentStatus
    metrics: dict[str, Any]
    warnings: list[str]
    errors: list[str]

@dataclass
class HeartbeatResponse:
    agent_id: str
    acknowledged: bool
    server_time: datetime
    instruction: dict[str, Any] | None

class SDKPlatformCommunicationServiceImpl:
    """gRPC Service Implementation für SDK-Platform Kommunikation"""

    def __init__(self, event_bus: PlatformEventBus):
        self.event_bus = event_bus

        # In-Memory Storage für Demo (in echter Implementierung würde Database verwendet)
        self.agents_storage: dict[str, dict[str, Any]] = {}
        self.functions_storage: dict[str, dict[str, Any]] = {}
        self.active_subscriptions: dict[str, asyncio.Queue] = {}

        # Metriken
        self.grpc_requests = 0
        self.grpc_errors = 0
        self.active_streams = 0

    async def register_agent(self, request: RegisterAgentRequest, context: grpc.aio.ServicerContext) -> RegisterAgentResponse:
        """SDK registriert neuen Agent bei Platform über gRPC"""
        try:
            self.grpc_requests += 1

            # Prüfe ob Agent bereits existiert
            if request.agent_id in self.agents_storage:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(f"Agent {request.agent_id} already exists")
                return RegisterAgentResponse(
                    agent_id=request.agent_id,
                    registered=False,
                    registered_at=datetime.now(UTC),
                    platform_agent_id="",
                    assigned_configuration={}
                )

            # Agent-Daten erstellen
            now = datetime.now(UTC)
            agent_data = {
                "agent_id": request.agent_id,
                "agent_type": request.agent_type.name,
                "status": "active",
                "capabilities": request.capabilities,
                "metadata": request.metadata,
                "configuration": request.configuration,
                "sdk_version": request.sdk_version,
                "created_at": now,
                "updated_at": now,
                "last_seen": now
            }

            # Agent speichern
            self.agents_storage[request.agent_id] = agent_data

            # Platform Event publizieren
            platform_event = PlatformEvent(
                event_type=PlatformEventType.AGENT_CREATED.value,
                event_id=str(uuid.uuid4()),
                source_service="platform-grpc-api",
                data={
                    "agent_id": request.agent_id,
                    "agent_type": request.agent_type.name,
                    "capabilities": request.capabilities,
                    "configuration": request.configuration,
                    "created_by": "grpc-api",
                    "sdk_version": request.sdk_version
                }
            )

            await self.event_bus.publish(platform_event)

            logger.info(f"Agent registriert via gRPC: {request.agent_id}")

            return RegisterAgentResponse(
                agent_id=request.agent_id,
                registered=True,
                registered_at=now,
                platform_agent_id=request.agent_id,  # Gleiche ID für Demo
                assigned_configuration=request.configuration
            )

        except Exception as e:
            self.grpc_errors += 1
            logger.error(f"Fehler beim Registrieren des Agents via gRPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def update_agent(self, request: UpdateAgentRequest, context: grpc.aio.ServicerContext) -> UpdateAgentResponse:
        """SDK aktualisiert Agent-Status über gRPC"""
        try:
            self.grpc_requests += 1

            if request.agent_id not in self.agents_storage:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Agent {request.agent_id} not found")
                return UpdateAgentResponse(
                    agent_id=request.agent_id,
                    updated=False,
                    updated_at=datetime.now(UTC)
                )

            agent_data = self.agents_storage[request.agent_id]
            previous_status = agent_data.get("status")

            # Agent-Daten aktualisieren
            agent_data["status"] = request.status.name
            agent_data["capabilities"] = request.capabilities
            agent_data["metadata"].update(request.metadata)
            agent_data["updated_at"] = datetime.now(UTC)
            agent_data["last_seen"] = datetime.now(UTC)

            # Platform Event publizieren bei Status-Änderung
            if request.status.name != previous_status:
                platform_event = PlatformEvent(
                    event_type=PlatformEventType.AGENT_STATUS_CHANGED.value,
                    event_id=str(uuid.uuid4()),
                    source_service="platform-grpc-api",
                    data={
                        "agent_id": request.agent_id,
                        "previous_status": previous_status,
                        "new_status": request.status.name,
                        "reason": "grpc_update",
                        "timestamp": datetime.now(UTC).isoformat()
                    }
                )
                await self.event_bus.publish(platform_event)

            logger.info(f"Agent aktualisiert via gRPC: {request.agent_id}")

            return UpdateAgentResponse(
                agent_id=request.agent_id,
                updated=True,
                updated_at=agent_data["updated_at"]
            )

        except Exception as e:
            self.grpc_errors += 1
            logger.error(f"Fehler beim Aktualisieren des Agents via gRPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def publish_event(self, request: PublishEventRequest, context: grpc.aio.ServicerContext) -> PublishEventResponse:
        """SDK publiziert Events über gRPC"""
        try:
            self.grpc_requests += 1

            # Platform Event erstellen
            event_id = str(uuid.uuid4())

            platform_event = PlatformEvent(
                event_type=request.event.get("event_type", "platform.sdk.event"),
                event_id=event_id,
                source_service="platform-grpc-api",
                data={
                    "source_agent_id": request.agent_id,
                    "event_data": request.event
                },
                correlation_id=request.event.get("correlation_id", event_id)
            )

            # Event über Platform Event Bus publizieren
            success = await self.event_bus.publish(platform_event)

            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to publish event")
                return PublishEventResponse(
                    event_id=event_id,
                    published=False,
                    published_at=datetime.now(UTC)
                )

            logger.info(f"Event publiziert via gRPC: {request.agent_id} -> {platform_event.event_type}")

            return PublishEventResponse(
                event_id=event_id,
                published=True,
                published_at=datetime.now(UTC)
            )

        except Exception as e:
            self.grpc_errors += 1
            logger.error(f"Fehler beim Publizieren des Events via gRPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def subscribe_events(self, request: SubscribeEventsRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[EventMessage]:
        """SDK abonniert Events über gRPC Streaming"""
        try:
            self.grpc_requests += 1
            self.active_streams += 1

            # Event Queue für diesen Stream erstellen
            event_queue = asyncio.Queue()
            subscription_id = f"{request.agent_id}_{uuid.uuid4()}"
            self.active_subscriptions[subscription_id] = event_queue

            logger.info(f"gRPC Event Subscription gestartet: {request.agent_id}")

            try:
                # Event-Handler für Platform Events registrieren
                def event_handler(platform_event: PlatformEvent):
                    """Synchroner Event-Handler für Platform Events"""
                    # Filter Events basierend auf Request
                    if request.event_types and platform_event.event_type not in request.event_types:
                        return

                    if request.source_agents:
                        source_agent = platform_event.data.get("source_agent_id")
                        if source_agent and source_agent not in request.source_agents:
                            return

                    # Event in Queue einreihen
                    event_message = EventMessage(
                        event_id=platform_event.event_id,
                        event_type=platform_event.event_type,
                        timestamp=platform_event.timestamp,
                        source_agent_id=platform_event.data.get("source_agent_id", "platform"),
                        payload=platform_event.data,
                        metadata={"correlation_id": platform_event.correlation_id}
                    )

                    try:
                        event_queue.put_nowait(event_message)
                    except asyncio.QueueFull:
                        logger.warning(f"Event queue full for subscription {subscription_id}")

                # Handler für alle relevanten Event-Typen registrieren
                for event_type in request.event_types or ["platform.*"]:
                    await self.event_bus.subscribe(event_type, event_handler)

                # Event-Stream senden
                while True:
                    try:
                        # Warte auf nächstes Event oder Context-Cancellation
                        event_message = await asyncio.wait_for(
                            event_queue.get(),
                            timeout=1.0
                        )

                        yield event_message

                    except TimeoutError:
                        # Prüfe ob Client noch verbunden ist
                        if context.cancelled():
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Fehler beim Senden des gRPC Events: {e}")
                        break

            finally:
                # Cleanup
                if subscription_id in self.active_subscriptions:
                    del self.active_subscriptions[subscription_id]
                self.active_streams -= 1
                logger.info(f"gRPC Event Subscription beendet: {request.agent_id}")

        except Exception as e:
            self.grpc_errors += 1
            self.active_streams -= 1
            logger.error(f"Fehler bei gRPC Event Subscription: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def call_function(self, request: FunctionCallRequest, context: grpc.aio.ServicerContext) -> FunctionCallResponse:
        """Platform ruft Agent-Function über gRPC auf"""
        try:
            self.grpc_requests += 1

            # Simuliere Function Call (in echter Implementierung würde hier Agent kontaktiert)
            call_id = str(uuid.uuid4())
            start_time = datetime.now(UTC)

            # Mock-Ausführung
            await asyncio.sleep(0.1)  # Simuliere Verarbeitungszeit

            # Mock-Ergebnis
            result = {
                "status": "success",
                "data": f"Function {request.function_name} executed",
                "parameters_received": request.parameters
            }

            execution_time = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            logger.info(f"Function Call via gRPC: {request.agent_id}.{request.function_name}")

            return FunctionCallResponse(
                call_id=call_id,
                function_id=request.function_id,
                status=SDKFunctionCallStatus.SDK_FUNCTION_CALL_STATUS_SUCCESS,
                result=result,
                error_message=None,
                error_code=None,
                execution_time_ms=execution_time,
                completed_at=datetime.now(UTC)
            )

        except Exception as e:
            self.grpc_errors += 1
            logger.error(f"Fehler beim Function Call via gRPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def stream_function_calls(self, request_iterator: AsyncIterator, context: grpc.aio.ServicerContext) -> AsyncIterator[FunctionCallResponse]:
        """Bidirektionales Streaming für Function Calls"""
        try:
            self.grpc_requests += 1
            self.active_streams += 1

            logger.info("gRPC Function Call Stream gestartet")

            async for request in request_iterator:
                try:
                    # Verarbeite Function Call Request
                    call_id = str(uuid.uuid4())

                    # Mock-Verarbeitung
                    result = {
                        "status": "success",
                        "stream_call": True,
                        "call_id": call_id
                    }

                    response = FunctionCallResponse(
                        call_id=call_id,
                        function_id=request.function_id,
                        status=SDKFunctionCallStatus.SDK_FUNCTION_CALL_STATUS_SUCCESS,
                        result=result,
                        error_message=None,
                        error_code=None,
                        execution_time_ms=50,
                        completed_at=datetime.now(UTC)
                    )

                    yield response

                except Exception as e:
                    logger.error(f"Fehler beim Verarbeiten des Stream Function Calls: {e}")

                    error_response = FunctionCallResponse(
                        call_id=str(uuid.uuid4()),
                        function_id="unknown",
                        status=SDKFunctionCallStatus.SDK_FUNCTION_CALL_STATUS_ERROR,
                        result=None,
                        error_message=str(e),
                        error_code="STREAM_ERROR",
                        execution_time_ms=0,
                        completed_at=datetime.now(UTC)
                    )

                    yield error_response

        except Exception as e:
            self.grpc_errors += 1
            logger.error(f"Fehler bei gRPC Function Call Stream: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
        finally:
            self.active_streams -= 1
            logger.info("gRPC Function Call Stream beendet")

    async def send_heartbeat(self, request: HeartbeatRequest, context: grpc.aio.ServicerContext) -> HeartbeatResponse:
        """SDK sendet Heartbeat an Platform"""
        try:
            self.grpc_requests += 1

            # Agent Last-Seen aktualisieren
            if request.agent_id in self.agents_storage:
                self.agents_storage[request.agent_id]["last_seen"] = datetime.now(UTC)
                self.agents_storage[request.agent_id]["status"] = request.status.name

            # Heartbeat verarbeiten
            logger.debug(f"Heartbeat empfangen via gRPC: {request.agent_id}")

            return HeartbeatResponse(
                agent_id=request.agent_id,
                acknowledged=True,
                server_time=datetime.now(UTC),
                instruction=None  # Keine speziellen Instruktionen
            )

        except Exception as e:
            self.grpc_errors += 1
            logger.error(f"Fehler beim Verarbeiten des Heartbeats via gRPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def get_metrics(self) -> dict[str, Any]:
        """Gibt gRPC Service Metriken zurück"""
        return {
            "grpc_requests": self.grpc_requests,
            "grpc_errors": self.grpc_errors,
            "active_streams": self.active_streams,
            "active_subscriptions": len(self.active_subscriptions),
            "registered_agents": len(self.agents_storage),
            "registered_functions": len(self.functions_storage),
            "error_rate": self.grpc_errors / max(self.grpc_requests, 1) * 100
        }
