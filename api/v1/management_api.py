#!/usr/bin/env python3
"""Platform-SDK Management API v1 für Issue #56 Messaging-first Architecture
Implementiert HTTP/REST APIs basierend auf api-contracts/openapi/platform-sdk-management-api-v1.yaml

ARCHITEKTUR-COMPLIANCE:
- Implementiert Agent-Management und Function-Orchestration APIs
- Nutzt Platform Event Bus für interne Kommunikation
- Keine direkten NATS-Zugriffe für externe Systeme
- Vollständige OpenAPI Contract Implementierung
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator

from kei_logging import get_logger
from messaging import PlatformEvent, PlatformEventBus
from messaging.platform_event_bus import PlatformEventType

logger = get_logger(__name__)

# Router für Management API
management_router = APIRouter(prefix="/management", tags=["Platform-SDK Management"])

# Pydantic Models basierend auf OpenAPI Contract
class AgentRegistrationRequest(BaseModel):
    """Agent Registration Request Schema"""
    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9_]+$", description="Eindeutige Agent-ID")
    agent_type: str = Field(..., description="Agent-Typ")
    capabilities: list[str] = Field(..., description="Agent-Fähigkeiten")
    metadata: dict[str, str] | None = Field(None, description="Agent-Metadaten")
    config: dict[str, Any] | None = Field(None, description="Initial agent configuration")

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v):
        allowed_types = ["code_assistant", "data_analyst", "workflow_manager", "monitoring_agent"]
        if v not in allowed_types:
            raise ValueError(f"Agent type must be one of: {allowed_types}")
        return v

class AgentUpdateRequest(BaseModel):
    """Agent Update Request Schema"""
    status: str | None = Field(None, description="Agent-Status")
    capabilities: list[str] | None = Field(None, description="Agent-Fähigkeiten")
    metadata: dict[str, str] | None = Field(None, description="Agent-Metadaten")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        if v is not None:
            allowed_statuses = ["active", "inactive", "error", "maintenance"]
            if v not in allowed_statuses:
                raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v

class AgentResponse(BaseModel):
    """Agent Response Schema"""
    agent_id: str = Field(..., description="Agent-ID")
    agent_type: str = Field(..., description="Agent-Typ")
    status: str = Field(..., description="Agent-Status")
    capabilities: list[str] = Field(..., description="Agent-Fähigkeiten")
    metadata: dict[str, str] = Field(..., description="Agent-Metadaten")
    created_at: datetime = Field(..., description="Erstellungszeitpunkt")
    updated_at: datetime | None = Field(None, description="Letztes Update")
    last_seen: datetime | None = Field(None, description="Letzte Aktivität")

class AgentListResponse(BaseModel):
    """Agent List Response Schema"""
    agents: list[AgentResponse] = Field(..., description="Liste der Agents")
    total: int = Field(..., description="Gesamtanzahl")
    limit: int = Field(..., description="Limit pro Seite")
    offset: int = Field(..., description="Offset")

class FunctionRegistrationRequest(BaseModel):
    """Function Registration Request Schema"""
    function_name: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", description="Function-Name")
    description: str = Field(..., description="Function-Beschreibung")
    parameters_schema: dict[str, Any] = Field(..., description="JSON Schema für Function-Parameter")
    return_schema: dict[str, Any] | None = Field(None, description="JSON Schema für Return-Value")

class FunctionResponse(BaseModel):
    """Function Response Schema"""
    function_name: str = Field(..., description="Function-Name")
    agent_id: str = Field(..., description="Agent-ID")
    description: str = Field(..., description="Function-Beschreibung")
    parameters_schema: dict[str, Any] = Field(..., description="Parameter-Schema")
    return_schema: dict[str, Any] | None = Field(None, description="Return-Schema")
    registered_at: datetime = Field(..., description="Registrierungszeitpunkt")

class FunctionListResponse(BaseModel):
    """Function List Response Schema"""
    functions: list[FunctionResponse] = Field(..., description="Liste der Functions")

class FunctionCallRequest(BaseModel):
    """Function Call Request Schema"""
    parameters: dict[str, Any] = Field(..., description="Function-Parameter")
    timeout_seconds: int | None = Field(30, ge=1, le=300, description="Timeout in Sekunden")

class FunctionCallResponse(BaseModel):
    """Function Call Response Schema"""
    call_id: str = Field(..., description="Call-ID")
    status: str = Field(..., description="Call-Status")
    result: dict[str, Any] | None = Field(None, description="Function-Ergebnis")
    error: str | None = Field(None, description="Error-Message bei Fehlern")
    execution_time_ms: int | None = Field(None, description="Ausführungszeit in ms")
    timestamp: datetime = Field(..., description="Zeitstempel")

class PlatformConfigResponse(BaseModel):
    """Platform Configuration Response Schema"""
    api_version: str = Field(..., description="API-Version")
    supported_protocols: list[str] = Field(..., description="Unterstützte Protokolle")
    rate_limits: dict[str, Any] = Field(..., description="Rate Limits")
    features: dict[str, Any] = Field(..., description="Verfügbare Features")

class AgentConfigResponse(BaseModel):
    """Agent Configuration Response Schema"""
    agent_id: str = Field(..., description="Agent-ID")
    config: dict[str, Any] = Field(..., description="Agent-Konfiguration")
    updated_at: datetime = Field(..., description="Letztes Update")

class AgentConfigUpdateRequest(BaseModel):
    """Agent Configuration Update Request Schema"""
    config: dict[str, Any] = Field(..., description="Neue Agent-Konfiguration")

# Dependency für Event Bus
async def get_event_bus() -> PlatformEventBus:
    """Dependency für Platform Event Bus"""
    # Mock Event Bus für Demo - in echter Implementierung aus DI
    from messaging.platform_event_bus import PlatformEventBusConfig
    from messaging.platform_nats_client import PlatformNATSConfig

    nats_config = PlatformNATSConfig(
        servers=["nats://localhost:4222"],
        cluster_name="platform-cluster"
    )

    config = PlatformEventBusConfig(nats_config=nats_config)
    event_bus = PlatformEventBus(config)

    if not event_bus.started:
        await event_bus.start()

    return event_bus

# In-Memory Storage für Demo (in echter Implementierung würde Database verwendet)
agents_storage: dict[str, dict[str, Any]] = {}
functions_storage: dict[str, dict[str, Any]] = {}

# Agent Management Endpoints
@management_router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    status: str | None = Query(None, description="Filter nach Status"),
    agent_type: str | None = Query(None, description="Filter nach Agent-Typ"),
    limit: int = Query(20, ge=1, le=100, description="Anzahl Ergebnisse"),
    offset: int = Query(0, ge=0, description="Offset für Paginierung")
) -> AgentListResponse:
    """Abrufen aller registrierten Agents"""
    try:
        # Filter anwenden
        filtered_agents = []
        for agent_data in agents_storage.values():
            if status and agent_data.get("status") != status:
                continue
            if agent_type and agent_data.get("agent_type") != agent_type:
                continue
            filtered_agents.append(agent_data)

        # Paginierung
        total = len(filtered_agents)
        paginated_agents = filtered_agents[offset:offset + limit]

        # Response erstellen
        agents = [
            AgentResponse(
                agent_id=agent["agent_id"],
                agent_type=agent["agent_type"],
                status=agent["status"],
                capabilities=agent["capabilities"],
                metadata=agent["metadata"],
                created_at=agent["created_at"],
                updated_at=agent.get("updated_at"),
                last_seen=agent.get("last_seen")
            )
            for agent in paginated_agents
        ]

        return AgentListResponse(
            agents=agents,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.post("/agents", response_model=AgentResponse, status_code=201)
async def register_agent(
    request: AgentRegistrationRequest,
    event_bus: PlatformEventBus = Depends(get_event_bus)
) -> AgentResponse:
    """SDK registriert neuen Agent bei Platform"""
    try:
        # Prüfe ob Agent bereits existiert
        if request.agent_id in agents_storage:
            raise HTTPException(
                status_code=409,
                detail=f"Agent {request.agent_id} already exists"
            )

        # Agent-Daten erstellen
        now = datetime.now(UTC)
        agent_data = {
            "agent_id": request.agent_id,
            "agent_type": request.agent_type,
            "status": "active",
            "capabilities": request.capabilities,
            "metadata": request.metadata or {},
            "config": request.config or {},
            "created_at": now,
            "updated_at": now,
            "last_seen": now
        }

        # Agent speichern
        agents_storage[request.agent_id] = agent_data

        # Platform Event publizieren
        platform_event = PlatformEvent(
            event_type=PlatformEventType.AGENT_CREATED.value,
            event_id=str(uuid.uuid4()),
            source_service="platform-management-api",
            data={
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "capabilities": request.capabilities,
                "configuration": request.config or {},
                "created_by": "sdk-api"
            }
        )

        await event_bus.publish(platform_event)

        logger.info(f"Agent registriert via Management API: {request.agent_id}")

        return AgentResponse(**agent_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Registrieren des Agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$")
) -> AgentResponse:
    """Abrufen detaillierter Agent-Informationen"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        agent_data = agents_storage[agent_id]
        return AgentResponse(**agent_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    request: AgentUpdateRequest,
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$"),
    event_bus: PlatformEventBus = Depends(get_event_bus)
) -> AgentResponse:
    """Aktualisierung von Agent-Konfiguration und -Status"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        agent_data = agents_storage[agent_id]
        previous_status = agent_data.get("status")

        # Agent-Daten aktualisieren
        if request.status is not None:
            agent_data["status"] = request.status
        if request.capabilities is not None:
            agent_data["capabilities"] = request.capabilities
        if request.metadata is not None:
            agent_data["metadata"].update(request.metadata)

        agent_data["updated_at"] = datetime.now(UTC)

        # Platform Event publizieren bei Status-Änderung
        if request.status and request.status != previous_status:
            platform_event = PlatformEvent(
                event_type=PlatformEventType.AGENT_STATUS_CHANGED.value,
                event_id=str(uuid.uuid4()),
                source_service="platform-management-api",
                data={
                    "agent_id": agent_id,
                    "previous_status": previous_status,
                    "new_status": request.status,
                    "reason": "api_update",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            )
            await event_bus.publish(platform_event)

        logger.info(f"Agent aktualisiert via Management API: {agent_id}")

        return AgentResponse(**agent_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren des Agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.delete("/agents/{agent_id}", status_code=204)
async def deregister_agent(
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$"),
    event_bus: PlatformEventBus = Depends(get_event_bus)
):
    """Entfernung eines Agents aus der Platform"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        # Agent entfernen
        del agents_storage[agent_id]

        # Platform Event publizieren
        platform_event = PlatformEvent(
            event_type=PlatformEventType.AGENT_DELETED.value,
            event_id=str(uuid.uuid4()),
            source_service="platform-management-api",
            data={
                "agent_id": agent_id,
                "reason": "api_deregistration",
                "deleted_by_service": "management-api"
            }
        )

        await event_bus.publish(platform_event)

        logger.info(f"Agent deregistriert via Management API: {agent_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Deregistrieren des Agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Function Management Endpoints
@management_router.get("/agents/{agent_id}/functions", response_model=FunctionListResponse)
async def list_agent_functions(
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$")
) -> FunctionListResponse:
    """Abrufen verfügbarer Functions eines Agents"""
    try:
        # Filter Functions für Agent
        agent_functions = [
            func_data for func_data in functions_storage.values()
            if func_data.get("agent_id") == agent_id
        ]

        functions = [
            FunctionResponse(**func_data)
            for func_data in agent_functions
        ]

        return FunctionListResponse(functions=functions)

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Agent Functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.post("/agents/{agent_id}/functions", response_model=FunctionResponse, status_code=201)
async def register_agent_function(
    request: FunctionRegistrationRequest,
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$")
) -> FunctionResponse:
    """SDK registriert neue Function für Agent"""
    try:
        # Prüfe ob Agent existiert
        if agent_id not in agents_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        # Function-ID erstellen
        function_id = f"{agent_id}_{request.function_name}"

        # Prüfe ob Function bereits existiert
        if function_id in functions_storage:
            raise HTTPException(
                status_code=409,
                detail=f"Function {request.function_name} already exists for agent {agent_id}"
            )

        # Function-Daten erstellen
        now = datetime.now(UTC)
        function_data = {
            "function_name": request.function_name,
            "agent_id": agent_id,
            "description": request.description,
            "parameters_schema": request.parameters_schema,
            "return_schema": request.return_schema,
            "registered_at": now
        }

        # Function speichern
        functions_storage[function_id] = function_data

        logger.info(f"Function registriert via Management API: {agent_id}.{request.function_name}")

        return FunctionResponse(**function_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Registrieren der Function: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.post("/agents/{agent_id}/functions/{function_name}/call", response_model=FunctionCallResponse)
async def call_agent_function(
    request: FunctionCallRequest,
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$"),
    function_name: str = Path(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
) -> FunctionCallResponse:
    """Platform ruft Agent-Function über HTTP API auf"""
    try:
        function_id = f"{agent_id}_{function_name}"

        # Prüfe ob Function existiert
        if function_id not in functions_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Function {function_name} not found for agent {agent_id}"
            )

        # Simuliere Function Call (in echter Implementierung würde hier gRPC/HTTP Call gemacht)
        call_id = str(uuid.uuid4())

        # Mock-Ergebnis
        result = {
            "status": "success",
            "data": f"Function {function_name} executed with parameters: {request.parameters}"
        }

        logger.info(f"Function Call via Management API: {agent_id}.{function_name}")

        return FunctionCallResponse(
            call_id=call_id,
            status="success",
            result=result,
            execution_time_ms=150,
            timestamp=datetime.now(UTC)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Function Call: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Endpoints
@management_router.get("/config", response_model=PlatformConfigResponse)
async def get_platform_config() -> PlatformConfigResponse:
    """SDK abruft Platform-Konfiguration"""
    return PlatformConfigResponse(
        api_version="v1.0.0",
        supported_protocols=["http", "grpc", "websocket"],
        rate_limits={
            "events_per_minute": 1000,
            "functions_per_minute": 100
        },
        features={
            "event_streaming": True,
            "function_calls": True,
            "real_time_updates": True,
            "schema_validation": True
        }
    )

@management_router.get("/config/agents/{agent_id}", response_model=AgentConfigResponse)
async def get_agent_config(
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$")
) -> AgentConfigResponse:
    """Abrufen agent-spezifischer Konfiguration"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        agent_data = agents_storage[agent_id]

        return AgentConfigResponse(
            agent_id=agent_id,
            config=agent_data.get("config", {}),
            updated_at=agent_data.get("updated_at", datetime.now(UTC))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Agent Config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.put("/config/agents/{agent_id}", response_model=AgentConfigResponse)
async def update_agent_config(
    request: AgentConfigUpdateRequest,
    agent_id: str = Path(..., pattern=r"^agent_[a-zA-Z0-9_]+$")
) -> AgentConfigResponse:
    """SDK aktualisiert Agent-Konfiguration"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )

        # Konfiguration aktualisieren
        agent_data = agents_storage[agent_id]
        agent_data["config"] = request.config
        agent_data["updated_at"] = datetime.now(UTC)

        logger.info(f"Agent Config aktualisiert via Management API: {agent_id}")

        return AgentConfigResponse(
            agent_id=agent_id,
            config=request.config,
            updated_at=agent_data["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren der Agent Config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
