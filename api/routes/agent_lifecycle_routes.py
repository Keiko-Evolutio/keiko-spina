# backend/api/routes/agent_lifecycle_routes.py
"""FastAPI-Routen für Agent-Lifecycle-Management.

Stellt REST-API-Endpunkte für alle Lifecycle-Operationen bereit:
register, initialize, activate, suspend, resume, terminate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from agents.lifecycle import (
    AgentLifecycleState,
    EventType,
    TaskPriority,
    agent_lifecycle_manager,
)
from auth import require_auth
from kei_logging import get_logger
from observability import trace_function
from quotas_limits.quota_middleware import check_api_quota

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/agents/lifecycle", tags=["Agent Lifecycle"])


# Request/Response Models

class AgentRegistrationRequest(BaseModel):
    """Request-Model für Agent-Registrierung."""
    agent_id: str = Field(..., description="Eindeutige Agent-ID")
    capabilities: set[str] | None = Field(None, description="Initiale Capabilities")


class AgentInitializationRequest(BaseModel):
    """Request-Model für Agent-Initialisierung."""
    agent_id: str = Field(..., description="Agent-ID")
    session_id: str | None = Field(None, description="Session-ID")
    thread_id: str | None = Field(None, description="Thread-ID")
    user_id: str | None = Field(None, description="User-ID")


class AgentSuspendRequest(BaseModel):
    """Request-Model für Agent-Suspend."""
    agent_id: str = Field(..., description="Agent-ID")
    reason: str | None = Field(None, description="Grund für Suspend")
    wait_for_tasks: bool = Field(True, description="Warten auf Task-Completion")
    timeout_seconds: int = Field(60, description="Timeout für Task-Completion")


class AgentTerminateRequest(BaseModel):
    """Request-Model für Agent-Termination."""
    agent_id: str = Field(..., description="Agent-ID")
    reason: str | None = Field(None, description="Grund für Termination")
    force: bool = Field(False, description="Erzwinge Termination ohne Cleanup")


class CapabilityAdvertisementRequest(BaseModel):
    """Request-Model für Capability-Advertisement."""
    agent_id: str = Field(..., description="Agent-ID")
    capabilities: set[str] = Field(..., description="Zu bewerbende Capabilities")
    replace_existing: bool = Field(False, description="Ersetze bestehende Capabilities")


class TaskSubmissionRequest(BaseModel):
    """Request-Model für Task-Submission."""
    agent_id: str = Field(..., description="Agent-ID")
    task_type: str = Field(..., description="Task-Typ")
    payload: dict[str, Any] = Field(..., description="Task-Payload")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="Task-Priorität")
    timeout_seconds: int = Field(300, description="Task-Timeout")
    max_retries: int = Field(3, description="Maximale Retry-Anzahl")


class EventSubmissionRequest(BaseModel):
    """Request-Model für Event-Submission."""
    agent_id: str = Field(..., description="Agent-ID")
    event_type: EventType = Field(..., description="Event-Typ")
    data: dict[str, Any] = Field(default_factory=dict, description="Event-Daten")
    correlation_id: str | None = Field(None, description="Correlation-ID")


class LifecycleOperationResponse(BaseModel):
    """Response-Model für Lifecycle-Operationen."""
    success: bool = Field(..., description="Operation erfolgreich")
    agent_id: str = Field(..., description="Agent-ID")
    operation: str = Field(..., description="Durchgeführte Operation")
    current_state: str = Field(..., description="Aktueller Lifecycle-State")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str | None = Field(None, description="Zusätzliche Nachricht")
    error: str | None = Field(None, description="Fehlermeldung falls aufgetreten")


class AgentStateResponse(BaseModel):
    """Response-Model für Agent-State."""
    agent_id: str
    current_state: str
    previous_state: str | None
    state_changed_at: datetime
    advertised_capabilities: list[str]
    pending_tasks_count: int
    active_tasks_count: int
    completed_tasks_count: int
    total_tasks_processed: int
    total_errors: int
    last_heartbeat: datetime | None
    suspended_at: datetime | None
    suspend_reason: str | None


class AgentListResponse(BaseModel):
    """Response-Model für Agent-Liste."""
    agents: list[AgentStateResponse]
    total_count: int
    active_count: int
    suspended_count: int
    terminated_count: int


# API-Endpunkte

@router.post("/register", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.register")
async def register_agent(
    request: AgentRegistrationRequest,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Registriert neuen Agent."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        success = await agent_lifecycle_manager.register_agent(
            request.agent_id,
            request.capabilities
        )

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(request.agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=request.agent_id,
                operation="register",
                current_state=current_state,
                message=f"Agent {request.agent_id} erfolgreich registriert"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent-Registrierung fehlgeschlagen für {request.agent_id}"
        )

    except Exception as e:
        logger.exception(f"Agent-Registrierung-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/initialize", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.initialize")
async def initialize_agent(
    request: AgentInitializationRequest,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Initialisiert registrierten Agent."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        # Execution-Context erstellen falls Daten vorhanden
        context = None
        if any([request.session_id, request.thread_id, request.user_id]):
            from agents.protocols.base_agent_protocol import AgentExecutionContext
            context = AgentExecutionContext(
                session_id=request.session_id or "",
                thread_id=request.thread_id or "",
                user_id=request.user_id or user_id
            )

        success = await agent_lifecycle_manager.initialize_agent(request.agent_id, context)

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(request.agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=request.agent_id,
                operation="initialize",
                current_state=current_state,
                message=f"Agent {request.agent_id} erfolgreich initialisiert"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent-Initialisierung fehlgeschlagen für {request.agent_id}"
        )

    except Exception as e:
        logger.exception(f"Agent-Initialisierung-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/activate", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.activate")
async def activate_agent(
    agent_id: str,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Aktiviert initialisierten Agent."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        success = await agent_lifecycle_manager.activate_agent(agent_id)

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=agent_id,
                operation="activate",
                current_state=current_state,
                message=f"Agent {agent_id} erfolgreich aktiviert"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent-Aktivierung fehlgeschlagen für {agent_id}"
        )

    except Exception as e:
        logger.exception(f"Agent-Aktivierung-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/suspend", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.suspend")
async def suspend_agent(
    request: AgentSuspendRequest,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Suspendiert aktiven Agent."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        success = await agent_lifecycle_manager.suspend_agent(
            request.agent_id,
            request.reason,
            request.wait_for_tasks,
            request.timeout_seconds
        )

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(request.agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=request.agent_id,
                operation="suspend",
                current_state=current_state,
                message=f"Agent {request.agent_id} erfolgreich suspendiert"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent-Suspend fehlgeschlagen für {request.agent_id}"
        )

    except Exception as e:
        logger.exception(f"Agent-Suspend-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/resume", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.resume")
async def resume_agent(
    agent_id: str,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Nimmt suspendierten Agent wieder in Betrieb."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        success = await agent_lifecycle_manager.resume_agent(agent_id)

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=agent_id,
                operation="resume",
                current_state=current_state,
                message=f"Agent {agent_id} erfolgreich resumed"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent-Resume fehlgeschlagen für {agent_id}"
        )

    except Exception as e:
        logger.exception(f"Agent-Resume-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/terminate", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.terminate")
async def terminate_agent(
    request: AgentTerminateRequest,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Terminiert Agent gracefully."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        success = await agent_lifecycle_manager.terminate_agent(
            request.agent_id,
            request.reason,
            request.force
        )

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(request.agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=request.agent_id,
                operation="terminate",
                current_state=current_state,
                message=f"Agent {request.agent_id} erfolgreich terminiert"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent-Termination fehlgeschlagen für {request.agent_id}"
        )

    except Exception as e:
        logger.exception(f"Agent-Termination-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/advertise-capabilities", response_model=LifecycleOperationResponse)
@trace_function("api.agent_lifecycle.advertise_capabilities")
async def advertise_capabilities(
    request: CapabilityAdvertisementRequest,
    user_id: str = Depends(require_auth)
) -> LifecycleOperationResponse:
    """Bewirbt Agent-Capabilities dynamisch."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        success = await agent_lifecycle_manager.advertise_capabilities(
            request.agent_id,
            request.capabilities,
            request.replace_existing
        )

        if success:
            agent_state = agent_lifecycle_manager.get_agent_state(request.agent_id)
            current_state = agent_state.current_state.value if agent_state else "unknown"

            return LifecycleOperationResponse(
                success=True,
                agent_id=request.agent_id,
                operation="advertise_capabilities",
                current_state=current_state,
                message=f"Capabilities für Agent {request.agent_id} erfolgreich beworben"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Capability-Advertisement fehlgeschlagen für {request.agent_id}"
        )

    except Exception as e:
        logger.exception(f"Capability-Advertisement-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/agents", response_model=AgentListResponse)
@trace_function("api.agent_lifecycle.list_agents")
async def list_agents(
    user_id: str = Depends(require_auth)
) -> AgentListResponse:
    """Listet alle Agents mit ihren States auf."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        all_agents = agent_lifecycle_manager.get_all_agents()

        agent_responses = []
        active_count = 0
        suspended_count = 0
        terminated_count = 0

        for agent_id, agent_state in all_agents.items():
            agent_response = AgentStateResponse(
                agent_id=agent_id,
                current_state=agent_state.current_state.value,
                previous_state=agent_state.previous_state.value if agent_state.previous_state else None,
                state_changed_at=agent_state.state_changed_at,
                advertised_capabilities=list(agent_state.advertised_capabilities),
                pending_tasks_count=len(agent_state.pending_tasks),
                active_tasks_count=len(agent_state.active_tasks),
                completed_tasks_count=len(agent_state.completed_tasks),
                total_tasks_processed=agent_state.total_tasks_processed,
                total_errors=agent_state.total_errors,
                last_heartbeat=agent_state.last_heartbeat,
                suspended_at=agent_state.suspended_at,
                suspend_reason=agent_state.suspend_reason
            )
            agent_responses.append(agent_response)

            # Zähle States
            if agent_state.current_state == AgentLifecycleState.RUNNING:
                active_count += 1
            elif agent_state.current_state == AgentLifecycleState.SUSPENDED:
                suspended_count += 1
            elif agent_state.current_state == AgentLifecycleState.TERMINATED:
                terminated_count += 1

        return AgentListResponse(
            agents=agent_responses,
            total_count=len(agent_responses),
            active_count=active_count,
            suspended_count=suspended_count,
            terminated_count=terminated_count
        )

    except Exception as e:
        logger.exception(f"Agent-Liste-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/agents/{agent_id}", response_model=AgentStateResponse)
@trace_function("api.agent_lifecycle.get_agent")
async def get_agent_state(
    agent_id: str,
    user_id: str = Depends(require_auth)
) -> AgentStateResponse:
    """Gibt detaillierten Agent-State zurück."""
    try:
        await check_api_quota(user_id, "agent_lifecycle")

        agent_state = agent_lifecycle_manager.get_agent_state(agent_id)

        if not agent_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} nicht gefunden"
            )

        return AgentStateResponse(
            agent_id=agent_id,
            current_state=agent_state.current_state.value,
            previous_state=agent_state.previous_state.value if agent_state.previous_state else None,
            state_changed_at=agent_state.state_changed_at,
            advertised_capabilities=list(agent_state.advertised_capabilities),
            pending_tasks_count=len(agent_state.pending_tasks),
            active_tasks_count=len(agent_state.active_tasks),
            completed_tasks_count=len(agent_state.completed_tasks),
            total_tasks_processed=agent_state.total_tasks_processed,
            total_errors=agent_state.total_errors,
            last_heartbeat=agent_state.last_heartbeat,
            suspended_at=agent_state.suspended_at,
            suspend_reason=agent_state.suspend_reason
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Agent-State-API-Fehler: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
