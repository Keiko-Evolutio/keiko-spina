# backend/api/routes/agents_routes.py
"""Agent Management API Routes für Azure AI Foundry."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException, Path
from pydantic import BaseModel, Field

from observability.deadline import run_with_deadline
from policy_engine import policy_engine
from security.kei_mcp_auth import require_auth
from services.limits.rate_limiter import check_agent_capability_quota

from .base import (
    check_agents_integration,
    create_health_response,
    create_router,
    get_agent_system_status,
)
from .common import AGENT_RESPONSES

# Router-Konfiguration
router = create_router("/agents", ["agents"])
router.responses.update(AGENT_RESPONSES)


# Datenmodelle
class AgentExecutionRequest(BaseModel):
    """Request für Agent-Task-Ausführung."""
    task_description: str = Field(..., description="Beschreibung der auszuführenden Aufgabe")
    context: dict[str, Any] | None = Field(None, description="Zusätzlicher Kontext")
    parameters: dict[str, Any] | None = Field(None, description="Ausführungsparameter")
    timeout: int | None = Field(60, ge=1, le=300, description="Timeout in Sekunden")


class AgentExecutionResponse(BaseModel):
    """Response für Agent-Task-Ausführung."""
    agent_id: str = Field(..., description="ID des ausführenden Agents")
    task_id: str = Field(..., description="Eindeutige Task-ID")
    status: str = Field(..., description="Ausführungsstatus")
    result: Any | None = Field(None, description="Ausführungsergebnis")
    started_at: datetime = Field(..., description="Startzeit")
    completed_at: datetime | None = Field(None, description="Endzeit")
    duration_ms: int | None = Field(None, description="Ausführungsdauer in Millisekunden")
    error: str | None = Field(None, description="Fehlermeldung falls vorhanden")


class AgentInfo(BaseModel):
    """Informationen über einen Agent."""
    id: str = Field(..., description="Agent-ID")
    name: str = Field(..., description="Agent-Name")
    description: str = Field(..., description="Agent-Beschreibung")
    capabilities: list[str] = Field(..., description="Agent-Fähigkeiten")
    status: str = Field(..., description="Agent-Status")
    created_at: datetime = Field(..., description="Erstellungszeit")


# Helper Funktionen
async def get_agents_safely() -> list[AgentInfo]:
    """Holt verfügbare Agents mit Fehlerbehandlung."""
    if not check_agents_integration():
        return []

    try:
        from agents import get_all_agents
        agents = await get_all_agents()
        return [
            AgentInfo(
                id=agent.get("id", f"agent_{i}"),
                name=agent.get("name", f"Agent {i}"),
                description=agent.get("description", "Keine Beschreibung"),
                capabilities=agent.get("capabilities", []),
                status=agent.get("status", "unknown"),
                created_at=agent.get("created_at", datetime.now())
            )
            for i, agent in enumerate(agents, 1)
        ]
    except Exception as e:
        from kei_logging import get_logger
        logger = get_logger(__name__)
        logger.warning(f"⚠️ Agents abrufen fehlgeschlagen: {e}")
        return []


async def execute_agent_safely(task: AgentExecutionRequest) -> AgentExecutionResponse:
    """Führt Agent-Task sicher aus."""
    task_id = f"task_{uuid4().hex[:8]}"
    start_time = datetime.now()

    if not check_agents_integration():
        return AgentExecutionResponse(
            agent_id="fallback",
            task_id=task_id,
            status="error",
            started_at=start_time,
            error="Agent System nicht verfügbar"
        )

    try:
        from agents import execute_agent_task, find_best_agent_for_task

        # Besten Agent finden
        agent_id, _ = await find_best_agent_for_task(task.task_description)

        # Task ausführen
        result = await execute_agent_task(
            agent_id=agent_id,
            task=task.task_description,
            context=task.context or {},
            parameters=task.parameters or {},
            timeout=task.timeout
        )

        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        return AgentExecutionResponse(
            agent_id=agent_id,
            task_id=task_id,
            status="success",
            result=result,
            started_at=start_time,
            completed_at=end_time,
            duration_ms=duration
        )

    except Exception as e:
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        return AgentExecutionResponse(
            agent_id="unknown",
            task_id=task_id,
            status="error",
            started_at=start_time,
            completed_at=end_time,
            duration_ms=duration,
            error=str(e)
        )


# API Endpunkte
@router.get("/", response_model=list[AgentInfo], dependencies=[Depends(require_auth)])
async def list_agents():
    """Listet alle verfügbaren Agents."""
    return await get_agents_safely()


@router.get("/{agent_id}", response_model=AgentInfo, dependencies=[Depends(require_auth)])
async def get_agent_details(agent_id: str = Path(..., description="Agent-ID")):
    """Holt Details eines spezifischen Agents."""
    agents = await get_agents_safely()
    agent = next((a for a in agents if a.id == agent_id), None)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")

    return agent


@router.post("/execute", response_model=AgentExecutionResponse, dependencies=[Depends(require_auth)])
async def execute_task(request: AgentExecutionRequest):
    """Führt eine Aufgabe mit dem besten verfügbaren Agent aus."""
    # Policy: Safety/PII vor Ausführung evaluieren; redaktierte Requestdaten verwenden
    tenant = None
    try:
        from fastapi import Request as _Req  # type: ignore
    except Exception:
        _Req = None  # type: ignore
    # Best-effort: Tenant kommt regulär aus Headern in Middlewares; hier nicht verfügbar
    decision = policy_engine.evaluate(tenant_id=tenant, payload=request.dict(), operation="agents.execute")
    if not decision.allow:
        raise HTTPException(status_code=422, detail={"error": "policy_blocked", "reason": decision.reason})
    # Quota: Agent/Capability (best effort – Capability optional aus Parametern ableiten)
    try:
        capability_id = None
        if request.parameters and isinstance(request.parameters, dict):
            capability_id = request.parameters.get("operation") or request.parameters.get("capability_id")
        # Agent-ID erst nach Auswahl bekannt – hier best effort mit None; echte Enforcer können auch in execute_agent_task erfolgen
        allowed, rl_headers, retry_after, limited_scope = await check_agent_capability_quota(
            agent_id=None,
            capability_id=str(capability_id) if capability_id else None,
            tenant_id=None,
        )
        if not allowed:
            resp_headers = {**rl_headers}
            if retry_after is not None:
                resp_headers["Retry-After"] = str(retry_after)
            raise HTTPException(status_code=429, detail={"error": "rate_limited", "scope": limited_scope}, headers=resp_headers)  # type: ignore[arg-type]
    except HTTPException:
        raise
    except Exception:
        # Quota-Best-Effort: bei Fehlern nicht blockieren
        pass

    # Ausführung mit Deadline-Unterstützung und Budget-Headern an Downstream
    # (Downstream-Aufrufe nutzen traced HTTP-Clients, die Header aus build_outgoing_budget_headers mitnehmen können)
    return await run_with_deadline(execute_agent_safely(request))


@router.get("/health")
async def agents_health_check():
    """Health Check für Agent System."""
    agents = await get_agents_safely()

    additional_data = {
        "agents": {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.status == "active"]),
            "available_capabilities": list({
                cap for agent in agents for cap in agent.capabilities
            })
        }
    }

    agent_status = get_agent_system_status()
    if agent_status:
        additional_data["agent_system_status"] = agent_status

    return create_health_response(additional_data)
