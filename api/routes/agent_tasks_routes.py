"""Agent Task API – submit_task, cancel_task, status mit Idempotenz/Korrelation.

Diese Routen implementieren eine minimal praktikable Task-Verwaltung für Agents.
Die Ausführung delegiert an das bestehende `agents_routes.execute_agent_safely`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Path
from pydantic import BaseModel, Field

from api.routes.agents_routes import AgentExecutionRequest, execute_agent_safely
from kei_logging import get_logger
from security.kei_mcp_auth import require_auth

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agent-tasks", tags=["agent-tasks"])


class SubmitTaskRequest(BaseModel):
    """Request für Task-Submission."""

    task_description: str = Field(..., min_length=1, description="Aufgabenbeschreibung")
    context: dict[str, Any] | None = Field(default=None, description="Zusatzkontext")
    parameters: dict[str, Any] | None = Field(default=None, description="Parameter")
    timeout: int | None = Field(default=60, ge=1, le=300, description="Timeout Sekunden")


class SubmitTaskResponse(BaseModel):
    """Antwort für Task-Submission."""

    task_id: str
    correlation_id: str | None = None
    status: str = "submitted"
    queued_at: datetime


class TaskStatusResponse(BaseModel):
    """Antwort für Task-Statusabfrage."""

    task_id: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None


_TASK_STORE: dict[str, TaskStatusResponse] = {}


@router.post("/submit", response_model=SubmitTaskResponse, dependencies=[Depends(require_auth)])
async def submit_task(
    body: SubmitTaskRequest,
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-Id"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SubmitTaskResponse:
    """Registriert einen Task und startet optional die Ausführung sofort.

    Idempotenz: Bei gleichem `Idempotency-Key` wird die ursprüngliche `task_id`
    zurückgegeben, falls bereits registriert.
    """
    # Idempotenz: Wiederverwendung bestehender Task-ID
    if idempotency_key and idempotency_key in _TASK_STORE:
        existing = _TASK_STORE[idempotency_key]
        return SubmitTaskResponse(
            task_id=existing.task_id,
            correlation_id=x_correlation_id,
            status=existing.status,
            queued_at=datetime.now(),
        )

    task_id = idempotency_key or f"task_{uuid4().hex[:12]}"
    _TASK_STORE[task_id] = TaskStatusResponse(task_id=task_id, status="queued")

    # Sofortige Ausführung anstoßen (synchroner Pfad über bestehende API)
    try:
        exec_req = AgentExecutionRequest(
            task_description=body.task_description,
            context=body.context,
            parameters=body.parameters,
            timeout=body.timeout,
        )
        result = await execute_agent_safely(exec_req)
        _TASK_STORE[task_id] = TaskStatusResponse(
            task_id=task_id,
            status=result.status,
            started_at=result.started_at,
            completed_at=result.completed_at,
            result=result.result,
            error=result.error,
        )
    except Exception as e:
        logger.exception(f"Task-Ausführung fehlgeschlagen: {e}")
        _TASK_STORE[task_id] = TaskStatusResponse(task_id=task_id, status="error", error=str(e))

    return SubmitTaskResponse(task_id=task_id, correlation_id=x_correlation_id, queued_at=datetime.now())


@router.post("/{task_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_task(task_id: str = Path(..., min_length=1)) -> dict[str, Any]:
    """Markiert einen Task als storniert (vereinfachte Implementierung)."""
    if task_id not in _TASK_STORE:
        raise HTTPException(status_code=404, detail="Task nicht gefunden")
    state = _TASK_STORE[task_id]
    if state.status in {"success", "error"}:
        return {"task_id": task_id, "status": state.status}
    state.status = "cancelled"
    _TASK_STORE[task_id] = state
    return {"task_id": task_id, "status": "cancelled"}


@router.get("/{task_id}/status", response_model=TaskStatusResponse, dependencies=[Depends(require_auth)])
async def get_task_status(task_id: str = Path(..., min_length=1)) -> TaskStatusResponse:
    """Liefert Task-Status zurück."""
    state = _TASK_STORE.get(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task nicht gefunden")
    return state
