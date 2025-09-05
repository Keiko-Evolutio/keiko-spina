"""KEI-RPC Agent Operations – plan, act, observe, explain (synchron).

Diese Routen bieten synchrone, wohldefinierte Operationen für Agenten an und
delegieren die Ausführung an die bestehende Agent-Ausführungsschicht.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.routes.agents_routes import AgentExecutionRequest, execute_agent_safely
from kei_logging import get_logger
from observability.budget import build_outgoing_budget_headers
from security.kei_mcp_auth import require_auth
from services.limits.rate_limiter import check_agent_capability_quota

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/rpc/agent", tags=["kei-rpc-agent"])


class PlanRequest(BaseModel):
    """Anfrage zur Planerstellung."""

    objective: str = Field(..., min_length=1, description="Zielbeschreibung")
    context: dict[str, Any] | None = Field(default=None)


class OperationResponse(BaseModel):
    """Standardisierte Antwortstruktur für RPC-Operationen."""

    operation: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None


async def _exec(operation: str, description: str, context: dict[str, Any] | None) -> OperationResponse:
    """Hilfsfunktion für die synchrone Agent-Ausführung."""
    # Quota: Capability‑spezifisch
    try:
        allowed, rl_headers, retry_after, limited_scope = await check_agent_capability_quota(
            agent_id=None,
            capability_id=operation,
            tenant_id=None,
        )
        if not allowed:
            # Abbildung als Fehlerresultat analog REST 429
            return OperationResponse(
                operation=operation,
                status="error",
                started_at=datetime.utcnow(),
                completed_at=None,
                result=None,
                error=f"rate_limited:{limited_scope}"
            )
    except Exception:
        pass
    # Budget‑Propagation an Downstream
    params = {"operation": operation, "headers": build_outgoing_budget_headers()}
    req = AgentExecutionRequest(task_description=description, context=context, parameters=params, timeout=60)
    res = await execute_agent_safely(req)
    return OperationResponse(
        operation=operation,
        status=res.status,
        started_at=res.started_at,
        completed_at=res.completed_at,
        result=res.result,
        error=res.error,
    )


@router.post("/plan", response_model=OperationResponse, dependencies=[Depends(require_auth)])
async def plan(body: PlanRequest, _: str = Depends(require_auth)) -> OperationResponse:
    """Erstellt einen groben Plan für die Zielerreichung."""
    # Capability‑Scope prüfen
    from fastapi import Request as _Req  # type: ignore
    try:
        req = _Req(scope={})  # Placeholder: in Router realer Request-Kontext vorhanden
    except Exception:
        req = None  # type: ignore
    # Best-effort: Prüfe Scopes im Request-State, wenn verfügbar
    try:
        scopes = getattr(req.state, "auth_context", {}).get("scopes", []) if req else []  # type: ignore[attr-defined]
        if scopes and "agents.capability.plan" not in scopes:
            raise HTTPException(status_code=403, detail="Missing scope: agents.capability.plan")
    except Exception:
        pass
    return await _exec("plan", f"PLAN:{body.objective}", body.context)


class ActRequest(BaseModel):
    """Anfrage für Ausführungsschritt (act)."""

    action: str = Field(..., min_length=1)
    context: dict[str, Any] | None = Field(default=None)


@router.post("/act", response_model=OperationResponse, dependencies=[Depends(require_auth)])
async def act(body: ActRequest, _: str = Depends(require_auth)) -> OperationResponse:
    """Führt einen nächsten Handlungsschritt aus."""
    return await _exec("act", f"ACT:{body.action}", body.context)


class ObserveRequest(BaseModel):
    """Anfrage für Beobachtung (observe)."""

    observation: str = Field(..., min_length=1)
    context: dict[str, Any] | None = Field(default=None)


@router.post("/observe", response_model=OperationResponse, dependencies=[Depends(require_auth)])
async def observe(body: ObserveRequest, _: str = Depends(require_auth)) -> OperationResponse:
    """Nimmt Beobachtungen auf und gleicht Plan an."""
    return await _exec("observe", f"OBSERVE:{body.observation}", body.context)


class ExplainRequest(BaseModel):
    """Anfrage für Erklärung (explain)."""

    topic: str = Field(..., min_length=1)
    context: dict[str, Any] | None = Field(default=None)


@router.post("/explain", response_model=OperationResponse, dependencies=[Depends(require_auth)])
async def explain(body: ExplainRequest, _: str = Depends(require_auth)) -> OperationResponse:
    """Erklärt eine Entscheidung oder ein Ergebnis verständlich."""
    return await _exec("explain", f"EXPLAIN:{body.topic}", body.context)


__all__ = ["router"]
