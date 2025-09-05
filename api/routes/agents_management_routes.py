"""Agent Management Endpoints – register/update/deprecate, capabilities, scaling hints, health/stats.

Diese Routen ergänzen die vorhandenen `agents_routes` um Management-Funktionalität.
Die Implementierung nutzt die vorhandene Dynamic Registry und Metadata-Services.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

# Vereinheitlicht: Unified Auth + require_scopes
# from auth.simple_enterprise_auth import Scope, require_scope
from agents.metadata.agent_metadata import AgentMetadata, CapabilityStatus
from agents.metadata.service import metadata_service
from agents.registry.dynamic_registry import dynamic_registry
from api.middleware.scope_middleware import require_scopes
from auth.enterprise_auth import Scope
from auth.unified_enterprise_auth import require_unified_auth
from kei_logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agents-mgmt", tags=["agents-management"])


async def perform_agent_handshake(agent_id: str, heartbeat_url: str | None) -> bool:
    """Führt einen Handshake mit dem Agent durch.

    Args:
        agent_id: Agent-ID
        heartbeat_url: URL für Heartbeat-Checks

    Returns:
        True wenn Handshake erfolgreich, False sonst
    """
    if not heartbeat_url:
        logger.warning(f"Kein Heartbeat-URL für Agent {agent_id} angegeben")
        return False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(heartbeat_url)

            if response.status_code == 200:
                logger.info(f"✅ Agent-Handshake erfolgreich: {agent_id}")
                return True
            logger.warning(f"⚠️ Agent-Handshake fehlgeschlagen: {agent_id} (Status: {response.status_code})")
            return False

    except Exception as e:
        logger.warning(f"⚠️ Agent-Handshake Fehler: {agent_id} - {e}")
        return False


class AgentRegisterRequest(BaseModel):
    """Registrierung eines (externen) Agents in der Dynamic Registry."""

    agent_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str | None = Field(None)
    capabilities: list[str] = Field(default_factory=list)
    category: str | None = Field(default="custom")
    owner: str | None = Field(default=None, description="Agent Owner (1..128 sichtbare Zeichen)")
    tenant: str | None = Field(default=None, description="Tenant (a-z0-9-_ 1..64)")
    tags: list[str] = Field(default_factory=list, description="Tags [a-zA-Z0-9_-]{1,50}, max 50")
    advertise_capabilities: list[str] | None = Field(default=None, description="Optionale initiale Capabilities (IDs)")
    heartbeat_url: str | None = Field(default=None, description="URL für Agent-Heartbeat (z.B. http://agent:8080/heartbeat)")


class AgentRegisterResponse(BaseModel):
    """Antwort auf eine Agent-Registrierung."""

    success: bool
    agent_id: str
    registered_at: datetime


class AgentCapabilityUpdateRequest(BaseModel):
    """Aktualisiert Capabilities eines Agents."""

    capabilities: list[str] = Field(default_factory=list)


class VersionUpdateRequest(BaseModel):
    """Setzt die Agent-Version (SemVer-validiert) und optional Links."""

    version: str = Field(..., min_length=1, description="Neue Version (SemVer)")
    migration_guide_url: str | None = Field(default=None)
    changelog_url: str | None = Field(default=None)
    pinned: bool | None = Field(default=None, description="Ob diese Version gepinnt werden soll")


class VersionEvaluationRequest(BaseModel):
    """Anfrage zur Bewertung von Deprecation/Migration gemäß SemVer."""

    target_version: str = Field(..., min_length=1, description="Zielversion (SemVer)")


class AgentIdentityRequest(BaseModel):
    """Validierte Identitätsdaten (Owner/Tenant/Tags)."""

    owner: str | None = Field(default=None, description="Agent Owner (1..128 sichtbare Zeichen)")
    tenant: str | None = Field(default=None, description="Tenant (a-z0-9-_ 1..64)")
    tags: list[str] = Field(default_factory=list, description="Tags [a-zA-Z0-9_-]{1,50}, max 50")








@router.post(
    "/register",
    response_model=AgentRegisterResponse,
)
async def register_agent(
    request: AgentRegisterRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_WRITE.value]))
) -> AgentRegisterResponse:
    """Registriert oder aktualisiert einen Agent in der Registry (Best-Effort)."""
    try:
        # Log Agent-Registrierung
        logger.info(
            f"🤖 Agent-Registrierung: {request.agent_id}",
            extra={
                "agent_id": request.agent_id,
                "agent_name": request.name,
                "description": request.description,
                "capabilities": request.capabilities or request.advertise_capabilities or [],
                "owner": request.owner,
                "tenant": request.tenant,
                "event_type": "agent_registration"
            }
        )

        # Handshake: Prüfe ob Agent erreichbar ist
        handshake_success = await perform_agent_handshake(request.agent_id, request.heartbeat_url)

        if not handshake_success:
            logger.warning(
                f"⚠️ Agent-Handshake fehlgeschlagen: {request.agent_id}",
                extra={"agent_id": request.agent_id, "event_type": "agent_handshake_failed"}
            )
            # Registrierung trotzdem erlauben, aber Status auf "unreachable" setzen

        # Metadaten in Registry-Struktur ablegen mit Heartbeat-Info
        import time

        # Erstelle Agent-Metadaten für Registry
        agent_metadata = AgentMetadata(
            agent_id=request.agent_id,
            agent_name=request.name,
            framework_type="CUSTOM",  # Default framework type
            framework_version="1.0.0",
            owner=request.owner,
            tenant=request.tenant,
            tags=request.tags or [],
        )

        # Füge Capabilities hinzu
        if request.capabilities or request.advertise_capabilities:
            capabilities = request.capabilities or request.advertise_capabilities or []
            for cap in capabilities:
                agent_metadata.add_capability({
                    "id": cap,
                    "name": cap,
                    "description": f"Capability: {cap}",
                    "status": "available" if handshake_success else "unreachable"
                })

        # Registriere Agent in Dynamic Registry
        try:
            await dynamic_registry.register_agent(agent_metadata)
        except Exception as e:
            logger.warning(f"Agent-Registry-Registrierung fehlgeschlagen: {e}")
            # Fallback: Speichere in temporärer Struktur
            if not hasattr(dynamic_registry, "_temp_agents"):
                dynamic_registry._temp_agents = {}
            dynamic_registry._temp_agents[request.agent_id] = type("_AdHocAgent", (), {
                "id": request.agent_id,
                "name": request.name,
                "description": request.description or "",
                "capabilities": request.capabilities or request.advertise_capabilities or [],
                "status": "available" if handshake_success else "unreachable",
                "owner": request.owner,
                "tenant": request.tenant,
                "tags": request.tags,
                "heartbeat_url": request.heartbeat_url,
                "last_heartbeat": time.time() if handshake_success else None,
                "registered_at": time.time(),
            })()
        # Optional gleich Capabilities bewerben (advertise)
        try:
            if request.advertise_capabilities:
                from api.routes.capabilities_routes import (
                    CapabilityAdvertiseItem,
                    CapabilityAdvertiseRequest,
                    advertise_capabilities,
                )
                body = CapabilityAdvertiseRequest(
                    capabilities=[CapabilityAdvertiseItem(id=c, name=c) for c in request.advertise_capabilities],
                    replace=False,
                )
                await advertise_capabilities(request.agent_id, body)
        except Exception:
            pass
        return AgentRegisterResponse(success=True, agent_id=request.agent_id, registered_at=datetime.utcnow())
    except Exception as e:
        logger.exception(f"Agent-Registrierung fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Agent-Registrierung fehlgeschlagen")


@router.put(
    "/{agent_id}/capabilities",
)
async def update_agent_capabilities(
    agent_id: str,
    body: AgentCapabilityUpdateRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_WRITE.value]))
) -> dict[str, Any]:
    """Aktualisiert Capabilities (vereinfachte Sicht in Registry)."""
    agent = await dynamic_registry.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    try:
        agent.capabilities = list(body.capabilities)
        # Update agent in registry
        try:
            await dynamic_registry.update_agent(agent_id, agent)
        except Exception as e:
            logger.warning(f"Agent-Registry-Update fehlgeschlagen: {e}")
            # Fallback: Update in temporary structure
            if hasattr(dynamic_registry, "_temp_agents") and agent_id in dynamic_registry._temp_agents:
                dynamic_registry._temp_agents[agent_id] = agent
        # Deprecation‑Hinweise automatisch befüllen, falls Metadata vorhanden
        meta = metadata_service.get_metadata(agent_id)
        deprecations: dict[str, str] = {}
        if meta:
            try:
                for cap_id in body.capabilities:
                    cap = meta.get_capability(cap_id)
                    if cap:
                        hint = meta.evaluate_capability_deprecation(cap)
                        if hint:
                            deprecations[cap_id] = hint
                            cap.status = cap.status or CapabilityStatus.AVAILABLE  # type: ignore[name-defined]
                            cap.status = CapabilityStatus.DEPRECATED  # type: ignore[name-defined]
                            meta.available_capabilities[cap_id] = cap
            except Exception:
                pass
        return {"agent_id": agent_id, "capabilities": list(body.capabilities), "deprecations": deprecations}
    except Exception as e:
        logger.exception(f"Capabilities Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Capabilities Update fehlgeschlagen")


@router.put(
    "/{agent_id}/identity",
)
async def update_agent_identity(
    agent_id: str,
    body: AgentIdentityRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_WRITE.value]))
) -> dict[str, Any]:
    """Setzt Owner/Tenant/Tags am Agent, inklusive Format-Validierung."""
    agent = await dynamic_registry.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")

    from agents.metadata.agent_metadata import AgentMetadata, FrameworkType
    temp = AgentMetadata(
        agent_id=agent_id,
        agent_name=getattr(agent, "name", agent_id),
        framework_type=FrameworkType.CUSTOM_MCP,
        framework_version="1.0.0",
    )
    try:
        temp.assign_identity(owner=body.owner, tenant=body.tenant, tags=body.tags)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

    try:
        agent.owner = temp.owner
        agent.tenant = temp.tenant
        agent.tags = temp.tags
        # Update agent in registry
        try:
            await dynamic_registry.update_agent(agent_id, agent)
        except Exception as e:
            logger.warning(f"Agent-Registry-Update fehlgeschlagen: {e}")
            # Fallback: Update in temporary structure
            if hasattr(dynamic_registry, "_temp_agents") and agent_id in dynamic_registry._temp_agents:
                dynamic_registry._temp_agents[agent_id] = agent
        return {"agent_id": agent_id, "owner": temp.owner, "tenant": temp.tenant, "tags": temp.tags}
    except Exception as e:
        logger.exception(f"Identity Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Identity Update fehlgeschlagen")


@router.get(
    "/{agent_id}/health",
)
async def get_agent_health(
    agent_id: str = Path(..., min_length=1),
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_READ.value]))
) -> dict[str, Any]:
    """Gibt einfachen Health-Status eines Agents zurück (Registry-Status)."""
    agent = await dynamic_registry.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    status = getattr(agent, "status", "unknown")
    return {"agent_id": agent_id, "status": status, "checked_at": datetime.utcnow().isoformat()}


class HeartbeatRequest(BaseModel):
    """Heartbeat-/Readiness-Informationen eines Agents."""

    health: str = Field(default="ok")
    readiness: str = Field(default="ready")
    queue_length: int | None = Field(default=None, ge=0)
    desired_concurrency: int | None = Field(default=None, ge=1)
    current_concurrency: int | None = Field(default=None, ge=0)
    hints: dict[str, Any] = Field(default_factory=dict)
    suspend: bool | None = Field(default=None, description="Optionaler Suspend-Flag (true=suspend, false=resume)")


@router.post(
    "/{agent_id}/heartbeat",
)
async def agent_heartbeat(
    agent_id: str,
    body: HeartbeatRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_WRITE.value]))
) -> dict[str, Any]:
    """Aktualisiert einfachen Heartbeat-Status in der Registry (Best-Effort)."""
    agent = await dynamic_registry.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    try:
        agent.status = body.health
        agent.readiness = body.readiness
        agent.queue_length = body.queue_length
        agent.desired_concurrency = body.desired_concurrency
        agent.concurrency = body.current_concurrency
        agent.hints = body.hints
        # Suspend/Resume Steuerung
        if body.suspend:
            agent.suspended = True
        elif body.suspend is False:
            agent.suspended = False
        # Update agent in registry
        try:
            await dynamic_registry.update_agent(agent_id, agent)
        except Exception as e:
            logger.warning(f"Agent-Registry-Update fehlgeschlagen: {e}")
            # Fallback: Update in temporary structure
            if hasattr(dynamic_registry, "_temp_agents") and agent_id in dynamic_registry._temp_agents:
                dynamic_registry._temp_agents[agent_id] = agent
        return {
            "agent_id": agent_id,
            "health": body.health,
            "readiness": body.readiness,
            "queue_length": body.queue_length,
            "desired_concurrency": body.desired_concurrency,
            "current_concurrency": body.current_concurrency,
            "suspended": getattr(agent, "suspended", False),
        }
    except Exception as e:
        logger.exception(f"Heartbeat Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Heartbeat Update fehlgeschlagen")


@router.post(
    "/{agent_id}/version/evaluate",
)
async def evaluate_agent_version(
    agent_id: str,
    body: VersionEvaluationRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_READ.value]))
) -> dict[str, Any]:
    """Berechnet Deprecation‑Warning und liefert Migrations-/Changelog‑Links."""
    meta = metadata_service.get_metadata(agent_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata nicht gefunden")
    meta.evaluate_deprecation_against(body.target_version)
    return {
        "agent_id": agent_id,
        "current_version": meta.agent_version,
        "target_version": body.target_version,
        "deprecation_warning": meta.deprecation_warning,
        "migration_guide_url": meta.migration_guide_url,
        "changelog_url": meta.changelog_url,
    }


@router.put(
    "/{agent_id}/version",
)
async def update_agent_version(
    agent_id: str,
    body: VersionUpdateRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_WRITE.value]))
) -> dict[str, Any]:
    """Setzt die Agent-Version nach SemVer‑Prüfung und optional Migrations-/Changelog‑Links."""
    meta = metadata_service.get_metadata(agent_id)
    if not meta:
        from agents.metadata.agent_metadata import FrameworkType
        meta = AgentMetadata(
            agent_id=agent_id,
            agent_name=agent_id,
            framework_type=FrameworkType.CUSTOM_MCP,
            framework_version="1.0.0",
        )
        metadata_service.register_metadata(meta)

    try:
        meta.set_version(body.version)
        if body.migration_guide_url is not None:
            meta.migration_guide_url = body.migration_guide_url
        if body.changelog_url is not None:
            meta.changelog_url = body.changelog_url
        # Optional Version Pinning
        if body.pinned is not None:
            with contextlib.suppress(Exception):
                meta.pinned = bool(body.pinned)
        return {"agent_id": agent_id, "version": meta.agent_version,
                "migration_guide_url": meta.migration_guide_url,
                "changelog_url": meta.changelog_url}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))


class ScalingHintsRequest(BaseModel):
    """Skalierungshinweise (z. B. gewünschte Concurrency)."""

    desired_concurrency: int = Field(..., ge=1, le=1024)
    max_queue_length: int | None = Field(default=None, ge=0)
    rollout: str | None = Field(default=None, description="Rollout-Strategie: canary|all|none")
    rollout_percentage: int | None = Field(default=None, ge=0, le=100, description="Canary-Prozentsatz (0-100)")


@router.put(
    "/{agent_id}/scaling-hints",
)
async def update_scaling_hints(
    agent_id: str,
    body: ScalingHintsRequest,
    _auth=Depends(require_unified_auth),
    _scopes=Depends(lambda request: require_scopes(request, [Scope.AGENTS_WRITE.value]))
) -> dict[str, Any]:
    """Setzt skalierungsrelevante Hinweise am Agent in der Registry."""
    agent = await dynamic_registry.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    try:
        hints = getattr(agent, "hints", {}) or {}
        hints.update({
            "desired_concurrency": body.desired_concurrency,
            "max_queue_length": body.max_queue_length,
            "rollout": body.rollout,
            "rollout_percentage": body.rollout_percentage,
        })
        agent.hints = hints
        # Update agent in registry
        try:
            await dynamic_registry.update_agent(agent_id, agent)
        except Exception as e:
            logger.warning(f"Agent-Registry-Update fehlgeschlagen: {e}")
            # Fallback: Update in temporary structure
            if hasattr(dynamic_registry, "_temp_agents") and agent_id in dynamic_registry._temp_agents:
                dynamic_registry._temp_agents[agent_id] = agent
        return {"agent_id": agent_id, "hints": hints}
    except Exception as e:
        logger.exception(f"Scaling Hints Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Scaling Hints Update fehlgeschlagen")
