"""Capabilities Management Endpoints – Advertisement und Health/Readiness pro Capability.

Diese Routen ergänzen die vorhandenen Management-Funktionen um eine explizite
Anzeige (advertise) von Capabilities sowie Status-/Health-Abfragen auf Capability‑Ebene.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from agents.base_agent import AgentMetadata
from agents.constants import AgentStatus as CapabilityStatus
from agents.metadata.service import AgentMetadataService

# Compatibility aliases
MCPCapabilityDescriptor = dict  # Simplified for migration
metadata_service = AgentMetadataService()  # Use proper metadata service
from kei_logging import get_logger

# Optional import - Registry-System ist nicht immer verfügbar
try:
    from agents.registry.dynamic_registry import dynamic_registry
except ImportError:
    # Fallback wenn Registry nicht verfügbar ist
    dynamic_registry = None
from security.kei_mcp_auth import require_auth, require_rate_limit

logger = get_logger(__name__)

# Router-Konfiguration
router = APIRouter(prefix="/api/v1/agents-mgmt", tags=["agents-capabilities"])


class CapabilityAdvertiseItem(BaseModel):
    """Einzelne Capability für Advertisement.

    - status: Availability‑Status gemäß CapabilityStatus
    - version: SemVer der Capability
    - parameters: Freie Metadaten (z. B. Schema‑Referenzen)
    - health/readiness: Optional als Metadaten abgelegt (parameters)
    """

    id: str = Field(..., min_length=1, description="Eindeutige Capability-ID")
    name: str = Field(..., min_length=1, description="Name der Capability")
    description: str | None = Field(default=None, description="Beschreibung")
    status: CapabilityStatus = Field(default=CapabilityStatus.AVAILABLE)
    version: str = Field(default="1.0.0", description="SemVer der Capability")
    parameters: dict[str, Any] = Field(default_factory=dict)
    health: str | None = Field(default=None, description="Health-Anzeige der Capability")
    readiness: str | None = Field(default=None, description="Readiness-Anzeige der Capability")
    endpoints: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Referenzen auf interoperable Endpunkte: {"
            "rpc: {service, methods: []}, grpc: {service, methods: []}, "
            "stream: {websocket: '/ws/...', sse: '/sse/...'}, bus: {subjects: []}, "
            "mcp: {server_id, tools: [], resources: []}"
            "}"
        ),
    )
    versions: dict[str, Any] | None = Field(
        default=None,
        description="Versionshinweise pro Interface, z. B. {rpc: 'v1', mcp: '2025-06-18'}",
    )


class CapabilityAdvertiseRequest(BaseModel):
    """Advertisement-Payload für mehrere Capabilities eines Agents."""

    capabilities: list[CapabilityAdvertiseItem] = Field(default_factory=list)
    replace: bool = Field(
        default=True,
        description="Ersetzt bestehende Capabilities vollständig (True) oder merged (False)",
    )


class CapabilityAdvertiseResponse(BaseModel):
    """Antwort auf Capability-Advertisement."""

    agent_id: str
    advertised_count: int
    replaced: bool
    updated_at: datetime


class CapabilityStatusUpdateRequest(BaseModel):
    """Status-Update einer einzelnen Capability."""

    status: CapabilityStatus = Field(...)


class CapabilityReadinessUpdateRequest(BaseModel):
    """Readiness-/Health-Update einer Capability (als Metadaten)."""

    health: str | None = Field(default=None)
    readiness: str | None = Field(default=None)
    parameters: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/{agent_id}/capabilities/advertise",
    response_model=CapabilityAdvertiseResponse,
    dependencies=[Depends(require_auth), Depends(lambda req: require_rate_limit(req, "default"))],
)
async def advertise_capabilities(agent_id: str, body: CapabilityAdvertiseRequest) -> CapabilityAdvertiseResponse:
    """Registriert/aktualisiert Capabilities eines Agents in den Agent-Metadaten.

    Hinweise:
    - Erstellt bei Bedarf eine neue AgentMetadata-Instanz (CUSTOM_MCP als Default).
    - Legt Health/Readiness als Parameter-Schlüssel ab (health/readiness), um pro Capability abfragbar zu sein.
    - Synchronisiert die Registry-Feldliste `capabilities` für Discovery.
    """
    # Metadata sicherstellen
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
        if body.replace:
            meta.available_capabilities.clear()

        # Capabilities übernehmen
        for cap in body.capabilities:
            params = dict(cap.parameters or {})
            if cap.health is not None:
                params["health"] = cap.health
            if cap.readiness is not None:
                params["readiness"] = cap.readiness
            if cap.endpoints is not None:
                params["endpoints"] = cap.endpoints
            if cap.versions is not None:
                params["versions"] = cap.versions

            descriptor = MCPCapabilityDescriptor(
                id=cap.id,
                name=cap.name,
                description=cap.description,
                status=cap.status,
                parameters=params,
            )
            meta.add_capability(descriptor)

        # Synchronisierung: Registry-Agent.capabilities (vereinfachte Sicht)
        if dynamic_registry is not None:
            try:
                reg_agent = await dynamic_registry.get_agent_by_id(agent_id)
                if reg_agent is not None:
                    reg_agent.capabilities = list(meta.available_capabilities.keys())
                    # Aggregierte Endpoints/Topics pro Capability ablegen (Best-Effort)
                    endpoints_map: dict[str, Any] = {}
                    for cid, cdesc in meta.available_capabilities.items():
                        ep = (cdesc.parameters or {}).get("endpoints") if hasattr(cdesc, "parameters") else None  # type: ignore[attr-defined]
                        if ep:
                            endpoints_map[cid] = ep
                    reg_agent.endpoints = endpoints_map
                    # Note: Agent capabilities are managed by the registry itself
            except Exception:
                # Fallback stumm – Registry ist Best-Effort
                pass

        return CapabilityAdvertiseResponse(
            agent_id=agent_id,
            advertised_count=len(body.capabilities),
            replaced=body.replace,
            updated_at=meta.updated_at,
        )
    except Exception as e:
        logger.exception(f"Capability-Advertisement fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Capability-Advertisement fehlgeschlagen")


@router.get(
    "/{agent_id}/capabilities",
    dependencies=[Depends(require_auth), Depends(lambda req: require_rate_limit(req, "discovery"))],
)
async def list_agent_capabilities(agent_id: str) -> dict[str, Any]:
    """Listet Capabilities eines Agents mit Status/Health/Readiness auf."""
    meta = metadata_service.get_metadata(agent_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata nicht gefunden")

    items: list[dict[str, Any]] = []
    for cap in meta.available_capabilities.values():
        params = cap.parameters or {}
        items.append(
            {
                "id": cap.id,
                "name": cap.name,
                "description": cap.description,
                "status": cap.status,
                "health": params.get("health"),
                "readiness": params.get("readiness"),
                "endpoints": params.get("endpoints"),
                "versions": params.get("versions"),
                "parameters": params,
            }
        )

    by_status: dict[str, int] = {}
    for it in items:
        by_status[str(it["status"])] = by_status.get(str(it["status"]), 0) + 1

    return {
        "agent_id": agent_id,
        "capabilities": items,
        "summary": {"by_status": by_status, "total": len(items)},
        "updated_at": metadata_service.get_metadata(agent_id).updated_at if metadata_service.get_metadata(agent_id) else datetime.utcnow(),
    }


@router.get(
    "/{agent_id}/capabilities/{capability_id}",
    dependencies=[Depends(require_auth), Depends(lambda req: require_rate_limit(req, "discovery"))],
)
async def get_capability(agent_id: str, capability_id: str = Path(..., min_length=1)) -> dict[str, Any]:
    """Liefert Details einer Capability inkl. Health/Readiness."""
    meta = metadata_service.get_metadata(agent_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata nicht gefunden")
    cap = meta.get_capability(capability_id)
    if not cap:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")
    params = cap.parameters or {}
    return {
        "agent_id": agent_id,
        "id": cap.id,
        "name": cap.name,
        "description": cap.description,
        "status": cap.status,
        "health": params.get("health"),
        "readiness": params.get("readiness"),
        "parameters": params,
    }


@router.put(
    "/{agent_id}/capabilities/{capability_id}/status",
    dependencies=[Depends(require_auth), Depends(lambda req: require_rate_limit(req, "default"))],
)
async def update_capability_status(agent_id: str, capability_id: str, body: CapabilityStatusUpdateRequest) -> dict[str, Any]:
    """Setzt den Status einer Capability (available/deprecated/...)."""
    meta = metadata_service.get_metadata(agent_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata nicht gefunden")
    cap = meta.get_capability(capability_id)
    if not cap:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")
    try:
        cap.status = body.status
        meta.available_capabilities[capability_id] = cap
        return {"agent_id": agent_id, "id": capability_id, "status": cap.status}
    except Exception as e:
        logger.exception(f"Capability-Status Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Capability-Status Update fehlgeschlagen")


@router.put(
    "/{agent_id}/capabilities/{capability_id}/readiness",
    dependencies=[Depends(require_auth), Depends(lambda req: require_rate_limit(req, "default"))],
)
async def update_capability_readiness(agent_id: str, capability_id: str, body: CapabilityReadinessUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Health/Readiness einer Capability (als Metadaten)."""
    meta = metadata_service.get_metadata(agent_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata nicht gefunden")
    cap = meta.get_capability(capability_id)
    if not cap:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")
    try:
        params = dict(cap.parameters or {})
        if body.health is not None:
            params["health"] = body.health
        if body.readiness is not None:
            params["readiness"] = body.readiness
        if body.parameters:
            params.update(body.parameters)
        cap.parameters = params
        meta.available_capabilities[capability_id] = cap
        return {
            "agent_id": agent_id,
            "id": capability_id,
            "status": cap.status,
            "health": params.get("health"),
            "readiness": params.get("readiness"),
        }
    except Exception as e:
        logger.exception(f"Capability-Readiness Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Capability-Readiness Update fehlgeschlagen")
