# backend/api/routes/enhanced_capabilities_routes.py
"""Enhanced Capabilities API-Routen für KEI-Agent System.

Implementiert vollständige Health/Readiness-Checks, Versionierung
und kategorie-spezifische Capability-Verwaltung.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from agents.capabilities import get_capability_manager
from agents.capabilities.enhanced_capabilities import (
    EnhancedCapability,
    VersionCompatibility,
    VersionInfo,
)
from kei_logging import get_logger
from observability import trace_function

# Globale Capability Manager Instanz
capability_manager = get_capability_manager()

if TYPE_CHECKING:
    from agents.metadata.agent_metadata import (
        CapabilityCategory,
        CapabilityStatus,
    )

logger = get_logger(__name__)

# Response Models
class CapabilityHealthResponse(BaseModel):
    """Response für Capability Health Check."""
    capability_id: str
    status: str
    response_time_ms: float
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)

class CapabilityMetricsResponse(BaseModel):
    """Response für Capability Metriken."""
    capability_id: str
    invocation_count: int
    success_rate: float
    avg_response_time_ms: float
    error_count: int
    last_invocation: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

class CapabilityReadinessResponse(BaseModel):
    """Response für Capability Readiness Check."""
    capability_id: str
    status: str
    dependencies_ready: bool
    response_time_ms: float
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)

class EnhancedCapabilityResponse(BaseModel):
    """Response für einzelne Enhanced Capability."""
    id: str
    name: str
    description: str
    category: str
    version: str
    status: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    endpoints: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

class EnhancedCapabilitiesResponse(BaseModel):
    """Response für Liste von Enhanced Capabilities."""
    capabilities: list[EnhancedCapabilityResponse]
    total_count: int
    filtered_count: int
    categories: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)

router = APIRouter(prefix="/api/v1/enhanced-capabilities", tags=["enhanced-capabilities"])


# ============================================================================
# REQUEST MODELS
# ============================================================================

class RegisterCapabilityRequest(BaseModel):
    """Request-Modell für Capability-Registrierung."""
    id: str = Field(..., description="Eindeutige Capability-ID")
    name: str = Field(..., description="Name der Capability")
    description: str = Field(..., description="Beschreibung")
    category: CapabilityCategory = Field(..., description="Kategorie")
    version: str = Field(..., description="Version (SemVer)")
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    endpoints: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateCapabilityRequest(BaseModel):
    """Request-Modell für Capability-Update."""
    name: str | None = Field(None, description="Neuer Name")
    description: str | None = Field(None, description="Neue Beschreibung")
    version: str | None = Field(None, description="Neue Version")
    status: CapabilityStatus | None = Field(None, description="Neuer Status")
    parameters: dict[str, Any] | None = Field(None, description="Neue Parameter")
    endpoints: dict[str, str] | None = Field(None, description="Neue Endpunkte")
    tags: list[str] | None = Field(None, description="Neue Tags")
    metadata: dict[str, Any] | None = Field(None, description="Neue Metadaten")


# ============================================================================
# CAPABILITY MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/register", response_model=dict[str, Any])
@trace_function("api.enhanced_capabilities.register")
async def register_capability(request: RegisterCapabilityRequest) -> dict[str, Any]:
    """Registriert neue Enhanced Capability."""
    try:
        # Erstelle VersionInfo
        version_info = VersionInfo(
            version=request.version,
            introduced_at=datetime.now(UTC),
            compatibility=VersionCompatibility.COMPATIBLE
        )

        # Erstelle Enhanced Capability
        capability = EnhancedCapability(
            id=request.id,
            name=request.name,
            description=request.description,
            category=request.category,
            version_info=version_info,
            parameters=request.parameters,
            dependencies=request.dependencies,
            endpoints=request.endpoints,
            tags=set(request.tags),
            metadata=request.metadata
        )

        # Registriere in Manager
        get_capability_manager().register_capability(capability)

        logger.info(f"Enhanced Capability registriert: {request.id}")

        return {
            "status": "success",
            "capability_id": request.id,
            "message": "Capability erfolgreich registriert",
            "timestamp": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.exception(f"Capability-Registrierung fehlgeschlagen: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=EnhancedCapabilitiesResponse)
@trace_function("api.enhanced_capabilities.list")
async def list_enhanced_capabilities(
    category: CapabilityCategory | None = Query(None, description="Filter nach Kategorie"),
    status: CapabilityStatus | None = Query(None, description="Filter nach Status"),
    available_only: bool = Query(False, description="Nur verfügbare Capabilities"),
    include_deprecated: bool = Query(False, description="Veraltete Capabilities einschließen")
) -> EnhancedCapabilitiesResponse:
    """Listet Enhanced Capabilities mit Filtern."""
    try:
        # Filter anwenden
        capabilities = get_capability_manager().list_capabilities(
            category=category,
            status=status,
            available_only=available_only
        )

        # Deprecated filtern falls gewünscht
        if not include_deprecated:
            capabilities = [c for c in capabilities if not c.is_deprecated()]

        # Response-Objekte erstellen
        capability_responses = []
        for cap in capabilities:
            response = EnhancedCapabilityResponse(
                id=cap.id,
                name=cap.name,
                description=cap.description,
                category=cap.category,
                version=cap.version_info.version,
                supported_versions=cap.supported_versions,
                status=cap.status.value,
                health_status=cap.health_status.value,
                readiness_status=cap.readiness_status.value,
                available=cap.is_available(),
                endpoints=cap.endpoints,
                parameters=cap.parameters,
                dependencies=cap.dependencies,
                tags=list(cap.tags),
                created_at=cap.created_at.isoformat(),
                updated_at=cap.updated_at.isoformat(),
                last_health_check=cap.last_health_check.isoformat() if cap.last_health_check else None,
                last_readiness_check=cap.last_readiness_check.isoformat() if cap.last_readiness_check else None
            )
            capability_responses.append(response)

        available_count = len([c for c in capabilities if c.is_available()])

        return EnhancedCapabilitiesResponse(
            api_version="1.0.0",
            capabilities=capability_responses,
            total_count=len(capabilities),
            available_count=available_count,
            timestamp=datetime.now(UTC).isoformat()
        )

    except Exception as e:
        logger.exception(f"Capability-Listing fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Interner Server-Fehler")


@router.get("/{capability_id}", response_model=EnhancedCapabilityResponse)
@trace_function("api.enhanced_capabilities.get")
async def get_capability(
    capability_id: str = Path(..., description="Capability-ID")
) -> EnhancedCapabilityResponse:
    """Gibt Details einer Enhanced Capability zurück."""
    capability = get_capability_manager().get_capability(capability_id)

    if not capability:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    return EnhancedCapabilityResponse(
        id=capability.id,
        name=capability.name,
        description=capability.description,
        category=capability.category,
        version=capability.version_info.version,
        supported_versions=capability.supported_versions,
        status=capability.status.value,
        health_status=capability.health_status.value,
        readiness_status=capability.readiness_status.value,
        available=capability.is_available(),
        endpoints=capability.endpoints,
        parameters=capability.parameters,
        dependencies=capability.dependencies,
        tags=list(capability.tags),
        created_at=capability.created_at.isoformat(),
        updated_at=capability.updated_at.isoformat(),
        last_health_check=capability.last_health_check.isoformat() if capability.last_health_check else None,
        last_readiness_check=capability.last_readiness_check.isoformat() if capability.last_readiness_check else None
    )


@router.put("/{capability_id}", response_model=dict[str, Any])
@trace_function("api.enhanced_capabilities.update")
async def update_capability(
    capability_id: str = Path(..., description="Capability-ID"),
    request: UpdateCapabilityRequest = ...
) -> dict[str, Any]:
    """Aktualisiert Enhanced Capability."""
    capability = get_capability_manager().get_capability(capability_id)

    if not capability:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    try:
        # Update-Felder anwenden
        updated = False

        if request.name is not None:
            capability.name = request.name
            updated = True

        if request.description is not None:
            capability.description = request.description
            updated = True

        if request.version is not None:
            # Neue Version erstellen
            new_version_info = VersionInfo(
                version=request.version,
                introduced_at=datetime.now(UTC),
                compatibility=VersionCompatibility.COMPATIBLE
            )
            capability.version_info = new_version_info
            updated = True

        if request.status is not None:
            capability.status = request.status
            updated = True

        if request.parameters is not None:
            capability.parameters = request.parameters
            updated = True

        if request.endpoints is not None:
            capability.endpoints = request.endpoints
            updated = True

        if request.tags is not None:
            capability.tags = set(request.tags)
            updated = True

        if request.metadata is not None:
            capability.metadata = request.metadata
            updated = True

        if updated:
            capability.updated_at = datetime.now(UTC)

        logger.info(f"Capability aktualisiert: {capability_id}")

        return {
            "status": "success",
            "capability_id": capability_id,
            "message": "Capability erfolgreich aktualisiert",
            "timestamp": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.exception(f"Capability-Update fehlgeschlagen: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{capability_id}", response_model=dict[str, Any])
@trace_function("api.enhanced_capabilities.delete")
async def unregister_capability(
    capability_id: str = Path(..., description="Capability-ID")
) -> dict[str, Any]:
    """Entfernt Enhanced Capability."""
    success = get_capability_manager().unregister_capability(capability_id)

    if not success:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    logger.info(f"Capability entfernt: {capability_id}")

    return {
        "status": "success",
        "capability_id": capability_id,
        "message": "Capability erfolgreich entfernt",
        "timestamp": datetime.now(UTC).isoformat()
    }


# ============================================================================
# HEALTH/READINESS CHECK ENDPOINTS
# ============================================================================

@router.get("/{capability_id}/health", response_model=CapabilityHealthResponse)
@trace_function("api.enhanced_capabilities.health_check")
async def check_capability_health(
    capability_id: str = Path(..., description="Capability-ID")
) -> CapabilityHealthResponse:
    """Führt Health-Check für Capability durch."""
    if capability_id not in get_capability_manager().capabilities:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    try:
        result = await get_capability_manager().check_capability_health(capability_id)

        if not result:
            raise HTTPException(status_code=500, detail="Health-Check fehlgeschlagen")

        return CapabilityHealthResponse(
            capability_id=capability_id,
            health_status=result.status.value,
            message=result.message,
            timestamp=result.timestamp.isoformat(),
            response_time_ms=result.response_time_ms,
            details=result.details
        )

    except Exception as e:
        logger.exception(f"Health-Check für {capability_id} fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Health-Check fehlgeschlagen")


@router.get("/{capability_id}/readiness", response_model=CapabilityReadinessResponse)
@trace_function("api.enhanced_capabilities.readiness_check")
async def check_capability_readiness(
    capability_id: str = Path(..., description="Capability-ID")
) -> CapabilityReadinessResponse:
    """Führt Readiness-Check für Capability durch."""
    if capability_id not in get_capability_manager().capabilities:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    try:
        result = await get_capability_manager().check_capability_readiness(capability_id)

        if not result:
            raise HTTPException(status_code=500, detail="Readiness-Check fehlgeschlagen")

        return CapabilityReadinessResponse(
            capability_id=capability_id,
            readiness_status=result.status.value,
            message=result.message,
            timestamp=result.timestamp.isoformat(),
            dependencies_ready=result.dependencies_ready,
            details=result.details
        )

    except Exception as e:
        logger.exception(f"Readiness-Check für {capability_id} fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Readiness-Check fehlgeschlagen")


@router.get("/{capability_id}/metrics", response_model=CapabilityMetricsResponse)
@trace_function("api.enhanced_capabilities.metrics")
async def get_capability_metrics(
    capability_id: str = Path(..., description="Capability-ID")
) -> CapabilityMetricsResponse:
    """Gibt Metriken für Capability zurück."""
    if capability_id not in get_capability_manager().capabilities:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    metrics = get_capability_manager().get_capability_metrics(capability_id)

    if not metrics:
        raise HTTPException(status_code=404, detail="Metriken nicht gefunden")

    return CapabilityMetricsResponse(
        capability_id=capability_id,
        total_invocations=metrics.total_invocations,
        successful_invocations=metrics.successful_invocations,
        failed_invocations=metrics.failed_invocations,
        success_rate=metrics.success_rate,
        error_rate=metrics.error_rate,
        average_response_time_ms=metrics.average_response_time_ms,
        last_invocation_at=metrics.last_invocation_at.isoformat() if metrics.last_invocation_at else None
    )


@router.post("/{capability_id}/metrics/record", response_model=dict[str, Any])
@trace_function("api.enhanced_capabilities.record_metrics")
async def record_capability_invocation(
    capability_id: str = Path(..., description="Capability-ID"),
    success: bool = Query(..., description="Ob Aufruf erfolgreich war"),
    response_time_ms: float = Query(..., description="Antwortzeit in Millisekunden")
) -> dict[str, Any]:
    """Zeichnet Capability-Aufruf für Metriken auf."""
    if capability_id not in get_capability_manager().capabilities:
        raise HTTPException(status_code=404, detail="Capability nicht gefunden")

    try:
        get_capability_manager().record_capability_invocation(
            capability_id=capability_id,
            success=success,
            response_time_ms=response_time_ms
        )

        return {
            "status": "success",
            "capability_id": capability_id,
            "message": "Metrik erfolgreich aufgezeichnet",
            "timestamp": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.exception(f"Metrik-Aufzeichnung für {capability_id} fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Metrik-Aufzeichnung fehlgeschlagen")


# ============================================================================
# LIFECYCLE MANAGEMENT
# ============================================================================

@router.post("/manager/start", response_model=dict[str, Any])
@trace_function("api.enhanced_capabilities.start_manager")
async def start_capability_manager() -> dict[str, Any]:
    """Startet Capability Manager."""
    try:
        await capability_manager.start()

        return {
            "status": "success",
            "message": "Capability Manager gestartet",
            "timestamp": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.exception(f"Start des Capability Managers fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Start fehlgeschlagen")


@router.post("/manager/stop", response_model=dict[str, Any])
@trace_function("api.enhanced_capabilities.stop_manager")
async def stop_capability_manager() -> dict[str, Any]:
    """Stoppt Capability Manager."""
    try:
        await capability_manager.stop()

        return {
            "status": "success",
            "message": "Capability Manager gestoppt",
            "timestamp": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.exception(f"Stop des Capability Managers fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Stop fehlgeschlagen")
