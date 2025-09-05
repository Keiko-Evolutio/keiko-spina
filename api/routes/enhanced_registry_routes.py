# backend/api/routes/enhanced_registry_routes.py
"""API-Endpoints für Enhanced Agent Registry.

Implementiert Registry-Management, Version-Pinning, Multi-Tenancy,
Advanced Discovery und Rollout-Management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field, field_validator

from agents.registry.discovery_engine import DiscoveryStrategy
from agents.registry.enhanced_models import (
    RolloutConfiguration,
    RolloutStrategy,
    TenantAccessLevel,
    TenantMetadata,
)
from agents.registry.enhanced_registry import enhanced_registry
from kei_logging import (
    AuthorizationError,
    BusinessLogicError,
    LogLinkedError,
    ValidationError,
    get_logger,
    with_log_links,
)

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/registry", tags=["Enhanced Registry Management"])


# Request/Response Models

class AgentRegistrationRequest(BaseModel):
    """Request-Model für Agent-Registrierung."""
    agent_id: str = Field(..., min_length=1, max_length=255, description="Agent-ID")
    name: str = Field(..., min_length=1, max_length=255, description="Agent-Name")
    version: str = Field(..., description="Agent-Version (SemVer)")
    tenant_id: str = Field(..., min_length=1, max_length=64, description="Tenant-ID")
    description: str | None = Field(None, max_length=1000, description="Agent-Beschreibung")
    capabilities: list[str] = Field(default_factory=list, description="Agent-Capabilities")
    tags: list[str] = Field(default_factory=list, description="Agent-Tags")
    owner: str | None = Field(None, max_length=128, description="Agent-Owner")
    access_level: TenantAccessLevel = Field(default=TenantAccessLevel.PRIVATE, description="Zugriffslevel")
    framework_version_constraint: str | None = Field(None, description="Framework-Versions-Constraint")

    @field_validator("version")
    def validate_version(cls, v):
        """Validiert Versions-Format."""
        from agents.registry.enhanced_models import SemanticVersion
        try:
            SemanticVersion.parse(v)
            return v
        except ValueError as e:
            raise ValueError(f"Ungültiges Versions-Format: {e}")

    @field_validator("capabilities", "tags")
    def validate_lists(cls, v):
        """Validiert Listen."""
        if len(v) > 50:
            raise ValueError("Maximal 50 Einträge erlaubt")
        return v


class TenantRegistrationRequest(BaseModel):
    """Request-Model für Tenant-Registrierung."""
    tenant_id: str = Field(..., min_length=1, max_length=64, description="Tenant-ID")
    tenant_name: str = Field(..., min_length=1, max_length=255, description="Tenant-Name")
    organization: str | None = Field(None, max_length=255, description="Organisation")
    contact_email: str | None = Field(None, description="Kontakt-E-Mail")
    access_level: TenantAccessLevel = Field(default=TenantAccessLevel.PRIVATE, description="Zugriffslevel")
    resource_quotas: dict[str, int] = Field(default_factory=dict, description="Ressourcen-Quotas")
    tags: list[str] = Field(default_factory=list, description="Tenant-Tags")

    @field_validator("contact_email")
    @classmethod
    def validate_email(cls, v):
        """Validiert E-Mail-Format."""
        if v is not None:
            import re
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, v):
                raise ValueError("Ungültiges E-Mail-Format")
        return v


class DiscoveryRequest(BaseModel):
    """Request-Model für Agent-Discovery."""
    agent_id: str | None = Field(None, description="Spezifische Agent-ID")
    version_constraint: str | None = Field(None, description="Versions-Constraint")
    capabilities: list[str] = Field(default_factory=list, description="Erforderliche Capabilities")
    tenant_id: str | None = Field(None, description="Tenant-ID für Zugriffsprüfung")
    strategy: DiscoveryStrategy = Field(default=DiscoveryStrategy.HYBRID, description="Discovery-Strategie")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximale Ergebnisse")
    min_health_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimaler Health-Score")
    max_load_factor: float = Field(default=0.9, ge=0.0, le=1.0, description="Maximaler Load-Factor")
    preferred_regions: list[str] = Field(default_factory=list, description="Bevorzugte Regionen")


class RolloutRequest(BaseModel):
    """Request-Model für Rollout-Start."""
    agent_id: str = Field(..., description="Agent-ID")
    source_version: str = Field(..., description="Quell-Version")
    target_version: str = Field(..., description="Ziel-Version")
    tenant_id: str = Field(..., description="Tenant-ID")
    strategy: RolloutStrategy = Field(default=RolloutStrategy.CANARY, description="Rollout-Strategie")
    canary_percentage: float = Field(default=10.0, ge=0.0, le=100.0, description="Canary-Prozentsatz")
    canary_duration_minutes: int = Field(default=60, ge=1, description="Canary-Dauer in Minuten")
    auto_rollback_on_error: bool = Field(default=True, description="Automatischer Rollback bei Fehlern")
    success_threshold_percentage: float = Field(default=95.0, ge=0.0, le=100.0, description="Erfolgs-Schwellwert")


class AgentResponse(BaseModel):
    """Response-Model für Agent-Informationen."""
    agent_id: str = Field(..., description="Agent-ID")
    name: str = Field(..., description="Agent-Name")
    version: str = Field(..., description="Agent-Version")
    tenant_id: str = Field(..., description="Tenant-ID")
    description: str | None = Field(None, description="Agent-Beschreibung")
    capabilities: list[str] = Field(default_factory=list, description="Agent-Capabilities")
    tags: list[str] = Field(default_factory=list, description="Agent-Tags")
    status: str = Field(..., description="Agent-Status")
    access_level: str = Field(..., description="Zugriffslevel")
    created_at: datetime = Field(..., description="Erstellungszeitpunkt")
    updated_at: datetime = Field(..., description="Letztes Update")
    owner: str | None = Field(None, description="Agent-Owner")


class TenantResponse(BaseModel):
    """Response-Model für Tenant-Informationen."""
    tenant_id: str = Field(..., description="Tenant-ID")
    tenant_name: str = Field(..., description="Tenant-Name")
    organization: str | None = Field(None, description="Organisation")
    access_level: str = Field(..., description="Zugriffslevel")
    total_agents: int = Field(..., description="Anzahl Agents")
    resource_quotas: dict[str, int] = Field(default_factory=dict, description="Ressourcen-Quotas")
    created_at: datetime = Field(..., description="Erstellungszeitpunkt")
    tags: list[str] = Field(default_factory=list, description="Tenant-Tags")


class RolloutResponse(BaseModel):
    """Response-Model für Rollout-Informationen."""
    rollout_id: str = Field(..., description="Rollout-ID")
    agent_id: str = Field(..., description="Agent-ID")
    source_version: str = Field(..., description="Quell-Version")
    target_version: str = Field(..., description="Ziel-Version")
    tenant_id: str = Field(..., description="Tenant-ID")
    status: str = Field(..., description="Rollout-Status")
    strategy: str = Field(..., description="Rollout-Strategie")
    started_at: datetime | None = Field(None, description="Start-Zeitpunkt")
    completed_at: datetime | None = Field(None, description="Abschluss-Zeitpunkt")
    success_rate: float = Field(default=0.0, description="Erfolgsrate")
    error_rate: float = Field(default=0.0, description="Fehlerrate")


# Dependency-Funktionen

async def validate_tenant_access(
    tenant_id: str,
    user_id: str | None = None,
    required_permission: str = "read"
) -> bool:
    """Validiert Tenant-Zugriff."""
    try:
        # Integration mit Enhanced Security System
        from enhanced_security import security_manager

        has_permission = await security_manager.check_permission(
            user_id=user_id or "system",
            resource=f"tenant:{tenant_id}",
            action=required_permission
        )

        if not has_permission:
            raise AuthorizationError(
                message=f"Keine Berechtigung für {required_permission} auf Tenant {tenant_id}",
                resource=f"tenant:{tenant_id}",
                action=required_permission,
                user_id=user_id
            )

        return True

    except ImportError:
        # Enhanced Security nicht verfügbar - erlaube Zugriff
        logger.warning("Enhanced Security System nicht verfügbar - Zugriff erlaubt")
        return True


# Registry Management Endpoints

@router.post("/agents", response_model=AgentResponse)
@with_log_links(component="registry_api", operation="register_agent")
async def register_agent(request: AgentRegistrationRequest) -> AgentResponse:
    """Registriert neuen Agent in Enhanced Registry.

    Args:
        request: Agent-Registrierungs-Request

    Returns:
        Agent-Informationen

    Raises:
        ValidationError: Bei ungültigen Eingabedaten
        BusinessLogicError: Bei Registrierungs-Fehlern
    """
    try:
        # Validiere Tenant-Zugriff
        await validate_tenant_access(request.tenant_id, required_permission="write")

        # Erstelle Enhanced Metadata
        from agents.registry.enhanced_models import (
            AgentStatus,
            AgentVersionMetadata,
            SemanticVersion,
        )

        metadata = AgentVersionMetadata(
            agent_id=request.agent_id,
            version=SemanticVersion.parse(request.version),
            tenant_id=request.tenant_id,
            name=request.name,
            description=request.description or "",
            capabilities=request.capabilities,
            tags=request.tags,
            owner=request.owner,
            status=AgentStatus.AVAILABLE,
            access_level=request.access_level
        )

        # Framework-Constraint setzen
        if request.framework_version_constraint:
            from agents.registry.enhanced_models import VersionConstraint
            metadata.framework_version_constraint = VersionConstraint.parse(
                request.framework_version_constraint
            )

        # Registriere Agent
        agent_id = await enhanced_registry.register_agent(metadata)

        # Erstelle Response
        return AgentResponse(
            agent_id=agent_id,
            name=metadata.name,
            version=str(metadata.version),
            tenant_id=metadata.tenant_id,
            description=metadata.description,
            capabilities=metadata.capabilities,
            tags=metadata.tags,
            status=metadata.status.value,
            access_level=metadata.access_level.value,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            owner=metadata.owner
        )

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Agent-Registrierung fehlgeschlagen: {e!s}",
            agent_id=request.agent_id,
            tenant_id=request.tenant_id,
            component="registry_api",
            operation="register_agent",
            cause=e
        )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
@with_log_links(component="registry_api", operation="get_agent")
async def get_agent(
    agent_id: str = Path(..., description="Agent-ID"),
    version: str = Query(default="latest", description="Versions-Constraint"),
    tenant_id: str | None = Query(None, description="Tenant-ID für Zugriffsprüfung")
) -> AgentResponse:
    """Holt Agent-Informationen basierend auf Versions-Constraint.

    Args:
        agent_id: Agent-ID
        version: Versions-Constraint
        tenant_id: Tenant-ID für Zugriffsprüfung

    Returns:
        Agent-Informationen

    Raises:
        ValidationError: Wenn Agent nicht gefunden
    """
    try:
        # Validiere Tenant-Zugriff falls angegeben
        if tenant_id:
            await validate_tenant_access(tenant_id, required_permission="read")

        # Hole Agent
        metadata = await enhanced_registry.get_agent(agent_id, version, tenant_id)

        if not metadata:
            raise ValidationError(
                message=f"Agent '{agent_id}' mit Version-Constraint '{version}' nicht gefunden",
                field="agent_id",
                value=agent_id,
                version_constraint=version,
                tenant_id=tenant_id
            )

        # Erstelle Response
        return AgentResponse(
            agent_id=metadata.agent_id,
            name=metadata.name,
            version=str(metadata.version),
            tenant_id=metadata.tenant_id,
            description=metadata.description,
            capabilities=metadata.capabilities,
            tags=metadata.tags,
            status=metadata.status.value,
            access_level=metadata.access_level.value,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            owner=metadata.owner
        )

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Agent-Abfrage fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            version_constraint=version,
            tenant_id=tenant_id,
            component="registry_api",
            operation="get_agent",
            cause=e
        )


@router.get("/agents", response_model=list[AgentResponse])
@with_log_links(component="registry_api", operation="list_agents")
async def list_agents(
    tenant_id: str | None = Query(None, description="Filter nach Tenant-ID"),
    include_versions: bool = Query(default=False, description="Alle Versionen einschließen"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximale Anzahl Ergebnisse")
) -> list[AgentResponse]:
    """Listet Agents auf.

    Args:
        tenant_id: Filter nach Tenant-ID
        include_versions: Alle Versionen einschließen
        limit: Maximale Anzahl Ergebnisse

    Returns:
        Liste von Agent-Informationen
    """
    try:
        # Validiere Tenant-Zugriff falls angegeben
        if tenant_id:
            await validate_tenant_access(tenant_id, required_permission="read")

        # Hole Agents
        agents_data = await enhanced_registry.list_agents(tenant_id, include_versions)

        responses = []
        count = 0

        for agent_id, agent_info in agents_data.items():
            if count >= limit:
                break

            if include_versions and "versions" in agent_info:
                # Mehrere Versionen
                for version in agent_info["versions"]:
                    if count >= limit:
                        break

                    metadata = await enhanced_registry.get_agent(agent_id, version, tenant_id)
                    if metadata:
                        responses.append(AgentResponse(
                            agent_id=metadata.agent_id,
                            name=metadata.name,
                            version=str(metadata.version),
                            tenant_id=metadata.tenant_id,
                            description=metadata.description,
                            capabilities=metadata.capabilities,
                            tags=metadata.tags,
                            status=metadata.status.value,
                            access_level=metadata.access_level.value,
                            created_at=metadata.created_at,
                            updated_at=metadata.updated_at,
                            owner=metadata.owner
                        ))
                        count += 1
            else:
                # Nur latest Version oder Legacy-Format
                if isinstance(agent_info, dict) and "agent_id" in agent_info:
                    # Enhanced Format
                    responses.append(AgentResponse(**agent_info))
                else:
                    # Legacy Format - hole latest
                    metadata = await enhanced_registry.get_agent(agent_id, "latest", tenant_id)
                    if metadata:
                        responses.append(AgentResponse(
                            agent_id=metadata.agent_id,
                            name=metadata.name,
                            version=str(metadata.version),
                            tenant_id=metadata.tenant_id,
                            description=metadata.description,
                            capabilities=metadata.capabilities,
                            tags=metadata.tags,
                            status=metadata.status.value,
                            access_level=metadata.access_level.value,
                            created_at=metadata.created_at,
                            updated_at=metadata.updated_at,
                            owner=metadata.owner
                        ))
                count += 1

        return responses

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Agent-Auflistung fehlgeschlagen: {e!s}",
            tenant_id=tenant_id,
            component="registry_api",
            operation="list_agents",
            cause=e
        )


# Discovery Endpoints

@router.post("/discovery", response_model=list[AgentResponse])
@with_log_links(component="registry_api", operation="discover_agents")
async def discover_agents(request: DiscoveryRequest) -> list[AgentResponse]:
    """Führt erweiterte Agent-Discovery durch.

    Args:
        request: Discovery-Request

    Returns:
        Liste von gefundenen Agents
    """
    try:
        # Validiere Tenant-Zugriff falls angegeben
        if request.tenant_id:
            await validate_tenant_access(request.tenant_id, required_permission="read")

        # Führe Discovery durch
        discovered_agents = await enhanced_registry.discover_agents(
            capabilities=request.capabilities,
            tenant_id=request.tenant_id,
            strategy=request.strategy,
            max_results=request.max_results,
            min_health_score=request.min_health_score,
            max_load_factor=request.max_load_factor,
            preferred_regions=request.preferred_regions
        )

        # Konvertiere zu Response-Format
        responses = []
        for metadata in discovered_agents:
            responses.append(AgentResponse(
                agent_id=metadata.agent_id,
                name=metadata.name,
                version=str(metadata.version),
                tenant_id=metadata.tenant_id,
                description=metadata.description,
                capabilities=metadata.capabilities,
                tags=metadata.tags,
                status=metadata.status.value,
                access_level=metadata.access_level.value,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                owner=metadata.owner
            ))

        logger.info(
            f"Agent-Discovery abgeschlossen: {len(responses)} Ergebnisse",
            extra={
                "strategy": request.strategy.value,
                "capabilities": request.capabilities,
                "tenant_id": request.tenant_id,
                "results_count": len(responses)
            }
        )

        return responses

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Agent-Discovery fehlgeschlagen: {e!s}",
            capabilities=request.capabilities,
            tenant_id=request.tenant_id,
            strategy=request.strategy.value,
            component="registry_api",
            operation="discover_agents",
            cause=e
        )


# Tenant Management Endpoints

@router.post("/tenants", response_model=TenantResponse)
@with_log_links(component="registry_api", operation="register_tenant")
async def register_tenant(request: TenantRegistrationRequest) -> TenantResponse:
    """Registriert neuen Tenant.

    Args:
        request: Tenant-Registrierungs-Request

    Returns:
        Tenant-Informationen
    """
    try:
        # Erstelle Tenant-Metadata
        tenant_metadata = TenantMetadata(
            tenant_id=request.tenant_id,
            tenant_name=request.tenant_name,
            organization=request.organization,
            contact_email=request.contact_email,
            access_level=request.access_level,
            resource_quotas=request.resource_quotas,
            tags=request.tags
        )

        # Registriere Tenant
        await enhanced_registry.register_tenant(tenant_metadata)

        # Erstelle Response
        return TenantResponse(
            tenant_id=tenant_metadata.tenant_id,
            tenant_name=tenant_metadata.tenant_name,
            organization=tenant_metadata.organization,
            access_level=tenant_metadata.access_level.value,
            total_agents=0,  # Neu registriert
            resource_quotas=tenant_metadata.resource_quotas,
            created_at=tenant_metadata.created_at,
            tags=tenant_metadata.tags
        )

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Tenant-Registrierung fehlgeschlagen: {e!s}",
            tenant_id=request.tenant_id,
            component="registry_api",
            operation="register_tenant",
            cause=e
        )


@router.get("/tenants/{tenant_id}", response_model=TenantResponse)
@with_log_links(component="registry_api", operation="get_tenant")
async def get_tenant(
    tenant_id: str = Path(..., description="Tenant-ID")
) -> TenantResponse:
    """Holt Tenant-Informationen.

    Args:
        tenant_id: Tenant-ID

    Returns:
        Tenant-Informationen
    """
    try:
        # Validiere Tenant-Zugriff
        await validate_tenant_access(tenant_id, required_permission="read")

        # Hole Tenant-Statistiken
        from agents.registry.tenant_manager import tenant_manager

        tenant_stats = tenant_manager.get_tenant_statistics(tenant_id)

        # Erstelle Response
        return TenantResponse(
            tenant_id=tenant_stats["tenant_id"],
            tenant_name=tenant_stats["tenant_name"],
            organization=tenant_stats["organization"],
            access_level=tenant_stats["access_level"],
            total_agents=tenant_stats["total_agents"],
            resource_quotas=tenant_stats["quota_usage"],
            created_at=datetime.fromisoformat(tenant_stats["created_at"]),
            tags=tenant_stats["tags"]
        )

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Tenant-Abfrage fehlgeschlagen: {e!s}",
            tenant_id=tenant_id,
            component="registry_api",
            operation="get_tenant",
            cause=e
        )


# Rollout Management Endpoints

@router.post("/rollouts", response_model=RolloutResponse)
@with_log_links(component="registry_api", operation="start_rollout")
async def start_rollout(request: RolloutRequest) -> RolloutResponse:
    """Startet Agent-Rollout.

    Args:
        request: Rollout-Request

    Returns:
        Rollout-Informationen
    """
    try:
        # Validiere Tenant-Zugriff
        await validate_tenant_access(request.tenant_id, required_permission="admin")

        # Erstelle Rollout-Konfiguration
        rollout_config = RolloutConfiguration(
            strategy=request.strategy,
            canary_percentage=request.canary_percentage,
            canary_duration_minutes=request.canary_duration_minutes,
            auto_rollback_on_error=request.auto_rollback_on_error,
            success_threshold_percentage=request.success_threshold_percentage
        )

        # Starte Rollout
        rollout_id = await enhanced_registry.start_agent_rollout(
            agent_id=request.agent_id,
            source_version=request.source_version,
            target_version=request.target_version,
            tenant_id=request.tenant_id,
            rollout_config=rollout_config
        )

        # Hole Rollout-Status
        from agents.registry.rollout_manager import rollout_manager

        rollout = rollout_manager.get_rollout_status(rollout_id)

        if not rollout:
            raise BusinessLogicError(
                message=f"Rollout {rollout_id} nicht gefunden nach Start",
                rollout_id=rollout_id
            )

        # Erstelle Response
        return RolloutResponse(
            rollout_id=rollout.rollout_id,
            agent_id=rollout.agent_id,
            source_version=str(rollout.source_version),
            target_version=str(rollout.target_version),
            tenant_id=rollout.tenant_id,
            status=rollout.status.value,
            strategy=rollout.config.strategy.value,
            started_at=rollout.started_at,
            completed_at=rollout.completed_at,
            success_rate=rollout.metrics.success_rate,
            error_rate=rollout.metrics.error_rate
        )

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Rollout-Start fehlgeschlagen: {e!s}",
            agent_id=request.agent_id,
            tenant_id=request.tenant_id,
            component="registry_api",
            operation="start_rollout",
            cause=e
        )


@router.get("/rollouts/{rollout_id}", response_model=RolloutResponse)
@with_log_links(component="registry_api", operation="get_rollout")
async def get_rollout(
    rollout_id: str = Path(..., description="Rollout-ID")
) -> RolloutResponse:
    """Holt Rollout-Status.

    Args:
        rollout_id: Rollout-ID

    Returns:
        Rollout-Informationen
    """
    try:
        # Hole Rollout-Status
        from agents.registry.rollout_manager import rollout_manager

        rollout = rollout_manager.get_rollout_status(rollout_id)

        if not rollout:
            raise ValidationError(
                message=f"Rollout '{rollout_id}' nicht gefunden",
                field="rollout_id",
                value=rollout_id
            )

        # Validiere Tenant-Zugriff
        await validate_tenant_access(rollout.tenant_id, required_permission="read")

        # Erstelle Response
        return RolloutResponse(
            rollout_id=rollout.rollout_id,
            agent_id=rollout.agent_id,
            source_version=str(rollout.source_version),
            target_version=str(rollout.target_version),
            tenant_id=rollout.tenant_id,
            status=rollout.status.value,
            strategy=rollout.config.strategy.value,
            started_at=rollout.started_at,
            completed_at=rollout.completed_at,
            success_rate=rollout.metrics.success_rate,
            error_rate=rollout.metrics.error_rate
        )

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Rollout-Abfrage fehlgeschlagen: {e!s}",
            rollout_id=rollout_id,
            component="registry_api",
            operation="get_rollout",
            cause=e
        )


# Statistics and Health Endpoints

@router.get("/statistics")
@with_log_links(component="registry_api", operation="get_statistics")
async def get_registry_statistics() -> dict[str, Any]:
    """Holt umfassende Registry-Statistiken.

    Returns:
        Registry-Statistiken
    """
    try:
        stats = await enhanced_registry.get_registry_statistics()

        logger.info(
            "Registry-Statistiken abgerufen",
            extra={
                "total_agents": stats.get("version_management", {}).get("total_agents", 0),
                "total_tenants": len(stats.get("tenant_management", {})),
                "active_rollouts": stats.get("rollout_management", {}).get("active_rollouts", 0)
            }
        )

        return stats

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Statistiken-Abfrage fehlgeschlagen: {e!s}",
            component="registry_api",
            operation="get_statistics",
            cause=e
        )


@router.post("/cleanup")
@with_log_links(component="registry_api", operation="cleanup_registry")
async def cleanup_registry(
    max_age_days: int = Query(default=90, ge=1, le=365, description="Maximales Alter in Tagen")
) -> dict[str, int]:
    """Führt Registry-Cleanup durch.

    Args:
        max_age_days: Maximales Alter für Cleanup

    Returns:
        Cleanup-Statistiken
    """
    try:
        cleanup_stats = await enhanced_registry.cleanup_registry(max_age_days)

        logger.info(
            f"Registry-Cleanup durchgeführt: {cleanup_stats}",
            extra={
                "max_age_days": max_age_days,
                **cleanup_stats
            }
        )

        return cleanup_stats

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Registry-Cleanup fehlgeschlagen: {e!s}",
            max_age_days=max_age_days,
            component="registry_api",
            operation="cleanup_registry",
            cause=e
        )
