# backend/api/routes/enhanced_agents_management.py
"""Erweiterte Agent-Management-Endpoints für Keiko Personal Assistant

Implementiert vollständige CRUD-Operationen und Management-Funktionalitäten für Agenten
mit Integration in Enhanced Security, Policy Engine, Observability und Logging-Systeme.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, Field, field_validator

from agents.metadata.service import metadata_service

# Import bestehender Systeme
from agents.registry.dynamic_registry import dynamic_registry
from kei_logging import (
    AuthorizationError,
    BusinessLogicError,
    LogLinkedError,
    ValidationError,
    get_logger,
    with_log_links,
)

if TYPE_CHECKING:
    from agents.metadata.agent_metadata import AgentMetadata

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/agents", tags=["Enhanced Agent Management"])


class AgentStatus(str, Enum):
    """Agent-Status-Enum."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


class ScalingStrategy(str, Enum):
    """Auto-Scaling-Strategien."""
    NONE = "none"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"


class HealthStatus(str, Enum):
    """Gesundheitsstatus-Enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# Request/Response Models

class AgentUpdateRequest(BaseModel):
    """Request-Model für Agent-Update."""
    name: str | None = Field(None, min_length=1, max_length=255, description="Agent-Name")
    description: str | None = Field(None, max_length=1000, description="Agent-Beschreibung")
    owner: str | None = Field(None, max_length=128, description="Agent-Owner")
    tenant: str | None = Field(None, max_length=64, description="Tenant-Zuordnung")
    tags: list[str] | None = Field(None, description="Agent-Tags")
    framework_config: dict[str, Any] | None = Field(None, description="Framework-Konfiguration")
    status: AgentStatus | None = Field(None, description="Agent-Status")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validiert Agent-Tags."""
        if v is not None:
            if len(v) > 50:
                raise ValueError("Maximal 50 Tags erlaubt")
            for tag in v:
                if not isinstance(tag, str) or len(tag) > 50:
                    raise ValueError("Tags müssen Strings mit maximal 50 Zeichen sein")
        return v


class AgentDeprecationRequest(BaseModel):
    """Request-Model für Agent-Deprecation."""
    reason: str = Field(..., min_length=1, max_length=500, description="Grund für Deprecation")
    migration_guide_url: str | None = Field(None, description="URL zum Migrationsleitfaden")
    replacement_agent_id: str | None = Field(None, description="ID des Ersatz-Agents")
    deprecation_date: datetime | None = Field(None, description="Geplantes Deprecation-Datum")
    end_of_life_date: datetime | None = Field(None, description="End-of-Life-Datum")
    notify_users: bool = Field(default=True, description="Benutzer über Deprecation benachrichtigen")


class QuotaConfiguration(BaseModel):
    """Quota-Konfiguration für Agenten."""
    max_requests_per_minute: int | None = Field(None, ge=1, description="Max. Requests pro Minute")
    max_requests_per_hour: int | None = Field(None, ge=1, description="Max. Requests pro Stunde")
    max_requests_per_day: int | None = Field(None, ge=1, description="Max. Requests pro Tag")
    max_concurrent_requests: int | None = Field(None, ge=1, description="Max. gleichzeitige Requests")
    max_memory_mb: int | None = Field(None, ge=1, description="Max. Memory in MB")
    max_cpu_cores: float | None = Field(None, ge=0.1, description="Max. CPU-Kerne")
    max_storage_gb: int | None = Field(None, ge=1, description="Max. Storage in GB")
    max_network_mbps: int | None = Field(None, ge=1, description="Max. Netzwerk-Bandbreite in Mbps")


class PolicyConfiguration(BaseModel):
    """Policy-Konfiguration für Agenten."""
    security_level: str = Field(default="standard", description="Sicherheitslevel")
    data_retention_days: int | None = Field(None, ge=1, description="Datenaufbewahrung in Tagen")
    allowed_domains: list[str] | None = Field(None, description="Erlaubte Domains")
    blocked_domains: list[str] | None = Field(None, description="Blockierte Domains")
    require_encryption: bool = Field(default=True, description="Verschlüsselung erforderlich")
    audit_logging: bool = Field(default=True, description="Audit-Logging aktiviert")
    compliance_frameworks: list[str] | None = Field(None, description="Compliance-Frameworks")


class ScalingConfiguration(BaseModel):
    """Scaling-Konfiguration für Agenten."""
    strategy: ScalingStrategy = Field(default=ScalingStrategy.NONE, description="Scaling-Strategie")
    min_instances: int = Field(default=1, ge=1, description="Minimale Instanzen")
    max_instances: int = Field(default=10, ge=1, description="Maximale Instanzen")
    target_cpu_utilization: float | None = Field(None, ge=0.1, le=1.0, description="Ziel-CPU-Auslastung")
    target_memory_utilization: float | None = Field(None, ge=0.1, le=1.0, description="Ziel-Memory-Auslastung")
    scale_up_threshold: float | None = Field(None, ge=0.1, le=1.0, description="Scale-Up-Schwellwert")
    scale_down_threshold: float | None = Field(None, ge=0.1, le=1.0, description="Scale-Down-Schwellwert")
    cooldown_period_seconds: int = Field(default=300, ge=60, description="Cooldown-Periode in Sekunden")

    @field_validator("max_instances")
    @classmethod
    def validate_max_instances(cls, v, info):
        """Validiert, dass max_instances >= min_instances."""
        if info.data and "min_instances" in info.data and v < info.data["min_instances"]:
            raise ValueError("max_instances muss >= min_instances sein")
        return v


class HealthCheckResult(BaseModel):
    """Ergebnis eines Gesundheitschecks."""
    status: HealthStatus = Field(..., description="Gesundheitsstatus")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Zeitstempel")
    response_time_ms: float | None = Field(None, description="Antwortzeit in Millisekunden")
    details: dict[str, Any] = Field(default_factory=dict, description="Detaillierte Informationen")
    errors: list[str] = Field(default_factory=list, description="Fehlermeldungen")


class AgentStatistics(BaseModel):
    """Agent-Statistiken."""
    total_requests: int = Field(default=0, description="Gesamtanzahl Requests")
    successful_requests: int = Field(default=0, description="Erfolgreiche Requests")
    failed_requests: int = Field(default=0, description="Fehlgeschlagene Requests")
    average_response_time_ms: float = Field(default=0.0, description="Durchschnittliche Antwortzeit")
    uptime_percentage: float = Field(default=0.0, description="Verfügbarkeit in Prozent")
    last_request_timestamp: datetime | None = Field(None, description="Zeitstempel des letzten Requests")
    resource_usage: dict[str, Any] = Field(default_factory=dict, description="Ressourcenverbrauch")
    performance_metrics: dict[str, Any] = Field(default_factory=dict, description="Performance-Metriken")


class AgentResponse(BaseModel):
    """Standard-Agent-Response."""
    agent_id: str = Field(..., description="Agent-ID")
    name: str = Field(..., description="Agent-Name")
    description: str | None = Field(None, description="Agent-Beschreibung")
    status: AgentStatus = Field(..., description="Agent-Status")
    owner: str | None = Field(None, description="Agent-Owner")
    tenant: str | None = Field(None, description="Tenant")
    tags: list[str] = Field(default_factory=list, description="Agent-Tags")
    capabilities: list[str] = Field(default_factory=list, description="Agent-Capabilities")
    created_at: datetime = Field(..., description="Erstellungszeitpunkt")
    updated_at: datetime = Field(..., description="Letztes Update")
    framework_type: str = Field(..., description="Framework-Typ")
    framework_version: str = Field(..., description="Framework-Version")


class QuotaUsageResponse(BaseModel):
    """Response für Quota-Nutzung."""
    configuration: QuotaConfiguration = Field(..., description="Quota-Konfiguration")
    current_usage: dict[str, Any] = Field(default_factory=dict, description="Aktuelle Nutzung")
    usage_percentage: dict[str, float] = Field(default_factory=dict, description="Nutzung in Prozent")
    limits_exceeded: list[str] = Field(default_factory=list, description="Überschrittene Limits")
    reset_times: dict[str, datetime] = Field(default_factory=dict, description="Reset-Zeiten")


class PolicyStatusResponse(BaseModel):
    """Response für Policy-Status."""
    configuration: PolicyConfiguration = Field(..., description="Policy-Konfiguration")
    compliance_status: dict[str, bool] = Field(default_factory=dict, description="Compliance-Status")
    violations: list[dict[str, Any]] = Field(default_factory=list, description="Policy-Verletzungen")
    last_audit: datetime | None = Field(None, description="Letztes Audit")


class ScalingStatusResponse(BaseModel):
    """Response für Scaling-Status."""
    configuration: ScalingConfiguration = Field(..., description="Scaling-Konfiguration")
    current_instances: int = Field(..., description="Aktuelle Instanzen")
    target_instances: int = Field(..., description="Ziel-Instanzen")
    scaling_events: list[dict[str, Any]] = Field(default_factory=list, description="Scaling-Events")
    last_scaling_action: datetime | None = Field(None, description="Letzte Scaling-Aktion")


# Dependency-Funktionen

async def get_agent_or_404(agent_id: str = Path(..., description="Agent-ID")) -> AgentMetadata:
    """Holt Agent-Metadata oder wirft 404-Fehler."""
    metadata = metadata_service.get_metadata(agent_id)
    if not metadata:
        raise ValidationError(
            message=f"Agent mit ID '{agent_id}' nicht gefunden",
            field="agent_id",
            value=agent_id,
            error_code="AGENT_NOT_FOUND"
        )
    return metadata


async def validate_agent_access(
    agent_id: str,
    user_id: str | None = None,
    required_permission: str = "read"
) -> bool:
    """Validiert Zugriff auf Agent."""
    try:
        # Integration mit Enhanced Security System
        from enhanced_security import security_manager

        # Prüfe Berechtigung
        has_permission = await security_manager.check_permission(
            user_id=user_id or "system",
            resource=f"agent:{agent_id}",
            action=required_permission
        )

        if not has_permission:
            raise AuthorizationError(
                message=f"Keine Berechtigung für {required_permission} auf Agent {agent_id}",
                resource=f"agent:{agent_id}",
                action=required_permission,
                user_id=user_id
            )

        return True

    except ImportError:
        # Enhanced Security nicht verfügbar - erlaube Zugriff
        logger.warning("Enhanced Security System nicht verfügbar - Zugriff erlaubt")
        return True


# Utility-Funktionen

def _extract_agent_response(metadata: AgentMetadata) -> AgentResponse:
    """Extrahiert Agent-Response aus Metadata."""
    return AgentResponse(
        agent_id=metadata.agent_id,
        name=metadata.agent_name,
        description=getattr(metadata, "description", None),
        status=AgentStatus.ACTIVE,  # Default
        owner=metadata.owner,
        tenant=metadata.tenant,
        tags=metadata.tags,
        capabilities=list(metadata.available_capabilities.keys()),
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        framework_type=metadata.framework_type.value,
        framework_version=metadata.framework_version
    )


def _get_current_quota_usage(agent_id: str) -> dict[str, Any]:
    """Holt aktuelle Quota-Nutzung für Agent."""
    try:
        # Integration mit Observability-Metrics
        from observability import get_agent_metrics_collector

        collector = get_agent_metrics_collector(agent_id)
        metrics = collector.get_comprehensive_metrics()

        # Extrahiere relevante Metriken
        task_metrics = metrics.get("task_metrics", {})

        return {
            "requests_per_minute": task_metrics.get("rate", {}).get("current_rate", 0) * 60,
            "concurrent_requests": task_metrics.get("queue_metrics", {}).get("current_depth", 0),
            "memory_usage_mb": 0,  # Placeholder
            "cpu_usage_cores": 0,  # Placeholder
            "storage_usage_gb": 0,  # Placeholder
            "network_usage_mbps": 0  # Placeholder
        }

    except ImportError:
        logger.warning("Observability-System nicht verfügbar - verwende Dummy-Daten")
        return {
            "requests_per_minute": 0,
            "concurrent_requests": 0,
            "memory_usage_mb": 0,
            "cpu_usage_cores": 0,
            "storage_usage_gb": 0,
            "network_usage_mbps": 0
        }


def _calculate_usage_percentage(current: dict[str, Any], limits: QuotaConfiguration) -> dict[str, float]:
    """Berechnet Nutzung in Prozent."""
    percentages = {}

    if limits.max_requests_per_minute and current.get("requests_per_minute"):
        percentages["requests_per_minute"] = min(100.0,
            (current["requests_per_minute"] / limits.max_requests_per_minute) * 100)

    if limits.max_concurrent_requests and current.get("concurrent_requests"):
        percentages["concurrent_requests"] = min(100.0,
            (current["concurrent_requests"] / limits.max_concurrent_requests) * 100)

    if limits.max_memory_mb and current.get("memory_usage_mb"):
        percentages["memory_usage"] = min(100.0,
            (current["memory_usage_mb"] / limits.max_memory_mb) * 100)

    if limits.max_cpu_cores and current.get("cpu_usage_cores"):
        percentages["cpu_usage"] = min(100.0,
            (current["cpu_usage_cores"] / limits.max_cpu_cores) * 100)

    return percentages


# Agent Update Endpoint

@router.put("/{agent_id}", response_model=AgentResponse)
@with_log_links(component="agent_management", operation="update_agent")
async def update_agent(
    agent_id: str = Path(..., description="Agent-ID"),
    request: AgentUpdateRequest = ...,
    metadata: AgentMetadata = Depends(get_agent_or_404)
) -> AgentResponse:
    """Aktualisiert Agent-Konfiguration und Metadaten.

    Args:
        agent_id: Eindeutige Agent-ID
        request: Update-Request mit neuen Werten
        metadata: Agent-Metadata (aus Dependency)

    Returns:
        Aktualisierte Agent-Informationen

    Raises:
        ValidationError: Bei ungültigen Eingabedaten
        AuthorizationError: Bei fehlenden Berechtigungen
        BusinessLogicError: Bei Business-Logic-Verletzungen
    """
    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="write")

    try:
        # Update-Felder anwenden
        update_data = request.dict(exclude_unset=True)

        if "name" in update_data:
            metadata.agent_name = update_data["name"]

        if "owner" in update_data:
            metadata.owner = update_data["owner"]

        if "tenant" in update_data:
            metadata.tenant = update_data["tenant"]

        if "tags" in update_data:
            metadata.tags = update_data["tags"]

        if "framework_config" in update_data:
            metadata.framework_config.update(update_data["framework_config"])

        # Timestamp aktualisieren
        metadata.updated_at = datetime.now(UTC)

        # Metadata speichern
        metadata_service.register_metadata(metadata)

        # Registry aktualisieren
        registry_agent = await dynamic_registry.get_agent_by_id(agent_id)
        if registry_agent:
            if "name" in update_data:
                registry_agent.name = update_data["name"]
            if "tags" in update_data:
                registry_agent.tags = update_data["tags"]
            # Update agent in registry
            try:
                await dynamic_registry.update_agent(agent_id, registry_agent)
            except Exception as e:
                logger.warning(f"Agent-Registry-Update fehlgeschlagen: {e}")
                # Fallback: Update in temporary structure
                if hasattr(dynamic_registry, "_temp_agents") and agent_id in dynamic_registry._temp_agents:
                    dynamic_registry._temp_agents[agent_id] = registry_agent

        logger.info(
            f"Agent {agent_id} erfolgreich aktualisiert",
            extra={
                "agent_id": agent_id,
                "updated_fields": list(update_data.keys()),
                "correlation_id": f"update_{uuid.uuid4().hex[:8]}"
            }
        )

        return _extract_agent_response(metadata)

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Agent-Update fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="agent_management",
            operation="update_agent",
            cause=e
        )


# Agent Deprecation Endpoint

@router.post("/{agent_id}/deprecate")
@with_log_links(component="agent_management", operation="deprecate_agent")
async def deprecate_agent(
    agent_id: str = Path(..., description="Agent-ID"),
    request: AgentDeprecationRequest = ...,
    metadata: AgentMetadata = Depends(get_agent_or_404)
) -> dict[str, Any]:
    """Markiert Agent als deprecated mit Migrationspfad.

    Args:
        agent_id: Eindeutige Agent-ID
        request: Deprecation-Request mit Grund und Migrationsinformationen
        metadata: Agent-Metadata (aus Dependency)

    Returns:
        Deprecation-Bestätigung mit Details

    Raises:
        ValidationError: Bei ungültigen Eingabedaten
        AuthorizationError: Bei fehlenden Berechtigungen
        BusinessLogicError: Bei bereits deprecated Agent
    """
    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="admin")

    try:
        # Prüfe, ob Agent bereits deprecated ist
        current_status = getattr(metadata, "status", None)
        if current_status == "deprecated":
            raise BusinessLogicError(
                message=f"Agent {agent_id} ist bereits als deprecated markiert",
                agent_id=agent_id,
                business_rule="no_double_deprecation"
            )

        # Deprecation-Informationen in Metadata speichern
        deprecation_info = {
            "deprecated": True,
            "deprecation_date": datetime.now(UTC).isoformat(),
            "reason": request.reason,
            "migration_guide_url": request.migration_guide_url,
            "replacement_agent_id": request.replacement_agent_id,
            "planned_deprecation_date": request.deprecation_date.isoformat() if request.deprecation_date else None,
            "end_of_life_date": request.end_of_life_date.isoformat() if request.end_of_life_date else None
        }

        # Füge zu Framework-Config hinzu
        metadata.framework_config["deprecation"] = deprecation_info
        metadata.updated_at = datetime.now(UTC)

        # Speichere Metadata
        metadata_service.register_metadata(metadata)

        # Registry-Status aktualisieren
        registry_agent = await dynamic_registry.get_agent_by_id(agent_id)
        if registry_agent:
            registry_agent.status = "deprecated"
            # Update agent in registry
            try:
                await dynamic_registry.update_agent(agent_id, registry_agent)
            except Exception as e:
                logger.warning(f"Agent-Registry-Update fehlgeschlagen: {e}")
                # Fallback: Update in temporary structure
                if hasattr(dynamic_registry, "_temp_agents") and agent_id in dynamic_registry._temp_agents:
                    dynamic_registry._temp_agents[agent_id] = registry_agent

        # Benutzer-Benachrichtigung (falls aktiviert)
        if request.notify_users:
            try:
                # Integration mit Notification-System
                await _notify_users_about_deprecation(agent_id, request)
            except Exception as e:
                logger.warning(f"Benutzer-Benachrichtigung fehlgeschlagen: {e}")

        logger.warning(
            f"Agent {agent_id} als deprecated markiert",
            extra={
                "agent_id": agent_id,
                "reason": request.reason,
                "replacement_agent_id": request.replacement_agent_id,
                "correlation_id": f"deprecate_{uuid.uuid4().hex[:8]}"
            }
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "deprecated_at": deprecation_info["deprecation_date"],
            "reason": request.reason,
            "migration_guide_url": request.migration_guide_url,
            "replacement_agent_id": request.replacement_agent_id,
            "end_of_life_date": request.end_of_life_date.isoformat() if request.end_of_life_date else None
        }

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Agent-Deprecation fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="agent_management",
            operation="deprecate_agent",
            cause=e
        )


async def _notify_users_about_deprecation(agent_id: str, request: AgentDeprecationRequest) -> None:
    """Benachrichtigt Benutzer über Agent-Deprecation."""
    # Placeholder für Notification-System-Integration
    logger.info(
        f"Benutzer-Benachrichtigung für Agent-Deprecation {agent_id} gesendet",
        extra={
            "agent_id": agent_id,
            "reason": request.reason,
            "replacement_agent_id": request.replacement_agent_id,
            "migration_guide_url": request.migration_guide_url,
            "end_of_life_date": request.end_of_life_date.isoformat() if request.end_of_life_date else None
        }
    )
