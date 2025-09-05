"""KEI-Stream Rate-Limiting Admin-Endpunkte.

Bietet Admin-Funktionalität für Rate-Limiting-Management:
- Konfiguration anzeigen und bearbeiten
- Rate-Limit-Status überwachen
- Hot-Reload von Konfigurationen
- Metriken und Statistiken

@version 1.0.0
"""

from typing import Any

# Admin-Access-Funktion (vereinfacht für Rate-Limiting-Admin)
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field

from config.unified_rate_limiting import (
    RateLimitTier as TenantTier,
)
from config.unified_rate_limiting import (
    UnifiedRateLimitConfig as RateLimitConfigModel,
)
from config.unified_rate_limiting import (
    get_unified_rate_limit_config as get_rate_limiting_config_manager,
)
from config.unified_rate_limiting import (
    reload_unified_rate_limit_config as reload_rate_limiting_config,
)
from middleware.kei_stream_rate_limiting import (
    KEIStreamEndpointType,
    KEIStreamIdentificationStrategy,
)
from middleware.rate_limiting_types import KEIStreamRateLimitStrategy
from observability import get_logger


def require_admin_access(_request: Request = None) -> None:
    """Vereinfachte Admin-Access-Prüfung für Rate-Limiting-Admin-Endpunkte.

    In einer Produktionsumgebung sollte hier eine echte Admin-Authentifizierung
    implementiert werden (z.B. über JWT-Scopes oder spezielle Admin-Tokens).
    """
    # Für jetzt: Einfache Implementierung, die immer erlaubt
    # TODO: Echte Admin-Authentifizierung implementieren - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/108

logger = get_logger(__name__)

router = APIRouter(prefix="/admin/rate-limiting", tags=["Rate-Limiting-Admin"])


class RateLimitStatusResponse(BaseModel):
    """Response-Modell für Rate-Limit-Status."""
    identifier: str
    strategy: str
    endpoint_type: str
    current_usage: dict[str, Any]
    limits: dict[str, Any]
    blocked: bool
    remaining_requests: int
    reset_time: float


class ConfigurationSummaryResponse(BaseModel):
    """Response-Modell für Konfigurationszusammenfassung."""
    tenants: dict[str, Any]
    endpoints: dict[str, Any]
    api_keys: dict[str, Any]
    tier_defaults: dict[str, Any]
    validation_warnings: list[str]


class TenantConfigRequest(BaseModel):
    """Request-Modell für Tenant-Konfiguration."""
    tenant_id: str = Field(description="Eindeutige Tenant-ID")
    tier: TenantTier = Field(description="Tenant-Tier-Level")
    custom_limits: RateLimitConfigModel | None = Field(None, description="Benutzerdefinierte Limits")
    api_keys: list[str] = Field(default_factory=list, description="Zugeordnete API-Keys")
    enabled: bool = Field(default=True, description="Tenant aktiviert")
    features: list[str] = Field(default_factory=list, description="Aktivierte Features")


@router.get("/status", response_model=ConfigurationSummaryResponse)
async def get_rate_limiting_status(
    _: None = Depends(require_admin_access)
) -> ConfigurationSummaryResponse:
    """Gibt aktuellen Status der Rate-Limiting-Konfiguration zurück."""
    try:
        config_manager = get_rate_limiting_config_manager()
        summary = config_manager.get_configuration_summary()
        warnings = config_manager.validate_configuration()

        return ConfigurationSummaryResponse(
            tenants=summary["tenants"],
            endpoints=summary["endpoints"],
            api_keys=summary["api_keys"],
            tier_defaults=summary["tier_defaults"],
            validation_warnings=warnings
        )
    except Exception as e:
        logger.exception(f"Fehler beim Abrufen des Rate-Limiting-Status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tenants")
async def list_tenants(
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Listet alle konfigurierten Tenants auf."""
    try:
        config_manager = get_rate_limiting_config_manager()
        tenants = {}

        for tenant_id, tenant_config in config_manager._tenant_configs.items():
            tenants[tenant_id] = {
                "tier": tenant_config.tier.value,
                "enabled": tenant_config.enabled,
                "api_keys_count": len(tenant_config.api_keys),
                "features": tenant_config.features,
                "has_custom_limits": tenant_config.custom_limits is not None
            }

        return {
            "tenants": tenants,
            "total_count": len(tenants)
        }
    except Exception as e:
        logger.exception(f"Fehler beim Auflisten der Tenants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tenants/{tenant_id}")
async def get_tenant_config(
    tenant_id: str = Path(description="Tenant-ID"),
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Gibt detaillierte Konfiguration für spezifischen Tenant zurück."""
    try:
        config_manager = get_rate_limiting_config_manager()

        if tenant_id not in config_manager._tenant_configs:
            raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' nicht gefunden")

        tenant_config = config_manager._tenant_configs[tenant_id]
        rate_limit_config = config_manager.get_tenant_config(tenant_id)

        return {
            "tenant_id": tenant_id,
            "tier": tenant_config.tier.value,
            "enabled": tenant_config.enabled,
            "api_keys": tenant_config.api_keys,
            "features": tenant_config.features,
            "custom_limits": tenant_config.custom_limits.dict() if tenant_config.custom_limits else None,
            "effective_limits": {
                "requests_per_second": rate_limit_config.requests_per_second,
                "burst_capacity": rate_limit_config.burst_capacity,
                "frames_per_second": rate_limit_config.frames_per_second,
                "max_concurrent_streams": rate_limit_config.max_concurrent_streams,
                "algorithm_strategy": rate_limit_config.algorithm_strategy.value,
                "identification_strategy": rate_limit_config.identification_strategy.value,
                "endpoint_type": rate_limit_config.endpoint_type.value
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Abrufen der Tenant-Konfiguration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints")
async def list_endpoints(
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Listet alle konfigurierten Endpoint-Patterns auf."""
    try:
        config_manager = get_rate_limiting_config_manager()
        endpoints = {}

        for pattern, endpoint_config in config_manager._endpoint_configs.items():
            endpoints[pattern] = {
                "description": endpoint_config.description,
                "tags": endpoint_config.tags,
                "endpoint_type": endpoint_config.config.endpoint_type.value,
                "algorithm_strategy": endpoint_config.config.algorithm_strategy.value,
                "identification_strategy": endpoint_config.config.identification_strategy.value,
                "requests_per_second": endpoint_config.config.requests_per_second,
                "burst_capacity": endpoint_config.config.burst_capacity,
                "frames_per_second": endpoint_config.config.frames_per_second,
                "enabled": endpoint_config.config.enabled
            }

        return {
            "endpoints": endpoints,
            "total_count": len(endpoints)
        }
    except Exception as e:
        logger.exception(f"Fehler beim Auflisten der Endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-keys/{api_key}/tenant")
async def get_tenant_for_api_key(
    api_key: str = Path(description="API-Key"),
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Ermittelt Tenant für spezifischen API-Key."""
    try:
        config_manager = get_rate_limiting_config_manager()
        tenant_id = config_manager.get_tenant_by_api_key(api_key)

        if not tenant_id:
            raise HTTPException(status_code=404, detail=f"API-Key '{api_key}' nicht gefunden")

        tenant_config = config_manager.get_tenant_config(tenant_id)

        return {
            "api_key": api_key,
            "tenant_id": tenant_id,
            "tenant_enabled": config_manager.is_tenant_enabled(tenant_id),
            "tenant_features": config_manager.get_tenant_features(tenant_id),
            "rate_limits": {
                "requests_per_second": tenant_config.requests_per_second,
                "burst_capacity": tenant_config.burst_capacity,
                "frames_per_second": tenant_config.frames_per_second,
                "max_concurrent_streams": tenant_config.max_concurrent_streams
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Abrufen des Tenants für API-Key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_configuration(
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Lädt Rate-Limiting-Konfiguration neu (Hot-Reload)."""
    try:
        success = reload_rate_limiting_config()

        if success:
            config_manager = get_rate_limiting_config_manager()
            summary = config_manager.get_configuration_summary()
            warnings = config_manager.validate_configuration()

            return {
                "success": True,
                "message": "Konfiguration erfolgreich neu geladen",
                "summary": summary,
                "warnings": warnings
            }
        return {
            "success": False,
            "message": "Fehler beim Neuladen der Konfiguration"
        }
    except Exception as e:
        logger.exception(f"Fehler beim Neuladen der Konfiguration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate")
async def validate_configuration(
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Validiert aktuelle Rate-Limiting-Konfiguration."""
    try:
        config_manager = get_rate_limiting_config_manager()
        warnings = config_manager.validate_configuration()

        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "warning_count": len(warnings)
        }
    except Exception as e:
        logger.exception(f"Fehler bei der Konfigurationsvalidierung: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_rate_limiting_metrics(
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Gibt Rate-Limiting-Metriken zurück."""
    try:
        # Hier würden wir normalerweise die Middleware-Instanz abrufen
        # Für jetzt geben wir Platzhalter-Metriken zurück
        return {
            "requests_allowed": 0,
            "requests_blocked": 0,
            "frames_allowed": 0,
            "frames_blocked": 0,
            "streams_created": 0,
            "streams_rejected": 0,
            "redis_errors": 0,
            "fallback_used": 0,
            "local_buckets_count": 0,
            "redis_connected": False
        }
    except Exception as e:
        logger.exception(f"Fehler beim Abrufen der Rate-Limiting-Metriken: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tiers")
async def list_available_tiers(
    _: None = Depends(require_admin_access)
) -> dict[str, Any]:
    """Listet verfügbare Tenant-Tiers auf."""
    try:
        config_manager = get_rate_limiting_config_manager()

        tiers = {}
        for tier, config in config_manager._tier_defaults.items():
            tiers[tier.value] = {
                "requests_per_second": config.requests_per_second,
                "burst_capacity": config.burst_capacity,
                "frames_per_second": config.frames_per_second,
                "max_concurrent_streams": config.max_concurrent_streams,
                "max_stream_duration_seconds": config.max_stream_duration_seconds,
                "algorithm_strategy": config.algorithm_strategy.value,
                "identification_strategy": config.identification_strategy.value,
                "enabled": config.enabled
            }

        return {
            "tiers": tiers,
            "available_algorithm_strategies": [strategy.value for strategy in KEIStreamRateLimitStrategy],
            "available_identification_strategies": [strategy.value for strategy in KEIStreamIdentificationStrategy],
            "available_endpoint_types": [endpoint_type.value for endpoint_type in KEIStreamEndpointType]
        }
    except Exception as e:
        logger.exception(f"Fehler beim Auflisten der Tiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))
