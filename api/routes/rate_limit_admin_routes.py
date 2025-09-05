"""Admin-APIs für IP-Whitelist/Blacklist Verwaltung (DDoS-Schutz)."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Query

from api.middleware.ddos_middleware import BLACKLIST_KEY, WHITELIST_KEY
from storage.cache.redis_cache import NoOpCache, get_cache_client

router = APIRouter(prefix="/api/v1/rate-limit", tags=["rate-limit-admin"])


@router.get("/lists")
async def get_lists() -> dict[str, Any]:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return {"whitelist": [], "blacklist": []}
    try:
        wl = list(await client.smembers(WHITELIST_KEY))  # type: ignore[attr-defined]
        bl = list(await client.smembers(BLACKLIST_KEY))  # type: ignore[attr-defined]
        return {"whitelist": wl, "blacklist": bl}
    except Exception:
        return {"whitelist": [], "blacklist": []}


@router.post("/whitelist/add")
async def add_whitelist(ip: str = Query(...)) -> dict[str, Any]:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return {"status": "no_cache"}
    try:
        await client.sadd(WHITELIST_KEY, ip)  # type: ignore[attr-defined]
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


@router.post("/whitelist/remove")
async def remove_whitelist(ip: str = Query(...)) -> dict[str, Any]:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return {"status": "no_cache"}
    try:
        await client.srem(WHITELIST_KEY, ip)  # type: ignore[attr-defined]
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


@router.post("/blacklist/add")
async def add_blacklist(ip: str = Query(...)) -> dict[str, Any]:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return {"status": "no_cache"}
    try:
        await client.sadd(BLACKLIST_KEY, ip)  # type: ignore[attr-defined]
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


@router.post("/blacklist/remove")
async def remove_blacklist(ip: str = Query(...)) -> dict[str, Any]:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return {"status": "no_cache"}
    try:
        await client.srem(BLACKLIST_KEY, ip)  # type: ignore[attr-defined]
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}

"""
Admin-API-Endpunkte für Rate Limit Management.

Diese Endpunkte ermöglichen die Verwaltung und Überwachung des
production-ready Rate Limiting Systems.
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from config.unified_rate_limiting import RateLimitTier, get_rate_limit_config
from config.unified_rate_limiting import (
    reload_unified_rate_limit_config as reload_rate_limit_config,
)
from kei_logging import get_logger
from middleware.rate_limiting import get_rate_limit_manager
from observability import trace_function
from security.kei_mcp_auth import require_auth

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# PYDANTIC-MODELLE
# ============================================================================

class RateLimitStatus(BaseModel):
    """Status eines Rate Limits."""
    client_id: str = Field(..., description="Client-Identifikation")
    operation: str = Field(..., description="Operation")
    current_usage: int = Field(..., description="Aktuelle Nutzung")
    limit: int = Field(..., description="Rate Limit")
    remaining: int = Field(..., description="Verbleibende Requests")
    reset_time: int = Field(..., description="Reset-Zeitstempel")
    window_start: float | None = Field(None, description="Window-Start-Zeit")
    tokens_remaining: float | None = Field(None, description="Verbleibende Tokens (Token Bucket)")
    bucket_capacity: int | None = Field(None, description="Bucket-Kapazität")


class RateLimitHealthStatus(BaseModel):
    """Gesundheitsstatus des Rate Limiting Systems."""
    backend_type: str = Field(..., description="Aktueller Backend-Typ")
    config_backend: str = Field(..., description="Konfigurierter Backend-Typ")
    redis_healthy: bool = Field(..., description="Redis-Gesundheitsstatus")
    memory_available: bool = Field(..., description="Memory-Backend verfügbar")
    cleanup_running: bool = Field(..., description="Cleanup-Task läuft")
    total_api_keys: int = Field(..., description="Anzahl registrierter API-Keys")
    whitelisted_ips: int = Field(..., description="Anzahl Whitelist-IPs")


class ApiKeyTierMapping(BaseModel):
    """API-Key zu Tier Mapping."""
    api_key: str = Field(..., description="API-Key (gekürzt für Anzeige)")
    tier: RateLimitTier = Field(..., description="Rate Limit Tier")
    created_at: str | None = Field(None, description="Erstellungszeitpunkt")


class RateLimitConfigSummary(BaseModel):
    """Zusammenfassung der Rate Limit Konfiguration."""
    backend: str = Field(..., description="Backend-Typ")
    default_tier: RateLimitTier = Field(..., description="Standard-Tier")
    redis_host: str = Field(..., description="Redis Host")
    redis_port: int = Field(..., description="Redis Port")
    memory_fallback_enabled: bool = Field(..., description="Memory-Fallback aktiviert")
    enable_metrics: bool = Field(..., description="Metriken aktiviert")
    include_rate_limit_headers: bool = Field(..., description="Rate Limit Headers aktiviert")
    cleanup_interval_seconds: int = Field(..., description="Cleanup-Intervall")


class RateLimitResetRequest(BaseModel):
    """Request für Rate Limit Reset."""
    client_id: str = Field(..., description="Client-ID zum Zurücksetzen")
    operation: str | None = Field(None, description="Spezifische Operation (optional)")


class ApiKeyTierRequest(BaseModel):
    """Request für API-Key-Tier-Management."""
    api_key: str = Field(..., description="API-Key")
    tier: RateLimitTier = Field(..., description="Rate Limit Tier")


class IpWhitelistRequest(BaseModel):
    """Request für IP-Whitelist-Management."""
    ip_address: str = Field(..., description="IP-Adresse")


# ============================================================================
# API-ENDPUNKTE
# ============================================================================

@router.get("/rate-limit/status", response_model=RateLimitHealthStatus)
@trace_function("api.rate_limit_admin.get_status")
async def get_rate_limit_status(
    _: str = Depends(require_auth)
) -> RateLimitHealthStatus:
    """Gibt Gesundheitsstatus des Rate Limiting Systems zurück.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()
        health_status = await rate_limit_manager.health_check()

        return RateLimitHealthStatus(
            backend_type=health_status["backend_type"],
            config_backend=health_status["config_backend"],
            redis_healthy=health_status["redis_healthy"],
            memory_available=health_status["memory_available"],
            cleanup_running=health_status["cleanup_running"],
            total_api_keys=len(rate_limit_manager._api_key_tiers),
            whitelisted_ips=len(rate_limit_manager._ip_whitelist)
        )

    except Exception as e:
        logger.exception(f"Fehler beim Abrufen des Rate Limit Status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Abrufen des Rate Limit Status"
        )


@router.get("/rate-limit/config", response_model=RateLimitConfigSummary)
@trace_function("api.rate_limit_admin.get_config")
async def get_rate_limit_config_summary(
    _: str = Depends(require_auth)
) -> RateLimitConfigSummary:
    """Gibt Zusammenfassung der Rate Limit Konfiguration zurück.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        config = get_rate_limit_config()

        return RateLimitConfigSummary(
            backend=config.backend.value,
            default_tier=config.default_tier,
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            memory_fallback_enabled=config.memory_fallback_enabled,
            enable_metrics=config.enable_metrics,
            include_rate_limit_headers=config.include_rate_limit_headers,
            cleanup_interval_seconds=config.cleanup_interval_seconds
        )

    except Exception as e:
        logger.exception(f"Fehler beim Abrufen der Rate Limit Konfiguration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Abrufen der Konfiguration"
        )


@router.get("/rate-limit/clients/{client_id}", response_model=dict[str, Any])
@trace_function("api.rate_limit_admin.get_client_info")
async def get_client_rate_limit_info(
    client_id: str = Path(..., description="Client-ID"),
    operation: str | None = Query(None, description="Spezifische Operation"),
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Gibt Rate Limit Informationen für spezifischen Client zurück.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        if operation:
            # Spezifische Operation
            info = await rate_limit_manager.get_rate_limit_info(client_id, operation)
            return {"client_id": client_id, "operation": operation, "info": info}
        # Alle Operationen
        operations = ["default", "register", "invoke", "discovery", "stats"]
        all_info = {}

        for op in operations:
            info = await rate_limit_manager.get_rate_limit_info(client_id, op)
            if info:
                all_info[op] = info

        return {"client_id": client_id, "operations": all_info}

    except Exception as e:
        logger.exception(f"Fehler beim Abrufen der Client Rate Limit Info für {client_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Abrufen der Client-Informationen"
        )


@router.post("/rate-limit/reset")
@trace_function("api.rate_limit_admin.reset_rate_limit")
async def reset_client_rate_limit(
    request: RateLimitResetRequest,
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Setzt Rate Limit für spezifischen Client zurück.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        success = await rate_limit_manager.reset_rate_limit(
            request.client_id,
            request.operation
        )

        if success:
            logger.info(f"Rate Limit für {request.client_id} zurückgesetzt "
                       f"(Operation: {request.operation or 'alle'})")

            return {
                "success": True,
                "message": f"Rate Limit für {request.client_id} erfolgreich zurückgesetzt",
                "client_id": request.client_id,
                "operation": request.operation,
                "reset_time": time.time()
            }
        raise HTTPException(
            status_code=404,
            detail=f"Keine Rate Limit Daten für Client {request.client_id} gefunden"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Zurücksetzen von Rate Limit für {request.client_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Zurücksetzen des Rate Limits"
        )


@router.get("/rate-limit/api-keys", response_model=list[ApiKeyTierMapping])
@trace_function("api.rate_limit_admin.list_api_key_tiers")
async def list_api_key_tiers(
    _: str = Depends(require_auth)
) -> list[ApiKeyTierMapping]:
    """Listet alle API-Key-Tier-Mappings auf.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        mappings = []
        for api_key, tier in rate_limit_manager._api_key_tiers.items():
            mappings.append(ApiKeyTierMapping(
                api_key=f"{api_key[:8]}...{api_key[-4:]}",  # Gekürzt für Sicherheit
                tier=tier
            ))

        return mappings

    except Exception as e:
        logger.exception(f"Fehler beim Auflisten der API-Key-Tiers: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Auflisten der API-Key-Tiers"
        )


@router.post("/rate-limit/api-keys")
@trace_function("api.rate_limit_admin.add_api_key_tier")
async def add_api_key_tier(
    request: ApiKeyTierRequest,
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Fügt API-Key-Tier-Mapping hinzu.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        await rate_limit_manager.add_api_key_tier(request.api_key, request.tier)

        return {
            "success": True,
            "message": "API-Key-Tier-Mapping erfolgreich hinzugefügt",
            "api_key": f"{request.api_key[:8]}...{request.api_key[-4:]}",
            "tier": request.tier.value,
            "created_at": time.time()
        }

    except Exception as e:
        logger.exception(f"Fehler beim Hinzufügen des API-Key-Tier-Mappings: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Hinzufügen des API-Key-Tier-Mappings"
        )


@router.delete("/rate-limit/api-keys/{api_key}")
@trace_function("api.rate_limit_admin.remove_api_key_tier")
async def remove_api_key_tier(
    api_key: str = Path(..., description="API-Key"),
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Entfernt API-Key-Tier-Mapping.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        if api_key not in rate_limit_manager._api_key_tiers:
            raise HTTPException(
                status_code=404,
                detail=f"API-Key-Tier-Mapping für {api_key[:8]}... nicht gefunden"
            )

        await rate_limit_manager.remove_api_key_tier(api_key)

        return {
            "success": True,
            "message": "API-Key-Tier-Mapping erfolgreich entfernt",
            "api_key": f"{api_key[:8]}...{api_key[-4:]}",
            "removed_at": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Entfernen des API-Key-Tier-Mappings: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Entfernen des API-Key-Tier-Mappings"
        )


@router.get("/rate-limit/whitelist", response_model=list[str])
@trace_function("api.rate_limit_admin.list_ip_whitelist")
async def list_ip_whitelist(
    _: str = Depends(require_auth)
) -> list[str]:
    """Listet alle IPs in der Whitelist auf.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()
        return list(rate_limit_manager._ip_whitelist)

    except Exception as e:
        logger.exception(f"Fehler beim Auflisten der IP-Whitelist: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Auflisten der IP-Whitelist"
        )


@router.post("/rate-limit/whitelist")
@trace_function("api.rate_limit_admin.add_ip_to_whitelist")
async def add_ip_to_whitelist(
    request: IpWhitelistRequest,
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Fügt IP zur Whitelist hinzu.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        rate_limit_manager.add_ip_to_whitelist(request.ip_address)

        return {
            "success": True,
            "message": f"IP {request.ip_address} erfolgreich zur Whitelist hinzugefügt",
            "ip_address": request.ip_address,
            "added_at": time.time()
        }

    except Exception as e:
        logger.exception(f"Fehler beim Hinzufügen der IP zur Whitelist: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Hinzufügen der IP zur Whitelist"
        )


@router.delete("/rate-limit/whitelist/{ip_address}")
@trace_function("api.rate_limit_admin.remove_ip_from_whitelist")
async def remove_ip_from_whitelist(
    ip_address: str = Path(..., description="IP-Adresse"),
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Entfernt IP von der Whitelist.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        if ip_address not in rate_limit_manager._ip_whitelist:
            raise HTTPException(
                status_code=404,
                detail=f"IP {ip_address} nicht in Whitelist gefunden"
            )

        rate_limit_manager.remove_ip_from_whitelist(ip_address)

        return {
            "success": True,
            "message": f"IP {ip_address} erfolgreich von Whitelist entfernt",
            "ip_address": ip_address,
            "removed_at": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Entfernen der IP von der Whitelist: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Entfernen der IP von der Whitelist"
        )


@router.post("/rate-limit/config/reload")
@trace_function("api.rate_limit_admin.reload_config")
async def reload_rate_limit_config_endpoint(
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Lädt Rate Limit Konfiguration neu.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        import time

        logger.info("Starte manuellen Config-Reload für Rate Limiting")

        old_config = get_rate_limit_config()
        new_config = reload_rate_limit_config()

        # Prüfe auf Änderungen
        changes = {}
        if old_config.backend != new_config.backend:
            changes["backend"] = {"old": old_config.backend.value, "new": new_config.backend.value}

        if old_config.default_tier != new_config.default_tier:
            changes["default_tier"] = {"old": old_config.default_tier.value, "new": new_config.default_tier.value}

        if old_config.redis_host != new_config.redis_host:
            changes["redis_host"] = {"old": old_config.redis_host, "new": new_config.redis_host}

        logger.info(f"Rate Limit Config-Reload abgeschlossen. Änderungen: {len(changes)}")

        return {
            "success": True,
            "message": "Rate Limit Konfiguration erfolgreich neu geladen",
            "changes": changes,
            "reload_time": time.time()
        }

    except Exception as e:
        logger.exception(f"Fehler beim Rate Limit Config-Reload: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Config-Reload"
        )


@router.post("/rate-limit/cleanup")
@trace_function("api.rate_limit_admin.manual_cleanup")
async def manual_rate_limit_cleanup(
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Führt manuelle Bereinigung abgelaufener Rate Limit Einträge durch.

    **Authentifizierung:** Bearer Token erforderlich (Admin-Berechtigung)
    """
    try:
        rate_limit_manager = get_rate_limit_manager()

        start_time = time.time()
        deleted_entries = await rate_limit_manager._current_backend.cleanup_expired()
        duration = time.time() - start_time

        logger.info(f"Manuelle Rate Limit Bereinigung abgeschlossen: "
                   f"{deleted_entries} Einträge in {duration:.2f}s")

        return {
            "success": True,
            "message": "Rate Limit Bereinigung erfolgreich durchgeführt",
            "deleted_entries": deleted_entries,
            "duration_seconds": duration,
            "cleanup_time": time.time()
        }

    except Exception as e:
        logger.exception(f"Fehler bei manueller Rate Limit Bereinigung: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler bei Rate Limit Bereinigung"
        )
