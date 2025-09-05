"""Voice Rate Limiting API Routes.
Admin-APIs für Voice Rate Limiting Management und Monitoring.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from core.container import get_container
from kei_logging import get_logger
from voice_rate_limiting.interfaces import (
    IVoiceRateLimitService,
    UserTier,
    VoiceOperation,
    VoiceRateLimitContext,
)

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/voice-rate-limiting", tags=["voice-rate-limiting"])


# Pydantic Models für API
class RateLimitStatusRequest(BaseModel):
    """Request Model für Rate Limit Status."""
    user_id: str
    session_id: str | None = None
    ip_address: str | None = None
    user_tier: UserTier = UserTier.STANDARD
    endpoint: str | None = None


class RateLimitTestRequest(BaseModel):
    """Request Model für Rate Limit Test."""
    operation: VoiceOperation
    user_id: str
    session_id: str | None = None
    ip_address: str | None = None
    user_tier: UserTier = UserTier.STANDARD
    amount: int = 1


class UserTierUpdateRequest(BaseModel):
    """Request Model für User Tier Update."""
    user_id: str
    new_tier: UserTier


def get_voice_rate_limit_service() -> IVoiceRateLimitService:
    """Dependency für Voice Rate Limiting Service."""
    try:
        container = get_container()
        return container.resolve(IVoiceRateLimitService)
    except Exception as e:
        logger.error(f"Failed to resolve voice rate limiting service: {e}")
        raise HTTPException(status_code=503, detail="Voice rate limiting service not available")


@router.get("/status", summary="Voice Rate Limiting Service Status")
async def get_service_status(
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Gibt Status des Voice Rate Limiting Services zurück.

    Returns:
        Dict mit Service-Status und Statistiken
    """
    try:
        # Dummy-Kontext für Service-Status
        context = VoiceRateLimitContext(user_id="system")
        status = await service.get_rate_limit_status(context)

        # Globale Statistiken hinzufügen
        global_stats = await service.get_global_statistics()
        status["global_statistics"] = global_stats

        return status
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@router.post("/status/user", summary="User-specific Rate Limit Status")
async def get_user_rate_limit_status(
    request: RateLimitStatusRequest,
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Gibt Rate Limit Status für spezifischen User zurück.

    Args:
        request: User-Kontext für Status-Abfrage
        service: Voice Rate Limit Service für Status-Abfrage

    Returns:
        Dict mit Rate Limit Status für alle Voice-Operationen
    """
    try:
        context = VoiceRateLimitContext(
            user_id=request.user_id,
            session_id=request.session_id,
            ip_address=request.ip_address,
            user_tier=request.user_tier,
            endpoint=request.endpoint
        )

        status = await service.get_rate_limit_status(context)
        return status
    except Exception as e:
        logger.error(f"Failed to get user rate limit status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user rate limit status")


@router.post("/test", summary="Test Rate Limit")
async def test_rate_limit(
    request: RateLimitTestRequest,
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Testet Rate Limit für spezifische Operation.

    Args:
        request: Test-Parameter

    Returns:
        Dict mit Rate Limit Test-Ergebnis
    """
    try:
        context = VoiceRateLimitContext(
            user_id=request.user_id,
            session_id=request.session_id,
            ip_address=request.ip_address,
            user_tier=request.user_tier
        )

        # Erst prüfen
        check_result = await service.rate_limiter.check_rate_limit(request.operation, context)

        # Dann konsumieren (falls erlaubt)
        if check_result.allowed:
            consume_result = await service.rate_limiter.consume_rate_limit(
                request.operation, context, request.amount
            )
        else:
            consume_result = check_result

        return {
            "operation": request.operation.value,
            "user_id": request.user_id,
            "user_tier": request.user_tier.value,
            "amount": request.amount,
            "check_result": {
                "allowed": check_result.allowed,
                "limit": check_result.limit,
                "remaining": check_result.remaining,
                "reset_time": check_result.reset_time.isoformat()
            },
            "consume_result": {
                "allowed": consume_result.allowed,
                "limit": consume_result.limit,
                "remaining": consume_result.remaining,
                "reset_time": consume_result.reset_time.isoformat(),
                "retry_after_seconds": consume_result.retry_after_seconds
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to test rate limit: {e}")
        raise HTTPException(status_code=500, detail="Failed to test rate limit")


@router.post("/reset/user/{user_id}", summary="Reset User Rate Limits")
async def reset_user_rate_limits(
    user_id: str = Path(..., description="User ID to reset"),
    session_id: str | None = Query(None, description="Optional session ID"),
    ip_address: str | None = Query(None, description="Optional IP address"),
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Setzt Rate Limits für User zurück (Admin-Funktion).

    Args:
        user_id: User ID
        session_id: Optional Session ID
        ip_address: Optional IP Address
        service: Voice Rate Limit Service für Reset-Operation

    Returns:
        Dict mit Reset-Bestätigung
    """
    try:
        context = VoiceRateLimitContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address
        )

        await service.reset_rate_limits(context)

        return {
            "message": f"Rate limits reset for user {user_id}",
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset user rate limits: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset user rate limits")


@router.post("/user-tier/update", summary="Update User Tier")
async def update_user_tier(
    request: UserTierUpdateRequest,
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Aktualisiert User-Tier (Admin-Funktion).

    Args:
        request: User Tier Update Request
        service: Voice Rate Limit Service für User-Tier-Update

    Returns:
        Dict mit Update-Bestätigung
    """
    try:
        await service.update_user_tier(request.user_id, request.new_tier)

        return {
            "message": f"User tier updated for {request.user_id}",
            "user_id": request.user_id,
            "old_tier": "unknown",  # In vollständiger Implementation aus DB holen
            "new_tier": request.new_tier.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to update user tier: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user tier")


@router.get("/health", summary="Voice Rate Limiting Health Check")
async def health_check(
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Führt Health Check für Voice Rate Limiting Service durch.

    Returns:
        Dict mit Health-Status
    """
    try:
        health = await service.health_check()

        if not health["healthy"]:
            raise HTTPException(status_code=503, detail="Voice rate limiting service unhealthy")

        return health
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/statistics", summary="Global Rate Limiting Statistics")
async def get_global_statistics(
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Gibt globale Rate Limiting Statistiken zurück.

    Returns:
        Dict mit globalen Statistiken
    """
    try:
        return await service.get_global_statistics()
    except Exception as e:
        logger.error(f"Failed to get global statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get global statistics")


@router.get("/operations", summary="List Voice Operations")
async def list_voice_operations() -> list[dict[str, str]]:
    """Gibt Liste aller Voice-Operationen zurück.

    Returns:
        Liste mit Voice-Operationen
    """
    return [
        {"value": op.value, "name": op.name}
        for op in VoiceOperation
    ]


@router.get("/user-tiers", summary="List User Tiers")
async def list_user_tiers() -> list[dict[str, str]]:
    """Gibt Liste aller User-Tiers zurück.

    Returns:
        Liste mit User-Tiers
    """
    return [
        {"value": tier.value, "name": tier.name}
        for tier in UserTier
    ]


@router.get("/adaptive/status", summary="Adaptive Rate Limiting Status")
async def get_adaptive_status(
    service: IVoiceRateLimitService = Depends(get_voice_rate_limit_service)
) -> dict[str, Any]:
    """Gibt Status des Adaptive Rate Limiting zurück.

    Returns:
        Dict mit Adaptive Rate Limiting Status
    """
    try:
        # Hole Adaptation-Status
        adaptation_status = service.adaptive_limiter.get_adaptation_status()

        # Aktuelle Limits für verschiedene Tiers sammeln
        current_limits = {}
        for operation in VoiceOperation:
            current_limits[operation.value] = {}
            for tier in UserTier:
                try:
                    config = await service.adaptive_limiter.get_current_limits(operation, tier)
                    current_limits[operation.value][tier.value] = {
                        "limit": config.limit,
                        "window_seconds": config.window_seconds,
                        "algorithm": config.algorithm.value,
                        "burst_limit": config.burst_limit
                    }
                except Exception as e:
                    current_limits[operation.value][tier.value] = {"error": str(e)}

        return {
            "adaptation_status": adaptation_status,
            "current_limits": current_limits,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get adaptive status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get adaptive status")
