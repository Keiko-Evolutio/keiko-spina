"""Core API Routes.

Zentrale API-Endpunkte für grundlegende Systemfunktionen.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/core", tags=["core"])


@router.get("/health", operation_id="core_health_check")
async def health_check():
    """Basis-Gesundheitscheck für Core-System."""
    return {"status": "healthy", "service": "core"}


@router.get("/info", operation_id="core_system_info")
async def system_info():
    """Grundlegende Systeminformationen."""
    return {
        "service": "keiko-backend",
        "version": "1.0.0",
        "status": "running"
    }
