"""Admin API Routes.

Administrative API-Endpunkte für Systemverwaltung.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health")
async def admin_health():
    """Gesundheitscheck für Admin-System."""
    return {"status": "healthy", "service": "admin"}


@router.get("/status")
async def system_status():
    """Systemstatus für Administratoren."""
    return {
        "system": "operational",
        "services": {
            "api": "running",
            "database": "connected",
            "cache": "available"
        }
    }
