"""KEI Agents API Routes.

Spezielle API-Endpunkte für KEI-Agent-System.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/kei-agents", tags=["kei-agents"])


@router.get("/health")
async def kei_agents_health():
    """Gesundheitscheck für KEI-Agents-System."""
    return {"status": "healthy", "service": "kei-agents"}


@router.get("/")
async def list_kei_agents():
    """Liste aller KEI-Agents."""
    return {"agents": [], "total": 0}
