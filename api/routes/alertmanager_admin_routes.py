"""Administrative Endpunkte für Alerting DLQ/Health.

Erlaubt das Abfragen der DLQ-Größe und das manuelle Replaying.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from services.webhooks.alerting import alert_dlq_replay, alert_dlq_size

router = APIRouter(prefix="/api/v1/alerts/admin", tags=["alerts-admin"])


@router.get("/dlq/size")
async def get_dlq_size() -> dict[str, int]:
    """Gibt die Größe der Alert-DLQ zurück."""
    return {"size": await alert_dlq_size()}


@router.post("/dlq/replay")
async def replay_dlq(max_items: int = Query(default=50, ge=1, le=1000)) -> dict[str, int]:
    """Replayed bis zu N DLQ-Einträge (best effort)."""
    return await alert_dlq_replay(max_items=max_items)


__all__ = ["router"]
