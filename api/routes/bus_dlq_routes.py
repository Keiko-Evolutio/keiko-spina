"""DLQ/Parking/Requeue Admin-APIs für KEI-Bus."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.messaging.dlq import ensure_dlq_stream, park_message, requeue_message

# Service-Imports
try:
    from services.messaging.service import get_bus_service
    BUS_SERVICE_AVAILABLE = True
except ImportError:
    get_bus_service = None
    BUS_SERVICE_AVAILABLE = False

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/bus/dlq", tags=["KEI-Bus-DLQ"])


class DLQReplayRequest(BaseModel):
    """Requeue/Replay Request."""

    original_subject: str = Field(..., description="Originales Subject zum Requeue")
    data_base64: str = Field(..., description="Nachrichtendaten (Base64)")
    headers: dict[str, Any] | None = None


@router.post("/requeue")
async def dlq_requeue(req: DLQReplayRequest):
    """Requeue einer DLQ-Message zurück ins Original-Subject."""
    try:
        import base64

        if not BUS_SERVICE_AVAILABLE:
            raise ImportError("Bus service not available")

        data = base64.b64decode(req.data_base64)
        from services.messaging.service import get_bus_service
        svc = get_bus_service()
        await svc.initialize()
        js = svc._nats.js  # type: ignore[attr-defined]
        await ensure_dlq_stream(js)
        await requeue_message(js, req.original_subject, data, req.headers or {})
        return {"status": "ok"}
    except Exception as exc:
        logger.exception(f"DLQ-Requeue fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="dlq_requeue_failed")


class DLQParkRequest(BaseModel):
    """Parking einer Message in Parking-Subject."""

    subject: str
    data_base64: str
    headers: dict[str, Any] | None = None


@router.post("/park")
async def dlq_park(req: DLQParkRequest):
    """Parkt eine Nachricht in ein Parking-Subject."""
    try:
        import base64

        if not BUS_SERVICE_AVAILABLE:
            raise ImportError("Bus service not available")

        data = base64.b64decode(req.data_base64)
        svc = get_bus_service()
        await svc.initialize()
        js = svc._nats.js  # type: ignore[attr-defined]
        await ensure_dlq_stream(js)
        await park_message(js, req.subject, data, req.headers or {})
        return {"status": "ok"}
    except Exception as exc:
        logger.exception(f"DLQ-Park fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="dlq_park_failed")
