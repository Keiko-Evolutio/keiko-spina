"""Management- und Test-API für KEI-Bus.

Stellt Endpunkte bereit für:
- Publish-Test
- Subscription-Setup (vereinfacht)
- Status/Health und Metrics-Übersicht
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.messaging import BusEnvelope, get_bus_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/bus", tags=["KEI-Bus"])


class PublishRequest(BaseModel):
    """Payload für Publish-Test."""

    subject: str = Field(..., description="Subject/Topic zum Publizieren")
    type: str = Field(..., description="Event-/Command-Typ")
    tenant: str | None = Field(None, description="Tenant/Namespace")
    key: str | None = Field(None, description="Ordering-/Partition-Key")
    payload: dict[str, Any] = Field(default_factory=dict, description="Nutzdaten")
    headers: dict[str, Any] = Field(default_factory=dict, description="Zusätzliche Header")


@router.post("/publish")
async def publish_message(req: PublishRequest) -> dict[str, Any]:
    """Veröffentlicht eine KEI-Bus-Nachricht (Test/Diagnostics)."""
    try:
        service = get_bus_service()
        await service.initialize()
        env = BusEnvelope(
            type=req.type,
            subject=req.subject,
            tenant=req.tenant,
            key=req.key,
            payload=req.payload,
            headers=req.headers,
        )
        await service.publish(env)
        return {"status": "ok", "message_id": env.id}
    except PermissionError as sec_err:
        raise HTTPException(status_code=403, detail=str(sec_err))
    except Exception as exc:
        logger.exception(f"Publish fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="publish_failed")


class SubscribeRequest(BaseModel):
    """Payload für dynamische Subscription (nur Demo)."""

    subject: str
    queue: str | None = None
    durable: str | None = None


@router.post("/subscribe")
async def subscribe_subject(req: SubscribeRequest) -> dict[str, Any]:
    """Richtet eine einfache Subscription ein und loggt eingehende Nachrichten."""
    try:
        service = get_bus_service()
        await service.initialize()

        async def _handler(env: BusEnvelope) -> None:
            logger.info(f"📨 Received on {env.subject}: {env.type} {env.id}")

        await service.subscribe(req.subject, req.queue, _handler, durable=req.durable)
        return {"status": "subscribed", "subject": req.subject, "queue": req.queue}
    except PermissionError as sec_err:
        raise HTTPException(status_code=403, detail=str(sec_err))
    except Exception as exc:
        logger.exception(f"Subscription fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="subscribe_failed")


@router.get("/status")
async def bus_status() -> dict[str, Any]:
    """Einfacher Status des Bus-Services."""
    try:
        get_bus_service()
        return {"initialized": True}
    except Exception:
        return {"initialized": False}
