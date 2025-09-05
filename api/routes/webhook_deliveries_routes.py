"""Delivery-Tracking API Routen für KEI-Webhook."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel

from kei_logging import get_logger
from services.webhooks.tracking import DeliveryTracker

from .base import create_router

logger = get_logger(__name__)
router = create_router("/webhooks", ["webhooks", "webhooks-deliveries"])


class DeliveryRecordResponse(BaseModel):
    """Vollständiger Delivery-Eintrag."""

    record: dict[str, Any]
    target: dict[str, Any]
    event: dict[str, Any]


class DeliveryListResponse(BaseModel):
    """Paginierte Liste von Deliveries."""

    items: list[DeliveryRecordResponse]
    total: int
    page: int
    limit: int


@router.get("/deliveries", response_model=DeliveryListResponse)
async def deliveries_list(
    request: Request,
    target_id: str | None = None,
    event_type: str | None = None,
    status: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
) -> DeliveryListResponse:
    """Listet Deliveries mit Filtern und Pagination."""
    tracker = DeliveryTracker()
    result = await tracker.list_deliveries(
        tenant_id=request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant"),
        page=page,
        limit=limit,
        target_id=target_id,
        event_type=event_type,
        status=status,
        from_date=from_date,
        to_date=to_date,
    )
    items = [DeliveryRecordResponse(**it) for it in result.get("items", [])]
    return DeliveryListResponse(items=items, total=result.get("total", 0), page=result.get("page", page), limit=result.get("limit", limit))


@router.get("/deliveries/{delivery_id}", response_model=DeliveryRecordResponse)
async def delivery_detail(request: Request, delivery_id: str) -> DeliveryRecordResponse:
    """Gibt Details einer einzelnen Delivery zurück."""
    tracker = DeliveryTracker()
    data = await tracker.get_detail(tenant_id=request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant"), delivery_id=delivery_id)
    if not data:
        raise HTTPException(status_code=404, detail="Delivery nicht gefunden")
    return DeliveryRecordResponse(**data)


class DeliveryRetryResponse(BaseModel):
    """Antwort für Retry einer Delivery."""

    new_delivery_id: str


@router.post("/deliveries/{delivery_id}/retry", response_model=DeliveryRetryResponse)
async def delivery_retry(request: Request, delivery_id: str) -> DeliveryRetryResponse:
    """Startet einen erneuten Zustellversuch (nur für failed/dlq)."""
    tracker = DeliveryTracker()
    new_id = await tracker.retry(tenant_id=request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant"), delivery_id=delivery_id)
    if not new_id:
        raise HTTPException(status_code=400, detail="Retry nicht möglich")
    return DeliveryRetryResponse(new_delivery_id=new_id)


class DeliveryCancelResponse(BaseModel):
    """Antwort für Cancellation einer Delivery."""

    cancelled: bool


@router.delete("/deliveries/{delivery_id}", response_model=DeliveryCancelResponse)
async def delivery_cancel(request: Request, delivery_id: str) -> DeliveryCancelResponse:
    """Bricht eine ausstehende/pending Delivery ab."""
    tracker = DeliveryTracker()
    ok = await tracker.cancel(tenant_id=request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant"), delivery_id=delivery_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Delivery nicht gefunden oder nicht pending")
    return DeliveryCancelResponse(cancelled=True)


class DeliveryStatsResponse(BaseModel):
    """Aggregierte Delivery-Statistiken."""

    total: int
    success_rate: float
    avg_latency_ms: float
    retry_rate: float


@router.get("/deliveries/stats", response_model=DeliveryStatsResponse)
async def deliveries_stats(request: Request) -> DeliveryStatsResponse:
    """Gibt aggregierte Metriken über Deliveries zurück."""
    tracker = DeliveryTracker()
    stats = await tracker.stats(tenant_id=request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant"))
    return DeliveryStatsResponse(**stats)


__all__ = ["router"]
