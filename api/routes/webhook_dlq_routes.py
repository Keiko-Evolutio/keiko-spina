"""DLQ Management Routen für KEI‑Webhook.

Bietet Auflistung, Retry (einzeln/bulk) und Lösch‑Operationen.
"""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.middleware.scope_middleware import require_scopes
from kei_logging import get_logger
from services.webhooks.audit_logger import WebhookAuditEventType, webhook_audit
from services.webhooks.manager import get_webhook_manager

from .base import create_router

logger = get_logger(__name__)
router = create_router("/api/v1/webhooks", ["webhooks", "webhooks-dlq"])


class DLQListQuery(BaseModel):
    """Query‑Parameter für DLQ‑Listing."""

    target_id: str | None = None
    event_type: str | None = None
    status: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sort: str | None = Field(default="created_at")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)


class DLQItem(BaseModel):
    """Ein DLQ‑Eintrag (vereinfachte Sicht)."""

    record: dict[str, Any]
    target: dict[str, Any]
    event: dict[str, Any]


class DLQListResponse(BaseModel):
    """Antwort für DLQ‑Listing."""

    items: list[DLQItem]
    total: int


@router.get("/dlq", response_model=DLQListResponse)
async def dlq_list(
    request: Request,
    target_id: str | None = None,
    event_type: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort: str | None = Query(default="created_at"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> DLQListResponse:
    """Listet DLQ‑Einträge mit Pagination/Filter/Sortierung."""
    require_scopes(request, ["webhook:dlq:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    filters: dict[str, Any] = {
        "target_id": target_id,
        "event_type": event_type,
        "status": status,
        "date_range": {"from": date_from, "to": date_to} if (date_from or date_to) else None,
    }
    result = await manager.dlq_list(limit=limit, offset=offset, filters=filters, sort=sort)
    items = [DLQItem(**it) for it in result.get("items", [])]
    return DLQListResponse(items=items, total=result.get("total", 0))


class RetryResponse(BaseModel):
    """Antwort für Einzel‑Retry."""

    success: bool


@router.post("/dlq/{delivery_id}/retry", response_model=RetryResponse)
async def dlq_retry_one(request: Request, delivery_id: str, reset_attempt: bool = Query(default=False)) -> RetryResponse:
    """Verschiebt einen DLQ‑Eintrag zurück in die Outbox."""
    require_scopes(request, ["webhook:dlq:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    ok = await manager.dlq_retry_one(delivery_id=delivery_id, reset_attempt=reset_attempt)
    if not ok:
        raise HTTPException(status_code=404, detail="DLQ Eintrag nicht gefunden")
    with contextlib.suppress(Exception):
        await webhook_audit.dlq_event(
            event=WebhookAuditEventType.DLQ_REQUEUE,
            delivery_id=delivery_id,
            target_id="unknown",
            tenant_id=None,
            correlation_id=None,
            details={"reset_attempt": reset_attempt},
        )
    return RetryResponse(success=True)


class BulkRetryRequest(BaseModel):
    """Request für Bulk‑Retry."""

    delivery_ids: list[str] | None = None
    target_id: str | None = None
    event_type: str | None = None
    status: str | None = None
    rate_limit_per_sec: float = Field(default=0.0, ge=0.0, le=50.0)
    reset_attempt: bool = False


class BulkRetryResponse(BaseModel):
    """Antwort für Bulk‑Retry."""

    requested: int
    retried: int
    failed: int


@router.post("/dlq/bulk-retry", response_model=BulkRetryResponse)
async def dlq_bulk_retry(req: Request, request: BulkRetryRequest) -> BulkRetryResponse:
    """Startet eine Bulk‑Retry Operation (IDs oder Filter)."""
    require_scopes(req, ["webhook:dlq:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    filters = {
        "target_id": request.target_id,
        "event_type": request.event_type,
        "status": request.status,
    }
    result = await manager.dlq_bulk_retry(
        delivery_ids=request.delivery_ids,
        filters=filters,
        reset_attempt=request.reset_attempt,
        rate_limit_per_sec=request.rate_limit_per_sec,
    )
    return BulkRetryResponse(**result)


class DeleteResponse(BaseModel):
    """Antwort für Lösch‑Operationen."""

    deleted: bool


@router.delete("/dlq/{delivery_id}", response_model=DeleteResponse)
async def dlq_delete_one(request: Request, delivery_id: str) -> DeleteResponse:
    """Löscht einen DLQ‑Eintrag endgültig."""
    require_scopes(request, ["webhook:dlq:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    ok = await manager.dlq_delete_one(delivery_id=delivery_id)
    if not ok:
        raise HTTPException(status_code=404, detail="DLQ Eintrag nicht gefunden")
    with contextlib.suppress(Exception):
        await webhook_audit.dlq_event(
            event=WebhookAuditEventType.DLQ_PURGE,
            delivery_id=delivery_id,
            target_id="unknown",
            tenant_id=None,
            correlation_id=None,
        )
    return DeleteResponse(deleted=True)


class PurgeResponse(BaseModel):
    """Antwort für Purge‑Operation."""

    purged: int


@router.delete("/dlq/purge", response_model=PurgeResponse)
async def dlq_purge(request: Request) -> PurgeResponse:
    """Leert die DLQ vollständig (nur Admin/mit Vorsicht!)."""
    require_scopes(request, ["webhook:dlq:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    count = await manager.dlq_purge()
    with contextlib.suppress(Exception):
        await webhook_audit.dlq_event(
            event=WebhookAuditEventType.DLQ_PURGE,
            delivery_id="bulk",
            target_id="bulk",
            tenant_id=None,
            correlation_id=None,
            details={"purged": count},
        )
    return PurgeResponse(purged=count)


__all__ = ["router"]
