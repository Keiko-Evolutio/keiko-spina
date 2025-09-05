"""KEI‑Webhook Admin Routen (DLQ/Outbox/Worker/Health)."""

from __future__ import annotations

import contextlib
from datetime import UTC
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from api.middleware.scope_middleware import require_scopes
from kei_logging import get_logger
from services.webhooks import get_webhook_manager
from services.webhooks.audit_logger import WebhookAuditEventType, webhook_audit
from services.webhooks.keys import dlq_key, outbox_key

# Service-Imports
try:
    from services.webhooks.audit_logger import get_webhook_audit_logger
    from services.webhooks.manager import get_webhook_manager as get_webhook_manager_direct
    from services.webhooks.models import WebhookTarget
    WEBHOOK_SERVICES_AVAILABLE = True
except ImportError:
    get_webhook_manager_direct = None
    WebhookTarget = None
    get_webhook_audit_logger = None
    WEBHOOK_SERVICES_AVAILABLE = False

from .base import create_router
from .webhook_admin_roles_routes import router as roles_router

logger = get_logger(__name__)

router = create_router("/api/v1/webhooks", ["webhooks-admin"])
try:
    # Rolle‑Admin Routen unter gleichem Prefix einhängen
    router.include_router(roles_router)
except (ImportError, AttributeError) as e:
    logger.debug(f"Webhook-Admin-Roles-Router konnte nicht eingehängt werden: {e}")
except Exception as e:
    logger.warning(f"Unerwarteter Fehler beim Einhängen des Webhook-Admin-Roles-Routers: {e}")


class DlqItem(BaseModel):
    """DLQ‑Elementrepräsentation."""

    record: dict[str, Any]
    target: dict[str, Any]
    event: dict[str, Any]


class DlqListResponse(BaseModel):
    """Antwort für DLQ‑Listing."""

    count: int
    items: list[DlqItem]


@router.get("/admin/dlq", response_model=DlqListResponse)
async def list_dlq(limit: int = 10) -> DlqListResponse:
    """Listet die ersten N Einträge der Webhook‑DLQ (Redis‑basiert)."""
    import json as _json

    from storage.cache.redis_cache import NoOpCache, get_cache_client

    try:
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return DlqListResponse(count=0, items=[])
        tenant_id = None  # Admin‑View: optionaler Header könnte später ergänzt werden
        total = await client.llen(dlq_key(tenant_id))  # type: ignore[attr-defined]
        raw_items = await client.lrange(dlq_key(tenant_id), 0, max(0, limit - 1))  # type: ignore[attr-defined]
        items: list[DlqItem] = []
        for raw in raw_items or []:
            try:
                items.append(DlqItem(**_json.loads(raw)))
            except Exception:
                continue
        return DlqListResponse(count=int(total or 0), items=items)
    except Exception:
        return DlqListResponse(count=0, items=[])


class DlqRequeueRequest(BaseModel):
    """Requeue‑Anfrage für DLQ."""

    count: int = 1


@router.post("/admin/dlq/requeue")
async def requeue_dlq(request: DlqRequeueRequest) -> dict[str, Any]:
    """Stellt bis zu N DLQ‑Nachrichten zurück in die Outbox."""
    from storage.cache.redis_cache import NoOpCache, get_cache_client

    try:
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            raise HTTPException(status_code=503, detail="DLQ nicht verfügbar")
        manager = get_webhook_manager()
        # Ermittele Shards aus aktueller Pool‑Konfiguration
        pool = manager.get_workers_status().get("pool", {})
        base = pool.get("queue_base", "default")
        configured = int(pool.get("configured", 1) or 1)
        shards = [base] if configured <= 1 else [f"{base}:{i}" for i in range(configured)]
        moved = 0
        for _ in range(max(0, request.count)):
            data = await client.rpop(dlq_key(None))
            if not data:
                break
            # Round‑Robin Requeue über Shards
            shard = shards[moved % len(shards)]
            await client.lpush(outbox_key(None, shard), data)
            moved += 1
        with contextlib.suppress(Exception):
            await webhook_audit.dlq_event(
                event=WebhookAuditEventType.DLQ_REQUEUE,
                delivery_id="bulk",
                target_id="bulk",
                tenant_id=None,
                correlation_id=None,
                details={"count": moved},
            )
        return {"status": "ok", "requeued": moved}
    except HTTPException:
        raise
    except (ConnectionError, TimeoutError) as exc:
        logger.error(f"DLQ Requeue Fehler - Verbindungsproblem: {exc}")
        raise HTTPException(status_code=503, detail="requeue_service_unavailable")
    except (ValueError, TypeError) as exc:
        logger.error(f"DLQ Requeue Fehler - Validierungsfehler: {exc}")
        raise HTTPException(status_code=400, detail="requeue_validation_failed")
    except Exception as exc:
        logger.exception(f"DLQ Requeue Fehler - Unerwarteter Fehler: {exc}")
        raise HTTPException(status_code=500, detail="requeue_failed")


class OutboxDepthResponse(BaseModel):
    """Antwortmodell für Outbox‑Tiefe."""

    timestamp: str
    queue: str
    depth: int
    status: str


@router.get("/admin/outbox/depth", response_model=OutboxDepthResponse)
async def get_outbox_depth() -> OutboxDepthResponse:
    """Gibt die aktuelle Tiefe der Outbox‑Queue zurück."""
    from datetime import datetime

    manager = get_webhook_manager()
    depth = await manager.get_outbox_depth()
    status = "ok" if depth >= 0 else "unknown"
    pool = manager.get_workers_status().get("pool", {})
    queue_base = pool.get("queue_base", "default")
    return OutboxDepthResponse(
        timestamp=datetime.now(UTC).isoformat(),
        queue=str(queue_base),
        depth=depth,
        status=status,
    )


class WorkerStatus(BaseModel):
    """Status eines einzelnen Workers."""

    queue_name: str
    active: bool
    processed: int
    failed: int
    last_activity: str
    uptime_seconds: int
    rate_per_minute: float
    poll_interval: float


class WorkersStatusResponse(BaseModel):
    """Antwortmodell für Worker‑Status."""

    workers: list[WorkerStatus]


@router.get("/admin/workers/status", response_model=WorkersStatusResponse)
async def get_workers_status() -> WorkersStatusResponse:
    """Gibt den Status aller aktiven Worker zurück."""
    manager = get_webhook_manager()
    raw = manager.get_workers_status()
    workers = [WorkerStatus(**w) for w in raw.get("workers", [])]
    return WorkersStatusResponse(workers=workers)


class BatchTargetsOp(BaseModel):
    """Batch‑Operation für Targets (create/update/delete)."""

    op: str
    data: dict[str, Any]


class BatchTargetsRequest(BaseModel):
    """Request für Batch‑Operationen auf Targets."""

    operations: list[BatchTargetsOp]


class BatchTargetsResponse(BaseModel):
    """Antwort auf Batch‑Operationen mit Erfolgs/Fehlerzählern."""

    success: int
    failed: int
    errors: list[str] = Field(default_factory=list)


@router.post("/targets/batch", response_model=BatchTargetsResponse)
async def batch_targets(payload: BatchTargetsRequest, request=None) -> BatchTargetsResponse:
    """Führt Bulk‑Operationen für Targets aus (create/update/delete)."""
    if request is not None:
        require_scopes(request, ["webhook:targets:manage"])  # type: ignore[name-defined]
    if not WEBHOOK_SERVICES_AVAILABLE:
        raise ImportError("Webhook services not available")
    manager = get_webhook_manager_direct()
    tenant_id = None
    try:
        if request is not None:
            tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
    except Exception:
        tenant_id = None
    registry = manager.get_targets_registry(tenant_id)

    applied: list[dict[str, Any]] = []
    success = 0
    errors: list[str] = []

    async def _rollback() -> None:
        for step in reversed(applied):
            try:
                if step["op"] == "create":
                    await registry.delete(step["id"])  # type: ignore[arg-type]
                elif step["op"] == "update" and step.get("before") is not None:
                    await registry.upsert(step["before"])  # type: ignore[arg-type]
            except (ConnectionError, TimeoutError) as e:
                logger.debug(f"Rollback-Operation fehlgeschlagen - Verbindungsproblem: {e}")
            except Exception as e:
                logger.warning(f"Rollback-Operation fehlgeschlagen - Unerwarteter Fehler: {e}")

    if not WEBHOOK_SERVICES_AVAILABLE:
        raise ImportError("Webhook services not available")

    for op in payload.operations:
        try:
            if op.op not in {"create", "update", "delete"}:
                raise ValueError("Ungültige Operation")
            if op.op in {"create", "update"}:
                target = WebhookTarget(**op.data)
                before = await registry.get(target.id)
                await registry.upsert(target)
                applied.append({"op": "create" if before is None else "update", "id": target.id, "before": before})
            else:
                tid = op.data.get("id")
                before = await registry.get(tid)
                await registry.delete(tid)
                applied.append({"op": "delete", "id": tid, "before": before})
            success += 1
        except (ValueError, TypeError) as exc:
            errors.append(f"Validierungsfehler: {exc}")
            await _rollback()
            break
        except (ConnectionError, TimeoutError) as exc:
            errors.append(f"Verbindungsproblem: {exc}")
            await _rollback()
            break
        except Exception as exc:
            errors.append(f"Unerwarteter Fehler: {exc}")
            await _rollback()
            break

    return BatchTargetsResponse(success=success, failed=len(payload.operations) - success, errors=errors)


class DLQBatchRetryRequest(BaseModel):
    """Request für Batch‑Retry (IDs oder Filter)."""

    delivery_ids: list[str] | None = None
    target_id: str | None = None
    event_type: str | None = None
    status: str | None = None
    reset_attempt: bool = False
    rate_limit_per_sec: float = Field(default=0.0, ge=0.0, le=50.0)


class DLQBatchPurgeRequest(BaseModel):
    """Request für Batch‑Purge per Filter."""

    target_id: str | None = None
    event_type: str | None = None
    status: str | None = None


class DLQBatchOpResponse(BaseModel):
    """Antwortmodell für DLQ Batch‑Operationen."""

    affected: int


class AuditVerifyRequest(BaseModel):
    """Requestmodell für Audit‑Log Verifikation."""

    payload: str


class AuditVerifyResponse(BaseModel):
    """Antwort auf Audit‑Verifikation."""

    valid: bool


@router.post("/admin/audit/verify", response_model=AuditVerifyResponse)
async def audit_verify(request: AuditVerifyRequest) -> AuditVerifyResponse:
    """Verifiziert eine Audit‑Log Zeile auf Integrität."""
    if not WEBHOOK_SERVICES_AVAILABLE:
        raise ImportError("Webhook services not available")

    lg = get_webhook_audit_logger()
    try:
        ok = lg.verify_payload(request.payload)
    except Exception:
        ok = False
    return AuditVerifyResponse(valid=ok)


@router.post("/dlq/batch-retry", response_model=DLQBatchOpResponse)
async def dlq_batch_retry(payload: DLQBatchRetryRequest) -> DLQBatchOpResponse:
    """Startet DLQ Batch‑Retry (IDs oder Filter)."""
    if not WEBHOOK_SERVICES_AVAILABLE:
        raise ImportError("Webhook services not available")
    manager = get_webhook_manager_direct()
    filters = {"target_id": payload.target_id, "event_type": payload.event_type, "status": payload.status}
    result = await manager.dlq_bulk_retry(
        delivery_ids=payload.delivery_ids,
        filters=filters,
        reset_attempt=payload.reset_attempt,
        rate_limit_per_sec=payload.rate_limit_per_sec,
    )
    return DLQBatchOpResponse(affected=int(result.get("retried", 0)))


@router.delete("/dlq/batch-purge", response_model=DLQBatchOpResponse)
async def dlq_batch_purge(target_id: str | None = None, event_type: str | None = None, status: str | None = None) -> DLQBatchOpResponse:
    """Löscht DLQ‑Einträge nach Filterkriterien in Batch."""
    if not WEBHOOK_SERVICES_AVAILABLE:
        raise ImportError("Webhook services not available")
    manager = get_webhook_manager_direct()
    filters = {"target_id": target_id, "event_type": event_type, "status": status}
    count = await manager.dlq_purge(filters=filters)  # type: ignore[arg-type]
    return DLQBatchOpResponse(affected=int(count or 0))


__all__ = ["router"]
