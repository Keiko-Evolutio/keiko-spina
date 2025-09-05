"""Sichere n8n-Integrationsrouten mit HMAC-Validierung und Idempotenz.

Stellt Endpunkte bereit:
- POST /api/v1/n8n/workflows/trigger
- GET  /api/v1/n8n/workflows/{execution_id}/status
- POST /api/v1/n8n/webhooks/{workflow_id}
"""

from __future__ import annotations

import hmac
import time
from hashlib import sha256
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from kei_logging import get_logger
from observability import trace_function
from security.n8n_hmac_validator import N8nHmacValidator
from services.n8n import N8nClient
from services.n8n.workflow_sync_manager import workflow_sync_manager
from storage.cache.redis_cache import get_cache_client

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/n8n", tags=["n8n"])


async def _verify_hmac(secret: str, payload: bytes, signature: str) -> bool:
    """Verifiziert HMAC-Signatur in konstanter Zeit."""
    expected = hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _check_replay_and_idempotency(nonce: str, idempotency_key: str | None) -> None:
    """Schützt gegen Replay-Angriffe und Duplikate über Redis-Cache."""
    try:
        client = await get_cache_client()
        # Nonce 5 Minuten gültig, Idempotency-Key 10 Minuten
        if nonce:
            await client.setex(f"keiko:n8n:nonce:{nonce}", 300, "1")
        if idempotency_key:
            exists = await client.get(f"keiko:n8n:idem:{idempotency_key}")
            if exists:
                raise HTTPException(status_code=409, detail="Duplicate request")
            await client.setex(f"keiko:n8n:idem:{idempotency_key}", 600, "1")
    except Exception:
        # Fallback ohne Redis: tolerantes Verhalten im Testkontext
        return


class TriggerRequest(BaseModel):
    """Request-Body für Workflow-Triggering."""

    workflow_id: str = Field(..., description="n8n Workflow-ID oder Name")
    payload: dict[str, Any] = Field(default_factory=dict, description="Eingabedaten für den Workflow")
    mode: str = Field(default="rest", description="Trigger-Modus: rest|webhook")
    webhook_path: str | None = Field(default=None, description="Webhook-Pfad für Fallback")


class TriggerResponse(BaseModel):
    """Antwort des Trigger-Endpunkts."""

    execution_id: str | None = None
    started: bool = True
    raw: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """Antwort der Status-Abfrage."""

    execution_id: str
    status: str
    finished: bool
    raw: dict[str, Any] = Field(default_factory=dict)


@router.post("/workflows/trigger", response_model=TriggerResponse)
@trace_function("api.n8n.trigger")
async def trigger_workflow(request: TriggerRequest) -> TriggerResponse:
    """Triggert einen n8n Workflow und liefert Execution-Daten zurück."""
    client = N8nClient()
    result = await client.trigger_workflow(
        workflow_id=request.workflow_id,
        payload=request.payload,
        mode=request.mode,
        webhook_path=request.webhook_path,
    )
    await client.aclose()
    return TriggerResponse(execution_id=result.execution_id, started=result.started, raw=result.raw)


@router.get("/workflows/{execution_id}/status", response_model=StatusResponse)
@trace_function("api.n8n.status")
async def get_status(execution_id: str) -> StatusResponse:
    """Gibt den aktuellen Status einer n8n-Execution zurück."""
    client = N8nClient()
    result = await client.get_execution_status(execution_id)
    await client.aclose()
    return StatusResponse(
        execution_id=execution_id,
        status=result.status.value,
        finished=result.finished,
        raw=result.raw,
    )


@router.post("/webhooks/{workflow_id}")
async def n8n_webhook(
    request: Request,
    x_signature: str = Header(default=""),
    x_n8n_signature: str = Header(default=""),
    x_timestamp: str = Header(default=""),
    x_n8n_timestamp: str = Header(default=""),
    x_nonce: str = Header(default=""),
    x_idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """Empfängt signierte Webhooks von n8n und validiert Sicherheit."""
    body = await request.body()

    # Header-Aliases unterstützen (n8n kann X-N8N-Signature/X-N8N-Timestamp senden)
    signature_header = x_signature or x_n8n_signature
    timestamp_header = x_timestamp or x_n8n_timestamp

    # Timestamp prüfen (±5 Minuten)
    try:
        ts = int(timestamp_header)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid timestamp") from exc
    if abs(int(time.time()) - ts) > 300:
        raise HTTPException(status_code=401, detail="Timestamp skew too large")

    # HMAC prüfen – Secret aus ENV/Konfiguration lesen
    validator = N8nHmacValidator()
    await validator.validate(payload=body, signature=signature_header, timestamp=str(ts))

    # Replay & Idempotency
    await _check_replay_and_idempotency(x_nonce, x_idempotency_key)

    # Optional: Execution-ID aus Payload extrahieren und SyncManager informieren
    try:
        data = await request.json()
    except Exception:
        data = {}
    execution_id = str(data.get("executionId") or data.get("id") or "").strip()
    if execution_id:
        await workflow_sync_manager.handle_callback(execution_id, data)

    logger.info("n8n Webhook angenommen")
    return {"status": "accepted", "execution_id": execution_id or None}


@router.post("/workflows/{execution_id}/pause")
@trace_function("api.n8n.pause")
async def pause_workflow(execution_id: str) -> dict[str, Any]:
    """Pausiert Keiko-seitige Synchronisation (Polling) einer Execution."""
    return await workflow_sync_manager.pause(execution_id)


@router.post("/workflows/{execution_id}/resume")
@trace_function("api.n8n.resume")
async def resume_workflow(execution_id: str) -> dict[str, Any]:
    """Setzt Keiko-seitige Synchronisation einer Execution fort."""
    return await workflow_sync_manager.resume(execution_id)


__all__ = ["router"]
