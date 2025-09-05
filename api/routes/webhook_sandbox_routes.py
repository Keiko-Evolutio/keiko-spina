"""Webhook Sandbox/Replay API.

Stellt Endpunkte für Sandbox-Inbound/Outbound, Listing und Replay bereit.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.messaging import BusEnvelope, get_messaging_service
from services.messaging.naming import subject_for_event
from services.webhooks.manager import get_webhook_manager
from services.webhooks.models import WebhookEventMeta
from storage.cache.redis_cache import NoOpCache, get_cache_client

from .base import create_router

logger = get_logger(__name__)

router = create_router("/api/v1/webhooks", ["webhooks-sandbox"])

# Redis Keys
KEY_SB_INBOUND = "kei:webhook:sandbox:inbound"
KEY_SB_OUTBOUND = "kei:webhook:sandbox:outbound"
KEY_SB_TARGETCALLS = "kei:webhook:sandbox:targetcalls"
MAX_ARCHIVE = 1000

# Öffentlicher Sandbox‑Test‑Key für HMAC (NICHT produktiv einsetzen)
SANDBOX_HMAC_SECRET = "kei-sandbox-secret"


class SandboxInboundItem(BaseModel):
    """Archiviertes Sandbox-Inbound-Event."""

    id: str = Field(..., description="Eindeutige ID")
    topic: str
    payload: dict[str, Any]
    headers: dict[str, Any] = Field(default_factory=dict)
    received_at: str


class SandboxOutboundItem(BaseModel):
    """Archiviertes Sandbox-Outbound-Event."""

    id: str
    target_id: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    meta: WebhookEventMeta | None = None
    enqueued_at: str


class ReplayRequest(BaseModel):
    """Replay-Request für Sandbox-Archive."""

    limit: int = Field(10, ge=1, le=200)


async def _archive_push(key: str, value: dict[str, Any]) -> None:
    """Speichert einen Eintrag im Redis-Archiv und trimmt die Liste."""
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return
    try:
        await client.lpush(key, json.dumps(value))
        await client.ltrim(key, 0, MAX_ARCHIVE - 1)  # type: ignore[attr-defined]
        try:
            # 24h Ablaufzeit pro Archivliste setzen (automatische Bereinigung)
            await client.expire(key, 24 * 3600)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - Best Effort
        logger.debug(f"Sandbox Archiv-Fehler: {exc}")


async def _archive_list(key: str, limit: int) -> list[dict[str, Any]]:
    """Liest die letzten N Einträge aus dem Archiv."""
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return []
    try:
        raw = await client.lrange(key, 0, max(0, limit - 1))  # type: ignore[attr-defined]
        out: list[dict[str, Any]] = []
        for item in raw or []:
            try:
                out.append(json.loads(item))
            except Exception:
                continue
        return out
    except Exception as exc:
        logger.debug(f"Sandbox Archiv-Read Fehler: {exc}")
        return []


@router.post("/sandbox/inbound/{topic}")
async def sandbox_inbound(topic: str, request: Request) -> dict[str, Any]:
    """Sandbox-Inbound: akzeptiert Payload ohne HMAC, publiziert auf KEI-Bus und archiviert."""
    try:
        body = await request.body()
        try:
            payload_json: dict[str, Any] = await request.json()
        except (ValueError, TypeError) as e:
            logger.debug(f"JSON-Parsing fehlgeschlagen - Format-/Typ-Fehler: {e}")
            payload_json = {"raw": body.decode("utf-8", errors="ignore")}
        except Exception as e:
            logger.warning(f"JSON-Parsing fehlgeschlagen - Unerwarteter Fehler: {e}")
            payload_json = {"raw": body.decode("utf-8", errors="ignore")}

        tenant = request.headers.get("x-tenant") or None
        subject = subject_for_event(tenant=tenant or "public", bounded_context="webhook", aggregate="inbound", event=topic, version=1)
        env = BusEnvelope(
            type=f"webhook.sandbox.inbound.{topic}",
            subject=subject,
            tenant=tenant,
            payload={"topic": topic, "data": payload_json, "sandbox": True},
            headers={"x-source": "webhook-sandbox"},
        )
        try:
            bus = get_messaging_service()
            await bus.initialize()
            await bus.publish(env)
        except (ConnectionError, TimeoutError) as exc:
            logger.debug(f"Sandbox Inbound Bus Publish fehlgeschlagen - Verbindungsproblem: {exc}")
        except Exception as exc:
            logger.debug(f"Sandbox Inbound Bus Publish fehlgeschlagen - Unerwarteter Fehler: {exc}")

        item = SandboxInboundItem(
            id=env.id,
            topic=topic,
            payload=payload_json,
            headers={k.lower(): v for k, v in (request.headers or {}).items()},
            received_at=datetime.now(UTC).isoformat(),
        ).model_dump(mode="json")
        await _archive_push(KEY_SB_INBOUND, item)

        return {"status": "accepted", "id": env.id}
    except HTTPException:
        raise
    except (ConnectionError, TimeoutError) as exc:
        logger.error(f"Sandbox Inbound Fehler - Verbindungsproblem: {exc}")
        raise HTTPException(status_code=503, detail="sandbox_inbound_service_unavailable")
    except (ValueError, TypeError) as exc:
        logger.error(f"Sandbox Inbound Fehler - Validierungsfehler: {exc}")
        raise HTTPException(status_code=400, detail="sandbox_inbound_validation_failed")
    except Exception as exc:
        logger.exception(f"Sandbox Inbound Fehler - Unerwarteter Fehler: {exc}")
        raise HTTPException(status_code=500, detail="sandbox_inbound_failed")


@router.get("/sandbox/inbound/recent", response_model=list[SandboxInboundItem])
async def sandbox_inbound_recent(limit: int = Query(20, ge=1, le=200)) -> list[SandboxInboundItem]:
    """Listet die letzten N Sandbox-Inbound-Events."""
    items = await _archive_list(KEY_SB_INBOUND, limit)
    return [SandboxInboundItem(**it) for it in items]


class SandboxOutboundEnqueueRequest(BaseModel):
    """Anfrage zum Enqueue eines Sandbox-Outbound-Events."""

    target_id: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    meta: WebhookEventMeta | None = None


class SandboxEnqueueResponse(BaseModel):
    """Antwort mit IDs für Nachverfolgung."""

    delivery_id: str
    event_id: str


@router.post("/sandbox/outbound/enqueue", response_model=SandboxEnqueueResponse)
async def sandbox_outbound_enqueue(request: SandboxOutboundEnqueueRequest) -> SandboxEnqueueResponse:
    """Enqueued ein Sandbox-Outbound-Event in die Outbox und archiviert den Auftrag."""
    try:
        manager = get_webhook_manager()
        delivery_id, event_id = await manager.enqueue_outbound(
            target_id=request.target_id,
            event_type=request.event_type,
            data=request.data,
            meta=request.meta,
        )
        item = SandboxOutboundItem(
            id=str(uuid.uuid4().hex),
            target_id=request.target_id,
            event_type=request.event_type,
            data=request.data,
            meta=request.meta,
            enqueued_at=datetime.now(UTC).isoformat(),
        ).model_dump(mode="json")
        await _archive_push(KEY_SB_OUTBOUND, item)
        return SandboxEnqueueResponse(delivery_id=delivery_id, event_id=event_id)
    except HTTPException:
        raise
    except (ConnectionError, TimeoutError) as exc:
        logger.error(f"Sandbox Outbound Enqueue Fehler - Verbindungsproblem: {exc}")
        raise HTTPException(status_code=503, detail="sandbox_outbound_service_unavailable")
    except (ValueError, TypeError) as exc:
        logger.error(f"Sandbox Outbound Enqueue Fehler - Validierungsfehler: {exc}")
        raise HTTPException(status_code=400, detail="sandbox_outbound_validation_failed")
    except Exception as exc:
        logger.exception(f"Sandbox Outbound Enqueue Fehler - Unerwarteter Fehler: {exc}")
        raise HTTPException(status_code=500, detail="sandbox_outbound_enqueue_failed")


@router.get("/sandbox/outbound/recent", response_model=list[SandboxOutboundItem])
async def sandbox_outbound_recent(limit: int = Query(20, ge=1, le=200)) -> list[SandboxOutboundItem]:
    """Listet die letzten N Sandbox-Outbound-Events (Auftragsdaten)."""
    items = await _archive_list(KEY_SB_OUTBOUND, limit)
    return [SandboxOutboundItem(**it) for it in items]


@router.post("/sandbox/replay/inbound")
async def sandbox_replay_inbound(request: ReplayRequest) -> dict[str, Any]:
    """Re-publiziert die letzten N Sandbox-Inbound-Events erneut auf den Bus."""
    try:
        items = await _archive_list(KEY_SB_INBOUND, request.limit)
        published = 0
        for it in reversed(items):  # Älteste zuerst
            topic = it.get("topic", "sandbox")
            payload = it.get("payload", {})
            subject = subject_for_event(tenant="public", bounded_context="webhook", aggregate="inbound", event=topic, version=1)
            env = BusEnvelope(
                type=f"webhook.sandbox.inbound.{topic}",
                subject=subject,
                tenant="public",
                payload={"topic": topic, "data": payload, "sandbox": True, "replay": True},
                headers={"x-source": "webhook-sandbox-replay"},
            )
            try:
                bus = get_messaging_service()
                await bus.initialize()
                await bus.publish(env)
                published += 1
            except Exception as exc:
                logger.debug(f"Sandbox Inbound Replay Publish fehlgeschlagen: {exc}")
        return {"status": "ok", "published": published}
    except Exception as exc:
        logger.exception(f"Sandbox Inbound Replay Fehler: {exc}")
        raise HTTPException(status_code=500, detail="sandbox_replay_inbound_failed")


@router.post("/sandbox/replay/outbound")
async def sandbox_replay_outbound(request: ReplayRequest) -> dict[str, Any]:
    """Re-enqueued die letzten N Sandbox-Outbound-Events in die Outbox."""
    try:
        items = await _archive_list(KEY_SB_OUTBOUND, request.limit)
        manager = get_webhook_manager()
        enqueued = 0
        for it in reversed(items):  # Älteste zuerst
            target_id = it.get("target_id")
            event_type = it.get("event_type")
            data = it.get("data", {})
            meta = it.get("meta")
            try:
                await manager.enqueue_outbound(target_id=target_id, event_type=event_type, data=data, meta=meta)
                enqueued += 1
            except Exception as exc:
                logger.debug(f"Sandbox Outbound Replay Enqueue fehlgeschlagen: {exc}")
        return {"status": "ok", "enqueued": enqueued}
    except Exception as exc:
        logger.exception(f"Sandbox Outbound Replay Fehler: {exc}")
        raise HTTPException(status_code=500, detail="sandbox_replay_outbound_failed")


# ---------------------------------------------------------------------------
# HMAC Test‑Utilities
# ---------------------------------------------------------------------------

class SandboxKeyResponse(BaseModel):
    """Antwort mit öffentlichem Sandbox‑HMAC‑Key."""

    hmac_secret: str


@router.get("/sandbox/keys", response_model=SandboxKeyResponse)
async def sandbox_keys() -> SandboxKeyResponse:
    """Gibt öffentlichen Sandbox‑HMAC‑Key zurück (nur für Tests)."""
    return SandboxKeyResponse(hmac_secret=SANDBOX_HMAC_SECRET)


class SignatureValidateRequest(BaseModel):
    """Request zur HMAC‑Signaturvalidierung."""

    payload: str = Field(..., description="Roh‑Payload als String")
    signature_hex: str = Field(..., description="Hex‑Signatur")
    secret: str | None = Field(default=None, description="Optionaler Secret‑Override")


class SignatureValidateResponse(BaseModel):
    """Antwort auf HMAC‑Signaturprüfung."""

    valid: bool


@router.post("/sandbox/validate-signature", response_model=SignatureValidateResponse)
async def validate_signature(request: SignatureValidateRequest) -> SignatureValidateResponse:
    """Validiert eine HMAC‑SHA256 Signatur gegen Sandbox‑Secret oder Override."""
    try:
        import hmac
        from hashlib import sha256
        secret = (request.secret or SANDBOX_HMAC_SECRET).encode("utf-8")
        calc = hmac.new(secret, request.payload.encode("utf-8"), sha256).hexdigest()
        return SignatureValidateResponse(valid=hmac.compare_digest(calc, request.signature_hex))
    except Exception:
        return SignatureValidateResponse(valid=False)


# ---------------------------------------------------------------------------
# Mock Targets (simulieren HTTP‑Antworten 200/400/500/Timeout)
# ---------------------------------------------------------------------------

class TargetCallItem(BaseModel):
    """Archivierter Aufruf an einen Mock‑Target Endpoint."""

    id: str
    path: str
    method: str
    status: int
    received_at: str
    payload: dict[str, Any] | None = None


async def _archive_target_call(path: str, method: str, status: int, payload: dict[str, Any] | None) -> None:
    """Archiviert einen Mock‑Target Aufruf für den Payload‑Inspector."""
    item = TargetCallItem(
        id=str(uuid.uuid4().hex),
        path=path,
        method=method,
        status=status,
        received_at=datetime.now(UTC).isoformat(),
        payload=payload,
    ).model_dump(mode="json")
    await _archive_push(KEY_SB_TARGETCALLS, item)


@router.post("/sandbox/mock/ok")
async def mock_ok(request: Request) -> dict[str, Any]:
    """Simuliert HTTP 200."""
    try:
        body = await request.json()
    except Exception:
        body = None
    await _archive_target_call("/sandbox/mock/ok", "POST", 200, body)
    return {"status": "ok"}


@router.post("/sandbox/mock/bad-request")
async def mock_bad_request(request: Request) -> dict[str, Any]:
    """Simuliert HTTP 400."""
    try:
        body = await request.json()
    except (ValueError, TypeError) as e:
        logger.debug(f"Mock Bad Request JSON-Parsing fehlgeschlagen: {e}")
        body = None
    except Exception as e:
        logger.warning(f"Mock Bad Request unerwarteter Fehler: {e}")
        body = None
    await _archive_target_call("/sandbox/mock/bad-request", "POST", 400, body)
    raise HTTPException(status_code=400, detail="bad_request")


@router.post("/sandbox/mock/server-error")
async def mock_server_error(request: Request) -> dict[str, Any]:
    """Simuliert HTTP 500."""
    try:
        body = await request.json()
    except (ValueError, TypeError) as e:
        logger.debug(f"Mock Server Error JSON-Parsing fehlgeschlagen: {e}")
        body = None
    except Exception as e:
        logger.warning(f"Mock Server Error unerwarteter Fehler: {e}")
        body = None
    await _archive_target_call("/sandbox/mock/server-error", "POST", 500, body)
    raise HTTPException(status_code=500, detail="server_error")


@router.post("/sandbox/mock/timeout")
async def mock_timeout(request: Request) -> dict[str, Any]:
    """Simuliert einen Timeout durch künstliche Verzögerung (2s)."""
    import asyncio

    try:
        body = await request.json()
    except Exception:
        body = None
    await _archive_target_call("/sandbox/mock/timeout", "POST", 599, body)
    await asyncio.sleep(2.0)
    return {"status": "delayed"}


@router.get("/sandbox/mock/recent", response_model=list[TargetCallItem])
async def mock_recent(limit: int = Query(20, ge=1, le=200)) -> list[TargetCallItem]:
    """Listet die letzten N Mock‑Target Aufrufe (Payload‑Inspector)."""
    items = await _archive_list(KEY_SB_TARGETCALLS, limit)
    return [TargetCallItem(**it) for it in items]


# ---------------------------------------------------------------------------
# Event‑Generator und Templates
# ---------------------------------------------------------------------------

class SandboxTemplateItem(BaseModel):
    """Sandbox‑Event‑Template Beschreibung."""

    event_type: str
    example: dict[str, Any]


TEMPLATES: list[SandboxTemplateItem] = [
    SandboxTemplateItem(event_type="task_completed", example={"task_id": "t-123", "status": "done", "duration_s": 3.2}).model_dump(),
    SandboxTemplateItem(event_type="document_uploaded", example={"document_id": "d-456", "mime": "application/pdf"}).model_dump(),
]


@router.get("/sandbox/templates", response_model=list[SandboxTemplateItem])
async def sandbox_templates() -> list[SandboxTemplateItem]:
    """Gibt verfügbare Test‑Templates zurück."""
    return [SandboxTemplateItem(**t) for t in TEMPLATES]


class GenerateEventRequest(BaseModel):
    """Request zum Generieren eines Test‑Events (optional enqueue)."""

    event_type: str | None = None
    data: dict[str, Any] | None = None
    target_id: str | None = Field(default=None, description="Optionales Target zum Enqueue")


class GenerateEventResponse(BaseModel):
    """Antwort mit erzeugtem Event und optionalen IDs."""

    event: dict[str, Any]
    delivery_id: str | None = None
    event_id: str | None = None


@router.post("/sandbox/generate", response_model=GenerateEventResponse)
async def sandbox_generate(request: GenerateEventRequest) -> GenerateEventResponse:
    """Generiert ein Test‑Event; optional wird Outbound enqueued."""
    etype = request.event_type or "task_completed"
    template = next((t for t in TEMPLATES if t.get("event_type") == etype), None)
    data = request.data or (template.get("example") if template else {"example": True})
    event = {
        "id": uuid.uuid4().hex,
        "event_type": etype,
        "occurred_at": datetime.now(UTC).isoformat(),
        "data": data,
        "meta": WebhookEventMeta().model_dump(mode="json"),
    }
    if request.target_id:
        try:
            manager = get_webhook_manager()
            delivery_id, event_id = await manager.enqueue_outbound(
                target_id=request.target_id,
                event_type=etype,
                data=data,
                meta=WebhookEventMeta(),
            )
            return GenerateEventResponse(event=event, delivery_id=delivery_id, event_id=event_id)
        except Exception:
            # Fallback: nur Event zurückgeben
            return GenerateEventResponse(event=event)
    return GenerateEventResponse(event=event)


# ---------------------------------------------------------------------------
# Health für Sandbox
# ---------------------------------------------------------------------------

@router.get("/sandbox/health")
async def sandbox_health() -> dict[str, Any]:
    """Einfacher Health‑Status der Sandbox (Archivgrößen)."""
    inbound = len(await _archive_list(KEY_SB_INBOUND, 100))
    outbound = len(await _archive_list(KEY_SB_OUTBOUND, 100))
    target_calls = len(await _archive_list(KEY_SB_TARGETCALLS, 100))
    return {"status": "ok", "inbound_recent": inbound, "outbound_recent": outbound, "mock_calls_recent": target_calls}


__all__ = ["router"]
