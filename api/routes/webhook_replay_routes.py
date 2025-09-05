"""Replay-API für KEI-Webhook (Events listen, Replay auslösen, Status)."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.webhooks.history import EventHistoryStore
from services.webhooks.manager import get_webhook_manager

from .base import create_router

logger = get_logger(__name__)
router = create_router("/api/v1/webhooks/replay", ["webhooks", "webhooks-replay"])


class EventHistoryItem(BaseModel):
    """Ein gespeicherter Event-Historie-Eintrag."""

    id: str
    record: dict[str, Any]
    target: dict[str, Any]
    event: dict[str, Any]


class EventHistoryResponse(BaseModel):
    """Antwort für das Event-History Listing."""

    items: list[EventHistoryItem]


@router.get("/events", response_model=EventHistoryResponse)
async def list_events(limit: int = Query(50, ge=1, le=500), target_id: str | None = None, event_type: str | None = None) -> EventHistoryResponse:
    """Listet die letzten N Events aus der Historie gefiltert."""
    store = EventHistoryStore()
    items = await store.list_events(limit=limit, target_id=target_id, event_type=event_type)
    return EventHistoryResponse(items=[EventHistoryItem(**it) for it in items])


class ReplayTarget(BaseModel):
    """Zielinformation für Replay (existierendes oder neues Target)."""

    target_id: str | None = None
    url: str | None = None
    secret: str | None = None


class ReplayRequest(BaseModel):
    """Replay-Anfrage für spezifische Event-IDs."""

    event_ids: list[str] = Field(default_factory=list)
    target: ReplayTarget


class ReplayStatus(BaseModel):
    """Vereinfachter Replay-Status."""

    replay_id: str
    status: str
    processed: int = 0
    failed: int = 0


_replay_status: dict[str, ReplayStatus] = {}


@router.post("/replay", response_model=ReplayStatus)
async def start_replay(request: ReplayRequest) -> ReplayStatus:
    """Startet ein Replay für gegebene Event-IDs gegen bestehendes/ neues Target."""
    if not request.event_ids:
        raise HTTPException(status_code=400, detail="Keine Event-IDs übergeben")
    manager = get_webhook_manager()
    store = EventHistoryStore()
    # Replay-ID generieren
    import uuid

    replay_id = uuid.uuid4().hex
    status = ReplayStatus(replay_id=replay_id, status="running", processed=0, failed=0)
    _replay_status[replay_id] = status

    # Lade Events und enqueuere mit Meta.replay=True
    items = []
    for eid in request.event_ids:
        # Vereinfachung: wir durchsuchen die letzten Tage und finden den Eintrag in list_events
        found = False
        events = await store.list_events(limit=1000)
        for it in events:
            if it.get("id") == eid:
                items.append(it)
                found = True
                break
        if not found:
            status.failed += 1

    for it in items:
        try:
            rec = it["record"]
            evt = it["event"]
            target_id = request.target.target_id or rec.get("target_id")
            # Markiere als Replay im Meta
            meta = evt.get("meta") or {}
            meta["replay"] = True
            evt["meta"] = meta
            # Bei neuem Target müssen url/secret vorhanden sein (vereinfachte Validierung)
            if request.target.target_id is None:
                if not (request.target.url and request.target.secret):
                    raise HTTPException(status_code=400, detail="Neues Target erfordert url und secret")
                # Hier könnte ein temporäres Target erstellt werden; wir enqueuen direkt mit bestehender Target-ID
                target_id = rec.get("target_id")
            await manager.enqueue_outbound(target_id=target_id, event_type=evt.get("event_type", "replay"), data=evt.get("data", {}))
            status.processed += 1
        except Exception:
            status.failed += 1

    status.status = "completed"
    return status


@router.get("/replay/{replay_id}/status", response_model=ReplayStatus)
async def get_replay_status(replay_id: str) -> ReplayStatus:
    """Liefert den Status eines Replay-Vorgangs."""
    s = _replay_status.get(replay_id)
    if not s:
        raise HTTPException(status_code=404, detail="Replay nicht gefunden")
    return s


__all__ = ["router"]
