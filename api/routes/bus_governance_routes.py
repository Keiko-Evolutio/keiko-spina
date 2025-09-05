"""Governance-APIs für KEI-Bus: Review-Gates, Topic-Registrierung, Policies."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.messaging.governance import (
    check_slo,
    check_topic_policy,
    review_schema,
    validate_rpc_naming,
    validate_subject_naming,
)
from services.messaging.management_registry import list_topics as reg_list_topics
from services.messaging.management_registry import (
    register_topic,
    set_allowed_keys,
    update_topic_policy,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/bus/governance", tags=["KEI-Bus-Governance"])


class TopicRegistration(BaseModel):
    """Registrierung eines Topics/Subjects."""

    name: str = Field(..., description="Subject/Topic Name")
    retention: str = Field("limits", description="Retention Policy")
    max_delivery: int = Field(5, ge=1, le=100, description="Max Redeliveries")


class PolicyUpdate(BaseModel):
    """Policy-Update für Topics."""

    name: str
    retention: str | None = None
    max_delivery: int | None = Field(None, ge=1, le=100)


_TOPIC_REGISTRY: dict[str, dict[str, Any]] = {}


@router.post("/topics/register")
async def register_topic(req: TopicRegistration):
    """Registriert Topic/Subject in einfacher Registry (In-Memory)."""
    try:
        await register_topic(req.name, retention=req.retention, max_delivery=req.max_delivery)
        return {"status": "ok"}
    except Exception as exc:
        logger.exception(f"Topic-Registrierung fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="topic_register_failed")


@router.post("/topics/policy")
async def update_policy(update: PolicyUpdate):
    """Aktualisiert Policy für registriertes Topic (persistent)."""
    await update_topic_policy(update.name, {k: v for k, v in update.model_dump().items() if v is not None})
    return {"status": "ok"}


@router.get("/topics")
async def list_topics():
    """Listet registrierte Topics/Subjects (persistent)."""
    return {"topics": await reg_list_topics()}


class SetKeysRequest(BaseModel):
    pattern: str
    role: str  # producer|consumer
    keys: list[str]


@router.post("/keys")
async def set_keys(req: SetKeysRequest):
    """Setzt erlaubte Producer/Consumer Keys für Pattern."""
    await set_allowed_keys(req.pattern, req.role, req.keys)
    return {"status": "ok"}


@router.get("/review/naming")
async def review_naming(subject: str, kind: str = "event"):
    """Prüft Naming-Konventionen für Subject (event|rpc)."""
    if kind == "rpc":
        return validate_rpc_naming(subject)
    return validate_subject_naming(subject)


@router.get("/review/schema")
async def review_schema_endpoint(uri: str):
    """Führt einfaches Schema-Review durch."""
    return review_schema(uri)


@router.get("/review/slo")
async def review_slo(subject: str, p95_target_ms: float | None = None):
    """Vergleicht p95 Latenz mit Zielwert (falls angegeben)."""
    return check_slo(subject, p95_target_ms)


@router.get("/review/policy")
async def review_policy(subject: str):
    """Prüft policy/quotas für konkretes Subject."""
    return check_topic_policy(subject)
