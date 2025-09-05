"""Administrative Bus-APIs: Schema-Registry, Outbox-Flush, Policies."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.messaging.chaos import list_replay, set_chaos_profile
from services.messaging.config import bus_settings
from services.messaging.dlq import ensure_dlq_stream, list_dlq_messages
from services.messaging.metrics import BusMetrics
from services.messaging.outbox import Outbox
from services.messaging.schema_registry import get_schema_registry

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/bus/admin", tags=["KEI-Bus-Admin"])


class RegisterSchemaRequest(BaseModel):
    """Request zum Registrieren eines JSON Schemas."""

    uri: str = Field(..., description="Eindeutige Schema-URI")
    schema_data: dict[str, Any] = Field(..., description="Schema (JSON/Avro/Protobuf)", alias="schema")
    schema_type: str = Field("json", description="Schema-Typ: json|avro|protobuf")
    compatibility: str = Field("backward", description="Kompatibilitätsregel: backward|forward")


@router.post("/schemas/register")
async def register_schema(req: RegisterSchemaRequest):
    """Registriert ein JSON Schema in der Registry."""
    try:
        registry = get_schema_registry()
        version = registry.register(req.uri, req.schema_data, req.schema_type, req.compatibility)
        return {"status": "ok", "version": version}
    except Exception as exc:
        logger.exception(f"Schema-Registrierung fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="schema_register_failed")


@router.get("/schemas/{uri}")
async def get_schema(uri: str):
    """Liefert aktuelles Schema (mit Typ)."""
    try:
        reg = get_schema_registry()
        schema, s_type = reg.get_with_type(uri)
        if not schema:
            raise HTTPException(status_code=404, detail="schema_not_found")
        return {"uri": uri, "type": s_type, "schema": schema}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Schema-Abruf fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="schema_get_failed")


@router.get("/schemas")
async def list_schemas():
    """Listet alle Schemata (URI, Version, Typ)."""
    try:
        reg = get_schema_registry()
        return {"schemas": reg.list_all()}
    except Exception as exc:
        logger.exception(f"Schema-Liste fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="schema_list_failed")


@router.get("/versioning")
async def versioning_info():
    """Gibt Envelope/SDK-Versionen und aktivierte Capabilities/Flags aus."""
    return {
        "envelope_version": bus_settings.envelope_version,
        "sdk_versions": bus_settings.sdk_versions,
        "capabilities": bus_settings.capabilities,
        "feature_flags": bus_settings.feature_flags,
        "deprecations": bus_settings.deprecations,
    }


@router.post("/outbox/{name}/flush")
async def flush_outbox(name: str):
    """Leert eine Outbox-Liste und gibt Anzahl der Elemente zurück."""
    try:
        outbox = Outbox(name)
        count = await outbox.flush()
        return {"status": "ok", "flushed": count}
    except Exception as exc:
        logger.exception(f"Outbox-Flush fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="outbox_flush_failed")


@router.get("/dlq")
async def list_dlq(subject_filter: str | None = None, max_items: int = 100):
    """Listet DLQ-Messages (vereinfachte Übersicht)."""
    try:
        from services.messaging.service import get_bus_service

        svc = get_bus_service()
        await svc.initialize()
        # JetStream Kontext aus Provider holen
        js = svc._nats.js  # type: ignore[attr-defined]
        await ensure_dlq_stream(js)
        items = await list_dlq_messages(js, subject_filter, max_items)
        return {"status": "ok", "items": items}
    except Exception as exc:
        logger.exception(f"DLQ-Liste fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="dlq_list_failed")


@router.get("/latency/percentiles")
async def latency_percentiles(subject: str | None = None):
    """Gibt p50/p95/p99 Latenzen aus lokal gepufferten Samples aus."""
    try:
        bm = BusMetrics()
        return {"subject": subject, "percentiles": bm.get_latency_percentiles(subject)}
    except Exception as exc:
        logger.exception(f"Latency-Percentiles Fehler: {exc}")
        raise HTTPException(status_code=500, detail="latency_percentiles_failed")


class ChaosProfileRequest(BaseModel):
    profile: str  # none|delay|drop|mixed
    delay_ms: int = 100


@router.post("/chaos/profile")
async def set_chaos(req: ChaosProfileRequest):
    """Setzt Chaos-Profil für Bus-Processing (Test/Sandbox)."""
    try:
        await set_chaos_profile(req.profile, req.delay_ms)
        return {"status": "ok"}
    except Exception as exc:
        logger.exception(f"Chaos-Set fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="chaos_set_failed")


@router.get("/replay")
async def get_replay(subject: str, count: int = 100):
    """Listet gespeicherte Rohframes für Replay/Analyse."""
    try:
        items = await list_replay(subject, count)
        return {"subject": subject, "items": items}
    except Exception as exc:
        logger.exception(f"Replay-Liste fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="replay_list_failed")
