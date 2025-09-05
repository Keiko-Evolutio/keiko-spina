"""Governance-Checks: Naming-Conventions, Schema-Review, SLO/SLA-Checks."""

from __future__ import annotations

import re
from typing import Any

from kei_logging import get_logger

from .management_registry import get_topic
from .metrics import BusMetrics
from .schema_registry import get_schema_registry

logger = get_logger(__name__)


_EVENT_PATTERN = re.compile(r"^kei\.[a-z0-9-]+\.[a-z0-9-]+\.[a-z0-9-]+\.[a-z0-9-]+\.v\d+$")
_RPC_PATTERN = re.compile(r"^kei\.rpc\.[a-z0-9-]+\.[a-z0-9-]+\.v\d+$")


def validate_subject_naming(subject: str) -> dict[str, Any]:
    """Prüft Naming-Konventionen für Event/Command Subjects.

    Format: kei.{tenant}.{bounded_context}.{aggregate}.{event}.vN
    """
    ok = bool(_EVENT_PATTERN.match(subject))
    parts = subject.split(".")
    detail = {
        "tenant": parts[1] if len(parts) > 1 else None,
        "bounded_context": parts[2] if len(parts) > 2 else None,
        "aggregate": parts[3] if len(parts) > 3 else None,
        "event": parts[4] if len(parts) > 4 else None,
        "version": parts[5] if len(parts) > 5 else None,
    }
    return {"valid": ok, "details": detail}


def validate_rpc_naming(subject: str) -> dict[str, Any]:
    """Prüft Naming-Konventionen für RPC Subjects.

    Format: kei.rpc.{service}.{method}.vN
    """
    ok = bool(_RPC_PATTERN.match(subject))
    parts = subject.split(".")
    detail = {
        "service": parts[2] if len(parts) > 2 else None,
        "method": parts[3] if len(parts) > 3 else None,
        "version": parts[4] if len(parts) > 4 else None,
    }
    return {"valid": ok, "details": detail}


def review_schema(uri: str) -> dict[str, Any]:
    """Einfaches Schema-Review: Existenz, Typ, minimaler Meta-Check."""
    reg = get_schema_registry()
    schema, s_type = reg.get_with_type(uri)
    if not schema or not s_type:
        return {"approved": False, "reason": "schema_not_found"}
    # JSON: muss Objekt sein
    if s_type == "json" and not isinstance(schema, dict):
        return {"approved": False, "reason": "invalid_json_schema"}
    # Avro/Protobuf: hier nur Existenz
    return {"approved": True, "type": s_type}


def check_slo(subject: str, p95_target_ms: float | None = None) -> dict[str, Any]:
    """Vergleicht p95-Latenz gegen Zielwert, falls verfügbar (lokale Samples)."""
    bm = BusMetrics()
    pct = bm.get_latency_percentiles(subject)
    if p95_target_ms is None:
        return {"ok": True, "percentiles": pct}
    ok = pct.get("p95", 0.0) <= float(p95_target_ms)
    return {"ok": ok, "percentiles": pct, "target_ms": p95_target_ms}


async def check_topic_policy(subject: str) -> dict[str, Any]:
    """Prüft, ob für ein Subject eine Policy/Quota vorhanden ist (optional)."""
    try:
        # Direkte Subject-Policies
        topic = await get_topic(subject)
        return {"has_policy": bool(topic), "policy": topic}
    except Exception:
        return {"has_policy": False, "policy": None}


__all__ = [
    "check_slo",
    "check_topic_policy",
    "review_schema",
    "validate_rpc_naming",
    "validate_subject_naming",
]
