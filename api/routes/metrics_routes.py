"""Client Performance Metrics API-Routen.

Stellt Endpunkte zum Empfangen (POST) und Abfragen (GET) von Client-Metriken
bereit, integriert in OpenTelemetry/Prometheus und unterstützt Multi‑Tenant via
`X-Tenant-Id` Header. Aggregation nach Zeitintervallen wird in‑memory durchgeführt.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Header, Query, Request, Response, status
from prometheus_client import REGISTRY, Counter, Histogram
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from kei_logging import get_logger
from monitoring import get_tracer
from services.webhooks.alerting import emit_warning
from storage.cache.redis_cache import NoOpCache, get_cache_client

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])
@router.get("/live", status_code=status.HTTP_200_OK)
async def live_metrics(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-Id"),
) -> dict[str, Any]:
    """Liefert Live-Metriken (leichtgewichtig) für Realtime-Views.

    Returns:
        Aggregiertes Snapshot-Objekt (Rolling Window basierend auf In-Memory Store)
    """
    items = _TENANT_STORE.get(x_tenant_id, [])[-50:]
    latest = items[-1] if items else None
    avg = {
        "lcp_ms": None,
        "fid_ms": None,
        "cls": None,
        "agent_response_ms": None,
    }
    if items:
        def mean(values: list[float]) -> float:
            return sum(values) / max(1, len(values))
        lcps = [float(s.data.webVitals.get("LCP").value) for s in items if s.data.webVitals.get("LCP")]
        fids = [float(s.data.webVitals.get("FID").value) for s in items if s.data.webVitals.get("FID")]
        clss = [float(s.data.webVitals.get("CLS").value) for s in items if s.data.webVitals.get("CLS")]
        agent_rt = [float(m.value) for s in items for m in s.data.business if m.name == "agent_response_time"]
        avg = {
            "lcp_ms": mean(lcps) if lcps else None,
            "fid_ms": mean(fids) if fids else None,
            "cls": mean(clss) if clss else None,
            "agent_response_ms": mean(agent_rt) if agent_rt else None,
        }
    return {
        "tenant": x_tenant_id,
        "latest": latest.data.model_dump() if latest else None,
        "avg": avg,
        "count": len(items),
    }


@router.get("/kpi/rollup", status_code=status.HTTP_200_OK)
async def kpi_rollup(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-Id"),
    period: str = Query(default="1h", pattern="^(1h|24h|7d)$"),
) -> dict[str, Any]:
    """Aggregiert KPIs über Zeiträume (1h, 24h, 7d) mit Redis-Zwischenspeicher.

    Returns:
        KPI-Objekt mit Metriken wie Webhook-Erfolgsraten, API-P95, WS-Fehlerraten.
    """
    cache_key = f"kpi:{x_tenant_id}:{period}"
    client = await get_cache_client()
    if client and not isinstance(client, NoOpCache):
        try:
            cached = await client.get(cache_key)  # type: ignore[attr-defined]
            if cached:
                import json
                return json.loads(cached)
        except Exception:
            pass

    # Minimalistische Aggregation aus In-Memory Snapshots
    items = _TENANT_STORE.get(x_tenant_id, [])
    horizon = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
    }[period]
    since = datetime.now(UTC) - horizon
    items = [i for i in items if i.received_at >= since]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    lcp = mean([float(s.data.webVitals.get("LCP").value) for s in items if s.data.webVitals.get("LCP")])
    fid = mean([float(s.data.webVitals.get("FID").value) for s in items if s.data.webVitals.get("FID")])
    cls = mean([float(s.data.webVitals.get("CLS").value) for s in items if s.data.webVitals.get("CLS")])
    agent_rt = mean([float(m.value) for s in items for m in s.data.business if m.name == "agent_response_time"])

    kpi = {
        "tenant": x_tenant_id,
        "period": period,
        "count": len(items),
        "api": {"p95_ms": lcp},
        "client": {"fid_ms": fid, "cls": cls},
        "agent": {"avg_response_ms": agent_rt},
        # Platzhalter für Webhook/WS KPIs (können aus Prometheus ergänzt werden)
        "webhook": {"success_rate": None},
        "websocket": {"connection_error_rate": None},
    }

    if client and not isinstance(client, NoOpCache):
        try:
            import json
            await client.setex(cache_key, 300, json.dumps(kpi))  # type: ignore[attr-defined]
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"KPI-Cache-Speicherung fehlgeschlagen - Verbindungsproblem: {e}")
        except (ValueError, TypeError) as e:
            logger.debug(f"KPI-Cache-Speicherung fehlgeschlagen - JSON-Serialisierungsfehler: {e}")
        except Exception as e:
            logger.warning(f"KPI-Cache-Speicherung fehlgeschlagen - Unerwarteter Fehler: {e}")
    return kpi


# Prometheus Metriken
CLIENT_METRICS_RECEIVED = Counter(
    "client_metrics_received_total",
    "Anzahl empfangener Client-Metrik-Snapshots",
    ["tenant"],
)

LCP_HIST = Histogram("client_web_vitals_lcp_ms", "Largest Contentful Paint (ms)", buckets=(500, 1000, 1500, 2500, 4000, 6000, 10000))
FID_HIST = Histogram("client_web_vitals_fid_ms", "First Input Delay (ms)", buckets=(16, 50, 100, 200, 500, 1000))
CLS_HIST = Histogram("client_web_vitals_cls", "Cumulative Layout Shift", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0))
AGENT_RT_HIST = Histogram("client_business_agent_response_ms", "Agent Response Time (ms)", buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000))

# Cache-Performance Metriken
CACHE_HIT_RATE = Histogram("client_cache_hit_rate", "Cache Hit Rate", ["tenant", "cache_type"], buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
CACHE_REQUESTS_TOTAL = Counter("client_cache_requests_total", "Total Cache Requests", ["tenant", "cache_type", "result"])


class WebVital(BaseModel):
    """Web Vitals Metrik."""

    name: str
    value: float
    rating: str | None = None
    timestamp: float


class NavigationMetrics(BaseModel):
    startTime: float
    domContentLoaded: float
    loadEventEnd: float
    firstPaint: float | None = None
    firstContentfulPaint: float | None = None
    transferSize: int | None = None


class ResourceMetric(BaseModel):
    name: str
    initiatorType: str | None = None
    duration: float
    transferSize: int | None = None
    startTime: float


class MemoryMetrics(BaseModel):
    usedJSHeapSize: int | None = None
    totalJSHeapSize: int | None = None
    jsHeapSizeLimit: int | None = None
    timestamp: int


class NetworkQualityMetrics(BaseModel):
    effectiveType: str | None = None
    rtt: int | None = None
    downlink: float | None = None
    saveData: bool | None = None
    timestamp: int


class BusinessMetric(BaseModel):
    name: str
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    timestamp: int


class CacheMetricByType(BaseModel):
    """Cache-Metrik pro Typ und Tenant."""

    cache_type: str = Field(..., description="Cache-Typ (agents, webhooks, etc.)")
    tenant_id: str | None = Field(None, description="Tenant-ID falls vorhanden")
    hits: int = Field(..., ge=0, description="Anzahl Cache-Hits")
    misses: int = Field(..., ge=0, description="Anzahl Cache-Misses")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Hit-Rate (0.0-1.0)")


class CacheMetrics(BaseModel):
    """Client-seitige Cache-Metriken."""

    hits: int = Field(..., ge=0, description="Gesamt Cache-Hits")
    misses: int = Field(..., ge=0, description="Gesamt Cache-Misses")
    miss_rate: float = Field(..., ge=0.0, le=1.0, description="Miss-Rate (0.0-1.0)")
    by_type: list[CacheMetricByType] = Field(default_factory=list, description="Metriken pro Cache-Typ")


class SessionInfo(BaseModel):
    id: str
    startedAt: int
    durationMs: int


class PerformanceSnapshot(BaseModel):
    webVitals: dict[str, WebVital] = Field(default_factory=dict)
    navigation: NavigationMetrics | None = None
    resources: list[ResourceMetric] = Field(default_factory=list)
    memory: MemoryMetrics | None = None
    network: NetworkQualityMetrics | None = None
    business: list[BusinessMetric] = Field(default_factory=list)
    errorsPerMinute: int | None = 0
    crashes: int | None = 0
    featureUsage: dict[str, int] = Field(default_factory=dict)
    session: SessionInfo
    cache_metrics: CacheMetrics | None = Field(None, description="Cache-Performance-Metriken")


@dataclass
class StoredSnapshot:
    tenant: str
    received_at: datetime
    data: PerformanceSnapshot


# In-Memory Speicher (pro Tenant), begrenzt auf N Items pro Tenant (Ringbuffer)
_TENANT_STORE: dict[str, list[StoredSnapshot]] = defaultdict(list)
_TENANT_STORE_LIMIT = 5000


def _append_snapshot(tenant: str, snap: PerformanceSnapshot) -> None:
    buff = _TENANT_STORE[tenant]
    buff.append(StoredSnapshot(tenant=tenant, received_at=datetime.now(UTC), data=snap))
    if len(buff) > _TENANT_STORE_LIMIT:
        del buff[: len(buff) - _TENANT_STORE_LIMIT]


def _aggregate(snapshots: list[StoredSnapshot], interval: timedelta) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    snapshots = sorted(snapshots, key=lambda s: s.received_at)
    start = snapshots[0].received_at.replace(second=0, microsecond=0)
    end = snapshots[-1].received_at
    buckets: list[tuple[datetime, datetime]] = []
    cur = start
    while cur <= end:
        buckets.append((cur, cur + interval))
        cur = cur + interval

    result: list[dict[str, Any]] = []
    for b_start, b_end in buckets:
        items = [s for s in snapshots if b_start <= s.received_at < b_end]
        if not items:
            result.append({"start": b_start.isoformat(), "end": b_end.isoformat(), "count": 0})
            continue
        # Durchschnittswerte berechnen (Beispiel: LCP/FID/CLS, Agent RT)
        def mean(values: list[float]) -> float:
            return sum(values) / max(1, len(values))

        lcps = [float(s.data.webVitals.get("LCP").value) for s in items if s.data.webVitals.get("LCP")]
        fids = [float(s.data.webVitals.get("FID").value) for s in items if s.data.webVitals.get("FID")]
        clss = [float(s.data.webVitals.get("CLS").value) for s in items if s.data.webVitals.get("CLS")]
        agent_rt = [float(m.value) for s in items for m in s.data.business if m.name == "agent_response_time"]

        result.append(
            {
                "start": b_start.isoformat(),
                "end": b_end.isoformat(),
                "count": len(items),
                "lcp_ms": mean(lcps) if lcps else None,
                "fid_ms": mean(fids) if fids else None,
                "cls": mean(clss) if clss else None,
                "agent_response_ms": mean(agent_rt) if agent_rt else None,
            }
        )
    return result


def _interval_to_delta(interval: str) -> timedelta:
    return {
        "1min": timedelta(minutes=1),
        "5min": timedelta(minutes=5),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }.get(interval, timedelta(minutes=1))


@router.post("/client", status_code=status.HTTP_202_ACCEPTED)
async def ingest_client_metrics(
    payload: PerformanceSnapshot,
    response: Response,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-Id"),
) -> dict[str, str]:
    """Empfängt Client-Metriken und integriert Observability.

    - OTEL Span mit Basis-Attributen
    - Prometheus Counter/Histograms
    - Optionales Alerting bei Grenzwertüberschreitungen
    - Speicherung in In‑Memory Store für spätere Abfragen
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("client_metrics_ingest") as span:  # type: ignore[call-arg]
        span.set_attribute("tenant.id", x_tenant_id)
        span.set_attribute("session.id", payload.session.id)
        CLIENT_METRICS_RECEIVED.labels(tenant=x_tenant_id).inc()

        # Prometheus Histograms füllen
        if payload.webVitals.get("LCP"):
            with contextlib.suppress(Exception):
                LCP_HIST.observe(float(payload.webVitals["LCP"].value))
        if payload.webVitals.get("FID"):
            with contextlib.suppress(Exception):
                FID_HIST.observe(float(payload.webVitals["FID"].value))
        if payload.webVitals.get("CLS"):
            with contextlib.suppress(Exception):
                CLS_HIST.observe(float(payload.webVitals["CLS"].value))

        for bm in payload.business:
            if bm.name == "agent_response_time":
                with contextlib.suppress(Exception):
                    AGENT_RT_HIST.observe(float(bm.value))

        # Cache-Metriken verarbeiten
        if payload.cache_metrics:
            cache_metrics = payload.cache_metrics

            # Gesamt-Cache-Hit-Rate
            try:
                CACHE_HIT_RATE.labels(tenant=x_tenant_id, cache_type="total").observe(1.0 - cache_metrics.miss_rate)
                CACHE_REQUESTS_TOTAL.labels(tenant=x_tenant_id, cache_type="total", result="hit").inc(cache_metrics.hits)
                CACHE_REQUESTS_TOTAL.labels(tenant=x_tenant_id, cache_type="total", result="miss").inc(cache_metrics.misses)
            except (AttributeError, ValueError) as e:
                logger.debug(f"Cache-Metriken-Verarbeitung fehlgeschlagen - Attribut-/Wert-Fehler: {e}")
            except Exception as e:
                logger.warning(f"Cache-Metriken-Verarbeitung fehlgeschlagen - Unerwarteter Fehler: {e}")

            # Cache-Metriken pro Typ
            for type_metric in cache_metrics.by_type:
                try:
                    cache_type = type_metric.cache_type
                    tenant_label = type_metric.tenant_id or x_tenant_id

                    CACHE_HIT_RATE.labels(tenant=tenant_label, cache_type=cache_type).observe(type_metric.hit_rate)
                    CACHE_REQUESTS_TOTAL.labels(tenant=tenant_label, cache_type=cache_type, result="hit").inc(type_metric.hits)
                    CACHE_REQUESTS_TOTAL.labels(tenant=tenant_label, cache_type=cache_type, result="miss").inc(type_metric.misses)
                except Exception:
                    pass

        # Alerts bei Grenzwerten
        try:
            lcp = float(payload.webVitals.get("LCP").value) if payload.webVitals.get("LCP") else None
            fid = float(payload.webVitals.get("FID").value) if payload.webVitals.get("FID") else None
            cls = float(payload.webVitals.get("CLS").value) if payload.webVitals.get("CLS") else None
            if lcp and lcp > 4000:
                await emit_warning("LCP hoch", {"tenant": x_tenant_id, "lcp_ms": lcp})
            if fid and fid > 200:
                await emit_warning("FID hoch", {"tenant": x_tenant_id, "fid_ms": fid})
            if cls and cls > 0.25:
                await emit_warning("CLS hoch", {"tenant": x_tenant_id, "cls": cls})
        except (AttributeError, ValueError, TypeError) as e:
            # Alerting ist best-effort
            logger.debug(f"Web-Vitals-Alerting fehlgeschlagen - Attribut-/Wert-/Typ-Fehler: {e}")
        except Exception as e:
            # Alerting ist best-effort
            logger.warning(f"Web-Vitals-Alerting fehlgeschlagen - Unerwarteter Fehler: {e}")

        # Snapshot speichern
        _append_snapshot(x_tenant_id, payload)

    response.headers["X-Tenant-Id"] = x_tenant_id
    return {"status": "accepted"}


@router.get("/stream", status_code=status.HTTP_200_OK)
async def stream_metrics(request: Request):
    """Server-Sent Events Stream für Live-Dashboards.

    Sendet periodisch (2s) kompakte KPI‑Snapshots. Beendet, wenn Client trennt.
    Rate‑Limitierung sollte vorgelagert per Middleware erfolgen.
    """
    import asyncio
    import json

    from starlette.responses import StreamingResponse

    async def event_gen():
        while True:
            if await request.is_disconnected():
                break
            snap = await live_metrics(x_tenant_id=request.headers.get("X-Tenant-Id", "default"))
            data = json.dumps(snap)
            yield f"data: {data}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/client", status_code=status.HTTP_200_OK)
async def query_client_metrics(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-Id"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=1000),
    start: str | None = Query(default=None, description="ISO Startzeit"),
    end: str | None = Query(default=None, description="ISO Endzeit"),
    interval: str = Query(default="1min", pattern="^(1min|5min|1h|1d)$"),
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    browser: str | None = Query(default=None),
) -> dict[str, Any]:
    """Liefert historische Metriken mit Paginierung und optionaler Aggregation."""
    items = _TENANT_STORE.get(x_tenant_id, [])
    if start:
        try:
            s = datetime.fromisoformat(start)
            items = [i for i in items if i.received_at >= s]
        except Exception:
            pass
    if end:
        try:
            e = datetime.fromisoformat(end)
            items = [i for i in items if i.received_at <= e]
        except Exception:
            pass

    # Segment-Filter
    if user_id:
      items = [i for i in items if (i.data.context or {}).get("userId") == user_id]
    if session_id:
      items = [i for i in items if i.data.session.id == session_id]
    if browser:
      items = [i for i in items if (i.data.context or {}).get("browser") and browser.lower() in str((i.data.context or {}).get("browser")).lower()]

    total = len(items)
    # Aggregation
    agg = _aggregate(items, _interval_to_delta(interval))

    # Pagination auf Roh‑Items
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = items[start_idx:end_idx]

    return {
        "tenant": x_tenant_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "received_at": it.received_at.isoformat(),
                "session_id": it.data.session.id,
                "web_vitals": {k: v.model_dump() for k, v in it.data.webVitals.items()},
            }
            for it in page_items
        ],
        "aggregates": agg,
    }


@router.get("/business-kpis", status_code=status.HTTP_200_OK)
async def business_kpis(
    request: Request,
    tenant_id: str | None = Query(default=None, alias="tenant_id"),
) -> JSONResponse:
    """Aggregiert Business‑KPIs (best effort) aus Prometheus Registry.

    Achtung: Für präzise Berechnungen produktiv Prometheus HTTP API abfragen.
    """
    try:
        tenant = tenant_id or request.headers.get("X-Tenant-Id") or ""
        metrics: dict[str, Any] = {"tenant_id": tenant}

        # Agent Task Erfolgsrate (optional vorhandene Metrik keiko_agent_tasks_total)
        success = 0.0
        total = 0.0
        for fam in REGISTRY.collect():
            if fam.name == "keiko_agent_tasks_total":
                for s in fam.samples:
                    if s.name != "keiko_agent_tasks_total":
                        continue
                    lbls = s.labels or {}
                    if tenant and lbls.get("tenant_id") not in (None, "", tenant):
                        continue
                    total += float(s.value)
                    if lbls.get("status") == "success":
                        success += float(s.value)
        metrics["agent_task_success_rate"] = (success / total) if total > 0 else None

        # WebSocket aktive Verbindungen
        ws_active = 0.0
        for fam in REGISTRY.collect():
            if fam.name == "keiko_websocket_connections_active":
                for s in fam.samples:
                    if s.name == "keiko_websocket_connections_active":
                        ws_active = float(s.value)
        metrics["websocket_active_connections"] = ws_active

        # Webhook Deliveries
        deliveries = 0.0
        for fam in REGISTRY.collect():
            if fam.name == "webhook_deliveries_total":
                for s in fam.samples:
                    if s.name != "webhook_deliveries_total":
                        continue
                    lbls = s.labels or {}
                    if tenant and lbls.get("tenant_id") not in (None, "", tenant):
                        continue
                    deliveries += float(s.value)
        metrics["webhook_deliveries_total"] = deliveries

        return JSONResponse(metrics)
    except Exception as e:
        logger.exception(f"KPI Aggregation Fehler: {e}")
        return JSONResponse({"error": "aggregation_failed"}, status_code=500)
