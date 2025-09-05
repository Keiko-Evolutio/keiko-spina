"""KEI-RPC REST API Router (v1).

Stellt standardisierte, stabile REST-Verträge bereit:
- Ressourcenorientierte Endpunkte mit Pagination, Filtering, Sorting
- Batch-Operationen
- Partial Updates via PATCH
- Idempotenz via `Idempotency-Key`
- ETags und Conditional Requests
- Einheitliches Fehlerformat (Problem+JSON kompatibel)
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kei_logging import get_logger

try:
    from prometheus_client import Histogram
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _PROM_AVAILABLE = False
import contextlib

from security.kei_mcp_auth import require_auth, require_rate_limit


class Scope(str):
    """Fix definierte Scopes für RPC-Operationen."""

    READ = "kei.rpc.read"
    WRITE = "kei.rpc.write"
    ADMIN = "kei.rpc.admin"


async def require_scope(
    request: Request,
    _: str = Depends(require_auth),
    required: str = Scope.READ,
) -> None:
    """Prüft erforderlichen Scope anhand des im Request-State gesetzten Auth-Kontexts.

    Raises:
        HTTPException: Wenn erforderlicher Scope fehlt.
    """
    # Robust gegen fehlenden State
    try:
        ctx = getattr(request.state, "auth_context", {}) or {}
        scopes = list(ctx.get("scopes", []) or [])
    except Exception:
        scopes = []

    # mTLS-Anfragen haben evtl. keine Scopes → policy: nur READ erlauben
    if not scopes and required != Scope.READ:
        raise HTTPException(status_code=403, detail="Insufficient scope for operation")

    if required not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing required scope: {required}")


def require_scope_dep(required: str):
    """Erzeugt eine FastAPI-Dependency, die einen Scope prüft."""

    async def _dep(request: Request, _: str = Depends(require_auth)) -> None:
        return await require_scope(request, _, required=required)

    return _dep


logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/rpc", tags=["kei-rpc"])


# ========= Modelle =========
# ========= Metriken =========
if _PROM_AVAILABLE:
    keiko_rpc_duration_seconds = Histogram(
        "keiko_rpc_duration_seconds",
        "Latenz der KEI-RPC REST Endpunkte",
        labelnames=["endpoint", "method", "status"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)
    )
else:
    keiko_rpc_duration_seconds = None  # type: ignore


class _RPCMetricsTimer:
    """Kontextmanager für Latenzmessung pro Request."""

    def __init__(self, endpoint: str, method: str):
        self.endpoint = endpoint
        self.method = method
        self._timer = None

    def __enter__(self):
        if keiko_rpc_duration_seconds:
            self._timer = keiko_rpc_duration_seconds.labels(self.endpoint, self.method, "").time()
        return self

    def set_status(self, response_status: int) -> None:
        if self._timer and keiko_rpc_duration_seconds:
            try:
                # Recreate with status label by observing elapsed time once
                elapsed = self._timer._HistogramTimer__hist._sum.get() - self._timer._HistogramTimer__last  # type: ignore[attr-defined]
            except Exception:
                elapsed = None
            try:
                self._timer.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            if elapsed is not None:
                with contextlib.suppress(Exception):
                    keiko_rpc_duration_seconds.labels(self.endpoint, self.method, str(response_status)).observe(elapsed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            with contextlib.suppress(Exception):
                self._timer.stop()


class Pagination(BaseModel):
    """Pagination-Metadaten."""

    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=200)
    total: int = Field(0, ge=0)


class RPCResource(BaseModel):
    """Beispiel-Ressource für Demonstrationszwecke."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime


class RPCResourceCreate(BaseModel):
    """Create-Request für Ressource."""

    name: str = Field(..., min_length=1, max_length=200)


class RPCResourceUpdate(BaseModel):
    """Partial-Update für Ressource."""

    name: str | None = Field(None, min_length=1, max_length=200)


class RPCResourceListResponse(BaseModel):
    """Antwort für Listen-Operationen mit Pagination."""

    items: list[RPCResource]
    pagination: Pagination


class ProblemJSON(BaseModel):
    """Problem+JSON kompatibles Fehlerobjekt."""

    type: str = Field(default="about:blank")
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


# ========= In-Memory Demo Storage =========
_TENANT_RESOURCES: dict[str, dict[str, RPCResource]] = {}
_TENANT_CONTENT: dict[str, dict[str, bytes]] = {}


def _make_etag(resource: RPCResource) -> str:
    """Erzeugt ETag basierend auf Updated-At und Id."""
    base = f"{resource.id}:{int(resource.updated_at.timestamp())}"
    return f'W/"{hash(base) & 0xFFFFFFFF:x}"'


def _parse_range_header(range_header: str, total_length: int) -> tuple[int, int] | None:
    """Parst einen HTTP Range-Header und gibt (start, end) zurück.

    Unterstützt nur Single-Range im Format "bytes=start-end".
    """
    try:
        if not range_header or not range_header.startswith("bytes="):
            return None
        spec = range_header[len("bytes="):]
        # Beispiele: "100-200", "100-", "-500"
        if spec.startswith("-"):
            # Letzte N Bytes
            suffix = int(spec[1:])
            if suffix <= 0:
                return None
            start = max(0, total_length - suffix)
            end = total_length - 1
            return start, end
        if "-" in spec:
            start_str, end_str = spec.split("-", 1)
            start = int(start_str)
            if start < 0 or start >= total_length:
                return None
            if end_str:
                end = int(end_str)
                if end < start:
                    return None
                end = min(end, total_length - 1)
                return start, end
            # Offenes Ende
            return start, total_length - 1
        return None
    except Exception:
        return None


# ========= Dependencies =========


async def rate_limit_discovery_dep(request: Request) -> None:
    """Rate Limiting Dependency für Discovery-Operationen."""
    await require_rate_limit(request, "discovery")


async def rate_limit_default_dep(request: Request) -> None:
    """Rate Limiting Dependency für Standard-Operationen."""
    await require_rate_limit(request, "default")


# ========= Endpunkte =========


@router.get("/resources", response_model=RPCResourceListResponse, responses={
    401: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    429: {"description": "Too Many Requests"}
})
async def list_resources(
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200),
    q: str | None = Query(None, description="Filter nach Name enthält"),
    sort: str = Query("-updated_at", description="Sortierfeld, Präfix '-' für absteigend"),
    _: str = Depends(require_auth),
    __: None = Depends(rate_limit_discovery_dep),
    ___: None = Depends(require_scope_dep(Scope.READ)),
) -> RPCResourceListResponse:
    """Listet Ressourcen mit Pagination/Filtering/Sorting."""
    if not x_tenant_id:
        problem = ProblemJSON(title="Tenant Required", status=403, detail="X-Tenant-Id header fehlt")
        return JSONResponse(content=problem.model_dump(), status_code=403, media_type="application/problem+json")

    tenant_store = _TENANT_RESOURCES.get(x_tenant_id, {})
    items = list(tenant_store.values())

    if q:
        items = [r for r in items if q.lower() in r.name.lower()]

    reverse = sort.startswith("-")
    key = sort.lstrip("-")
    if key not in {"name", "created_at", "updated_at"}:
        key = "updated_at"
    items.sort(key=lambda r: getattr(r, key), reverse=reverse)

    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]

    resp = RPCResourceListResponse(items=page_items, pagination=Pagination(page=page, per_page=per_page, total=total))
    return cast("RPCResourceListResponse", resp)


@router.post(
    "/resources",
    response_model=RPCResource,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
        412: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
        429: {"description": "Too Many Requests"}
    }
)
async def create_resource(
    body: RPCResourceCreate,
    response: Response,
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    _: str = Depends(require_auth),
    __: None = Depends(rate_limit_default_dep),
    ___: None = Depends(require_scope_dep(Scope.WRITE)),
) -> RPCResource:
    """Erstellt Ressource. Unterstützt ETag-Precondition und Idempotenz."""
    if not x_tenant_id:
        problem = ProblemJSON(title="Tenant Required", status=403, detail="X-Tenant-Id header fehlt")
        return JSONResponse(content=problem.model_dump(), status_code=403, media_type="application/problem+json")

    now = datetime.utcnow()
    # Einfache deterministische Idempotenz per Key (pro Tenant)
    tenant_store = _TENANT_RESOURCES.setdefault(x_tenant_id, {})
    if idempotency_key and idempotency_key in tenant_store:
        res = tenant_store[idempotency_key]
        response.headers["ETag"] = _make_etag(res)
        return res

    new_id = idempotency_key or f"{x_tenant_id}_res_{len(tenant_store)+1}"
    res = RPCResource(id=new_id, name=body.name, created_at=now, updated_at=now)
    tenant_store[new_id] = res
    response.headers["ETag"] = _make_etag(res)
    return res


@router.get("/resources/{resource_id}", response_model=RPCResource, responses={
    304: {"description": "Not Modified"},
    401: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    404: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    429: {"description": "Too Many Requests"}
})
async def get_resource(
    resource_id: str = Path(..., min_length=1),
    if_none_match: str | None = Header(None, alias="If-None-Match"),
    response: Response = None,  # type: ignore[assignment]
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    _: str = Depends(require_auth),
    __: None = Depends(rate_limit_default_dep),
    ___: None = Depends(require_scope_dep(Scope.READ)),
) -> RPCResource:
    """Liest Ressource mit ETag-Unterstützung."""
    if not x_tenant_id:
        problem = ProblemJSON(title="Tenant Required", status=403, detail="X-Tenant-Id header fehlt")
        return JSONResponse(content=problem.model_dump(), status_code=403, media_type="application/problem+json")

    tenant_store = _TENANT_RESOURCES.get(x_tenant_id, {})
    res = tenant_store.get(resource_id)
    if not res:
        problem = ProblemJSON(title="Not Found", status=404, detail="Resource not found")
        return JSONResponse(content=problem.model_dump(), status_code=404, media_type="application/problem+json")

    etag = _make_etag(res)
    if if_none_match and if_none_match == etag:
        # 304 ohne Body, aber mit ETag
        return Response(status_code=304, headers={"ETag": etag})

    if response is not None:
        response.headers["ETag"] = etag
    return res


@router.get("/resources/{resource_id}/content", responses={
    200: {"description": "Vollständiger Inhalt"},
    206: {"description": "Teilinhalt (Partial Content)"},
    401: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    404: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    416: {"description": "Range Not Satisfiable"},
    429: {"description": "Too Many Requests"}
})
async def get_resource_content(
    resource_id: str = Path(..., min_length=1),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    range_header: str | None = Header(None, alias="Range"),
    _: str = Depends(require_auth),
    __: None = Depends(rate_limit_default_dep),
    ___: None = Depends(require_scope_dep(Scope.READ)),
):
    """Liefert binären Ressourceninhalt mit Range-Unterstützung.

    Setzt `Accept-Ranges: bytes` und beantwortet gültige `Range`-Header mit 206
    inklusive `Content-Range`. Bei ungültigen Bereichen wird 416 mit
    `Content-Range: bytes */total` zurückgegeben.
    """
    if not x_tenant_id:
        problem = ProblemJSON(title="Tenant Required", status=403, detail="X-Tenant-Id header fehlt")
        return Response(content=problem.model_dump_json(), status_code=403, media_type="application/problem+json")

    # Ressource prüfen
    tenant_store = _TENANT_RESOURCES.get(x_tenant_id, {})
    res = tenant_store.get(resource_id)
    if not res:
        problem = ProblemJSON(title="Not Found", status=404, detail="Resource not found")
        return Response(content=problem.model_dump_json(), status_code=404, media_type="application/problem+json")

    # Demo-Inhalt bereitstellen (persistiert pro Tenant/Resource, falls nicht vorhanden)
    content_store = _TENANT_CONTENT.setdefault(x_tenant_id, {})
    blob = content_store.get(resource_id)
    if blob is None:
        # Erzeuge deterministischen Demo-Content (~5MB)
        chunk = (f"{resource_id}-content-" * 100).encode("utf-8")
        target_size = 5 * 1024 * 1024
        parts: list[bytes] = []
        size = 0
        while size < target_size:
            parts.append(chunk)
            size += len(chunk)
        blob = b"".join(parts)[:target_size]
        content_store[resource_id] = blob

    total = len(blob)
    headers: dict[str, str] = {"Accept-Ranges": "bytes"}

    # Range-Handling
    rng = _parse_range_header(range_header or "", total)
    if rng is None and (range_header or "") != "":
        # Ungültiger Range → 416
        headers["Content-Range"] = f"bytes */{total}"
        return Response(status_code=416, headers=headers)

    if rng is None:
        # Voller Inhalt (200)
        headers["Content-Length"] = str(total)
        return Response(content=blob, media_type="application/octet-stream", headers=headers)

    start, end = rng
    if start >= total or end >= total or start > end:
        headers["Content-Range"] = f"bytes */{total}"
        return Response(status_code=416, headers=headers)

    part = blob[start : end + 1]
    headers.update({
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(len(part)),
    })
    return Response(status_code=206, content=part, media_type="application/octet-stream", headers=headers)


@router.patch("/resources/{resource_id}", response_model=RPCResource, responses={
    401: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    404: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    412: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    429: {"description": "Too Many Requests"}
})
async def patch_resource(
    resource_id: str,
    body: RPCResourceUpdate,
    if_match: str | None = Header(None, alias="If-Match"),
    response: Response = None,  # type: ignore[assignment]
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    _: str = Depends(require_auth),
    __: None = Depends(rate_limit_default_dep),
    ___: None = Depends(require_scope_dep(Scope.WRITE)),
) -> RPCResource:
    """Teil-Update mit ETag-Precondition (If-Match)."""
    if not x_tenant_id:
        problem = ProblemJSON(title="Tenant Required", status=403, detail="X-Tenant-Id header fehlt")
        return JSONResponse(content=problem.model_dump(), status_code=403, media_type="application/problem+json")

    tenant_store = _TENANT_RESOURCES.get(x_tenant_id, {})
    res = tenant_store.get(resource_id)
    if not res:
        problem = ProblemJSON(title="Not Found", status=404, detail="Resource not found")
        return JSONResponse(content=problem.model_dump(), status_code=404, media_type="application/problem+json")

    current_etag = _make_etag(res)
    if if_match and if_match != current_etag:
        problem = ProblemJSON(title="Precondition Failed", status=412, detail="ETag mismatch")
        return JSONResponse(content=problem.model_dump(), status_code=412, media_type="application/problem+json")

    if body.name is not None:
        res.name = body.name
        res.updated_at = datetime.utcnow()

    tenant_store[resource_id] = res
    if response is not None:
        response.headers["ETag"] = _make_etag(res)
    return res


@router.post("/resources:batch", response_model=list[RPCResource], responses={
    401: {"content": {"application/problem+json": {"schema": ProblemJSON.model_json_schema()}}},
    429: {"description": "Too Many Requests"}
})
async def batch_create(
    items: list[RPCResourceCreate],
    response: Response,
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    _: str = Depends(require_auth),
    __: None = Depends(rate_limit_default_dep),
    ___: None = Depends(require_scope_dep(Scope.WRITE)),
) -> list[RPCResource]:
    """Batch-Erstellung von Ressourcen."""
    if not x_tenant_id:
        problem = ProblemJSON(title="Tenant Required", status=403, detail="X-Tenant-Id header fehlt")
        return JSONResponse(content=problem.model_dump(), status_code=403, media_type="application/problem+json")

    tenant_store = _TENANT_RESOURCES.setdefault(x_tenant_id, {})
    created: list[RPCResource] = []
    for item in items:
        now = datetime.utcnow()
        new_id = f"{x_tenant_id}_res_{len(tenant_store)+1}"
        res = RPCResource(id=new_id, name=item.name, created_at=now, updated_at=now)
        tenant_store[new_id] = res
        created.append(res)
    response.headers["X-Batch-Count"] = str(len(created))
    return created


__all__ = ["router"]
