"""KEI-RPC gRPC Servicer Implementierung (v1).

Implementiert CRUD/Batch-Operationen und einen bidi-Streaming-Handler
für Long-running/Realtime-Aufgaben. Nutzt eine einfache In-Memory-Storage,
analog zur REST-Demo in `api/routes/rpc_routes.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import grpc
from google.protobuf.any_pb2 import Any as AnyProto  # type: ignore
from google.rpc import error_details_pb2, status_pb2  # type: ignore

from kei_logging import get_logger

# Logger früh initialisieren für Import-Fehlerbehandlung
logger = get_logger(__name__)

if TYPE_CHECKING:
    import builtins
    from collections.abc import AsyncIterator

try:
    from grpc_status import rpc_status  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    rpc_status = None  # type: ignore

# Die generierten Klassen werden zur Laufzeit erwartet. Falls Codegen
# noch nicht ausgeführt ist, importiert dieser Servicer nicht.
rpc_pb2 = None
rpc_pb2_grpc = None

try:  # pragma: no cover - abhängig von protoc
    import rpc.proto.kei_rpc_pb2 as rpc_pb2  # type: ignore
    import rpc.proto.kei_rpc_pb2_grpc as rpc_pb2_grpc  # type: ignore
except (ImportError, ModuleNotFoundError) as _imp_err:  # pragma: no cover
    # Fallback-Dummies, damit Import gelingt, bis Runtime-Codegen erfolgt
    logger.debug(f"gRPC Proto-Module nicht verfügbar: {_imp_err}")
    rpc_pb2 = None
    rpc_pb2_grpc = None
except Exception as _imp_err:  # pragma: no cover
    # Unerwartete Import-Fehler
    logger.warning(f"Unerwarteter Fehler beim Import der gRPC Proto-Module: {_imp_err}")
    rpc_pb2 = None
    rpc_pb2_grpc = None

# Fallback-Dummies, damit Import gelingt, bis Runtime-Codegen erfolgt
if rpc_pb2 is None or rpc_pb2_grpc is None:
    class _DummyRPCModule:  # type: ignore
        class Resource:
            """Dummy Placeholder."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class Pagination:
            """Dummy Placeholder."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class ListResourcesResponse:
            """Dummy Placeholder."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class StreamMessage:
            """Dummy Placeholder."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

    class _DummyRPCModuleGRPC:  # type: ignore
        class KEIRPCServiceServicer:
            """Dummy Base Servicer."""

    rpc_pb2 = _DummyRPCModule()  # type: ignore
    rpc_pb2_grpc = _DummyRPCModuleGRPC()  # type: ignore


@dataclass
class _Resource:
    """Interne Repräsentation einer Ressource."""

    id: str
    name: str
    created_at: str
    updated_at: str


class _InMemoryStore:
    """Einfacher In-Memory Store analog zu REST-Routen."""

    def __init__(self) -> None:
        # Pro-Tenant Storage
        self.items_by_tenant: dict[str, dict[str, _Resource]] = {}

    def list(
        self, tenant_id: str, page: int, per_page: int, q: str | None, sort: str
    ) -> tuple[builtins.list[_Resource], int]:
        data = list(self.items_by_tenant.get(tenant_id, {}).values())
        if q:
            data = [d for d in data if q.lower() in d.name.lower()]
        reverse = sort.startswith("-")
        key = sort.lstrip("-") or "updated_at"
        data.sort(key=lambda d: getattr(d, key), reverse=reverse)
        total = len(data)
        start = (page - 1) * per_page
        end = start + per_page
        return data[start:end], total

    def create(self, tenant_id: str, name: str, idem: str | None) -> _Resource:
        now = datetime.now(UTC).isoformat()
        store = self.items_by_tenant.setdefault(tenant_id, {})
        new_id = idem or f"{tenant_id}_res_{len(store)+1}"
        if new_id in store:
            return store[new_id]
        obj = _Resource(id=new_id, name=name, created_at=now, updated_at=now)
        store[new_id] = obj
        return obj

    def get(self, tenant_id: str, item_id: str) -> _Resource | None:
        return self.items_by_tenant.get(tenant_id, {}).get(item_id)

    def patch(self, tenant_id: str, item_id: str, name: str | None) -> _Resource | None:
        store = self.items_by_tenant.get(tenant_id, {})
        obj = store.get(item_id)
        if not obj:
            return None
        if name is not None:
            obj.name = name
            obj.updated_at = datetime.now(UTC).isoformat()
        return obj


_STORE = _InMemoryStore()


def _to_proto(res: _Resource) -> Any:
    """Konvertiert interne Ressource in Proto Resource."""
    return rpc_pb2.Resource(
        id=res.id,
        name=res.name,
        created_at=res.created_at,
        updated_at=res.updated_at,
    )


def _get_metadata_value(context: grpc.aio.ServicerContext, key: str) -> str | None:
    """Liest Metadata schlüsselunabhängig aus dem Kontext."""
    try:
        md = context.invocation_metadata() or []
        lower = key.lower()
        for pair in md:
            if pair.key.lower() == lower:
                return pair.value
    except (AttributeError, TypeError, ValueError):
        return None
    return None


async def _require_tenant(context: grpc.aio.ServicerContext) -> str:
    """Erzwingt `x-tenant-id` Metadata und liefert deren Wert zurück."""
    tenant_id = _get_metadata_value(context, "x-tenant-id")
    if not tenant_id:
        # Rich Status Fehler konstruieren
        if rpc_status is not None:
            st = status_pb2.Status(
                code=grpc.StatusCode.INVALID_ARGUMENT.value[0],
                message="Missing required metadata: x-tenant-id",
                details=[
                    AnyProto.Pack(
                        error_details_pb2.ErrorInfo(
                            reason="TENANT_REQUIRED",
                            domain="kei.rpc",
                            metadata={"missing": "x-tenant-id"},
                        )
                    )
                ],
            )
            if rpc_status is not None and hasattr(context, "abort_with_status"):
                context.abort_with_status(rpc_status.to_status(st))
            else:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Missing required metadata: x-tenant-id"
                )
        else:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Missing required metadata: x-tenant-id"
            )
    return tenant_id  # type: ignore[return-value]


def _make_etag(res: _Resource) -> str:
    """Berechnet schwaches ETag analog REST-Implementierung."""
    base = f"{res.id}:{res.updated_at}"
    # einfache Hash-Repräsentation
    return f'W/"{abs(hash(base)) & 0xFFFFFFFF:x}"'


# Sichere Basis-Klasse für KEIRPCService
_ServicerBase = getattr(rpc_pb2_grpc, "KEIRPCServiceServicer", object) if rpc_pb2_grpc else object

class KEIRPCService(_ServicerBase):  # type: ignore[misc]
    """gRPC Servicer für KEI-RPC."""

    @staticmethod
    async def list_resources(
        request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        """Listet Ressourcen mit Pagination/Filter/Sort."""
        # Optional: einfache Scope-Prüfung (nur wenn Scopes vorhanden)
        try:
            scopes = (_get_metadata_value(context, "x-scopes") or "").split()
            if scopes and "kei.rpc.read" not in scopes:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Missing required scope: kei.rpc.read"
                )
        except (AttributeError, ValueError, TypeError):
            pass
        tenant_id = await _require_tenant(context)
        page = request.page or 1
        per_page = request.per_page or 20
        q = request.q or None
        sort = request.sort or "-updated_at"
        items, total = _STORE.list(tenant_id, page, per_page, q, sort)
        return rpc_pb2.ListResourcesResponse(
            items=[_to_proto(it) for it in items],
            pagination=rpc_pb2.Pagination(page=page, per_page=per_page, total=total),
        )

    @staticmethod
    async def create_resource(
        request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        """Erstellt Ressource mit einfacher Idempotenz."""
        try:
            scopes = (_get_metadata_value(context, "x-scopes") or "").split()
            if scopes and "kei.rpc.write" not in scopes:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Missing required scope: kei.rpc.write"
                )
        except (AttributeError, ValueError, TypeError):
            pass
        tenant_id = await _require_tenant(context)
        name = request.name
        idem = request.idempotency_key or None
        res = _STORE.create(tenant_id, name, idem)
        # ETag als Trailing-Metadata zurückgeben
        with contextlib.suppress(Exception):
            context.set_trailing_metadata((("etag", _make_etag(res)),))
        return _to_proto(res)

    @staticmethod
    async def get_resource(
        request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        """Liest Ressource; 404 bei Nichtfinden."""
        # Scope-Prüfung für READ (analog List)
        try:
            scopes = (_get_metadata_value(context, "x-scopes") or "").split()
            if scopes and "kei.rpc.read" not in scopes:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Missing required scope: kei.rpc.read"
                )
        except (AttributeError, ValueError) as e:
            logger.debug(f"Fehler beim Parsen der Scopes für GetResource: {e}")
        except Exception as e:
            logger.warning(f"Unerwarteter Fehler bei Scope-Prüfung für GetResource: {e}")
        tenant_id = await _require_tenant(context)
        res = _STORE.get(tenant_id, request.id)
        if not res:
            # Rich Status NOT_FOUND
            if rpc_status is not None:
                st = status_pb2.Status(
                    code=grpc.StatusCode.NOT_FOUND.value[0],
                    message="Resource not found",
                    details=[
                        AnyProto.Pack(
                            error_details_pb2.ResourceInfo(
                                resource_type="Resource", resource_name=request.id
                            )
                        )
                    ],
                )
                if rpc_status is not None and hasattr(context, "abort_with_status"):
                    context.abort_with_status(rpc_status.to_status(st))
                else:
                    await context.abort(grpc.StatusCode.NOT_FOUND, "Resource not found")
            else:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Resource not found")
        # ETag anhängen
        with contextlib.suppress(Exception):
            context.set_trailing_metadata((("etag", _make_etag(res)),))
        return _to_proto(res)  # type: ignore[arg-type]

    @staticmethod
    async def patch_resource(
        request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        """Teil-Update der Ressource."""
        try:
            scopes = (_get_metadata_value(context, "x-scopes") or "").split()
            if scopes and "kei.rpc.write" not in scopes:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Missing required scope: kei.rpc.write"
                )
        except (AttributeError, ValueError, TypeError):
            pass
        tenant_id = await _require_tenant(context)
        # If-Match ETag prüfen
        current = _STORE.get(tenant_id, request.id)
        if not current:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Resource not found")
        current_etag = _make_etag(current)  # type: ignore[arg-type]
        if getattr(request, "if_match", None) and request.if_match != current_etag:
            if rpc_status is not None:
                st = status_pb2.Status(
                    code=grpc.StatusCode.FAILED_PRECONDITION.value[0],
                    message="ETag mismatch",
                    details=[
                        AnyProto.Pack(
                            error_details_pb2.PreconditionFailure(
                                violations=[
                                    error_details_pb2.PreconditionFailure.Violation(
                                        type="ETAG",
                                        subject=request.id,
                                        description="If-Match does not match current ETag",
                                    )
                                ]
                            )
                        )
                    ],
                )
                if rpc_status is not None and hasattr(context, "abort_with_status"):
                    context.abort_with_status(rpc_status.to_status(st))
                else:
                    await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "ETag mismatch")
            else:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "ETag mismatch")

        res = _STORE.patch(tenant_id, request.id, request.name or None)
        if not res:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Resource not found")
        # Optional: If-Match/ETag könnte in Metadata geprüft werden
        with contextlib.suppress(Exception):
            context.set_trailing_metadata((("etag", _make_etag(res)),))
        return _to_proto(res)  # type: ignore[arg-type]

    @staticmethod
    async def batch_create(
        request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        """Batch-Erstellung von Ressourcen."""
        created: list[_Resource] = []
        tenant_id = await _require_tenant(context)
        for item in request.items:
            res = _STORE.create(tenant_id, item.name, idem=None)
            created.append(res)
        return rpc_pb2.ListResourcesResponse(
            items=[_to_proto(it) for it in created],
            pagination=rpc_pb2.Pagination(page=1, per_page=len(created), total=len(created)),
        )

    @staticmethod
    async def stream_operations(
        request_iterator: AsyncIterator[Any],
        _: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        """Einfacher bidi-Streaming-Handler mit Flow-Control.

        Echo-artige Umsetzung mit Status-Updates. In einer echten Implementierung
        würden Operationen orchestriert und Zwischenergebnisse gestreamt.
        """
        try:
            # Einfache kreditorientierte Flow-Control via Semaphore
            max_inflight = 10
            sem = asyncio.Semaphore(max_inflight)

            async def _process(one_msg: Any) -> list[Any]:
                # Simulierte Verarbeitung
                status_msg = rpc_pb2.StreamMessage(
                    id=one_msg.id or "",
                    type="status",
                    operation=one_msg.operation or "noop",
                    payload_json='{"status":"processing"}',
                )
                result_msg = rpc_pb2.StreamMessage(
                    id=one_msg.id or "",
                    type="result",
                    operation=one_msg.operation or "noop",
                    payload_json=one_msg.payload_json or "{}",
                )
                return [status_msg, result_msg]

            async for msg in request_iterator:
                await sem.acquire()
                try:
                    # Ergebnisse sofort streamen, nicht puffern
                    for out in await _process(msg):
                        yield out
                finally:
                    sem.release()
        except asyncio.CancelledError:  # type: ignore[name-defined]  # pragma: no cover
            # Client hat Stream beendet
            return

    # ==============================
    # Agent Operations (plan/act/observe/explain)
    # ==============================

    async def plan(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Erstellt einen Plan über die bestehende Agent-Ausführungsschicht."""
        return await self._agent_operation(
            context,
            op="plan",
            description_prefix="PLAN:",
            text=request.objective,
            context_json=getattr(request, "context_json", None),
        )

    async def act(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Führt eine Aktion aus."""
        return await self._agent_operation(
            context,
            op="act",
            description_prefix="ACT:",
            text=request.action,
            context_json=getattr(request, "context_json", None),
        )

    async def observe(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Nimmt Beobachtungen auf."""
        return await self._agent_operation(
            context,
            op="observe",
            description_prefix="OBSERVE:",
            text=request.observation,
            context_json=getattr(request, "context_json", None),
        )

    async def explain(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Erklärt ein Thema/Ergebnis."""
        return await self._agent_operation(
            context,
            op="explain",
            description_prefix="EXPLAIN:",
            text=request.topic,
            context_json=getattr(request, "context_json", None),
        )

    @staticmethod
    async def _agent_operation(
        context: grpc.aio.ServicerContext,
        *,
        op: str,
        description_prefix: str,
        text: str,
        context_json: str | None,
    ) -> Any:
        """Hilfsfunktion: Führt die Operation über REST-Logik aus und mappt auf Proto."""
        # Optional: Scope-Prüfung
        try:
            scopes = (_get_metadata_value(context, "x-scopes") or "").split()
            required = f"kei.rpc.{op}"
            if scopes and required not in scopes:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, f"Missing required scope: {required}"
                )
        except (AttributeError, ValueError) as e:
            logger.debug(f"Fehler beim Parsen der Scopes für Agent-Operation {op}: {e}")
        except Exception as e:
            logger.warning(f"Unerwarteter Fehler bei Scope-Prüfung für Agent-Operation {op}: {e}")

        started = datetime.now(UTC).isoformat()
        # REST-Hilfslogik wiederverwenden (AgentExecutionRequest)
        try:
            import json as _json

            from api.routes.agents_routes import (
                AgentExecutionRequest,  # type: ignore
                execute_agent_safely,
            )

            ctx: dict[str, Any] | None = None
            if context_json:
                try:
                    ctx = _json.loads(context_json)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Fehler beim Parsen des Context-JSON für Agent-Operation {op}: {e}")
                    ctx = {"_raw": context_json}
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Unerwarteter Fehler beim JSON-Parsing für Agent-Operation {op}: {e}")
                    ctx = {"_raw": context_json}
            req = AgentExecutionRequest(
                task_description=f"{description_prefix}{text}",
                context=ctx,
                parameters={"operation": op},
                timeout=60,
            )
            res = await execute_agent_safely(req)
            completed = res.completed_at.isoformat() if getattr(res, "completed_at", None) else ""
            # Build proto response
            return rpc_pb2.OperationResponse(  # type: ignore[attr-defined]
                operation=op,
                status=res.status,
                started_at=started,
                completed_at=completed,
                result_json=(
                    _json.dumps(res.result) if getattr(res, "result", None) is not None else "{}"
                ),
                error=res.error or "",
            )
        except (ValueError, TypeError, RuntimeError) as e:
            # Fehlerfall
            try:
                return rpc_pb2.OperationResponse(  # type: ignore[attr-defined]
                    operation=op,
                    status="error",
                    started_at=started,
                    completed_at=datetime.now(UTC).isoformat(),
                    result_json="{}",
                    error=str(e),
                )
            except (ImportError, AttributeError) as proto_err:
                # Wenn Protos noch nicht generiert, generischer Abort
                logger.debug(f"Proto-Klassen nicht verfügbar für Agent-Operation {op}: {proto_err}")
            except (ValueError, TypeError) as proto_err:
                # Unerwarteter Fehler beim Erstellen der Proto-Response
                logger.warning(f"Unerwarteter Fehler beim Erstellen der Proto-Response für {op}: {proto_err}")
                # Wenn Protos noch nicht generiert, generischer Abort
                from core.exceptions import AgentError

                exc = AgentError(
                    "Agent-Operation fehlgeschlagen",
                    details={"op": op, "reason": str(e)},
                    severity="HIGH",
                )
                # gRPC Mapping Interceptor fängt KeikoException ab; als Fallback: INTERNAL
                await context.abort(grpc.StatusCode.INTERNAL, exc.message)


__all__ = ["KEIRPCService"]
