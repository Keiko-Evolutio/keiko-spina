"""KEI-RPC gRPC Client mit dediziertem Circuit Breaker und Deadline-Propagation.

Dieser Client kapselt gRPC-Aufrufe an den Service `api.grpc.v1.KEIRPCService`.
Er unterstützt:

- Circuit Breaker pro Upstream-Service
- Deadline/Timeout-Propagation aus eingehenden HTTP-Requests
- Optionale TLS-Verbindung
- Authorization-Metadata (Bearer Token)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import grpc

from kei_logging import get_logger
from services.core.circuit_breaker import CircuitBreaker, CircuitPolicy
from services.core.constants import KEI_GRPC_CIRCUIT_BREAKER_CONFIG

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


def _load_stubs() -> tuple[Any, Any]:
    """Lädt gRPC Stubs dynamisch mit Fallback auf Code-Generierung.

    Returns:
        Tuple (rpc_pb2, rpc_pb2_grpc)
    """
    try:  # pragma: no cover - abhängig von protoc
        import rpc.proto.kei_rpc_pb2 as rpc_pb2  # type: ignore
        import rpc.proto.kei_rpc_pb2_grpc as rpc_pb2_grpc  # type: ignore
        return rpc_pb2, rpc_pb2_grpc
    except Exception:
        try:
            # Fallback: Protos zur Laufzeit generieren
            from rpc.proto.generate import generate_protos  # type: ignore

            if generate_protos():
                import rpc.proto.kei_rpc_pb2 as rpc_pb2  # type: ignore
                import rpc.proto.kei_rpc_pb2_grpc as rpc_pb2_grpc  # type: ignore
                return rpc_pb2, rpc_pb2_grpc
        except Exception as e:
            logger.warning(f"gRPC Protos konnten nicht generiert werden: {e}")
        # Dummies zurückgeben, um Import zu erlauben (Methoden würden dann fehlschlagen)
        class _Dummy:
            pass

        return _Dummy(), _Dummy()


@dataclass(slots=True)
class KEIRPCGRPCClientConfig:
    """Konfiguration für gRPC-Client."""

    target: str = "localhost:50051"
    api_token: str | None = None
    use_tls: bool = False
    root_cert_path: str | None = None
    timeout_seconds_default: float = 5.0
    tenant_id: str = "default"


class KEIRPCGRPCClient:
    """gRPC-Client für KEI-RPC Service mit Circuit Breaker und Deadlines."""

    def __init__(self, config: KEIRPCGRPCClientConfig) -> None:
        """Initialisiert den gRPC-Client.

        Args:
            config: Client-Konfiguration
        """
        self._config = config
        self._channel: grpc.aio.Channel | None = None
        self._stub: Any | None = None
        self._cb = CircuitBreaker(
            name=f"grpc:{config.target}",
            policy=CircuitPolicy(**KEI_GRPC_CIRCUIT_BREAKER_CONFIG),
        )

    async def _ensure_channel(self) -> None:
        """Stellt sicher, dass Channel/Stub vorhanden sind."""
        if self._channel is not None and self._stub is not None:
            return

        if self._config.use_tls:
            try:
                # credentials = None  # TODO: Verwende für erweiterte TLS-Konfiguration - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/110
                if self._config.root_cert_path:
                    with open(self._config.root_cert_path, "rb") as f:
                        root_certs = f.read()
                    credentials = grpc.ssl_channel_credentials(root_certificates=root_certs)
                else:
                    credentials = grpc.ssl_channel_credentials()
                self._channel = grpc.aio.secure_channel(self._config.target, credentials)
            except Exception as e:
                logger.warning(f"TLS-Channel Erstellung fehlgeschlagen, fallback auf insecure: {e}")
                self._channel = grpc.aio.insecure_channel(self._config.target)
        else:
            self._channel = grpc.aio.insecure_channel(self._config.target)

        rpc_pb2, rpc_pb2_grpc = _load_stubs()
        try:
            self._stub = rpc_pb2_grpc.KEIRPCServiceStub(self._channel)  # type: ignore[attr-defined]
        except Exception as e:
            logger.exception(f"Stub-Erstellung fehlgeschlagen: {e}")
            self._stub = None

    # ---------------------------------------------------------------------
    # Deadline-Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def compute_timeout_seconds(
        *,
        request: Any | None = None,
        explicit_timeout: float | None = None,
        default_timeout: float,
    ) -> float:
        """Berechnet Timeout für gRPC-Call mit optionaler Budget-Propagation.

        Regeln:
        - Wenn `explicit_timeout` gesetzt ist, wird dieser genutzt.
        - Wenn HTTP-Header `X-Request-Timeout-Ms` oder `X-Deadline-Ms` gesetzt sind,
          wird daraus ein Timeout abgeleitet.
        - Wenn `request.state.start_time` existiert und `X-Request-Budget-Ms` gesetzt ist,
          wird Rest-Budget berechnet.
        - Sonst `default_timeout`.
        """
        if explicit_timeout is not None and explicit_timeout > 0:
            return float(explicit_timeout)

        # Header-basierte Deadlines
        try:
            if request is not None and hasattr(request, "headers"):
                hdr = request.headers
                if "X-Request-Timeout-Ms" in hdr:
                    ms = max(0, int(hdr.get("X-Request-Timeout-Ms", "0")))
                    return max(0.001, ms / 1000.0)
                if "X-Deadline-Ms" in hdr:
                    ms = max(0, int(hdr.get("X-Deadline-Ms", "0")))
                    return max(0.001, ms / 1000.0)
                if "X-Request-Budget-Ms" in hdr and hasattr(request.state, "start_time"):
                    budget_ms = max(0, int(hdr.get("X-Request-Budget-Ms", "0")))
                    start = float(getattr(request.state, "start_time", time.time()))
                    elapsed_ms = int((time.time() - start) * 1000)
                    remaining_ms = max(0, budget_ms - elapsed_ms)
                    return max(0.001, remaining_ms / 1000.0)
        except Exception:
            pass

        return float(default_timeout)

    def _auth_metadata(self) -> tuple[tuple[str, str], ...]:
        """Erzeugt Authorization-Metadata falls Token konfiguriert."""
        md = []
        if self._config.api_token:
            md.append(("authorization", f"Bearer {self._config.api_token}"))
        # Tenant-Metadaten stets senden (Server verlangt Tenant-Isolation)
        if self._config.tenant_id:
            md.append(("x-tenant-id", self._config.tenant_id))
        return tuple(md)

    # ---------------------------------------------------------------------
    # RPC-Methoden (mit Circuit Breaker und Timeout)
    # ---------------------------------------------------------------------
    async def list_resources(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        q: str | None = None,
        sort: str = "-updated_at",
        request: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Listet Ressourcen via gRPC mit Paginierung.

        Returns:
            Dictionary mit `items` und `pagination` (serialisiert)
        """
        await self._ensure_channel()
        rpc_pb2, _ = _load_stubs()

        params = {
            "page": page,
            "per_page": per_page,
            "q": q or "",
            "sort": sort,
        }
        msg = rpc_pb2.ListResourcesRequest(**params)  # type: ignore[attr-defined]
        metadata = self._auth_metadata()
        to = self.compute_timeout_seconds(
            request=request, explicit_timeout=timeout_seconds, default_timeout=self._config.timeout_seconds_default
        )

        async def _call() -> dict[str, Any]:
            assert self._stub is not None
            resp = await self._stub.ListResources(msg, timeout=to, metadata=metadata)  # type: ignore[attr-defined]
            # Serialisierung in Dict
            items = [
                {"id": it.id, "name": it.name, "created_at": it.created_at, "updated_at": it.updated_at}
                for it in getattr(resp, "items", [])
            ]
            pag = getattr(resp, "pagination", None)
            pagination = {"page": pag.page, "per_page": pag.per_page, "total": pag.total} if pag else {}
            return {"items": items, "pagination": pagination}

        return await self._cb.call(_call)

    async def create_resource(
        self,
        name: str,
        *,
        idempotency_key: str | None = None,
        request: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Erstellt Ressource via gRPC mit optionalem Idempotenzschlüssel."""
        await self._ensure_channel()
        rpc_pb2, _ = _load_stubs()
        msg = rpc_pb2.CreateResourceRequest(name=name, idempotency_key=idempotency_key or "")  # type: ignore[attr-defined]
        metadata = self._auth_metadata()
        to = self.compute_timeout_seconds(
            request=request, explicit_timeout=timeout_seconds, default_timeout=self._config.timeout_seconds_default
        )

        async def _call() -> dict[str, Any]:
            assert self._stub is not None
            res = await self._stub.CreateResource(msg, timeout=to, metadata=metadata)  # type: ignore[attr-defined]
            return {"id": res.id, "name": res.name, "created_at": res.created_at, "updated_at": res.updated_at}

        return await self._cb.call(_call)

    async def get_resource(
        self,
        resource_id: str,
        *,
        request: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Liest Ressource via gRPC."""
        await self._ensure_channel()
        rpc_pb2, _ = _load_stubs()
        msg = rpc_pb2.GetResourceRequest(id=resource_id)  # type: ignore[attr-defined]
        metadata = self._auth_metadata()
        to = self.compute_timeout_seconds(
            request=request, explicit_timeout=timeout_seconds, default_timeout=self._config.timeout_seconds_default
        )

        async def _call() -> dict[str, Any]:
            assert self._stub is not None
            res = await self._stub.GetResource(msg, timeout=to, metadata=metadata)  # type: ignore[attr-defined]
            return {"id": res.id, "name": res.name, "created_at": res.created_at, "updated_at": res.updated_at}

        return await self._cb.call(_call)

    async def patch_resource(
        self,
        resource_id: str,
        *,
        name: str | None = None,
        _if_match: str | None = None,  # aktuell nicht im proto ausgewertet, Platzhalter
        request: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Teil-Update einer Ressource via gRPC."""
        await self._ensure_channel()
        rpc_pb2, _ = _load_stubs()
        msg = rpc_pb2.PatchResourceRequest(id=resource_id, name=name or "")  # type: ignore[attr-defined]
        metadata = self._auth_metadata()
        to = self.compute_timeout_seconds(
            request=request, explicit_timeout=timeout_seconds, default_timeout=self._config.timeout_seconds_default
        )

        async def _call() -> dict[str, Any]:
            assert self._stub is not None
            res = await self._stub.PatchResource(msg, timeout=to, metadata=metadata)  # type: ignore[attr-defined]
            return {"id": res.id, "name": res.name, "created_at": res.created_at, "updated_at": res.updated_at}

        return await self._cb.call(_call)

    async def stream_operations(
        self,
        messages: AsyncIterator[dict[str, Any]],
        *,
        request: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Bidirektionales Streaming von Operationen."""
        await self._ensure_channel()
        rpc_pb2, _ = _load_stubs()
        metadata = self._auth_metadata()
        to = self.compute_timeout_seconds(
            request=request, explicit_timeout=timeout_seconds, default_timeout=self._config.timeout_seconds_default
        )

        async def _gen():
            async for m in messages:
                yield rpc_pb2.StreamMessage(
                    id=str(m.get("id", "")),
                    type=str(m.get("type", "request")),
                    operation=str(m.get("operation", "noop")),
                    payload_json=str(m.get("payload_json", "{}")),
                )

        assert self._stub is not None

        # Circuit Breaker um Stream-Aufbau, nicht jede Nachricht
        async def _call() -> AsyncIterator[dict[str, Any]]:
            call = self._stub.StreamOperations(_gen(), timeout=to, metadata=metadata)  # type: ignore[attr-defined]
            async for resp in call:
                yield {"id": resp.id, "type": resp.type, "operation": resp.operation, "payload_json": resp.payload_json}

        # Der CB erwartet eine Coroutine, daher kapseln wir die Iteration
        queue: asyncio.Queue = asyncio.Queue()

        async def _runner():
            try:
                async for item in _call():
                    await queue.put((True, item))
            except Exception as e:
                await queue.put((False, e))
            finally:
                await queue.put((None, None))

        async def _cb_wrapper() -> None:
            await self._cb.call(lambda: _runner())

        asyncio.create_task(_cb_wrapper())

        while True:
            ok, value = await queue.get()
            if ok is True:
                yield value
            elif ok is False:
                raise value
            else:
                break

    async def close(self) -> None:
        """Schließt den Channel."""
        try:
            if self._channel:
                await self._channel.close()
        except Exception:
            pass


__all__ = ["KEIRPCGRPCClient", "KEIRPCGRPCClientConfig"]
