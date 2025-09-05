"""RPC-API über KEI-Bus (Request/Reply)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kei_logging import get_logger
from services.messaging.envelope import BusEnvelope
from services.messaging.naming import subject_for_rpc
from services.messaging.rpc import RPCError, rpc_request

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/bus/rpc", tags=["KEI-Bus-RPC"])


class RpcRequest(BaseModel):
    """Request für RPC-Aufruf über Bus."""

    service: str
    method: str
    version: int = 1
    tenant: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 5.0


@router.post("/invoke")
async def invoke_rpc(req: RpcRequest):
    """Führt RPC über Bus aus und liefert Antwort zurück."""
    try:
        result = await rpc_request(
            service=req.service,
            method=req.method,
            payload=req.payload,
            version=req.version,
            tenant=req.tenant,
            timeout_seconds=req.timeout_seconds,
        )
        return {"status": "ok", "result": result}
    except RPCError as re:
        # Standardisierte Fehlerantwort (HTTP 200 mit Fehlerobjekt, um Bus-Semantik widerzuspiegeln)
        return {"status": "error", "error": {"code": re.code, "message": re.message, "retryable": re.retryable, "details": re.details}}
    except TimeoutError:
        raise HTTPException(status_code=504, detail="rpc_timeout")
    except Exception as exc:
        logger.exception(f"RPC fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="rpc_failed")


class RpcEchoSubscribeRequest(BaseModel):
    """Request zum Starten eines Echo-RPC-Handlers (Test)."""

    service: str
    method: str
    version: int = 1
    tenant: str | None = None


@router.post("/subscribe_echo")
async def subscribe_echo(req: RpcEchoSubscribeRequest):
    """Registriert einen Echo-Handler, der eingehende RPC-Requests beantwortet."""
    try:
        from services.messaging.service import get_bus_service
        bus = get_bus_service()
        await bus.initialize()

        subject = subject_for_rpc(service=req.service, method=req.method, version=req.version, tenant=req.tenant)

        async def _handler(env: BusEnvelope) -> None:
            try:
                reply_to = env.payload.get("reply_to")
                if not reply_to:
                    return
                response = BusEnvelope(
                    type="rpc_response",
                    subject=reply_to,
                    tenant=req.tenant,
                    payload={"echo": env.payload.get("data")},
                    corr_id=env.corr_id or env.id,
                    causation_id=env.id,
                )
                await bus.publish(response)
            except Exception as handler_error:
                logger.exception(f"Fehler im Echo-RPC-Handler: {handler_error}")

        await bus.subscribe(subject, queue=None, handler=_handler)
        return {"status": "subscribed", "subject": subject}
    except Exception as exc:
        logger.exception(f"Echo-Subscription fehlgeschlagen: {exc}")
        raise HTTPException(status_code=500, detail="subscribe_echo_failed")
