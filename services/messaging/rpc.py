"""Request/Reply (RPC) über KEI-Bus."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from kei_logging import get_logger

from .config import bus_settings
from .envelope import BusEnvelope
from .metrics import inject_trace
from .naming import subject_for_rpc
from .service import get_bus_service

logger = get_logger(__name__)

# Konstanten für RPC
RPC_REQUEST_TYPE = "rpc_request"
RPC_REPLY_SUBJECT_PREFIX = "kei.rpc.reply"


class RPCError(Exception):
    """Standardisierte RPC-Fehler mit Katalogfeldern."""

    def __init__(self, code: str, message: str, retryable: bool = False, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


def _create_reply_subject() -> str:
    """Erstellt temporäres Reply-Subject."""
    return f"{RPC_REPLY_SUBJECT_PREFIX}.{uuid4().hex}"


def _create_reply_handler(future: asyncio.Future[dict[str, Any]]) -> Callable[[BusEnvelope], Awaitable[None]]:
    """Erstellt Reply-Handler für RPC-Response."""
    async def _on_reply(env: BusEnvelope) -> None:
        try:
            if not future.done():
                future.set_result(env.payload)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
    return _on_reply


def _create_rpc_envelope(
    service: str,
    method: str,
    payload: dict[str, Any],
    reply_subject: str,
    version: int = 1,
    tenant: str | None = None,
) -> BusEnvelope:
    """Erstellt RPC-Request-Envelope."""
    request_subject = subject_for_rpc(service=service, method=method, version=version, tenant=tenant)

    env = BusEnvelope(
        type=RPC_REQUEST_TYPE,
        subject=request_subject,
        payload={"data": payload, "reply_to": reply_subject},
        tenant=tenant,
    )
    # Trace Header hinzufügen
    env.headers = inject_trace(env.headers)
    return env


async def rpc_request(
    *,
    service: str,
    method: str,
    payload: dict[str, Any],
    version: int = 1,
    tenant: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Sendet RPC Request und wartet auf Antwort.

    Args:
        service: Service-Name
        method: Method-Name
        payload: Request-Payload
        version: API-Version
        tenant: Tenant-ID
        timeout_seconds: Timeout in Sekunden

    Returns:
        Response-Payload

    Raises:
        RPCError: Bei RPC-Fehlern oder Timeout
    """
    bus = get_bus_service()
    await bus.initialize()

    # Setup für RPC-Request/Reply
    reply_subject = _create_reply_subject()
    loop = asyncio.get_event_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()

    # Reply-Handler registrieren
    reply_handler = _create_reply_handler(future)
    await bus.subscribe(reply_subject, queue=None, handler=reply_handler)

    # Request-Envelope erstellen und senden
    envelope = _create_rpc_envelope(service, method, payload, reply_subject, version, tenant)
    await bus.publish(envelope)

    # Auf Antwort warten
    try:
        effective_timeout = timeout_seconds if timeout_seconds is not None else bus_settings.rpc.default_timeout_seconds
        return await asyncio.wait_for(future, timeout=effective_timeout)
    except TimeoutError:
        raise RPCError(
            code="RPC_TIMEOUT",
            message="RPC Timeout",
            retryable=True,
            details={"service": service, "method": method}
        )
