"""Asynchrone Polling-Helfer für n8n Workflow-Executions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from kei_logging import get_logger
from observability import trace_span

from .n8n_client import DEFAULT_POLL_INTERVAL, N8nClient

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)


async def run_polling_loop(
    execution_id: str,
    *,
    handle_result: Callable[[dict], Awaitable[bool]],
    sleep_seconds: float = DEFAULT_POLL_INTERVAL,
) -> None:
    """Führt Polling durch und ruft `handle_result` mit Rohdaten auf.

    `handle_result` soll True zurückgeben, wenn beendet werden soll.
    """
    client = N8nClient()
    try:
        while True:
            with trace_span("n8n.sync.poll_iteration", {"execution_id": execution_id}):
                res = await client.get_execution_status(execution_id)
                should_stop = await handle_result({
                    "status": res.status.value if hasattr(res.status, "value") else str(res.status),
                    "finished": bool(res.finished),
                    "raw": res.raw,
                })
                if should_stop:
                    break
            await asyncio.sleep(sleep_seconds)
    except Exception as exc:
        logger.warning(f"Polling Fehler für {execution_id}: {exc}")
    finally:
        await client.aclose()


__all__ = ["run_polling_loop"]
