"""Kompatibler WorkflowSyncManager zur Verwaltung von n8n-Polling-Kontexten.

Dieser Manager wird von bestehenden Tests verwendet und stellt eine einfache
API zum Starten/Stoppen sowie Pause/Resume bereit. Intern wird auf den
`N8nClient` und das Polling zurückgegriffen.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger
from observability import trace_function, trace_span

from .n8n_client import N8nClient
from .sync_models import SyncContext

logger = get_logger(__name__)


@dataclass(slots=True)
class _PollingTask:
    """Interner Container für einen laufenden Polling-Task."""

    task: asyncio.Task[None]
    stop_event: asyncio.Event


class WorkflowSyncManager:
    """Verwaltet n8n-Execution-Polling und Zustände pro Execution-ID."""

    def __init__(self) -> None:
        self._contexts: dict[str, SyncContext] = {}
        self._tasks: dict[str, _PollingTask] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @trace_function("n8n.sync_mgr.start")
    async def start(self, execution_id: str, *, connection_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        """Startet Polling für Execution-ID."""
        if execution_id in self._contexts:
            return {"ack": True, "already_started": True}

        ctx = SyncContext(execution_id=execution_id, connection_id=connection_id, agent_id=agent_id, started_at=datetime.now(UTC))
        self._contexts[execution_id] = ctx
        self._start_polling(execution_id)
        return {"ack": True}

    @trace_function("n8n.sync_mgr.stop")
    async def stop(self, execution_id: str) -> dict[str, Any]:
        """Stoppt Polling und entfernt Kontext."""
        task = self._tasks.pop(execution_id, None)
        if task:
            task.stop_event.set()
            with contextlib.suppress(Exception):
                await task.task
        self._contexts.pop(execution_id, None)
        return {"ack": True}

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    async def start_workflow(self, **kwargs) -> dict[str, Any]:
        """Kompatibilitätsmethode für Tests: startet Dummy-Workflow.

        Diese Methode wird in Tests via monkeypatch ersetzt. Hier liefern wir
        eine minimale Default-Implementierung zurück.
        """
        exec_id = kwargs.get("execution_id") or "E_TEST"
        await self.start(exec_id)
        return {"execution_id": exec_id, "started": True}
    @trace_function("n8n.sync_mgr.pause")
    async def pause(self, execution_id: str) -> dict[str, Any]:
        """Pausiert Polling (Flag im Kontext)."""
        ctx = self._contexts.get(execution_id)
        if not ctx:
            return {"ack": False, "reason": "not_found"}
        ctx.paused = True
        return {"ack": True}

    @trace_function("n8n.sync_mgr.resume")
    async def resume(self, execution_id: str) -> dict[str, Any]:
        """Setzt Polling fort (Flag im Kontext)."""
        ctx = self._contexts.get(execution_id)
        if not ctx:
            return {"ack": False, "reason": "not_found"}
        ctx.paused = False
        return {"ack": True}

    @trace_function("n8n.sync_mgr.handle_callback")
    async def handle_callback(self, execution_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Behandelt Callback-Result und bereinigt bei Abschluss."""
        finished = bool(payload.get("finished"))
        if finished:
            await self.stop(execution_id)
        return {"ack": True}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _start_polling(self, execution_id: str) -> None:
        """Startet Hintergrund-Polling-Task."""
        stop_event = asyncio.Event()

        async def _runner() -> None:
            client = N8nClient()
            try:
                while not stop_event.is_set():
                    with trace_span("n8n.sync_mgr.poll_iter", {"execution_id": execution_id}):
                        res = await client.get_execution_status(execution_id)
                        ctx = self._contexts.get(execution_id)
                        if ctx and not ctx.paused and getattr(res, "finished", False):
                            break
                    await asyncio.sleep(0.1)
            except Exception as exc:
                logger.debug(f"Polling-Fehler: {exc}")
            finally:
                await client.aclose()

        task = asyncio.create_task(_runner())
        self._tasks[execution_id] = _PollingTask(task=task, stop_event=stop_event)


# Singleton
workflow_sync_manager = WorkflowSyncManager()


__all__ = ["WorkflowSyncManager", "workflow_sync_manager"]
