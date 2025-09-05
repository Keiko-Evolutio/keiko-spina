"""Hilfsfunktionen für WebSocket- und Mesh-Events in der n8n-Synchronisation."""

from __future__ import annotations

from agents.mesh.agent_event_bus import AgentEvent, AgentEventBus
from data_models.websocket import create_agent_response, create_status_update
from kei_logging import get_logger
from services.streaming import websocket_manager

logger = get_logger(__name__)


async def emit_status_update(
    *,
    connection_id: str | None,
    event_bus: AgentEventBus,
    execution_id: str,
    agent_id: str | None,
    status: str,
    progress: float | None = None,
    details: str | None = None,
) -> None:
    """Sendet Status an WebSocket und publiziert Mesh-Event."""
    if connection_id:
        try:
            await websocket_manager.send_json_to_connection(
                connection_id, create_status_update(status=status, progress=progress, details=details).model_dump()
            )
        except Exception as exc:
            logger.debug(f"WebSocket-Status-Update fehlgeschlagen ({connection_id}): {exc}")

    event_bus.publish(
        AgentEvent(
            event_type="n8n.status_update",
            payload={
                "execution_id": execution_id,
                "status": status,
                "progress": progress,
                "details": details,
                "agent_id": agent_id,
            },
            idempotency_key=execution_id,
        )
    )


async def emit_final_result(
    *,
    connection_id: str | None,
    event_bus: AgentEventBus,
    execution_id: str,
    agent_id: str | None,
    success: bool,
    result: dict,
) -> None:
    """Sendet Abschlussresultat an WebSocket und publiziert Mesh-Event."""
    if connection_id:
        try:
            await websocket_manager.send_json_to_connection(
                connection_id,
                create_agent_response(content=("Workflow erfolgreich" if success else "Workflow fehlgeschlagen"), is_final=True).model_dump(),
            )
        except Exception as exc:
            logger.debug(f"WebSocket-Abschluss fehlgeschlagen ({connection_id}): {exc}")

    event_bus.publish(
        AgentEvent(
            event_type="n8n.execution_finished",
            payload={
                "execution_id": execution_id,
                "success": success,
                "result": result,
                "agent_id": agent_id,
            },
            idempotency_key=execution_id,
        )
    )


__all__ = ["emit_final_result", "emit_status_update"]
