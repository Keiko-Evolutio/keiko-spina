# backend/websocket/router.py
"""WebSocket Router - Optimierte Routing-Logik."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from data_models.websocket import StatusUpdateEvent
from kei_logging import get_logger

from .manager import websocket_manager
from .utils import handle_ping_message, is_ping_event

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["websockets"])



# Hinweis: KEI-Stream hat eigene WS-Route unter `/stream/ws/{session_id}`

@router.websocket("/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str) -> None:
    """Universeller WebSocket-Endpoint für alle Verbindungen.
    Verwaltet Verbindung, Message-Routing und Fehlerbehandlung.
    """
    connection_id = await websocket_manager.connect(websocket, user_id)

    # Willkommensnachricht senden
    welcome_event = StatusUpdateEvent(
        status="connected",
        details=f"Verbunden als {user_id}",
        correlation_id=connection_id
    )
    await websocket_manager.send_to_connection(connection_id, welcome_event)

    try:
        while True:
            connection = websocket_manager.get_connection(connection_id)
            if not connection:
                break

            message_data = await connection.receive_message()
            if message_data is None:
                break

            await _handle_message(connection_id, message_data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket getrennt: {user_id}")
    except Exception as e:
        logger.exception(f"WebSocket Fehler für {user_id}: {e}")
    finally:
        await websocket_manager.disconnect(connection_id)


async def _handle_message(connection_id: str, data: dict[str, Any]) -> None:
    """Einheitliches Message-Handling für alle Verbindungen."""
    event_type = data.get("event_type", "unknown")

    # Ping-Pong für Keepalive mit korrekter Latenz-Messung
    if is_ping_event(event_type):
        await handle_ping_message(
            connection_id,
            data,
            websocket_manager.send_json_to_connection
        )
        return

    # An Event-Handler-System weiterleiten
    await websocket_manager.handle_message(connection_id, data)


@router.get("/stats")
async def get_websocket_stats() -> dict[str, Any]:
    """WebSocket Statistiken Endpoint."""
    return websocket_manager.get_stats()


# =====================================================================
# Health Check Endpoint
# =====================================================================

@router.get("/health")
async def websocket_health() -> dict[str, Any]:
    """WebSocket Service Health Check."""
    stats = websocket_manager.get_stats()

    return {
        "service": "websocket",
        "status": "healthy",
        "stats": stats,
        "timestamp": datetime.now(UTC).isoformat()
    }
