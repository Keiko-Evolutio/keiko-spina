# backend/websocket/handlers.py
"""WebSocket Event-Handler."""

from typing import Any

from data_models.websocket import (
    AgentResponseEvent,
    EventType,
    StatusUpdateEvent,
)
from kei_logging import get_logger

from .manager import websocket_manager
from .system_heartbeat_handler import handle_system_heartbeat_subscription

logger = get_logger(__name__)


# =====================================================================
# Event Handlers
# =====================================================================

async def handle_voice_input(connection_id: str, message_data: dict[str, Any]) -> None:
    """Handler für Voice-Input-Events."""
    logger.info(f"Spracheingabe empfangen von {connection_id}")

    # Status-Update senden
    status_event = StatusUpdateEvent(
        status="processing_voice",
        progress=0.0,
        details="Verarbeite Spracheingabe...",
        correlation_id=message_data.get("correlation_id")
    )
    await websocket_manager.send_to_connection(connection_id, status_event)


async def handle_function_confirmation(connection_id: str, message_data: dict[str, Any]) -> None:
    """Handler für Funktions-Bestätigungen."""
    function_call_id = message_data.get("function_call_id")
    confirmed = message_data.get("confirmed", False)

    logger.info(
        f"Funktion {function_call_id} {'bestätigt' if confirmed else 'abgelehnt'} "
        f"von {connection_id}"
    )

    # Bestätigungsantwort senden
    response_event = AgentResponseEvent(
        content=f"Funktionsaufruf wurde {'bestätigt' if confirmed else 'abgelehnt'}.",
        is_final=True,
        correlation_id=message_data.get("correlation_id")
    )
    await websocket_manager.send_to_connection(connection_id, response_event)


async def handle_system_heartbeat_message(connection_id: str, message_data: dict[str, Any]) -> None:
    """Handler für System Heartbeat Subscription-Nachrichten."""
    await handle_system_heartbeat_subscription(connection_id, message_data)


def register_default_handlers() -> None:
    """Registriert Standard-Event-Handler."""
    websocket_manager.add_event_handler(EventType.VOICE_INPUT, handle_voice_input)
    websocket_manager.add_event_handler(EventType.FUNCTION_CALL, handle_function_confirmation)
    websocket_manager.add_event_handler(EventType.SYSTEM_HEARTBEAT, handle_system_heartbeat_message)
    logger.info("Standard Event-Handler registriert")
