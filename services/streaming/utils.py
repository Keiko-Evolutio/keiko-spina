# backend/websocket/utils.py
"""WebSocket Utility-Funktionen für gemeinsame Patterns."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

# Konstanten für Ping-Events
PING_EVENT_TYPES: set[str] = {"ping", "heartbeat"}
PONG_EVENT_TYPE = "pong"


def create_server_timestamp() -> str:
    """Erstellt einen standardisierten Server-Timestamp im ISO-Format.

    Returns:
        ISO-formatierter UTC-Timestamp
    """
    return datetime.now(UTC).isoformat()


def create_pong_response(
    client_time: str | None = None,
    event_type: str = PONG_EVENT_TYPE,
    include_server_timestamp: bool = True
) -> dict[str, Any]:
    """Erstellt eine standardisierte Pong-Response.

    Args:
        client_time: Ursprüngliche client_time aus der Ping-Nachricht für Latenz-Messung
        event_type: Event-Type für die Response (default: "pong")
        include_server_timestamp: Ob server_timestamp hinzugefügt werden soll

    Returns:
        Dictionary mit Pong-Response-Daten
    """
    response: dict[str, Any] = {
        "event_type": event_type
    }

    if include_server_timestamp:
        response["server_timestamp"] = create_server_timestamp()

    # Gebe client_time zurück für Round-Trip-Latenz-Messung
    if client_time:
        response["client_time"] = client_time

    return response


def is_ping_event(event_type: str) -> bool:
    """Prüft ob ein Event-Type ein Ping/Heartbeat-Event ist.

    Args:
        event_type: Der zu prüfende Event-Type

    Returns:
        True wenn es sich um ein Ping-Event handelt
    """
    return event_type in PING_EVENT_TYPES


def extract_client_time(data: dict[str, Any]) -> str | None:
    """Extrahiert client_time aus einer Ping-Nachricht.

    Args:
        data: Die Nachrichtendaten

    Returns:
        client_time wenn vorhanden, sonst None
    """
    return data.get("client_time")


async def handle_ping_message(
    connection_id: str,
    data: dict[str, Any],
    send_function,
    log_debug: bool = True
) -> None:
    """Behandelt eine Ping-Nachricht und sendet entsprechende Pong-Response.

    Args:
        connection_id: ID der WebSocket-Verbindung
        data: Die Ping-Nachrichtendaten
        send_function: Async-Funktion zum Senden der Response (connection_id, response_data)
        log_debug: Ob Debug-Logs ausgegeben werden sollen
    """
    client_time = extract_client_time(data)

    # Erstelle Pong-Response
    response = create_pong_response(client_time)

    # Debug-Logging
    if log_debug and client_time:
        logger.debug(f"Ping-Pong: client_time={client_time}, connection={connection_id}")

    # Sende Response
    await send_function(connection_id, response)


def create_heartbeat_response(
    original_timestamp: str | None = None,
    response_type: str = "heartbeat"
) -> dict[str, Any]:
    """Erstellt eine Heartbeat-Response für KEI-Stream.

    Args:
        original_timestamp: Ursprünglicher Timestamp aus der Heartbeat-Nachricht
        response_type: Response-Type (default: "heartbeat")

    Returns:
        Dictionary mit Heartbeat-Response-Daten
    """
    response: dict[str, Any] = {
        "type": response_type
    }

    if original_timestamp:
        response["ts"] = original_timestamp
    else:
        response["ts"] = create_server_timestamp()

    return response


async def handle_heartbeat_message(
    connection_id: str,
    frame_data: dict[str, Any],
    send_function
) -> None:
    """Behandelt eine KEI-Stream Heartbeat-Nachricht.

    Args:
        connection_id: ID der WebSocket-Verbindung
        frame_data: Die Heartbeat-Frame-Daten
        send_function: Async-Funktion zum Senden der Response
    """
    original_ts = frame_data.get("ts")
    response = create_heartbeat_response(original_ts)
    await send_function(connection_id, response)


# Event-Handler-Abstraktion
EventHandlerFunction = Callable[[str, dict[str, Any]], Any]


class BaseEventHandler(ABC):
    """Basis-Klasse für Event-Handler."""

    @abstractmethod
    async def handle(self, connection_id: str, data: dict[str, Any]) -> None:
        """Behandelt ein Event.

        Args:
            connection_id: ID der WebSocket-Verbindung
            data: Event-Daten
        """

    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Prüft ob dieser Handler das Event behandeln kann.

        Args:
            event_type: Der Event-Type

        Returns:
            True wenn der Handler das Event behandeln kann
        """


class FunctionEventHandler(BaseEventHandler):
    """Event-Handler der eine Funktion kapselt."""

    def __init__(self, event_types: str | set[str], handler_func: EventHandlerFunction):
        """Initialisiert den Handler.

        Args:
            event_types: Event-Type(s) die dieser Handler behandelt
            handler_func: Die Handler-Funktion
        """
        if isinstance(event_types, str):
            self.event_types = {event_types}
        else:
            self.event_types = set(event_types)
        self.handler_func = handler_func

    def can_handle(self, event_type: str) -> bool:
        """Prüft ob dieser Handler das Event behandeln kann."""
        return event_type in self.event_types

    async def handle(self, connection_id: str, data: dict[str, Any]) -> None:
        """Führt die Handler-Funktion aus."""
        if asyncio.iscoroutinefunction(self.handler_func):
            await self.handler_func(connection_id, data)
        else:
            self.handler_func(connection_id, data)


class EventRouter:
    """Zentraler Event-Router für einheitliches Event-Handling."""

    def __init__(self):
        self.handlers: list[BaseEventHandler] = []
        self.fallback_handler: EventHandlerFunction | None = None

    def add_handler(self, handler: BaseEventHandler) -> None:
        """Fügt einen Event-Handler hinzu."""
        self.handlers.append(handler)

    def add_function_handler(
        self,
        event_types: str | set[str],
        handler_func: EventHandlerFunction
    ) -> None:
        """Fügt einen funktionsbasierten Handler hinzu."""
        handler = FunctionEventHandler(event_types, handler_func)
        self.add_handler(handler)

    def set_fallback_handler(self, handler_func: EventHandlerFunction) -> None:
        """Setzt einen Fallback-Handler für unbekannte Events."""
        self.fallback_handler = handler_func

    async def route_event(self, connection_id: str, data: dict[str, Any]) -> bool:
        """Routet ein Event an den entsprechenden Handler.

        Args:
            connection_id: ID der WebSocket-Verbindung
            data: Event-Daten

        Returns:
            True wenn ein Handler das Event behandelt hat
        """
        event_type = data.get("event_type", "unknown")

        # Suche passenden Handler
        for handler in self.handlers:
            if handler.can_handle(event_type):
                try:
                    await handler.handle(connection_id, data)
                    return True
                except Exception as e:
                    logger.exception(f"Fehler in Event-Handler für {event_type}: {e}")

        # Fallback-Handler verwenden
        if self.fallback_handler:
            try:
                if asyncio.iscoroutinefunction(self.fallback_handler):
                    await self.fallback_handler(connection_id, data)
                else:
                    self.fallback_handler(connection_id, data)
                return True
            except Exception as e:
                logger.exception(f"Fehler in Fallback-Handler für {event_type}: {e}")

        return False


# Kompatibilitäts-Funktionen für bestehenden Code
def now_iso() -> str:
    """Alias für create_server_timestamp() für Rückwärtskompatibilität."""
    return create_server_timestamp()
