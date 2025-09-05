# backend/websocket/manager.py
"""WebSocket Manager - Zentrales Connection-Management für Echtzeit-Kommunikation."""

import asyncio
import contextlib
import json
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from prometheus_client import Counter, Gauge

from data_models.websocket import ErrorEvent, EventType, WebSocketEvent
from kei_logging import get_logger

logger = get_logger(__name__)

# Konstanten
DEFAULT_CONNECTION_ID_SUFFIX_LENGTH = 8

# Prometheus Metriken für WebSocket‑Zustand
WS_CONNECTION_ATTEMPTS = Counter(
    "keiko_websocket_connection_attempts_total",
    "WebSocket Verbindungsversuche",
)
WS_CONNECTION_ERRORS = Counter(
    "keiko_websocket_connection_errors_total",
    "WebSocket Verbindungsfehler",
)
WS_CONNECTIONS_ACTIVE = Gauge(
    "keiko_websocket_connections_active",
    "Aktive WebSocket Verbindungen",
)


class WebSocketConnection:
    """Einzelne WebSocket-Verbindung mit optimierter Fehlerbehandlung."""

    def __init__(self, websocket: WebSocket, connection_id: str, user_id: str) -> None:
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.is_active = True

    async def send_event(self, event: WebSocketEvent | dict[str, Any]) -> bool:
        """Sendet Event an den Client."""
        if not self.is_active:
            return False

        try:
            # Behandle sowohl Pydantic-Modelle als auch dict-Objekte
            if hasattr(event, "model_dump_json"):
                # Pydantic-Modell
                event_data = event.model_dump_json()
            else:
                # dict-Objekt - konvertiere zu JSON mit datetime-Unterstützung
                event_data = self._serialize_to_json(event)

            await self.websocket.send_text(event_data)
            return True
        except Exception as e:
            logger.exception(f"Fehler beim Senden an {self.connection_id}: {e}")
            with contextlib.suppress(Exception):
                WS_CONNECTION_ERRORS.inc()
            self.is_active = False
            return False

    def _serialize_to_json(self, data: dict[str, Any]) -> str:
        """Serialisiert dict zu JSON mit datetime-Unterstützung."""
        import json
        from datetime import datetime

        def json_serializer(obj):
            """Custom JSON serializer für datetime und andere Objekte."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        return json.dumps(data, default=json_serializer)

    async def send_json(self, data: dict[str, Any]) -> bool:
        """Sendet JSON-Daten direkt."""
        if not self.is_active:
            return False

        try:
            await self.websocket.send_json(data)
            return True
        except Exception as e:
            logger.exception(f"Fehler beim Senden an {self.connection_id}: {e}")
            with contextlib.suppress(Exception):
                WS_CONNECTION_ERRORS.inc()
            self.is_active = False
            return False

    async def receive_message(self) -> dict[str, Any] | None:
        """Empfängt Nachricht vom Client."""
        try:
            message = await self.websocket.receive_text()
            return json.loads(message)
        except WebSocketDisconnect:
            self.is_active = False
            return None
        except Exception as e:
            logger.exception(f"Fehler beim Empfangen von {self.connection_id}: {e}")
            return None

class WebSocketManager:
    """Zentraler WebSocket-Manager mit optimierter Architektur."""

    def __init__(self) -> None:
        # Anzahl der Zeichen des zufälligen Suffixes in der Connection-ID
        # Beispiel: "{user_id}_<suffix>"
        self._connection_id_suffix_len: int = DEFAULT_CONNECTION_ID_SUFFIX_LENGTH

        # Registriert aktive Verbindungen (connection_id -> Verbindung)
        self.connections: dict[str, WebSocketConnection] = {}



        # Gruppiert Verbindungen pro Nutzer (user_id -> Menge connection_ids)
        self.user_connections: dict[str, set[str]] = defaultdict(set)

        # Registrierte Event-Handler je EventType
        self.event_handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._message_count = 0

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Neue Verbindung etablieren."""
        WS_CONNECTION_ATTEMPTS.inc()
        # WebSocket wurde bereits in den Handler-Funktionen akzeptiert
        # await websocket.accept() - ENTFERNT: Verhindert ASGI-Fehler

        connection_id = f"{user_id}_{uuid.uuid4().hex[:self._connection_id_suffix_len]}"
        connection = WebSocketConnection(websocket, connection_id, user_id)

        self.connections[connection_id] = connection
        self.user_connections[user_id].add(connection_id)

        logger.info(f"WebSocket verbunden: {user_id} ({connection_id})")
        with contextlib.suppress(Exception):
            WS_CONNECTIONS_ACTIVE.set(len(self.connections))
        return connection_id



    async def disconnect(self, connection_id: str) -> None:
        """Verbindung trennen und Ressourcen freigeben."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        user_id = connection.user_id

        # Connection schließen
        try:
            await connection.websocket.close()
        except Exception:
            pass  # Fehler beim Cleanup ignorieren

        # Aus Registry entfernen
        del self.connections[connection_id]
        self.user_connections[user_id].discard(connection_id)

        if not self.user_connections[user_id]:
            del self.user_connections[user_id]

        logger.info(f"WebSocket getrennt: {connection_id}")
        with contextlib.suppress(Exception):
            WS_CONNECTIONS_ACTIVE.set(len(self.connections))

    async def send_to_connection(self, connection_id: str, event: WebSocketEvent) -> bool:
        """Event an spezifische Verbindung senden."""
        if connection_id not in self.connections:
            return False

        success = await self.connections[connection_id].send_event(event)
        if success:
            self._message_count += 1
        return success

    async def send_json_to_connection(self, connection_id: str, data: dict[str, Any]) -> bool:
        """JSON an spezifische Verbindung senden."""
        if connection_id not in self.connections:
            return False

        success = await self.connections[connection_id].send_json(data)
        if success:
            self._message_count += 1
        return success

    async def send_to_user(self, user_id: str, event: WebSocketEvent) -> int:
        """Event an alle Verbindungen eines Benutzers senden."""
        if user_id not in self.user_connections:
            return 0

        sent_count = 0
        for connection_id in self.user_connections[user_id].copy():
            if await self.send_to_connection(connection_id, event):
                sent_count += 1

        return sent_count

    async def broadcast(self, event: WebSocketEvent) -> int:
        """Event an alle aktiven Verbindungen senden."""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if await self.send_to_connection(connection_id, event):
                sent_count += 1

        return sent_count

    def add_event_handler(self, event_type: EventType, handler: Callable[[str, dict[str, Any]], Any]) -> None:
        """Event-Handler registrieren."""
        self.event_handlers[event_type].append(handler)

    async def handle_message(self, connection_id: str, message_data: dict[str, Any]) -> None:
        """Eingehende Nachrichten verarbeiten."""
        try:
            # Spezielle Client-Nachricht (nicht Teil der EventType-Enum)
            # Wird verwendet, um einen zuvor angeforderten Funktionsaufruf zu bestätigen/abzulehnen
            message_type = message_data.get("message_type")
            if message_type == "function_confirmation":
                # Lazy-Import, um Zirkularimporte zu vermeiden
                from .handlers import handle_function_confirmation

                await handle_function_confirmation(connection_id, message_data)
                return

            event_type = EventType(message_data.get("event_type", ""))

            # Event-Handler ausführen
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(connection_id, message_data)
                    else:
                        handler(connection_id, message_data)
                except Exception as e:
                    logger.exception(f"Fehler in Event-Handler für {event_type}: {e}")

        except ValueError:
            # Unbekannter Event-Type
            error_event = ErrorEvent(
                error_code="INVALID_EVENT_TYPE",
                message="Unbekannter Event-Type"
            )
            await self.send_to_connection(connection_id, error_event)

        except Exception as e:
            logger.exception(f"Fehler beim Verarbeiten der Nachricht von {connection_id}: {e}")
            error_event = ErrorEvent(
                error_code="PROCESSING_ERROR",
                message="Fehler beim Verarbeiten der Nachricht"
            )
            await self.send_to_connection(connection_id, error_event)

    def get_connection(self, connection_id: str) -> WebSocketConnection | None:
        """Connection-Objekt abrufen."""
        return self.connections.get(connection_id)

    def get_stats(self) -> dict[str, Any]:
        """Manager-Statistiken abrufen."""
        return {
            "active_connections": len(self.connections),
            "unique_users": len(self.user_connections),
            "messages_sent": self._message_count,
        }


# Globale Manager-Instanz
websocket_manager = WebSocketManager()

# Standard-Handler registrieren
try:
    from .handlers import register_default_handlers
    register_default_handlers()
except ImportError:
    # Handler werden später registriert
    pass
