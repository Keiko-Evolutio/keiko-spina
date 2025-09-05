"""WebSocket-Endpunkt für Live System Heartbeat Streaming
"""

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from kei_logging import get_logger
from services.system_heartbeat_service import get_system_heartbeat_service

logger = get_logger(__name__)

router = APIRouter()

# Aktive WebSocket-Verbindungen
active_connections: set[WebSocket] = set()


class SystemHeartbeatStreamer:
    """Streamt System-Heartbeat-Daten an alle verbundenen Clients."""

    def __init__(self):
        self.running = False
        self.stream_task: asyncio.Task = None
        self.last_heartbeat: dict[str, Any] = None

    async def start_streaming(self):
        """Startet den Heartbeat-Stream."""
        if self.running:
            return

        self.running = True
        self.stream_task = asyncio.create_task(self._stream_loop())
        logger.info("💓 System Heartbeat Streaming gestartet")

    async def stop_streaming(self):
        """Stoppt den Heartbeat-Stream."""
        self.running = False
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 System Heartbeat Streaming gestoppt")

    async def _stream_loop(self):
        """Haupt-Stream-Loop."""
        while self.running:
            try:
                # Aktuellen Heartbeat abrufen
                heartbeat_service = get_system_heartbeat_service()
                if heartbeat_service:
                    current_heartbeat = heartbeat_service.get_current_heartbeat()

                    # Nur senden wenn sich etwas geändert hat
                    if current_heartbeat and current_heartbeat != self.last_heartbeat:
                        await self._broadcast_heartbeat(current_heartbeat)
                        self.last_heartbeat = current_heartbeat

                # Alle 5 Sekunden aktualisieren
                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Heartbeat-Stream: {e}")
                await asyncio.sleep(5.0)

    async def _broadcast_heartbeat(self, heartbeat_data: dict[str, Any]):
        """Sendet Heartbeat-Daten an alle verbundenen Clients."""
        if not active_connections:
            return

        message = {
            "type": "heartbeat_update",
            "data": heartbeat_data,
            "timestamp": heartbeat_data.get("timestamp")
        }

        # Disconnected connections entfernen
        disconnected = set()

        for websocket in active_connections.copy():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    disconnected.add(websocket)
            except Exception as e:
                logger.warning(f"Fehler beim Senden an WebSocket: {e}")
                disconnected.add(websocket)

        # Disconnected connections entfernen
        for ws in disconnected:
            active_connections.discard(ws)

        if disconnected:
            logger.debug(f"Entfernte {len(disconnected)} disconnected WebSocket-Verbindungen")


# Globaler Streamer
heartbeat_streamer = SystemHeartbeatStreamer()


async def authenticate_heartbeat_websocket(websocket: WebSocket) -> dict | None:
    """Authentifiziert WebSocket-Verbindungen für System Heartbeat.

    System Heartbeat erfordert mindestens 'system' oder 'admin' Berechtigung.
    """
    try:
        # Token aus Query-Parametern oder Headers extrahieren
        token = None
        if hasattr(websocket, "query_params") and websocket.query_params:
            token = websocket.query_params.get("token") or websocket.query_params.get("access_token")

        if not token and hasattr(websocket, "headers"):
            auth_header = websocket.headers.get("authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                token = auth_header[7:]

        if not token:
            return None

        # Enterprise Auth System für Validierung
        from auth.enterprise_auth import auth

        mock_request = type("MockRequest", (), {})()
        result = await auth.validator.validate(token, mock_request)

        if result.success and result.context:
            # Prüfe erforderliche Berechtigung für System Heartbeat
            required_scopes = {"system", "admin", "monitoring"}
            user_scopes = {scope.value for scope in result.context.scopes}

            if required_scopes.intersection(user_scopes):
                return {
                    "user_id": result.context.subject,
                    "scopes": list(user_scopes),
                    "privilege_level": result.context.privilege.value,
                    "authenticated_at": time.time()
                }
            logger.warning(f"Unzureichende Berechtigung für System Heartbeat: {user_scopes}")
            return None
        return None

    except Exception as e:
        logger.error(f"System Heartbeat WebSocket-Authentifizierung Fehler: {e}")
        return None


@router.websocket("/ws/system/heartbeat")
async def websocket_system_heartbeat(websocket: WebSocket):
    """Sicherer WebSocket-Endpunkt für Live System Heartbeat Updates.

    Erfordert Authentifizierung mit 'system', 'admin' oder 'monitoring' Berechtigung.
    Token über Query-Parameter: ?token=<your_token>
    """
    import os

    # Prüfe ob Authentifizierung erforderlich ist
    environment = os.getenv("ENVIRONMENT", "development").lower()
    auth_required = environment == "production" or os.getenv("WEBSOCKET_AUTH_ENABLED", "false").lower() == "true"

    user_context = None
    if auth_required:
        user_context = await authenticate_heartbeat_websocket(websocket)
        if not user_context:
            await websocket.close(code=4003, reason="Insufficient privileges for system heartbeat")
            return

    await websocket.accept()
    active_connections.add(websocket)

    try:
        if user_context:
            logger.info(f"💓 Authentifizierter WebSocket-Client verbunden für System Heartbeat: {user_context['user_id']} (Total: {len(active_connections)})")
        else:
            logger.info(f"💓 WebSocket-Client verbunden für System Heartbeat (Development Mode, Total: {len(active_connections)})")

        # Streaming starten wenn erster Client
        if len(active_connections) == 1:
            await heartbeat_streamer.start_streaming()

        # Initialen Heartbeat mit Authentifizierungsinfo senden
        heartbeat_service = get_system_heartbeat_service()
        if heartbeat_service:
            initial_heartbeat = heartbeat_service.get_current_heartbeat()
            if initial_heartbeat:
                heartbeat_message = {
                    "type": "heartbeat_update",
                    "data": initial_heartbeat,
                    "timestamp": initial_heartbeat.get("timestamp"),
                    "authenticated": user_context is not None,
                    "connection_id": id(websocket)
                }
                if user_context:
                    heartbeat_message["user_id"] = user_context["user_id"]
                    heartbeat_message["privilege_level"] = user_context["privilege_level"]

                await websocket.send_text(json.dumps(heartbeat_message))

        # Keep-alive Loop
        while True:
            try:
                # Ping/Pong für Connection-Health
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WebSocket Fehler: {e}")
                break

    except WebSocketDisconnect:
        logger.info("💓 WebSocket-Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket Fehler: {e}")
    finally:
        # Connection entfernen
        active_connections.discard(websocket)
        logger.info(f"💓 WebSocket-Client entfernt (Verbleibend: {len(active_connections)})")

        # Streaming stoppen wenn keine Clients mehr
        if len(active_connections) == 0:
            await heartbeat_streamer.stop_streaming()


@router.get("/system/heartbeat/status")
async def heartbeat_stream_status():
    """Status des Heartbeat-Streams."""
    return {
        "streaming": heartbeat_streamer.running,
        "active_connections": len(active_connections),
        "last_update": heartbeat_streamer.last_heartbeat.get("timestamp") if heartbeat_streamer.last_heartbeat else None
    }


# Startup/Shutdown Events
async def startup_heartbeat_streaming():
    """Initialisiert das Heartbeat-Streaming beim Startup."""
    logger.info("💓 System Heartbeat Streaming initialisiert")


async def shutdown_heartbeat_streaming():
    """Stoppt das Heartbeat-Streaming beim Shutdown."""
    await heartbeat_streamer.stop_streaming()
    logger.info("💓 System Heartbeat Streaming beendet")
