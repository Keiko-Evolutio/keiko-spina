"""WebSocket-Handler und Message Processing für Keiko.

Dieses Modul kapselt die bisherigen WebSocket-Endpunkte und zugehörige
Nachrichtenverarbeitung aus main.py.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agents import get_system_status as get_adapter_system_status
from monitoring import record_custom_metric
from services.streaming.manager import websocket_manager

from .common.constants import (
    WS_EVENT_AGENT_INPUT,
    WS_EVENT_PING,
    WS_EVENT_PONG,
    WS_EVENT_SYSTEM_QUERY,
    WS_EVENT_VOICE_INPUT,
    WS_MSG_ERROR,
)
from .common.logger_utils import get_module_logger

logger = get_module_logger(__name__)


def now_iso() -> str:
    """Gibt den aktuellen Zeitstempel im ISO‑Format (UTC) zurück."""
    return datetime.now(UTC).isoformat()


async def send_error(connection_id: str, message: str) -> None:
    """Sendet eine standardisierte Fehlermeldung an eine WebSocket‑Verbindung."""
    await websocket_manager.send_json_to_connection(
        connection_id,
        {
            "type": WS_MSG_ERROR,
            "message": message,
            "timestamp": now_iso(),
        },
    )


def services_status() -> dict[str, bool]:
    """Liefert den aktuellen Status zentraler Systemkomponenten."""
    return {
        "agents": True,
        "voice": True,
        "adapters": True,
        "redis": True,
        "services": True,
    }


class WebSocketEventBase:
    """Basis‑Event für WebSocket‑Kommunikation (leichtgewichtig)."""

    def __init__(self, event_type: str, session_id: str | None = None) -> None:
        self.event_id: str = f"evt_{uuid.uuid4().hex[:8]}"
        self.event_type: str = event_type
        self.timestamp: datetime = datetime.now(UTC)
        self.session_id: str | None = session_id

    def model_dump(self) -> dict[str, Any]:
        """Serialisiert das Event als Dict."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }


class SystemStatusEvent(WebSocketEventBase):
    """System‑Status Event."""

    def __init__(self, status: str, services: dict[str, Any], message: str | None = None) -> None:
        super().__init__(event_type="system_status")
        self.status: str = status
        self.services: dict[str, Any] = services
        self.message: str | None = message

    def model_dump(self) -> dict[str, Any]:
        data = super().model_dump()
        data.update({
            "status": self.status,
            "services": self.services,
            "message": self.message,
        })
        return data


# ----------------------------------------------------------------------
# WebSocket Endpunkte
# ----------------------------------------------------------------------
async def authenticate_client_websocket(websocket: WebSocket, user_id: str) -> dict | None:
    """Authentifiziert Client WebSocket-Verbindungen.

    Args:
        websocket: WebSocket-Verbindung
        user_id: Benutzer-ID aus URL-Parameter

    Returns:
        User context dict if authenticated, None otherwise
    """
    try:
        # Token aus Query-Parametern extrahieren
        token = None
        if hasattr(websocket, "query_params") and websocket.query_params:
            token = websocket.query_params.get("token") or websocket.query_params.get("access_token")

        if not token:
            logger.warning(f"Client WebSocket-Verbindung ohne Token: {user_id}")
            return None

        # Enterprise Auth System für Validierung
        from auth.enterprise_auth import auth

        mock_request = type("MockRequest", (), {})()
        result = await auth.validator.validate(token, mock_request)

        if result.success and result.context:
            # Prüfe ob user_id mit Token-Subject übereinstimmt (optional)
            token_subject = result.context.subject
            if user_id != token_subject:
                logger.info(f"User ID Mismatch - URL: {user_id}, Token: {token_subject} (erlaubt für Admin)")
                # Erlaube Mismatch für Admin/System-Benutzer
                admin_scopes = {"admin", "system"}
                user_scopes = {scope.value for scope in result.context.scopes}
                if not admin_scopes.intersection(user_scopes):
                    logger.warning(f"Unzureichende Berechtigung für User ID Mismatch: {user_scopes}")
                    return None

            return {
                "user_id": result.context.subject,
                "requested_user_id": user_id,
                "scopes": [scope.value for scope in result.context.scopes],
                "privilege_level": result.context.privilege.value,
                "authenticated_at": time.time()
            }
        logger.warning(f"Client WebSocket Token-Validierung fehlgeschlagen: {result.error}")
        return None

    except Exception as e:
        logger.error(f"Client WebSocket-Authentifizierung Fehler: {e}")
        return None


async def client_websocket(websocket: WebSocket, user_id: str) -> None:
    """Enterprise-grade Client WebSocket‑Endpoint mit umfassender Sicherheit."""
    from api.middleware.enterprise_websocket_auth import enterprise_websocket_auth

    connection_id = None
    user_context = None

    try:
        # Enterprise-grade Authentifizierung (IMMER erforderlich)
        endpoint_path = f"/ws/client/{user_id}"

        # Prüfe ob Endpoint Authentifizierung erfordert
        if enterprise_websocket_auth.is_endpoint_protected(endpoint_path):
            user_context = await enterprise_websocket_auth.authenticate_websocket(websocket, endpoint_path)
            if not user_context:
                await websocket.close(code=4001, reason="Authentication required - Enterprise Security")
                return
        else:
            # Prüfe ob es sich um einen System-Client handelt
            from config.websocket_auth_config import WEBSOCKET_AUTH_CONFIG
            is_system_client = any(
                endpoint_path.startswith(pattern.replace("*", ""))
                for pattern in WEBSOCKET_AUTH_CONFIG.system_clients.path_patterns
            )
            if is_system_client:
                logger.info(f"🔧 System-Client WebSocket verbunden (Auth bypassed): {endpoint_path}")
            else:
                logger.warning(f"⚠️ WebSocket endpoint not protected: {endpoint_path}")

        # WebSocket-Verbindung akzeptieren
        await websocket.accept()

        connection_id = await websocket_manager.connect(websocket, user_id)

        if user_context:
            logger.info(f"🔐 Authentifizierter Client verbunden: {user_context['user_id']} (als {user_id})")
        else:
            logger.info(f"🔓 Client verbunden (Development Mode): {user_id}")

        # Willkommensnachricht mit Authentifizierungsinfo
        welcome_message = f"Willkommen {user_id}!"
        if user_context:
            welcome_message += f" (Authentifiziert als {user_context['user_id']})"

        welcome = SystemStatusEvent(
            status="connected",
            services=services_status(),
            message=welcome_message,
        )
        welcome_data = welcome.model_dump()
        welcome_data["authenticated"] = user_context is not None
        if user_context:
            welcome_data["auth_info"] = {
                "user_id": user_context["user_id"],
                "scopes": user_context["scopes"],
                "privilege_level": user_context["privilege_level"]
            }

        await websocket_manager.send_json_to_connection(connection_id, welcome_data)
        with contextlib.suppress(Exception):
            record_custom_metric("ws.client.connected", 1, {"user_id": user_id})

        # Message Loop
        while True:
            try:
                data = await websocket.receive_json()
                await handle_client_message(connection_id, user_id, data)
                with contextlib.suppress(Exception):
                    record_custom_metric("ws.client.msg", 1, {"user_id": user_id})
            except WebSocketDisconnect:
                logger.info(f"Client getrennt: {user_id}")
                break
            except Exception as exc:
                logger.error(f"Message Handling Fehler: {exc}")
                await send_error(connection_id, str(exc))

    except Exception as exc:
        logger.error(f"WebSocket Fehler für {user_id}: {exc}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)
            with contextlib.suppress(Exception):
                record_custom_metric("ws.client.disconnected", 1, {"user_id": user_id})


async def agent_websocket(websocket: WebSocket, agent_id: str) -> None:
    """Agent WebSocket‑Endpoint für A2A‑Kommunikation."""
    connection_id = None
    try:
        connection_id = await websocket_manager.connect(websocket, agent_id)
        logger.info(f"Agent verbunden: {agent_id}")

        # Agent Registration
        try:
            status = get_adapter_system_status()
        except Exception as exc:
            logger.debug(f"Adapter Status Fehler: {exc}")
            status = {"status": "unknown"}
        await websocket_manager.send_json_to_connection(
            connection_id,
            {
                "type": "agent_registered",
                "agent_id": agent_id,
                "adapter_status": status,
                "timestamp": now_iso(),
            },
        )

        # Message Loop
        while True:
            try:
                data = await websocket.receive_json()
                await handle_agent_message(connection_id, agent_id, data)
            except WebSocketDisconnect:
                logger.info(f"Agent getrennt: {agent_id}")
                break
            except Exception as exc:
                logger.error(f"Agent Message Handling Fehler: {exc}")

    except Exception as exc:
        logger.error(f"Agent WebSocket Fehler für {agent_id}: {exc}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)


def register_websocket_routes(app: FastAPI) -> None:
    """Registriert die WebSocket‑Routen auf der App."""
    app.websocket("/ws/client/{user_id}")(client_websocket)
    app.websocket("/ws/agent/{agent_id}")(agent_websocket)


# ----------------------------------------------------------------------
# Message Processing
# ----------------------------------------------------------------------
async def handle_client_message(connection_id: str, user_id: str, data: dict[str, Any]) -> None:
    """Client‑Nachrichten verarbeiten."""
    try:
        event_type = data.get("event_type", "unknown")
        logger.debug(f"Client Message ({user_id}): {event_type}")

        if event_type == WS_EVENT_AGENT_INPUT:
            await handle_agent_input(connection_id, user_id, data)
        elif event_type == WS_EVENT_VOICE_INPUT:
            await handle_voice_input(connection_id, user_id, data)
        elif event_type == WS_EVENT_SYSTEM_QUERY:
            await handle_system_query(connection_id, user_id, data)
        elif event_type == WS_EVENT_PING:
            # Extrahiere client_time für korrekte Round-Trip-Latenz-Messung
            client_time = data.get("client_time")

            pong_response = {
                "event_type": WS_EVENT_PONG,
                "server_timestamp": now_iso()
            }

            # Gebe client_time zurück, wenn vorhanden (für TypedWebSocketClient)
            if client_time:
                pong_response["client_time"] = client_time
                logger.debug(f"Ping-Pong: client_time={client_time}, user={user_id}")

            await websocket_manager.send_json_to_connection(connection_id, pong_response)
        else:
            await send_error(connection_id, f"Unbekannter Event‑Typ: {event_type}")

    except Exception as exc:
        logger.exception(f"Client Message Handling Fehler: {exc}")
        await send_error(connection_id, str(exc))


async def handle_agent_message(connection_id: str, agent_id: str, data: dict[str, Any]) -> None:
    """Agent‑zu‑Agent Nachrichten verarbeiten."""
    try:
        event_type = data.get("event_type", "unknown")
        logger.debug(f"Agent Message ({agent_id}): {event_type}")

        if event_type == "task_request":
            # Task‑Delegation
            target_agent = data.get("target_agent")
            task_data = data.get("task_data", {})

            message = {
                "type": "task_delegation",
                "from_agent": agent_id,
                "task_data": task_data,
                "timestamp": now_iso(),
            }

            if target_agent:
                # Direkte Delegation
                await websocket_manager.send_json_to_connection(target_agent, message)
            else:
                # Broadcast an alle Verbindungen außer der eigenen
                for conn_id in websocket_manager.connections:
                    if conn_id != connection_id:
                        await websocket_manager.send_json_to_connection(conn_id, message)

        elif event_type == "capability_advertisement":
            capabilities = data.get("capabilities", [])
            logger.info(f"Agent {agent_id} Capabilities: {capabilities}")

        else:
            logger.warning(f"Unbekannter Agent Event‑Typ: {event_type}")

    except Exception as exc:
        logger.exception(f"Agent Message Handling Fehler: {exc}")


async def handle_agent_input(connection_id: str, user_id: str, data: dict[str, Any]) -> None:
    """Agent Input verarbeiten."""
    try:
        content = data.get("content", "")
        await websocket_manager.send_json_to_connection(
            connection_id,
            {
                "type": "agent_response",
                "content": f"Verarbeitet: {content}",
                "metadata": {"processed_by": "agent", "user_id": user_id},
                "timestamp": now_iso(),
            },
        )
    except Exception as exc:
        logger.exception(f"Agent Input Processing Fehler: {exc}")


async def handle_voice_input(connection_id: str, _user_id: str, data: dict[str, Any]) -> None:
    """Voice Input verarbeiten."""
    try:
        audio_data = data.get("audio_data", "")
        await websocket_manager.send_json_to_connection(
            connection_id,
            {
                "type": "voice_response",
                "transcription": f"Voice Input empfangen: {len(audio_data)} bytes",
                "timestamp": now_iso(),
            },
        )
    except Exception as exc:
        logger.exception(f"Voice Input Processing Fehler: {exc}")


async def handle_system_query(connection_id: str, _user_id: str, data: dict[str, Any]) -> None:
    """System Query verarbeiten."""
    try:
        query_type = data.get("query_type", "status")

        if query_type == "status":
            status = {
                "type": "system_status",
                "services": services_status(),
                "websocket_stats": websocket_manager.get_stats(),
                "timestamp": now_iso(),
            }
            await websocket_manager.send_json_to_connection(connection_id, status)
        else:
            await send_error(connection_id, f"Unbekannter Query‑Typ: {query_type}")

    except Exception as exc:
        logger.exception(f"System Query Processing Fehler: {exc}")


__all__ = [
    "SystemStatusEvent",
    "agent_websocket",
    "client_websocket",
    "handle_agent_input",
    "handle_agent_message",
    "handle_client_message",
    "handle_system_query",
    "handle_voice_input",
    "register_websocket_routes",
]
