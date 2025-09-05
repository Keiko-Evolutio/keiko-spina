"""WebRTC Signaling Server

WebSocket-basierter Signaling Server für WebRTC-Verbindungen.
Behandelt Offer/Answer-Austausch und ICE Candidate-Handling.

@version 1.0.0
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from kei_logging import get_logger

from .config import WebRTCConfig
from .types import (
    SignalingMessage,
    SignalingMessageType,
    WebRTCPeer,
    WebRTCSession,
    WebRTCSessionState,
    create_connection_id,
    create_session_id,
    parse_signaling_message,
    validate_signaling_message,
)

logger = get_logger(__name__)

# =============================================================================
# WebSocket Connection Management
# =============================================================================

class WebRTCConnection:
    """WebRTC WebSocket-Verbindung."""

    def __init__(
        self,
        websocket: WebSocket,
        user_id: str,
        session_id: str,
        connection_id: str,
        ip_address: str
    ):
        self.websocket = websocket
        self.user_id = user_id
        self.session_id = session_id
        self.connection_id = connection_id
        self.ip_address = ip_address
        self.created_at = datetime.now(UTC)
        self.last_activity = datetime.now(UTC)
        self.is_active = True

    async def send_message(self, message: SignalingMessage) -> bool:
        """Sendet Signaling Message an Client."""
        try:
            message_dict = message.model_dump() if hasattr(message, "model_dump") else message.__dict__
            await self.websocket.send_text(json.dumps(message_dict))
            self.last_activity = datetime.now(UTC)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Senden der Message an {self.user_id}: {e}")
            self.is_active = False
            return False

    async def close(self, code: int = 1000, reason: str = "Normal closure") -> None:
        """Schließt WebSocket-Verbindung."""
        try:
            await self.websocket.close(code, reason)
        except Exception as e:
            logger.debug(f"Fehler beim Schließen der WebSocket-Verbindung: {e}")
        finally:
            self.is_active = False

# =============================================================================
# WebRTC Signaling Server
# =============================================================================

class WebRTCSignalingServer:
    """WebRTC Signaling Server für P2P-Verbindungen."""

    def __init__(self, config: WebRTCConfig):
        self.config = config

        # Connection Management
        self.connections: dict[str, WebRTCConnection] = {}  # connection_id -> connection
        self.user_connections: dict[str, set[str]] = {}  # user_id -> set of connection_ids
        self.session_connections: dict[str, set[str]] = {}  # session_id -> set of connection_ids

        # Session Management
        self.sessions: dict[str, WebRTCSession] = {}  # session_id -> session

        # Monitoring
        self.total_connections = 0
        self.total_sessions = 0
        self.active_connections = 0
        self.message_count = 0

        # Cleanup Task
        self.cleanup_task: asyncio.Task | None = None
        self.is_running = False

        logger.info("WebRTC Signaling Server initialisiert", extra={
            "signaling_port": config.signaling_port,
            "debug_mode": config.debug_mode
        })

    # =============================================================================
    # Server Lifecycle
    # =============================================================================

    async def start(self) -> None:
        """Startet den Signaling Server."""
        if self.is_running:
            logger.warning("Signaling Server bereits gestartet")
            return

        self.is_running = True

        # Cleanup Task starten
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("WebRTC Signaling Server gestartet")

    async def stop(self) -> None:
        """Stoppt den Signaling Server."""
        if not self.is_running:
            return

        self.is_running = False

        # Cleanup Task stoppen
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Alle Verbindungen schließen
        await self._close_all_connections()

        logger.info("WebRTC Signaling Server gestoppt")

    async def is_healthy(self) -> bool:
        """Prüft Server-Gesundheit."""
        return self.is_running and self.cleanup_task is not None and not self.cleanup_task.done()

    # =============================================================================
    # WebSocket Connection Handling
    # =============================================================================

    async def handle_websocket_connection(
        self,
        websocket: WebSocket,
        user_id: str,
        session_id: str | None = None,
        ip_address: str = "unknown"
    ) -> None:
        """Behandelt neue WebSocket-Verbindung."""
        # Session ID generieren falls nicht vorhanden
        if not session_id:
            session_id = create_session_id()

        connection_id = create_connection_id()

        try:
            # WebSocket akzeptieren
            await websocket.accept()

            # Connection erstellen
            connection = WebRTCConnection(
                websocket=websocket,
                user_id=user_id,
                session_id=session_id,
                connection_id=connection_id,
                ip_address=ip_address
            )

            # Connection registrieren
            await self._register_connection(connection)

            logger.info(f"WebRTC-Verbindung hergestellt: {user_id} (Session: {session_id})")

            # Message Loop
            await self._handle_connection_messages(connection)

        except WebSocketDisconnect:
            logger.info(f"WebRTC-Verbindung getrennt: {user_id}")
        except Exception as e:
            logger.error(f"Fehler in WebRTC-Verbindung für {user_id}: {e}")
        finally:
            # Connection cleanup
            await self._unregister_connection(connection_id)

    async def _handle_connection_messages(self, connection: WebRTCConnection) -> None:
        """Behandelt Messages von WebSocket-Verbindung."""
        try:
            while connection.is_active and self.is_running:
                # Message empfangen
                message_text = await connection.websocket.receive_text()

                try:
                    # Message parsen
                    message_data = json.loads(message_text)

                    # Message validieren
                    if not validate_signaling_message(message_data):
                        logger.warning(f"Ungültige Signaling Message von {connection.user_id}")
                        continue

                    # Message verarbeiten
                    await self._process_signaling_message(connection, message_data)

                    # Aktivität aktualisieren
                    connection.last_activity = datetime.now(UTC)
                    self.message_count += 1

                    # Debug Logging
                    if self.config.log_signaling_messages:
                        logger.debug(f"Signaling Message verarbeitet: {message_data.get('type')} von {connection.user_id}")

                except json.JSONDecodeError:
                    logger.warning(f"Ungültiges JSON von {connection.user_id}")
                except Exception as e:
                    logger.error(f"Fehler beim Verarbeiten der Message von {connection.user_id}: {e}")

        except WebSocketDisconnect:
            logger.debug(f"WebSocket getrennt: {connection.user_id}")
        except Exception as e:
            logger.error(f"Fehler in Message Loop für {connection.user_id}: {e}")

    # =============================================================================
    # Signaling Message Processing
    # =============================================================================

    async def _process_signaling_message(
        self,
        connection: WebRTCConnection,
        message_data: dict[str, Any]
    ) -> None:
        """Verarbeitet Signaling Message."""
        message_type = message_data.get("type")
        message_data.get("session_id", connection.session_id)

        try:
            # Message parsen
            message = parse_signaling_message(message_data)

            # Message-Type-spezifische Verarbeitung
            if message_type == SignalingMessageType.OFFER:
                await self._handle_offer(connection, message)
            elif message_type == SignalingMessageType.ANSWER:
                await self._handle_answer(connection, message)
            elif message_type == SignalingMessageType.ICE_CANDIDATE:
                await self._handle_ice_candidate(connection, message)
            elif message_type == SignalingMessageType.PING:
                await self._handle_ping(connection, message)
            elif message_type == SignalingMessageType.ERROR:
                await self._handle_error_message(connection, message)
            else:
                logger.warning(f"Unbekannter Message Type: {message_type} von {connection.user_id}")

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der Signaling Message: {e}")
            await self._send_error_to_connection(
                connection,
                "MESSAGE_PROCESSING_ERROR",
                f"Fehler beim Verarbeiten der Message: {e!s}"
            )

    async def _handle_offer(self, connection: WebRTCConnection, message: SignalingMessage) -> None:
        """Behandelt SDP Offer."""
        session_id = message.session_id

        # Session erstellen oder aktualisieren
        await self._get_or_create_session(session_id, connection)

        # Offer an andere Teilnehmer weiterleiten
        await self._relay_message_to_session_peers(connection, message, session_id)

        logger.info(f"SDP Offer verarbeitet für Session {session_id}")

    async def _handle_answer(self, connection: WebRTCConnection, message: SignalingMessage) -> None:
        """Behandelt SDP Answer."""
        session_id = message.session_id

        # Answer an andere Teilnehmer weiterleiten
        await self._relay_message_to_session_peers(connection, message, session_id)

        logger.info(f"SDP Answer verarbeitet für Session {session_id}")

    async def _handle_ice_candidate(self, connection: WebRTCConnection, message: SignalingMessage) -> None:
        """Behandelt ICE Candidate."""
        session_id = message.session_id

        # ICE Candidate an andere Teilnehmer weiterleiten
        await self._relay_message_to_session_peers(connection, message, session_id)

        if self.config.log_ice_candidates:
            logger.debug(f"ICE Candidate verarbeitet für Session {session_id}")

    async def _handle_ping(self, connection: WebRTCConnection, message: SignalingMessage) -> None:
        """Behandelt Ping Message."""
        # Pong zurücksenden
        pong_message = {
            "type": SignalingMessageType.PONG,
            "session_id": message.session_id,
            "user_id": connection.user_id,
            "timestamp": datetime.now(UTC).isoformat()
        }

        await connection.send_message(pong_message)

    async def _handle_error_message(self, connection: WebRTCConnection, message: SignalingMessage) -> None:
        """Behandelt Error Message."""
        logger.warning(f"Error Message von {connection.user_id}: {message}")

        # Error an andere Session-Teilnehmer weiterleiten
        await self._relay_message_to_session_peers(connection, message, message.session_id)

    # =============================================================================
    # Session Management
    # =============================================================================

    async def _get_or_create_session(self, session_id: str, connection: WebRTCConnection) -> WebRTCSession:
        """Holt oder erstellt WebRTC Session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Responder hinzufügen falls noch nicht vorhanden
            if not session.responder and session.initiator.user_id != connection.user_id:
                session.responder = WebRTCPeer(
                    user_id=connection.user_id,
                    session_id=session_id,
                    connection_id=connection.connection_id,
                    ip_address=connection.ip_address
                )
                session.state = WebRTCSessionState.CONNECTING
                logger.info(f"Responder zu Session {session_id} hinzugefügt: {connection.user_id}")

            return session

        # Neue Session erstellen
        peer = WebRTCPeer(
            user_id=connection.user_id,
            session_id=session_id,
            connection_id=connection.connection_id,
            ip_address=connection.ip_address
        )

        session = WebRTCSession(
            session_id=session_id,
            initiator=peer,
            state=WebRTCSessionState.CREATED
        )

        self.sessions[session_id] = session
        self.total_sessions += 1

        logger.info(f"Neue WebRTC Session erstellt: {session_id} (Initiator: {connection.user_id})")
        return session

    async def _relay_message_to_session_peers(
        self,
        sender_connection: WebRTCConnection,
        message: SignalingMessage,
        session_id: str
    ) -> None:
        """Leitet Message an andere Session-Teilnehmer weiter."""
        if session_id not in self.session_connections:
            logger.warning(f"Session {session_id} nicht gefunden für Message Relay")
            return

        connection_ids = self.session_connections[session_id]

        # Message an alle anderen Connections in der Session senden
        for connection_id in connection_ids:
            if connection_id == sender_connection.connection_id:
                continue  # Nicht an Sender zurücksenden

            connection = self.connections.get(connection_id)
            if connection and connection.is_active:
                success = await connection.send_message(message)
                if not success:
                    logger.warning(f"Fehler beim Weiterleiten der Message an {connection.user_id}")

    # =============================================================================
    # Connection Management
    # =============================================================================

    async def _register_connection(self, connection: WebRTCConnection) -> None:
        """Registriert neue WebSocket-Verbindung."""
        # Connection registrieren
        self.connections[connection.connection_id] = connection

        # User-Connection-Mapping
        if connection.user_id not in self.user_connections:
            self.user_connections[connection.user_id] = set()
        self.user_connections[connection.user_id].add(connection.connection_id)

        # Session-Connection-Mapping
        if connection.session_id not in self.session_connections:
            self.session_connections[connection.session_id] = set()
        self.session_connections[connection.session_id].add(connection.connection_id)

        # Statistiken aktualisieren
        self.total_connections += 1
        self.active_connections += 1

        logger.debug(f"Connection registriert: {connection.user_id} ({connection.connection_id})")

    async def _unregister_connection(self, connection_id: str) -> None:
        """Entfernt WebSocket-Verbindung."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Connection aus Mappings entfernen
        if connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]

        if connection.session_id in self.session_connections:
            self.session_connections[connection.session_id].discard(connection_id)
            if not self.session_connections[connection.session_id]:
                del self.session_connections[connection.session_id]

                # Session cleanup wenn keine Connections mehr
                await self._cleanup_session(connection.session_id)

        # Connection entfernen
        del self.connections[connection_id]

        # Statistiken aktualisieren
        self.active_connections -= 1

        logger.debug(f"Connection entfernt: {connection.user_id} ({connection_id})")

    async def _cleanup_session(self, session_id: str) -> None:
        """Räumt Session auf wenn keine Connections mehr vorhanden."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.state = WebRTCSessionState.DISCONNECTED
            session.disconnected_at = datetime.now(UTC)

            # Session nach Timeout entfernen
            await asyncio.sleep(60)  # 1 Minute Cleanup-Delay
            if session_id in self.sessions and session_id not in self.session_connections:
                del self.sessions[session_id]
                logger.info(f"Session {session_id} aufgeräumt")

    async def _close_all_connections(self) -> None:
        """Schließt alle WebSocket-Verbindungen."""
        close_tasks = []

        for connection in self.connections.values():
            if connection.is_active:
                close_tasks.append(connection.close(1001, "Server shutdown"))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Alle Mappings leeren
        self.connections.clear()
        self.user_connections.clear()
        self.session_connections.clear()
        self.sessions.clear()

        self.active_connections = 0

        logger.info("Alle WebRTC-Verbindungen geschlossen")

    # =============================================================================
    # Utility Methods
    # =============================================================================

    async def _send_error_to_connection(
        self,
        connection: WebRTCConnection,
        error_code: str,
        error_message: str
    ) -> None:
        """Sendet Error Message an Connection."""
        error_msg = {
            "type": SignalingMessageType.ERROR,
            "session_id": connection.session_id,
            "user_id": connection.user_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "error_code": error_code,
            "error_message": error_message
        }

        await connection.send_message(error_msg)

    async def _cleanup_loop(self) -> None:
        """Cleanup Loop für inaktive Connections und Sessions."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.session_cleanup_interval)
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Cleanup Loop: {e}")

    async def _perform_cleanup(self) -> None:
        """Führt Cleanup von inaktiven Connections und Sessions durch."""
        now = datetime.now(UTC)
        cleanup_threshold = now.timestamp() - self.config.session_timeout

        # Inaktive Connections finden
        inactive_connections = []
        for connection in self.connections.values():
            if connection.last_activity.timestamp() < cleanup_threshold:
                inactive_connections.append(connection.connection_id)

        # Inaktive Connections entfernen
        for connection_id in inactive_connections:
            connection = self.connections.get(connection_id)
            if connection:
                await connection.close(1000, "Session timeout")
                await self._unregister_connection(connection_id)
                logger.info(f"Inaktive Connection entfernt: {connection.user_id}")

        # Session Cleanup
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if session.last_activity.timestamp() < cleanup_threshold:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self._cleanup_session(session_id)

    # =============================================================================
    # Status und Monitoring
    # =============================================================================

    def get_server_stats(self) -> dict[str, Any]:
        """Gibt Server-Statistiken zurück."""
        return {
            "is_running": self.is_running,
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "active_sessions": len(self.sessions),
            "total_sessions": self.total_sessions,
            "message_count": self.message_count,
            "users_online": len(self.user_connections),
            "timestamp": datetime.now(UTC).isoformat()
        }

# =============================================================================
# Factory Function
# =============================================================================

def create_signaling_server(config: WebRTCConfig) -> WebRTCSignalingServer:
    """Erstellt WebRTC Signaling Server."""
    return WebRTCSignalingServer(config)
