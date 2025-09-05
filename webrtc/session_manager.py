"""WebRTC Session Manager für Keiko Personal Assistant.

Dieses Modul implementiert den Session Manager für die Verwaltung
von WebRTC-Sessions und Peer-Verbindungen.
"""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
from .config import WebRTCConfig
from .types import (
    ConnectionStateChangeEvent,
    IceConnectionState,
    IceGatheringState,
    SignalingState,
    WebRTCEvent,
    WebRTCPeer,
    WebRTCSession,
    WebRTCSessionState,
)

logger = get_logger(__name__)


@dataclass
class SessionMetrics:
    """Metriken für Session Manager."""
    total_sessions_created: int = 0
    active_sessions: int = 0
    completed_sessions: int = 0
    failed_sessions: int = 0
    average_session_duration_seconds: float = 0.0
    peak_concurrent_sessions: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SessionCleanupInfo:
    """Informationen für Session-Cleanup."""
    session_id: str
    last_activity: datetime
    state: WebRTCSessionState
    cleanup_reason: str


class WebRTCSessionManager:
    """Enterprise WebRTC Session Manager für intelligente Session-Verwaltung.

    Implementiert umfassende Session-Verwaltung mit:
    - Automatische Session-Lifecycle-Verwaltung
    - Peer-Connection-Tracking und -Monitoring
    - Session-Timeout und Cleanup-Mechanismen
    - Event-basierte Session-State-Verwaltung
    - Performance-Monitoring und Metriken
    """

    def __init__(self, config: WebRTCConfig | None = None):
        """Initialisiert den WebRTC Session Manager.

        Args:
            config: WebRTC-Konfiguration
        """
        self.config = config or WebRTCConfig()

        # Session-Management
        self._sessions: dict[str, WebRTCSession] = {}
        self._user_sessions: dict[str, set[str]] = {}  # user_id -> session_ids
        self._session_lock = asyncio.Lock()

        # Cleanup und Monitoring
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Konfiguration
        self.session_timeout = timedelta(seconds=self.config.session_timeout)
        self.cleanup_interval = timedelta(seconds=self.config.session_cleanup_interval)
        self.max_sessions_per_user = self.config.max_sessions_per_user

        # Metriken
        self.metrics = SessionMetrics()

        # Event-Callbacks
        self._session_callbacks: list[Callable[[WebRTCEvent], None]] = []

        logger.info("WebRTC Session Manager initialisiert")

    async def start(self) -> None:
        """Startet den Session Manager und Cleanup-Tasks."""
        if self._running:
            logger.warning("Session Manager bereits gestartet")
            return

        self._running = True

        # Cleanup-Task starten
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="webrtc-session-cleanup"
        )

        logger.info("WebRTC Session Manager gestartet")

    async def stop(self) -> None:
        """Stoppt den Session Manager und alle Sessions."""
        if not self._running:
            return

        self._running = False

        # Cleanup-Task stoppen
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Alle Sessions schließen
        await self._close_all_sessions()

        logger.info("WebRTC Session Manager gestoppt")

    async def create_session(
        self,
        user_id: str,
        connection_id: str,
        ip_address: str,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Erstellt eine neue WebRTC-Session.

        Args:
            user_id: Benutzer-ID
            connection_id: Verbindungs-ID
            ip_address: IP-Adresse des Clients
            user_agent: User-Agent-String
            metadata: Zusätzliche Metadaten

        Returns:
            Session-ID

        Raises:
            ValueError: Wenn Session-Limit erreicht
        """
        async with self._session_lock:
            # Session-Limit pro Benutzer prüfen
            user_session_count = len(self._user_sessions.get(user_id, set()))
            if user_session_count >= self.max_sessions_per_user:
                raise ValueError(f"Session-Limit erreicht für Benutzer {user_id}")

            # Session-ID generieren
            session_id = str(uuid.uuid4())

            # Peer erstellen
            initiator = WebRTCPeer(
                user_id=user_id,
                session_id=session_id,
                connection_id=connection_id,
                ip_address=ip_address,
                user_agent=user_agent
            )

            # Session erstellen
            session = WebRTCSession(
                session_id=session_id,
                initiator=initiator,
                metadata=metadata or {}
            )

            # Session speichern
            self._sessions[session_id] = session

            # User-Session-Mapping aktualisieren
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)

            # Metriken aktualisieren
            self.metrics.total_sessions_created += 1
            self.metrics.active_sessions = len(self._sessions)
            self.metrics.peak_concurrent_sessions = max(
                self.metrics.peak_concurrent_sessions,
                self.metrics.active_sessions
            )

            # Event senden
            event = ConnectionStateChangeEvent(
                session_id=session_id,
                previous_state=WebRTCSessionState.CREATED,
                new_state=WebRTCSessionState.CREATED
            )
            await self._emit_event(event)

            logger.info(f"WebRTC-Session erstellt: {session_id} für Benutzer {user_id}")

            return session_id

    async def get_session(self, session_id: str) -> WebRTCSession | None:
        """Gibt eine Session zurück.

        Args:
            session_id: Session-ID

        Returns:
            WebRTC-Session oder None
        """
        async with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_activity = datetime.now(UTC)
            return session

    async def update_session_state(
        self,
        session_id: str,
        new_state: WebRTCSessionState,
        signaling_state: SignalingState | None = None,
        ice_connection_state: IceConnectionState | None = None,
        ice_gathering_state: IceGatheringState | None = None
    ) -> bool:
        """Aktualisiert den Session-State.

        Args:
            session_id: Session-ID
            new_state: Neuer Session-State
            signaling_state: Neuer Signaling-State (optional)
            ice_connection_state: Neuer ICE-Connection-State (optional)
            ice_gathering_state: Neuer ICE-Gathering-State (optional)

        Returns:
            True wenn erfolgreich aktualisiert
        """
        async with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session nicht gefunden: {session_id}")
                return False

            previous_state = session.state
            session.state = new_state
            session.last_activity = datetime.now(UTC)

            # Zusätzliche States aktualisieren
            if signaling_state:
                session.signaling_state = signaling_state
            if ice_connection_state:
                session.ice_connection_state = ice_connection_state
            if ice_gathering_state:
                session.ice_gathering_state = ice_gathering_state

            # Timestamps setzen
            if new_state == WebRTCSessionState.CONNECTED and not session.connected_at:
                session.connected_at = datetime.now(UTC)
            elif new_state in [WebRTCSessionState.DISCONNECTED, WebRTCSessionState.CLOSED, WebRTCSessionState.FAILED]:
                if not session.disconnected_at:
                    session.disconnected_at = datetime.now(UTC)

            # Event senden
            event = ConnectionStateChangeEvent(
                session_id=session_id,
                previous_state=previous_state,
                new_state=new_state
            )
            await self._emit_event(event)

            logger.debug(f"Session-State aktualisiert: {session_id} {previous_state.value} -> {new_state.value}")

            return True

    async def add_responder(
        self,
        session_id: str,
        user_id: str,
        connection_id: str,
        ip_address: str,
        user_agent: str | None = None
    ) -> bool:
        """Fügt einen Responder zu einer Session hinzu.

        Args:
            session_id: Session-ID
            user_id: Benutzer-ID des Responders
            connection_id: Verbindungs-ID
            ip_address: IP-Adresse
            user_agent: User-Agent-String

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        async with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session nicht gefunden: {session_id}")
                return False

            if session.responder:
                logger.warning(f"Session {session_id} hat bereits einen Responder")
                return False

            # Responder erstellen
            responder = WebRTCPeer(
                user_id=user_id,
                session_id=session_id,
                connection_id=connection_id,
                ip_address=ip_address,
                user_agent=user_agent
            )

            session.responder = responder
            session.last_activity = datetime.now(UTC)

            # User-Session-Mapping aktualisieren
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)

            logger.info(f"Responder hinzugefügt zu Session {session_id}: {user_id}")

            return True

    async def close_session(self, session_id: str, reason: str = "normal") -> bool:
        """Schließt eine Session.

        Args:
            session_id: Session-ID
            reason: Grund für das Schließen

        Returns:
            True wenn erfolgreich geschlossen
        """
        async with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            # Session-State aktualisieren
            previous_state = session.state
            session.state = WebRTCSessionState.CLOSED
            session.disconnected_at = datetime.now(UTC)
            session.metadata["close_reason"] = reason

            # User-Session-Mappings entfernen
            self._remove_session_from_user_mappings(session_id, session)

            # Session entfernen
            del self._sessions[session_id]

            # Metriken aktualisieren
            self.metrics.active_sessions = len(self._sessions)
            if previous_state == WebRTCSessionState.CONNECTED:
                self.metrics.completed_sessions += 1
            else:
                self.metrics.failed_sessions += 1

            # Durchschnittliche Session-Dauer berechnen
            if session.connected_at and session.disconnected_at:
                duration = (session.disconnected_at - session.connected_at).total_seconds()
                self._update_average_duration(duration)

            # Event senden
            event = ConnectionStateChangeEvent(
                session_id=session_id,
                previous_state=previous_state,
                new_state=WebRTCSessionState.CLOSED,
                data={"reason": reason}
            )
            await self._emit_event(event)

            logger.info(f"Session geschlossen: {session_id} (Grund: {reason})")

            return True

    async def get_user_sessions(self, user_id: str) -> list[WebRTCSession]:
        """Gibt alle Sessions eines Benutzers zurück.

        Args:
            user_id: Benutzer-ID

        Returns:
            Liste der Sessions
        """
        async with self._session_lock:
            session_ids = self._user_sessions.get(user_id, set())
            return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def get_active_sessions(self) -> list[WebRTCSession]:
        """Gibt alle aktiven Sessions zurück.

        Returns:
            Liste der aktiven Sessions
        """
        async with self._session_lock:
            return [
                session for session in self._sessions.values()
                if session.state not in [WebRTCSessionState.CLOSED, WebRTCSessionState.FAILED]
            ]

    def add_session_callback(self, callback: Callable[[WebRTCEvent], None]) -> None:
        """Fügt Callback für Session-Events hinzu.

        Args:
            callback: Callback-Funktion
        """
        self._session_callbacks.append(callback)

    def _remove_session_from_user_mappings(self, session_id: str, session: WebRTCSession) -> None:
        """Entfernt Session aus User-Mappings."""
        # Initiator entfernen
        if session.initiator.user_id in self._user_sessions:
            self._user_sessions[session.initiator.user_id].discard(session_id)
            if not self._user_sessions[session.initiator.user_id]:
                del self._user_sessions[session.initiator.user_id]

        # Responder entfernen
        if session.responder and session.responder.user_id in self._user_sessions:
            self._user_sessions[session.responder.user_id].discard(session_id)
            if not self._user_sessions[session.responder.user_id]:
                del self._user_sessions[session.responder.user_id]

    def _update_average_duration(self, duration_seconds: float) -> None:
        """Aktualisiert durchschnittliche Session-Dauer."""
        total_completed = self.metrics.completed_sessions
        if total_completed > 0:
            current_total = self.metrics.average_session_duration_seconds * (total_completed - 1)
            self.metrics.average_session_duration_seconds = (current_total + duration_seconds) / total_completed

    async def _emit_event(self, event: WebRTCEvent) -> None:
        """Sendet Event an alle Callbacks."""
        for callback in self._session_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Fehler in Session-Callback: {e}")

    async def _cleanup_loop(self) -> None:
        """Cleanup-Loop für abgelaufene Sessions."""
        while self._running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(self.cleanup_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Session-Cleanup: {e}")
                await asyncio.sleep(5)

    async def _cleanup_expired_sessions(self) -> None:
        """Bereinigt abgelaufene Sessions."""
        current_time = datetime.now(UTC)
        expired_sessions = []

        async with self._session_lock:
            for session_id, session in self._sessions.items():
                # Timeout prüfen
                time_since_activity = current_time - session.last_activity
                if time_since_activity > self.session_timeout:
                    expired_sessions.append(SessionCleanupInfo(
                        session_id=session_id,
                        last_activity=session.last_activity,
                        state=session.state,
                        cleanup_reason="timeout"
                    ))

        # Abgelaufene Sessions schließen
        for cleanup_info in expired_sessions:
            await self.close_session(cleanup_info.session_id, cleanup_info.cleanup_reason)
            logger.info(f"Session wegen Timeout bereinigt: {cleanup_info.session_id}")

    async def _close_all_sessions(self) -> None:
        """Schließt alle Sessions."""
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id, "shutdown")

    async def get_session_metrics(self) -> dict[str, Any]:
        """Gibt Session-Metriken zurück."""
        async with self._session_lock:
            self.metrics.last_updated = datetime.now(UTC)

            return {
                "total_sessions_created": self.metrics.total_sessions_created,
                "active_sessions": self.metrics.active_sessions,
                "completed_sessions": self.metrics.completed_sessions,
                "failed_sessions": self.metrics.failed_sessions,
                "average_session_duration_seconds": self.metrics.average_session_duration_seconds,
                "peak_concurrent_sessions": self.metrics.peak_concurrent_sessions,
                "session_timeout_seconds": self.session_timeout.total_seconds(),
                "max_sessions_per_user": self.max_sessions_per_user,
                "last_updated": self.metrics.last_updated.isoformat()
            }


def create_session_manager(config: WebRTCConfig | None = None) -> WebRTCSessionManager:
    """Factory-Funktion für WebRTC Session Manager.

    Args:
        config: WebRTC-Konfiguration

    Returns:
        Neue WebRTCSessionManager-Instanz
    """
    return WebRTCSessionManager(config)
