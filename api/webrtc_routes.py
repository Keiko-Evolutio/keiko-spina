"""WebRTC API Routes

FastAPI Routes für WebRTC-Integration im Voice-Service-System.
Bietet Endpoints für Signaling, Konfiguration und Monitoring.

@version 1.0.0
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel

from kei_logging import get_logger
from webrtc import (
    get_webrtc_service,
    get_webrtc_system_status,
    is_webrtc_system_healthy,
    perform_webrtc_health_check,
)
from webrtc.config import export_webrtc_config_for_frontend, get_webrtc_config
from webrtc.types import WebRTCSessionState

logger = get_logger(__name__)

# =============================================================================
# Authentication Dependencies
# =============================================================================

def get_current_user_id(
    user_id: str | None = Header(None, alias="X-User-ID"),
    authorization: str | None = Header(None)
) -> str:
    """Extrahiert User-ID aus Request-Headers.

    Prüft zuerst X-User-ID Header, dann Authorization Bearer Token.
    """
    # Direkte User-ID aus Header
    if user_id:
        return user_id

    # Fallback: User-ID aus Authorization Token extrahieren
    if authorization and authorization.startswith("Bearer "):
        try:
            import base64
            import json

            token = authorization.replace("Bearer ", "")
            parts = token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                # Base64 Padding hinzufügen falls nötig
                payload += "=" * (4 - len(payload) % 4)
                decoded = base64.b64decode(payload)
                jwt_data = json.loads(decoded)

                # User-ID aus verschiedenen JWT-Claims extrahieren
                extracted_user_id = (
                    jwt_data.get("sub") or
                    jwt_data.get("user_id") or
                    jwt_data.get("uid")
                )
                if extracted_user_id:
                    return str(extracted_user_id)
        except Exception:
            # JWT-Parsing fehlgeschlagen - weiter mit Fehler
            pass

    # Keine User-ID gefunden
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="User-ID erforderlich - X-User-ID Header oder Authorization Bearer Token"
    )

# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(
    prefix="/webrtc",
    tags=["WebRTC"],
    responses={404: {"description": "Not found"}}
)

# =============================================================================
# Request/Response Models
# =============================================================================

class WebRTCConfigResponse(BaseModel):
    """WebRTC Konfiguration Response."""
    iceServers: list[dict[str, Any]]
    iceTransportPolicy: str
    bundlePolicy: str
    audioCodecs: list[dict[str, Any]]
    audioConfig: dict[str, Any]

class WebRTCSessionInfo(BaseModel):
    """WebRTC Session Information."""
    session_id: str
    state: str
    initiator_user_id: str
    responder_user_id: str | None = None
    created_at: str
    connected_at: str | None = None
    last_activity: str

class WebRTCStatsResponse(BaseModel):
    """WebRTC Server Statistiken."""
    is_running: bool
    active_connections: int
    total_connections: int
    active_sessions: int
    total_sessions: int
    message_count: int
    users_online: int
    timestamp: str

class WebRTCHealthResponse(BaseModel):
    """WebRTC Health Check Response."""
    healthy: bool
    timestamp: str
    checks: dict[str, Any]

# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get("/config", response_model=WebRTCConfigResponse)
async def get_webrtc_configuration(
    user_id: str = Depends(get_current_user_id)
) -> WebRTCConfigResponse:
    """Holt WebRTC-Konfiguration für Frontend.

    Gibt ICE Server, Audio-Codecs und andere WebRTC-Einstellungen zurück.
    """
    try:
        config = get_webrtc_config()
        frontend_config = export_webrtc_config_for_frontend(config)

        logger.info(f"WebRTC-Konfiguration abgerufen für User {user_id}")

        return WebRTCConfigResponse(**frontend_config)

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der WebRTC-Konfiguration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Abrufen der WebRTC-Konfiguration"
        )

@router.get("/ice-servers")
async def get_ice_servers(
    user_id: str = Depends(get_current_user_id)
) -> dict[str, Any]:
    """Holt ICE Server-Konfiguration.

    Gibt STUN/TURN Server für NAT Traversal zurück.
    """
    try:
        logger.info(f"ICE Server-Konfiguration angefordert von User {user_id}")
        config = get_webrtc_config()

        ice_servers = []

        # STUN Server
        for stun_url in config.stun_servers:
            ice_servers.append({"urls": stun_url})

        # TURN Server
        for turn_config in config.turn_servers:
            ice_servers.append(turn_config)

        return {
            "iceServers": ice_servers,
            "iceTransportPolicy": "all"
        }

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der ICE Server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Abrufen der ICE Server"
        )

# =============================================================================
# Session Management Endpoints
# =============================================================================

@router.get("/sessions", response_model=list[WebRTCSessionInfo])
async def get_user_sessions(
    user_id: str = Depends(get_current_user_id)
) -> list[WebRTCSessionInfo]:
    """Holt aktive WebRTC-Sessions für User.
    """
    try:
        signaling_server = get_webrtc_service("signaling_server")
        if not signaling_server:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="WebRTC Signaling Server nicht verfügbar"
            )

        # User-Sessions aus Signaling Server holen
        user_sessions = []
        for session in signaling_server.sessions.values():
            if (session.initiator.user_id == user_id or
                (session.responder and session.responder.user_id == user_id)):

                session_info = WebRTCSessionInfo(
                    session_id=session.session_id,
                    state=session.state.value,
                    initiator_user_id=session.initiator.user_id,
                    responder_user_id=session.responder.user_id if session.responder else None,
                    created_at=session.created_at.isoformat(),
                    connected_at=session.connected_at.isoformat() if session.connected_at else None,
                    last_activity=session.last_activity.isoformat()
                )
                user_sessions.append(session_info)

        return user_sessions

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der User-Sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Abrufen der Sessions"
        )

@router.delete("/sessions/{session_id}")
async def terminate_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
) -> dict[str, str]:
    """Beendet WebRTC-Session.
    """
    try:
        signaling_server = get_webrtc_service("signaling_server")
        if not signaling_server:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="WebRTC Signaling Server nicht verfügbar"
            )

        # Session finden
        session = signaling_server.sessions.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session nicht gefunden"
            )

        # Berechtigung prüfen
        if (session.initiator.user_id != user_id and
            (not session.responder or session.responder.user_id != user_id)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Keine Berechtigung für diese Session"
            )

        # Session beenden
        session.state = WebRTCSessionState.DISCONNECTED

        # Connections in Session schließen
        if session_id in signaling_server.session_connections:
            connection_ids = list(signaling_server.session_connections[session_id])
            for connection_id in connection_ids:
                connection = signaling_server.connections.get(connection_id)
                if connection:
                    await connection.close(1000, "Session terminated by user")

        logger.info(f"WebRTC-Session {session_id} beendet von User {user_id}")

        return {"message": "Session erfolgreich beendet"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Beenden der Session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Beenden der Session"
        )

# =============================================================================
# Monitoring Endpoints
# =============================================================================

@router.get("/stats", response_model=WebRTCStatsResponse)
async def get_webrtc_stats(
    user_id: str = Depends(get_current_user_id)
) -> WebRTCStatsResponse:
    """Holt WebRTC Server-Statistiken.
    """
    try:
        logger.info(f"WebRTC-Statistiken angefordert von User {user_id}")
        signaling_server = get_webrtc_service("signaling_server")
        if not signaling_server:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="WebRTC Signaling Server nicht verfügbar"
            )

        stats = signaling_server.get_server_stats()
        return WebRTCStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der WebRTC-Statistiken: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Abrufen der Statistiken"
        )

@router.get("/health", response_model=WebRTCHealthResponse)
async def get_webrtc_health() -> WebRTCHealthResponse:
    """WebRTC Health Check.
    """
    try:
        health_status = await perform_webrtc_health_check()
        return WebRTCHealthResponse(**health_status)

    except Exception as e:
        logger.error(f"Fehler beim WebRTC Health Check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Health Check"
        )

@router.get("/status")
async def get_webrtc_status() -> dict[str, Any]:
    """WebRTC System Status.
    """
    try:
        return get_webrtc_system_status()

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des WebRTC-Status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Abrufen des Status"
        )

# =============================================================================
# WebSocket Signaling Endpoint
# =============================================================================

@router.websocket("/signaling")
async def websocket_signaling_endpoint(
    websocket: WebSocket,
    session_id: str | None = None,
    user_id: str | None = None
):
    """WebSocket Endpoint für WebRTC Signaling.

    Behandelt Offer/Answer-Austausch und ICE Candidate-Handling.
    """
    # Authentication (in Production sollte dies über Token erfolgen)
    if not user_id:
        await websocket.close(code=4001, reason="Authentication required")
        return

    # Signaling Server holen
    signaling_server = get_webrtc_service("signaling_server")
    if not signaling_server:
        await websocket.close(code=4003, reason="Signaling server unavailable")
        return

    # Client IP ermitteln
    client_ip = websocket.client.host if websocket.client else "unknown"

    try:
        # WebSocket-Verbindung an Signaling Server delegieren
        await signaling_server.handle_websocket_connection(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            ip_address=client_ip
        )

    except WebSocketDisconnect:
        logger.info(f"WebRTC Signaling WebSocket getrennt: {user_id}")
    except Exception as e:
        logger.error(f"Fehler in WebRTC Signaling WebSocket für {user_id}: {e}")
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except (ConnectionResetError, RuntimeError) as close_error:
            # WebSocket bereits geschlossen oder Verbindung unterbrochen
            logger.debug(f"WebSocket-Close fehlgeschlagen: {close_error}")

# =============================================================================
# Utility Endpoints
# =============================================================================

@router.post("/test-connection")
async def test_webrtc_connection(
    user_id: str = Depends(get_current_user_id)
) -> dict[str, Any]:
    """Testet WebRTC-Verbindungsfähigkeit.
    """
    try:
        logger.info(f"WebRTC Connection Test gestartet von User {user_id}")
        config = get_webrtc_config()

        # Basis-Connectivity-Test
        test_results = {
            "webrtc_available": is_webrtc_system_healthy(),
            "signaling_server_running": get_webrtc_service("signaling_server") is not None,
            "ice_servers_configured": len(config.stun_servers) > 0,
            "turn_servers_configured": len(config.turn_servers) > 0,
            "audio_codecs_available": len(config.audio_codecs) > 0
        }

        # Overall Test Result
        test_results["connection_possible"] = all([
            test_results["webrtc_available"],
            test_results["signaling_server_running"],
            test_results["ice_servers_configured"]
        ])

        return {
            "test_results": test_results,
            "recommendations": _get_connection_recommendations(test_results),
            "timestamp": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.error(f"Fehler beim WebRTC Connection Test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fehler beim Connection Test"
        )

def _get_connection_recommendations(test_results: dict[str, bool]) -> list[str]:
    """Gibt Empfehlungen basierend auf Test-Ergebnissen zurück."""
    recommendations = []

    if not test_results.get("webrtc_available"):
        recommendations.append("WebRTC-System ist nicht verfügbar - prüfen Sie die Server-Konfiguration")

    if not test_results.get("signaling_server_running"):
        recommendations.append("Signaling Server läuft nicht - starten Sie den WebRTC-Service")

    if not test_results.get("ice_servers_configured"):
        recommendations.append("Keine ICE Server konfiguriert - STUN/TURN Server erforderlich")

    if not test_results.get("turn_servers_configured"):
        recommendations.append("Keine TURN Server konfiguriert - für NAT Traversal empfohlen")

    if not test_results.get("audio_codecs_available"):
        recommendations.append("Keine Audio-Codecs konfiguriert - Opus wird empfohlen")

    if test_results.get("connection_possible"):
        recommendations.append("WebRTC-Verbindung sollte möglich sein")

    return recommendations
