"""WebSocket API Routes.

Sichere WebSocket-Endpunkte für Echtzeit-Kommunikation mit Enterprise-Grade Authentifizierung.
"""

import json
import time

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer

from auth.enterprise_auth import AuthContext
from auth.unified_enterprise_auth import require_websocket_auth
from kei_logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="", tags=["websocket"])
security = HTTPBearer(auto_error=False)





@router.websocket("/connect")
async def websocket_endpoint(websocket: WebSocket, auth: AuthContext = Depends(require_websocket_auth)):
    """Sichere WebSocket-Verbindung mit vereinheitlichter Authentifizierung.

    Unterstützt sowohl Authorization-Header (Bearer) als auch Query-Parameter (?token=, ?access_token=)
    über die Unified WebSocket Auth Dependency.
    """
    # Verbindung akzeptieren erst nach erfolgreicher Auth
    await websocket.accept()

    try:
        logger.info(f"🔐 Authentifizierte WebSocket-Verbindung: {auth.subject}")

        # Willkommensnachricht mit Authentifizierungsinfo
        welcome_message = {
            "type": "connection_established",
            "message": f"Willkommen {auth.subject}!",
            "authenticated": True,
            "scopes": [s.value for s in (auth.scopes or [])],
            "timestamp": time.time()
        }
        await websocket.send_text(json.dumps(welcome_message))

        # Echo-Loop mit Authentifizierungskontext
        while True:
            data = await websocket.receive_text()

            try:
                # Parse JSON wenn möglich
                message = json.loads(data)
                response = {
                    "type": "echo",
                    "original_message": message,
                    "user": auth.subject,
                    "timestamp": time.time()
                }
            except Exception:
                # Falls kein JSON, als Plain-Text Echoen
                response = {
                    "type": "echo",
                    "original_message": data,
                    "user": auth.subject,
                    "timestamp": time.time()
                }

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket-Verbindung getrennt")
    except Exception as e:
        logger.error(f"WebSocket-Fehler: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@router.get("/health")
async def websocket_health():
    """Gesundheitscheck für WebSocket-System."""
    return {"status": "healthy", "service": "websocket"}
