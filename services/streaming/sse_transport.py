"""KEI-Stream SSE (Server-Sent Events) Read-only Transport.

Stellt einen einfachen SSE-Endpoint bereit, der Frames aus einem Stream
als Event-Stream publiziert. Für niedrige Komplexität ist dies read-only.
Umfasst vollständige Prometheus-Metriken für Observability.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from kei_logging import get_logger

from .constants import DEFAULT_SSE_POLLING_INTERVAL_SECONDS
from .session import session_manager
from .sse_metrics import sse_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


router = APIRouter(prefix="/stream/sse", tags=["kei-stream-sse"])


async def _check_client_disconnect(request: Request, endpoint: str, session_id: str, stream_id: str) -> bool:
    """Prüft ob Client getrennt wurde und zeichnet Disconnect auf.

    Returns:
        True wenn Client getrennt wurde
    """
    if await request.is_disconnected():
        sse_metrics.record_client_disconnect(endpoint, session_id, "normal")
        logger.debug(f"SSE-Client getrennt: session={session_id}, stream={stream_id}")
        return True
    return False


async def _poll_and_process_frames(
    session_id: str,
    stream_id: str,
    last_seq: int,
    endpoint: str
) -> tuple[list, int]:
    """Pollt Frames vom Session Manager und verarbeitet sie.

    Returns:
        Tuple von (frames, new_last_seq)
    """
    polling_start = time.time()

    # Replay-Fenster pollend streamen
    replay = await session_manager.resume_from(session_id, stream_id, last_seq)

    # Polling-Dauer aufzeichnen
    polling_duration = time.time() - polling_start
    sse_metrics.record_polling_duration(endpoint, session_id, stream_id, polling_duration)

    # Replay-Fenster-Größe aktualisieren
    sse_metrics.update_replay_window_size(endpoint, session_id, stream_id, len(replay))

    # Neue last_seq berechnen
    new_last_seq = last_seq
    for frame in replay:
        new_last_seq = max(new_last_seq, frame.seq)

    return replay, new_last_seq


def _serialize_frame_to_sse(frame) -> bytes:
    """Serialisiert Frame zu SSE-Format.

    Returns:
        SSE-formatierte Bytes
    """
    data = json.dumps(frame.model_dump())
    return f"data: {data}\n\n".encode()


def _extract_frame_type(frame) -> str:
    """Extrahiert Frame-Type als String für Metriken.

    Returns:
        Frame-Type als String
    """
    return getattr(frame.type, "value", str(frame.type)) if hasattr(frame.type, "value") else str(frame.type)


@router.get("/{session_id}/{stream_id}")
async def stream_sse(session_id: str, stream_id: str, request: Request) -> StreamingResponse:
    """SSE-Endpoint für einen gegebenen Stream mit umfassenden Prometheus-Metriken.

    Achtung: Dies ist read-only und dient z. B. für Monitoring/Progress.
    Umfasst vollständige Observability für Verbindungsmanagement, Nachrichtenversand und Fehlerbehandlung.
    """
    # Endpoint-Identifikation für Metriken
    endpoint = f"/stream/sse/{session_id}/{stream_id}"
    connection_start_time = time.time()

    # Verbindungsstart aufzeichnen
    sse_metrics.record_connection_start(endpoint, session_id, "connected")

    logger.info(f"SSE-Verbindung gestartet: session={session_id}, stream={stream_id}")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        """Event-Generator mit umfassender Metriken-Aufzeichnung."""
        last_seq = 0
        disconnect_reason = "normal"

        try:
            while True:
                # Client-Disconnect prüfen
                if await _check_client_disconnect(request, endpoint, session_id, stream_id):
                    disconnect_reason = "client_disconnect"
                    break

                try:
                    # Frames vom Session Manager abrufen
                    replay, last_seq = await _poll_and_process_frames(
                        session_id, stream_id, last_seq, endpoint
                    )

                    # Frames verarbeiten und senden
                    for frame in replay:
                        # Frame serialisieren
                        message_bytes = _serialize_frame_to_sse(frame)

                        # Metriken für gesendete Nachricht aufzeichnen
                        frame_type = _extract_frame_type(frame)
                        sse_metrics.record_message_sent(
                            endpoint, session_id, stream_id, frame_type, len(message_bytes)
                        )

                        yield message_bytes

                except Exception as e:
                    # Fehler bei Replay/Polling aufzeichnen
                    sse_metrics.record_error(endpoint, session_id, "polling_error")
                    logger.exception(f"SSE-Polling-Fehler: session={session_id}, stream={stream_id}, error={e}")
                    disconnect_reason = "polling_error"
                    break

                # Polling-Intervall
                await asyncio.sleep(DEFAULT_SSE_POLLING_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            # Normale Cancellation (z.B. bei Server-Shutdown)
            disconnect_reason = "cancelled"
            logger.debug(f"SSE-Generator abgebrochen: session={session_id}, stream={stream_id}")

        except Exception as e:
            # Unerwarteter Fehler
            disconnect_reason = "unexpected_error"
            sse_metrics.record_error(endpoint, session_id, "generator_error")
            logger.exception(f"SSE-Generator-Fehler: session={session_id}, stream={stream_id}, error={e}")

        finally:
            # Verbindungsende aufzeichnen
            connection_duration = time.time() - connection_start_time
            sse_metrics.record_connection_end(endpoint, session_id, connection_duration, disconnect_reason)
            logger.info(f"SSE-Verbindung beendet: session={session_id}, stream={stream_id}, "
                       f"duration={connection_duration:.2f}s, reason={disconnect_reason}")

    try:
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        # Fehler beim Erstellen der StreamingResponse
        connection_duration = time.time() - connection_start_time
        sse_metrics.record_error(endpoint, session_id, "response_creation_error")
        sse_metrics.record_connection_end(endpoint, session_id, connection_duration, "response_error")
        logger.exception(f"SSE-Response-Erstellungsfehler: session={session_id}, stream={stream_id}, error={e}")
        raise
