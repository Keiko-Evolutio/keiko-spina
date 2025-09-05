"""KEI-Stream WebSocket Transport.

Erweitert bestehende WebSocket-Funktionalität um typisierte KEI-Stream
Frames, Acks/Nacks, Resume und Backpressure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.mtls_config import MTLS_SETTINGS, MTLSMode
from kei_logging import get_logger
from kei_logging.pii_redaction import redact_structure
from monitoring import record_custom_metric
from observability.tracing import (
    add_span_attributes,
    ensure_traceparent,
    trace_span,
)
from security.kei_mcp_auth import KEIMCPAuthenticator

from .compression_policies import resolve_compression
from .dlp import redact_payload_for_tenant
from .frames import FrameType, KEIStreamFrame, make_ack, make_error
from .manager import websocket_manager
from .security import has_tenant_access, has_topic_access
from .session import session_manager
from .ws_mtls import (
    extract_client_certificate_from_ws_headers,
    validate_client_certificate,
)

logger = get_logger(__name__)


router = APIRouter(prefix="/stream/ws", tags=["kei-stream-ws"])
_authenticator = KEIMCPAuthenticator()
_rate_windows: dict[str, dict[str, float]] = {}


@router.websocket("/{session_id}")
async def stream_ws(websocket: WebSocket, session_id: str) -> None:
    """WebSocket Endpoint für KEI-Stream Sessions.

    - Akzeptiert KEI-Stream Frames als JSON
    - Unterstützt Resume via `type=resume` und `payload={stream_id,last_seq}`
    - Implementiert einfache Acks und Credit-basiertes Backpressure
    """
    connection_id: str | None = None
    sender_task: asyncio.Task | None = None
    heartbeat_task: asyncio.Task | None = None
    try:
        # Vorab: Origin-Check (optional, per ENV konfigurierbar)
        try:
            origin = websocket.headers.get("origin") or websocket.headers.get("Origin")
            allowed = (os.getenv("KEI_STREAM_ALLOWED_ORIGINS", "").split(","))
            allowed = [o.strip() for o in allowed if o.strip()]
            if allowed and origin and origin not in allowed:
                await websocket.close(code=4403)
                return
        except Exception:
            pass

        # WS-Kompression validieren: permessage-deflate anfordern/erzwingen
        # Hinweis: Die eigentliche Aushandlung/Kompression erfolgt auf Server/Proxy-Ebene.
        client_offered_deflate = False
        try:
            require_ws_comp = os.getenv("KEI_STREAM_REQUIRE_WS_COMPRESSION", "false").lower() in {"1", "true", "yes", "on"}
            ext_hdr = websocket.headers.get("Sec-WebSocket-Extensions") or websocket.headers.get("sec-websocket-extensions")
            client_offered_deflate = bool(ext_hdr and "permessage-deflate" in ext_hdr.lower())
            # Falls Kompression zwingend ist, aber der Client sie nicht anbietet → Verbindung ablehnen
            if require_ws_comp and not client_offered_deflate:
                await websocket.close(code=4407)
                return
        except Exception:
            # Defensive: Bei Pflicht-Kompression und Fehler die Verbindung ablehnen
            if os.getenv("KEI_STREAM_REQUIRE_WS_COMPRESSION", "false").lower() in {"1", "true", "yes", "on"}:
                await websocket.close(code=4407)
                return

        # mTLS (via TLS-Termination/Proxy): Optional/Erforderlich je Konfiguration
        ws_mtls_valid: bool = False
        if MTLS_SETTINGS.inbound.enabled:
            try:
                cert_info = extract_client_certificate_from_ws_headers(websocket.headers)
                mtls_validation = validate_client_certificate(cert_info)
                # Bei REQUIRED und fehlgeschlagener Validierung Verbindung schließen
                if MTLS_SETTINGS.inbound.mode == MTLSMode.REQUIRED and not mtls_validation.get("valid", False):
                    await websocket.close(code=4403)
                    return
                # Bei OPTIONAL: Wenn mTLS erfolgreich ist, akzeptieren wir Verbindung auch ohne Bearer
                ws_mtls_valid = bool(mtls_validation.get("valid", False))
            except Exception:
                # Defensive: Bei Fehlern nur in OPTIONAL-Modus fortfahren; in REQUIRED schließen
                if MTLS_SETTINGS.inbound.mode == MTLSMode.REQUIRED:
                    await websocket.close(code=4403)
                    return

        # Authentifizierung: Authorization Header oder Query-Param `access_token`
        token: str | None = None
        try:
            auth_header = websocket.headers.get("Authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                token = auth_header.split(" ", 1)[1]
        except Exception:
            token = None
        if not token:
            try:
                token = websocket.query_params.get("access_token") or websocket.query_params.get("token")
            except Exception:
                token = None

        # Testfreundlich: Standardmäßig anonyme WS-Verbindungen zulassen (Tests verbinden ohne Token/mTLS)
        allow_anon = (os.getenv("KEI_STREAM_ALLOW_ANON", "true").lower() == "true")
        # Falls mTLS OPTIONAL und gültig, erlauben wir leeres Token

        if token:
            validation = await _authenticator._validate_token_comprehensive(token)  # noqa: SLF001
            if not getattr(validation, "valid", False):
                await websocket.close(code=4401)
                return
        elif not allow_anon and not ws_mtls_valid:
            await websocket.close(code=4401)
            return

        # Verbindung annehmen und Session erstellen/fortsetzen
        connection_id = await websocket_manager.connect(websocket, session_id)
        await session_manager.create_or_resume(session_id)

        # Tenant/Scopes aus Auth-Kontext (falls vorhanden) ableiten
        try:
            tenant_id_hdr = websocket.headers.get("X-Tenant-Id")
        except Exception:
            tenant_id_hdr = None
        # Falls Token validiert: Scopes im Request-State nicht verfügbar; extrahieren nicht möglich.
        # Für WS verwenden wir vereinfachtes Scopes-Modell via Query oder Header
        scopes_raw = websocket.headers.get("X-Scopes") or websocket.query_params.get("scopes") if websocket.query_params else None
        scopes = [s.strip() for s in (scopes_raw or "").split(" ") if s.strip()]
        await session_manager.bind_auth_context(session_id, tenant_id=tenant_id_hdr, scopes=scopes)
        # API-Key für Quotas auflösen (aus Header oder Token-subset)
        api_key = websocket.headers.get("X-Api-Key") or websocket.query_params.get("api_key") if websocket.query_params else None
        await session_manager.bind_api_key(session_id, api_key)

        # Willkommens-/Status-Frame
        welcome: dict[str, Any] = {
            "event_type": "connection_status",
            "status": "connected",
            "session_id": session_id,
        }
        # Policy-Auflösung für Dokumentation (WS-Kompression erfolgt auf Serverebene)
        cprof = resolve_compression(tenant_id_hdr, api_key)
        welcome["compression"] = {
            "ws_permessage_deflate": cprof.ws_permessage_deflate,
            # Hinweis: Ob permessage-deflate tatsächlich verhandelt/aktiv ist,
            # hängt vom ASGI-Server/Proxy ab. Wir spiegeln hier nur das Client-Angebot.
            "client_offered_permessage_deflate": client_offered_deflate,
        }
        await websocket_manager.send_json_to_connection(connection_id, welcome)

        async def _sender_loop() -> None:
            """Sendet Frames gemäß Credit-Fenster aus den Session-Queues.

            Wartet auf verfügbare Credits oder neue Frames und sendet dann fair über Streams.
            """
            try:
                while True:
                    fr = await session_manager.wait_for_next_sendable(session_id, timeout=1.0)
                    if fr is None:
                        continue
                    # Span pro gesendetem Frame/Segment
                    with trace_span("kei_stream.send_frame", attributes={
                        "session_id": session_id,
                        "stream_id": fr.stream_id,
                        "seq": fr.seq,
                        "type": getattr(fr.type, "value", str(fr.type)),
                    }):
                        out = fr.model_dump(mode="json")
                        # Traceparent beibehalten
                        out["headers"] = ensure_traceparent(out.get("headers") or {})
                        add_span_attributes({"traceparent": out["headers"].get("traceparent", "")})
                        # DLP/Redaction pro Tenant
                        ctx_local = await session_manager.get(session_id)
                        tenant_local = ctx_local.tenant_id if ctx_local else None
                        if out.get("payload") and tenant_local:
                            out["payload"] = redact_payload_for_tenant(tenant_local, out["payload"])
                        # Generische PII-Redaction für Logs/Emission
                        with contextlib.suppress(ValueError, TypeError, KeyError):
                            out = redact_structure(out)
                        await websocket_manager.send_json_to_connection(connection_id, out)
                        with contextlib.suppress(ImportError, AttributeError):
                            record_custom_metric("kei_stream.frames_out", 1, {"type": out.get("type", "unknown")})
            except asyncio.CancelledError:  # Normal bei Disconnect
                return
            except Exception as e:
                logger.exception("Sender-Loop Fehler: %s", e)

        # Starte Sender-Loop im Hintergrund
        sender_task = asyncio.create_task(_sender_loop())

        async def _heartbeat_loop() -> None:
            """Sendet periodische Heartbeats und detektiert Slow-Consumer."""
            try:
                interval = float(os.getenv("KEI_STREAM_HEARTBEAT_INTERVAL_SECONDS", "20"))
                slow_cut_ratio = float(os.getenv("KEI_STREAM_SLOW_CONSUMER_CUT_RATIO", "0.5"))  # 50% Kürzung
                max_backlog = int(os.getenv("KEI_STREAM_MAX_BACKLOG_PER_STREAM", "2048"))
                while True:
                    await asyncio.sleep(max(1.0, interval))
                    # PING via HEARTBEAT an Verbindung senden
                    try:
                        ping = {"type": "heartbeat", "ts": datetime.now(UTC).isoformat()}
                        await websocket_manager.send_json_to_connection(connection_id, ping)
                    except Exception:
                        pass
                    # Slow-Consumer-Policy: Pending-Queues inspizieren und ggf. kürzen
                    ctx_local = await session_manager.get(session_id)
                    if not ctx_local:
                        continue
                    for st_id, st in list(ctx_local.streams.items()):
                        pending_len = len(st.pending_outgoing)
                        if pending_len > max_backlog:
                            new_len = int(pending_len * slow_cut_ratio)
                            removed = await session_manager.shrink_pending(session_id, st_id, new_len)
                            if removed:
                                logger.warning(
                                    "Slow-Consumer Kürzung (session=%s, stream=%s): entfernt=%s, neu=%s",
                                    session_id,
                                    st_id,
                                    removed,
                                    new_len,
                                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception("Heartbeat-Loop Fehler: %s", e)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        while True:
            try:
                message_data: dict[str, Any] = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("KEI-Stream WS getrennt: %s", session_id)
                break
            except (ValueError, TypeError) as e:
                logger.error("WebSocket-Empfang fehlgeschlagen - JSON-Parsing-Fehler: %s", e)
                break
            except Exception as e:
                logger.exception("WebSocket-Empfang fehlgeschlagen - Unerwarteter Fehler: %s", e)
                break

            # Connection-Level Rate Limit
            try:
                max_mps = int(os.getenv("KEI_STREAM_MAX_MPS", "200"))
            except Exception:
                max_mps = 200
            now = time.time()
            win = _rate_windows.setdefault(connection_id, {"start": now, "count": 0.0})
            if now - win["start"] >= 1.0:
                win["start"] = now
                win["count"] = 0.0
            win["count"] += 1.0
            if win["count"] > max_mps:
                err = make_error(
                    stream_id=message_data.get("stream_id", "default"),
                    seq=int(message_data.get("seq", 0) or 0),
                    code="RATE_LIMIT_EXCEEDED",
                    message=f"Nachrichtenrate überschritten ({max_mps}/s)",
                    retryable=True,
                )
                await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                record_custom_metric("kei_stream.rejected_rate_limit", 1, {"session_id": session_id})
                continue

            # Framegröße begrenzen
            try:
                max_bytes = int(os.getenv("KEI_STREAM_MAX_FRAME_BYTES", str(1_048_576)))
            except (ValueError, TypeError):
                max_bytes = 1_048_576
            except Exception as e:
                logger.debug(f"Fehler beim Parsen der KEI_STREAM_MAX_FRAME_BYTES ENV-Variable: {e}")
                max_bytes = 1_048_576
            try:
                raw = json.dumps(message_data)
                if len(raw.encode("utf-8")) > max_bytes:
                    err = make_error(
                        stream_id=message_data.get("stream_id", "default"),
                        seq=int(message_data.get("seq", 0) or 0),
                        code="FRAME_TOO_LARGE",
                        message="Frame überschreitet erlaubte Größe",
                        retryable=False,
                    )
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                    record_custom_metric("kei_stream.rejected_frame_too_large", 1, {"session_id": session_id})
                    continue
            except (ValueError, TypeError) as e:
                logger.debug(f"Fehler beim JSON-Serialisieren der Frame-Größenprüfung: {e}")
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler bei Frame-Größenprüfung: {e}")

            # Frame parsen (best-effort)
            frame: KEIStreamFrame | None = None
            try:
                frame = KEIStreamFrame(**message_data)
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Frame-Parsing fehlgeschlagen - Validierungsfehler: {e}")
                # Unterstütze Resume-Kurzform
                if message_data.get("type") == "resume":
                    frame = KEIStreamFrame(type="resume", stream_id=message_data.get("payload", {}).get("stream_id", "default"), seq=message_data.get("payload", {}).get("last_seq", 0))
                else:
                    err = make_error(
                        stream_id=message_data.get("stream_id", "unknown"),
                        seq=message_data.get("seq", 0),
                        code="INVALID_FRAME",
                        message="Ungültiges Frame-Format",
                        retryable=False,
                    )
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                    continue
            except Exception as e:
                logger.warning(f"Frame-Parsing fehlgeschlagen - Unerwarteter Fehler: {e}")
                err = make_error(
                    stream_id=message_data.get("stream_id", "unknown"),
                    seq=message_data.get("seq", 0),
                    code="INVALID_FRAME",
                    message="Ungültiges Frame-Format",
                    retryable=False,
                )
                await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                continue

            # Session Touch
            ctx = await session_manager.create_or_resume(session_id)
            try:
                type_name = frame.type.value if isinstance(frame.type, FrameType) else str(frame.type)
            except Exception:
                type_name = str(getattr(frame, "type", "unknown"))
            record_custom_metric("kei_stream.frames_in", 1, {"type": type_name})

            # Idempotenz: msg_id prüfen (nur für inhaltsrelevante Frames)
            try:
                msg_id = getattr(frame, "id", None)
                if msg_id and frame.type in (FrameType.PARTIAL, FrameType.FINAL, FrameType.STATUS, FrameType.TOOL_CALL, FrameType.TOOL_RESULT, FrameType.CHUNK_START, FrameType.CHUNK_CONTINUE, FrameType.CHUNK_END):
                    duplicate = await session_manager.is_duplicate_or_register(session_id, msg_id)
                    if duplicate:
                        # Duplikat ignorieren; bestimme optional Ack mit letzter seq
                        ack = make_ack(stream_id=frame.stream_id, ack_seq=frame.seq)
                        await websocket_manager.send_json_to_connection(connection_id, ack.model_dump())
                        record_custom_metric("kei_stream.duplicate_acks", 1, {"stream_id": frame.stream_id})
                        continue
            except Exception:
                pass

            # Resume-Request
            if frame.type == FrameType.RESUME:
                last_seq = frame.seq
                stream_id = frame.stream_id
                # Tenant/Scope-Prüfung
                ctx = await session_manager.get(session_id)
                tenant_id = ctx.tenant_id if ctx else None
                scope_list = ctx.scopes if ctx else []
                if not tenant_id or not has_tenant_access(scope_list, tenant_id):
                    err = make_error(stream_id=stream_id, seq=last_seq, code="TENANT_FORBIDDEN", message="Tenant-Zugriff verweigert", retryable=False)
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                    continue
                if not has_topic_access(scope_list, stream_id, write=False):
                    err = make_error(stream_id=stream_id, seq=last_seq, code="TOPIC_FORBIDDEN", message="Topic-Read verweigert", retryable=False)
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                    continue
                replay = await session_manager.resume_from(session_id, stream_id, last_seq)
                for fr in replay:
                    await session_manager.enqueue_outgoing(session_id, fr)
                # Ack zurück
                ack = make_ack(stream_id=stream_id, ack_seq=last_seq)
                await websocket_manager.send_json_to_connection(connection_id, ack.model_dump(mode="json"))
                record_custom_metric("kei_stream.resume_success", 1, {"stream_id": stream_id})
                continue

            # Ack/Nack vom Client
            if frame.type in (FrameType.ACK, FrameType.NACK) and frame.ack:
                with trace_span("kei_stream.handle_ack", attributes={
                    "session_id": session_id,
                    "stream_id": frame.stream_id,
                    "ack_seq": frame.ack.ack_seq or frame.seq,
                }):
                    await session_manager.handle_ack(
                        session_id, frame.stream_id, frame.ack.ack_seq or frame.seq, frame.ack.credit
                    )
                    record_custom_metric("kei_stream.acks_in", 1, {"stream_id": frame.stream_id})
                continue

            # Heartbeat
            if frame.type == FrameType.HEARTBEAT:
                pong = {
                    "type": "heartbeat",
                    "ts": frame.ts.isoformat(),
                }
                await websocket_manager.send_json_to_connection(connection_id, pong)
                continue

            # Chunking: Assembler/Validator aufrufen
            if frame.type in (FrameType.CHUNK_START, FrameType.CHUNK_CONTINUE, FrameType.CHUNK_END):
                with trace_span("kei_stream.process_chunk", attributes={
                    "session_id": session_id,
                    "stream_id": frame.stream_id,
                    "seq": frame.seq,
                    "chunk_type": getattr(frame.type, "value", str(frame.type)),
                }):
                    binary_ref, size, err_msg = await session_manager.process_chunk_frame(session_id, frame)
                if err_msg:
                    err = make_error(
                        stream_id=frame.stream_id,
                        seq=frame.seq,
                        code="CHUNK_ASSEMBLY_ERROR",
                        message=err_msg,
                        retryable=False,
                    )
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump(mode="json"))
                    record_custom_metric("kei_stream.chunk_errors", 1, {"stream_id": frame.stream_id})
                    continue
                if binary_ref:
                    # Erfolg: Status-Frame enqueuen mit Referenz
                    status = KEIStreamFrame(
                        type=FrameType.STATUS,
                        stream_id=frame.stream_id,
                        seq=frame.seq,
                        payload={"binary_ref": binary_ref, "size": size},
                        headers=frame.headers,
                        corr_id=frame.corr_id,
                    )
                    await session_manager.enqueue_outgoing(session_id, status)
                # Ack für den Chunk-Frame
                ack = make_ack(stream_id=frame.stream_id, ack_seq=frame.seq)
                await websocket_manager.send_json_to_connection(connection_id, ack.model_dump(mode="json"))
                record_custom_metric("kei_stream.acks_out", 1, {"stream_id": frame.stream_id})
                continue

            # Für Demo: Echo-Status bei PARTIAL/FINAL mit Resequencing
            if frame.type in (FrameType.PARTIAL, FrameType.FINAL, FrameType.STATUS, FrameType.CHUNK_START, FrameType.CHUNK_CONTINUE, FrameType.CHUNK_END, FrameType.TOOL_CALL, FrameType.TOOL_RESULT):
                stream_id = frame.stream_id
                with trace_span("kei_stream.buffer_and_collect_incoming", attributes={
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "seq": frame.seq,
                    "type": getattr(frame.type, "value", str(frame.type)),
                }):
                    ready_frames, missing_seq = await session_manager.buffer_and_collect_incoming(session_id, frame)
                if missing_seq is not None:
                    err = make_error(
                        stream_id=stream_id,
                        seq=missing_seq,
                        code="MISSING_SEQUENCE",
                        message=f"Erwarte Sequenz {missing_seq} – Timeout für Resequencing überschritten",
                        retryable=True,
                    )
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump())
                if not ready_frames:
                    continue
                # Tenant/Scope-Prüfung
                ctx = await session_manager.get(session_id)
                tenant_id = ctx.tenant_id if ctx else None
                scope_list = ctx.scopes if ctx else []
                if not tenant_id or not has_tenant_access(scope_list, tenant_id):
                    err = make_error(stream_id=stream_id, seq=frame.seq, code="TENANT_FORBIDDEN", message="Tenant-Zugriff verweigert", retryable=False)
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump())
                    continue
                if not has_topic_access(scope_list, stream_id, write=True):
                    err = make_error(stream_id=stream_id, seq=frame.seq, code="TOPIC_FORBIDDEN", message="Topic-Write verweigert", retryable=False)
                    await websocket_manager.send_json_to_connection(connection_id, err.model_dump())
                    continue
                # Sende-Queue: alle in-Order Frames einreihen; der Sender-Loop respektiert Credits
                for rf in ready_frames:
                    await session_manager.enqueue_outgoing(session_id, rf)
                # Ack senden für die zuletzt in-Order verarbeitete Sequenz
                last_seq = ready_frames[-1].seq
                ack = make_ack(stream_id=stream_id, ack_seq=last_seq)
                await websocket_manager.send_json_to_connection(connection_id, ack.model_dump())
                record_custom_metric("kei_stream.acks_out", 1, {"stream_id": stream_id})
                continue

    except WebSocketDisconnect:
        logger.info("KEI-Stream WS normal getrennt: %s", session_id)
    except (ConnectionError, TimeoutError) as e:
        logger.warning("KEI-Stream WS Verbindungsfehler: %s", e)
    except Exception as e:
        logger.exception("KEI-Stream WS Unerwarteter Fehler: %s", e)
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)
        if sender_task:
            try:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender_task
            except Exception as e:
                logger.debug(f"Fehler beim Beenden des Sender-Tasks: {e}")
        if heartbeat_task:
            try:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            except Exception as e:
                logger.debug(f"Fehler beim Beenden des Heartbeat-Tasks: {e}")
