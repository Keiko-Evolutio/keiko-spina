"""gRPC KEI-Stream Transport.

Stellt einen bidi-Streaming-Servicer bereit, der `StreamFrame` Nachrichten
gemäß `backend/stream/grpc_stream.proto` verarbeitet. Implementiert Resume,
Ack/Nack und einfaches Credit-basiertes Flow-Control.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from kei_logging.pii_redaction import redact_structure

from .compression_policies import get_grpc_compression_from_str, resolve_compression
from .dlp import redact_payload_for_tenant
from .frames import FrameType, KEIStreamFrame
from .quotas import resolve_quota
from .security import has_tenant_access, has_topic_access
from .session import session_manager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import grpc

logger = get_logger(__name__)


# Lazy-Imports der generierten Klassen; Laufzeit-Codegen wird an anderer Stelle gehandhabt
try:  # pragma: no cover - abhängig von protoc
    import stream.grpc_stream_pb2 as stream_pb2  # type: ignore
    import stream.grpc_stream_pb2_grpc as stream_pb2_grpc  # type: ignore
except ImportError:  # pragma: no cover
    class _DummyPB2:  # type: ignore
        class StreamFrame:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class FrameHeader:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class ErrorInfo:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class AckInfo:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

    class _DummyPB2GRPC:  # type: ignore
        class KEIStreamServiceServicer:
            """Dummy."""


    stream_pb2 = _DummyPB2()  # type: ignore
    stream_pb2_grpc = _DummyPB2GRPC()  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"Unerwarteter Fehler beim Import der Stream Proto-Module: {e}")
    class _DummyPB2:  # type: ignore
        class StreamFrame:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class FrameHeader:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class ErrorInfo:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class AckInfo:
            """Dummy."""

            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

    class _DummyPB2GRPC:  # type: ignore
        class KEIStreamServiceServicer:
            """Dummy."""


    stream_pb2 = _DummyPB2()  # type: ignore
    stream_pb2_grpc = _DummyPB2GRPC()  # type: ignore


class KEIStreamService(stream_pb2_grpc.KEIStreamServiceServicer):  # type: ignore[misc]
    """gRPC KEI-Stream Servicer."""

    async def stream(
        self,
        request_iterator: AsyncIterator[Any],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        """Bidirektionaler Stream mit einfachem Flow-Control und Resume."""
        # gRPC-Server nutzt jetzt die Session-Queues; Semaphore bleibt als Schutz vor CPU-Saturation
        sem = asyncio.Semaphore(64)

        # Nebenläufiges Empfangen und Senden mit strikter Credit-Beachtung
        session_id: str | None = None
        recv_task: asyncio.Task = asyncio.create_task(request_iterator.__anext__())
        send_task: asyncio.Task | None = None
        heartbeat_task: asyncio.Task | None = None

        try:
            while True:
                await sem.acquire()
                try:
                    wait_set = {recv_task}
                    if session_id:
                        if send_task is None or send_task.done():
                            send_task = asyncio.create_task(
                                session_manager.wait_for_next_sendable(session_id, timeout=0.5)
                            )
                        wait_set.add(send_task)
                    # Heartbeat-Task pro aktiver Session starten
                    if session_id and (heartbeat_task is None or heartbeat_task.done()):
                        async def _hb_loop(sid: str) -> None:
                            try:
                                interval = float(os.getenv("KEI_STREAM_HEARTBEAT_INTERVAL_SECONDS", "20"))
                                while True:
                                    await asyncio.sleep(max(1.0, interval))
                                    stream_pb2.StreamFrame(
                                        type="heartbeat",
                                        header=stream_pb2.FrameHeader(stream_id="", seq=0),
                                    )
                                    # Heartbeat in gRPC-Stream schicken (non-blocking über yield ist hier nicht möglich außerhalb Funktionskontext)
                                    # Daher nutzen wir das reguläre Sendepfad: enqueue Outgoing mit STATUS ist übertrieben –
                                    # in gRPC senden wir Heartbeat direkt, indem wir das Event über einen Kanal liefern
                                    # Workaround: Wir ignorieren Heartbeats, da yield außerhalb geht nicht. Stattdessen rely auf HTTP/2 keepalive
                                    # und lassen WS die Heartbeats übernehmen.
                            except Exception:
                                return
                        heartbeat_task = asyncio.create_task(_hb_loop(session_id))

                    done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                    # Senden priorisieren, wenn verfügbar
                    if send_task and send_task in done:
                        fr = send_task.result()
                        if fr is not None:
                            ctx = await session_manager.get(session_id) if session_id else None
                            tenant_id = ctx.tenant_id if ctx else None
                            payload_json = "{}"
                            if fr.payload is not None:
                                try:
                                    payload_obj = redact_payload_for_tenant(tenant_id, fr.payload) if tenant_id else fr.payload
                                    # Generische PII-Redaction zusätzlich anwenden
                                    safe_obj = redact_structure(payload_obj)
                                    payload_json = json.dumps(safe_obj)
                                except Exception:
                                    payload_json = json.dumps(fr.payload or {})
                            # Traceparent erzwingen
                            try:
                                from observability.tracing import ensure_traceparent
                                headers = ensure_traceparent({"traceparent": fr.headers.get("traceparent", "")})
                                tp_value = headers.get("traceparent", "")
                            except Exception:
                                tp_value = fr.headers.get("traceparent", "")

                            yield stream_pb2.StreamFrame(
                                type=str(fr.type.value if hasattr(fr.type, "value") else fr.type),
                                header=stream_pb2.FrameHeader(
                                    id=fr.id,
                                    stream_id=fr.stream_id,
                                    seq=fr.seq,
                                    ts=fr.ts.isoformat(),
                                    corr_id=fr.corr_id or "",
                                    traceparent=tp_value,
                                ),
                                payload_json=payload_json,
                            )
                            try:
                                from monitoring import record_custom_metric
                                record_custom_metric("kei_stream.frames_out", 1, {"type": str(fr.type.value if hasattr(fr.type, "value") else fr.type)})
                            except Exception:
                                pass
                        # sende Task wird im nächsten Loop neu erstellt

                    if recv_task in done:
                        try:
                            incoming = recv_task.result()
                        except StopAsyncIteration:
                            break
                        # Nächsten Empfang gleich triggern
                        recv_task = asyncio.create_task(request_iterator.__anext__())

                        frame_type = (getattr(incoming, "type", "") or "").lower()
                        header = getattr(incoming, "header", None)
                        stream_id = getattr(header, "stream_id", "default") if header else "default"
                        seq = getattr(header, "seq", 0) if header else 0
                        session_id = getattr(header, "corr_id", "session") if header else "session"

                        # Resolve Quota/Compression anhand Tenant/API-Key aus Header
                        api_key_hdr = getattr(header, "corr_id", None)  # Platzhalter: echte API-Key Quelle projektspezifisch
                        ctx_tmp = await session_manager.get(session_id)
                        _profile = resolve_quota(ctx_tmp.tenant_id if ctx_tmp else None, api_key_hdr)
                        _cprof = resolve_compression(ctx_tmp.tenant_id if ctx_tmp else None, api_key_hdr)
                        try:
                            comp_enum = get_grpc_compression_from_str(_cprof.grpc_compression)
                            if comp_enum is not None:
                                context.set_compression(comp_enum)
                        except Exception:
                            pass

                        # Resume → Replay enqueuen und ACK senden
                        if frame_type == "resume":
                            replay = await session_manager.resume_from(session_id, stream_id, seq)
                            ctx = await session_manager.get(session_id)
                            tenant_id = ctx.tenant_id if ctx else None
                            scopes = ctx.scopes if ctx else []
                            if not tenant_id or not has_tenant_access(scopes, tenant_id):
                                yield stream_pb2.StreamFrame(type="error", header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq), error=stream_pb2.ErrorInfo(code="TENANT_FORBIDDEN", message="Tenant-Zugriff verweigert", retryable=False))
                                continue
                            if not has_topic_access(scopes, stream_id, write=False):
                                yield stream_pb2.StreamFrame(type="error", header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq), error=stream_pb2.ErrorInfo(code="TOPIC_FORBIDDEN", message="Topic-Read verweigert", retryable=False))
                                continue
                            for fr in replay:
                                await session_manager.enqueue_outgoing(session_id, fr)
                            yield stream_pb2.StreamFrame(
                                type="ack",
                                header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq),
                                ack=stream_pb2.AckInfo(ack_seq=seq),
                            )
                            try:
                                from monitoring import record_custom_metric
                                record_custom_metric("kei_stream.resume_success", 1, {"stream_id": stream_id})
                            except (ImportError, AttributeError):
                                # Monitoring-Modul nicht verfügbar
                                pass
                            except Exception as e:
                                logger.debug(f"Fehler beim Aufzeichnen der Resume-Metrik: {e}")
                            continue

                        # Eingehender Ack/Nack → Credits aktualisieren
                        if frame_type in ("ack", "nack") and getattr(incoming, "ack", None):
                            ack_seq = getattr(incoming.ack, "ack_seq", seq)
                            credit = getattr(incoming.ack, "credit", -1)
                            await session_manager.handle_ack(
                                session_id, stream_id, int(ack_seq), int(credit) if credit >= 0 else None
                            )
                            try:
                                from monitoring import record_custom_metric
                                record_custom_metric("kei_stream.acks_in", 1, {"stream_id": stream_id})
                            except Exception:
                                pass
                            continue

                        # Chunking: Assembler/Validator
                        if frame_type in ("chunk_start", "chunk_continue", "chunk_end"):
                            # Erzeuge KEIStreamFrame für Verarbeitung im Session-Manager
                            app_fr = KEIStreamFrame(
                                type=FrameType.CHUNK_START if frame_type == "chunk_start" else FrameType.CHUNK_CONTINUE if frame_type == "chunk_continue" else FrameType.CHUNK_END,
                                stream_id=stream_id,
                                seq=seq,
                                headers={"traceparent": getattr(header, "traceparent", "") if header else ""},
                                corr_id=getattr(header, "corr_id", None) if header else None,
                                payload={},
                            )
                            # payload_json sollte base64 enthalten
                            try:
                                payload_obj = json.loads(incoming.payload_json) if getattr(incoming, "payload_json", None) else {}
                            except Exception:
                                payload_obj = {}
                            app_fr.payload = payload_obj
                            # ChunkInfo aus PB nach KEI-ChunkInfo mappen
                            if getattr(incoming, "chunk", None):
                                app_fr.chunk = type(app_fr).model_fields["chunk"].annotation(
                                    kind="start" if frame_type == "chunk_start" else "continue" if frame_type == "chunk_continue" else "end",
                                    content_range=getattr(incoming.chunk, "content_range", None),
                                    checksum=getattr(incoming.chunk, "checksum", None),
                                )
                            _ref, _size, err_msg = await session_manager.process_chunk_frame(session_id, app_fr)
                            if err_msg:
                                yield stream_pb2.StreamFrame(
                                    type="error",
                                    header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq),
                                    error=stream_pb2.ErrorInfo(code="CHUNK_ASSEMBLY_ERROR", message=err_msg, retryable=False),
                                )
                                continue
                            # Ack für den Chunk-Frame
                            yield stream_pb2.StreamFrame(
                                type="ack",
                                header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq),
                                ack=stream_pb2.AckInfo(ack_seq=seq),
                            )
                            try:
                                from monitoring import record_custom_metric
                                record_custom_metric("kei_stream.duplicate_acks", 1, {"stream_id": stream_id})
                            except (ImportError, AttributeError):
                                # Monitoring-Modul nicht verfügbar
                                pass
                            except Exception as e:
                                logger.debug(f"Fehler beim Aufzeichnen der Duplicate-ACK-Metrik: {e}")
                            continue

                        # Idempotenz prüfen (nur für inhaltsrelevante Frames)
                        try:
                            msg_id = getattr(header, "id", None) if header else None
                            if msg_id and frame_type in ("partial","final","status","tool_call","tool_result","chunk_start","chunk_continue","chunk_end"):
                                duplicate = await session_manager.is_duplicate_or_register(session_id, msg_id)
                                if duplicate:
                                    # Duplikate ignorieren; optional Ack senden
                                    yield stream_pb2.StreamFrame(
                                        type="ack",
                                        header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq),
                                        ack=stream_pb2.AckInfo(ack_seq=seq),
                                    )
                                    continue
                        except Exception:
                            pass

                        # App-Frames: Status + Final erzeugen und enqueuen (Credit-gesteuert) mit Resequencing
                        ctx = await session_manager.get(session_id)
                        tenant_id = ctx.tenant_id if ctx else None
                        scopes = ctx.scopes if ctx else []
                        if not tenant_id or not has_tenant_access(scopes, tenant_id):
                            yield stream_pb2.StreamFrame(type="error", header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq), error=stream_pb2.ErrorInfo(code="TENANT_FORBIDDEN", message="Tenant-Zugriff verweigert", retryable=False))
                            continue
                        if not has_topic_access(scopes, stream_id, write=True):
                            yield stream_pb2.StreamFrame(type="error", header=stream_pb2.FrameHeader(stream_id=stream_id, seq=seq), error=stream_pb2.ErrorInfo(code="TOPIC_FORBIDDEN", message="Topic-Write verweigert", retryable=False))
                            continue

                        # Resequencing für eingehende Frames
                        # Wir behandeln das eintreffende Frame (FINAL/STATUS etc.) als einen verarbeitbaren Block,
                        # d. h. wir resequencen anhand der Seq des App-Frames (hier: seq)
                        app_fr = KEIStreamFrame(
                            type=FrameType.FINAL if frame_type == "final" else FrameType.PARTIAL if frame_type == "partial" else FrameType.STATUS,
                            stream_id=stream_id,
                            seq=seq,
                            headers={"traceparent": getattr(header, "traceparent", "") if header else ""},
                            corr_id=getattr(header, "corr_id", None) if header else None,
                            payload={}
                        )
                        if getattr(incoming, "payload_json", None):
                            try:
                                app_fr.payload = json.loads(incoming.payload_json) or {}
                            except (ValueError, TypeError) as e:
                                logger.debug(f"JSON-Parsing des Payload fehlgeschlagen: {e}")
                                app_fr.payload = {}
                            except Exception as e:
                                logger.warning(f"Unerwarteter Fehler beim Payload-Parsing: {e}")
                                app_fr.payload = {}
                        ready_frames, missing_seq = await session_manager.buffer_and_collect_incoming(session_id, app_fr)
                        if missing_seq is not None:
                            yield stream_pb2.StreamFrame(
                                type="error",
                                header=stream_pb2.FrameHeader(stream_id=stream_id, seq=missing_seq),
                                error=stream_pb2.ErrorInfo(code="MISSING_SEQUENCE", message=f"Erwarte Sequenz {missing_seq} – Timeout für Resequencing überschritten", retryable=True),
                            )
                        if not ready_frames:
                            continue
                        for rf in ready_frames:
                            await session_manager.enqueue_outgoing(session_id, rf)
                finally:
                    sem.release()
        finally:
            if send_task:
                send_task.cancel()
                with contextlib.suppress(Exception):
                    await send_task
            if heartbeat_task:
                heartbeat_task.cancel()
                with contextlib.suppress(Exception):
                    await heartbeat_task


__all__ = ["KEIStreamService"]
