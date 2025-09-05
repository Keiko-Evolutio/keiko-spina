"""KEI-Stream Session- und Resume-Management.

Implementiert Session-IDs, Reconnect/Resume mit Replay-Fenster, Sequenz-
Tracking, Credit-basiertes Flow-Control und Slow-Consumer-Policy.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

from kei_logging import get_logger
from monitoring import record_custom_metric

from .frames import FrameType, KEIStreamFrame
from .quotas import resolve_quota

logger = get_logger(__name__)


DEFAULT_TOKENS_PER_SEC: float = float(os.getenv("KEI_STREAM_PER_STREAM_TOKENS_PER_SEC", "50"))
DEFAULT_BUCKET_CAPACITY: int = int(os.getenv("KEI_STREAM_PER_STREAM_BUCKET_CAP", "100"))
DEFAULT_FRAME_COST: float = float(os.getenv("KEI_STREAM_PER_STREAM_FRAME_COST", "1"))


@dataclass
class StreamState:
    """Zustand eines logischen Streams innerhalb einer Session."""

    stream_id: str
    next_seq_out: int = 1
    next_seq_in_expected: int = 1
    credit_window: int = 16  # Standard-Kreditfenster
    # Bereits gesendete, aber noch nicht bestätigte Frames
    buffered_outgoing: deque[KEIStreamFrame] = field(default_factory=deque)
    # Ausstehende Frames, die noch nicht gesendet wurden (warten auf Credits)
    pending_outgoing: deque[KEIStreamFrame] = field(default_factory=deque)
    replay_buffer: deque[KEIStreamFrame] = field(default_factory=deque)
    replay_retention: int = 256
    # Resequencing für eingehende Frames
    reseq_buffer: dict[int, KEIStreamFrame] = field(default_factory=dict)
    gap_started_at: datetime | None = None
    # Token-Bucket für Per-Stream-Rate-Control
    tokens_per_sec: float = DEFAULT_TOKENS_PER_SEC
    bucket_capacity: int = DEFAULT_BUCKET_CAPACITY
    tokens: float = float(DEFAULT_BUCKET_CAPACITY)
    last_refill_monotonic: float = field(default_factory=time.monotonic)
    # Sendezeitpunkte für ACK-Latenz (seq -> monotonic timestamp)
    send_time_by_seq: dict[int, float] = field(default_factory=dict)
    # Gleitendes Fenster für ACK-Latenzen (in ms) zur Prozentil-Aggregation
    ack_latency_window: deque[float] = field(default_factory=lambda: deque(maxlen=int(os.getenv("KEI_STREAM_ACK_LATENCY_WINDOW", "200"))))

    def record_sent(self, frame: KEIStreamFrame) -> None:
        """Merkt gesendete Frames für Replay innerhalb des Fensters.

        - Hält Replay-Fenster begrenzt
        - Speichert Sendezeitpunkt für spätere ACK-Latenzberechnung
        """
        self.replay_buffer.append(frame)
        while len(self.replay_buffer) > self.replay_retention:
            self.replay_buffer.popleft()
        with contextlib.suppress(Exception):
            self.send_time_by_seq[frame.seq] = time.monotonic()

    def ack(self, ack_seq: int, new_credit: int | None) -> None:
        """Verarbeitet eingehenden ACK/NACK und passt Flow-Control an.

        - Aktualisiert Credit-Fenster
        - Entfernt bestätigte Frames aus dem Buffer
        - Erfasst ACK-Latenzmetriken
        """
        if new_credit is not None:
            self.credit_window = max(0, new_credit)
        # Bereits bestätigte Frames aus dem Buffer entfernen und Latenz messen
        while self.buffered_outgoing and self.buffered_outgoing[0].seq <= ack_seq:
            fr = self.buffered_outgoing.popleft()
            try:
                t0 = self.send_time_by_seq.pop(fr.seq, None)
                if t0 is not None:
                    latency_ms = max(0.0, (time.monotonic() - t0) * 1000.0)
                    record_custom_metric(
                        "kei_stream.ack_latency_ms",
                        latency_ms,
                        {"stream_id": fr.stream_id}
                    )
                    record_custom_metric(
                        "kei_stream.ack_in_total",
                        1,
                        {"stream_id": fr.stream_id}
                    )
                    # Latenz in Fenster aufnehmen und Prozentile berechnen
                    try:
                        self.ack_latency_window.append(latency_ms)
                        # Prozentile p50/p95/p99 berechnen
                        values = sorted(self.ack_latency_window)
                        if values:
                            def _percentile(v: list[float], q: float) -> float:
                                # Einfaches Perzentil (Nearest-Rank)
                                if not v:
                                    return 0.0
                                idx = max(0, min(len(v) - 1, round(q * (len(v) - 1))))
                                return float(v[idx])

                            p50 = _percentile(values, 0.50)
                            p95 = _percentile(values, 0.95)
                            p99 = _percentile(values, 0.99)
                            record_custom_metric("kei_stream.ack_latency_ms_p50", p50, {"stream_id": fr.stream_id})
                            record_custom_metric("kei_stream.ack_latency_ms_p95", p95, {"stream_id": fr.stream_id})
                            record_custom_metric("kei_stream.ack_latency_ms_p99", p99, {"stream_id": fr.stream_id})
                    except Exception:
                        pass
            except Exception:
                pass


@dataclass
class ChunkAssemblyState:
    """Interner Zustand für Chunk-Zusammenbau eines Binary-Streams."""

    stream_id: str
    chunk_id: str
    total_size: int
    expected_next_offset: int
    checksum_expected: str | None
    buffer: BytesIO = field(default_factory=BytesIO)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Hinweis: ChunkAssemblyState enthält keine Replay-/ACK-Logik


@dataclass
class SessionContext:
    """Session-Kontext für eine aktive oder wiederaufgenommene KEI-Stream Session."""

    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    streams: dict[str, StreamState] = field(default_factory=dict)
    tenant_id: str | None = None
    scopes: list[str] = field(default_factory=list)
    api_key: str | None = None
    closed: bool = False
    idle_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    # Bedingungsvariable für Flow-Control (Signal bei Enqueue/Credit-Änderung)
    send_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    # Idempotenz-/Dedup-Cache
    idempotency_ttl: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    idempotency_max: int = 10_000
    seen_msg_ids: dict[str, datetime] = field(default_factory=dict)
    seen_msg_order: deque[tuple[str, datetime]] = field(default_factory=deque)
    # Chunk-Zusammenbau und Speicher
    chunk_assemblies: dict[tuple[str, str], ChunkAssemblyState] = field(default_factory=dict)
    chunk_sink: str = field(default_factory=lambda: os.getenv("KEI_STREAM_CHUNK_SINK", "memory"))
    chunk_dir: str = field(default_factory=lambda: os.getenv("KEI_STREAM_CHUNK_DIR", "/tmp/kei_stream_chunks"))
    assembled_memory_store: dict[str, bytes] = field(default_factory=dict)

    def touch(self) -> None:
        """Aktualisiert Last-Access Zeitpunkt."""
        self.last_seen_at = datetime.now(UTC)

    def get_or_create_stream(self, stream_id: str) -> StreamState:
        """Liefert StreamState, erstellt wenn nicht vorhanden."""
        if stream_id not in self.streams:
            # Quota prüfen: max_streams
            max_streams = resolve_quota(self.tenant_id, self.api_key).max_streams
            if len(self.streams) >= max_streams:
                from core.exceptions import KeikoRateLimitError
                raise KeikoRateLimitError("Maximale Anzahl gleichzeitiger Streams überschritten", details={"limit": max_streams})
            self.streams[stream_id] = StreamState(stream_id=stream_id)
        return self.streams[stream_id]

    def remove_stream(self, stream_id: str) -> bool:
        """Entfernt einen Stream aus der Session.

        Returns:
            True wenn entfernt, False sonst
        """
        return self.streams.pop(stream_id, None) is not None


class SessionManager:
    """Verwaltet KEI-Stream Sessions inkl. Resume und Replay-Fenster."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._lock = asyncio.Lock()
        # Maximale Anzahl ausstehender Frames pro Stream, um Memory-Pressure zu begrenzen
        try:
            self._max_pending_per_stream: int = int(os.getenv("KEI_STREAM_MAX_PENDING_PER_STREAM", "1024"))
        except Exception:
            self._max_pending_per_stream = 1024
        try:
            self._idemp_ttl_secs: int = int(os.getenv("KEI_STREAM_IDEMPOTENCY_TTL_SECONDS", "600"))
        except Exception:
            self._idemp_ttl_secs = 600
        try:
            self._idemp_max_ids: int = int(os.getenv("KEI_STREAM_IDEMPOTENCY_MAX_IDS", "10000"))
        except Exception:
            self._idemp_max_ids = 10000
        try:
            self._resequence_timeout_ms: int = int(os.getenv("KEI_STREAM_RESEQUENCE_TIMEOUT_MS", "500"))
        except Exception:
            self._resequence_timeout_ms = 500
        try:
            self._resequence_max_buffer: int = int(os.getenv("KEI_STREAM_RESEQUENCE_MAX_BUFFER", "256"))
        except Exception:
            self._resequence_max_buffer = 256
        try:
            self._chunk_checksum_algo: str = os.getenv("KEI_STREAM_CHUNK_CHECKSUM_ALGO", "sha256").lower()
        except Exception:
            self._chunk_checksum_algo = "sha256"

    async def create_or_resume(self, session_id: str) -> SessionContext:
        """Erstellt neue Session oder setzt bestehende fort."""
        async with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is None:
                ctx = SessionContext(
                    session_id=session_id,
                    idempotency_ttl=timedelta(seconds=self._idemp_ttl_secs),
                    idempotency_max=self._idemp_max_ids,
                )
                self._sessions[session_id] = ctx
                logger.info("Neue KEI-Stream Session erstellt: %s", session_id)
            else:
                logger.info("Session fortgesetzt: %s", session_id)
            ctx.touch()
            return ctx

    async def bind_auth_context(self, session_id: str, *, tenant_id: str | None, scopes: list[str] | None) -> None:
        """Bindet Tenant/Scopes an Session, falls neu oder leer."""
        async with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                ctx = SessionContext(session_id=session_id, tenant_id=tenant_id or None, scopes=list(scopes or []))
                self._sessions[session_id] = ctx
            else:
                if tenant_id and not ctx.tenant_id:
                    ctx.tenant_id = tenant_id
                if scopes and not ctx.scopes:
                    ctx.scopes = list(scopes)
            ctx.touch()

    async def bind_api_key(self, session_id: str, api_key: str | None) -> None:
        """Bindet einen API-Key an die Session für Quota-Auflösung."""
        async with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                ctx = SessionContext(session_id=session_id, api_key=api_key or None)
                self._sessions[session_id] = ctx
            elif api_key and not ctx.api_key:
                ctx.api_key = api_key
            ctx.touch()

    async def get(self, session_id: str) -> SessionContext | None:
        """Gibt SessionContext zurück, falls vorhanden."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_stream_state(self, session_id: str, stream_id: str) -> StreamState | None:
        """Gibt StreamState einer Session zurück, None wenn unbekannt."""
        ctx = await self.get(session_id)
        if not ctx:
            return None
        return ctx.get_or_create_stream(stream_id)

    async def close(self, session_id: str) -> None:
        """Schließt Session und entfernt sie aus dem Speicher."""
        async with self._lock:
            ctx = self._sessions.pop(session_id, None)
            if ctx is not None:
                ctx.closed = True
                logger.info("Session geschlossen: %s", session_id)

    async def cleanup_idle(self) -> int:
        """Entfernt inaktive Sessions gemäß Idle-Timeout.

        Returns:
            Anzahl der entfernten Sessions
        """
        async with self._lock:
            now = datetime.now(UTC)
            to_delete = [
                sid
                for sid, ctx in self._sessions.items()
                if (now - ctx.last_seen_at) > ctx.idle_timeout or ctx.closed
            ]
            for sid in to_delete:
                del self._sessions[sid]
            if to_delete:
                logger.info("%d inaktive Sessions bereinigt", len(to_delete))
            return len(to_delete)

    async def cancel_stream(self, session_id: str, stream_id: str) -> bool:
        """Bricht einen Stream innerhalb einer Session ab."""
        ctx = await self.get(session_id)
        if not ctx:
            return False
        removed = ctx.remove_stream(stream_id)
        ctx.touch()
        return removed

    async def stats(self) -> dict[str, Any]:
        """Gibt einfache Statistiken über Sessions/Streams zurück."""
        out: dict[str, Any] = {"sessions": 0, "streams": 0}
        for sid in list(self._sessions.keys()):
            ctx = self._sessions.get(sid)
            if not ctx:
                continue
            out["sessions"] += 1
            out["streams"] += len(ctx.streams)
        return out

    async def handle_ack(
        self, session_id: str, stream_id: str, ack_seq: int, credit: int | None
    ) -> None:
        """Verarbeitet ACK/NACK Frames und aktualisiert Flow-Control."""
        ctx = await self.get(session_id)
        if not ctx:
            return
        async with ctx.send_condition:
            stream = ctx.get_or_create_stream(stream_id)
            stream.ack(ack_seq, credit)
            ctx.touch()
            # Sender über frei gewordene Credits informieren
            ctx.send_condition.notify_all()

    async def record_sent(self, session_id: str, frame: KEIStreamFrame) -> None:
        """Puffert gesendeten Frame für mögliches Replay."""
        ctx = await self.get(session_id)
        if not ctx:
            return
        stream = ctx.get_or_create_stream(frame.stream_id)
        stream.record_sent(frame)
        ctx.touch()

    async def verify_and_advance_incoming(self, session_id: str, stream_id: str, seq: int) -> bool:
        """Prüft Reihenfolge eingehender Frames und erhöht Erwartungswert.

        Returns:
            True bei korrekter Reihenfolge, False sonst
        """
        ctx = await self.get(session_id)
        if not ctx:
            return True
        stream = ctx.get_or_create_stream(stream_id)
        if seq < stream.next_seq_in_expected:
            return False
        stream.next_seq_in_expected = seq + 1
        ctx.touch()
        return True

    async def buffer_and_collect_incoming(self, session_id: str, frame: KEIStreamFrame) -> tuple[list[KEIStreamFrame], int | None]:
        """Resequencing für eingehende Frames mit Timeout.

        Garantiert in-order Auslieferung pro Stream durch Zwischenspeicherung
        von out-of-order Frames bis entweder die Lücke geschlossen ist oder
        ein Timeout überschritten wurde.

        Args:
            session_id: Session-Identifikator
            frame: Eingehender Frame

        Returns:
            (ready_frames, missing_seq): Liste der jetzt verarbeitbaren Frames
            in korrekter Reihenfolge und optional die fehlende Sequenznummer,
            falls ein Timeout eingetreten ist.
        """
        ctx = await self.get(session_id)
        if not ctx:
            return [frame], None
        st = ctx.get_or_create_stream(frame.stream_id)
        now = datetime.now(UTC)
        expected = st.next_seq_in_expected

        # Duplikat/zu alt
        if frame.seq < expected:
            return [], None

        # Exakt erwartet → ausliefern und evtl. Buffer flushen
        if frame.seq == expected:
            ready: list[KEIStreamFrame] = [frame]
            st.next_seq_in_expected = expected + 1
            # Nachfolgende aus Buffer einsammeln
            while st.next_seq_in_expected in st.reseq_buffer:
                nxt = st.reseq_buffer.pop(st.next_seq_in_expected)
                ready.append(nxt)
                st.next_seq_in_expected += 1
            # Gap zurücksetzen, wenn Buffer leer
            if not st.reseq_buffer:
                st.gap_started_at = None
            ctx.touch()
            return ready, None

        # frame.seq > expected → in Buffer ablegen
        st.reseq_buffer[frame.seq] = frame
        if st.gap_started_at is None:
            st.gap_started_at = now
        # Buffer begrenzen: Drop-Highest bis Limit erreicht
        if len(st.reseq_buffer) > self._resequence_max_buffer:
            try:
                highest_seq = max(st.reseq_buffer.keys())
                st.reseq_buffer.pop(highest_seq, None)
                logger.warning(
                    "Resequence-Buffer voll, verwerfe höchsten Seq (session=%s, stream=%s, seq=%s)",
                    session_id,
                    frame.stream_id,
                    highest_seq,
                )
            except (ImportError, AttributeError):
                # Monitoring-Modul nicht verfügbar
                pass
            except Exception as e:
                logger.debug(f"Fehler beim Aufzeichnen der Metrik: {e}")
        # Timeout prüfen
        gap_ms = (now - st.gap_started_at).total_seconds() * 1000.0 if st.gap_started_at else 0.0
        if st.gap_started_at and gap_ms >= self._resequence_timeout_ms:
            # Signalisiere fehlende Sequenz
            missing_seq = expected
            # Gap neu starten, um wiederholt zu signalisieren, aber nicht zu spammer
            st.gap_started_at = now
            ctx.touch()
            return [], missing_seq

        ctx.touch()
        return [], None

    async def resume_from(self, session_id: str, stream_id: str, last_seq: int) -> list[KEIStreamFrame]:
        """Liefert Frames ab gegebener Sequenznummer aus dem Replay-Fenster."""
        ctx = await self.get(session_id)
        if not ctx:
            return []
        stream = ctx.get_or_create_stream(stream_id)
        replay: list[KEIStreamFrame] = []
        for fr in stream.replay_buffer:
            if fr.seq > last_seq and fr.stream_id == stream_id:
                replay.append(fr)
        logger.info("Resume liefert %d Frames (session=%s, stream=%s, after=%s)", len(replay), session_id, stream_id, last_seq)
        return replay

    async def is_duplicate_or_register(self, session_id: str, msg_id: str) -> bool:
        """Prüft, ob `msg_id` bereits verarbeitet wurde, und registriert sie ansonsten.

        Gewährleistet Idempotenz mit TTL und Kapazitätsgrenze pro Session.

        Args:
            session_id: Session-Identifikator
            msg_id: Nachrichten-ID

        Returns:
            True, wenn Duplikat (bereits gesehen), sonst False
        """
        async with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                return False
            now = datetime.now(UTC)
            # Cleanup: TTL-basiert
            ttl_deadline = now - ctx.idempotency_ttl
            while ctx.seen_msg_order and (ctx.seen_msg_order[0][1] < ttl_deadline or ctx.seen_msg_order[0][0] not in ctx.seen_msg_ids):
                old_id, _ts = ctx.seen_msg_order.popleft()
                # nur entfernen, wenn wirklich alt
                ts_in_map = ctx.seen_msg_ids.get(old_id)
                if ts_in_map and ts_in_map < ttl_deadline:
                    ctx.seen_msg_ids.pop(old_id, None)

            # Bereits gesehen?
            if msg_id in ctx.seen_msg_ids:
                return True

            # Registrieren
            ctx.seen_msg_ids[msg_id] = now
            ctx.seen_msg_order.append((msg_id, now))

            # Kapazitätsgrenze durchsetzen (Drop-Oldest)
            while len(ctx.seen_msg_ids) > ctx.idempotency_max and ctx.seen_msg_order:
                drop_id, _ts = ctx.seen_msg_order.popleft()
                ctx.seen_msg_ids.pop(drop_id, None)

            return False

    async def shrink_pending(self, session_id: str, stream_id: str, cutoff: int) -> int:
        """Kürzt die Pending-Queue eines Streams auf die gegebene Obergrenze.

        Args:
            session_id: Session-Identifikator
            stream_id: Stream-Identifikator
            cutoff: Verbleibende maximale Länge nach Kürzung

        Returns:
            Anzahl tatsächlich entfernter Frames
        """
        ctx = await self.get(session_id)
        if not ctx:
            return 0
        async with ctx.send_condition:
            st = ctx.get_or_create_stream(stream_id)
            removed = 0
            while len(st.pending_outgoing) > max(0, cutoff):
                try:
                    st.pending_outgoing.popleft()
                    removed += 1
                except Exception:
                    break
            if removed:
                ctx.touch()
                ctx.send_condition.notify_all()
            return removed

    async def enqueue_outgoing(self, session_id: str, frame: KEIStreamFrame) -> bool:
        """Fügt einen ausgehenden Frame in die Stream-Queue ein.

        Respektiert eine Obergrenze der ausstehenden Frames, um Speicher zu schützen.

        Args:
            session_id: Ziel-Session
            frame: Frame, der gesendet werden soll

        Returns:
            True, wenn der Frame eingequeued wurde, False bei Ablehnung
        """
        ctx = await self.get(session_id)
        if not ctx:
            return False
        async with ctx.send_condition:
            stream = ctx.get_or_create_stream(frame.stream_id)
            # Begrenzung anwenden (Drop-Oldest, um Tail-Latency zu bevorzugen)
            if len(stream.pending_outgoing) >= self._max_pending_per_stream:
                try:
                    dropped = stream.pending_outgoing.popleft()
                    logger.warning(
                        "Pending-Queue voll, verwerfe ältesten Frame (session=%s, stream=%s, dropped_seq=%s)",
                        session_id,
                        frame.stream_id,
                        getattr(dropped, "seq", -1),
                    )
                    with contextlib.suppress(ImportError, AttributeError):
                        record_custom_metric("kei_stream.drops_total", 1, {"stream_id": frame.stream_id})
                except (ImportError, AttributeError):
                    # Monitoring-Modul nicht verfügbar
                    pass
                except Exception as e:
                    logger.debug(f"Fehler beim Aufzeichnen der Drop-Metrik: {e}")
            stream.pending_outgoing.append(frame)
            ctx.touch()
            # Sende-Condition wecken
            ctx.send_condition.notify_all()
            return True

    def _refill_tokens(self, st: StreamState) -> None:
        """Füllt den Token-Bucket des Streams gemäß Rate nach."""
        now = time.monotonic()
        elapsed = now - st.last_refill_monotonic
        if elapsed <= 0:
            return
        st.tokens = min(float(st.bucket_capacity), st.tokens + elapsed * st.tokens_per_sec)
        st.last_refill_monotonic = now

    async def _find_next_sendable(self, ctx: SessionContext) -> KEIStreamFrame | None:
        """Sucht den nächsten sendbaren Frame über alle Streams der Session.

        Sendbar bedeutet: Es liegt mindestens ein Frame an und das Credit-Fenster ist > 0.
        """
        # Faire Auswahl: sortiere Streams nach dem Zeitpunkt der letzten Auffüllung, um Hunger zu reduzieren
        for stream_id, st in list(ctx.streams.items()):
            # Refill Tokens
            self._refill_tokens(st)
            if st.credit_window <= 0 or not st.pending_outgoing:
                if st.credit_window <= 0:
                    with contextlib.suppress(Exception):
                        record_custom_metric("kei_stream.backpressure_blocked", 1, {"stream_id": stream_id})
                continue
            # Prüfe Token-Bucket
            required = DEFAULT_FRAME_COST
            if st.tokens < required:
                with contextlib.suppress(Exception):
                    record_custom_metric("kei_stream.token_waits", 1, {"stream_id": stream_id})
                continue
            # Senden möglich
            fr = st.pending_outgoing.popleft()
            # Kredit dekrementieren und für Replay/ACK-Tracking registrieren
            st.credit_window = max(0, st.credit_window - 1)
            st.tokens -= required
            st.buffered_outgoing.append(fr)
            st.record_sent(fr)
            return fr
        return None

    async def wait_for_next_sendable(self, session_id: str, timeout: float | None = None) -> KEIStreamFrame | None:
        """Wartet bis ein Frame gemäß Credits gesendet werden darf und liefert ihn zurück.

        Args:
            session_id: Session, aus der gesendet werden soll
            timeout: Optionales Timeout in Sekunden

        Returns:
            Nächster sendbarer Frame oder None bei Timeout/fehlender Session
        """
        ctx = await self.get(session_id)
        if not ctx:
            return None
        async with ctx.send_condition:
            # Schneller Versuch ohne Warten
            fr = await self._find_next_sendable(ctx)
            if fr is not None:
                return fr
            # Warten bis Credits/Queue verfügbar
            try:
                if timeout is not None:
                    try:
                        await asyncio.wait_for(ctx.send_condition.wait(), timeout=timeout)
                    except TimeoutError:
                        return None
                else:
                    await ctx.send_condition.wait()
            except (TimeoutError, asyncio.CancelledError):
                return None
            except Exception as e:
                logger.debug(f"Fehler beim Warten auf sendbare Frames: {e}")
                return None
            # Nach dem Wecken erneut versuchen
            return await self._find_next_sendable(ctx)

    # ------------------------- Chunking/Assembler -------------------------

    @staticmethod
    def _parse_content_range(range_header: str) -> tuple[int, int, int]:
        """Parst einen HTTP-ähnlichen Content-Range Header.

        Erwartetes Format: "bytes <start>-<end>/<total>" (inkl. inklusivem Endoffset).
        """
        m = re.match(r"^bytes\s+(\d+)-(\d+)/(\d+)$", range_header or "")
        if not m:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Ungültiges content_range Format")
        start = int(m.group(1))
        end = int(m.group(2))
        total = int(m.group(3))
        if start < 0 or end < start or total <= 0 or end >= total:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Ungültige content_range Werte")
        return start, end, total

    def _get_checksum(self, data: bytes) -> str:
        """Berechnet die Prüfsumme gemäß konfiguriertem Algorithmus."""
        algo = self._chunk_checksum_algo
        if algo == "sha256":
            return hashlib.sha256(data).hexdigest()
        if algo == "sha1":
            return hashlib.sha1(data).hexdigest()  # noqa: S324
        if algo == "md5":
            return hashlib.md5(data).hexdigest()  # noqa: S324
        # Fallback
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _decode_chunk_payload(frame: KEIStreamFrame) -> bytes:
        """Dekodiert Chunk-Daten aus dem Frame-Payload.

        Erwartet base64-kodierte Daten in payload["data_b64"].
        """
        if not frame.payload or "data_b64" not in frame.payload:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Chunk-Payload fehlt 'data_b64'")
        try:
            return base64.b64decode(frame.payload["data_b64"], validate=True)
        except (ValueError, TypeError) as exc:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Ungültige Base64-Daten im Chunk-Payload") from exc
        except Exception as exc:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Unerwarteter Fehler beim Base64-Dekodieren") from exc

    async def process_chunk_frame(self, session_id: str, frame: KEIStreamFrame) -> tuple[str | None, int | None, str | None]:
        """Verarbeitet CHUNK_* Frames und assembliert Binärdaten.

        Args:
            session_id: Session-ID
            frame: Eingehender Frame mit type in {chunk_start, chunk_continue, chunk_end}

        Returns:
            (binary_ref, size, error_message)
        """
        if not frame.chunk or frame.type not in (FrameType.CHUNK_START, FrameType.CHUNK_CONTINUE, FrameType.CHUNK_END):
            return None, None, None
        ctx = await self.get(session_id)
        if not ctx:
            return None, None, "Session nicht gefunden"

        # Schlüssel: (stream_id, chunk_id). Für Chunk-ID verwenden wir Frame-ID.
        chunk_id = frame.id
        key = (frame.stream_id, chunk_id)

        # Content-Range prüfen und Daten dekodieren
        if not frame.chunk.content_range:
            return None, None, "content_range im Chunk fehlt"
        start, end, total = self._parse_content_range(frame.chunk.content_range)
        data = self._decode_chunk_payload(frame)
        expected_len = end - start + 1
        if len(data) != expected_len:
            return None, None, "Chunk-Datenlänge stimmt nicht mit content_range überein"

        # State holen/erstellen
        st = ctx.chunk_assemblies.get(key)

        if frame.type == FrameType.CHUNK_START:
            if st is not None:
                # Restart, vorhandenen verwerfen
                ctx.chunk_assemblies.pop(key, None)
            if start != 0:
                return None, None, "Erster Chunk muss bei Offset 0 beginnen"
            st = ChunkAssemblyState(
                stream_id=frame.stream_id,
                chunk_id=chunk_id,
                total_size=total,
                expected_next_offset=end + 1,
                checksum_expected=frame.chunk.checksum,
            )
            st.buffer.write(data)
            ctx.chunk_assemblies[key] = st
            ctx.touch()
            return None, None, None

        if st is None:
            return None, None, "Kein laufender Chunk-Zusammenbau für diesen Stream"

        # Continue/Ende
        if start != st.expected_next_offset:
            return None, None, "Unerwarteter Start-Offset (Lücke/Überschneidung)"
        st.buffer.write(data)
        st.expected_next_offset = end + 1
        if frame.chunk.checksum and not st.checksum_expected:
            st.checksum_expected = frame.chunk.checksum
        ctx.touch()

        if frame.type == FrameType.CHUNK_END:
            # Abschlussprüfungen
            if end != st.total_size - 1:
                return None, None, "Letzter Chunk hat falsches End-Offset"
            assembled = st.buffer.getvalue()
            # Checksummenprüfung (optional, wenn erwartet)
            if st.checksum_expected:
                calc = self._get_checksum(assembled)
                if calc.lower() != st.checksum_expected.lower():
                    # Zusammenbau verwerfen
                    ctx.chunk_assemblies.pop(key, None)
                    return None, None, "Checksumme stimmt nicht überein"
            # Persistieren gemäß Sink
            binary_ref, size = await self._persist_assembled(ctx, st, assembled)
            # State entfernen
            ctx.chunk_assemblies.pop(key, None)
            return binary_ref, size, None

        return None, None, None

    async def _persist_assembled(
        self, ctx: SessionContext, st: ChunkAssemblyState, data: bytes
    ) -> tuple[str, int]:
        """Persistiert den zusammengebauten Chunk gemäß Sink-Konfiguration."""
        size = len(data)
        if ctx.chunk_sink == "file":
            base_dir = Path(ctx.chunk_dir) / ctx.session_id
            with contextlib.suppress(Exception):
                base_dir.mkdir(parents=True, exist_ok=True)
            file_path = base_dir / f"{st.stream_id}_{st.chunk_id}_{size}.bin"
            with open(file_path, "wb") as f:
                f.write(data)
            return f"file://{file_path}", size
        # Default: memory
        mem_key = f"mem://{ctx.session_id}/{st.stream_id}/{st.chunk_id}"
        ctx.assembled_memory_store[mem_key] = data
        return mem_key, size


# Globale Manager-Instanz
session_manager = SessionManager()


__all__ = [
    "SessionContext",
    "SessionManager",
    "StreamState",
    "session_manager",
]
