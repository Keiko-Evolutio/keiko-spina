"""KEI-Stream Frame- und Fehler-/Ack-Schemata.

Dieses Modul definiert die typsicheren Frame-Strukturen für KEI-Stream
gemäß Spezifikation in `prompts_for_dev/api/KEI-Stream.md`.

Die Frames werden einheitlich über WebSocket, gRPC und optional SSE
transportiert. Alle Docstrings sind auf Deutsch, während Bezeichner
englisch bleiben.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class FrameType(str, Enum):
    """Alle unterstützten Frame-Typen für KEI-Stream."""

    PARTIAL = "partial"
    FINAL = "final"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"
    RESUME = "resume"
    CHUNK_START = "chunk_start"
    CHUNK_CONTINUE = "chunk_continue"
    CHUNK_END = "chunk_end"


class ChunkInfo(BaseModel):
    """Metadaten für Chunking großer Payloads."""

    kind: Literal["start", "continue", "end"] = Field(
        ...,
        description="Art des Chunks (start|continue|end)",
    )
    content_range: str | None = Field(
        None,
        description="Byte-Range im Format 'bytes <start>-<end>/<total>'",
    )
    checksum: str | None = Field(
        None,
        description="Checksumme (z. B. SHA256) für Integritätsprüfung",
    )


class ErrorInfo(BaseModel):
    """Standardisiertes Fehlerobjekt für Fehler-Frames."""

    code: str = Field(..., description="Fehlercode (z. B. RATE_LIMIT_EXCEEDED)")
    message: str = Field(..., description="Benutzerfreundliche Fehlermeldung")
    retryable: bool = Field(True, description="Ob der Fehler retrybar ist")
    details: dict[str, Any] | None = Field(
        None, description="Zusätzliche Fehlerdetails"
    )


class AckInfo(BaseModel):
    """Ack/Nack-Informationen für at-least-once Zustellung."""

    ack_seq: int | None = Field(
        None, description="Letzte bestätigte Sequenznummer des Empfängers"
    )
    credit: int | None = Field(
        None,
        ge=0,
        description=(
            "Zurückgemeldetes Sende-Kreditsfenster; erhöht Server-Sendekapazität"
        ),
    )
    reason: str | None = Field(
        None, description="Optionaler Grund für NACK oder Flow-Control-Hinweis"
    )


class KEIStreamFrame(BaseModel):
    """Einheitlicher KEI-Stream Frame.

    Der Frame ist bewusst generisch gehalten. Die `payload`-Struktur ist
    JSON-basiert und je nach `type` unterschiedlich. Binärdaten werden
    referenziert (z. B. über `binary_ref`) oder in Chunks übertragen.
    """

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(
        default_factory=lambda: f"msg_{uuid4().hex[:12]}",
        description="Eindeutige Nachrichten-ID für Idempotenz",
    )
    type: FrameType = Field(..., description="Frame-Typ")
    stream_id: str = Field(..., description="Stream-ID (logischer Kanal)")
    seq: int = Field(0, ge=0, description="Sequenznummer pro Stream-ID")
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Zeitstempel")
    corr_id: str | None = Field(
        None, description="Korrelations-ID über Interaktionen/Services hinweg"
    )
    headers: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Zusätzliche Header, inkl. 'traceparent' für W3C Trace-Propagation"
        ),
    )
    payload: dict[str, Any] | None = Field(
        default=None, description="Inhalts-Payload; Aufbau abhängig vom Typ"
    )
    binary_ref: str | None = Field(
        default=None, description="Referenz auf externe Binärdaten (z. B. Blob-URI)"
    )
    chunk: ChunkInfo | None = Field(
        default=None, description="Chunk-Metadaten für große Payloads"
    )
    error: ErrorInfo | None = Field(
        default=None, description="Fehlerinformationen für ERROR-Frames"
    )
    ack: AckInfo | None = Field(
        default=None, description="ACK/NACK-Informationen für Flow-Control"
    )


def make_ack(
    *, stream_id: str, ack_seq: int, credit: int | None = None, corr_id: str | None = None
) -> KEIStreamFrame:
    """Erstellt einen ACK-Frame.

    Args:
        stream_id: Logische Stream-ID
        ack_seq: Bestätigte Sequenznummer
        credit: Optionales Kreditfenster zum Erhöhen der Sendequote
        corr_id: Optionale Korrelations-ID

    Returns:
        ACK-Frame
    """
    return KEIStreamFrame(
        type=FrameType.ACK,
        stream_id=stream_id,
        seq=ack_seq,
        corr_id=corr_id,
        headers={},
        payload=None,
        ack=AckInfo(ack_seq=ack_seq, credit=credit),
    )


def make_error(
    *,
    stream_id: str,
    seq: int,
    code: str,
    message: str,
    retryable: bool = True,
    details: dict[str, Any] | None = None,
    corr_id: str | None = None,
) -> KEIStreamFrame:
    """Erstellt einen standardisierten ERROR-Frame."""
    return KEIStreamFrame(
        type=FrameType.ERROR,
        stream_id=stream_id,
        seq=seq,
        corr_id=corr_id,
        headers={},
        error=ErrorInfo(code=code, message=message, retryable=retryable, details=details),
    )


__all__ = [
    "AckInfo",
    "ChunkInfo",
    "ErrorInfo",
    "FrameType",
    "KEIStreamFrame",
    "make_ack",
    "make_error",
]
