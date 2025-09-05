"""Message Envelope für KEI-Bus.

Definiert ein standardisiertes Envelope-Format mit Tracing, Korrelation und Schema-Referenzen.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BusEnvelope(BaseModel):
    """Standard-Envelope für Bus-Nachrichten.

    Attributes:
        id: Eindeutige Message-ID
        type: Event-/Command-Typ
        subject: NATS Subject / Kafka Topic
        ts: Zeitstempel ISO-8601 (UTC)
        corr_id: Korrelations-ID
        causation_id: Auslösende Message-ID
        tenant: Tenant-/Projekt-Namespace
        key: Partitionierungs-/Ordering-Key
        headers: Beliebige Header (inkl. Security-Claims)
        traceparent: W3C Trace Context
        schema_ref: Referenz auf Schema (z. B. JSON Schema URI)
        payload: Nutzdaten
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    type: str
    subject: str
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    corr_id: str | None = None
    causation_id: str | None = None
    tenant: str | None = None
    key: str | None = None
    headers: dict[str, Any] = Field(default_factory=dict)
    traceparent: str | None = None
    schema_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def with_trace(self, traceparent: str | None) -> BusEnvelope:
        """Gibt Envelope mit gesetztem Traceparent zurück."""
        if traceparent:
            self.traceparent = traceparent
        return self
