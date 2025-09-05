"""Typisierte Datenmodelle für Agent-to-Agent (A2A) Kommunikation.

Stellt strukturierte, versionsierte Message-Formate für A2A-Nachrichten bereit,
inkl. Tool-Calls, Attachments, Korrelation und W3C Trace-Kontext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .constants import (
    A2A_ERROR_CONTENT_REQUIRED,
    A2A_ERROR_CORR_ID_EMPTY,
    A2A_ERROR_INVALID_URI,
    A2A_ERROR_TOOL_NAME_REQUIRED,
    A2A_ERROR_TRACEPARENT_EMPTY,
    A2A_ROLE_ASSISTANT,
    A2A_ROLE_SYSTEM,
    A2A_ROLE_TOOL,
    A2A_ROLE_USER,
)
from .utils import ValidationMixin, ensure_dict_default, ensure_list_default, utc_now

if TYPE_CHECKING:
    from datetime import datetime


class A2ARole(str, Enum):
    """Rollen für A2A-Nachrichteninhalte."""

    USER = A2A_ROLE_USER
    ASSISTANT = A2A_ROLE_ASSISTANT
    SYSTEM = A2A_ROLE_SYSTEM
    TOOL = A2A_ROLE_TOOL


@dataclass(slots=True)
class A2AAttachment:
    """Metadaten zu angehängten Ressourcen (z. B. Dateien, URLs)."""

    id: str
    type: str
    uri: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=ensure_dict_default)


@dataclass(slots=True)
class A2AToolCall:
    """Struktur für Tool-Aufrufe innerhalb einer Nachricht."""

    name: str
    arguments: dict[str, Any] = field(default_factory=ensure_dict_default)
    call_id: str | None = None


@dataclass(slots=True)
class A2AMessage(ValidationMixin):
    """A2A-Nachricht mit Korrelation und Trace-Kontext.

    Attributes:
        protocol_version: Version des A2A-Protokolls
        role: Semantische Rolle der Nachricht
        content: Freier Textinhalt
        tool_calls: Optionale Tool-Aufrufe
        attachments: Optionale Anhänge
        corr_id: Korrelations-ID für Ende-zu-Ende Nachverfolgung
        causation_id: Ursprungs-ID für Kausalkette
        traceparent: W3C Traceparent-Header
        timestamp: Zeitstempel in UTC
        headers: Zusätzliche strukturierte Header
        metadata: Frei erweiterbare Metadaten
    """

    protocol_version: str
    role: A2ARole
    content: str
    tool_calls: list[A2AToolCall] = field(default_factory=ensure_list_default)
    attachments: list[A2AAttachment] = field(default_factory=ensure_list_default)
    corr_id: str | None = None
    causation_id: str | None = None
    traceparent: str | None = None
    timestamp: datetime = field(default_factory=utc_now)
    headers: dict[str, Any] = field(default_factory=ensure_dict_default)
    metadata: dict[str, Any] = field(default_factory=ensure_dict_default)

    def _validate_content(self) -> None:
        """Validiert den Nachrichteninhalt."""
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError(A2A_ERROR_CONTENT_REQUIRED)

    def _validate_tool_calls(self) -> None:
        """Validiert alle Tool-Aufrufe."""
        for tc in self.tool_calls:
            if not tc.name or not isinstance(tc.name, str):
                raise ValueError(A2A_ERROR_TOOL_NAME_REQUIRED)

    def _validate_attachments(self) -> None:
        """Validiert alle Anhänge."""
        for att in self.attachments:
            try:
                self.validate_uri_field(att.uri, "Attachment-URI")
            except ValueError as exc:
                raise ValueError(f"{A2A_ERROR_INVALID_URI}: {att.uri}") from exc

    def _validate_optional_fields(self) -> None:
        """Validiert optionale Felder."""
        if self.corr_id is not None and not str(self.corr_id).strip():
            raise ValueError(A2A_ERROR_CORR_ID_EMPTY)
        if self.traceparent is not None and not str(self.traceparent).strip():
            raise ValueError(A2A_ERROR_TRACEPARENT_EMPTY)

    def validate(self) -> None:
        """Validiert die A2A-Nachricht gemäß Schema.

        - content darf nicht leer sein
        - attachments: gültige URI
        - tool_calls: Name Pflicht
        - role: muss A2ARole sein (durch Enum gesichert)
        - traceparent/corr_id: falls gesetzt, nicht leer
        """
        self._validate_content()
        self._validate_tool_calls()
        self._validate_attachments()
        self._validate_optional_fields()


@dataclass(slots=True)
class A2AEnvelope:
    """Transport-Envelope für A2A über KEI-Bus."""

    from_agent_id: str
    to_agent_id: str
    message: A2AMessage
    tenant: str | None = None


__all__ = [
    "A2AAttachment",
    "A2AEnvelope",
    "A2AMessage",
    "A2ARole",
    "A2AToolCall",
]
