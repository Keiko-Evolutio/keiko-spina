"""Utility-Funktionen für KEI-Bus.

Konsolidiert gemeinsame Hilfsfunktionen und wiederkehrende Patterns.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kei_logging import get_logger

from .constants import (
    AUTHORIZATION_HEADER,
    BEARER_PREFIX,
    ENVELOPE_FIELD_CAUSATION_ID,
    ENVELOPE_FIELD_CORRELATION_ID,
    ENVELOPE_FIELD_MESSAGE_ID,
    ENVELOPE_FIELD_MESSAGE_TYPE,
    ENVELOPE_FIELD_SUBJECT,
    ENVELOPE_FIELD_TENANT,
)

logger = get_logger(__name__)


def create_subject_hash(subject: str, key: str) -> str:
    """Erstellt Hash für Subject-Key-Kombination.

    Args:
        subject: Subject-Name
        key: Partitionierungs-Key

    Returns:
        Hash-String (12 Zeichen)
    """
    return hashlib.sha1(f"{subject}:{key}".encode()).hexdigest()[:12]


def create_keyed_subject(subject: str, key: str) -> str:
    """Erstellt Subject mit Key-Hash für Partitionierung.

    Args:
        subject: Basis-Subject
        key: Partitionierungs-Key

    Returns:
        Subject mit Key-Hash
    """
    key_hash = create_subject_hash(subject, key)
    return f"{subject}.key.{key_hash}"


def extract_bearer_token(headers: dict[str, Any]) -> str | None:
    """Extrahiert Bearer-Token aus Headers.

    Args:
        headers: HTTP/Message-Headers

    Returns:
        Token-String oder None
    """
    auth_header = headers.get(AUTHORIZATION_HEADER) or headers.get("Authorization")
    if not isinstance(auth_header, str):
        return None

    if not auth_header.lower().startswith(BEARER_PREFIX):
        return None

    return auth_header.split(" ", 1)[1] if " " in auth_header else None


def create_structured_log_context(
    correlation_id: str | None = None,
    causation_id: str | None = None,
    tenant: str | None = None,
    subject: str | None = None,
    message_type: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Erstellt strukturierten Logging-Kontext.

    Args:
        correlation_id: Korrelations-ID
        causation_id: Auslösende Message-ID
        tenant: Tenant-ID
        subject: Subject/Topic
        message_type: Message-Typ
        message_id: Message-ID

    Returns:
        Logging-Kontext-Dictionary
    """
    context = {}

    if correlation_id:
        context[ENVELOPE_FIELD_CORRELATION_ID] = correlation_id
    if causation_id:
        context[ENVELOPE_FIELD_CAUSATION_ID] = causation_id
    if tenant:
        context[ENVELOPE_FIELD_TENANT] = tenant
    if subject:
        context[ENVELOPE_FIELD_SUBJECT] = subject
    if message_type:
        context[ENVELOPE_FIELD_MESSAGE_TYPE] = message_type
    if message_id:
        context[ENVELOPE_FIELD_MESSAGE_ID] = message_id

    return context


def safe_json_serialize(data: Any) -> str:
    """Sichere JSON-Serialisierung mit Fallback.

    Args:
        data: Zu serialisierende Daten

    Returns:
        JSON-String
    """
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        logger.debug("JSON-Serialisierung fehlgeschlagen: %s", exc)
        return json.dumps({"error": "serialization_failed", "type": str(type(data))})


def safe_json_deserialize(json_str: str) -> Any:
    """Sichere JSON-Deserialisierung mit Fallback.

    Args:
        json_str: JSON-String

    Returns:
        Deserialisierte Daten oder None bei Fehler
    """
    try:
        return json.loads(json_str)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("JSON-Deserialisierung fehlgeschlagen: %s", exc)
        return None


def create_cache_key(prefix: str, *parts: str) -> str:
    """Erstellt Cache-Key aus Prefix und Teilen.

    Args:
        prefix: Cache-Key-Prefix
        *parts: Key-Teile

    Returns:
        Vollständiger Cache-Key
    """
    return ":".join([prefix, *list(parts)])


def hash_content(content: str, algorithm: str = "sha256") -> str:
    """Erstellt Hash von Content.

    Args:
        content: Zu hashender Content
        algorithm: Hash-Algorithmus

    Returns:
        Hash-String
    """
    if algorithm == "sha256":
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    if algorithm == "sha1":
        return hashlib.sha1(content.encode("utf-8")).hexdigest()
    if algorithm == "md5":
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def validate_subject_format(subject: str) -> bool:
    """Validiert Subject-Format (Basis-Check).

    Args:
        subject: Subject-String

    Returns:
        True wenn gültig
    """
    if not subject or not isinstance(subject, str):
        return False

    # Basis-Validierung: muss mit "kei." beginnen
    if not subject.startswith("kei."):
        return False

    # Keine Leerzeichen oder Sonderzeichen
    return not (" " in subject or "\t" in subject or "\n" in subject)


def sanitize_subject(subject: str) -> str:
    """Bereinigt Subject-String.

    Args:
        subject: Roher Subject-String

    Returns:
        Bereinigter Subject-String
    """
    if not subject:
        return ""

    # Whitespace entfernen
    subject = subject.strip()

    # Mehrfache Punkte durch einzelne ersetzen
    while ".." in subject:
        subject = subject.replace("..", ".")

    # Führende/nachfolgende Punkte entfernen
    return subject.strip(".")



def create_error_context(
    error: Exception,
    operation: str,
    additional_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt Error-Kontext für Logging.

    Args:
        error: Exception-Objekt
        operation: Operation die fehlgeschlagen ist
        additional_context: Zusätzlicher Kontext

    Returns:
        Error-Kontext-Dictionary
    """
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "operation": operation,
    }

    if additional_context:
        context.update(additional_context)

    return context


def is_retryable_error(error: Exception) -> bool:
    """Prüft, ob ein Fehler retry-fähig ist.

    Args:
        error: Exception-Objekt

    Returns:
        True wenn retry-fähig
    """
    # Permission-Errors sind nicht retry-fähig (höchste Priorität)
    if isinstance(error, PermissionError):
        return False

    # ValueError/TypeError sind meist nicht retry-fähig
    if isinstance(error, ValueError | TypeError):
        return False

    # Netzwerk-/Verbindungsfehler sind meist retry-fähig
    retryable_types = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    if isinstance(error, retryable_types):
        return True

    # Default: nicht retry-fähig
    return False


__all__ = [
    # Cache-Utilities
    "create_cache_key",
    "create_error_context",
    "create_keyed_subject",
    # Logging-Utilities
    "create_structured_log_context",
    # Subject-Utilities
    "create_subject_hash",
    # Token-Utilities
    "extract_bearer_token",
    # Hash-Utilities
    "hash_content",
    # Error-Utilities
    "is_retryable_error",
    "safe_json_deserialize",
    # JSON-Utilities
    "safe_json_serialize",
    "sanitize_subject",
    "validate_subject_format",
]
