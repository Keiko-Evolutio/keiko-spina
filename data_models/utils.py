# backend/data_models/utils.py
"""Utility-Funktionen für Data Models - Gemeinsame Hilfsfunktionen."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse


def generate_uuid() -> str:
    """Generiert eine neue UUID als String.

    Returns:
        str: UUID als String
    """
    return str(uuid.uuid4())


def generate_short_id(prefix: str = "") -> str:
    """Generiert eine kurze ID mit optionalem Prefix.

    Args:
        prefix: Optionaler Prefix für die ID

    Returns:
        str: Kurze ID mit Prefix
    """
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}{short_uuid}" if prefix else short_uuid


def utc_now() -> datetime:
    """Gibt aktuellen UTC-Zeitstempel zurück.

    Returns:
        datetime: Aktueller UTC-Zeitstempel
    """
    return datetime.now(UTC)


def validate_non_empty_string(value: Any, field_name: str) -> None:
    """Validiert dass ein Wert ein nicht-leerer String ist.

    Args:
        value: Zu validierender Wert
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValueError: Wenn Wert kein nicht-leerer String ist
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} ist erforderlich und darf nicht leer sein")


def validate_uri(uri: str, field_name: str = "URI") -> None:
    """Validiert dass eine URI gültig ist.

    Args:
        uri: Zu validierende URI
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValueError: Wenn URI ungültig ist
    """
    try:
        parsed = urlparse(uri)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"{field_name} ist ungültig: {uri}")
    except Exception as e:
        raise ValueError(f"{field_name} ist ungültig: {uri}") from e


def validate_optional_non_empty_string(value: str | None, field_name: str) -> None:
    """Validiert dass ein optionaler String, falls gesetzt, nicht leer ist.

    Args:
        value: Zu validierender optionaler String
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValueError: Wenn String gesetzt aber leer ist
    """
    if value is not None and not str(value).strip():
        raise ValueError(f"{field_name} darf nicht leer sein")


def ensure_dict_default() -> dict[str, Any]:
    """Factory-Funktion für leere Dictionaries.

    Returns:
        Dict[str, Any]: Leeres Dictionary
    """
    return {}


def ensure_list_default() -> list[Any]:
    """Factory-Funktion für leere Listen.

    Returns:
        List[Any]: Leere Liste
    """
    return []


def create_event_id(prefix: str = "evt") -> str:
    """Erstellt eine Event-ID mit Prefix.

    Args:
        prefix: Prefix für die Event-ID

    Returns:
        str: Event-ID mit Prefix
    """
    return generate_short_id(f"{prefix}_")


def create_update_id() -> str:
    """Erstellt eine Update-ID.

    Returns:
        str: Update-ID
    """
    return generate_uuid()


def safe_string_conversion(value: Any) -> str:
    """Konvertiert einen Wert sicher zu String.

    Args:
        value: Zu konvertierender Wert

    Returns:
        str: String-Repräsentation des Werts
    """
    if value is None:
        return ""
    return str(value)


class ValidationMixin:
    """Mixin-Klasse für gemeinsame Validierungslogik."""

    def validate_required_string(self, value: Any, field_name: str) -> None:
        """Validiert ein erforderliches String-Feld.

        Args:
            value: Zu validierender Wert
            field_name: Name des Feldes

        Raises:
            ValueError: Wenn Validierung fehlschlägt
        """
        validate_non_empty_string(value, field_name)

    def validate_optional_string(self, value: str | None, field_name: str) -> None:
        """Validiert ein optionales String-Feld.

        Args:
            value: Zu validierender optionaler String
            field_name: Name des Feldes

        Raises:
            ValueError: Wenn Validierung fehlschlägt
        """
        validate_optional_non_empty_string(value, field_name)

    def validate_uri_field(self, uri: str, field_name: str = "URI") -> None:
        """Validiert ein URI-Feld.

        Args:
            uri: Zu validierende URI
            field_name: Name des Feldes

        Raises:
            ValueError: Wenn Validierung fehlschlägt
        """
        validate_uri(uri, field_name)


def create_default_metadata() -> dict[str, Any]:
    """Erstellt Standard-Metadaten-Dictionary.

    Returns:
        Dict[str, Any]: Standard-Metadaten
    """
    return {
        "created_at": utc_now(),
        "version": "1.0.0"
    }


def merge_metadata(
        base: dict[str, Any],
        additional: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Führt Metadaten-Dictionaries zusammen.

    Args:
        base: Basis-Metadaten
        additional: Zusätzliche Metadaten

    Returns:
        Dict[str, Any]: Zusammengeführte Metadaten
    """
    result = base.copy()
    if additional:
        result.update(additional)
    return result
