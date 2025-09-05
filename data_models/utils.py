# backend/data_models/utils.py
"""Utility-Funktionen für Data Models - Gemeinsame Hilfsfunktionen."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any


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




