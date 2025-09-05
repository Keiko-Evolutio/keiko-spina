"""Gemeinsame Validierungs-Utilities für API-Endpoints.

Eliminiert duplizierte Validierungslogik und stellt wiederverwendbare
Validierungsfunktionen für alle API-Komponenten bereit.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .constants import (
    MAX_INPUT_LENGTH,
    MAX_SERVER_NAME_LENGTH,
    MIN_SERVER_NAME_LENGTH,
    ValidationPatterns,
)
from .error_handlers import ValidationError

# ============================================================================
# COMPILED REGEX PATTERNS
# ============================================================================

# Kompilierte Regex-Pattern für bessere Performance
_UUID_PATTERN = re.compile(ValidationPatterns.UUID_V4, re.IGNORECASE)
_CORRELATION_ID_PATTERN = re.compile(ValidationPatterns.CORRELATION_ID, re.IGNORECASE)
_ALPHANUMERIC_PATTERN = re.compile(ValidationPatterns.ALPHANUMERIC)
_SERVER_NAME_PATTERN = re.compile(ValidationPatterns.SERVER_NAME)
_EMAIL_PATTERN = re.compile(ValidationPatterns.EMAIL, re.IGNORECASE)


# ============================================================================
# BASIC VALIDATION FUNCTIONS
# ============================================================================

def validate_correlation_id(correlation_id: str) -> str:
    """Validiert Korrelations-ID-Format.

    Args:
        correlation_id: Zu validierende Korrelations-ID

    Returns:
        Validierte Korrelations-ID

    Raises:
        ValidationError: Bei ungültigem Format
    """
    if not correlation_id:
        raise ValidationError(
            "Correlation ID ist erforderlich",
            field="correlation_id"
        )

    if not _CORRELATION_ID_PATTERN.match(correlation_id):
        raise ValidationError(
            "Ungültiges Correlation ID Format",
            field="correlation_id",
            value=correlation_id
        )

    return correlation_id


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validiert UUID-Format.

    Args:
        value: Zu validierende UUID
        field_name: Name des Feldes (für Fehlermeldungen)

    Returns:
        Validierte UUID

    Raises:
        ValidationError: Bei ungültiger UUID
    """
    if not value:
        raise ValidationError(
            f"{field_name} ist erforderlich",
            field=field_name
        )

    try:
        # Versuche UUID zu parsen
        uuid.UUID(value)
        return value
    except ValueError:
        raise ValidationError(
            f"Ungültiges UUID-Format für {field_name}",
            field=field_name,
            value=value
        )


def validate_server_name(server_name: str) -> str:
    """Validiert MCP-Server-Name.

    Args:
        server_name: Zu validierender Server-Name

    Returns:
        Validierter Server-Name

    Raises:
        ValidationError: Bei ungültigem Server-Name
    """
    if not server_name:
        raise ValidationError(
            "Server-Name ist erforderlich",
            field="server_name"
        )

    if len(server_name) < MIN_SERVER_NAME_LENGTH or len(server_name) > MAX_SERVER_NAME_LENGTH:
        raise ValidationError(
            f"Server-Name muss zwischen {MIN_SERVER_NAME_LENGTH} und {MAX_SERVER_NAME_LENGTH} Zeichen lang sein",
            field="server_name",
            value=server_name
        )

    if not _SERVER_NAME_PATTERN.match(server_name):
        raise ValidationError(
            "Server-Name darf nur alphanumerische Zeichen, Bindestriche und Unterstriche enthalten",
            field="server_name",
            value=server_name
        )

    return server_name


def validate_email(email: str, field_name: str = "email") -> str:
    """Validiert E-Mail-Adresse.

    Args:
        email: Zu validierende E-Mail-Adresse
        field_name: Name des Feldes (für Fehlermeldungen)

    Returns:
        Validierte E-Mail-Adresse

    Raises:
        ValidationError: Bei ungültiger E-Mail
    """
    if not email:
        raise ValidationError(
            f"{field_name} ist erforderlich",
            field=field_name
        )

    if not _EMAIL_PATTERN.match(email):
        raise ValidationError(
            f"Ungültiges E-Mail-Format für {field_name}",
            field=field_name,
            value=email
        )

    return email.lower()  # Normalisierung zu Kleinbuchstaben


# ============================================================================
# ADVANCED VALIDATION FUNCTIONS
# ============================================================================

def validate_json_payload(
    payload: dict[str, Any],
    model_class: type[BaseModel],
    field_name: str = "payload"
) -> BaseModel:
    """Validiert JSON-Payload gegen Pydantic-Model.

    Args:
        payload: Zu validierendes JSON-Payload
        model_class: Pydantic-Model-Klasse
        field_name: Name des Feldes (für Fehlermeldungen)

    Returns:
        Validiertes Model-Objekt

    Raises:
        ValidationError: Bei Validierungsfehlern
    """
    try:
        return model_class(**payload)
    except PydanticValidationError as exc:
        # Erste Validierungsfehler extrahieren
        first_error = exc.errors()[0]
        field_path = ".".join(str(loc) for loc in first_error["loc"])

        raise ValidationError(
            f"Validierungsfehler in {field_name}.{field_path}: {first_error['msg']}",
            field=f"{field_name}.{field_path}"
        )


def sanitize_input(
    value: str,
    max_length: int | None = None,
    allow_html: bool = False,
    field_name: str = "input"
) -> str:
    """Sanitisiert Eingabe-String.

    Args:
        value: Zu sanitisierender String
        max_length: Maximale Länge (Standard: MAX_INPUT_LENGTH)
        allow_html: HTML-Tags erlauben
        field_name: Name des Feldes

    Returns:
        Sanitisierter String

    Raises:
        ValidationError: Bei ungültiger Eingabe
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} muss ein String sein",
            field=field_name,
            value=type(value).__name__
        )

    # Whitespace trimmen
    value = value.strip()

    # Standard-Maximum verwenden falls nicht angegeben
    effective_max_length = max_length or MAX_INPUT_LENGTH

    # Länge prüfen
    if len(value) > effective_max_length:
        raise ValidationError(
            f"{field_name} darf maximal {effective_max_length} Zeichen lang sein",
            field=field_name,
            value=f"{len(value)} Zeichen"
        )

    # HTML-Tags entfernen falls nicht erlaubt
    if not allow_html:
        # Einfache HTML-Tag-Entfernung (für komplexere Fälle bleach verwenden)
        value = re.sub(r"<[^>]+>", "", value)

    return value


def check_required_fields(
    data: dict[str, Any],
    required_fields: list[str],
    context: str = "request"
) -> None:
    """Prüft ob alle erforderlichen Felder vorhanden sind.

    Args:
        data: Zu prüfende Daten
        required_fields: Liste erforderlicher Felder
        context: Kontext für Fehlermeldungen

    Raises:
        ValidationError: Bei fehlenden Feldern
    """
    missing_fields = []

    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)

    if missing_fields:
        raise ValidationError(
            f"Erforderliche Felder fehlen in {context}: {', '.join(missing_fields)}",
            field="required_fields"
        )


def validate_enum_value(
    value: str,
    enum_class: type,
    field_name: str = "value"
) -> Any:
    """Validiert Enum-Wert.

    Args:
        value: Zu validierender Wert
        enum_class: Enum-Klasse
        field_name: Name des Feldes

    Returns:
        Validierter Enum-Wert

    Raises:
        ValidationError: Bei ungültigem Enum-Wert
    """
    try:
        return enum_class(value)
    except ValueError:
        valid_values = [e.value for e in enum_class]
        raise ValidationError(
            f"Ungültiger Wert für {field_name}. Erlaubte Werte: {', '.join(valid_values)}",
            field=field_name,
            value=value
        )


# ============================================================================
# VALIDATION DECORATORS
# ============================================================================

def validate_request_data(required_fields: list[str] | None = None):
    """Decorator für Request-Daten-Validierung.

    Args:
        required_fields: Liste erforderlicher Felder

    Returns:
        Decorator-Funktion
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Request-Daten aus kwargs extrahieren
            request_data = kwargs.get("request_data", {})

            if required_fields:
                check_required_fields(request_data, required_fields, "request")

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_path_params(**param_validators):
    """Decorator für Path-Parameter-Validierung.

    Args:
        **param_validators: Dictionary von Parameter-Namen zu Validator-Funktionen

    Returns:
        Decorator-Funktion
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for param_name, validator in param_validators.items():
                if param_name in kwargs:
                    kwargs[param_name] = validator(kwargs[param_name])

            return await func(*args, **kwargs)
        return wrapper
    return decorator
