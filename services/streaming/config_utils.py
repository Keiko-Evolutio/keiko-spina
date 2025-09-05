"""Konfiguration-Utilities für KEI-Stream.

Zentrale Funktionen für das Parsen von Umgebungsvariablen und
Konfigurationswerten mit konsistenter Fehlerbehandlung.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, TypeVar

from kei_logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def get_env_bool(name: str, default: bool = False) -> bool:
    """Liest Boolean-Wert aus Umgebungsvariable.

    Args:
        name: Name der Umgebungsvariable
        default: Fallback-Wert wenn Variable nicht gesetzt

    Returns:
        Boolean-Wert der Umgebungsvariable oder Default
    """
    value = os.getenv(name, "").lower().strip()
    if not value:
        return default
    return value in {"true", "1", "yes", "on", "enabled"}


def get_env_int(name: str, default: int = 0) -> int:
    """Liest Integer-Wert aus Umgebungsvariable.

    Args:
        name: Name der Umgebungsvariable
        default: Fallback-Wert wenn Variable nicht gesetzt oder ungültig

    Returns:
        Integer-Wert der Umgebungsvariable oder Default
    """
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return int(value.strip())
    except (ValueError, AttributeError) as e:
        logger.warning(f"Ungültiger Integer-Wert für {name}: {e}, verwende Default {default}")
        return default


def get_env_float(name: str, default: float = 0.0) -> float:
    """Liest Float-Wert aus Umgebungsvariable.

    Args:
        name: Name der Umgebungsvariable
        default: Fallback-Wert wenn Variable nicht gesetzt oder ungültig

    Returns:
        Float-Wert der Umgebungsvariable oder Default
    """
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return float(value.strip())
    except (ValueError, AttributeError) as e:
        logger.warning(f"Ungültiger Float-Wert für {name}: {e}, verwende Default {default}")
        return default


def get_env_str(name: str, default: str = "") -> str:
    """Liest String-Wert aus Umgebungsvariable.

    Args:
        name: Name der Umgebungsvariable
        default: Fallback-Wert wenn Variable nicht gesetzt

    Returns:
        String-Wert der Umgebungsvariable oder Default
    """
    return os.getenv(name, default).strip()


def get_env_json(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Liest JSON-Objekt aus Umgebungsvariable.

    Args:
        name: Name der Umgebungsvariable
        default: Fallback-Wert wenn Variable nicht gesetzt oder ungültiges JSON

    Returns:
        Geparste JSON-Daten oder Default
    """
    if default is None:
        default = {}

    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        parsed = json.loads(raw_value)
        if not isinstance(parsed, dict):
            logger.warning(f"JSON in {name} ist kein Dictionary, verwende Default")
            return default
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Ungültiges JSON in {name}: {e}, verwende Default")
        return default


def get_env_list(name: str, separator: str = ",", default: list | None = None) -> list:
    """Liest Liste aus Umgebungsvariable.

    Args:
        name: Name der Umgebungsvariable
        separator: Trennzeichen für Liste (Standard: Komma)
        default: Fallback-Wert wenn Variable nicht gesetzt

    Returns:
        Liste von Strings oder Default
    """
    if default is None:
        default = []

    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    return [item.strip() for item in raw_value.split(separator) if item.strip()]


def validate_env_range(
    name: str,
    value: int | float,
    min_value: int | float | None = None,
    max_value: int | float | None = None
) -> bool:
    """Validiert ob ein Umgebungsvariablen-Wert in einem gültigen Bereich liegt.

    Args:
        name: Name der Umgebungsvariable (für Logging)
        value: Zu validierender Wert
        min_value: Minimaler erlaubter Wert
        max_value: Maximaler erlaubter Wert

    Returns:
        True wenn Wert im gültigen Bereich liegt
    """
    if min_value is not None and value < min_value:
        logger.warning(f"Wert für {name} ({value}) ist kleiner als Minimum ({min_value})")
        return False

    if max_value is not None and value > max_value:
        logger.warning(f"Wert für {name} ({value}) ist größer als Maximum ({max_value})")
        return False

    return True


def get_env_with_validation[T](
    name: str,
    parser_func: Callable,
    default: T,
    min_value: int | float | None = None,
    max_value: int | float | None = None
) -> T:
    """Liest und validiert Umgebungsvariable mit benutzerdefinierter Parser-Funktion.

    Args:
        name: Name der Umgebungsvariable
        parser_func: Funktion zum Parsen des Werts (z.B. get_env_int)
        default: Fallback-Wert
        min_value: Minimaler erlaubter Wert (optional)
        max_value: Maximaler erlaubter Wert (optional)

    Returns:
        Geparster und validierter Wert oder Default
    """
    try:
        value = parser_func(name, default)

        # Validierung nur für numerische Werte
        if isinstance(value, int | float) and (min_value is not None or max_value is not None):
            if not validate_env_range(name, value, min_value, max_value):
                logger.warning(f"Verwende Default-Wert {default} für {name}")
                return default

        return value
    except Exception as e:
        logger.exception(f"Fehler beim Parsen von {name}: {e}, verwende Default {default}")
        return default


def resolve_hierarchical_config(
    base_config: dict[str, Any],
    tenant_id: str | None = None,
    api_key: str | None = None,
    config_key: str = "default"
) -> dict[str, Any]:
    """Löst hierarchische Konfiguration auf (API-Key > Tenant > Default).

    Args:
        base_config: Basis-Konfigurationsdictionary
        tenant_id: Tenant-ID für tenant-spezifische Konfiguration
        api_key: API-Key für key-spezifische Konfiguration
        config_key: Schlüssel für Default-Konfiguration

    Returns:
        Aufgelöste Konfiguration
    """
    # Starte mit Default-Konfiguration
    effective_config = base_config.get(config_key, {}).copy()

    # Überschreibe mit Tenant-spezifischer Konfiguration
    if tenant_id and isinstance(base_config.get("tenants"), dict):
        tenant_config = base_config["tenants"].get(tenant_id, {})
        if isinstance(tenant_config, dict):
            effective_config.update(tenant_config)

    # Überschreibe mit API-Key-spezifischer Konfiguration (höchste Priorität)
    if api_key and isinstance(base_config.get("api_keys"), dict):
        api_config = base_config["api_keys"].get(api_key, {})
        if isinstance(api_config, dict):
            effective_config.update(api_config)

    return effective_config


def get_tenant_specific_env[T](
    base_name: str,
    tenant_id: str | None = None,
    api_key: str | None = None,
    parser_func: Callable = get_env_str,
    default: T = None
) -> T:
    """Liest tenant- oder API-key-spezifische Umgebungsvariable.

    Auflösungsreihenfolge:
    1. {base_name}_APIKEY_{api_key}
    2. {base_name}_TENANT_{tenant_id}
    3. {base_name}

    Args:
        base_name: Basis-Name der Umgebungsvariable
        tenant_id: Tenant-ID für tenant-spezifische Variable
        api_key: API-Key für key-spezifische Variable
        parser_func: Parser-Funktion (Standard: get_env_str)
        default: Fallback-Wert

    Returns:
        Wert der spezifischsten verfügbaren Umgebungsvariable
    """
    # API-Key hat höchste Priorität
    if api_key:
        api_key_var = f"{base_name}_APIKEY_{api_key.upper()}"
        if os.getenv(api_key_var):
            return parser_func(api_key_var, default)

    # Tenant-spezifische Variable
    if tenant_id:
        tenant_var = f"{base_name}_TENANT_{tenant_id.upper()}"
        if os.getenv(tenant_var):
            return parser_func(tenant_var, default)

    # Fallback auf Basis-Variable
    return parser_func(base_name, default)


__all__ = [
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "get_env_json",
    "get_env_list",
    "get_env_str",
    "get_env_with_validation",
    "get_tenant_specific_env",
    "resolve_hierarchical_config",
    "validate_env_range",
]
