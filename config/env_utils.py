"""Zentrale Utilities für Environment Variable Loading.

Eliminiert duplizierte Environment Loading Logic und bietet
typsichere, konsistente Funktionen für alle Config-Module.
"""

import os
from typing import TypeVar

from kei_logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def get_env_str(
    key: str,
    default: str = "",
    required: bool = False
) -> str:
    """Lädt String-Wert aus Umgebungsvariable.

    Args:
        key: Environment Variable Name
        default: Default-Wert falls Variable nicht gesetzt
        required: Ob Variable erforderlich ist

    Returns:
        String-Wert der Environment Variable

    Raises:
        ValueError: Wenn required=True und Variable nicht gesetzt
    """
    value = os.getenv(key, default)

    if required and not value:
        raise ValueError(f"Erforderliche Environment Variable nicht gesetzt: {key}")

    return value


def get_env_int(
    key: str,
    default: int = 0,
    min_value: int | None = None,
    max_value: int | None = None,
    required: bool = False
) -> int:
    """Lädt Integer-Wert aus Umgebungsvariable.

    Args:
        key: Environment Variable Name
        default: Default-Wert falls Variable nicht gesetzt
        min_value: Minimaler erlaubter Wert
        max_value: Maximaler erlaubter Wert
        required: Ob Variable erforderlich ist

    Returns:
        Integer-Wert der Environment Variable

    Raises:
        ValueError: Bei ungültigen Werten oder Verletzung der Constraints
    """
    value_str = os.getenv(key)

    if value_str is None:
        if required:
            raise ValueError(f"Erforderliche Environment Variable nicht gesetzt: {key}")
        return default

    try:
        value = int(value_str)
    except ValueError:
        logger.warning(f"Ungültiger Integer-Wert für {key}: {value_str}, verwende Default: {default}")
        return default

    # Validiere Constraints
    if min_value is not None and value < min_value:
        logger.warning(f"Wert für {key} ({value}) unter Minimum ({min_value}), verwende Minimum")
        return min_value

    if max_value is not None and value > max_value:
        logger.warning(f"Wert für {key} ({value}) über Maximum ({max_value}), verwende Maximum")
        return max_value

    return value


def get_env_float(
    key: str,
    default: float = 0.0,
    min_value: float | None = None,
    max_value: float | None = None,
    required: bool = False
) -> float:
    """Lädt Float-Wert aus Umgebungsvariable.

    Args:
        key: Environment Variable Name
        default: Default-Wert falls Variable nicht gesetzt
        min_value: Minimaler erlaubter Wert
        max_value: Maximaler erlaubter Wert
        required: Ob Variable erforderlich ist

    Returns:
        Float-Wert der Environment Variable

    Raises:
        ValueError: Bei ungültigen Werten oder Verletzung der Constraints
    """
    value_str = os.getenv(key)

    if value_str is None:
        if required:
            raise ValueError(f"Erforderliche Environment Variable nicht gesetzt: {key}")
        return default

    try:
        value = float(value_str)
    except ValueError:
        logger.warning(f"Ungültiger Float-Wert für {key}: {value_str}, verwende Default: {default}")
        return default

    # Validiere Constraints
    if min_value is not None and value < min_value:
        logger.warning(f"Wert für {key} ({value}) unter Minimum ({min_value}), verwende Minimum")
        return min_value

    if max_value is not None and value > max_value:
        logger.warning(f"Wert für {key} ({value}) über Maximum ({max_value}), verwende Maximum")
        return max_value

    return value


def get_env_bool(
    key: str,
    default: bool = False,
    required: bool = False
) -> bool:
    """Lädt Boolean-Wert aus Umgebungsvariable.

    Args:
        key: Environment Variable Name
        default: Default-Wert falls Variable nicht gesetzt
        required: Ob Variable erforderlich ist

    Returns:
        Boolean-Wert der Environment Variable

    Raises:
        ValueError: Wenn required=True und Variable nicht gesetzt
    """
    value_str = os.getenv(key)

    if value_str is None:
        if required:
            raise ValueError(f"Erforderliche Environment Variable nicht gesetzt: {key}")
        return default

    # Erkenne verschiedene Boolean-Repräsentationen
    true_values = {"true", "1", "yes", "on", "enabled"}
    false_values = {"false", "0", "no", "off", "disabled"}

    value_lower = value_str.lower().strip()

    if value_lower in true_values:
        return True
    if value_lower in false_values:
        return False
    logger.warning(f"Ungültiger Boolean-Wert für {key}: {value_str}, verwende Default: {default}")
    return default


def get_env_list(
    key: str,
    default: list[str] | None = None,
    separator: str = ",",
    strip_items: bool = True,
    filter_empty: bool = True,
    required: bool = False
) -> list[str]:
    """Lädt Liste von Strings aus Umgebungsvariable.

    Args:
        key: Environment Variable Name
        default: Default-Liste falls Variable nicht gesetzt
        separator: Trennzeichen für Liste
        strip_items: Ob Whitespace von Items entfernt werden soll
        filter_empty: Ob leere Items gefiltert werden sollen
        required: Ob Variable erforderlich ist

    Returns:
        Liste von Strings aus Environment Variable

    Raises:
        ValueError: Wenn required=True und Variable nicht gesetzt
    """
    if default is None:
        default = []

    value_str = os.getenv(key)

    if value_str is None:
        if required:
            raise ValueError(f"Erforderliche Environment Variable nicht gesetzt: {key}")
        return default

    # Splitte und verarbeite Items
    items = value_str.split(separator)

    if strip_items:
        items = [item.strip() for item in items]

    if filter_empty:
        items = [item for item in items if item]

    return items


def get_env_enum[T](
    key: str,
    enum_class: type,
    default: T | None = None,
    required: bool = False
) -> T:
    """Lädt Enum-Wert aus Umgebungsvariable.

    Args:
        key: Environment Variable Name
        enum_class: Enum-Klasse
        default: Default-Wert falls Variable nicht gesetzt
        required: Ob Variable erforderlich ist

    Returns:
        Enum-Wert der Environment Variable

    Raises:
        ValueError: Bei ungültigen Enum-Werten oder wenn required=True und Variable nicht gesetzt
    """
    value_str = os.getenv(key)

    if value_str is None:
        if required:
            raise ValueError(f"Erforderliche Environment Variable nicht gesetzt: {key}")
        return default

    try:
        # Versuche Enum-Wert zu erstellen
        return enum_class(value_str.lower())
    except ValueError:
        valid_values = [e.value for e in enum_class]
        logger.warning(f"Ungültiger Enum-Wert für {key}: {value_str}, "
                      f"gültige Werte: {valid_values}, verwende Default: {default}")
        return default


def get_env_optional_str(key: str) -> str | None:
    """Lädt optionalen String aus Umgebungsvariable.

    Args:
        key: Environment Variable Name

    Returns:
        String-Wert oder None falls nicht gesetzt
    """
    value = os.getenv(key)
    return value if value else None


def load_env_config(
    prefix: str,
    config_mapping: dict,
    logger_name: str | None = None
) -> dict:
    """Lädt Konfiguration basierend auf Prefix und Mapping.

    Args:
        prefix: Prefix für Environment Variables (z.B. "KEI_MCP_")
        config_mapping: Mapping von Config-Keys zu (env_suffix, loader_func, default)
        logger_name: Name für Logger (optional)

    Returns:
        Dictionary mit geladener Konfiguration

    Example:
        config_mapping = {
            "timeout": ("TIMEOUT", get_env_float, 30.0),
            "retries": ("MAX_RETRIES", get_env_int, 3),
            "enabled": ("ENABLED", get_env_bool, True)
        }

        config = load_env_config("KEI_MCP_", config_mapping)
        # Lädt KEI_MCP_TIMEOUT, KEI_MCP_MAX_RETRIES, KEI_MCP_ENABLED
    """
    local_logger = get_logger(logger_name) if logger_name else logger

    config = {}

    for config_key, (env_suffix, loader_func, default_value) in config_mapping.items():
        env_key = f"{prefix}{env_suffix}"

        try:
            if loader_func in (get_env_str, get_env_bool) or loader_func in [get_env_int, get_env_float]:
                config[config_key] = loader_func(env_key, default_value)
            elif loader_func == get_env_list:
                config[config_key] = loader_func(env_key, default_value or [])
            else:
                # Fallback für custom loader functions
                config[config_key] = loader_func(env_key, default_value)

        except Exception as e:
            local_logger.exception(f"Fehler beim Laden von {env_key}: {e}")
            config[config_key] = default_value

    local_logger.debug(f"Konfiguration geladen mit Prefix {prefix}: {len(config)} Werte")
    return config


def validate_required_env_vars(required_vars: list[str]) -> None:
    """Validiert dass alle erforderlichen Environment Variables gesetzt sind.

    Args:
        required_vars: Liste der erforderlichen Environment Variable Namen

    Raises:
        ValueError: Wenn eine oder mehrere erforderliche Variables nicht gesetzt sind
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Erforderliche Environment Variables nicht gesetzt: {', '.join(missing_vars)}")


def get_env_with_fallbacks(keys: list[str], default: str = "") -> str:
    """Lädt String-Wert mit Fallback-Keys.

    Args:
        keys: Liste von Environment Variable Namen (in Prioritätsreihenfolge)
        default: Default-Wert falls keine Variable gesetzt

    Returns:
        Erster gefundene Wert oder Default
    """
    for key in keys:
        value = os.getenv(key)
        if value:
            return value

    return default


# Convenience Functions für häufige Patterns
def get_redis_config(prefix: str = "REDIS_") -> dict:
    """Lädt Standard-Redis-Konfiguration.

    Args:
        prefix: Prefix für Redis Environment Variables

    Returns:
        Dictionary mit Redis-Konfiguration
    """
    return load_env_config(prefix, {
        "host": ("HOST", get_env_str, "localhost"),
        "port": ("PORT", get_env_int, 6379),
        "db": ("DB", get_env_int, 0),
        "password": ("PASSWORD", lambda key, default: get_env_optional_str(key), None),
        "ssl": ("SSL", get_env_bool, False),
        "timeout": ("TIMEOUT", get_env_float, 5.0),
    })


def get_auth_config(prefix: str = "AUTH_") -> dict:
    """Lädt Standard-Auth-Konfiguration.

    Args:
        prefix: Prefix für Auth Environment Variables

    Returns:
        Dictionary mit Auth-Konfiguration
    """
    return load_env_config(prefix, {
        "enabled": ("ENABLED", get_env_bool, False),
        "mode": ("MODE", get_env_str, "disabled"),
        "secret_key": ("SECRET_KEY", get_env_optional_str, None),
        "algorithm": ("ALGORITHM", get_env_str, "HS256"),
        "timeout": ("TIMEOUT", get_env_int, 3600),
    })
