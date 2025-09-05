# backend/agents/slo_sla/utils.py
"""Utility-Funktionen für das SLO/SLA Management-System.

Konsolidiert wiederkehrende Code-Patterns und bietet wiederverwendbare
Funktionen für History-Management, Thread-Safety, Serialisierung und
andere gemeinsame Operationen.
"""

import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

from .constants import (
    DEFAULT_HISTORY_LIMIT,
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PERCENTAGE,
    MAX_TAG_KEY_LENGTH,
    MAX_TAG_VALUE_LENGTH,
    MIN_PERCENTAGE,
)

logger = get_logger(__name__)

T = TypeVar("T")

# =============================================================================
# PROTOCOLS UND INTERFACES
# =============================================================================

@runtime_checkable
class Serializable(Protocol):
    """Protocol für serialisierbare Objekte."""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Objekt zu Dictionary."""
        ...


@runtime_checkable
class HasTimestamp(Protocol):
    """Protocol für Objekte mit Timestamp."""

    timestamp: float


# =============================================================================
# THREAD-SAFETY UTILITIES
# =============================================================================

class ThreadSafeManager:
    """Basis-Klasse für Thread-Safe-Management mit RLock."""

    def __init__(self):
        """Initialisiert Thread-Safe-Manager."""
        self._lock = threading.RLock()

    def with_lock(self, func, *args, **kwargs):
        """Führt Funktion mit Lock aus.

        Args:
            func: Auszuführende Funktion
            *args: Funktions-Argumente
            **kwargs: Funktions-Keyword-Argumente

        Returns:
            Funktions-Rückgabewert
        """
        with self._lock:
            return func(*args, **kwargs)


def create_thread_safe_lock() -> threading.RLock:
    """Erstellt einen neuen Thread-Safe RLock.

    Returns:
        Neuer RLock
    """
    return threading.RLock()


# =============================================================================
# HISTORY-MANAGEMENT UTILITIES
# =============================================================================

def create_limited_history(max_size: int = DEFAULT_HISTORY_LIMIT) -> deque:
    """Erstellt eine begrenzte History-Deque.

    Args:
        max_size: Maximale Größe der History

    Returns:
        Begrenzte Deque
    """
    return deque(maxlen=max_size)


def add_to_limited_history(
    history: list[T],
    item: T,
    max_size: int = DEFAULT_HISTORY_LIMIT
) -> None:
    """Fügt Item zu begrenzter History-Liste hinzu.

    Args:
        history: History-Liste
        item: Hinzuzufügendes Item
        max_size: Maximale Größe der History
    """
    history.append(item)

    # Behalte nur die letzten max_size Items
    if len(history) > max_size:
        history[:] = history[-max_size:]


def cleanup_old_items_by_timestamp(
    items: list[HasTimestamp],
    max_age_seconds: float
) -> list[HasTimestamp]:
    """Entfernt alte Items basierend auf Timestamp.

    Args:
        items: Liste von Items mit Timestamp
        max_age_seconds: Maximales Alter in Sekunden

    Returns:
        Gefilterte Liste ohne alte Items
    """
    cutoff_time = time.time() - max_age_seconds
    return [item for item in items if item.timestamp >= cutoff_time]


# =============================================================================
# SERIALISIERUNG UTILITIES
# =============================================================================

def safe_to_dict(obj: Any) -> dict[str, Any]:
    """Konvertiert Objekt sicher zu Dictionary.

    Unterstützt Dataclasses, Enums, Serializable-Objekte und primitive Typen.

    Args:
        obj: Zu konvertierendes Objekt

    Returns:
        Dictionary-Repräsentation
    """
    if obj is None:
        return {}

    # Dataclass
    if is_dataclass(obj):
        result = {}
        for key, value in asdict(obj).items():
            result[key] = safe_to_dict(value) if not _is_primitive(value) else value
        return result

    # Serializable Protocol
    if isinstance(obj, Serializable):
        return obj.to_dict()

    # Enum
    if isinstance(obj, Enum):
        return obj.value

    # List/Tuple
    if isinstance(obj, (list, tuple)):
        return [safe_to_dict(item) if not _is_primitive(item) else item for item in obj]

    # Dict
    if isinstance(obj, dict):
        return {
            key: safe_to_dict(value) if not _is_primitive(value) else value
            for key, value in obj.items()
        }

    # Primitive oder unbekannt
    return obj


def _is_primitive(value: Any) -> bool:
    """Prüft ob Wert ein primitiver Typ ist.

    Args:
        value: Zu prüfender Wert

    Returns:
        True wenn primitiv
    """
    return isinstance(value, (str, int, float, bool, type(None)))


def create_standardized_to_dict(
    obj: Any,
    additional_fields: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt standardisierte to_dict-Implementierung.

    Args:
        obj: Objekt für Serialisierung
        additional_fields: Zusätzliche Felder

    Returns:
        Standardisiertes Dictionary
    """
    result = safe_to_dict(obj)

    if additional_fields:
        result.update(additional_fields)

    return result


# =============================================================================
# METRICS UTILITIES
# =============================================================================

class MetricsHelper:
    """Helper-Klasse für einheitliche Metrics-Collection."""

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        """Initialisiert Metrics-Helper.

        Args:
            metrics_collector: Optional MetricsCollector-Instanz
        """
        self._metrics_collector = metrics_collector or MetricsCollector()

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None
    ) -> None:
        """Zeichnet Counter-Metric auf.

        Args:
            name: Metric-Name
            value: Metric-Wert
            tags: Tags für Metric
        """
        try:
            self._metrics_collector.increment_counter(name, value, tags or {})
        except Exception as e:
            logger.warning(f"Fehler beim Aufzeichnen von Counter-Metric {name}: {e}")

    def record_gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None
    ) -> None:
        """Zeichnet Gauge-Metric auf.

        Args:
            name: Metric-Name
            value: Metric-Wert
            tags: Tags für Metric
        """
        try:
            self._metrics_collector.record_gauge(name, value, tags or {})
        except Exception as e:
            logger.warning(f"Fehler beim Aufzeichnen von Gauge-Metric {name}: {e}")


def create_metrics_helper() -> MetricsHelper:
    """Erstellt neuen MetricsHelper.

    Returns:
        Neuer MetricsHelper
    """
    return MetricsHelper()


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_percentage(value: float, field_name: str = "percentage") -> None:
    """Validiert Percentage-Wert.

    Args:
        value: Zu validierender Wert
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValueError: Wenn Wert ungültig
    """
    if not MIN_PERCENTAGE <= value <= MAX_PERCENTAGE:
        raise ValueError(
            f"{field_name} muss zwischen {MIN_PERCENTAGE} und {MAX_PERCENTAGE} liegen, "
            f"erhalten: {value}"
        )


def validate_string_length(
    value: str,
    max_length: int,
    field_name: str = "string"
) -> None:
    """Validiert String-Länge.

    Args:
        value: Zu validierender String
        max_length: Maximale Länge
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValueError: Wenn String zu lang
    """
    if len(value) > max_length:
        raise ValueError(
            f"{field_name} darf maximal {max_length} Zeichen lang sein, "
            f"erhalten: {len(value)} Zeichen"
        )


def validate_name(name: str) -> None:
    """Validiert Namen (SLO/SLA-Namen, etc.).

    Args:
        name: Zu validierender Name

    Raises:
        ValueError: Wenn Name ungültig
    """
    if not name or not name.strip():
        raise ValueError("Name darf nicht leer sein")

    validate_string_length(name, MAX_NAME_LENGTH, "Name")


def validate_description(description: str) -> None:
    """Validiert Beschreibung.

    Args:
        description: Zu validierende Beschreibung

    Raises:
        ValueError: Wenn Beschreibung ungültig
    """
    validate_string_length(description, MAX_DESCRIPTION_LENGTH, "Beschreibung")


def validate_tags(tags: dict[str, str]) -> None:
    """Validiert Tag-Dictionary.

    Args:
        tags: Zu validierende Tags

    Raises:
        ValueError: Wenn Tags ungültig
    """
    for key, value in tags.items():
        validate_string_length(key, MAX_TAG_KEY_LENGTH, f"Tag-Key '{key}'")
        validate_string_length(value, MAX_TAG_VALUE_LENGTH, f"Tag-Value für '{key}'")


# =============================================================================
# TIME UTILITIES
# =============================================================================

def get_current_timestamp() -> float:
    """Gibt aktuellen Timestamp zurück.

    Returns:
        Aktueller Unix-Timestamp
    """
    return time.time()


def is_timestamp_recent(timestamp: float, max_age_seconds: float) -> bool:
    """Prüft ob Timestamp recent ist.

    Args:
        timestamp: Zu prüfender Timestamp
        max_age_seconds: Maximales Alter in Sekunden

    Returns:
        True wenn Timestamp recent ist
    """
    return (get_current_timestamp() - timestamp) <= max_age_seconds


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

def safe_execute(func, default_value=None, log_errors: bool = True):
    """Führt Funktion sicher aus mit Error-Handling.

    Args:
        func: Auszuführende Funktion
        default_value: Default-Wert bei Fehler
        log_errors: Ob Fehler geloggt werden sollen

    Returns:
        Funktions-Rückgabewert oder default_value
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Fehler bei Ausführung von {func.__name__}: {e}")
        return default_value


def log_operation_duration(operation_name: str):
    """Decorator für Logging der Operationsdauer.

    Args:
        operation_name: Name der Operation

    Returns:
        Decorator-Funktion
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = get_current_timestamp()
            try:
                result = func(*args, **kwargs)
                duration = get_current_timestamp() - start_time
                logger.debug(f"{operation_name} abgeschlossen in {duration:.3f}s")
                return result
            except Exception as e:
                duration = get_current_timestamp() - start_time
                logger.error(f"{operation_name} fehlgeschlagen nach {duration:.3f}s: {e}")
                raise
        return wrapper
    return decorator
