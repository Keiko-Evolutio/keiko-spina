# backend/task_management/utils.py
"""Utility-Funktionen für Task-Management-System.

Konsolidiert gemeinsame Funktionalitäten um Code-Duplikation zu eliminieren
und Clean Code Prinzipien zu befolgen.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, TypeVar

from kei_logging import get_logger

# Type Variables für generische Funktionen
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Globaler Logger für Utilities
_logger = get_logger(__name__)


def generate_uuid() -> str:
    """Generiert eine neue UUID als String.

    Returns:
        UUID als String
    """
    return str(uuid.uuid4())


def get_current_utc_datetime() -> datetime:
    """Gibt aktuelles UTC-Datetime zurück.

    Returns:
        Aktuelles UTC-Datetime
    """
    return datetime.now(UTC)


def get_module_logger(module_name: str):
    """Erstellt Logger für ein Modul.

    Args:
        module_name: Name des Moduls

    Returns:
        Konfigurierter Logger
    """
    return get_logger(module_name)


async def create_async_lock() -> asyncio.Lock:
    """Erstellt einen neuen Async-Lock.

    Returns:
        Neuer asyncio.Lock
    """
    return asyncio.Lock()


def safe_dict_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Sichere Dictionary-Zugriff mit Default-Wert.

    Args:
        data: Dictionary
        key: Schlüssel
        default: Default-Wert

    Returns:
        Wert oder Default
    """
    return data.get(key, default)


def validate_required_fields(data: dict[str, Any], required_fields: list[str]) -> None:
    """Validiert erforderliche Felder in Dictionary.

    Args:
        data: Zu validierendes Dictionary
        required_fields: Liste erforderlicher Felder

    Raises:
        ValueError: Wenn erforderliche Felder fehlen
    """
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Erforderliche Felder fehlen: {missing_fields}")


def sanitize_string(value: str, max_length: int | None = None) -> str:
    """Bereinigt String-Werte.

    Args:
        value: Zu bereinigender String
        max_length: Maximale Länge

    Returns:
        Bereinigter String
    """
    if not isinstance(value, str):
        value = str(value)

    # Entferne führende/nachfolgende Whitespaces
    value = value.strip()

    # Kürze auf maximale Länge
    if max_length and len(value) > max_length:
        value = value[:max_length]

    return value


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Führt mehrere Dictionaries zusammen.

    Args:
        *dicts: Dictionaries zum Zusammenführen

    Returns:
        Zusammengeführtes Dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def calculate_execution_time_ms(start_time: datetime, end_time: datetime) -> float:
    """Berechnet Ausführungszeit in Millisekunden.

    Args:
        start_time: Startzeit
        end_time: Endzeit

    Returns:
        Ausführungszeit in Millisekunden
    """
    return (end_time - start_time).total_seconds() * 1000


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 60.0
):
    """Decorator für Retry-Logic mit exponential backoff.

    Args:
        max_retries: Maximale Anzahl Wiederholungen
        base_delay: Basis-Verzögerung in Sekunden
        backoff_multiplier: Backoff-Multiplikator
        max_delay: Maximale Verzögerung
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        break

                    # Berechne Verzögerung
                    delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)

                    _logger.warning(
                        f"Versuch {attempt + 1}/{max_retries + 1} fehlgeschlagen: {e}. "
                        f"Wiederholung in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

            # Alle Versuche fehlgeschlagen
            raise last_exception

        return wrapper
    return decorator


def format_error_message(error: Exception, context: dict[str, Any] | None = None) -> str:
    """Formatiert Error-Message mit Kontext.

    Args:
        error: Exception
        context: Zusätzlicher Kontext

    Returns:
        Formatierte Error-Message
    """
    message = f"{type(error).__name__}: {error!s}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" (Kontext: {context_str})"

    return message


async def safe_async_call(
    func: Callable[..., Awaitable[T]],
    *args,
    timeout_seconds: float | None = None,
    default_value: T | None = None,
    **kwargs
) -> T | None:
    """Sichere Ausführung einer async Funktion mit Timeout.

    Args:
        func: Async Funktion
        *args: Positionelle Argumente
        timeout_seconds: Timeout in Sekunden
        default_value: Default-Wert bei Fehler
        **kwargs: Keyword-Argumente

    Returns:
        Funktionsergebnis oder Default-Wert
    """
    try:
        if timeout_seconds:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
        return await func(*args, **kwargs)
    except Exception as e:
        _logger.exception(f"Async-Call fehlgeschlagen: {format_error_message(e)}")
        return default_value


class AsyncContextManager:
    """Base-Klasse für Async Context Manager."""

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._async_enter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self._async_exit(exc_type, exc_val, exc_tb)

    async def _async_enter(self):
        """Override in Subclasses."""

    async def _async_exit(self, exc_type, exc_val, exc_tb):
        """Override in Subclasses."""


__all__ = [
    "AsyncContextManager",
    "calculate_execution_time_ms",
    "create_async_lock",
    "format_error_message",
    "generate_uuid",
    "get_current_utc_datetime",
    "get_module_logger",
    "merge_dicts",
    "retry_with_backoff",
    "safe_async_call",
    "safe_dict_get",
    "sanitize_string",
    "validate_required_fields",
]
