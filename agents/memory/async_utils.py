"""Async/Sync Utility-Funktionen für Memory-Module."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from kei_logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def run_async_safe(
    coro: Coroutine[Any, Any, T],
    *,
    raise_on_running_loop: bool = False,
    default_on_error: T | None = None
) -> T | None:
    """Führt Coroutine synchron aus.

    Args:
        coro: Coroutine zum Ausführen
        raise_on_running_loop: Ob Exception bei laufendem Loop geworfen werden soll
        default_on_error: Default-Wert bei Fehlern

    Returns:
        Ergebnis der Coroutine oder default_on_error bei Fehlern

    Raises:
        RuntimeError: Wenn raise_on_running_loop=True und Loop läuft
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if raise_on_running_loop:
                raise RuntimeError(
                    "Kann Coroutine nicht synchron im laufenden Event Loop ausführen"
                )
            logger.warning("Kann Coroutine nicht synchron im laufenden Event Loop ausführen")
            return default_on_error
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
    except Exception as e:
        logger.warning(f"Async-Ausführung fehlgeschlagen: {e}")
        return default_on_error


def fire_and_forget_async(
    coro: asyncio.Future[Any] | Coroutine[Any, Any, Any]
) -> None:
    """Startet Coroutine ohne auf Abschluss zu warten.

    Args:
        coro: Coroutine zum Ausführen
    """
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(coro)  # type: ignore[arg-type]
    except RuntimeError:
        asyncio.run(coro)  # pragma: no cover


MICROSECOND_OFFSET: float = 0.001


def validate_session_id(session_id: str, max_length: int = 255) -> None:
    """Validiert Session-ID Parameter.

    Args:
        session_id: Zu validierende Session-ID
        max_length: Maximale Länge der Session-ID

    Raises:
        ValueError: Bei ungültiger Session-ID
    """
    if not session_id or not session_id.strip():
        raise ValueError("Session-ID darf nicht leer sein")

    if len(session_id) > max_length:
        raise ValueError(f"Session-ID zu lang (max {max_length} Zeichen)")


def validate_thread_id(thread_id: str, max_length: int = 255) -> None:
    """Validiert Thread-ID Parameter.

    Args:
        thread_id: Zu validierende Thread-ID
        max_length: Maximale Länge der Thread-ID

    Raises:
        ValueError: Bei ungültiger Thread-ID
    """
    if not thread_id or not thread_id.strip():
        raise ValueError("Thread-ID darf nicht leer sein")

    if len(thread_id) > max_length:
        raise ValueError(f"Thread-ID zu lang (max {max_length} Zeichen)")


def create_unique_id(prefix: str, identifier: str, timestamp: float, suffix: str = "") -> str:
    """Erstellt eine eindeutige ID für Cosmos DB Items.

    Args:
        prefix: ID-Prefix
        identifier: Eindeutiger Identifier (z.B. session_id, thread_id)
        timestamp: Zeitstempel für Eindeutigkeit
        suffix: Optionaler Suffix

    Returns:
        Eindeutige ID
    """
    base_id = f"{identifier}-{timestamp}"
    if suffix:
        base_id += f"-{suffix}"

    return f"{prefix}{base_id}" if prefix else base_id


__all__ = [
    "MICROSECOND_OFFSET",
    "create_unique_id",
    "fire_and_forget_async",
    "run_async_safe",
    "validate_session_id",
    "validate_thread_id",
]
