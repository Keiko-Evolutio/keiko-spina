# backend/agents/core/utils.py
"""Gemeinsame Utility-Funktionen für das Core-Modul

Konsolidiert wiederkehrende Patterns und eliminiert Code-Duplikate:
- Validation-Utils für Dataclasses
- Common Error-Handling Patterns
- Logging-Utilities
- Async-Utilities
"""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

try:
    from kei_logging import get_logger
except ImportError:
    # Fallback für lokale Entwicklung
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Konstanten
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONCURRENT_TASKS = 10
DEFAULT_TASK_TIMEOUT_SECONDS = 30.0
DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_CLEANUP_INTERVAL_SECONDS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 10.0
DEFAULT_RETRY_BACKOFF_FACTOR = 2.0
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 60.0
DEFAULT_LOG_LEVEL = "INFO"

logger = get_logger(__name__)


class ValidationError(ValueError):
    """Spezifische Exception für Validierungsfehler."""


def validate_positive_number(value: float | int, field_name: str) -> None:
    """Validiert dass ein Wert positiv ist.

    Args:
        value: Zu validierender Wert
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValidationError: Wenn Wert nicht positiv ist
    """
    if value <= 0:
        raise ValidationError(f"{field_name} muss positiv sein, erhalten: {value}")


def validate_non_empty_string(value: str, field_name: str) -> None:
    """Validiert dass ein String nicht leer ist.

    Args:
        value: Zu validierender String
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValidationError: Wenn String leer ist
    """
    if not value or not value.strip():
        raise ValidationError(f"{field_name} darf nicht leer sein")


def validate_required_field(value: Any, field_name: str) -> None:
    """Validiert dass ein Pflichtfeld gesetzt ist.

    Args:
        value: Zu validierender Wert
        field_name: Name des Feldes für Fehlermeldung

    Raises:
        ValidationError: Wenn Wert None oder leer ist
    """
    if value is None:
        raise ValidationError(f"{field_name} ist erforderlich")

    if isinstance(value, str) and not value.strip():
        raise ValidationError(f"{field_name} darf nicht leer sein")


def generate_task_id() -> str:
    """Generiert eine eindeutige Task-ID.

    Returns:
        UUID-String als Task-ID
    """
    return str(uuid.uuid4())


def get_module_logger(module_name: str) -> Any:
    """Erstellt Logger für ein Modul mit konsistenter Konfiguration.

    Args:
        module_name: Name des Moduls (__name__)

    Returns:
        Konfigurierter Logger
    """
    return get_logger(module_name)


@asynccontextmanager
async def async_timeout_context(timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
    """Async Context Manager für Timeout-Handling.

    Args:
        timeout_seconds: Timeout in Sekunden

    Yields:
        Context für zeitbegrenzte Operationen

    Raises:
        asyncio.TimeoutError: Bei Timeout
    """
    start_time = time.time()
    try:
        async with asyncio.timeout(timeout_seconds):
            yield
    except TimeoutError:
        duration = time.time() - start_time
        logger.warning(f"Operation timeout nach {duration:.2f}s (Limit: {timeout_seconds}s)")
        raise


@asynccontextmanager
async def async_error_context(
    operation_name: str,
    log_errors: bool = True,
    raise_on_error: bool = True,
    default_return: Any = None
):
    """Async Context Manager für Error-Handling mit Logging.

    Args:
        operation_name: Name der Operation für Logging
        log_errors: Ob Fehler geloggt werden sollen
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        default_return: Rückgabewert bei Fehlern

    Yields:
        Context-Dictionary mit Ergebnis-Informationen
    """
    start_time = time.time()
    context = {
        "operation": operation_name,
        "success": True,
        "error": None,
        "result": None,
        "duration": 0.0
    }

    try:
        yield context
        context["duration"] = time.time() - start_time
        if log_errors:
            logger.debug(f"{operation_name} erfolgreich nach {context['duration']:.3f}s")
    except Exception as e:
        context["success"] = False
        context["error"] = e
        context["result"] = default_return
        context["duration"] = time.time() - start_time

        if log_errors:
            logger.warning(
                f"{operation_name} fehlgeschlagen nach {context['duration']:.3f}s: {e}"
            )

        if raise_on_error:
            raise


def with_error_handling(
    default_return: Any = None,
    log_errors: bool = True,
    error_message: str | None = None,
    raise_on_error: bool = False,
    allowed_exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """Decorator für einheitliches Error-Handling.

    Args:
        default_return: Rückgabewert bei Fehlern
        log_errors: Ob Fehler geloggt werden sollen
        error_message: Custom Error-Message
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        allowed_exceptions: Tuple von Exception-Typen die gefangen werden

    Returns:
        Decorator-Funktion
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except allowed_exceptions as e:
                if log_errors:
                    msg = error_message or f"{func.__name__} fehlgeschlagen"
                    logger.warning(f"{msg}: {e}")

                if raise_on_error:
                    raise

                return default_return
        return wrapper
    return decorator


def with_async_error_handling(
    default_return: Any = None,
    log_errors: bool = True,
    error_message: str | None = None,
    raise_on_error: bool = False,
    allowed_exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Async Decorator für einheitliches Error-Handling.

    Args:
        default_return: Rückgabewert bei Fehlern
        log_errors: Ob Fehler geloggt werden sollen
        error_message: Custom Error-Message
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        allowed_exceptions: Tuple von Exception-Typen die gefangen werden

    Returns:
        Async Decorator-Funktion
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except allowed_exceptions as e:
                if log_errors:
                    msg = error_message or f"{func.__name__} fehlgeschlagen"
                    logger.warning(f"{msg}: {e}")

                if raise_on_error:
                    raise

                return default_return
        return wrapper
    return decorator


class MetricsCollector:
    """Gemeinsame Basis-Klasse für Metriken-Sammlung."""

    def __init__(self):
        """Initialisiert Metriken-Collector."""
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_execution_time = 0.0
        self.start_time = time.time()

    def record_operation(self, success: bool, execution_time: float) -> None:
        """Zeichnet Operation-Ergebnis auf.

        Args:
            success: Ob Operation erfolgreich war
            execution_time: Ausführungszeit in Sekunden
        """
        self.total_operations += 1
        self.total_execution_time += execution_time

        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

    @property
    def success_rate(self) -> float:
        """Berechnet Success-Rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations

    @property
    def average_execution_time(self) -> float:
        """Berechnet durchschnittliche Ausführungszeit."""
        if self.total_operations == 0:
            return 0.0
        return self.total_execution_time / self.total_operations

    @property
    def uptime_seconds(self) -> float:
        """Berechnet Uptime in Sekunden."""
        return time.time() - self.start_time

    def get_metrics_dict(self) -> dict[str, Any]:
        """Gibt Metriken als Dictionary zurück.

        Returns:
            Dictionary mit allen Metriken
        """
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.average_execution_time,
            "uptime_seconds": self.uptime_seconds
        }
