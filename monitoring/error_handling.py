"""Gemeinsame Error-Handling-Patterns für das Monitoring-System.

Konsolidiert wiederkehrende Try-Catch-Patterns und Error-Handling-Logic
aus dem gesamten Monitoring-Modul. Stellt konsistente Fehlerbehandlung
und Logging sicher.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from kei_logging import get_logger

logger = get_logger(__name__)

# Type Variables für Generic Functions
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# ============================================================================
# EXCEPTION KLASSEN
# ============================================================================

class MonitoringError(Exception):
    """Basis-Exception für Monitoring-Fehler."""

class MetricsCollectionError(MonitoringError):
    """Fehler bei der Metriken-Sammlung."""

class HealthCheckError(MonitoringError):
    """Fehler bei Health-Checks."""

class TracingError(MonitoringError):
    """Fehler beim Tracing."""

class ConfigurationError(MonitoringError):
    """Fehler in der Konfiguration."""

# ============================================================================
# DECORATOR FÜR SAFE OPERATIONS
# ============================================================================

def safe_operation(
    default_return: Any = None,
    log_errors: bool = True,
    error_message: str | None = None,
    raise_on_error: bool = False,
    allowed_exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """Decorator für sichere Operationen mit einheitlichem Error-Handling.

    Args:
        default_return: Rückgabewert bei Fehlern
        log_errors: Ob Fehler geloggt werden sollen
        error_message: Custom Error-Message
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        allowed_exceptions: Tuple von Exception-Typen die gefangen werden
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

def safe_async_operation(
    default_return: Any = None,
    log_errors: bool = True,
    error_message: str | None = None,
    raise_on_error: bool = False,
    allowed_exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """Async-Version des safe_operation Decorators."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
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

# ============================================================================
# CONTEXT MANAGER FÜR SAFE OPERATIONS
# ============================================================================

@contextmanager
def safe_context(
    operation_name: str,
    log_errors: bool = True,
    raise_on_error: bool = False,
    default_return: Any = None
) -> Iterator[dict[str, Any]]:
    """Context Manager für sichere Operationen.

    Args:
        operation_name: Name der Operation für Logging
        log_errors: Ob Fehler geloggt werden sollen
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        default_return: Default-Rückgabewert bei Fehlern
    """
    context = {"success": True, "error": None, "result": None}
    start_time = time.time()

    try:
        yield context
    except Exception as e:
        context["success"] = False
        context["error"] = e
        context["result"] = default_return

        if log_errors:
            duration = time.time() - start_time
            logger.warning(
                f"{operation_name} fehlgeschlagen nach {duration:.3f}s: {e}"
            )

        if raise_on_error:
            raise
    else:
        duration = time.time() - start_time
        logger.debug(f"{operation_name} erfolgreich nach {duration:.3f}s")

@asynccontextmanager
async def safe_async_context(
    operation_name: str,
    log_errors: bool = True,
    raise_on_error: bool = False,
    default_return: Any = None
) -> AsyncIterator[dict[str, Any]]:
    """Async-Version des safe_context Context Managers."""
    context = {"success": True, "error": None, "result": None}
    start_time = time.time()

    try:
        yield context
    except Exception as e:
        context["success"] = False
        context["error"] = e
        context["result"] = default_return

        if log_errors:
            duration = time.time() - start_time
            logger.warning(
                f"{operation_name} fehlgeschlagen nach {duration:.3f}s: {e}"
            )

        if raise_on_error:
            raise
    else:
        duration = time.time() - start_time
        logger.debug(f"{operation_name} erfolgreich nach {duration:.3f}s")

# ============================================================================
# TIMEOUT HANDLING
# ============================================================================

def with_timeout(
    timeout_seconds: float,
    operation_name: str = "Operation",
    default_return: Any = None
) -> Callable[[F], F]:
    """Decorator für Timeout-Handling bei synchronen Operationen."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Für synchrone Funktionen verwenden wir einen einfachen Timeout-Check
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if duration > timeout_seconds:
                    logger.warning(
                        f"{operation_name} dauerte {duration:.3f}s "
                        f"(Timeout: {timeout_seconds}s)"
                    )
                return result
            except Exception as e:
                logger.warning(f"{operation_name} fehlgeschlagen: {e}")
                return default_return
        return wrapper
    return decorator

def with_async_timeout(
    timeout_seconds: float,
    operation_name: str = "Operation",
    default_return: Any = None
) -> Callable[[F], F]:
    """Decorator für Timeout-Handling bei asynchronen Operationen."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except TimeoutError:
                logger.warning(
                    f"{operation_name} Timeout nach {timeout_seconds}s"
                )
                return default_return
            except Exception as e:
                logger.warning(f"{operation_name} fehlgeschlagen: {e}")
                return default_return
        return wrapper
    return decorator

# ============================================================================
# RETRY LOGIC
# ============================================================================

def with_retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    operation_name: str = "Operation"
) -> Callable[[F], F]:
    """Decorator für Retry-Logic bei fehlgeschlagenen Operationen."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = delay_seconds * (backoff_factor ** attempt)
                        logger.debug(
                            f"{operation_name} Versuch {attempt + 1} fehlgeschlagen, "
                            f"Retry in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            f"{operation_name} fehlgeschlagen nach {max_attempts} Versuchen: {e}"
                        )

            # Wenn alle Versuche fehlgeschlagen sind
            if last_exception:
                raise last_exception
            return None

        return wrapper
    return decorator

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_and_ignore_error(
    operation: Callable[[], T],
    operation_name: str,
    default_return: T = None
) -> T:
    """Führt Operation aus und ignoriert Fehler mit Logging."""
    try:
        return operation()
    except Exception as e:
        logger.warning(f"{operation_name} fehlgeschlagen: {e}")
        return default_return

async def log_and_ignore_async_error(
    operation: Callable[[], T],
    operation_name: str,
    default_return: T = None
) -> T:
    """Async-Version von log_and_ignore_error."""
    try:
        if asyncio.iscoroutinefunction(operation):
            return await operation()
        return operation()
    except (ValueError, TypeError) as e:
        logger.warning(f"{operation_name} fehlgeschlagen - Validierungsfehler: {e}")
        return default_return
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"{operation_name} fehlgeschlagen - Verbindungsproblem: {e}")
        return default_return
    except Exception as e:
        logger.warning(f"{operation_name} fehlgeschlagen - Unerwarteter Fehler: {e}")
        return default_return

def safe_dict_get(
    dictionary: dict[str, Any],
    key: str,
    default: Any = None,
    log_missing: bool = False
) -> Any:
    """Sicherer Dictionary-Zugriff mit optionalem Logging."""
    try:
        return dictionary[key]
    except KeyError:
        if log_missing:
            logger.debug(f"Key '{key}' nicht in Dictionary gefunden")
        return default

def validate_config(config: Any, config_name: str) -> bool:
    """Validiert Konfiguration mit einheitlichem Error-Handling."""
    try:
        if hasattr(config, "validate"):
            config.validate()
            return True
        logger.warning(f"{config_name} hat keine validate() Methode")
        return True
    except Exception as e:
        logger.exception(f"{config_name} Validierung fehlgeschlagen: {e}")
        return False

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ConfigurationError",
    "HealthCheckError",
    "MetricsCollectionError",
    # Exception Classes
    "MonitoringError",
    "TracingError",
    "log_and_ignore_async_error",
    # Utility Functions
    "log_and_ignore_error",
    "safe_async_context",
    "safe_async_operation",
    # Context Managers
    "safe_context",
    "safe_dict_get",
    # Decorators
    "safe_operation",
    "validate_config",
    "with_async_timeout",
    "with_retry",
    "with_timeout",
]
