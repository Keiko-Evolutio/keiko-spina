"""Logger-Utilities für das App-Modul.

Gemeinsame Logger-Funktionen und -Konfigurationen um Code-Duplikation
zu vermeiden und konsistente Logging-Patterns zu gewährleisten.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from kei_logging import get_logger

from .constants import LOG_LEVEL_DEBUG, LOG_LEVEL_ERROR, LOG_LEVEL_INFO

# Logfire-Integration (optional)
try:
    from .logfire_logger import get_logfire_logger, initialize_logfire_logging
    LOGFIRE_LOGGER_AVAILABLE = True
except ImportError:
    get_logfire_logger = None
    initialize_logfire_logging = None
    LOGFIRE_LOGGER_AVAILABLE = False

# Type Variable für Decorator
F = TypeVar("F", bound=Callable[..., Any])


def get_module_logger(module_name: str, use_logfire: bool = True) -> Any:
    """Erstellt einen Logger für das angegebene Modul.

    Args:
        module_name: Name des Moduls (normalerweise __name__)
        use_logfire: Ob Logfire-Integration verwendet werden soll

    Returns:
        Logger-Instanz für das Modul (mit Logfire-Support falls verfügbar)
    """
    if use_logfire and LOGFIRE_LOGGER_AVAILABLE:
        try:
            return get_logfire_logger(module_name)
        except Exception:
            # Fallback auf Standard-Logger
            pass

    return get_logger(module_name)


def log_execution_time(logger: Any, level: str = LOG_LEVEL_DEBUG) -> Callable[[F], F]:
    """Decorator um Ausführungszeit von Funktionen zu loggen.

    Args:
        logger: Logger-Instanz
        level: Log-Level (default: DEBUG)

    Returns:
        Decorator-Funktion
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log(
                    getattr(logger, level.lower(), logger.debug),
                    f"{func.__name__} ausgeführt in {execution_time:.3f}s"
                )
                return result
            except Exception as exc:
                execution_time = time.time() - start_time
                logger.exception(
                    f"{func.__name__} fehlgeschlagen nach {execution_time:.3f}s: {exc}"
                )
                raise
        return cast("F", wrapper)
    return decorator


def log_async_execution_time(logger: Any, level: str = LOG_LEVEL_DEBUG) -> Callable[[F], F]:
    """Decorator um Ausführungszeit von async Funktionen zu loggen.

    Args:
        logger: Logger-Instanz
        level: Log-Level (default: DEBUG)

    Returns:
        Decorator-Funktion für async Funktionen
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                log_method = getattr(logger, level.lower(), logger.debug)
                log_method(f"{func.__name__} ausgeführt in {execution_time:.3f}s")
                return result
            except Exception as exc:
                execution_time = time.time() - start_time
                logger.exception(
                    f"{func.__name__} fehlgeschlagen nach {execution_time:.3f}s: {exc}"
                )
                raise
        return cast("F", wrapper)
    return decorator


def log_service_operation(
    logger: Any,
    service_name: str,
    operation: str,
    level: str = LOG_LEVEL_INFO
) -> Callable[[F], F]:
    """Decorator um Service-Operationen zu loggen.

    Args:
        logger: Logger-Instanz
        service_name: Name des Services
        operation: Name der Operation (z.B. "startup", "shutdown")
        level: Log-Level (default: INFO)

    Returns:
        Decorator-Funktion
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(f"{service_name} {operation} gestartet...")

            try:
                result = await func(*args, **kwargs)
                log_method(f"{service_name} {operation} erfolgreich abgeschlossen")
                return result
            except Exception as exc:
                logger.exception(f"{service_name} {operation} fehlgeschlagen: {exc}")
                raise
        return cast("F", wrapper)
    return decorator


def log_with_context(
    logger: Any,
    level: str,
    message: str,
    **context: Any
) -> None:
    """Loggt eine Nachricht mit zusätzlichem Kontext.

    Args:
        logger: Logger-Instanz
        level: Log-Level
        message: Log-Nachricht
        **context: Zusätzliche Kontext-Informationen
    """
    log_method = getattr(logger, level.lower(), logger.info)

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} [{context_str}]"
    else:
        full_message = message

    log_method(full_message)


def safe_log_exception(
    logger: Any,
    exc: Exception,
    message: str = "Unerwarteter Fehler aufgetreten",
    level: str = LOG_LEVEL_ERROR,
    **context: Any
) -> None:
    """Loggt eine Exception sicher ohne weitere Exceptions zu werfen.

    Args:
        logger: Logger-Instanz
        exc: Exception-Objekt
        message: Zusätzliche Nachricht
        level: Log-Level (default: ERROR)
        **context: Zusätzliche Kontext-Informationen
    """
    try:
        log_method = getattr(logger, level.lower(), logger.error)

        context_str = ""
        if context:
            context_str = f" [{', '.join(f'{k}={v}' for k, v in context.items())}]"

        full_message = f"{message}: {exc.__class__.__name__}: {exc}{context_str}"
        log_method(full_message)

    except Exception:
        # Fallback: Verwende print wenn Logger fehlschlägt
        pass


__all__ = [
    "get_module_logger",
    "log_async_execution_time",
    "log_execution_time",
    "log_service_operation",
    "log_with_context",
    "safe_log_exception",
]
