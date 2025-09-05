"""Standardisierte Error Handler für das App-Modul.

Gemeinsame Error-Handling-Patterns um Code-Duplikation zu vermeiden
und konsistente Fehlerbehandlung zu gewährleisten.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

from .constants import (
    LOG_LEVEL_WARNING,
)
from .logger_utils import get_module_logger, safe_log_exception

# Type Variable für Decorator
F = TypeVar("F", bound=Callable[..., Any])

logger = get_module_logger(__name__)


class ServiceError(Exception):
    """Basis-Exception für Service-Fehler."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        error_code: str | None = None,
        original_exception: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.service_name = service_name
        self.error_code = error_code
        self.original_exception = original_exception


class ServiceUnavailableError(ServiceError):
    """Exception für nicht verfügbare Services."""

    def __init__(
        self,
        service_name: str,
        message: str | None = None,
        original_exception: Exception | None = None
    ) -> None:
        message = message or f"{service_name} ist nicht verfügbar"
        super().__init__(
            message=message,
            service_name=service_name,
            error_code="SERVICE_UNAVAILABLE",
            original_exception=original_exception
        )


class ConfigurationError(ServiceError):
    """Exception für Konfigurationsfehler."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        original_exception: Exception | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            original_exception=original_exception
        )
        self.config_key = config_key


def handle_service_errors(
    service_name: str,
    fallback_value: Any = None,
    raise_on_error: bool = False,
    log_level: str = LOG_LEVEL_WARNING
) -> Callable[[F], F]:
    """Decorator für standardisierte Service-Error-Behandlung.

    Args:
        service_name: Name des Services für Logging
        fallback_value: Wert der bei Fehlern zurückgegeben wird
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        log_level: Log-Level für Fehler

    Returns:
        Decorator-Funktion
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                safe_log_exception(
                    logger,
                    exc,
                    f"{service_name} Operation fehlgeschlagen",
                    level=log_level,
                    service=service_name,
                    function=func.__name__
                )

                if raise_on_error:
                    raise ServiceError(
                        message=f"{service_name} Operation fehlgeschlagen: {exc}",
                        service_name=service_name,
                        original_exception=exc
                    )

                return fallback_value
        return cast("F", wrapper)
    return decorator


def handle_async_service_errors(
    service_name: str,
    fallback_value: Any = None,
    raise_on_error: bool = False,
    log_level: str = LOG_LEVEL_WARNING
) -> Callable[[F], F]:
    """Decorator für standardisierte async Service-Error-Behandlung.

    Args:
        service_name: Name des Services für Logging
        fallback_value: Wert der bei Fehlern zurückgegeben wird
        raise_on_error: Ob Exceptions weitergeworfen werden sollen
        log_level: Log-Level für Fehler

    Returns:
        Decorator-Funktion für async Funktionen
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                safe_log_exception(
                    logger,
                    exc,
                    f"{service_name} async Operation fehlgeschlagen",
                    level=log_level,
                    service=service_name,
                    function=func.__name__
                )

                if raise_on_error:
                    raise ServiceError(
                        message=f"{service_name} async Operation fehlgeschlagen: {exc}",
                        service_name=service_name,
                        original_exception=exc
                    )

                return fallback_value
        return cast("F", wrapper)
    return decorator


def safe_service_call(
    service_call: Callable[[], Any],
    service_name: str,
    fallback_value: Any = None,
    log_errors: bool = True
) -> Any:
    """Führt einen Service-Aufruf sicher aus.

    Args:
        service_call: Funktion die aufgerufen werden soll
        service_name: Name des Services für Logging
        fallback_value: Wert der bei Fehlern zurückgegeben wird
        log_errors: Ob Fehler geloggt werden sollen

    Returns:
        Ergebnis des Service-Aufrufs oder fallback_value
    """
    try:
        return service_call()
    except Exception as exc:
        if log_errors:
            safe_log_exception(
                logger,
                exc,
                f"Service-Aufruf für {service_name} fehlgeschlagen",
                service=service_name
            )
        return fallback_value


async def safe_async_service_call(
    service_call: Callable[[], Any],
    service_name: str,
    fallback_value: Any = None,
    log_errors: bool = True
) -> Any:
    """Führt einen async Service-Aufruf sicher aus.

    Args:
        service_call: Async Funktion die aufgerufen werden soll
        service_name: Name des Services für Logging
        fallback_value: Wert der bei Fehlern zurückgegeben wird
        log_errors: Ob Fehler geloggt werden sollen

    Returns:
        Ergebnis des Service-Aufrufs oder fallback_value
    """
    try:
        return await service_call()
    except Exception as exc:
        if log_errors:
            safe_log_exception(
                logger,
                exc,
                f"Async Service-Aufruf für {service_name} fehlgeschlagen",
                service=service_name
            )
        return fallback_value


def create_error_response(
    error: Exception | str,
    error_code: str | None = None,
    status_code: int = 500
) -> dict[str, Any]:
    """Erstellt eine standardisierte Error-Response.

    Args:
        error: Exception oder Error-Message
        error_code: Optionaler Error-Code
        status_code: HTTP Status Code

    Returns:
        Dictionary mit Error-Informationen
    """
    if isinstance(error, Exception):
        message = str(error)
        error_type = error.__class__.__name__
    else:
        message = error
        error_type = "Error"

    response = {
        "error": True,
        "message": message,
        "error_type": error_type,
        "status_code": status_code
    }

    if error_code:
        response["error_code"] = error_code

    return response


__all__ = [
    "ConfigurationError",
    "ServiceError",
    "ServiceUnavailableError",
    "create_error_response",
    "handle_async_service_errors",
    "handle_service_errors",
    "safe_async_service_call",
    "safe_service_call",
]
