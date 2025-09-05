# backend/services/clients/common/retry_utils.py
"""Retry-Utilities für Client Services.

Bietet wiederverwendbare Retry-Logik mit Exponential Backoff.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from kei_logging import get_logger

from .constants import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_INITIAL_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class RetryConfig:
    """Konfiguration für Retry-Verhalten."""

    max_retries: int = DEFAULT_MAX_RETRIES
    initial_delay: float = DEFAULT_INITIAL_DELAY
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    max_delay: float = DEFAULT_MAX_DELAY
    exceptions: tuple[type[Exception], ...] = (Exception,)


class RetryExhaustedException(Exception):
    """Exception wenn alle Retry-Versuche fehlgeschlagen sind."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry nach {attempts} Versuchen fehlgeschlagen: {last_error}")


async def retry_with_backoff[T](
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Führt eine async Funktion mit Exponential Backoff Retry aus.

    Args:
        func: Die auszuführende async Funktion
        *args: Positionale Argumente für die Funktion
        config: Retry-Konfiguration (optional)
        **kwargs: Keyword-Argumente für die Funktion

    Returns:
        Das Ergebnis der Funktion

    Raises:
        RetryExhaustedException: Wenn alle Versuche fehlgeschlagen sind
    """
    if config is None:
        config = RetryConfig()

    delay = config.initial_delay
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            logger.debug({
                "event": "retry_attempt",
                "attempt": attempt,
                "max_retries": config.max_retries,
                "function": func.__name__,
            })

            result = await func(*args, **kwargs)

            if attempt > 0:
                logger.info({
                    "event": "retry_success",
                    "attempt": attempt,
                    "function": func.__name__,
                })

            return result

        except config.exceptions as e:
            last_error = e

            logger.warning({
                "event": "retry_attempt_failed",
                "attempt": attempt,
                "max_retries": config.max_retries,
                "function": func.__name__,
                "error": str(e),
                "error_type": type(e).__name__,
            })

            if attempt >= config.max_retries:
                break

            # Exponential backoff mit Jitter
            actual_delay = min(delay, config.max_delay)
            logger.debug({
                "event": "retry_delay",
                "delay": actual_delay,
                "attempt": attempt,
            })

            await asyncio.sleep(actual_delay)
            delay *= config.backoff_multiplier

    # Alle Versuche fehlgeschlagen
    logger.error({
        "event": "retry_exhausted",
        "attempts": config.max_retries + 1,
        "function": func.__name__,
        "last_error": str(last_error),
    })

    if last_error is not None:
        raise RetryExhaustedException(config.max_retries + 1, last_error) from last_error
    raise RetryExhaustedException(config.max_retries + 1, Exception("Unbekannter Fehler"))


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator für automatische Retry-Logik.

    Args:
        config: Retry-Konfiguration (optional)

    Returns:
        Decorator-Funktion

    Example:
        @with_retry(RetryConfig(max_retries=3))
        async def my_function():
            # Funktion die fehlschlagen könnte
            pass
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


class RetryableClient:
    """Base-Klasse für Clients mit eingebauter Retry-Logik."""

    def __init__(self, retry_config: RetryConfig | None = None) -> None:
        """Initialisiert den Client mit Retry-Konfiguration.

        Args:
            retry_config: Retry-Konfiguration (optional)
        """
        self._retry_config = retry_config or RetryConfig()

    async def _execute_with_retry(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Führt eine Funktion mit der konfigurierten Retry-Logik aus.

        Args:
            func: Die auszuführende Funktion
            *args: Positionale Argumente
            **kwargs: Keyword-Argumente

        Returns:
            Das Ergebnis der Funktion
        """
        return await retry_with_backoff(
            func,
            *args,
            config=self._retry_config,
            **kwargs
        )


def create_content_safety_retry_config() -> RetryConfig:
    """Erstellt Retry-Konfiguration für Content Safety Client."""
    return RetryConfig(
        max_retries=2,
        initial_delay=0.5,
        exceptions=(Exception,)
    )


def create_image_generation_retry_config() -> RetryConfig:
    """Erstellt Retry-Konfiguration für Image Generation Service."""
    return RetryConfig(
        max_retries=2,
        initial_delay=0.5,
        exceptions=(Exception,)
    )


def create_http_retry_config() -> RetryConfig:
    """Erstellt Standard Retry-Konfiguration für HTTP-Requests."""
    return RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        exceptions=(Exception,)
    )
