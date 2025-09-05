"""Test-Helper-Funktionen für das KEI-Webhook System."""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator für async Test-Funktionen."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: str = "Condition not met within timeout"
) -> None:
    """Wartet bis eine Bedingung erfüllt ist."""
    start_time = asyncio.get_event_loop().time()

    while True:
        if condition():
            return

        if asyncio.get_event_loop().time() - start_time > timeout:
            raise WebhookTimeoutError(error_message)

        await asyncio.sleep(interval)


@contextmanager
def capture_logs(logger_name: str, level: int = logging.DEBUG):
    """Context-Manager zum Erfassen von Log-Nachrichten."""
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    handler = logging.StreamHandler()
    handler.setLevel(level)

    captured_logs: list[logging.LogRecord] = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            captured_logs.append(record)

    capturing_handler = CapturingHandler()
    capturing_handler.setLevel(level)

    logger.addHandler(capturing_handler)
    logger.setLevel(level)

    try:
        yield captured_logs
    finally:
        logger.removeHandler(capturing_handler)
        logger.setLevel(original_level)


class AsyncContextManager:
    """Helper für async Context-Manager in Tests."""

    def __init__(self, async_func: Callable[[], Any]):
        self.async_func = async_func
        self.result = None

    async def __aenter__(self):
        self.result = await self.async_func()
        return self.result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.result, "close"):
            await self.result.close()


def mock_redis_client(mock_client):
    """Decorator zum Mocken des Redis-Clients."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with patch("services.webhooks.utils.redis_manager.get_cache_client",
                      return_value=mock_client):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def mock_settings(**settings_overrides):
    """Decorator zum Mocken von Settings."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            patches = []
            for key, value in settings_overrides.items():
                patches.append(patch(f"config.settings.settings.{key}", value))

            for p in patches:
                p.start()

            try:
                return func(*args, **kwargs)
            finally:
                for p in patches:
                    p.stop()
        return wrapper
    return decorator


class WebhookTimeoutError(Exception):
    """Timeout-Fehler für Tests."""


__all__ = [
    "AsyncContextManager",
    "WebhookTimeoutError",
    "async_test",
    "capture_logs",
    "mock_redis_client",
    "mock_settings",
    "wait_for_condition",
]
