# backend/quotas_limits/utils.py
"""Utility-Funktionen für das Quotas/Limits System.

Gemeinsame Funktionalitäten zur Reduzierung von Code-Duplikation
und Verbesserung der Wartbarkeit.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from kei_logging import get_logger

from .constants import (
    CACHE_TTL_SECONDS,
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
    PATTERN_UUID,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from decimal import Decimal

logger = get_logger(__name__)

T = TypeVar("T")


def generate_uuid() -> str:
    """Generiert eine neue UUID als String.

    Returns:
        str: UUID als String
    """
    return str(uuid.uuid4())


def validate_uuid(uuid_string: str) -> bool:
    """Validiert UUID-Format.

    Args:
        uuid_string: Zu validierende UUID

    Returns:
        bool: True wenn gültige UUID
    """
    import re
    return bool(re.match(PATTERN_UUID, uuid_string))


def get_current_timestamp() -> datetime:
    """Gibt aktuellen UTC-Timestamp zurück.

    Returns:
        datetime: Aktueller UTC-Timestamp
    """
    return datetime.now(UTC)


def calculate_time_difference_seconds(start: datetime, end: datetime | None = None) -> float:
    """Berechnet Zeitdifferenz in Sekunden.

    Args:
        start: Start-Zeitpunkt
        end: End-Zeitpunkt (default: jetzt)

    Returns:
        float: Zeitdifferenz in Sekunden
    """
    if end is None:
        end = get_current_timestamp()
    return (end - start).total_seconds()


def is_expired(timestamp: datetime, ttl_seconds: int) -> bool:
    """Prüft ob Timestamp abgelaufen ist.

    Args:
        timestamp: Zu prüfender Timestamp
        ttl_seconds: Time-to-Live in Sekunden

    Returns:
        bool: True wenn abgelaufen
    """
    expiry_time = timestamp + timedelta(seconds=ttl_seconds)
    return get_current_timestamp() > expiry_time


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Sichere Division mit Default-Wert bei Division durch Null.

    Args:
        numerator: Zähler
        denominator: Nenner
        default: Default-Wert bei Division durch Null

    Returns:
        float: Ergebnis der Division oder Default-Wert
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Begrenzt Wert auf Min/Max-Bereich.

    Args:
        value: Zu begrenzender Wert
        min_value: Minimum
        max_value: Maximum

    Returns:
        float: Begrenzter Wert
    """
    return max(min_value, min(value, max_value))


def format_decimal_currency(amount: Decimal, currency: str = "EUR") -> str:
    """Formatiert Decimal als Währung.

    Args:
        amount: Betrag
        currency: Währung

    Returns:
        str: Formatierter Währungsbetrag
    """
    return f"{amount:.2f} {currency}"


def calculate_percentage(part: float, total: float) -> float:
    """Berechnet Prozentsatz.

    Args:
        part: Teil-Wert
        total: Gesamt-Wert

    Returns:
        float: Prozentsatz (0-100)
    """
    return safe_divide(part * 100, total, 0.0)


class CacheEntry:
    """Cache-Eintrag mit TTL."""

    def __init__(self, value: Any, ttl_seconds: int = CACHE_TTL_SECONDS):
        """Initialisiert Cache-Eintrag.

        Args:
            value: Zu cachender Wert
            ttl_seconds: Time-to-Live in Sekunden
        """
        self.value = value
        self.timestamp = get_current_timestamp()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Prüft ob Cache-Eintrag abgelaufen ist.

        Returns:
            bool: True wenn abgelaufen
        """
        return is_expired(self.timestamp, self.ttl_seconds)


class SimpleCache:
    """Einfacher In-Memory-Cache mit TTL."""

    def __init__(self, max_size: int = 1000):
        """Initialisiert Cache.

        Args:
            max_size: Maximale Anzahl Cache-Einträge
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Holt Wert aus Cache.

        Args:
            key: Cache-Schlüssel

        Returns:
            Optional[Any]: Cached Wert oder None
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.value
            if entry:
                # Entferne abgelaufenen Eintrag
                del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        """Setzt Wert in Cache.

        Args:
            key: Cache-Schlüssel
            value: Zu cachender Wert
            ttl_seconds: Time-to-Live in Sekunden
        """
        async with self._lock:
            # Prüfe Cache-Größe
            if len(self._cache) >= self._max_size:
                await self._cleanup_expired()

                # Falls immer noch zu groß, entferne älteste Einträge
                if len(self._cache) >= self._max_size:
                    oldest_keys = sorted(
                        self._cache.keys(),
                        key=lambda k: self._cache[k].timestamp
                    )[:len(self._cache) - self._max_size + 1]

                    for old_key in oldest_keys:
                        del self._cache[old_key]

            self._cache[key] = CacheEntry(value, ttl_seconds)

    async def delete(self, key: str) -> bool:
        """Löscht Eintrag aus Cache.

        Args:
            key: Cache-Schlüssel

        Returns:
            bool: True wenn Eintrag existierte
        """
        async with self._lock:
            return self._cache.pop(key, None) is not None

    async def clear(self) -> None:
        """Leert Cache komplett."""
        async with self._lock:
            self._cache.clear()

    async def _cleanup_expired(self) -> None:
        """Entfernt abgelaufene Cache-Einträge."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück.

        Returns:
            Dict[str, Any]: Cache-Statistiken
        """
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate": 0.0  # Würde Tracking erfordern
        }


async def retry_with_backoff[T](
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    **kwargs
) -> T:
    """Führt Funktion mit Retry und Exponential Backoff aus.

    Args:
        func: Auszuführende Funktion
        *args: Funktions-Argumente
        max_retries: Maximale Anzahl Wiederholungen
        delay_seconds: Initiale Verzögerung
        backoff_multiplier: Backoff-Multiplikator
        **kwargs: Funktions-Keyword-Argumente

    Returns:
        T: Funktions-Ergebnis

    Raises:
        Exception: Letzte Exception nach allen Versuchen
    """
    last_exception = None
    current_delay = delay_seconds

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {current_delay}s: {e}")
                await asyncio.sleep(current_delay)
                current_delay *= backoff_multiplier
            else:
                logger.exception(f"All {max_retries + 1} attempts failed")

    raise last_exception


def measure_execution_time(func: Callable) -> Callable:
    """Decorator zur Messung der Ausführungszeit.

    Args:
        func: Zu messende Funktion

    Returns:
        Callable: Dekorierte Funktion
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.exception(f"{func.__name__} failed after {execution_time:.2f}ms: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.exception(f"{func.__name__} failed after {execution_time:.2f}ms: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def create_error_context(
    operation: str,
    details: dict[str, Any] | None = None,
    exception: Exception | None = None
) -> dict[str, Any]:
    """Erstellt standardisierten Error-Context.

    Args:
        operation: Name der Operation
        details: Zusätzliche Details
        exception: Aufgetretene Exception

    Returns:
        Dict[str, Any]: Error-Context
    """
    context = {
        "operation": operation,
        "timestamp": get_current_timestamp().isoformat(),
        "details": details or {}
    }

    if exception:
        context["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception)
        }

    return context
