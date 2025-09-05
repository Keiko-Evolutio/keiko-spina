"""Zentraler Redis-Client-Manager für das KEI-Webhook System.

Stellt eine einheitliche Abstraktion für Redis-Operationen bereit mit
automatischem Fallback auf NoOpCache, Retry-Logic und Connection-Pooling.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

from ..constants import (
    ALERT_RETRY_BACKOFF_SECONDS,
    RETRY_RATE_LIMIT_MAX_ATTEMPTS,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

T = TypeVar("T")


class RedisManager:
    """Zentraler Manager für Redis-Operationen mit Retry-Logic und Fallbacks."""

    def __init__(self) -> None:
        self._client_cache: Any | None = None
        self._cache_valid = False

    async def get_client(self) -> Any:
        """Holt Redis-Client mit Caching und Fallback-Behandlung.

        Returns:
            Redis-Client oder NoOpCache-Instanz
        """
        if not self._cache_valid or self._client_cache is None:
            try:
                self._client_cache = await get_cache_client()
                self._cache_valid = True
            except (OSError, RuntimeError, ConnectionError) as exc:
                logger.debug("Redis-Verbindung fehlgeschlagen, verwende NoOpCache: %s", exc)
                self._client_cache = NoOpCache()
                self._cache_valid = True

        return self._client_cache

    def invalidate_cache(self) -> None:
        """Invalidiert den Client-Cache (z.B. nach Connection-Fehlern)."""
        self._cache_valid = False
        self._client_cache = None

    async def is_available(self) -> bool:
        """Prüft, ob Redis verfügbar ist.

        Returns:
            True wenn Redis verfügbar, False bei NoOpCache
        """
        client = await self.get_client()
        return not isinstance(client, NoOpCache)

    async def execute_with_retry(
        self,
        operation: Callable[[Any], Awaitable[T]],
        *,
        max_attempts: int = RETRY_RATE_LIMIT_MAX_ATTEMPTS,
        backoff_seconds: float = ALERT_RETRY_BACKOFF_SECONDS,
        fallback_value: T | None = None,
    ) -> T | None:
        """Führt Redis-Operation mit Retry-Logic aus.

        Args:
            operation: Async-Funktion die Redis-Client als Parameter erwartet
            max_attempts: Maximale Anzahl Versuche
            backoff_seconds: Backoff-Zeit zwischen Versuchen
            fallback_value: Rückgabewert bei dauerhaftem Fehler

        Returns:
            Ergebnis der Operation oder fallback_value
        """
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                client = await self.get_client()
                if isinstance(client, NoOpCache):
                    logger.debug("Redis nicht verfügbar, verwende Fallback")
                    return fallback_value

                return await operation(client)

            except (TimeoutError, OSError, ConnectionError) as exc:
                last_exception = exc
                logger.debug(
                    "Redis-Operation fehlgeschlagen (Versuch %d/%d): %s",
                    attempt, max_attempts, exc
                )

                # Cache invalidieren für nächsten Versuch
                self.invalidate_cache()

                if attempt < max_attempts:
                    await asyncio.sleep(backoff_seconds * attempt)

            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Unerwartete Fehler nicht wiederholen
                logger.warning("Unerwarteter Redis-Fehler: %s", exc)
                return fallback_value

        logger.warning(
            "Redis-Operation nach %d Versuchen fehlgeschlagen: %s",
            max_attempts, last_exception
        )
        return fallback_value

    @asynccontextmanager
    async def transaction(self):
        """Context-Manager für Redis-Transaktionen.

        Yields:
            Redis-Client für Transaktions-Operationen
        """
        client = await self.get_client()
        if isinstance(client, NoOpCache):
            # NoOpCache unterstützt keine Transaktionen
            yield client
            return

        try:
            # Redis-Pipeline für atomare Operationen
            if hasattr(client, "pipeline"):
                async with client.pipeline() as pipe:
                    yield pipe
            else:
                yield client
        except Exception as exc:
            logger.warning("Redis-Transaktion fehlgeschlagen: %s", exc)
            self.invalidate_cache()
            raise

    async def safe_set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Sichere Redis SET-Operation mit Fallback.

        Args:
            key: Redis-Key
            value: Wert
            ex: Expiration in Sekunden
            nx: Nur setzen wenn Key nicht existiert

        Returns:
            True bei Erfolg, False bei Fehler oder NoOpCache
        """
        async def _set_operation(client: Any) -> bool:
            if isinstance(client, NoOpCache):
                return False

            kwargs = {}
            if ex is not None:
                kwargs["ex"] = ex
            if nx:
                kwargs["nx"] = nx

            result = await client.set(key, value, **kwargs)
            return bool(result)

        result = await self.execute_with_retry(_set_operation, fallback_value=False)
        return result or False

    async def safe_get(self, key: str) -> str | None:
        """Sichere Redis GET-Operation mit Fallback.

        Args:
            key: Redis-Key

        Returns:
            Wert oder None bei Fehler/NoOpCache
        """
        async def _get_operation(client: Any) -> str | None:
            if isinstance(client, NoOpCache):
                return None
            return await client.get(key)

        return await self.execute_with_retry(_get_operation, fallback_value=None)

    async def safe_delete(self, *keys: str) -> int:
        """Sichere Redis DELETE-Operation mit Fallback.

        Args:
            *keys: Redis-Keys zum Löschen

        Returns:
            Anzahl gelöschter Keys
        """
        async def _delete_operation(client: Any) -> int:
            if isinstance(client, NoOpCache):
                return 0
            return await client.delete(*keys)

        result = await self.execute_with_retry(_delete_operation, fallback_value=0)
        return result or 0

    async def safe_lpush(self, key: str, *values: str) -> int:
        """Sichere Redis LPUSH-Operation mit Fallback.

        Args:
            key: Redis-Key
            *values: Werte zum Hinzufügen

        Returns:
            Neue Länge der Liste
        """
        async def _lpush_operation(client: Any) -> int:
            if isinstance(client, NoOpCache):
                return 0
            return await client.lpush(key, *values)

        result = await self.execute_with_retry(_lpush_operation, fallback_value=0)
        return result or 0

    async def safe_rpop(self, key: str) -> str | None:
        """Sichere Redis RPOP-Operation mit Fallback.

        Args:
            key: Redis-Key

        Returns:
            Gepoppter Wert oder None
        """
        async def _rpop_operation(client: Any) -> str | None:
            if isinstance(client, NoOpCache):
                return None
            return await client.rpop(key)

        return await self.execute_with_retry(_rpop_operation, fallback_value=None)

    async def safe_llen(self, key: str) -> int:
        """Sichere Redis LLEN-Operation mit Fallback.

        Args:
            key: Redis-Key

        Returns:
            Länge der Liste
        """
        async def _llen_operation(client: Any) -> int:
            if isinstance(client, NoOpCache):
                return 0
            return await client.llen(key)

        result = await self.execute_with_retry(_llen_operation, fallback_value=0)
        return result or 0

    async def safe_hset(self, key: str, field: str, value: str) -> bool:
        """Sichere Redis HSET-Operation mit Fallback.

        Args:
            key: Redis-Key
            field: Hash-Field
            value: Wert

        Returns:
            True bei Erfolg
        """
        async def _hset_operation(client: Any) -> bool:
            if isinstance(client, NoOpCache):
                return False
            result = await client.hset(key, field, value)
            return bool(result)

        result = await self.execute_with_retry(_hset_operation, fallback_value=False)
        return result or False

    async def safe_hgetall(self, key: str) -> dict[str, str]:
        """Sichere Redis HGETALL-Operation mit Fallback.

        Args:
            key: Redis-Key

        Returns:
            Hash-Dictionary oder leeres Dict
        """
        async def _hgetall_operation(client: Any) -> dict[str, str]:
            if isinstance(client, NoOpCache):
                return {}
            result = await client.hgetall(key)
            return dict(result) if result else {}

        result = await self.execute_with_retry(_hgetall_operation, fallback_value={})
        return result or {}

    async def safe_hget(self, key: str, field: str) -> str | None:
        """Sichere Redis HGET-Operation mit Fallback.

        Args:
            key: Redis-Key
            field: Hash-Field

        Returns:
            Wert oder None
        """
        async def _hget_operation(client: Any) -> str | None:
            if isinstance(client, NoOpCache):
                return None
            return await client.hget(key, field)

        return await self.execute_with_retry(_hget_operation, fallback_value=None)

    async def safe_hdel(self, key: str, *fields: str) -> int:
        """Sichere Redis HDEL-Operation mit Fallback.

        Args:
            key: Redis-Key
            *fields: Hash-Fields zum Löschen

        Returns:
            Anzahl gelöschter Fields
        """
        async def _hdel_operation(client: Any) -> int:
            if isinstance(client, NoOpCache):
                return 0
            return await client.hdel(key, *fields)

        result = await self.execute_with_retry(_hdel_operation, fallback_value=0)
        return result or 0


# Globale Singleton-Instanz
_redis_manager: RedisManager | None = None


def get_redis_manager() -> RedisManager:
    """Holt die globale RedisManager-Instanz.

    Returns:
        RedisManager-Singleton
    """
    # pylint: disable=global-statement
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager


async def get_redis_client() -> Any:
    """Convenience-Funktion für Redis-Client-Zugriff.

    Returns:
        Redis-Client oder NoOpCache
    """
    manager = get_redis_manager()
    return await manager.get_client()


__all__ = ["RedisManager", "get_redis_client", "get_redis_manager"]
