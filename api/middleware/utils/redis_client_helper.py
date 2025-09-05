"""Redis Client Helper für Middleware.

Zentrale Utility-Klasse für einheitliches Redis-Client-Handling mit Error-Handling,
Fallback-Logik und Performance-Tracking für alle Middleware-Komponenten.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

logger = get_logger(__name__)

T = TypeVar("T")


class RedisOperationResult:
    """Ergebnis einer Redis-Operation mit Metadaten."""

    def __init__(
        self,
        success: bool,
        value: Any = None,
        error: str | None = None,
        latency_ms: float | None = None,
        used_fallback: bool = False
    ):
        self.success = success
        self.value = value
        self.error = error
        self.latency_ms = latency_ms
        self.used_fallback = used_fallback


class RedisClientHelper:
    """Zentrale Redis-Client-Helper-Klasse für Middleware.

    Bietet einheitliche Redis-Operationen mit automatischem Error-Handling,
    Fallback-Mechanismen und Performance-Tracking.
    """

    def __init__(self, component_name: str = "middleware"):
        """Initialisiert Redis Client Helper.

        Args:
            component_name: Name der Komponente für Logging und Metriken
        """
        self.component_name = component_name
        self._client: Any | NoOpCache | None = None
        self._client_initialized = False

    async def _ensure_client(self) -> Any | NoOpCache | None:
        """Stellt sicher, dass Redis-Client verfügbar ist."""
        if not self._client_initialized:
            try:
                self._client = await get_cache_client()
                self._client_initialized = True
                if self._client and not isinstance(self._client, NoOpCache):
                    logger.debug(f"Redis-Client für {self.component_name} initialisiert")
                else:
                    logger.debug(f"NoOpCache-Fallback für {self.component_name} aktiviert")
            except Exception as e:
                logger.warning(f"Redis-Client-Initialisierung für {self.component_name} fehlgeschlagen: {e}")
                self._client = None
                self._client_initialized = True

        return self._client

    def _is_available(self, client: Any) -> bool:
        """Prüft, ob Redis-Client verfügbar und verwendbar ist."""
        return client is not None and not isinstance(client, NoOpCache)

    @asynccontextmanager
    async def _track_operation(self, operation_name: str):
        """Context Manager für Performance-Tracking von Redis-Operationen."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(f"Redis {operation_name} für {self.component_name}: {latency_ms:.2f}ms")

    async def safe_get(self, key: str, default: T = None) -> RedisOperationResult:
        """Sichere Redis GET-Operation mit Fallback.

        Args:
            key: Redis-Schlüssel
            default: Standardwert bei Fehler oder Cache-Miss

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        client = await self._ensure_client()

        if not self._is_available(client):
            return RedisOperationResult(
                success=False,
                value=default,
                error="Redis nicht verfügbar",
                used_fallback=True
            )

        try:
            async with self._track_operation("GET") as _:
                start_time = time.perf_counter()
                value = await client.get(key)  # type: ignore[attr-defined]
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return RedisOperationResult(
                    success=True,
                    value=value if value is not None else default,
                    latency_ms=latency_ms
                )
        except Exception as e:
            logger.debug(f"Redis GET für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                value=default,
                error=str(e),
                used_fallback=True
            )

    async def safe_set(self, key: str, value: Any, ttl: int | None = None) -> RedisOperationResult:
        """Sichere Redis SET/SETEX-Operation mit Fallback.

        Args:
            key: Redis-Schlüssel
            value: Zu speichernder Wert
            ttl: TTL in Sekunden (optional)

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        client = await self._ensure_client()

        if not self._is_available(client):
            return RedisOperationResult(
                success=False,
                error="Redis nicht verfügbar",
                used_fallback=True
            )

        try:
            async with self._track_operation("SET") as _:
                start_time = time.perf_counter()

                if ttl is not None:
                    result = await client.setex(key, ttl, value)  # type: ignore[attr-defined]
                else:
                    result = await client.set(key, value)  # type: ignore[attr-defined]

                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return RedisOperationResult(
                    success=True,
                    value=result,
                    latency_ms=latency_ms
                )
        except Exception as e:
            logger.debug(f"Redis SET für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                error=str(e),
                used_fallback=True
            )

    async def safe_incr(self, key: str) -> RedisOperationResult:
        """Sichere Redis INCR-Operation mit Fallback.

        Args:
            key: Redis-Schlüssel

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        client = await self._ensure_client()

        if not self._is_available(client):
            return RedisOperationResult(
                success=False,
                value=0,
                error="Redis nicht verfügbar",
                used_fallback=True
            )

        try:
            async with self._track_operation("INCR") as _:
                start_time = time.perf_counter()
                value = await client.incr(key)  # type: ignore[attr-defined]
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return RedisOperationResult(
                    success=True,
                    value=value,
                    latency_ms=latency_ms
                )
        except Exception as e:
            logger.debug(f"Redis INCR für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                value=0,
                error=str(e),
                used_fallback=True
            )

    async def safe_expire(self, key: str, ttl: int) -> RedisOperationResult:
        """Sichere Redis EXPIRE-Operation mit Fallback.

        Args:
            key: Redis-Schlüssel
            ttl: TTL in Sekunden

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        client = await self._ensure_client()

        if not self._is_available(client):
            return RedisOperationResult(
                success=False,
                error="Redis nicht verfügbar",
                used_fallback=True
            )

        try:
            async with self._track_operation("EXPIRE") as _:
                start_time = time.perf_counter()
                result = await client.expire(key, ttl)  # type: ignore[attr-defined]
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return RedisOperationResult(
                    success=True,
                    value=result,
                    latency_ms=latency_ms
                )
        except Exception as e:
            logger.debug(f"Redis EXPIRE für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                error=str(e),
                used_fallback=True
            )


    async def safe_sismember(self, key: str, member: str) -> RedisOperationResult:
        """Sichere Redis SISMEMBER-Operation mit Fallback.

        Args:
            key: Redis-Set-Schlüssel
            member: Zu prüfendes Set-Member

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        client = await self._ensure_client()

        if not self._is_available(client):
            return RedisOperationResult(
                success=False,
                value=False,
                error="Redis nicht verfügbar",
                used_fallback=True
            )

        try:
            async with self._track_operation("SISMEMBER") as _:
                start_time = time.perf_counter()
                is_member = await client.sismember(key, member)  # type: ignore[attr-defined]
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return RedisOperationResult(
                    success=True,
                    value=bool(is_member),
                    latency_ms=latency_ms
                )
        except Exception as e:
            logger.debug(f"Redis SISMEMBER für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                value=False,
                error=str(e),
                used_fallback=True
            )

    async def safe_sadd(self, key: str, *members: str) -> RedisOperationResult:
        """Sichere Redis SADD-Operation mit Fallback.

        Args:
            key: Redis-Set-Schlüssel
            members: Hinzuzufügende Set-Members

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        client = await self._ensure_client()

        if not self._is_available(client):
            return RedisOperationResult(
                success=False,
                value=0,
                error="Redis nicht verfügbar",
                used_fallback=True
            )

        try:
            async with self._track_operation("SADD") as _:
                start_time = time.perf_counter()
                added_count = await client.sadd(key, *members)  # type: ignore[attr-defined]
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                return RedisOperationResult(
                    success=True,
                    value=added_count,
                    latency_ms=latency_ms
                )
        except Exception as e:
            logger.debug(f"Redis SADD für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                value=0,
                error=str(e),
                used_fallback=True
            )

    async def safe_json_get(self, key: str, default: T = None) -> RedisOperationResult:
        """Sichere Redis GET-Operation mit JSON-Deserialisierung.

        Args:
            key: Redis-Schlüssel
            default: Standardwert bei Fehler oder Cache-Miss

        Returns:
            RedisOperationResult mit deserialisiertem JSON-Wert
        """
        result = await self.safe_get(key)

        if not result.success or result.value is None:
            return RedisOperationResult(
                success=result.success,
                value=default,
                error=result.error,
                latency_ms=result.latency_ms,
                used_fallback=result.used_fallback
            )

        try:
            parsed_value = json.loads(result.value)
            return RedisOperationResult(
                success=True,
                value=parsed_value,
                latency_ms=result.latency_ms
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"JSON-Deserialisierung für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                value=default,
                error=f"JSON-Deserialisierung fehlgeschlagen: {e}",
                used_fallback=True
            )

    async def safe_json_set(self, key: str, value: Any, ttl: int | None = None) -> RedisOperationResult:
        """Sichere Redis SET-Operation mit JSON-Serialisierung.

        Args:
            key: Redis-Schlüssel
            value: Zu serialisierender und speichernder Wert
            ttl: TTL in Sekunden (optional)

        Returns:
            RedisOperationResult mit Ergebnis und Metadaten
        """
        try:
            serialized_value = json.dumps(value)
        except (TypeError, ValueError) as e:
            logger.debug(f"JSON-Serialisierung für {self.component_name} fehlgeschlagen: {e}")
            return RedisOperationResult(
                success=False,
                error=f"JSON-Serialisierung fehlgeschlagen: {e}",
                used_fallback=True
            )

        return await self.safe_set(key, serialized_value, ttl)


__all__ = ["RedisClientHelper", "RedisOperationResult"]
