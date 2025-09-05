"""Redis-basierter Rate Limiter mit Sliding Window und Token Bucket Algorithmen.

Implementiert production-ready Rate Limiting mit Redis-Backend für
horizontale Skalierbarkeit und verteilte Koordination.
"""

import asyncio
import time
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from config.unified_rate_limiting import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitPolicy,
)
from kei_logging import get_logger
from services.interfaces.rate_limiter import RateLimiterBackend, RateLimitResult

logger = get_logger(__name__)


class RedisRateLimiter(RateLimiterBackend):
    """Redis-basierter Rate Limiter mit Sliding Window und Token Bucket.

    Implementiert sowohl RateLimiterBackend als auch RateLimiterService Interface.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._redis: aioredis.Redis | None = None
        self._connection_healthy = False

    async def _get_redis_client(self) -> aioredis.Redis:
        """Erstellt oder gibt Redis-Client zurück."""
        if self._redis is None:
            try:
                self._redis = aioredis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    ssl=self.config.redis_ssl,
                    socket_timeout=self.config.redis_timeout,
                    socket_connect_timeout=self.config.redis_timeout,
                    decode_responses=True,
                    retry_on_timeout=True,
                    health_check_interval=30
                )

                # Verbindung testen
                await self._redis.ping()
                self._connection_healthy = True
                logger.info(f"✅ Redis Rate Limiter verbunden: {self.config.redis_host}:{self.config.redis_port}")

            except Exception as e:
                logger.exception(f"❌ Redis Rate Limiter Verbindung fehlgeschlagen: {e}")
                self._connection_healthy = False
                raise

        return self._redis

    async def check_rate_limit(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float | None = None
    ) -> RateLimitResult:
        """Prüft Rate Limit mit konfigurierbarem Algorithmus."""
        if current_time is None:
            current_time = time.time()

        try:
            if policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._sliding_window_check(key, policy, current_time)
            if policy.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._token_bucket_check(key, policy, current_time)
            # FIXED_WINDOW
            return await self._fixed_window_check(key, policy, current_time)

        except (RedisError, RedisConnectionError) as e:
            logger.warning(f"Redis Rate Limiter Fehler für {key}: {e}")
            self._connection_healthy = False
            # Fallback auf erlaubt bei Redis-Fehlern
            return RateLimitResult(
                allowed=True,
                remaining=policy.requests_per_minute,
                reset_time=int(current_time + policy.window_size_seconds),
                current_usage=0,
                limit=policy.requests_per_minute
            )

    async def _sliding_window_check(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float
    ) -> RateLimitResult:
        """Sliding Window Rate Limiting mit Redis Sorted Sets."""
        redis = await self._get_redis_client()

        # Redis-Keys
        window_key = f"rl:sw:{key}"

        # Sliding Window Zeitbereich
        window_start = current_time - policy.window_size_seconds

        # Lua-Script für atomare Operationen
        lua_script = """
        local window_key = KEYS[1]
        local current_time = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])

        -- Entferne abgelaufene Einträge
        redis.call('ZREMRANGEBYSCORE', window_key, '-inf', window_start)

        -- Zähle aktuelle Requests im Fenster
        local current_count = redis.call('ZCARD', window_key)

        -- Prüfe Limit
        if current_count >= limit then
            return {0, current_count, limit}
        end

        -- Füge neuen Request hinzu
        redis.call('ZADD', window_key, current_time, current_time)
        redis.call('EXPIRE', window_key, ttl)

        return {1, current_count + 1, limit}
        """

        # Script ausführen
        result = await redis.eval(
            lua_script,
            1,
            window_key,
            str(current_time),
            str(window_start),
            policy.requests_per_minute,
            policy.window_size_seconds * 2  # TTL etwas länger als Window
        )

        allowed = bool(result[0])
        current_usage = int(result[1])
        limit = int(result[2])

        remaining = max(0, limit - current_usage)
        reset_time = int(current_time + policy.window_size_seconds)

        # Soft-Limit-Prüfung
        soft_limit_exceeded = current_usage >= (limit * policy.soft_limit_factor)

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=policy.window_size_seconds if not allowed else None,
            current_usage=current_usage,
            limit=limit,
            window_start=window_start,
            soft_limit_exceeded=soft_limit_exceeded
        )

    async def _token_bucket_check(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float
    ) -> RateLimitResult:
        """Token Bucket Rate Limiting für Burst-Handling."""
        redis = await self._get_redis_client()

        # Redis-Keys
        bucket_key = f"rl:tb:{key}"

        # Lua-Script für Token Bucket
        lua_script = """
        local bucket_key = KEYS[1]
        local current_time = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local refill_rate = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])

        -- Hole aktuelle Bucket-Daten
        local bucket_data = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket_data[1]) or capacity
        local last_refill = tonumber(bucket_data[2]) or current_time

        -- Berechne neue Tokens basierend auf verstrichener Zeit
        local time_passed = current_time - last_refill
        local new_tokens = math.min(capacity, tokens + (time_passed * refill_rate))

        -- Prüfe ob Request erlaubt ist
        if new_tokens >= 1 then
            -- Token verbrauchen
            new_tokens = new_tokens - 1

            -- Bucket-Status aktualisieren
            redis.call('HMSET', bucket_key, 'tokens', new_tokens, 'last_refill', current_time)
            redis.call('EXPIRE', bucket_key, ttl)

            return {1, new_tokens, capacity}
        else
            -- Nicht genug Tokens
            redis.call('HMSET', bucket_key, 'tokens', new_tokens, 'last_refill', current_time)
            redis.call('EXPIRE', bucket_key, ttl)

            return {0, new_tokens, capacity}
        end
        """

        # Script ausführen
        result = await redis.eval(
            lua_script,
            1,
            bucket_key,
            current_time,
            policy.burst_size,
            policy.burst_refill_rate,
            policy.window_size_seconds * 2
        )

        allowed = bool(result[0])
        tokens_remaining = float(result[1])
        bucket_capacity = int(result[2])

        # Berechne Retry-After basierend auf Token-Refill
        retry_after = None
        if not allowed:
            tokens_needed = 1.0 - tokens_remaining
            retry_after = int(tokens_needed / policy.burst_refill_rate) + 1

        return RateLimitResult(
            allowed=allowed,
            remaining=int(tokens_remaining),
            reset_time=int(current_time + (bucket_capacity / policy.burst_refill_rate)),
            retry_after=retry_after,
            current_usage=bucket_capacity - int(tokens_remaining),
            limit=bucket_capacity,
            tokens_remaining=tokens_remaining,
            bucket_capacity=bucket_capacity
        )

    async def _fixed_window_check(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float
    ) -> RateLimitResult:
        """Fixed Window Rate Limiting."""
        redis = await self._get_redis_client()

        # Window-basierter Key
        window_start = int(current_time // policy.window_size_seconds) * policy.window_size_seconds
        window_key = f"rl:fw:{key}:{window_start}"

        # Atomare Increment-Operation
        current_count = await redis.incr(window_key)

        # TTL setzen falls neuer Key
        if current_count == 1:
            await redis.expire(window_key, policy.window_size_seconds)

        allowed = current_count <= policy.requests_per_minute
        remaining = max(0, policy.requests_per_minute - current_count)
        reset_time = int(window_start + policy.window_size_seconds)

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=reset_time - int(current_time) if not allowed else None,
            current_usage=current_count,
            limit=policy.requests_per_minute,
            window_start=window_start
        )

    async def reset_rate_limit(self, key: str) -> bool:
        """Setzt Rate Limit für Key zurück."""
        try:
            redis = await self._get_redis_client()

            # Alle möglichen Keys für diesen Rate Limit Key löschen
            patterns = [
                f"rl:sw:{key}",
                f"rl:tb:{key}",
                f"rl:fw:{key}:*"
            ]

            deleted = 0
            for pattern in patterns:
                if "*" in pattern:
                    # Pattern-basierte Löschung
                    keys = await redis.keys(pattern)
                    if keys:
                        deleted += await redis.delete(*keys)
                else:
                    # Direkte Löschung
                    deleted += await redis.delete(pattern)

            logger.info(f"Rate Limit für {key} zurückgesetzt: {deleted} Keys gelöscht")
            return deleted > 0

        except Exception as e:
            logger.exception(f"Fehler beim Zurücksetzen von Rate Limit für {key}: {e}")
            return False

    async def get_rate_limit_info(self, key: str) -> dict[str, Any] | None:
        """Gibt Rate Limit Informationen für Key zurück."""
        try:
            redis = await self._get_redis_client()

            info = {
                "key": key,
                "sliding_window": None,
                "token_bucket": None,
                "fixed_window": None
            }

            # Sliding Window Info
            sw_key = f"rl:sw:{key}"
            if await redis.exists(sw_key):
                count = await redis.zcard(sw_key)
                ttl = await redis.ttl(sw_key)
                info["sliding_window"] = {
                    "current_count": count,
                    "ttl": ttl
                }

            # Token Bucket Info
            tb_key = f"rl:tb:{key}"
            if await redis.exists(tb_key):
                bucket_data = await redis.hmget(tb_key, "tokens", "last_refill")
                ttl = await redis.ttl(tb_key)
                info["token_bucket"] = {
                    "tokens": float(bucket_data[0]) if bucket_data[0] else 0,
                    "last_refill": float(bucket_data[1]) if bucket_data[1] else 0,
                    "ttl": ttl
                }

            # Fixed Window Info
            fw_pattern = f"rl:fw:{key}:*"
            fw_keys = await redis.keys(fw_pattern)
            if fw_keys:
                fw_info = []
                for fw_key in fw_keys:
                    count = await redis.get(fw_key)
                    ttl = await redis.ttl(fw_key)
                    fw_info.append({
                        "window": fw_key.split(":")[-1],
                        "count": int(count) if count else 0,
                        "ttl": ttl
                    })
                info["fixed_window"] = fw_info

            return info

        except Exception as e:
            logger.exception(f"Fehler beim Abrufen von Rate Limit Info für {key}: {e}")
            return None

    async def cleanup_expired(self) -> int:
        """Bereinigt abgelaufene Rate Limit Einträge."""
        try:
            redis = await self._get_redis_client()

            # Cleanup-Script für alle Rate Limit Patterns
            lua_script = """
            local patterns = {'rl:sw:*', 'rl:tb:*', 'rl:fw:*'}
            local deleted = 0

            for _, pattern in ipairs(patterns) do
                local keys = redis.call('KEYS', pattern)
                for _, key in ipairs(keys) do
                    local ttl = redis.call('TTL', key)
                    if ttl == -1 then
                        -- Key ohne TTL, setze Standard-TTL
                        redis.call('EXPIRE', key, 3600)
                    elseif ttl == -2 then
                        -- Key existiert nicht mehr
                        deleted = deleted + 1
                    end
                end
            end

            return deleted
            """

            deleted = await redis.eval(lua_script, 0)

            if deleted > 0:
                logger.info(f"Rate Limit Cleanup: {deleted} abgelaufene Einträge bereinigt")

            return int(deleted)

        except Exception as e:
            logger.exception(f"Fehler beim Rate Limit Cleanup: {e}")
            return 0

    async def health_check(self) -> bool:
        """Prüft Redis-Verbindung."""
        try:
            redis = await self._get_redis_client()
            await redis.ping()
            self._connection_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Redis Rate Limiter Health Check fehlgeschlagen: {e}")
            self._connection_healthy = False
            return False

    @property
    def is_healthy(self) -> bool:
        """Gibt Gesundheitsstatus zurück."""
        return self._connection_healthy

    async def get_statistics(self) -> dict[str, Any]:
        """Liefert Rate-Limiting-Statistiken."""
        try:
            redis = await self._get_redis_client()
            info = await redis.info()
            return {
                "backend": "redis",
                "healthy": self._connection_healthy,
                "redis_info": {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "uptime": info.get("uptime_in_seconds", 0)
                }
            }
        except Exception as e:
            logger.exception(f"Fehler beim Abrufen der Redis-Statistiken: {e}")
            return {
                "backend": "redis",
                "healthy": False,
                "error": str(e)
            }

    async def initialize(self) -> None:
        """Initialisiert den Redis Rate Limiter."""
        try:
            # Test Redis-Verbindung
            redis = await self._get_redis_client()
            await redis.ping()
            self._connection_healthy = True
            logger.info("Redis Rate Limiter erfolgreich initialisiert")
        except Exception as e:
            logger.warning(f"Redis Rate Limiter Initialisierung fehlgeschlagen: {e}")
            self._connection_healthy = False

    async def shutdown(self) -> None:
        """Fährt den Redis Rate Limiter herunter."""
        try:
            if hasattr(self, "_redis_pool") and self._redis_pool:
                await self._redis_pool.disconnect()
            logger.info("Redis Rate Limiter heruntergefahren")
        except Exception as e:
            logger.warning(f"Redis Rate Limiter Shutdown-Fehler: {e}")


class MemoryRateLimiter(RateLimiterBackend):
    """In-Memory Rate Limiter als Fallback.

    Implementiert sowohl RateLimiterBackend als auch RateLimiterService Interface.
    """

    def __init__(self):
        self._sliding_windows: dict[str, list[float]] = {}
        self._token_buckets: dict[str, dict[str, float]] = {}
        self._fixed_windows: dict[str, dict[str, int]] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float | None = None
    ) -> RateLimitResult:
        """In-Memory Rate Limit Check."""
        if current_time is None:
            current_time = time.time()

        async with self._lock:
            if policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._memory_sliding_window(key, policy, current_time)
            if policy.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._memory_token_bucket(key, policy, current_time)
            # FIXED_WINDOW
            return await self._memory_fixed_window(key, policy, current_time)

    async def _memory_sliding_window(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float
    ) -> RateLimitResult:
        """In-Memory Sliding Window."""
        if key not in self._sliding_windows:
            self._sliding_windows[key] = []

        window = self._sliding_windows[key]
        window_start = current_time - policy.window_size_seconds

        # Entferne abgelaufene Einträge
        window[:] = [t for t in window if t > window_start]

        # Prüfe Limit
        if len(window) >= policy.requests_per_minute:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=int(current_time + policy.window_size_seconds),
                retry_after=policy.window_size_seconds,
                current_usage=len(window),
                limit=policy.requests_per_minute
            )

        # Füge Request hinzu
        window.append(current_time)

        return RateLimitResult(
            allowed=True,
            remaining=policy.requests_per_minute - len(window),
            reset_time=int(current_time + policy.window_size_seconds),
            current_usage=len(window),
            limit=policy.requests_per_minute
        )

    async def _memory_token_bucket(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float
    ) -> RateLimitResult:
        """In-Memory Token Bucket."""
        if key not in self._token_buckets:
            self._token_buckets[key] = {
                "tokens": float(policy.burst_size),
                "last_refill": current_time
            }

        bucket = self._token_buckets[key]

        # Token-Refill
        time_passed = current_time - bucket["last_refill"]
        new_tokens = min(
            policy.burst_size,
            bucket["tokens"] + (time_passed * policy.burst_refill_rate)
        )

        bucket["tokens"] = new_tokens
        bucket["last_refill"] = current_time

        # Prüfe Token-Verfügbarkeit
        if new_tokens >= 1:
            bucket["tokens"] -= 1
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket["tokens"]),
                reset_time=int(current_time + (policy.burst_size / policy.burst_refill_rate)),
                tokens_remaining=bucket["tokens"],
                bucket_capacity=policy.burst_size
            )
        retry_after = int((1.0 - new_tokens) / policy.burst_refill_rate) + 1
        return RateLimitResult(
            allowed=False,
            remaining=0,
            reset_time=int(current_time + retry_after),
            retry_after=retry_after,
            tokens_remaining=bucket["tokens"],
            bucket_capacity=policy.burst_size
        )

    async def _memory_fixed_window(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: float
    ) -> RateLimitResult:
        """In-Memory Fixed Window."""
        window_start = int(current_time // policy.window_size_seconds) * policy.window_size_seconds
        window_key = f"{key}:{window_start}"

        if key not in self._fixed_windows:
            self._fixed_windows[key] = {}

        # Cleanup alte Windows
        current_windows = {}
        for wk, count in self._fixed_windows[key].items():
            if int(wk.split(":")[-1]) >= window_start - policy.window_size_seconds:
                current_windows[wk] = count
        self._fixed_windows[key] = current_windows

        # Aktueller Window-Count
        current_count = self._fixed_windows[key].get(window_key, 0) + 1
        self._fixed_windows[key][window_key] = current_count

        allowed = current_count <= policy.requests_per_minute
        remaining = max(0, policy.requests_per_minute - current_count)
        reset_time = window_start + policy.window_size_seconds

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=reset_time - int(current_time) if not allowed else None,
            current_usage=current_count,
            limit=policy.requests_per_minute
        )

    async def reset_rate_limit(self, key: str) -> bool:
        """Setzt In-Memory Rate Limit zurück."""
        async with self._lock:
            deleted = 0

            if key in self._sliding_windows:
                del self._sliding_windows[key]
                deleted += 1

            if key in self._token_buckets:
                del self._token_buckets[key]
                deleted += 1

            if key in self._fixed_windows:
                del self._fixed_windows[key]
                deleted += 1

            return deleted > 0

    async def get_rate_limit_info(self, key: str) -> dict[str, Any] | None:
        """Gibt In-Memory Rate Limit Info zurück."""
        async with self._lock:
            info = {"key": key}

            if key in self._sliding_windows:
                info["sliding_window"] = {
                    "current_count": len(self._sliding_windows[key])
                }

            if key in self._token_buckets:
                info["token_bucket"] = self._token_buckets[key].copy()

            if key in self._fixed_windows:
                info["fixed_window"] = self._fixed_windows[key].copy()

            return info if len(info) > 1 else None

    async def cleanup_expired(self) -> int:
        """Bereinigt abgelaufene In-Memory Einträge."""
        current_time = time.time()
        deleted = 0

        async with self._lock:
            # Cleanup Sliding Windows (älter als 1 Stunde)
            expired_sw = []
            for key, window in self._sliding_windows.items():
                if not window or max(window) < current_time - 3600:
                    expired_sw.append(key)

            for key in expired_sw:
                del self._sliding_windows[key]
                deleted += 1

            # Cleanup Token Buckets (älter als 1 Stunde)
            expired_tb = []
            for key, bucket in self._token_buckets.items():
                if bucket["last_refill"] < current_time - 3600:
                    expired_tb.append(key)

            for key in expired_tb:
                del self._token_buckets[key]
                deleted += 1

            # Cleanup Fixed Windows
            for key in list(self._fixed_windows.keys()):
                windows = self._fixed_windows[key]
                current_windows = {}
                for wk, count in windows.items():
                    window_time = int(wk.split(":")[-1])
                    if window_time >= current_time - 3600:
                        current_windows[wk] = count

                if current_windows:
                    self._fixed_windows[key] = current_windows
                else:
                    del self._fixed_windows[key]
                    deleted += 1

        return deleted

    async def get_statistics(self) -> dict[str, Any]:
        """Liefert In-Memory Rate-Limiting-Statistiken."""
        async with self._lock:
            return {
                "backend": "memory",
                "healthy": True,
                "sliding_windows": len(self._sliding_windows),
                "token_buckets": len(self._token_buckets),
                "fixed_windows": len(self._fixed_windows),
                "total_keys": len(self._sliding_windows) + len(self._token_buckets) + len(self._fixed_windows)
            }

    async def initialize(self) -> None:
        """Initialisiert den Memory Rate Limiter."""
        logger.info("Memory Rate Limiter erfolgreich initialisiert")

    async def shutdown(self) -> None:
        """Fährt den Memory Rate Limiter herunter."""
        async with self._lock:
            self._sliding_windows.clear()
            self._token_buckets.clear()
            self._fixed_windows.clear()
        logger.info("Memory Rate Limiter heruntergefahren")
