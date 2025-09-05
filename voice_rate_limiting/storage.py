"""Rate Limit Storage Implementations.
Redis-basierte und In-Memory Storage für Rate Limiting.
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

from kei_logging import get_logger

from .interfaces import IRateLimitStore

logger = get_logger(__name__)


class InMemoryRateLimitStore(IRateLimitStore):
    """In-Memory Rate Limit Store für lokale Entwicklung.
    Thread-safe Implementation mit Sliding Window Support.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Sliding Window Storage: key -> deque of timestamps
        self._windows: dict[str, deque] = defaultdict(lambda: deque())

        # Concurrent Connection Storage: key -> count
        self._concurrent: dict[str, int] = defaultdict(int)

        # Cleanup-Task
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        logger.info("In-memory rate limit store initialized")

    async def start_cleanup_task(self) -> None:
        """Startet periodische Cleanup-Task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug("Rate limit store cleanup task started")

    async def stop_cleanup_task(self) -> None:
        """Stoppt Cleanup-Task."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.debug("Rate limit store cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Periodische Cleanup-Schleife."""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(60)  # Cleanup every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit cleanup loop: {e}")
                await asyncio.sleep(5)

    async def _cleanup_expired_entries(self) -> None:
        """Entfernt abgelaufene Einträge."""
        current_time = time.time()

        with self._lock:
            # Cleanup sliding windows (entferne Einträge älter als 1 Stunde)
            cutoff_time = current_time - 3600  # 1 hour

            keys_to_remove = []
            for key, window in self._windows.items():
                # Entferne alte Timestamps
                while window and window[0] < cutoff_time:
                    window.popleft()

                # Entferne leere Windows
                if not window:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._windows[key]

            logger.debug(f"Cleaned up {len(keys_to_remove)} expired rate limit entries")

    async def get_count(self, key: str, window_seconds: int) -> int:
        """Gibt aktuelle Anzahl für Schlüssel zurück."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self._lock:
            window = self._windows[key]

            # Entferne alte Einträge
            while window and window[0] < cutoff_time:
                window.popleft()

            return len(window)

    async def increment(self, key: str, window_seconds: int, amount: int = 1) -> int:
        """Erhöht Counter und gibt neue Anzahl zurück."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self._lock:
            window = self._windows[key]

            # Entferne alte Einträge
            while window and window[0] < cutoff_time:
                window.popleft()

            # Füge neue Einträge hinzu
            for _ in range(amount):
                window.append(current_time)

            return len(window)

    async def get_reset_time(self, key: str, window_seconds: int) -> datetime:
        """Gibt Reset-Zeit für Schlüssel zurück."""
        current_time = time.time()

        with self._lock:
            window = self._windows[key]

            if not window:
                # Kein Window vorhanden, Reset ist sofort
                return datetime.fromtimestamp(current_time)

            # Reset-Zeit ist wenn der älteste Eintrag abläuft
            oldest_timestamp = window[0]
            reset_timestamp = oldest_timestamp + window_seconds

            return datetime.fromtimestamp(reset_timestamp)

    async def clear(self, key: str) -> None:
        """Löscht Counter für Schlüssel."""
        with self._lock:
            if key in self._windows:
                del self._windows[key]
            if key in self._concurrent:
                del self._concurrent[key]

    async def get_concurrent_count(self, key: str) -> int:
        """Gibt aktuelle Concurrent-Anzahl zurück."""
        with self._lock:
            return self._concurrent[key]

    async def increment_concurrent(self, key: str) -> int:
        """Erhöht Concurrent-Counter."""
        with self._lock:
            self._concurrent[key] += 1
            return self._concurrent[key]

    async def decrement_concurrent(self, key: str) -> int:
        """Verringert Concurrent-Counter."""
        with self._lock:
            if self._concurrent[key] > 0:
                self._concurrent[key] -= 1
            return self._concurrent[key]

    def get_stats(self) -> dict[str, int]:
        """Gibt Storage-Statistiken zurück."""
        with self._lock:
            return {
                "total_windows": len(self._windows),
                "total_concurrent": len(self._concurrent),
                "total_entries": sum(len(window) for window in self._windows.values())
            }


class RedisRateLimitStore(IRateLimitStore):
    """Redis-basierte Rate Limit Store für Production.
    Unterstützt Horizontal Scaling und High Availability.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._connected = False

        # Lua Scripts für atomare Operationen
        self._sliding_window_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local amount = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local cutoff = now - window

        -- Entferne alte Einträge
        redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)

        -- Füge neue Einträge hinzu
        for i = 1, amount do
            redis.call('ZADD', key, now, now .. ':' .. i)
        end

        -- Setze Expiration
        redis.call('EXPIRE', key, window + 60)

        -- Gib aktuelle Anzahl zurück
        return redis.call('ZCARD', key)
        """

        self._get_count_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local cutoff = now - window

        -- Entferne alte Einträge
        redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)

        -- Gib aktuelle Anzahl zurück
        return redis.call('ZCARD', key)
        """

        logger.info(f"Redis rate limit store configured: {redis_url}")

    async def connect(self) -> None:
        """Verbindet mit Redis."""
        if self._connected:
            return

        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            await self._redis.ping()
            self._connected = True

            logger.info("Redis rate limit store connected successfully")

        except ImportError:
            logger.error("redis package not installed, falling back to in-memory store")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Trennt Redis-Verbindung."""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("Redis rate limit store disconnected")

    async def get_count(self, key: str, window_seconds: int) -> int:
        """Gibt aktuelle Anzahl für Schlüssel zurück."""
        if not self._connected:
            await self.connect()

        try:
            current_time = time.time()
            result = await self._redis.eval(
                self._get_count_script,
                1,
                f"rl:{key}",
                window_seconds,
                current_time
            )
            return int(result)

        except Exception as e:
            logger.error(f"Redis get_count failed: {e}")
            raise

    async def increment(self, key: str, window_seconds: int, amount: int = 1) -> int:
        """Erhöht Counter und gibt neue Anzahl zurück."""
        if not self._connected:
            await self.connect()

        try:
            current_time = time.time()
            result = await self._redis.eval(
                self._sliding_window_script,
                1,
                f"rl:{key}",
                window_seconds,
                amount,
                current_time
            )
            return int(result)

        except Exception as e:
            logger.error(f"Redis increment failed: {e}")
            raise

    async def get_reset_time(self, key: str, window_seconds: int) -> datetime:
        """Gibt Reset-Zeit für Schlüssel zurück."""
        if not self._connected:
            await self.connect()

        try:
            # Hole ältesten Eintrag
            oldest_entries = await self._redis.zrange(f"rl:{key}", 0, 0, withscores=True)

            if not oldest_entries:
                # Kein Eintrag vorhanden
                return datetime.fromtimestamp(time.time())

            oldest_timestamp = oldest_entries[0][1]
            reset_timestamp = oldest_timestamp + window_seconds

            return datetime.fromtimestamp(reset_timestamp)

        except Exception as e:
            logger.error(f"Redis get_reset_time failed: {e}")
            # Fallback
            return datetime.fromtimestamp(time.time() + window_seconds)

    async def clear(self, key: str) -> None:
        """Löscht Counter für Schlüssel."""
        if not self._connected:
            await self.connect()

        try:
            await self._redis.delete(f"rl:{key}", f"cc:{key}")

        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            raise

    async def get_concurrent_count(self, key: str) -> int:
        """Gibt aktuelle Concurrent-Anzahl zurück."""
        if not self._connected:
            await self.connect()

        try:
            result = await self._redis.get(f"cc:{key}")
            return int(result) if result else 0

        except Exception as e:
            logger.error(f"Redis get_concurrent_count failed: {e}")
            return 0

    async def increment_concurrent(self, key: str) -> int:
        """Erhöht Concurrent-Counter."""
        if not self._connected:
            await self.connect()

        try:
            result = await self._redis.incr(f"cc:{key}")
            await self._redis.expire(f"cc:{key}", 3600)  # 1 hour expiration
            return int(result)

        except Exception as e:
            logger.error(f"Redis increment_concurrent failed: {e}")
            raise

    async def decrement_concurrent(self, key: str) -> int:
        """Verringert Concurrent-Counter."""
        if not self._connected:
            await self.connect()

        try:
            result = await self._redis.decr(f"cc:{key}")
            # Verhindere negative Werte
            if result < 0:
                await self._redis.set(f"cc:{key}", 0)
                result = 0
            return int(result)

        except Exception as e:
            logger.error(f"Redis decrement_concurrent failed: {e}")
            raise


def create_rate_limit_store(redis_url: str | None = None, force_in_memory: bool = False) -> IRateLimitStore:
    """Factory-Funktion für Rate Limit Store.

    Args:
        redis_url: Redis URL, falls None wird In-Memory Store verwendet
        force_in_memory: Erzwingt In-Memory Store (für Development)

    Returns:
        Rate Limit Store Instance
    """
    # Force In-Memory Store für Development oder wenn explizit gewünscht
    if force_in_memory or not redis_url:
        logger.info("Using in-memory rate limit store")
        return InMemoryRateLimitStore()

    # Versuche Redis Store zu erstellen
    try:
        # Prüfe ob redis package verfügbar ist
        import redis.asyncio as redis
        logger.info(f"Using Redis rate limit store: {redis_url}")
        return RedisRateLimitStore(redis_url)
    except ImportError:
        logger.warning("redis package not installed, falling back to in-memory store")
        return InMemoryRateLimitStore()
    except Exception as e:
        logger.warning(f"Failed to create Redis store ({e}), falling back to in-memory store")
        return InMemoryRateLimitStore()
