"""Rate Limiting Algorithmen Implementation.
Sliding Window, Token Bucket, Fixed Window und Leaky Bucket Algorithmen.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime

from kei_logging import get_logger

from .interfaces import (
    IRateLimitAlgorithm,
    IRateLimitStore,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitResult,
)

logger = get_logger(__name__)


class SlidingWindowAlgorithm(IRateLimitAlgorithm):
    """Sliding Window Rate Limiting Algorithm.
    Präzise Rate Limiting mit gleitendem Zeitfenster.
    """

    async def check_limit(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Prüft Rate Limit für Schlüssel."""
        current_count = await store.get_count(key, config.window_seconds)
        remaining = max(0, config.limit - current_count)
        allowed = current_count < config.limit

        reset_time = await store.get_reset_time(key, config.window_seconds)
        retry_after = None if allowed else int((reset_time - datetime.now()).total_seconds())

        return RateLimitResult(
            allowed=allowed,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after_seconds=retry_after,
            algorithm_used=RateLimitAlgorithm.SLIDING_WINDOW,
            key=key
        )

    async def consume(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Schlüssel."""
        # Erst prüfen, dann konsumieren für Atomarität
        current_count = await store.get_count(key, config.window_seconds)

        if current_count + amount > config.limit:
            # Limit würde überschritten
            remaining = max(0, config.limit - current_count)
            reset_time = await store.get_reset_time(key, config.window_seconds)
            retry_after = int((reset_time - datetime.now()).total_seconds())

            return RateLimitResult(
                allowed=False,
                limit=config.limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                algorithm_used=RateLimitAlgorithm.SLIDING_WINDOW,
                key=key
            )

        # Konsumiere
        new_count = await store.increment(key, config.window_seconds, amount)
        remaining = max(0, config.limit - new_count)
        reset_time = await store.get_reset_time(key, config.window_seconds)

        return RateLimitResult(
            allowed=True,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            algorithm_used=RateLimitAlgorithm.SLIDING_WINDOW,
            key=key
        )


@dataclass
class TokenBucketState:
    """Token Bucket State."""
    tokens: float
    last_refill: float


class TokenBucketAlgorithm(IRateLimitAlgorithm):
    """Token Bucket Rate Limiting Algorithm.
    Erlaubt Burst-Traffic bis zur Bucket-Kapazität.
    """

    def __init__(self):
        # In-Memory Token Bucket States (für Redis könnte man Lua Scripts verwenden)
        self._buckets: dict[str, TokenBucketState] = {}
        self._lock = asyncio.Lock()

    async def check_limit(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Prüft Rate Limit für Schlüssel."""
        async with self._lock:
            bucket = await self._get_or_create_bucket(key, config)
            await self._refill_bucket(bucket, config)

            allowed = bucket.tokens >= 1
            remaining = int(bucket.tokens)

            # Reset-Zeit berechnen (wann wird nächster Token verfügbar)
            if bucket.tokens < 1:
                tokens_needed = 1 - bucket.tokens
                refill_rate = config.limit / config.window_seconds
                seconds_to_wait = tokens_needed / refill_rate
                reset_time = datetime.fromtimestamp(time.time() + seconds_to_wait)
                retry_after = int(seconds_to_wait)
            else:
                reset_time = datetime.fromtimestamp(time.time())
                retry_after = None

            return RateLimitResult(
                allowed=allowed,
                limit=config.burst_limit or config.limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                algorithm_used=RateLimitAlgorithm.TOKEN_BUCKET,
                key=key
            )

    async def consume(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Schlüssel."""
        async with self._lock:
            bucket = await self._get_or_create_bucket(key, config)
            await self._refill_bucket(bucket, config)

            if bucket.tokens >= amount:
                # Konsumiere Tokens
                bucket.tokens -= amount
                allowed = True
            else:
                allowed = False

            remaining = int(bucket.tokens)

            # Reset-Zeit berechnen
            if bucket.tokens < amount:
                tokens_needed = amount - bucket.tokens
                refill_rate = config.limit / config.window_seconds
                seconds_to_wait = tokens_needed / refill_rate
                reset_time = datetime.fromtimestamp(time.time() + seconds_to_wait)
                retry_after = int(seconds_to_wait) if not allowed else None
            else:
                reset_time = datetime.fromtimestamp(time.time())
                retry_after = None

            return RateLimitResult(
                allowed=allowed,
                limit=config.burst_limit or config.limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                algorithm_used=RateLimitAlgorithm.TOKEN_BUCKET,
                key=key
            )

    async def _get_or_create_bucket(self, key: str, config: RateLimitConfig) -> TokenBucketState:
        """Holt oder erstellt Token Bucket."""
        if key not in self._buckets:
            bucket_capacity = config.burst_limit or config.limit
            self._buckets[key] = TokenBucketState(
                tokens=float(bucket_capacity),
                last_refill=time.time()
            )

        return self._buckets[key]

    async def _refill_bucket(self, bucket: TokenBucketState, config: RateLimitConfig) -> None:
        """Füllt Token Bucket auf."""
        current_time = time.time()
        time_passed = current_time - bucket.last_refill

        # Berechne neue Tokens
        refill_rate = config.limit / config.window_seconds  # Tokens pro Sekunde
        new_tokens = time_passed * refill_rate

        # Füge Tokens hinzu (bis zur Kapazität)
        bucket_capacity = config.burst_limit or config.limit
        bucket.tokens = min(bucket_capacity, bucket.tokens + new_tokens)
        bucket.last_refill = current_time


class FixedWindowAlgorithm(IRateLimitAlgorithm):
    """Fixed Window Rate Limiting Algorithm.
    Einfacher Algorithmus mit festen Zeitfenstern.
    """

    async def check_limit(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Prüft Rate Limit für Schlüssel."""
        window_key = self._get_window_key(key, config.window_seconds)
        current_count = await store.get_count(window_key, config.window_seconds)

        remaining = max(0, config.limit - current_count)
        allowed = current_count < config.limit

        reset_time = self._get_window_reset_time(config.window_seconds)
        retry_after = None if allowed else int((reset_time - datetime.now()).total_seconds())

        return RateLimitResult(
            allowed=allowed,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after_seconds=retry_after,
            algorithm_used=RateLimitAlgorithm.FIXED_WINDOW,
            key=key
        )

    async def consume(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Schlüssel."""
        window_key = self._get_window_key(key, config.window_seconds)

        # Prüfe aktuellen Count
        current_count = await store.get_count(window_key, config.window_seconds)

        if current_count + amount > config.limit:
            # Limit würde überschritten
            remaining = max(0, config.limit - current_count)
            reset_time = self._get_window_reset_time(config.window_seconds)
            retry_after = int((reset_time - datetime.now()).total_seconds())

            return RateLimitResult(
                allowed=False,
                limit=config.limit,
                remaining=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                algorithm_used=RateLimitAlgorithm.FIXED_WINDOW,
                key=key
            )

        # Konsumiere
        new_count = await store.increment(window_key, config.window_seconds, amount)
        remaining = max(0, config.limit - new_count)
        reset_time = self._get_window_reset_time(config.window_seconds)

        return RateLimitResult(
            allowed=True,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            algorithm_used=RateLimitAlgorithm.FIXED_WINDOW,
            key=key
        )

    def _get_window_key(self, key: str, window_seconds: int) -> str:
        """Generiert Window-spezifischen Schlüssel."""
        current_time = int(time.time())
        window_start = (current_time // window_seconds) * window_seconds
        return f"{key}:window:{window_start}"

    def _get_window_reset_time(self, window_seconds: int) -> datetime:
        """Berechnet Reset-Zeit für aktuelles Window."""
        current_time = int(time.time())
        window_start = (current_time // window_seconds) * window_seconds
        window_end = window_start + window_seconds
        return datetime.fromtimestamp(window_end)


class LeakyBucketAlgorithm(IRateLimitAlgorithm):
    """Leaky Bucket Rate Limiting Algorithm.
    Glättet Traffic durch konstante Ausgabe-Rate.
    """

    def __init__(self):
        # Vereinfachte Implementation - ähnlich wie Token Bucket
        self._buckets: dict[str, TokenBucketState] = {}
        self._lock = asyncio.Lock()

    async def check_limit(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Prüft Rate Limit für Schlüssel."""
        async with self._lock:
            bucket = await self._get_or_create_bucket(key, config)
            await self._leak_bucket(bucket, config)

            # Bei Leaky Bucket ist der "Bucket" die Warteschlange
            bucket_capacity = config.burst_limit or config.limit
            allowed = bucket.tokens < bucket_capacity
            remaining = int(bucket_capacity - bucket.tokens)

            if not allowed:
                # Berechne Wartezeit bis Platz frei wird
                leak_rate = config.limit / config.window_seconds
                seconds_to_wait = 1 / leak_rate  # Zeit bis nächster Slot frei wird
                reset_time = datetime.fromtimestamp(time.time() + seconds_to_wait)
                retry_after = int(seconds_to_wait)
            else:
                reset_time = datetime.fromtimestamp(time.time())
                retry_after = None

            return RateLimitResult(
                allowed=allowed,
                limit=bucket_capacity,
                remaining=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                algorithm_used=RateLimitAlgorithm.LEAKY_BUCKET,
                key=key
            )

    async def consume(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Schlüssel."""
        async with self._lock:
            bucket = await self._get_or_create_bucket(key, config)
            await self._leak_bucket(bucket, config)

            bucket_capacity = config.burst_limit or config.limit

            if bucket.tokens + amount <= bucket_capacity:
                # Füge zur Warteschlange hinzu
                bucket.tokens += amount
                allowed = True
            else:
                allowed = False

            remaining = int(bucket_capacity - bucket.tokens)

            if not allowed:
                leak_rate = config.limit / config.window_seconds
                seconds_to_wait = amount / leak_rate
                reset_time = datetime.fromtimestamp(time.time() + seconds_to_wait)
                retry_after = int(seconds_to_wait)
            else:
                reset_time = datetime.fromtimestamp(time.time())
                retry_after = None

            return RateLimitResult(
                allowed=allowed,
                limit=bucket_capacity,
                remaining=remaining,
                reset_time=reset_time,
                retry_after_seconds=retry_after,
                algorithm_used=RateLimitAlgorithm.LEAKY_BUCKET,
                key=key
            )

    async def _get_or_create_bucket(self, key: str, config: RateLimitConfig) -> TokenBucketState:
        """Holt oder erstellt Leaky Bucket."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucketState(
                tokens=0.0,  # Leaky Bucket startet leer
                last_refill=time.time()
            )

        return self._buckets[key]

    async def _leak_bucket(self, bucket: TokenBucketState, config: RateLimitConfig) -> None:
        """Lässt Bucket "lecken" (verarbeitet Warteschlange)."""
        current_time = time.time()
        time_passed = current_time - bucket.last_refill

        # Berechne wie viele Requests verarbeitet wurden
        leak_rate = config.limit / config.window_seconds  # Requests pro Sekunde
        leaked_amount = time_passed * leak_rate

        # Reduziere Bucket-Inhalt
        bucket.tokens = max(0, bucket.tokens - leaked_amount)
        bucket.last_refill = current_time


def create_rate_limit_algorithm(algorithm: RateLimitAlgorithm) -> IRateLimitAlgorithm:
    """Factory-Funktion für Rate Limiting Algorithmen.

    Args:
        algorithm: Gewünschter Algorithmus

    Returns:
        Algorithm-Instance
    """
    if algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
        return SlidingWindowAlgorithm()
    if algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
        return TokenBucketAlgorithm()
    if algorithm == RateLimitAlgorithm.FIXED_WINDOW:
        return FixedWindowAlgorithm()
    if algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
        return LeakyBucketAlgorithm()
    raise ValueError(f"Unknown rate limiting algorithm: {algorithm}")
