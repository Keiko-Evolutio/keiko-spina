"""Token Bucket Implementation für Rate Limiting.

Implementiert einen Token-Bucket-Algorithmus für Rate Limiting mit konfigurierbarer
Kapazität und Refill-Rate.
"""

from __future__ import annotations

import asyncio
import time

from .constants import DEFAULT_TOKEN_BUCKET_CAPACITY, DEFAULT_TOKEN_BUCKET_REFILL_RATE


class TokenBucket:
    """Token Bucket für Rate Limiting.

    Ein Token Bucket ist ein Algorithmus für Rate Limiting, der eine bestimmte
    Anzahl von Tokens in einem "Bucket" speichert. Tokens werden mit einer
    konfigurierbaren Rate nachgefüllt.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_TOKEN_BUCKET_CAPACITY,
        refill_rate: float = DEFAULT_TOKEN_BUCKET_REFILL_RATE,
        initial_tokens: int | None = None
    ) -> None:
        """Initialisiert den Token Bucket.

        Args:
            capacity: Maximale Anzahl von Tokens im Bucket
            refill_rate: Anzahl der Tokens pro Sekunde, die nachgefüllt werden
            initial_tokens: Anfängliche Anzahl von Tokens (Standard: capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Versucht, eine bestimmte Anzahl von Tokens zu verbrauchen.

        Args:
            tokens: Anzahl der zu verbrauchenden Tokens

        Returns:
            True wenn genügend Tokens verfügbar waren, False sonst
        """
        async with self._lock:
            await self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False



    async def _refill(self) -> None:
        """Füllt Tokens basierend auf der verstrichenen Zeit nach."""
        now = time.time()
        elapsed = now - self.last_refill

        # Berechne neue Tokens basierend auf der Refill-Rate
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    async def get_available_tokens(self) -> float:
        """Gibt die aktuelle Anzahl verfügbarer Tokens zurück."""
        async with self._lock:
            await self._refill()
            return self.tokens

    async def wait_for_tokens(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Wartet, bis genügend Tokens verfügbar sind.

        Args:
            tokens: Anzahl der benötigten Tokens
            timeout: Maximale Wartezeit in Sekunden

        Returns:
            True wenn Tokens verfügbar wurden, False bei Timeout
        """
        start_time = time.time()

        while True:
            if await self.consume(tokens):
                return True

            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Berechne Wartezeit bis genügend Tokens verfügbar sind
            async with self._lock:
                await self._refill()
                if self.tokens >= tokens:
                    continue

                needed_tokens = tokens - self.tokens
                wait_time = needed_tokens / self.refill_rate

                # Begrenze Wartezeit auf maximal 1 Sekunde pro Iteration
                wait_time = min(wait_time, 1.0)

            await asyncio.sleep(wait_time)

    def reset(self, tokens: int | None = None) -> None:
        """Setzt den Token Bucket zurück.

        Args:
            tokens: Neue Anzahl von Tokens (Standard: capacity)
        """
        self.tokens = tokens if tokens is not None else self.capacity
        self.last_refill = time.time()


class DistributedTokenBucket(TokenBucket):
    """Verteilter Token Bucket mit Redis-Backend (Placeholder)."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        redis_key: str,
        initial_tokens: int | None = None
    ) -> None:
        """Initialisiert den verteilten Token Bucket.

        Args:
            capacity: Maximale Anzahl von Tokens im Bucket
            refill_rate: Anzahl der Tokens pro Sekunde, die nachgefüllt werden
            redis_key: Redis-Schlüssel für die Speicherung
            initial_tokens: Anfängliche Anzahl von Tokens (Standard: capacity)
        """
        super().__init__(capacity, refill_rate, initial_tokens)
        self.redis_key = redis_key
        # Redis-Integration ist für zukünftige Versionen geplant

    async def consume(self, tokens: int = 1) -> bool:
        """Versucht, Tokens zu verbrauchen (mit Redis-Synchronisation).

        Aktuell: Fallback auf lokale Implementierung.
        """
        return await super().consume(tokens)


__all__ = [
    "DistributedTokenBucket",
    "TokenBucket",
]
