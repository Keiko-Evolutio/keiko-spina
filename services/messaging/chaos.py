"""Chaos-Profile (Delay/Drop) und Replay/Time-Travel Store (einfach).

Replay speichert Rohnachrichten pro Subject (begrenzte Anzahl) für Wiedereinspielung.
"""

from __future__ import annotations

import asyncio
import random

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

from .config import bus_settings

logger = get_logger(__name__)

# Konstanten für Chaos-Engineering
_CHAOS_PROFILE_KEY = "bus:chaos:profile"  # none|delay|drop|mixed
_CHAOS_DELAY_MS = "bus:chaos:delay_ms"
_REPLAY_LIST = "bus:replay:{subject}"

# Chaos-Wahrscheinlichkeiten und Konvertierungen
MILLISECONDS_TO_SECONDS = 1000.0
DROP_PROBABILITY_THRESHOLD = 0.5
MIXED_DROP_PROBABILITY_THRESHOLD = 0.7


async def set_chaos_profile(profile: str, delay_ms: int = 100) -> None:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return
    await client.set(_CHAOS_PROFILE_KEY, profile)
    await client.set(_CHAOS_DELAY_MS, str(delay_ms))


async def apply_chaos(subject: str, raw_data: bytes) -> bool:
    """Wendet Chaos-Profil an.

    Returns:
        bool: True = weiter verarbeiten; False = Drop
    """
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return True
    try:
        profile = await client.get(_CHAOS_PROFILE_KEY)
        if not profile or profile == "none":
            return True
        delay_ms_raw = await client.get(_CHAOS_DELAY_MS)
        delay_ms = int(delay_ms_raw) if delay_ms_raw else 0
        if profile == "delay":
            await asyncio.sleep(max(0, delay_ms) / MILLISECONDS_TO_SECONDS)
            return True
        if profile == "drop":
            return random.random() > DROP_PROBABILITY_THRESHOLD
        if profile == "mixed":
            if random.random() > MIXED_DROP_PROBABILITY_THRESHOLD:
                return False
            await asyncio.sleep(max(0, delay_ms) / MILLISECONDS_TO_SECONDS)
            return True
        return True
    except Exception:
        return True


async def store_replay(subject: str, raw_data: bytes) -> None:
    """Speichert Rohnachricht für Replay (begrenzte Länge)."""
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return
    key = _REPLAY_LIST.format(subject=subject)
    try:
        await client.lpush(key, raw_data.decode("utf-8"))
        limit = max(10, bus_settings.replay_store_limit_per_subject)
        await client.ltrim(key, 0, limit - 1)
    except Exception:
        pass


async def list_replay(subject: str, count: int = 100) -> list[str]:
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        return []
    key = _REPLAY_LIST.format(subject=subject)
    try:
        items = await client.lrange(key, 0, max(0, count - 1))
        return [str(i) for i in (items or [])]  # type: ignore[union-attr]
    except Exception:
        return []
