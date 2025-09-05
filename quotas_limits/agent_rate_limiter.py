# backend/quotas_limits/agent_rate_limiter.py
"""Agent-spezifischer Rate Limiter für Keiko Personal Assistant

Implementiert individuelle Request-Limits pro Agent-ID, Capability-spezifische
Rate-Limits und konfigurierbare Zeitfenster mit Burst-Handling.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .base_manager import BaseManager, ManagerConfig
from .constants import (
    DEFAULT_REQUESTS_PER_DAY,
    DEFAULT_REQUESTS_PER_HOUR,
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_REQUESTS_PER_SECOND,
)
from .utils import (
    get_current_timestamp,
)

logger = get_logger(__name__)


class RateLimitWindow(str, Enum):
    """Zeitfenster für Rate-Limiting."""
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


class PriorityLevel(int, Enum):
    """Prioritätslevel für Rate-Limiting."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class RateLimitAlgorithm(str, Enum):
    """Rate-Limiting-Algorithmen."""
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class BurstConfig:
    """Konfiguration für Burst-Handling."""
    burst_size: int
    burst_refill_rate: float  # Tokens pro Sekunde
    burst_window_seconds: int = 60
    burst_penalty_factor: float = 1.5  # Verlangsamung nach Burst

    def calculate_refill_amount(self, time_elapsed: float) -> float:
        """Berechnet Refill-Menge basierend auf verstrichener Zeit."""
        return min(self.burst_size, time_elapsed * self.burst_refill_rate)


@dataclass
class AgentRateLimit:
    """Rate-Limit-Konfiguration für Agent."""
    agent_id: str

    # Basis-Limits
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND
    requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE
    requests_per_hour: float = DEFAULT_REQUESTS_PER_HOUR
    requests_per_day: float = DEFAULT_REQUESTS_PER_DAY

    # Burst-Konfiguration
    burst_config: BurstConfig | None = None

    # Algorithmus
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW

    # Priorität
    priority: PriorityLevel = PriorityLevel.NORMAL

    # Capability-spezifische Overrides
    capability_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    # Gültigkeit
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Metadaten
    created_at: datetime = field(default_factory=get_current_timestamp)
    tenant_id: str | None = None
    tags: set[str] = field(default_factory=set)

    def get_limit_for_window(self, window: RateLimitWindow) -> float:
        """Gibt Limit für spezifisches Zeitfenster zurück."""
        window_mapping = {
            RateLimitWindow.PER_SECOND: self.requests_per_second,
            RateLimitWindow.PER_MINUTE: self.requests_per_minute,
            RateLimitWindow.PER_HOUR: self.requests_per_hour,
            RateLimitWindow.PER_DAY: self.requests_per_day
        }
        return window_mapping.get(window, self.requests_per_minute)

    def get_capability_limit(self, capability_id: str, window: RateLimitWindow) -> float | None:
        """Gibt Capability-spezifisches Limit zurück."""
        if capability_id in self.capability_overrides:
            overrides = self.capability_overrides[capability_id]
            return overrides.get(window.value)
        return None

    def is_valid(self) -> bool:
        """Prüft, ob Rate-Limit gültig ist."""
        if not self.enabled:
            return False

        now = datetime.now(UTC)

        if self.valid_from and now < self.valid_from:
            return False

        return not (self.valid_until and now > self.valid_until)


@dataclass
class CapabilityRateLimit:
    """Rate-Limit-Konfiguration für Capability."""
    capability_id: str

    # Basis-Limits
    operations_per_second: float = 5.0
    operations_per_minute: float = 300.0
    operations_per_hour: float = 18000.0

    # Datenvolumen-Limits
    data_volume_per_minute_mb: float = 100.0
    data_volume_per_hour_mb: float = 6000.0

    # Compute-Limits
    compute_time_per_minute_seconds: float = 60.0
    compute_time_per_hour_seconds: float = 3600.0

    # Concurrent-Limits
    max_concurrent_operations: int = 10

    # Priorität
    priority: PriorityLevel = PriorityLevel.NORMAL

    # Gültigkeit
    enabled: bool = True

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    description: str = ""


@dataclass
class RateLimitResult:
    """Ergebnis einer Rate-Limit-Prüfung."""
    allowed: bool
    agent_id: str
    capability_id: str | None = None

    # Aktuelle Nutzung
    current_usage: float = 0.0
    remaining: float = 0.0
    limit: float = 0.0

    # Zeitfenster-Info
    window: RateLimitWindow = RateLimitWindow.PER_MINUTE
    window_reset_time: datetime | None = None

    # Retry-Info
    retry_after_seconds: int | None = None

    # Burst-Info
    burst_used: bool = False
    burst_remaining: float = 0.0

    # Metadaten
    algorithm_used: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    priority_applied: PriorityLevel = PriorityLevel.NORMAL

    @property
    def usage_percentage(self) -> float:
        """Gibt Nutzung in Prozent zurück."""
        if self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100

    @property
    def is_near_limit(self) -> bool:
        """Prüft, ob nahe am Limit."""
        return self.usage_percentage > 80.0


class TokenBucket:
    """Token-Bucket-Implementierung für Rate-Limiting."""

    def __init__(self, capacity: float, refill_rate: float):
        """Initialisiert Token Bucket.

        Args:
            capacity: Maximale Anzahl Tokens
            refill_rate: Tokens pro Sekunde
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: float = 1.0) -> bool:
        """Versucht Tokens zu verbrauchen."""
        async with self._lock:
            now = time.time()

            # Refill tokens
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + (time_passed * self.refill_rate))
            self.last_refill = now

            # Prüfe verfügbare Tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def get_status(self) -> dict[str, float]:
        """Gibt aktuellen Status zurück."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            current_tokens = min(self.capacity, self.tokens + (time_passed * self.refill_rate))

            return {
                "current_tokens": current_tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "utilization": 1.0 - (current_tokens / self.capacity)
            }


class SlidingWindow:
    """Sliding-Window-Implementierung für Rate-Limiting."""

    def __init__(self, window_size_seconds: int, limit: float):
        """Initialisiert Sliding Window.

        Args:
            window_size_seconds: Größe des Zeitfensters
            limit: Maximale Anzahl Requests im Fenster
        """
        self.window_size = window_size_seconds
        self.limit = limit
        self.requests = deque()
        self._lock = asyncio.Lock()

    async def add_request(self, timestamp: float | None = None) -> bool:
        """Fügt Request hinzu und prüft Limit."""
        if timestamp is None:
            timestamp = time.time()

        async with self._lock:
            # Entferne alte Requests
            cutoff_time = timestamp - self.window_size
            while self.requests and self.requests[0] <= cutoff_time:
                self.requests.popleft()

            # Prüfe Limit
            if len(self.requests) >= self.limit:
                return False

            # Füge Request hinzu
            self.requests.append(timestamp)
            return True

    async def get_current_count(self) -> int:
        """Gibt aktuelle Anzahl Requests zurück."""
        async with self._lock:
            now = time.time()
            cutoff_time = now - self.window_size

            # Entferne alte Requests
            while self.requests and self.requests[0] <= cutoff_time:
                self.requests.popleft()

            return len(self.requests)

    async def get_remaining(self) -> float:
        """Gibt verbleibende Kapazität zurück."""
        current_count = await self.get_current_count()
        return max(0, self.limit - current_count)


class AgentRateLimiter(BaseManager):
    """Agent-spezifischer Rate Limiter."""

    def __init__(self, config: ManagerConfig | None = None):
        """Initialisiert Agent Rate Limiter."""
        super().__init__(config)

        self._agent_limits: dict[str, AgentRateLimit] = {}
        self._capability_limits: dict[str, CapabilityRateLimit] = {}

        # Rate-Limiting-Implementierungen
        self._token_buckets: dict[str, TokenBucket] = {}
        self._sliding_windows: dict[str, SlidingWindow] = {}

        # Zusätzliche Locks für spezifische Operationen
        self._agent_lock = asyncio.Lock()
        self._capability_lock = asyncio.Lock()

    def get_manager_type(self) -> str:
        """Gibt Manager-Typ zurück."""
        return "AgentRateLimiter"

    def register_agent_limit(self, agent_limit: AgentRateLimit) -> None:
        """Registriert Agent-Rate-Limit."""
        self._agent_limits[agent_limit.agent_id] = agent_limit
        if self.cache:
            asyncio.create_task(self.cache.delete(f"agent_limit:{agent_limit.agent_id}"))
        logger.info(f"Agent-Rate-Limit registriert: {agent_limit.agent_id}")

    def register_capability_limit(self, capability_limit: CapabilityRateLimit) -> None:
        """Registriert Capability-Rate-Limit."""
        self._capability_limits[capability_limit.capability_id] = capability_limit
        logger.info(f"Capability-Rate-Limit registriert: {capability_limit.capability_id}")

    @trace_function("rate_limit.check_agent")
    async def check_agent_rate_limit(
        self,
        agent_id: str,
        capability_id: str | None = None,
        window: RateLimitWindow = RateLimitWindow.PER_MINUTE,
        requested_amount: float = 1.0,
        context: dict[str, Any] | None = None
    ) -> RateLimitResult:
        """Prüft Agent-Rate-Limit."""
        cache_key = f"rate_limit:{agent_id}:{capability_id}:{window.value}:{requested_amount}"

        return await self.execute_operation(
            operation_name="check_agent_rate_limit",
            operation_func=self._check_agent_rate_limit_impl,
            agent_id=agent_id,
            capability_id=capability_id,
            window=window,
            requested_amount=requested_amount,
            context=context,
            cache_key=cache_key
        )

    async def _check_agent_rate_limit_impl(
        self,
        agent_id: str,
        capability_id: str | None,
        window: RateLimitWindow,
        requested_amount: float,
        context: dict[str, Any] | None
    ) -> RateLimitResult:
        """Implementierung der Agent-Rate-Limit-Prüfung."""
        # Hole Agent-Limit
        agent_limit = self._agent_limits.get(agent_id)
        if not agent_limit or not agent_limit.is_valid():
            return RateLimitResult(
                allowed=False,
                agent_id=agent_id,
                capability_id=capability_id,
                window=window,
                retry_after_seconds=60
            )

        # Prüfe Capability-spezifisches Limit
        if capability_id:
            capability_result = await self._check_capability_limit(
                agent_id, capability_id, window, requested_amount, context
            )
            if not capability_result.allowed:
                return capability_result

        # Bestimme effektives Limit
        effective_limit = agent_limit.get_limit_for_window(window)
        if capability_id:
            capability_override = agent_limit.get_capability_limit(capability_id, window)
            if capability_override:
                effective_limit = capability_override

        # Wähle Algorithmus
        if agent_limit.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            result = await self._check_token_bucket(
                agent_id, effective_limit, requested_amount, agent_limit.burst_config
            )
        elif agent_limit.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            result = await self._check_sliding_window(
                agent_id, window, effective_limit, requested_amount
            )
        else:  # Fixed Window oder Leaky Bucket
            result = await self._check_fixed_window(
                agent_id, window, effective_limit, requested_amount
            )

        return result

    async def _check_capability_limit(
        self,
        agent_id: str,
        capability_id: str,
        window: RateLimitWindow,
        requested_amount: float,
        context: dict[str, Any] | None
    ) -> RateLimitResult:
        """Prüft Capability-spezifisches Limit."""
        capability_limit = self._capability_limits.get(capability_id)
        if not capability_limit or not capability_limit.enabled:
            return RateLimitResult(
                allowed=True,
                agent_id=agent_id,
                capability_id=capability_id,
                window=window
            )

        # Bestimme Capability-Limit basierend auf Fenster
        if window == RateLimitWindow.PER_SECOND:
            limit = capability_limit.operations_per_second
        elif window == RateLimitWindow.PER_MINUTE:
            limit = capability_limit.operations_per_minute
        elif window == RateLimitWindow.PER_HOUR:
            limit = capability_limit.operations_per_hour
        else:
            limit = capability_limit.operations_per_minute

        # Prüfe mit Sliding Window
        window_key = f"capability:{capability_id}:{window.value}"

        if window_key not in self._sliding_windows:
            window_seconds = self._get_window_seconds(window)
            self._sliding_windows[window_key] = SlidingWindow(window_seconds, limit)

        sliding_window = self._sliding_windows[window_key]
        allowed = await sliding_window.add_request()

        current_count = await sliding_window.get_current_count()
        remaining = await sliding_window.get_remaining()

        return RateLimitResult(
            allowed=allowed,
            agent_id=agent_id,
            capability_id=capability_id,
            current_usage=current_count,
            remaining=remaining,
            limit=limit,
            window=window,
            algorithm_used=RateLimitAlgorithm.SLIDING_WINDOW,
            retry_after_seconds=None if allowed else self._get_window_seconds(window)
        )

    async def _check_token_bucket(
        self,
        agent_id: str,
        limit: float,
        requested_amount: float,
        burst_config: BurstConfig | None
    ) -> RateLimitResult:
        """Prüft mit Token-Bucket-Algorithmus."""
        bucket_key = f"agent:{agent_id}:token_bucket"

        if bucket_key not in self._token_buckets:
            refill_rate = limit / 60.0  # Pro Sekunde
            capacity = burst_config.burst_size if burst_config else limit
            self._token_buckets[bucket_key] = TokenBucket(capacity, refill_rate)

        bucket = self._token_buckets[bucket_key]
        allowed = await bucket.consume(requested_amount)

        status = await bucket.get_status()

        return RateLimitResult(
            allowed=allowed,
            agent_id=agent_id,
            current_usage=status["capacity"] - status["current_tokens"],
            remaining=status["current_tokens"],
            limit=status["capacity"],
            window=RateLimitWindow.PER_MINUTE,
            algorithm_used=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_used=burst_config is not None and allowed,
            burst_remaining=status["current_tokens"] if burst_config else 0.0,
            retry_after_seconds=None if allowed else int(requested_amount / status["refill_rate"])
        )

    async def _check_sliding_window(
        self,
        agent_id: str,
        window: RateLimitWindow,
        limit: float,
        requested_amount: float
    ) -> RateLimitResult:
        """Prüft mit Sliding-Window-Algorithmus."""
        window_key = f"agent:{agent_id}:{window.value}"

        if window_key not in self._sliding_windows:
            window_seconds = self._get_window_seconds(window)
            self._sliding_windows[window_key] = SlidingWindow(window_seconds, limit)

        sliding_window = self._sliding_windows[window_key]
        allowed = await sliding_window.add_request()

        current_count = await sliding_window.get_current_count()
        remaining = await sliding_window.get_remaining()

        return RateLimitResult(
            allowed=allowed,
            agent_id=agent_id,
            current_usage=current_count,
            remaining=remaining,
            limit=limit,
            window=window,
            algorithm_used=RateLimitAlgorithm.SLIDING_WINDOW,
            retry_after_seconds=None if allowed else self._get_window_seconds(window)
        )

    async def _check_fixed_window(
        self,
        agent_id: str,
        window: RateLimitWindow,
        limit: float,
        requested_amount: float
    ) -> RateLimitResult:
        """Prüft mit Fixed-Window-Algorithmus."""
        # Vereinfachte Fixed-Window-Implementierung
        # In Produktion würde hier eine echte Fixed-Window-Implementierung stehen
        return await self._check_sliding_window(agent_id, window, limit, requested_amount)

    def _get_window_seconds(self, window: RateLimitWindow) -> int:
        """Gibt Zeitfenster in Sekunden zurück."""
        window_mapping = {
            RateLimitWindow.PER_SECOND: 1,
            RateLimitWindow.PER_MINUTE: 60,
            RateLimitWindow.PER_HOUR: 3600,
            RateLimitWindow.PER_DAY: 86400
        }
        return window_mapping.get(window, 60)

    def get_agent_limits(self) -> list[AgentRateLimit]:
        """Gibt alle Agent-Limits zurück."""
        return list(self._agent_limits.values())

    def get_capability_limits(self) -> list[CapabilityRateLimit]:
        """Gibt alle Capability-Limits zurück."""
        return list(self._capability_limits.values())

    def remove_agent_limit(self, agent_id: str) -> bool:
        """Entfernt Agent-Rate-Limit."""
        if agent_id in self._agent_limits:
            del self._agent_limits[agent_id]
            if self.cache:
                asyncio.create_task(self.cache.delete(f"agent_limit:{agent_id}"))
            logger.info(f"Agent-Rate-Limit entfernt: {agent_id}")
            return True
        return False

    def get_rate_limiter_statistics(self) -> dict[str, Any]:
        """Gibt Rate-Limiter-spezifische Statistiken zurück."""
        base_status = self.get_status()
        rate_limiter_stats = {
            "registered_agents": len(self._agent_limits),
            "registered_capabilities": len(self._capability_limits),
            "active_token_buckets": len(self._token_buckets),
            "active_sliding_windows": len(self._sliding_windows)
        }

        # Kombiniere mit Base-Manager-Statistiken
        base_status.update(rate_limiter_stats)
        return base_status


# Globale Agent Rate Limiter Instanz
agent_rate_limiter = AgentRateLimiter()
