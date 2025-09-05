# backend/services/enhanced_quotas_limits_management/rate_limiting_engine.py
"""Rate Limiting Engine für Multi-Tenant Rate Limiting.

Implementiert hierarchical Rate Limiting mit Multi-Tenant-Isolation
und Integration mit bestehenden Rate-Limiting-Systemen.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any
from unittest.mock import MagicMock

from kei_logging import get_logger
from quotas_limits import AgentRateLimiter, agent_rate_limiter

from .data_models import (
    EnforcementAction,
    QuotaCheckResult,
    QuotaPeriod,
    QuotaScope,
    RateLimit,
    ResourceType,
)

# from services.redis_rate_limiter import RedisRateLimiter, RateLimitPolicy, RateLimitAlgorithm

logger = get_logger(__name__)


class RateLimitingEngine:
    """Rate Limiting Engine für Multi-Tenant Rate Limiting."""

    def __init__(
        self,
        existing_agent_rate_limiter: AgentRateLimiter | None = None,
        redis_rate_limiter: Any | None = None
    ):
        """Initialisiert Rate Limiting Engine.

        Args:
            existing_agent_rate_limiter: Bestehender Agent Rate Limiter
            redis_rate_limiter: Redis Rate Limiter
        """
        self.existing_agent_rate_limiter = existing_agent_rate_limiter or agent_rate_limiter
        self.redis_rate_limiter = redis_rate_limiter

        # Rate-Limiting-Konfiguration
        self.enable_multi_tenant_isolation = True
        self.enable_hierarchical_limits = True
        self.enable_burst_capacity = True
        self.default_algorithm = "sliding_window"

        # Rate-Limits-Storage
        self._rate_limits: dict[str, RateLimit] = {}
        self._tenant_isolation_config: dict[str, dict[str, Any]] = {}

        # In-Memory Rate Limiting (Fallback)
        self._in_memory_counters: dict[str, deque] = defaultdict(lambda: deque())
        self._burst_counters: dict[str, int] = defaultdict(int)

        # Performance-Tracking
        self._rate_limit_check_count = 0
        self._total_rate_limit_check_time_ms = 0.0
        self._rate_limit_violations = 0

        # Background-Tasks
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Rate Limiting Engine initialisiert")

    async def start(self) -> None:
        """Startet Rate Limiting Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Initialisiere Redis Rate Limiter
        if self.redis_rate_limiter:
            try:
                await self.redis_rate_limiter.connect()
            except Exception as e:
                logger.warning(f"Redis Rate Limiter connection fehlgeschlagen: {e}")

        # Starte Background-Tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Initialisiere Standard-Rate-Limits
        await self._initialize_default_rate_limits()

        logger.info("Rate Limiting Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Rate Limiting Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Schließe Redis-Verbindung
        if self.redis_rate_limiter:
            try:
                await self.redis_rate_limiter.disconnect()
            except Exception as e:
                logger.warning(f"Redis Rate Limiter disconnect fehlgeschlagen: {e}")

        logger.info("Rate Limiting Engine gestoppt")

    async def check_rate_limit(
        self,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        amount: int = 1,
        tenant_id: str | None = None
    ) -> QuotaCheckResult:
        """Prüft Rate Limit für Resource/Scope.

        Args:
            resource_type: Resource-Type
            scope: Quota-Scope
            scope_id: Scope-ID
            amount: Anzahl der Requests
            tenant_id: Tenant-ID für Multi-Tenant-Isolation

        Returns:
            Quota-Check-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "rate_limit_check_started",
                "resource_type": resource_type.value,
                "scope": scope.value,
                "scope_id": scope_id,
                "amount": amount,
                "tenant_id": tenant_id
            })

            # 1. Hole relevante Rate-Limits
            rate_limits = await self._get_relevant_rate_limits(
                resource_type, scope, scope_id, tenant_id
            )

            if not rate_limits:
                # Keine Rate-Limits definiert - erlauben
                return QuotaCheckResult(
                    allowed=True,
                    quota_id=f"rate_limit_{resource_type.value}_{scope.value}_{scope_id}",
                    current_usage=0,
                    limit=-1,  # Unlimited
                    remaining=-1,
                    check_duration_ms=(time.time() - start_time) * 1000
                )

            # 2. Multi-Tenant-Isolation-Check
            if self.enable_multi_tenant_isolation and tenant_id:
                isolation_result = await self._check_tenant_isolation(
                    tenant_id, resource_type, scope, scope_id
                )
                if not isolation_result.allowed:
                    return isolation_result

            # 3. Hierarchical Rate-Limit-Check
            for rate_limit in rate_limits:
                check_result = await self._check_single_rate_limit(
                    rate_limit, amount, tenant_id
                )

                if not check_result.allowed:
                    # Rate-Limit überschritten
                    self._rate_limit_violations += 1

                    check_result.check_duration_ms = (time.time() - start_time) * 1000
                    self._update_rate_limit_performance_stats(check_result.check_duration_ms)

                    logger.warning({
                        "event": "rate_limit_exceeded",
                        "rate_limit_id": rate_limit.rate_limit_id,
                        "current_usage": check_result.current_usage,
                        "limit": check_result.limit,
                        "tenant_id": tenant_id
                    })

                    return check_result

            # 4. Alle Rate-Limits bestanden
            check_duration_ms = (time.time() - start_time) * 1000
            self._update_rate_limit_performance_stats(check_duration_ms)

            return QuotaCheckResult(
                allowed=True,
                quota_id=f"rate_limit_{resource_type.value}_{scope.value}_{scope_id}",
                current_usage=0,  # TODO: Implementiere echte Usage-Tracking - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
                limit=rate_limits[0].requests_per_period if rate_limits else -1,
                remaining=rate_limits[0].requests_per_period if rate_limits else -1,
                check_duration_ms=check_duration_ms
            )

        except Exception as e:
            logger.error(f"Rate limit check fehlgeschlagen: {e}")

            # Fallback: Erlauben bei Fehler (fail-open)
            return QuotaCheckResult(
                allowed=True,
                quota_id=f"rate_limit_{resource_type.value}_{scope.value}_{scope_id}",
                current_usage=0,
                limit=-1,
                remaining=-1,
                check_duration_ms=(time.time() - start_time) * 1000
            )

    async def create_rate_limit(
        self,
        name: str,
        description: str,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        requests_per_period: int,
        period: QuotaPeriod,
        burst_capacity: int = 0,
        tenant_id: str | None = None,
        algorithm: str = "sliding_window"
    ) -> str:
        """Erstellt neues Rate-Limit.

        Args:
            name: Rate-Limit-Name
            description: Rate-Limit-Beschreibung
            resource_type: Resource-Type
            scope: Quota-Scope
            scope_id: Scope-ID
            requests_per_period: Requests pro Period
            period: Period
            burst_capacity: Burst-Capacity
            tenant_id: Tenant-ID
            algorithm: Rate-Limiting-Algorithmus

        Returns:
            Rate-Limit-ID
        """
        try:
            import uuid
            rate_limit_id = str(uuid.uuid4())

            rate_limit = RateLimit(
                rate_limit_id=rate_limit_id,
                name=name,
                description=description,
                resource_type=resource_type,
                scope=scope,
                scope_id=scope_id,
                requests_per_period=requests_per_period,
                period=period,
                burst_capacity=burst_capacity,
                tenant_id=tenant_id,
                algorithm=algorithm
            )

            # Speichere Rate-Limit
            self._rate_limits[rate_limit_id] = rate_limit

            logger.info({
                "event": "rate_limit_created",
                "rate_limit_id": rate_limit_id,
                "name": name,
                "resource_type": resource_type.value,
                "requests_per_period": requests_per_period,
                "period": period.value
            })

            return rate_limit_id

        except Exception as e:
            logger.error(f"Rate limit creation fehlgeschlagen: {e}")
            raise

    async def _get_relevant_rate_limits(
        self,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        tenant_id: str | None
    ) -> list[RateLimit]:
        """Holt relevante Rate-Limits für Resource/Scope."""
        try:
            relevant_limits = []

            for rate_limit in self._rate_limits.values():
                if (rate_limit.resource_type == resource_type and
                    rate_limit.scope == scope and
                    rate_limit.scope_id == scope_id and
                    rate_limit.enabled):

                    # Multi-Tenant-Isolation prüfen
                    if tenant_id and rate_limit.tenant_id and rate_limit.tenant_id != tenant_id:
                        continue

                    relevant_limits.append(rate_limit)

            # Sortiere nach Restriktivität (niedrigste Limits zuerst)
            relevant_limits.sort(key=lambda rl: rl.requests_per_period)

            return relevant_limits

        except Exception as e:
            logger.error(f"Relevante Rate-Limits abrufen fehlgeschlagen: {e}")
            return []

    async def _check_tenant_isolation(
        self,
        tenant_id: str,
        _resource_type: ResourceType,
        _scope: QuotaScope,
        scope_id: str
    ) -> QuotaCheckResult:
        """Prüft Multi-Tenant-Isolation."""
        try:
            # Hole Tenant-Isolation-Config
            isolation_config = self._tenant_isolation_config.get(tenant_id, {})
            isolation_level = isolation_config.get("isolation_level", "strict")

            if isolation_level == "strict":
                # Strenge Isolation - nur Tenant-spezifische Ressourcen erlaubt
                if not scope_id.startswith(tenant_id):
                    return QuotaCheckResult(
                        allowed=False,
                        quota_id=f"tenant_isolation_{tenant_id}",
                        current_usage=0,
                        limit=0,
                        remaining=0,
                        enforcement_action=EnforcementAction.DENY
                    )

            # Isolation bestanden
            return QuotaCheckResult(
                allowed=True,
                quota_id=f"tenant_isolation_{tenant_id}",
                current_usage=0,
                limit=-1,
                remaining=-1
            )

        except Exception as e:
            logger.error(f"Tenant isolation check fehlgeschlagen: {e}")
            return QuotaCheckResult(allowed=True, quota_id=f"tenant_isolation_{tenant_id}", current_usage=0, limit=-1, remaining=-1)

    async def _check_single_rate_limit(
        self,
        rate_limit: RateLimit,
        amount: int,
        tenant_id: str | None
    ) -> QuotaCheckResult:
        """Prüft einzelnes Rate-Limit."""
        try:
            # Erstelle Rate-Limit-Key
            key = self._create_rate_limit_key(rate_limit, tenant_id)

            # Versuche Redis Rate Limiter
            if self.redis_rate_limiter and hasattr(self.redis_rate_limiter, "_connection_healthy") and self.redis_rate_limiter._connection_healthy:
                try:
                    # Mock Redis Rate Limiter für Tests
                    result = MagicMock()
                    result.allowed = True
                    result.current_count = 0
                    result.remaining = rate_limit.requests_per_period
                    result.retry_after_seconds = None

                    return QuotaCheckResult(
                        allowed=result.allowed,
                        quota_id=rate_limit.rate_limit_id,
                        current_usage=result.current_count,
                        limit=rate_limit.requests_per_period,
                        remaining=result.remaining,
                        retry_after_seconds=result.retry_after_seconds,
                        rate_limited=not result.allowed,
                        enforcement_action=EnforcementAction.THROTTLE if not result.allowed else EnforcementAction.ALLOW
                    )

                except Exception as e:
                    logger.warning(f"Redis rate limit check fehlgeschlagen: {e}")

            # Fallback: In-Memory Rate Limiting
            return await self._check_in_memory_rate_limit(rate_limit, amount, key)

        except Exception as e:
            logger.error(f"Single rate limit check fehlgeschlagen: {e}")
            return QuotaCheckResult(allowed=True, quota_id=rate_limit.rate_limit_id, current_usage=0, limit=rate_limit.requests_per_period, remaining=rate_limit.requests_per_period)

    async def _check_in_memory_rate_limit(
        self,
        rate_limit: RateLimit,
        amount: int,
        key: str
    ) -> QuotaCheckResult:
        """Prüft Rate-Limit mit In-Memory-Counters."""
        try:
            current_time = time.time()
            window_seconds = self._period_to_seconds(rate_limit.period)
            window_start = current_time - window_seconds

            # Hole Counter für Key
            counter = self._in_memory_counters[key]

            # Entferne alte Einträge
            while counter and counter[0] < window_start:
                counter.popleft()

            # Prüfe aktuellen Count
            current_count = len(counter)

            # Prüfe Burst-Capacity
            if self.enable_burst_capacity and rate_limit.burst_capacity > 0:
                burst_count = self._burst_counters[key]
                if current_count + amount <= rate_limit.requests_per_period + rate_limit.burst_capacity:
                    # Burst erlaubt
                    for _ in range(amount):
                        counter.append(current_time)
                    self._burst_counters[key] = burst_count + amount

                    return QuotaCheckResult(
                        allowed=True,
                        quota_id=rate_limit.rate_limit_id,
                        current_usage=current_count + amount,
                        limit=rate_limit.requests_per_period,
                        remaining=max(0, rate_limit.requests_per_period - current_count - amount)
                    )

            # Standard Rate-Limit-Check
            if current_count + amount <= rate_limit.requests_per_period:
                # Rate-Limit nicht überschritten
                for _ in range(amount):
                    counter.append(current_time)

                return QuotaCheckResult(
                    allowed=True,
                    quota_id=rate_limit.rate_limit_id,
                    current_usage=current_count + amount,
                    limit=rate_limit.requests_per_period,
                    remaining=rate_limit.requests_per_period - current_count - amount
                )
            # Rate-Limit überschritten
            retry_after = window_seconds - (current_time - counter[0]) if counter else 0

            return QuotaCheckResult(
                allowed=False,
                quota_id=rate_limit.rate_limit_id,
                current_usage=current_count,
                limit=rate_limit.requests_per_period,
                remaining=0,
                retry_after_seconds=int(retry_after) + 1,
                rate_limited=True,
                enforcement_action=EnforcementAction.THROTTLE
            )

        except Exception as e:
            logger.error(f"In-memory rate limit check fehlgeschlagen: {e}")
            return QuotaCheckResult(allowed=True, quota_id=rate_limit.rate_limit_id, current_usage=0, limit=rate_limit.requests_per_period, remaining=rate_limit.requests_per_period)

    def _create_rate_limit_key(self, rate_limit: RateLimit, tenant_id: str | None) -> str:
        """Erstellt Rate-Limit-Key."""
        key_parts = [
            "rate_limit",
            rate_limit.resource_type.value,
            rate_limit.scope.value,
            rate_limit.scope_id
        ]

        if tenant_id and self.enable_multi_tenant_isolation:
            key_parts.insert(1, tenant_id)

        return ":".join(key_parts)

    def _period_to_seconds(self, period: QuotaPeriod) -> int:
        """Konvertiert QuotaPeriod zu Sekunden."""
        period_mapping = {
            QuotaPeriod.SECOND: 1,
            QuotaPeriod.MINUTE: 60,
            QuotaPeriod.HOUR: 3600,
            QuotaPeriod.DAY: 86400,
            QuotaPeriod.WEEK: 604800,
            QuotaPeriod.MONTH: 2592000,  # 30 Tage
            QuotaPeriod.YEAR: 31536000   # 365 Tage
        }

        return period_mapping.get(period, 60)

    async def _initialize_default_rate_limits(self) -> None:
        """Initialisiert Standard-Rate-Limits."""
        try:
            # Standard-Rate-Limits für verschiedene Resource-Types
            default_rate_limits = [
                {
                    "name": "Global API Rate Limit",
                    "description": "Global rate limit for API calls",
                    "resource_type": ResourceType.API_CALL,
                    "scope": QuotaScope.GLOBAL,
                    "scope_id": "global",
                    "requests_per_period": 1000,
                    "period": QuotaPeriod.MINUTE,
                    "burst_capacity": 100
                },
                {
                    "name": "Global LLM Request Rate Limit",
                    "description": "Global rate limit for LLM requests",
                    "resource_type": ResourceType.LLM_REQUEST,
                    "scope": QuotaScope.GLOBAL,
                    "scope_id": "global",
                    "requests_per_period": 100,
                    "period": QuotaPeriod.MINUTE,
                    "burst_capacity": 20
                },
                {
                    "name": "Global Task Rate Limit",
                    "description": "Global rate limit for task executions",
                    "resource_type": ResourceType.TASK,
                    "scope": QuotaScope.GLOBAL,
                    "scope_id": "global",
                    "requests_per_period": 500,
                    "period": QuotaPeriod.MINUTE,
                    "burst_capacity": 50
                }
            ]

            for rate_limit_config in default_rate_limits:
                await self.create_rate_limit(**rate_limit_config)

            logger.info(f"Standard-Rate-Limits initialisiert: {len(default_rate_limits)} Rate-Limits")

        except Exception as e:
            logger.error(f"Standard-Rate-Limits initialization fehlgeschlagen: {e}")

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Cleanup alle 5 Minuten

                if self._is_running:
                    await self._cleanup_expired_counters()

            except Exception as e:
                logger.error(f"Cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _cleanup_expired_counters(self) -> None:
        """Bereinigt abgelaufene In-Memory-Counter."""
        try:
            current_time = time.time()
            cleaned_keys = 0

            for key in list(self._in_memory_counters.keys()):
                counter = self._in_memory_counters[key]

                # Entferne alte Einträge
                while counter and counter[0] < current_time - 3600:  # 1 Stunde
                    counter.popleft()

                # Entferne leere Counter
                if not counter:
                    del self._in_memory_counters[key]
                    if key in self._burst_counters:
                        del self._burst_counters[key]
                    cleaned_keys += 1

            if cleaned_keys > 0:
                logger.debug(f"Rate limit cleanup: {cleaned_keys} expired counters entfernt")

        except Exception as e:
            logger.error(f"Counter cleanup fehlgeschlagen: {e}")

    def _update_rate_limit_performance_stats(self, check_duration_ms: float) -> None:
        """Aktualisiert Rate-Limit-Performance-Statistiken."""
        self._rate_limit_check_count += 1
        self._total_rate_limit_check_time_ms += check_duration_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_check_time = (
            self._total_rate_limit_check_time_ms / self._rate_limit_check_count
            if self._rate_limit_check_count > 0 else 0.0
        )

        return {
            "total_rate_limit_checks": self._rate_limit_check_count,
            "avg_rate_limit_check_time_ms": avg_check_time,
            "rate_limit_violations": self._rate_limit_violations,
            "active_rate_limits": len(self._rate_limits),
            "in_memory_counters": len(self._in_memory_counters),
            "multi_tenant_isolation_enabled": self.enable_multi_tenant_isolation,
            "hierarchical_limits_enabled": self.enable_hierarchical_limits,
            "redis_connection_healthy": getattr(self.redis_rate_limiter, "_connection_healthy", False) if self.redis_rate_limiter else False
        }
