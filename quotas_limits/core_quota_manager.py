# backend/quotas_limits/core_quota_manager.py
"""Core Quota Manager für Keiko Personal Assistant

Implementiert zentrale Quota-Verwaltung, -Enforcement und -Monitoring
mit asynchroner Verarbeitung und Circuit-Breaker-Pattern.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .base_manager import BaseManager, ManagerConfig
from .constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
)
from .utils import (
    calculate_time_difference_seconds,
    generate_uuid,
    get_current_timestamp,
)

logger = get_logger(__name__)


class QuotaType(str, Enum):
    """Typen von Quotas."""
    REQUEST_RATE = "request_rate"
    OPERATION_COUNT = "operation_count"
    DATA_VOLUME = "data_volume"
    COMPUTE_TIME = "compute_time"
    MEMORY_USAGE = "memory_usage"
    STORAGE_QUOTA = "storage_quota"
    BANDWIDTH_LIMIT = "bandwidth_limit"
    CONCURRENT_OPERATIONS = "concurrent_operations"


class QuotaScope(str, Enum):
    """Scope von Quotas."""
    GLOBAL = "global"
    TENANT = "tenant"
    AGENT = "agent"
    CAPABILITY = "capability"
    USER = "user"
    OPERATION = "operation"


class QuotaViolationType(str, Enum):
    """Typen von Quota-Verletzungen."""
    HARD_LIMIT_EXCEEDED = "hard_limit_exceeded"
    SOFT_LIMIT_EXCEEDED = "soft_limit_exceeded"
    BURST_LIMIT_EXCEEDED = "burst_limit_exceeded"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class QuotaPolicy:
    """Definition einer Quota-Policy."""
    policy_id: str
    name: str
    description: str
    quota_type: QuotaType
    scope: QuotaScope

    # Limits
    hard_limit: float
    soft_limit: float | None = None
    burst_limit: float | None = None

    # Zeitfenster
    window_seconds: int = 3600  # 1 Stunde Standard

    # Priorität und Vererbung
    priority: int = 100
    inheritable: bool = True

    # Gültigkeit
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = None
    tags: set[str] = field(default_factory=set)

    def is_valid(self) -> bool:
        """Prüft, ob Policy gültig ist."""
        if not self.enabled:
            return False

        now = datetime.now(UTC)

        if self.valid_from and now < self.valid_from:
            return False

        return not (self.valid_until and now > self.valid_until)

    @property
    def effective_soft_limit(self) -> float:
        """Gibt effektives Soft-Limit zurück."""
        return self.soft_limit or (self.hard_limit * 0.8)


@dataclass
class QuotaUsage:
    """Aktuelle Quota-Nutzung."""
    quota_id: str
    scope_id: str
    current_usage: float
    peak_usage: float
    usage_count: int
    window_start: datetime
    window_end: datetime
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def reset_window(self, window_seconds: int) -> None:
        """Setzt Nutzungsfenster zurück."""
        now = datetime.now(UTC)
        self.window_start = now
        self.window_end = now + timedelta(seconds=window_seconds)
        self.current_usage = 0.0
        self.peak_usage = 0.0
        self.usage_count = 0
        self.last_updated = now

    def add_usage(self, amount: float) -> None:
        """Fügt Nutzung hinzu."""
        self.current_usage += amount
        self.peak_usage = max(self.peak_usage, self.current_usage)
        self.usage_count += 1
        self.last_updated = datetime.now(UTC)

    def is_window_expired(self) -> bool:
        """Prüft, ob Zeitfenster abgelaufen ist."""
        return datetime.now(UTC) > self.window_end


@dataclass
class QuotaViolation:
    """Repräsentiert eine Quota-Verletzung."""
    violation_type: QuotaViolationType
    quota_id: str
    scope_id: str
    current_usage: float
    limit_exceeded: float
    severity: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuotaCheckResult:
    """Ergebnis einer Quota-Prüfung."""
    allowed: bool
    quota_id: str
    scope_id: str
    current_usage: float
    remaining: float
    limit: float
    violations: list[QuotaViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retry_after_seconds: int | None = None

    @property
    def has_violations(self) -> bool:
        """Prüft, ob Verletzungen vorhanden sind."""
        return len(self.violations) > 0

    @property
    def usage_percentage(self) -> float:
        """Gibt Nutzung in Prozent zurück."""
        if self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100


@dataclass
class QuotaAllocation:
    """Quota-Zuteilung für spezifischen Scope."""
    allocation_id: str
    quota_id: str
    scope: QuotaScope
    scope_id: str
    allocated_amount: float
    used_amount: float = 0.0
    reserved_amount: float = 0.0
    priority: int = 100
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    @property
    def available_amount(self) -> float:
        """Gibt verfügbare Menge zurück."""
        return self.allocated_amount - self.used_amount - self.reserved_amount

    @property
    def utilization_percentage(self) -> float:
        """Gibt Auslastung in Prozent zurück."""
        if self.allocated_amount == 0:
            return 0.0
        return (self.used_amount / self.allocated_amount) * 100

    def is_expired(self) -> bool:
        """Prüft, ob Allocation abgelaufen ist."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at


class QuotaEnforcer(ABC):
    """Basis-Klasse für Quota-Enforcer."""

    @abstractmethod
    async def check_quota(
        self,
        policy: QuotaPolicy,
        scope_id: str,
        requested_amount: float,
        context: dict[str, Any] | None = None
    ) -> QuotaCheckResult:
        """Prüft Quota für gegebene Policy."""

    @abstractmethod
    async def consume_quota(
        self,
        policy: QuotaPolicy,
        scope_id: str,
        amount: float,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Verbraucht Quota-Menge."""


class DefaultQuotaEnforcer(QuotaEnforcer):
    """Standard-Quota-Enforcer."""

    def __init__(self):
        """Initialisiert Default Quota Enforcer."""
        self._usage_cache: dict[str, QuotaUsage] = {}
        self._cache_lock = asyncio.Lock()

    async def check_quota(
        self,
        policy: QuotaPolicy,
        scope_id: str,
        requested_amount: float,
        context: dict[str, Any] | None = None
    ) -> QuotaCheckResult:
        """Prüft Quota für gegebene Policy."""
        async with self._cache_lock:
            usage_key = f"{policy.policy_id}:{scope_id}"

            # Hole oder erstelle Usage
            if usage_key not in self._usage_cache:
                self._usage_cache[usage_key] = QuotaUsage(
                    quota_id=policy.policy_id,
                    scope_id=scope_id,
                    current_usage=0.0,
                    peak_usage=0.0,
                    usage_count=0,
                    window_start=datetime.now(UTC),
                    window_end=datetime.now(UTC) + timedelta(seconds=policy.window_seconds)
                )

            usage = self._usage_cache[usage_key]

            # Prüfe Zeitfenster
            if usage.is_window_expired():
                usage.reset_window(policy.window_seconds)

            # Berechne neue Nutzung
            new_usage = usage.current_usage + requested_amount
            violations = []
            warnings = []

            # Prüfe Hard-Limit
            if new_usage > policy.hard_limit:
                violation = QuotaViolation(
                    violation_type=QuotaViolationType.HARD_LIMIT_EXCEEDED,
                    quota_id=policy.policy_id,
                    scope_id=scope_id,
                    current_usage=new_usage,
                    limit_exceeded=policy.hard_limit,
                    severity="critical",
                    message=f"Hard limit exceeded: {new_usage} > {policy.hard_limit}"
                )
                violations.append(violation)

            # Prüfe Soft-Limit
            elif new_usage > policy.effective_soft_limit:
                if new_usage > policy.hard_limit:
                    violation = QuotaViolation(
                        violation_type=QuotaViolationType.SOFT_LIMIT_EXCEEDED,
                        quota_id=policy.policy_id,
                        scope_id=scope_id,
                        current_usage=new_usage,
                        limit_exceeded=policy.effective_soft_limit,
                        severity="warning",
                        message=f"Soft limit exceeded: {new_usage} > {policy.effective_soft_limit}"
                    )
                    violations.append(violation)
                else:
                    warnings.append(f"Approaching quota limit: {new_usage:.1f}/{policy.hard_limit}")

            # Prüfe Burst-Limit
            if policy.burst_limit and new_usage > policy.burst_limit:
                violation = QuotaViolation(
                    violation_type=QuotaViolationType.BURST_LIMIT_EXCEEDED,
                    quota_id=policy.policy_id,
                    scope_id=scope_id,
                    current_usage=new_usage,
                    limit_exceeded=policy.burst_limit,
                    severity="high",
                    message=f"Burst limit exceeded: {new_usage} > {policy.burst_limit}"
                )
                violations.append(violation)

            # Bestimme Retry-After
            retry_after = None
            if violations:
                remaining_window = (usage.window_end - datetime.now(UTC)).total_seconds()
                retry_after = max(1, int(remaining_window))

            return QuotaCheckResult(
                allowed=len(violations) == 0,
                quota_id=policy.policy_id,
                scope_id=scope_id,
                current_usage=usage.current_usage,
                remaining=max(0, policy.hard_limit - usage.current_usage),
                limit=policy.hard_limit,
                violations=violations,
                warnings=warnings,
                retry_after_seconds=retry_after
            )

    async def consume_quota(
        self,
        policy: QuotaPolicy,
        scope_id: str,
        amount: float,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Verbraucht Quota-Menge."""
        async with self._cache_lock:
            usage_key = f"{policy.policy_id}:{scope_id}"

            if usage_key in self._usage_cache:
                usage = self._usage_cache[usage_key]

                # Prüfe Zeitfenster
                if usage.is_window_expired():
                    usage.reset_window(policy.window_seconds)

                # Verbrauche Quota
                usage.add_usage(amount)
                return True

            return False


class CircuitBreaker:
    """Circuit Breaker für Quota-Checks."""

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: int = CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """Führt Funktion mit Circuit Breaker aus."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Prüft, ob Reset-Versuch gemacht werden soll."""
        if not self.last_failure_time:
            return False

        return calculate_time_difference_seconds(self.last_failure_time) > self.recovery_timeout

    def _on_success(self):
        """Behandelt erfolgreiche Ausführung."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Behandelt fehlgeschlagene Ausführung."""
        self.failure_count += 1
        self.last_failure_time = get_current_timestamp()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class QuotaManager(BaseManager):
    """Zentrale Quota-Verwaltung."""

    def __init__(self, config: ManagerConfig | None = None):
        """Initialisiert Quota Manager."""
        super().__init__(config)

        self._policies: dict[str, QuotaPolicy] = {}
        self._allocations: dict[str, QuotaAllocation] = {}
        self._enforcers: dict[QuotaType, QuotaEnforcer] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Standard-Enforcer registrieren
        default_enforcer = DefaultQuotaEnforcer()
        for quota_type in QuotaType:
            self._enforcers[quota_type] = default_enforcer
            self._circuit_breakers[quota_type.value] = CircuitBreaker()

    def get_manager_type(self) -> str:
        """Gibt Manager-Typ zurück."""
        return "QuotaManager"

    def register_policy(self, policy: QuotaPolicy) -> None:
        """Registriert Quota-Policy."""
        self._policies[policy.policy_id] = policy
        if self.cache:
            asyncio.create_task(self.cache.clear())
        logger.info(f"Quota-Policy registriert: {policy.policy_id} ({policy.quota_type.value})")

    def register_enforcer(self, quota_type: QuotaType, enforcer: QuotaEnforcer) -> None:
        """Registriert Quota-Enforcer."""
        self._enforcers[quota_type] = enforcer
        logger.info(f"Quota-Enforcer registriert für {quota_type.value}")

    @trace_function("quota.check")
    async def check_quota(
        self,
        policy_id: str,
        scope_id: str,
        requested_amount: float = 1.0,
        context: dict[str, Any] | None = None
    ) -> QuotaCheckResult:
        """Prüft Quota für gegebene Policy."""
        cache_key = f"{policy_id}:{scope_id}:{requested_amount}"

        return await self.execute_operation(
            operation_name="check_quota",
            operation_func=self._check_quota_impl,
            policy_id=policy_id,
            scope_id=scope_id,
            requested_amount=requested_amount,
            context=context,
            cache_key=cache_key
        )

    async def _check_quota_impl(
        self,
        policy_id: str,
        scope_id: str,
        requested_amount: float,
        context: dict[str, Any] | None
    ) -> QuotaCheckResult:
        """Implementierung der Quota-Prüfung."""
        policy = self._policies.get(policy_id)
        if not policy or not policy.is_valid():
            return QuotaCheckResult(
                allowed=False,
                quota_id=policy_id,
                scope_id=scope_id,
                current_usage=0.0,
                remaining=0.0,
                limit=0.0,
                violations=[QuotaViolation(
                    violation_type=QuotaViolationType.HARD_LIMIT_EXCEEDED,
                    quota_id=policy_id,
                    scope_id=scope_id,
                    current_usage=0.0,
                    limit_exceeded=0.0,
                    severity="critical",
                    message="Policy not found or invalid"
                )]
            )

        enforcer = self._enforcers.get(policy.quota_type)
        if not enforcer:
            raise ValueError(f"No enforcer for quota type: {policy.quota_type}")

        circuit_breaker = self._circuit_breakers[policy.quota_type.value]

        return await circuit_breaker.call(
            enforcer.check_quota,
            policy, scope_id, requested_amount, context
        )


    async def consume_quota(
        self,
        policy_id: str,
        scope_id: str,
        amount: float = 1.0,
        context: dict[str, Any] | None = None
    ) -> bool:
        """Verbraucht Quota-Menge."""
        return await self.execute_operation(
            operation_name="consume_quota",
            operation_func=self._consume_quota_impl,
            policy_id=policy_id,
            scope_id=scope_id,
            amount=amount,
            context=context
        )

    async def _consume_quota_impl(
        self,
        policy_id: str,
        scope_id: str,
        amount: float,
        context: dict[str, Any] | None
    ) -> bool:
        """Implementierung des Quota-Verbrauchs."""
        policy = self._policies.get(policy_id)
        if not policy or not policy.is_valid():
            return False

        enforcer = self._enforcers.get(policy.quota_type)
        if not enforcer:
            return False

        circuit_breaker = self._circuit_breakers[policy.quota_type.value]

        return await circuit_breaker.call(
            enforcer.consume_quota,
            policy, scope_id, amount, context
        )

    def create_allocation(
        self,
        quota_id: str,
        scope: QuotaScope,
        scope_id: str,
        amount: float,
        priority: int = 100,
        expires_at: datetime | None = None
    ) -> str:
        """Erstellt Quota-Allocation."""
        allocation_id = generate_uuid()
        allocation = QuotaAllocation(
            allocation_id=allocation_id,
            quota_id=quota_id,
            scope=scope,
            scope_id=scope_id,
            allocated_amount=amount,
            priority=priority,
            expires_at=expires_at
        )

        self._allocations[allocation_id] = allocation
        logger.info(f"Quota-Allocation erstellt: {allocation_id}")
        return allocation_id

    def get_allocations(self, scope_id: str) -> list[QuotaAllocation]:
        """Gibt Allocations für Scope zurück."""
        return [
            allocation for allocation in self._allocations.values()
            if allocation.scope_id == scope_id and not allocation.is_expired()
        ]

    def get_policies(self, scope: QuotaScope | None = None) -> list[QuotaPolicy]:
        """Gibt Quota-Policies zurück."""
        if scope:
            return [policy for policy in self._policies.values() if policy.scope == scope]
        return list(self._policies.values())

    def remove_policy(self, policy_id: str) -> bool:
        """Entfernt Quota-Policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            if self.cache:
                asyncio.create_task(self.cache.clear())
            logger.info(f"Quota-Policy entfernt: {policy_id}")
            return True
        return False

    def get_quota_statistics(self) -> dict[str, Any]:
        """Gibt Quota-spezifische Statistiken zurück."""
        base_status = self.get_status()
        quota_stats = {
            "total_policies": len(self._policies),
            "total_allocations": len(self._allocations),
            "registered_enforcers": len(self._enforcers),
            "circuit_breakers": {
                cb_type: cb.state for cb_type, cb in self._circuit_breakers.items()
            }
        }

        # Kombiniere mit Base-Manager-Statistiken
        base_status.update(quota_stats)
        return base_status


# Globale Quota Manager Instanz
quota_manager = QuotaManager()
