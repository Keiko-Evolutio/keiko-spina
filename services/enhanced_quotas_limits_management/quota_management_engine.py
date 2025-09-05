# backend/services/enhanced_quotas_limits_management/quota_management_engine.py
"""Enhanced Quota Management Engine.

Implementiert Enterprise-Grade Resource-Management mit Integration
aller bestehenden Quota-Systeme und Enhanced Security Integration.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime

from kei_logging import get_logger
from quotas_limits import QuotaManager, quota_manager
from services.enhanced_security_integration import (
    EnhancedSecurityIntegrationEngine,
    SecurityContext,
    SecurityLevel,
)

from .data_models import (
    EnforcementAction,
    QuotaCheckResult,
    QuotaPerformanceMetrics,
    QuotaPeriod,
    QuotaScope,
    QuotaStatus,
    QuotaUsage,
    QuotaViolation,
    ResourceQuota,
    ResourceType,
)
from .quota_analytics_engine import QuotaAnalyticsEngine
from .quota_enforcement_engine import QuotaEnforcementEngine
from .rate_limiting_engine import RateLimitingEngine

logger = get_logger(__name__)


class EnhancedQuotaManagementEngine:
    """Enhanced Quota Management Engine für Enterprise-Grade Resource-Management."""

    def __init__(
        self,
        security_integration_engine: EnhancedSecurityIntegrationEngine,
        existing_quota_manager: QuotaManager | None = None,
        rate_limiting_engine: RateLimitingEngine | None = None,
        enforcement_engine: QuotaEnforcementEngine | None = None,
        analytics_engine: QuotaAnalyticsEngine | None = None
    ):
        """Initialisiert Enhanced Quota Management Engine.

        Args:
            security_integration_engine: Enhanced Security Integration Engine
            existing_quota_manager: Bestehender Quota Manager
            rate_limiting_engine: Rate Limiting Engine
            enforcement_engine: Quota Enforcement Engine
            analytics_engine: Quota Analytics Engine
        """
        self.security_integration_engine = security_integration_engine
        self.existing_quota_manager = existing_quota_manager or quota_manager
        self.rate_limiting_engine = rate_limiting_engine or RateLimitingEngine()
        self.enforcement_engine = enforcement_engine or QuotaEnforcementEngine()
        self.analytics_engine = analytics_engine or QuotaAnalyticsEngine()

        # Enhanced Quota-Konfiguration
        self.enable_enhanced_quotas = True
        self.enable_hierarchical_limits = True
        self.enable_security_integration = True
        self.quota_check_timeout_ms = 50.0  # < 50ms SLA

        # Quota-Storage
        self._resource_quotas: dict[str, ResourceQuota] = {}
        self._quota_usage: dict[str, QuotaUsage] = {}
        self._quota_violations: list[QuotaViolation] = []

        # Hierarchical Quota-Tree
        self._quota_hierarchy: dict[str, list[str]] = defaultdict(list)

        # Performance-Tracking
        self._quota_check_count = 0
        self._total_quota_check_time_ms = 0.0
        self._enforcement_count = 0
        self._violation_count = 0

        # Cache für Performance-Optimierung
        self._quota_cache: dict[str, QuotaCheckResult] = {}
        self._cache_ttl_seconds = 60
        self._cache_timestamps: dict[str, float] = {}

        # Background-Tasks
        self._usage_tracking_task: asyncio.Task | None = None
        self._cache_cleanup_task: asyncio.Task | None = None
        self._analytics_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Enhanced Quota Management Engine initialisiert")

    async def start(self) -> None:
        """Startet Enhanced Quota Management Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Komponenten
        await self.rate_limiting_engine.start()
        await self.enforcement_engine.start()
        await self.analytics_engine.start()

        # Starte Background-Tasks
        self._usage_tracking_task = asyncio.create_task(self._usage_tracking_loop())
        self._cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._analytics_task = asyncio.create_task(self._analytics_loop())

        # Initialisiere Standard-Quotas
        await self._initialize_default_quotas()

        logger.info("Enhanced Quota Management Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Enhanced Quota Management Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._usage_tracking_task:
            self._usage_tracking_task.cancel()
        if self._cache_cleanup_task:
            self._cache_cleanup_task.cancel()
        if self._analytics_task:
            self._analytics_task.cancel()

        await asyncio.gather(
            self._usage_tracking_task,
            self._cache_cleanup_task,
            self._analytics_task,
            return_exceptions=True
        )

        # Stoppe Komponenten
        await self.rate_limiting_engine.stop()
        await self.enforcement_engine.stop()
        await self.analytics_engine.stop()

        logger.info("Enhanced Quota Management Engine gestoppt")

    async def check_quota_with_security(
        self,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        amount: int = 1,
        security_context: SecurityContext | None = None
    ) -> QuotaCheckResult:
        """Führt Quota-Check mit Security-Integration durch.

        Args:
            resource_type: Resource-Type
            scope: Quota-Scope
            scope_id: Scope-ID
            amount: Anzahl der zu konsumierenden Ressourcen
            security_context: Security-Context

        Returns:
            Quota-Check-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "quota_check_with_security_started",
                "resource_type": resource_type.value,
                "scope": scope.value,
                "scope_id": scope_id,
                "amount": amount
            })

            # 1. Security-Check (falls Security-Context vorhanden)
            if security_context and self.enable_security_integration:
                security_result = await self.security_integration_engine.perform_comprehensive_security_check(
                    security_context=security_context,
                    resource_type=resource_type,
                    resource_id=f"{scope.value}:{scope_id}",
                    action="consume"
                )

                if not security_result.is_secure:
                    logger.warning({
                        "event": "quota_security_check_failed",
                        "scope_id": scope_id,
                        "security_score": security_result.security_score
                    })

                    return QuotaCheckResult(
                        allowed=False,
                        quota_id=f"{resource_type.value}_{scope.value}_{scope_id}",
                        current_usage=0,
                        limit=0,
                        remaining=0,
                        enforcement_action=EnforcementAction.DENY,
                        check_duration_ms=(time.time() - start_time) * 1000
                    )

            # 2. Hole relevante Quotas
            quotas = await self._get_relevant_quotas(resource_type, scope, scope_id)

            if not quotas:
                # Keine Quotas definiert - erlauben mit Warnung
                logger.warning(f"Keine Quotas definiert für {resource_type.value}:{scope.value}:{scope_id}")
                return QuotaCheckResult(
                    allowed=True,
                    quota_id=f"{resource_type.value}_{scope.value}_{scope_id}",
                    current_usage=0,
                    limit=-1,  # Unlimited
                    remaining=-1,
                    check_duration_ms=(time.time() - start_time) * 1000
                )

            # 3. Prüfe Cache
            cache_key = f"{resource_type.value}_{scope.value}_{scope_id}_{amount}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                cached_result.check_duration_ms = (time.time() - start_time) * 1000
                return cached_result

            # 4. Hierarchical Quota-Check
            check_result = await self._perform_hierarchical_quota_check(
                quotas, resource_type, scope, scope_id, amount
            )

            # 5. Rate-Limiting-Check
            if check_result.allowed:
                rate_limit_result = await self.rate_limiting_engine.check_rate_limit(
                    resource_type=resource_type,
                    scope=scope,
                    scope_id=scope_id,
                    amount=amount
                )

                if not rate_limit_result.allowed:
                    check_result.allowed = False
                    check_result.rate_limited = True
                    check_result.retry_after_seconds = rate_limit_result.retry_after_seconds
                    check_result.enforcement_action = EnforcementAction.THROTTLE

            # 6. Enforcement (falls nicht erlaubt)
            if not check_result.allowed:
                await self.enforcement_engine.enforce_quota_violation(
                    quota_check_result=check_result,
                    security_context=security_context
                )

            # 7. Usage-Tracking (falls erlaubt)
            if check_result.allowed:
                await self._track_usage(resource_type, scope, scope_id, amount)

            # 8. Cache Result
            self._cache_result(cache_key, check_result)

            # Performance-Tracking
            check_duration_ms = (time.time() - start_time) * 1000
            check_result.check_duration_ms = check_duration_ms
            self._update_quota_performance_stats(check_duration_ms)

            logger.debug({
                "event": "quota_check_with_security_completed",
                "allowed": check_result.allowed,
                "current_usage": check_result.current_usage,
                "limit": check_result.limit,
                "check_duration_ms": check_duration_ms
            })

            return check_result

        except Exception as e:
            logger.error(f"Quota check mit Security fehlgeschlagen: {e}")

            # Fallback: Verweigern bei Fehler
            return QuotaCheckResult(
                allowed=False,
                quota_id=f"{resource_type.value}_{scope.value}_{scope_id}",
                current_usage=0,
                limit=0,
                remaining=0,
                enforcement_action=EnforcementAction.DENY,
                check_duration_ms=(time.time() - start_time) * 1000
            )

    async def create_resource_quota(
        self,
        name: str,
        description: str,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        limit: int,
        period: QuotaPeriod,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        enforcement_action: EnforcementAction = EnforcementAction.DENY,
        parent_quota_id: str | None = None
    ) -> str:
        """Erstellt neue Resource-Quota.

        Args:
            name: Quota-Name
            description: Quota-Beschreibung
            resource_type: Resource-Type
            scope: Quota-Scope
            scope_id: Scope-ID
            limit: Quota-Limit
            period: Quota-Period
            security_level: Security-Level
            enforcement_action: Enforcement-Action
            parent_quota_id: Parent-Quota-ID

        Returns:
            Quota-ID
        """
        try:
            import uuid
            quota_id = str(uuid.uuid4())

            quota = ResourceQuota(
                quota_id=quota_id,
                name=name,
                description=description,
                resource_type=resource_type,
                scope=scope,
                scope_id=scope_id,
                limit=limit,
                period=period,
                security_level=security_level,
                enforcement_action=enforcement_action,
                parent_quota_id=parent_quota_id
            )

            # Speichere Quota
            self._resource_quotas[quota_id] = quota

            # Aktualisiere Hierarchie
            if parent_quota_id:
                self._quota_hierarchy[parent_quota_id].append(quota_id)
                if parent_quota_id in self._resource_quotas:
                    self._resource_quotas[parent_quota_id].child_quota_ids.add(quota_id)

            logger.info({
                "event": "resource_quota_created",
                "quota_id": quota_id,
                "name": name,
                "resource_type": resource_type.value,
                "scope": scope.value,
                "limit": limit
            })

            return quota_id

        except Exception as e:
            logger.error(f"Resource Quota creation fehlgeschlagen: {e}")
            raise

    async def _get_relevant_quotas(
        self,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str
    ) -> list[ResourceQuota]:
        """Holt relevante Quotas für Resource/Scope."""
        try:
            relevant_quotas = []

            for quota in self._resource_quotas.values():
                if (quota.resource_type == resource_type and
                    quota.scope == scope and
                    quota.scope_id == scope_id and
                    quota.status == QuotaStatus.ACTIVE):
                    relevant_quotas.append(quota)

            return relevant_quotas

        except Exception as e:
            logger.error(f"Relevante Quotas abrufen fehlgeschlagen: {e}")
            return []

    async def _perform_hierarchical_quota_check(
        self,
        quotas: list[ResourceQuota],
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        amount: int
    ) -> QuotaCheckResult:
        """Führt hierarchical Quota-Check durch."""
        try:
            # Finde primäre Quota
            primary_quota = quotas[0] if quotas else None

            if not primary_quota:
                return QuotaCheckResult(
                    allowed=False,
                    quota_id=f"{resource_type.value}_{scope.value}_{scope_id}",
                    current_usage=0,
                    limit=0,
                    remaining=0,
                    enforcement_action=EnforcementAction.DENY
                )

            # Hole aktuelle Usage
            current_usage = await self._get_current_usage(
                resource_type, scope, scope_id, primary_quota.period
            )

            # Prüfe Limit
            new_usage = current_usage + amount
            allowed = new_usage <= primary_quota.limit
            remaining = max(0, primary_quota.limit - current_usage)

            # Prüfe Parent-Quotas (falls hierarchical)
            if self.enable_hierarchical_limits and primary_quota.parent_quota_id:
                parent_check = await self._check_parent_quotas(
                    primary_quota.parent_quota_id, amount
                )
                if not parent_check.allowed:
                    allowed = False

            return QuotaCheckResult(
                allowed=allowed,
                quota_id=primary_quota.quota_id,
                current_usage=current_usage,
                limit=primary_quota.limit,
                remaining=remaining,
                enforcement_action=primary_quota.enforcement_action if not allowed else EnforcementAction.ALLOW
            )

        except Exception as e:
            logger.error(f"Hierarchical quota check fehlgeschlagen: {e}")
            return QuotaCheckResult(
                allowed=False,
                quota_id=f"{resource_type.value}_{scope.value}_{scope_id}",
                current_usage=0,
                limit=0,
                remaining=0,
                enforcement_action=EnforcementAction.DENY
            )

    async def _get_current_usage(
        self,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        period: QuotaPeriod
    ) -> int:
        """Holt aktuelle Usage für Resource/Scope/Period."""
        try:
            # Vereinfachte Implementation - in Realität würde dies aus Datenbank/Cache kommen
            usage_key = f"{resource_type.value}_{scope.value}_{scope_id}_{period.value}"

            if usage_key in self._quota_usage:
                return self._quota_usage[usage_key].current_usage

            return 0

        except Exception as e:
            logger.error(f"Current usage abrufen fehlgeschlagen: {e}")
            return 0

    async def _check_parent_quotas(self, parent_quota_id: str, amount: int) -> QuotaCheckResult:
        """Prüft Parent-Quotas rekursiv."""
        try:
            parent_quota = self._resource_quotas.get(parent_quota_id)
            if not parent_quota:
                return QuotaCheckResult(allowed=True, quota_id=parent_quota_id, current_usage=0, limit=-1, remaining=-1)

            # Prüfe Parent-Quota
            current_usage = await self._get_current_usage(
                parent_quota.resource_type,
                parent_quota.scope,
                parent_quota.scope_id,
                parent_quota.period
            )

            new_usage = current_usage + amount
            allowed = new_usage <= parent_quota.limit

            if not allowed:
                return QuotaCheckResult(
                    allowed=False,
                    quota_id=parent_quota_id,
                    current_usage=current_usage,
                    limit=parent_quota.limit,
                    remaining=max(0, parent_quota.limit - current_usage),
                    enforcement_action=parent_quota.enforcement_action
                )

            # Prüfe Grandparent-Quotas
            if parent_quota.parent_quota_id:
                return await self._check_parent_quotas(parent_quota.parent_quota_id, amount)

            return QuotaCheckResult(allowed=True, quota_id=parent_quota_id, current_usage=current_usage, limit=parent_quota.limit, remaining=parent_quota.limit - current_usage)

        except Exception as e:
            logger.error(f"Parent quota check fehlgeschlagen: {e}")
            return QuotaCheckResult(allowed=False, quota_id=parent_quota_id, current_usage=0, limit=0, remaining=0)

    async def _track_usage(
        self,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        amount: int
    ) -> None:
        """Trackt Usage für Analytics."""
        try:
            usage_key = f"{resource_type.value}_{scope.value}_{scope_id}"

            if usage_key not in self._quota_usage:
                import uuid
                self._quota_usage[usage_key] = QuotaUsage(
                    usage_id=str(uuid.uuid4()),
                    quota_id=usage_key,
                    resource_type=resource_type,
                    scope=scope,
                    scope_id=scope_id,
                    current_usage=0,
                    peak_usage=0,
                    average_usage=0.0,
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow()
                )

            # Aktualisiere Usage
            usage = self._quota_usage[usage_key]
            usage.current_usage += amount
            usage.peak_usage = max(usage.peak_usage, usage.current_usage)
            usage.request_count += 1

        except Exception as e:
            logger.error(f"Usage tracking fehlgeschlagen: {e}")

    def _get_cached_result(self, cache_key: str) -> QuotaCheckResult | None:
        """Holt Cached Quota-Check-Result."""
        try:
            if cache_key not in self._quota_cache:
                return None

            # Prüfe TTL
            if cache_key in self._cache_timestamps:
                age = time.time() - self._cache_timestamps[cache_key]
                if age > self._cache_ttl_seconds:
                    del self._quota_cache[cache_key]
                    del self._cache_timestamps[cache_key]
                    return None

            return self._quota_cache[cache_key]

        except Exception as e:
            logger.error(f"Cache lookup fehlgeschlagen: {e}")
            return None

    def _cache_result(self, cache_key: str, result: QuotaCheckResult) -> None:
        """Cached Quota-Check-Result."""
        try:
            self._quota_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

            # Memory-Limit prüfen
            if len(self._quota_cache) > 10000:
                # Entferne älteste Einträge
                oldest_keys = sorted(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k]
                )[:1000]

                for key in oldest_keys:
                    del self._quota_cache[key]
                    del self._cache_timestamps[key]

        except Exception as e:
            logger.error(f"Cache storage fehlgeschlagen: {e}")

    async def _initialize_default_quotas(self) -> None:
        """Initialisiert Standard-Quotas."""
        try:
            # Standard-Quotas für verschiedene Resource-Types
            default_quotas = [
                {
                    "name": "Global Agent Quota",
                    "description": "Global limit for agent instances",
                    "resource_type": ResourceType.AGENT,
                    "scope": QuotaScope.GLOBAL,
                    "scope_id": "global",
                    "limit": 1000,
                    "period": QuotaPeriod.HOUR
                },
                {
                    "name": "Global Task Quota",
                    "description": "Global limit for task executions",
                    "resource_type": ResourceType.TASK,
                    "scope": QuotaScope.GLOBAL,
                    "scope_id": "global",
                    "limit": 10000,
                    "period": QuotaPeriod.HOUR
                },
                {
                    "name": "Global API Call Quota",
                    "description": "Global limit for API calls",
                    "resource_type": ResourceType.API_CALL,
                    "scope": QuotaScope.GLOBAL,
                    "scope_id": "global",
                    "limit": 100000,
                    "period": QuotaPeriod.HOUR
                }
            ]

            for quota_config in default_quotas:
                await self.create_resource_quota(**quota_config)

            logger.info(f"Standard-Quotas initialisiert: {len(default_quotas)} Quotas")

        except Exception as e:
            logger.error(f"Standard-Quotas initialization fehlgeschlagen: {e}")

    async def _usage_tracking_loop(self) -> None:
        """Background-Loop für Usage-Tracking."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Update alle Minute

                if self._is_running:
                    await self._update_usage_statistics()

            except Exception as e:
                logger.error(f"Usage tracking loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _cache_cleanup_loop(self) -> None:
        """Background-Loop für Cache-Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Cleanup alle 5 Minuten

                if self._is_running:
                    await self._cleanup_expired_cache()

            except Exception as e:
                logger.error(f"Cache cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _analytics_loop(self) -> None:
        """Background-Loop für Analytics."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Analytics alle Stunde

                if self._is_running:
                    await self.analytics_engine.generate_quota_analytics()

            except Exception as e:
                logger.error(f"Analytics loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _update_usage_statistics(self) -> None:
        """Aktualisiert Usage-Statistiken."""
        try:
            # TODO: Implementiere Usage-Statistics-Update - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
            logger.debug("Usage statistics aktualisiert")

        except Exception as e:
            logger.error(f"Usage statistics update fehlgeschlagen: {e}")

    async def _cleanup_expired_cache(self) -> None:
        """Bereinigt abgelaufene Cache-Einträge."""
        try:
            current_time = time.time()
            expired_keys = []

            for key, timestamp in self._cache_timestamps.items():
                if current_time - timestamp > self._cache_ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._quota_cache[key]
                del self._cache_timestamps[key]

            if expired_keys:
                logger.debug(f"Cache cleanup: {len(expired_keys)} expired entries entfernt")

        except Exception as e:
            logger.error(f"Cache cleanup fehlgeschlagen: {e}")

    def _update_quota_performance_stats(self, check_duration_ms: float) -> None:
        """Aktualisiert Quota-Performance-Statistiken."""
        self._quota_check_count += 1
        self._total_quota_check_time_ms += check_duration_ms

    def get_performance_metrics(self) -> QuotaPerformanceMetrics:
        """Gibt Performance-Metriken zurück."""
        avg_quota_check_time = (
            self._total_quota_check_time_ms / self._quota_check_count
            if self._quota_check_count > 0 else 0.0
        )

        return QuotaPerformanceMetrics(
            total_quota_checks=self._quota_check_count,
            avg_quota_check_time_ms=avg_quota_check_time,
            total_enforcements=self._enforcement_count,
            meets_quota_sla=avg_quota_check_time < self.quota_check_timeout_ms,
            quota_sla_threshold_ms=self.quota_check_timeout_ms,
            cache_hit_rate=len(self._quota_cache) / max(1, self._quota_check_count),
            sample_count=self._quota_check_count
        )
