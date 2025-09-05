# backend/quotas_limits/quota_middleware.py
"""Quota Enforcement Middleware für Keiko Personal Assistant

Implementiert FastAPI-Middleware für automatische Quota-Checks,
Integration mit Enhanced Security und Policy Engine.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger
from observability import trace_function

from .agent_rate_limiter import RateLimitWindow, agent_rate_limiter
from .budget_manager import BudgetType, CostCategory, budget_manager
from .core_quota_manager import QuotaScope, QuotaType, quota_manager
from .quota_monitoring import quota_monitor

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class QuotaConfig:
    """Konfiguration für Quota-Enforcement."""
    # Quota-Management
    quota_enabled: bool = True
    default_quota_checks: bool = True
    quota_fail_open: bool = False

    # Rate-Limiting
    rate_limiting_enabled: bool = True
    default_rate_limits: bool = True
    rate_limit_fail_open: bool = False

    # Budget-Management
    budget_tracking_enabled: bool = True
    auto_cost_calculation: bool = True
    budget_fail_open: bool = False

    # Monitoring
    monitoring_enabled: bool = True
    metrics_collection: bool = True
    alert_on_violations: bool = True

    # Integration
    security_integration: bool = True
    policy_integration: bool = True

    # Performance
    cache_quota_results: bool = True
    cache_ttl_seconds: int = 60
    async_processing: bool = True

    # Ausgeschlossene Pfade
    excluded_paths: set[str] = None

    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = {
                "/health", "/metrics", "/docs", "/openapi.json", "/redoc",
                "/quota/status", "/quota/health"
            }


@dataclass
class QuotaEnforcementResult:
    """Ergebnis der Quota-Enforcement."""
    allowed: bool
    quota_violations: list[str] = None
    rate_limit_violations: list[str] = None
    budget_violations: list[str] = None
    warnings: list[str] = None

    # Quota-Details
    quota_remaining: dict[str, float] = None
    rate_limit_remaining: dict[str, float] = None
    budget_remaining: dict[str, float] = None

    # Retry-Info
    retry_after_seconds: int | None = None

    # Performance
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.quota_violations is None:
            self.quota_violations = []
        if self.rate_limit_violations is None:
            self.rate_limit_violations = []
        if self.budget_violations is None:
            self.budget_violations = []
        if self.warnings is None:
            self.warnings = []
        if self.quota_remaining is None:
            self.quota_remaining = {}
        if self.rate_limit_remaining is None:
            self.rate_limit_remaining = {}
        if self.budget_remaining is None:
            self.budget_remaining = {}

    @property
    def has_violations(self) -> bool:
        """Prüft, ob Verletzungen vorhanden sind."""
        return (len(self.quota_violations) > 0 or
                len(self.rate_limit_violations) > 0 or
                len(self.budget_violations) > 0)


class QuotaEnforcementMiddleware(BaseHTTPMiddleware):
    """Middleware für automatische Quota-Enforcement."""

    def __init__(self, app, config: QuotaConfig):
        """Initialisiert Quota Enforcement Middleware.

        Args:
            app: FastAPI-App
            config: Quota-Konfiguration
        """
        super().__init__(app)
        self.config = config

        # Statistiken
        self._requests_processed = 0
        self._requests_blocked = 0
        self._quota_violations = 0
        self._rate_limit_violations = 0
        self._budget_violations = 0

        # Cache für Performance
        self._enforcement_cache: dict[str, QuotaEnforcementResult] = {}
        self._cache_timestamps: dict[str, float] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet Request mit Quota-Enforcement."""
        start_time = time.time()

        try:
            # Prüfe ausgeschlossene Pfade
            if request.url.path in self.config.excluded_paths:
                return await call_next(request)

            self._requests_processed += 1

            # Extrahiere Request-Context
            request_context = await self._extract_request_context(request)

            # Pre-Request Quota-Enforcement
            enforcement_result = await self._enforce_quotas(request_context, "pre_request")

            if not enforcement_result.allowed and not self._should_fail_open(enforcement_result):
                self._requests_blocked += 1
                return self._create_quota_error_response(enforcement_result)

            # Führe Request aus
            response = await call_next(request)

            # Post-Request Quota-Enforcement (für Verbrauch)
            await self._consume_quotas(request_context, response)

            # Füge Quota-Headers hinzu
            self._add_quota_headers(response, enforcement_result)

            # Monitoring
            if self.config.monitoring_enabled:
                processing_time = (time.time() - start_time) * 1000
                quota_monitor.metrics_collector.record_quota_check(
                    processing_time,
                    cache_hit=self._was_cache_hit(request_context)
                )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Quota-Enforcement-Fehler: {e}")

            if self.config.quota_fail_open:
                return await call_next(request)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Quota enforcement error"
            )

    async def _extract_request_context(self, request: Request) -> dict[str, Any]:
        """Extrahiert Request-Context für Quota-Checks."""
        # Extrahiere User/Agent/Tenant-Info aus Request
        user_id = None
        agent_id = None
        tenant_id = None

        # Aus Headers
        agent_id = request.headers.get("X-Agent-ID")
        tenant_id = request.headers.get("X-Tenant-ID")
        user_id = request.headers.get("X-User-ID")

        # Aus Request-State (falls von Security-Middleware gesetzt)
        if hasattr(request.state, "principal"):
            principal = getattr(request.state, "principal", {})
            user_id = user_id or principal.get("id")
            agent_id = agent_id or principal.get("agent_id")

        if hasattr(request.state, "tenant_context"):
            tenant_context = getattr(request.state, "tenant_context", {})
            tenant_id = tenant_id or tenant_context.get("tenant_id")

        # Request-Details
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)

        # Content-Length für Cost-Calculation
        content_length = int(request.headers.get("content-length", 0))

        return {
            "user_id": user_id,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "method": method,
            "path": path,
            "query_params": query_params,
            "content_length": content_length,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.now(UTC)
        }

    @trace_function("quota_middleware.enforce")
    async def _enforce_quotas(
        self,
        context: dict[str, Any],
        phase: str
    ) -> QuotaEnforcementResult:
        """Führt Quota-Enforcement durch."""
        start_time = time.time()
        cache_key = None  # Initialize to avoid unbound variable

        # Cache-Check
        if self.config.cache_quota_results:
            cache_key = self._generate_cache_key(context, phase)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

        result = QuotaEnforcementResult(allowed=True)

        try:
            # 1. Quota-Checks
            if self.config.quota_enabled:
                await self._check_quotas(context, result)

            # 2. Rate-Limiting
            if self.config.rate_limiting_enabled:
                await self._check_rate_limits(context, result)

            # 3. Budget-Checks
            if self.config.budget_tracking_enabled and phase == "pre_request":
                await self._check_budgets(context, result)

            # Bestimme Overall-Erlaubnis
            result.allowed = not result.has_violations

            # Cache Result
            if self.config.cache_quota_results and cache_key is not None:
                self._cache_result(cache_key, result)

            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time

            return result

        except Exception as e:
            logger.exception(f"Quota-Enforcement fehlgeschlagen: {e}")

            # Fallback-Verhalten
            if self.config.quota_fail_open:
                result.allowed = True
                result.warnings.append(f"Quota check failed: {e!s}")
            else:
                result.allowed = False
                result.quota_violations.append(f"Quota enforcement error: {e!s}")

            return result

    async def _check_quotas(self, context: dict[str, Any], result: QuotaEnforcementResult) -> None:
        """Prüft Quota-Limits."""
        agent_id = context.get("agent_id")
        tenant_id = context.get("tenant_id")

        if not agent_id and not tenant_id:
            return

        # Standard-Quota-Checks
        quota_checks = [
            ("request_rate", QuotaType.REQUEST_RATE, agent_id or tenant_id),
            ("operation_count", QuotaType.OPERATION_COUNT, agent_id or tenant_id),
        ]

        for quota_name, quota_type, scope_id in quota_checks:
            if not scope_id:
                continue

            # Finde passende Policy
            policies = quota_manager.get_policies(QuotaScope.AGENT if agent_id else QuotaScope.TENANT)
            matching_policy = None

            for policy in policies:
                if policy.quota_type == quota_type:
                    matching_policy = policy
                    break

            if matching_policy:
                quota_result = await quota_manager.check_quota(
                    matching_policy.policy_id,
                    scope_id,
                    1.0,  # Standard-Request-Menge
                    context
                )

                if not quota_result.allowed:
                    result.quota_violations.extend([v.message for v in quota_result.violations])
                    self._quota_violations += len(quota_result.violations)

                    if quota_result.retry_after_seconds:
                        result.retry_after_seconds = quota_result.retry_after_seconds

                result.quota_remaining[quota_name] = quota_result.remaining

    async def _check_rate_limits(self, context: dict[str, Any], result: QuotaEnforcementResult) -> None:
        """Prüft Rate-Limits."""
        agent_id = context.get("agent_id")

        if not agent_id:
            return

        # Rate-Limit-Check
        rate_limit_result = await agent_rate_limiter.check_agent_rate_limit(
            agent_id=agent_id,
            window=RateLimitWindow.PER_MINUTE,
            requested_amount=1.0,
            context=context
        )

        if not rate_limit_result.allowed:
            result.rate_limit_violations.append(
                f"Rate limit exceeded for agent {agent_id}: {rate_limit_result.current_usage}/{rate_limit_result.limit}"
            )
            self._rate_limit_violations += 1

            if rate_limit_result.retry_after_seconds:
                result.retry_after_seconds = rate_limit_result.retry_after_seconds

        result.rate_limit_remaining["per_minute"] = rate_limit_result.remaining

    async def _check_budgets(self, context: dict[str, Any], result: QuotaEnforcementResult) -> None:
        """Prüft Budget-Limits."""
        agent_id = context.get("agent_id")

        if not agent_id:
            return

        # Berechne geschätzte Kosten
        estimated_cost = self._calculate_estimated_cost(context)

        # Finde Agent-Budgets
        agent_budgets = budget_manager.get_agent_budgets(agent_id)

        for budget_info in agent_budgets:
            if budget_info["is_exhausted"]:
                result.budget_violations.append(
                    f"Budget exhausted for agent {agent_id}: {budget_info['budget_type']}"
                )
                self._budget_violations += 1
            elif budget_info["available_amount"] < estimated_cost:
                result.budget_violations.append(
                    f"Insufficient budget for agent {agent_id}: {budget_info['budget_type']}"
                )
                self._budget_violations += 1

            result.budget_remaining[budget_info["budget_type"]] = budget_info["available_amount"]

    async def _consume_quotas(self, context: dict[str, Any], response: Response) -> None:
        """Verbraucht Quotas nach Request-Ausführung."""
        if not self.config.async_processing:
            await self._consume_quotas_sync(context, response)
        else:
            # Asynchrone Verarbeitung
            asyncio.create_task(self._consume_quotas_async(context, response))

    async def _consume_quotas_sync(self, context: dict[str, Any], response: Response) -> None:
        """Synchroner Quota-Verbrauch."""
        agent_id = context.get("agent_id")

        if not agent_id:
            return

        try:
            # Berechne tatsächliche Kosten
            actual_cost = self._calculate_actual_cost(context, response)

            # Verbrauche Budget
            if self.config.budget_tracking_enabled and actual_cost.total_cost > 0:
                # Finde passendes Budget
                agent_budgets = budget_manager.get_agent_budgets(agent_id)

                for budget_info in agent_budgets:
                    if budget_info["budget_type"] == "api_calls":
                        await budget_manager.charge_budget(
                            budget_info["budget_id"],
                            actual_cost,
                            agent_id
                        )
                        break

            # Verbrauche Quota
            # (Quota-Verbrauch erfolgt bereits im Check)

        except Exception as e:
            logger.exception(f"Quota-Consumption fehlgeschlagen: {e}")

    async def _consume_quotas_async(self, context: dict[str, Any], response: Response) -> None:
        """Asynchroner Quota-Verbrauch."""
        # Gleiche Logik wie sync, aber ohne await auf Response
        await self._consume_quotas_sync(context, response)

    def _calculate_estimated_cost(self, context: dict[str, Any]) -> float:
        """Berechnet geschätzte Kosten für Request."""
        # Vereinfachte Kosten-Schätzung
        base_cost = 0.001  # Basis-Kosten pro Request

        # Zusätzliche Kosten basierend auf Content-Length
        content_length = context.get("content_length", 0)
        data_cost = (content_length / 1024) * 0.0001  # Pro KB

        return base_cost + data_cost

    def _calculate_actual_cost(self, context: dict[str, Any], response: Response) -> Any:
        """Berechnet tatsächliche Kosten nach Request."""
        from decimal import Decimal

        # Berechne Kosten basierend auf Request und Response
        operation_type = f"{context['method']} {context['path']}"

        # Basis-Kosten
        Decimal("0.001")

        # Variable Kosten basierend auf Response-Size
        response_size = len(response.body) if hasattr(response, "body") else 0
        Decimal(str(response_size / 1024)) * Decimal("0.0001")  # Pro KB

        return budget_manager.cost_tracker.calculate_operation_cost(
            operation_type=operation_type,
            category=CostCategory.API_CALLS,
            api_calls_count=1,
            data_volume_mb=response_size / (1024 * 1024)
        )

    def _should_fail_open(self, result: QuotaEnforcementResult) -> bool:
        """Bestimmt, ob bei Violations fail-open verhalten werden soll."""
        if result.quota_violations and not self.config.quota_fail_open:
            return False
        if result.rate_limit_violations and not self.config.rate_limit_fail_open:
            return False
        return not (result.budget_violations and not self.config.budget_fail_open)

    def _create_quota_error_response(self, result: QuotaEnforcementResult) -> Response:
        """Erstellt Error-Response für Quota-Violations."""
        error_details = {
            "error": "Quota limit exceeded",
            "quota_violations": result.quota_violations,
            "rate_limit_violations": result.rate_limit_violations,
            "budget_violations": result.budget_violations
        }

        if result.retry_after_seconds:
            error_details["retry_after"] = result.retry_after_seconds

        # Bestimme HTTP-Status-Code
        if result.rate_limit_violations:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif result.budget_violations:
            status_code = status.HTTP_402_PAYMENT_REQUIRED
        else:
            status_code = status.HTTP_403_FORBIDDEN

        response = Response(
            content=str(error_details),
            status_code=status_code,
            media_type="application/json"
        )

        # Füge Retry-After-Header hinzu
        if result.retry_after_seconds:
            response.headers["Retry-After"] = str(result.retry_after_seconds)

        return response

    def _add_quota_headers(self, response: Response, result: QuotaEnforcementResult) -> None:
        """Fügt Quota-Headers zur Response hinzu."""
        # Quota-Remaining-Headers
        for quota_name, remaining in result.quota_remaining.items():
            response.headers[f"X-Quota-Remaining-{quota_name.replace('_', '-')}"] = str(remaining)

        # Rate-Limit-Headers
        for window, remaining in result.rate_limit_remaining.items():
            response.headers[f"X-RateLimit-Remaining-{window}"] = str(remaining)

        # Budget-Headers
        for budget_type, remaining in result.budget_remaining.items():
            response.headers[f"X-Budget-Remaining-{budget_type.replace('_', '-')}"] = str(remaining)

        # Processing-Time-Header
        response.headers["X-Quota-Processing-Time"] = f"{result.processing_time_ms:.2f}ms"

    def _generate_cache_key(self, context: dict[str, Any], phase: str) -> str:
        """Generiert Cache-Key für Enforcement-Result."""
        key_parts = [
            phase,
            context.get("agent_id", ""),
            context.get("tenant_id", ""),
            context.get("method", ""),
            context.get("path", "")
        ]
        return ":".join(key_parts)

    def _get_cached_result(self, cache_key: str) -> QuotaEnforcementResult | None:
        """Gibt gecachtes Enforcement-Result zurück."""
        if cache_key not in self._enforcement_cache:
            return None

        # Prüfe TTL
        cache_time = self._cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > self.config.cache_ttl_seconds:
            del self._enforcement_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._enforcement_cache[cache_key]

    def _cache_result(self, cache_key: str, result: QuotaEnforcementResult) -> None:
        """Cached Enforcement-Result."""
        self._enforcement_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

    def _was_cache_hit(self, context: dict[str, Any]) -> bool:
        """Prüft, ob Request ein Cache-Hit war."""
        # Vereinfachte Cache-Hit-Detection
        return False  # In Produktion würde hier echte Cache-Hit-Detection stehen

    def get_enforcement_statistics(self) -> dict[str, Any]:
        """Gibt Enforcement-Statistiken zurück."""
        return {
            "requests_processed": self._requests_processed,
            "requests_blocked": self._requests_blocked,
            "quota_violations": self._quota_violations,
            "rate_limit_violations": self._rate_limit_violations,
            "budget_violations": self._budget_violations,
            "block_rate": self._requests_blocked / max(self._requests_processed, 1),
            "cache_size": len(self._enforcement_cache),
            "config": {
                "quota_enabled": self.config.quota_enabled,
                "rate_limiting_enabled": self.config.rate_limiting_enabled,
                "budget_tracking_enabled": self.config.budget_tracking_enabled,
                "monitoring_enabled": self.config.monitoring_enabled
            }
        }


# Decorator für Quota-Compliance
def require_quota_compliance(
    quota_types: list[QuotaType] | None = None,
    rate_limit_check: bool = True,
    budget_check: bool = True,
    fail_open: bool = False
):
    """Decorator für Quota-Compliance-Checks.

    Args:
        quota_types: Zu prüfende Quota-Typen
        rate_limit_check: Rate-Limit-Check aktivieren
        budget_check: Budget-Check aktivieren
        fail_open: Bei Fehlern durchlassen
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Request aus Argumenten extrahieren
            request = None
            for arg in args:
                if hasattr(arg, "method") and hasattr(arg, "url"):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                if fail_open:
                    return await func(*args, **kwargs)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No request context for quota check"
                )

            # Quota-Checks durchführen
            try:
                # Vereinfachte Quota-Checks für Decorator
                agent_id = request.headers.get("X-Agent-ID")

                if rate_limit_check and agent_id:
                    rate_result = await agent_rate_limiter.check_agent_rate_limit(agent_id)
                    if not rate_result.allowed:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded"
                        )

                if budget_check and agent_id:
                    # Berechne geschätzte Kosten für diesen Request
                    context = {
                        "agent_id": agent_id,
                        "request": request,
                        "endpoint": str(request.url.path),
                        "method": request.method
                    }
                    
                    # Erstelle temporäres Result-Objekt für Budget-Check
                    from .models import QuotaEnforcementResult
                    temp_result = QuotaEnforcementResult()
                    
                    # Importiere Budget-Manager
                    from .budget_manager import budget_manager
                    
                    # Berechne geschätzte Kosten
                    estimated_cost = 0.01  # Default-Schätzung
                    agent_budgets = budget_manager.get_agent_budgets(agent_id)
                    
                    for budget_info in agent_budgets:
                        if budget_info["is_exhausted"]:
                            raise HTTPException(
                                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                                detail=f"Budget exhausted for agent {agent_id}: {budget_info['budget_type']}"
                            )
                        elif budget_info["available_amount"] < estimated_cost:
                            raise HTTPException(
                                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                                detail=f"Insufficient budget for agent {agent_id}: {budget_info['budget_type']}"
                            )

                return await func(*args, **kwargs)

            except HTTPException:
                raise
            except Exception as e:
                if fail_open:
                    logger.warning(f"Quota-Check fehlgeschlagen, fail-open: {e}")
                    return await func(*args, **kwargs)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Quota check error"
                )

        return wrapper
    return decorator


def enforce_agent_quotas(agent_id: str, quota_types: list[QuotaType] | None = None):
    """Decorator für Agent-spezifische Quota-Enforcement."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Agent-Quota-Checks
            if quota_types:
                for quota_type in quota_types:
                    # Prüfe Quota für Agent und Typ
                    policy_id = f"agent_{agent_id}_{quota_type.value}"
                    result = await quota_manager.check_quota(
                        policy_id=policy_id,
                        scope_id=agent_id,
                        requested_amount=1.0
                    )

                    if not result.allowed:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Quota exceeded for {quota_type.value}"
                        )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def enforce_budget_limits(budget_types: list[BudgetType] | None = None):
    """Decorator für Budget-Limit-Enforcement."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Budget-Checks
            if budget_types:
                for budget_type in budget_types:
                    # Finde Agent-ID aus Argumenten
                    agent_id = kwargs.get("agent_id") or (args[0] if args else None)
                    if not agent_id:
                        continue

                    # Prüfe Budget-Verfügbarkeit
                    agent_budgets = budget_manager.get_agent_budgets(agent_id)
                    budget_available = any(
                        b.get("budget_type") == budget_type.value and
                        b.get("status") == "active" and
                        b.get("remaining_amount", 0) > 0
                        for b in agent_budgets
                    )

                    if not budget_available:
                        raise HTTPException(
                            status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            detail=f"Insufficient budget for {budget_type.value}"
                        )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


async def check_api_quota(user_id: str, quota_type: str) -> None:
    """Prüft API-Quota für Benutzer und Quota-Typ.

    Args:
        user_id: Benutzer-ID für Quota-Prüfung
        quota_type: Typ der Quota (z.B. "agent_lifecycle")

    Raises:
        HTTPException: Bei Quota-Überschreitung
    """
    try:
        # Erstelle Policy-ID basierend auf Benutzer und Quota-Typ
        policy_id = f"user_{quota_type}"

        # Prüfe Quota über Quota-Manager
        result = await quota_manager.check_quota(
            policy_id=policy_id,
            scope_id=user_id,
            requested_amount=1.0
        )

        if not result.allowed:
            # Bestimme Retry-After aus Violations
            retry_after = None
            if result.violations:
                for violation in result.violations:
                    if hasattr(violation, "retry_after_seconds") and violation.retry_after_seconds:
                        retry_after = violation.retry_after_seconds
                        break

            # Erstelle detaillierte Fehlermeldung
            detail = f"API quota exceeded for {quota_type}"
            if result.violations:
                detail += f": {result.violations[0].message}"

            # Erstelle HTTPException mit Retry-After Header
            headers = {}
            if retry_after:
                headers["Retry-After"] = str(int(retry_after))

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=detail,
                headers=headers
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"API-Quota-Prüfung fehlgeschlagen für {user_id}/{quota_type}: {e}")
        # Fail-open: Bei Fehlern erlauben wir die Anfrage
        logger.warning(f"API-Quota-Prüfung fehlgeschlagen, fail-open für {user_id}/{quota_type}")
        return
