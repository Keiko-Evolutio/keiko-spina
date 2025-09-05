# backend/services/enhanced_quotas_limits_management/quota_analytics_engine.py
"""Quota Analytics Engine für Usage Tracking und Analytics.

Implementiert umfassende Quota-Analytics mit Trend-Analysis,
Predictive Analytics und Real-time Dashboards.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .data_models import QuotaAnalytics, QuotaScope, QuotaViolation, ResourceType

logger = get_logger(__name__)


class QuotaAnalyticsEngine:
    """Quota Analytics Engine für Usage Tracking und Analytics."""

    def __init__(self):
        """Initialisiert Quota Analytics Engine."""
        # Analytics-Konfiguration
        self.enable_real_time_analytics = True
        self.enable_predictive_analytics = True
        self.enable_trend_analysis = True
        self.analytics_retention_days = 90

        # Analytics-Storage
        self._quota_analytics: dict[str, list[QuotaAnalytics]] = defaultdict(list)
        self._usage_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._violation_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Real-time Metrics
        self._real_time_usage: dict[str, dict[str, Any]] = {}
        self._performance_metrics: dict[str, list[float]] = defaultdict(list)

        # Trend-Analysis-Cache
        self._trend_cache: dict[str, dict[str, Any]] = {}
        self._trend_cache_ttl = 300  # 5 Minuten
        self._trend_cache_timestamps: dict[str, float] = {}

        # Performance-Tracking
        self._analytics_generation_count = 0
        self._total_analytics_time_ms = 0.0

        # Background-Tasks
        self._analytics_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._real_time_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Quota Analytics Engine initialisiert")

    async def start(self) -> None:
        """Startet Quota Analytics Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._real_time_task = asyncio.create_task(self._real_time_loop())

        logger.info("Quota Analytics Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Quota Analytics Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._analytics_task:
            self._analytics_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._real_time_task:
            self._real_time_task.cancel()

        await asyncio.gather(
            self._analytics_task,
            self._cleanup_task,
            self._real_time_task,
            return_exceptions=True
        )

        logger.info("Quota Analytics Engine gestoppt")

    async def track_quota_usage(
        self,
        quota_id: str,
        resource_type: ResourceType,
        scope: QuotaScope,
        scope_id: str,
        usage_amount: int,
        response_time_ms: float = 0.0,
        success: bool = True
    ) -> None:
        """Trackt Quota-Usage für Analytics.

        Args:
            quota_id: Quota-ID
            resource_type: Resource-Type
            scope: Quota-Scope
            scope_id: Scope-ID
            usage_amount: Usage-Amount
            response_time_ms: Response-Zeit
            success: Erfolg-Status
        """
        try:
            current_time = datetime.utcnow()

            # Erstelle Usage-Metric
            usage_metric = {
                "timestamp": current_time,
                "quota_id": quota_id,
                "resource_type": resource_type.value,
                "scope": scope.value,
                "scope_id": scope_id,
                "usage_amount": usage_amount,
                "response_time_ms": response_time_ms,
                "success": success
            }

            # Speichere Usage-Metric
            self._usage_metrics[quota_id].append(usage_metric)

            # Aktualisiere Real-time Usage
            if quota_id not in self._real_time_usage:
                self._real_time_usage[quota_id] = {
                    "current_usage": 0,
                    "peak_usage": 0,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "avg_response_time_ms": 0.0,
                    "last_updated": current_time
                }

            real_time = self._real_time_usage[quota_id]
            real_time["current_usage"] += usage_amount
            real_time["peak_usage"] = max(real_time["peak_usage"], real_time["current_usage"])
            real_time["total_requests"] += 1

            if success:
                real_time["successful_requests"] += 1
            else:
                real_time["failed_requests"] += 1

            # Aktualisiere Average Response Time
            total_response_time = real_time["avg_response_time_ms"] * (real_time["total_requests"] - 1)
            real_time["avg_response_time_ms"] = (total_response_time + response_time_ms) / real_time["total_requests"]
            real_time["last_updated"] = current_time

            # Performance-Tracking
            self._performance_metrics[quota_id].append(response_time_ms)

            # Limitiere Metrics-Größe
            if len(self._usage_metrics[quota_id]) > 10000:
                self._usage_metrics[quota_id] = self._usage_metrics[quota_id][-10000:]

            if len(self._performance_metrics[quota_id]) > 1000:
                self._performance_metrics[quota_id] = self._performance_metrics[quota_id][-1000:]

        except Exception as e:
            logger.error(f"Quota usage tracking fehlgeschlagen: {e}")

    async def track_quota_violation(
        self,
        violation: QuotaViolation,
        enforcement_action_taken: bool = False
    ) -> None:
        """Trackt Quota-Violation für Analytics.

        Args:
            violation: Quota-Violation
            enforcement_action_taken: Enforcement-Action durchgeführt
        """
        try:
            violation_metric = {
                "timestamp": violation.violation_timestamp,
                "violation_id": violation.violation_id,
                "quota_id": violation.quota_id,
                "resource_type": violation.resource_type.value,
                "scope": violation.scope.value,
                "scope_id": violation.scope_id,
                "limit_value": violation.limit_value,
                "actual_value": violation.actual_value,
                "excess_amount": violation.excess_amount,
                "enforcement_action": violation.enforcement_action.value,
                "action_taken": enforcement_action_taken,
                "user_id": violation.user_id,
                "request_id": violation.request_id
            }

            # Speichere Violation-Metric
            self._violation_metrics[violation.quota_id].append(violation_metric)

            # Limitiere Metrics-Größe
            if len(self._violation_metrics[violation.quota_id]) > 1000:
                self._violation_metrics[violation.quota_id] = self._violation_metrics[violation.quota_id][-1000:]

        except Exception as e:
            logger.error(f"Quota violation tracking fehlgeschlagen: {e}")

    async def generate_quota_analytics(
        self,
        quota_id: str | None = None,
        period_hours: int = 24
    ) -> list[QuotaAnalytics]:
        """Generiert Quota-Analytics für Zeitraum.

        Args:
            quota_id: Quota-ID (alle falls None)
            period_hours: Zeitraum in Stunden

        Returns:
            Liste von Quota-Analytics
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "quota_analytics_generation_started",
                "quota_id": quota_id,
                "period_hours": period_hours
            })

            analytics_results = []

            # Bestimme Quota-IDs
            quota_ids = [quota_id] if quota_id else list(self._usage_metrics.keys())

            for qid in quota_ids:
                analytics = await self._generate_single_quota_analytics(qid, period_hours)
                if analytics:
                    analytics_results.append(analytics)

            # Speichere Analytics
            for analytics in analytics_results:
                self._quota_analytics[analytics.quota_id].append(analytics)

                # Limitiere Analytics-Größe
                if len(self._quota_analytics[analytics.quota_id]) > 100:
                    self._quota_analytics[analytics.quota_id] = self._quota_analytics[analytics.quota_id][-100:]

            # Performance-Tracking
            analytics_time_ms = (time.time() - start_time) * 1000
            self._update_analytics_performance_stats(analytics_time_ms)

            logger.debug({
                "event": "quota_analytics_generation_completed",
                "analytics_count": len(analytics_results),
                "analytics_time_ms": analytics_time_ms
            })

            return analytics_results

        except Exception as e:
            logger.error(f"Quota analytics generation fehlgeschlagen: {e}")
            return []

    async def _generate_single_quota_analytics(
        self,
        quota_id: str,
        period_hours: int
    ) -> QuotaAnalytics | None:
        """Generiert Analytics für einzelne Quota."""
        try:
            current_time = datetime.utcnow()
            period_start = current_time - timedelta(hours=period_hours)

            # Hole Usage-Metrics für Zeitraum
            usage_metrics = [
                m for m in self._usage_metrics.get(quota_id, [])
                if m["timestamp"] >= period_start
            ]

            if not usage_metrics:
                return None

            # Hole Violation-Metrics für Zeitraum
            violation_metrics = [
                v for v in self._violation_metrics.get(quota_id, [])
                if v["timestamp"] >= period_start
            ]

            # Berechne Statistiken
            total_requests = len(usage_metrics)
            successful_requests = sum(1 for m in usage_metrics if m["success"])
            failed_requests = total_requests - successful_requests

            response_times = [m["response_time_ms"] for m in usage_metrics if m["response_time_ms"] > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            p95_response_time = self._calculate_percentile(response_times, 95) if response_times else 0.0
            p99_response_time = self._calculate_percentile(response_times, 99) if response_times else 0.0

            # Berechne Usage-Statistiken
            usage_amounts = [m["usage_amount"] for m in usage_metrics]
            peak_usage = max(usage_amounts) if usage_amounts else 0
            average_usage = sum(usage_amounts) / len(usage_amounts) if usage_amounts else 0.0

            # Berechne Violation-Statistiken
            violation_count = len(violation_metrics)
            violation_rate = violation_count / total_requests if total_requests > 0 else 0.0

            throttled_requests = sum(1 for v in violation_metrics if v["enforcement_action"] == "throttle")
            denied_requests = sum(1 for v in violation_metrics if v["enforcement_action"] == "deny")

            # Trend-Analysis
            usage_trend = await self._analyze_usage_trend(quota_id, usage_metrics)
            predicted_exhaustion = await self._predict_quota_exhaustion(quota_id, usage_metrics)

            # Quota-Utilization (benötigt Limit-Information)
            quota_utilization = 0.0  # TODO: Implementiere mit echten Quota-Limits - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114

            # Erstelle Analytics
            import uuid
            analytics = QuotaAnalytics(
                analytics_id=str(uuid.uuid4()),
                quota_id=quota_id,
                period_start=period_start,
                period_end=current_time,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                throttled_requests=throttled_requests,
                denied_requests=denied_requests,
                avg_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                peak_usage=peak_usage,
                average_usage=average_usage,
                quota_utilization=quota_utilization,
                usage_trend=usage_trend,
                predicted_exhaustion=predicted_exhaustion,
                violation_count=violation_count,
                violation_rate=violation_rate
            )

            return analytics

        except Exception as e:
            logger.error(f"Single quota analytics generation fehlgeschlagen für {quota_id}: {e}")
            return None

    async def _analyze_usage_trend(
        self,
        quota_id: str,
        usage_metrics: list[dict[str, Any]]
    ) -> str:
        """Analysiert Usage-Trend."""
        try:
            if len(usage_metrics) < 10:
                return "stable"

            # Cache-Check
            cache_key = f"trend_{quota_id}"
            if cache_key in self._trend_cache:
                cache_age = time.time() - self._trend_cache_timestamps.get(cache_key, 0)
                if cache_age < self._trend_cache_ttl:
                    return self._trend_cache[cache_key]["trend"]

            # Berechne Trend basierend auf Usage-Amounts über Zeit
            time_points = [(m["timestamp"], m["usage_amount"]) for m in usage_metrics[-50:]]  # Letzten 50 Punkte
            time_points.sort(key=lambda x: x[0])

            if len(time_points) < 5:
                return "stable"

            # Einfache Trend-Berechnung
            first_half = time_points[:len(time_points)//2]
            second_half = time_points[len(time_points)//2:]

            first_avg = sum(p[1] for p in first_half) / len(first_half)
            second_avg = sum(p[1] for p in second_half) / len(second_half)

            change_ratio = (second_avg - first_avg) / first_avg if first_avg > 0 else 0

            if change_ratio > 0.2:
                trend = "increasing"
            elif change_ratio < -0.2:
                trend = "decreasing"
            elif abs(change_ratio) > 0.1:
                trend = "volatile"
            else:
                trend = "stable"

            # Cache Result
            self._trend_cache[cache_key] = {"trend": trend, "change_ratio": change_ratio}
            self._trend_cache_timestamps[cache_key] = time.time()

            return trend

        except Exception as e:
            logger.error(f"Usage trend analysis fehlgeschlagen: {e}")
            return "stable"

    async def _predict_quota_exhaustion(
        self,
        _quota_id: str,
        usage_metrics: list[dict[str, Any]]
    ) -> datetime | None:
        """Vorhersage der Quota-Erschöpfung."""
        try:
            if not self.enable_predictive_analytics or len(usage_metrics) < 20:
                return None

            # Einfache lineare Extrapolation
            recent_metrics = usage_metrics[-20:]  # Letzten 20 Datenpunkte

            if not recent_metrics:
                return None

            # Berechne durchschnittliche Usage-Rate
            time_span_hours = (recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"]).total_seconds() / 3600
            total_usage = sum(m["usage_amount"] for m in recent_metrics)

            if time_span_hours <= 0:
                return None

            usage_rate_per_hour = total_usage / time_span_hours

            # TODO: Implementiere mit echten Quota-Limits - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
            # Placeholder: Annahme eines Limits von 10000
            assumed_limit = 10000
            current_usage = sum(m["usage_amount"] for m in usage_metrics[-10:])  # Aktuelle Usage

            if usage_rate_per_hour <= 0:
                return None

            remaining_quota = max(0, assumed_limit - current_usage)
            hours_to_exhaustion = remaining_quota / usage_rate_per_hour

            if hours_to_exhaustion > 0 and hours_to_exhaustion < 168:  # Innerhalb einer Woche
                return datetime.utcnow() + timedelta(hours=hours_to_exhaustion)

            return None

        except Exception as e:
            logger.error(f"Quota exhaustion prediction fehlgeschlagen: {e}")
            return None

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Berechnet Percentile für Liste von Werten."""
        try:
            if not values:
                return 0.0

            sorted_values = sorted(values)
            index = int((percentile / 100.0) * len(sorted_values))
            index = min(index, len(sorted_values) - 1)

            return sorted_values[index]

        except Exception as e:
            logger.error(f"Percentile calculation fehlgeschlagen: {e}")
            return 0.0

    async def get_real_time_dashboard_data(self) -> dict[str, Any]:
        """Holt Real-time Dashboard-Daten."""
        try:
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "quotas": {},
                "global_stats": {
                    "total_quotas": len(self._real_time_usage),
                    "total_requests": sum(q["total_requests"] for q in self._real_time_usage.values()),
                    "total_violations": sum(len(v) for v in self._violation_metrics.values()),
                    "avg_response_time_ms": 0.0
                }
            }

            # Berechne globale Average Response Time
            all_response_times = []
            for quota_data in self._real_time_usage.values():
                if quota_data["avg_response_time_ms"] > 0:
                    all_response_times.append(quota_data["avg_response_time_ms"])

            if all_response_times:
                dashboard_data["global_stats"]["avg_response_time_ms"] = sum(all_response_times) / len(all_response_times)

            # Quota-spezifische Daten
            for quota_id, quota_data in self._real_time_usage.items():
                dashboard_data["quotas"][quota_id] = {
                    "current_usage": quota_data["current_usage"],
                    "peak_usage": quota_data["peak_usage"],
                    "total_requests": quota_data["total_requests"],
                    "success_rate": quota_data["successful_requests"] / max(1, quota_data["total_requests"]),
                    "avg_response_time_ms": quota_data["avg_response_time_ms"],
                    "last_updated": quota_data["last_updated"].isoformat(),
                    "recent_violations": len([
                        v for v in self._violation_metrics.get(quota_id, [])
                        if v["timestamp"] > datetime.utcnow() - timedelta(hours=1)
                    ])
                }

            return dashboard_data

        except Exception as e:
            logger.error(f"Real-time dashboard data generation fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def _analytics_loop(self) -> None:
        """Background-Loop für Analytics-Generation."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Analytics alle Stunde

                if self._is_running:
                    await self.generate_quota_analytics()

            except Exception as e:
                logger.error(f"Analytics loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Cleanup alle Stunde

                if self._is_running:
                    await self._cleanup_old_analytics()

            except Exception as e:
                logger.error(f"Cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _real_time_loop(self) -> None:
        """Background-Loop für Real-time Updates."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Real-time updates alle Minute

                if self._is_running:
                    await self._update_real_time_metrics()

            except Exception as e:
                logger.error(f"Real-time loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_analytics(self) -> None:
        """Bereinigt alte Analytics-Daten."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.analytics_retention_days)

            for quota_id in list(self._quota_analytics.keys()):
                original_count = len(self._quota_analytics[quota_id])
                self._quota_analytics[quota_id] = [
                    a for a in self._quota_analytics[quota_id]
                    if a.generated_at > cutoff_time
                ]

                cleaned_count = original_count - len(self._quota_analytics[quota_id])
                if cleaned_count > 0:
                    logger.debug(f"Analytics cleanup für {quota_id}: {cleaned_count} alte Analytics entfernt")

            # Cleanup Usage-Metrics
            for quota_id in list(self._usage_metrics.keys()):
                original_count = len(self._usage_metrics[quota_id])
                self._usage_metrics[quota_id] = [
                    m for m in self._usage_metrics[quota_id]
                    if m["timestamp"] > cutoff_time
                ]

                cleaned_count = original_count - len(self._usage_metrics[quota_id])
                if cleaned_count > 0:
                    logger.debug(f"Usage metrics cleanup für {quota_id}: {cleaned_count} alte Metrics entfernt")

        except Exception as e:
            logger.error(f"Analytics cleanup fehlgeschlagen: {e}")

    async def _update_real_time_metrics(self) -> None:
        """Aktualisiert Real-time Metrics."""
        try:
            # Reset current usage (wird durch neue Requests wieder aufgebaut)
            for quota_data in self._real_time_usage.values():
                quota_data["current_usage"] = 0

        except Exception as e:
            logger.error(f"Real-time metrics update fehlgeschlagen: {e}")

    def _update_analytics_performance_stats(self, analytics_time_ms: float) -> None:
        """Aktualisiert Analytics-Performance-Statistiken."""
        self._analytics_generation_count += 1
        self._total_analytics_time_ms += analytics_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_analytics_time = (
            self._total_analytics_time_ms / self._analytics_generation_count
            if self._analytics_generation_count > 0 else 0.0
        )

        return {
            "total_analytics_generations": self._analytics_generation_count,
            "avg_analytics_time_ms": avg_analytics_time,
            "tracked_quotas": len(self._usage_metrics),
            "total_usage_metrics": sum(len(metrics) for metrics in self._usage_metrics.values()),
            "total_violation_metrics": sum(len(metrics) for metrics in self._violation_metrics.values()),
            "real_time_analytics_enabled": self.enable_real_time_analytics,
            "predictive_analytics_enabled": self.enable_predictive_analytics,
            "trend_analysis_enabled": self.enable_trend_analysis,
            "analytics_retention_days": self.analytics_retention_days
        }
