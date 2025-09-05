"""Dead Letter Queue Analytics und Monitoring System.
Umfassende Analytics für Failed Task Management und Pattern Detection.
"""

import asyncio
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .dead_letter_queue import FailedTask, FailureReason, TaskStatus
from .voice_workflow_dlq import VoiceWorkflowDLQ

logger = get_logger(__name__)


@dataclass
class FailurePattern:
    """Repräsentiert ein erkanntes Failure-Pattern."""
    pattern_id: str
    pattern_type: str  # temporal, frequency, cascade, resource
    description: str

    # Pattern Metrics
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    frequency_per_hour: float

    # Pattern Details
    failure_reasons: list[FailureReason]
    affected_users: set[str] = field(default_factory=set)
    affected_sessions: set[str] = field(default_factory=set)

    # Impact Assessment
    severity: str = "medium"  # low, medium, high, critical
    business_impact: str = "unknown"

    # Recommendations
    recommended_actions: list[str] = field(default_factory=list)
    auto_mitigation_possible: bool = False


@dataclass
class PerformanceImpact:
    """Repräsentiert Performance-Impact von Failed Tasks."""
    impact_id: str

    # Latency Impact
    baseline_latency_ms: float
    failed_task_latency_ms: float
    latency_degradation_percent: float

    # Throughput Impact
    baseline_throughput_rps: float
    degraded_throughput_rps: float
    throughput_reduction_percent: float

    # Resource Impact
    cpu_overhead_percent: float
    memory_overhead_mb: float

    # User Experience Impact
    user_satisfaction_score: float = 0.0
    session_abandonment_rate: float = 0.0

    # Recovery Impact
    recovery_time_ms: float = 0.0
    recovery_success_rate: float = 0.0


class DLQAnalyticsEngine:
    """Dead Letter Queue Analytics Engine.
    Analysiert Failed Tasks für Pattern Detection und Performance Impact.
    """

    def __init__(self, dlq: VoiceWorkflowDLQ):
        self.dlq = dlq

        # Analytics Storage
        self._failure_patterns: dict[str, FailurePattern] = {}
        self._performance_impacts: dict[str, PerformanceImpact] = {}

        # Pattern Detection
        self._pattern_detection_enabled = True
        self._pattern_detection_window_hours = 24
        self._pattern_threshold_count = 5

        # Analytics Cache
        self._analytics_cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 Minuten

        # Background Tasks
        self._analytics_task: asyncio.Task | None = None
        self._pattern_detection_task: asyncio.Task | None = None
        self._running = False

        logger.info("DLQ analytics engine initialized")

    async def start(self) -> None:
        """Startet Analytics Background-Tasks."""
        if self._running:
            return

        self._running = True

        # Starte Analytics Task
        self._analytics_task = asyncio.create_task(self._analytics_loop())

        # Starte Pattern Detection Task
        if self._pattern_detection_enabled:
            self._pattern_detection_task = asyncio.create_task(self._pattern_detection_loop())

        logger.info("DLQ analytics background tasks started")

    async def stop(self) -> None:
        """Stoppt Analytics Background-Tasks."""
        self._running = False

        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass

        if self._pattern_detection_task:
            self._pattern_detection_task.cancel()
            try:
                await self._pattern_detection_task
            except asyncio.CancelledError:
                pass

        logger.info("DLQ analytics background tasks stopped")

    async def analyze_failure_trends(
        self,
        time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Analysiert Failure-Trends über Zeitfenster."""
        cache_key = f"failure_trends_{time_window_hours}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Sammle Failed Tasks im Zeitfenster
        recent_tasks = [
            task for task in self.dlq._failed_tasks.values()
            if task.failed_at >= cutoff_time
        ]

        if not recent_tasks:
            return {"error": "No failed tasks in time window"}

        # Failure Reason Trends
        failure_reason_counts = Counter(task.failure_reason for task in recent_tasks)
        failure_category_counts = Counter(task.failure_category for task in recent_tasks)

        # Temporal Analysis
        hourly_failures = defaultdict(int)
        for task in recent_tasks:
            hour_key = task.failed_at.strftime("%Y-%m-%d %H:00")
            hourly_failures[hour_key] += 1

        # User Impact Analysis
        affected_users = set(task.user_id for task in recent_tasks if task.user_id)
        affected_sessions = set(task.session_id for task in recent_tasks if task.session_id)

        # Recovery Analysis
        recovery_stats = self._analyze_recovery_performance(recent_tasks)

        result = {
            "time_window_hours": time_window_hours,
            "total_failures": len(recent_tasks),
            "failure_trends": {
                "by_reason": dict(failure_reason_counts.most_common()),
                "by_category": dict(failure_category_counts.most_common()),
                "hourly_distribution": dict(hourly_failures)
            },
            "impact_analysis": {
                "affected_users": len(affected_users),
                "affected_sessions": len(affected_sessions),
                "user_impact_rate": len(affected_users) / max(len(recent_tasks), 1)
            },
            "recovery_analysis": recovery_stats,
            "generated_at": datetime.utcnow().isoformat()
        }

        self._cache_result(cache_key, result)
        return result

    async def detect_failure_patterns(self) -> list[FailurePattern]:
        """Erkennt Failure-Patterns in Failed Tasks."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self._pattern_detection_window_hours)

        recent_tasks = [
            task for task in self.dlq._failed_tasks.values()
            if task.failed_at >= cutoff_time
        ]

        detected_patterns = []

        # Temporal Patterns
        temporal_patterns = await self._detect_temporal_patterns(recent_tasks)
        detected_patterns.extend(temporal_patterns)

        # Frequency Patterns
        frequency_patterns = await self._detect_frequency_patterns(recent_tasks)
        detected_patterns.extend(frequency_patterns)

        # Cascade Patterns
        cascade_patterns = await self._detect_cascade_patterns(recent_tasks)
        detected_patterns.extend(cascade_patterns)

        # Resource Patterns
        resource_patterns = await self._detect_resource_patterns(recent_tasks)
        detected_patterns.extend(resource_patterns)

        # Speichere erkannte Patterns
        for pattern in detected_patterns:
            self._failure_patterns[pattern.pattern_id] = pattern

        logger.info(f"Detected {len(detected_patterns)} failure patterns")
        return detected_patterns

    async def analyze_performance_impact(
        self,
        baseline_metrics: dict[str, float]
    ) -> PerformanceImpact:
        """Analysiert Performance-Impact von Failed Tasks."""
        # Sammle aktuelle Metriken
        current_metrics = await self._collect_current_metrics()

        # Berechne Impact
        latency_degradation = (
            (current_metrics["average_latency_ms"] - baseline_metrics["average_latency_ms"]) /
            baseline_metrics["average_latency_ms"] * 100
        ) if baseline_metrics["average_latency_ms"] > 0 else 0

        throughput_reduction = (
            (baseline_metrics["throughput_rps"] - current_metrics["throughput_rps"]) /
            baseline_metrics["throughput_rps"] * 100
        ) if baseline_metrics["throughput_rps"] > 0 else 0

        impact = PerformanceImpact(
            impact_id=f"impact_{int(time.time())}",
            baseline_latency_ms=baseline_metrics["average_latency_ms"],
            failed_task_latency_ms=current_metrics["average_latency_ms"],
            latency_degradation_percent=latency_degradation,
            baseline_throughput_rps=baseline_metrics["throughput_rps"],
            degraded_throughput_rps=current_metrics["throughput_rps"],
            throughput_reduction_percent=throughput_reduction,
            cpu_overhead_percent=current_metrics.get("cpu_overhead_percent", 0),
            memory_overhead_mb=current_metrics.get("memory_overhead_mb", 0)
        )

        self._performance_impacts[impact.impact_id] = impact
        return impact

    async def get_proactive_recommendations(self) -> list[dict[str, Any]]:
        """Gibt proaktive Empfehlungen basierend auf Analytics zurück."""
        recommendations = []

        # Pattern-basierte Empfehlungen
        for pattern in self._failure_patterns.values():
            if pattern.auto_mitigation_possible:
                recommendations.append({
                    "type": "pattern_mitigation",
                    "priority": pattern.severity,
                    "description": f"Auto-mitigation available for pattern: {pattern.description}",
                    "actions": pattern.recommended_actions,
                    "pattern_id": pattern.pattern_id
                })

        # Performance-basierte Empfehlungen
        for impact in self._performance_impacts.values():
            if impact.latency_degradation_percent > 20:
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "description": f"Latency degradation detected: {impact.latency_degradation_percent:.1f}%",
                    "actions": [
                        "Increase retry delays",
                        "Implement circuit breaker",
                        "Scale resources"
                    ]
                })

        # Resource-basierte Empfehlungen
        dlq_stats = await self.dlq.get_statistics()
        queue_usage = dlq_stats["queue_sizes"]["retry_queue"] / self.dlq.max_queue_size

        if queue_usage > 0.8:
            recommendations.append({
                "type": "resource_scaling",
                "priority": "critical",
                "description": f"DLQ queue usage high: {queue_usage:.1%}",
                "actions": [
                    "Increase queue size",
                    "Implement batch processing",
                    "Add more retry workers"
                ]
            })

        return recommendations

    async def generate_failure_report(
        self,
        time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Generiert umfassenden Failure-Report."""
        # Sammle alle Analytics
        failure_trends = await self.analyze_failure_trends(time_window_hours)
        detected_patterns = await self.detect_failure_patterns()
        voice_analytics = await self.dlq.get_voice_failure_analytics()
        recommendations = await self.get_proactive_recommendations()

        # DLQ Statistics
        dlq_stats = await self.dlq.get_statistics()

        return {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "report_type": "comprehensive_failure_analysis"
            },
            "executive_summary": {
                "total_failures": failure_trends.get("total_failures", 0),
                "affected_users": failure_trends.get("impact_analysis", {}).get("affected_users", 0),
                "patterns_detected": len(detected_patterns),
                "critical_recommendations": len([r for r in recommendations if r.get("priority") == "critical"])
            },
            "failure_analysis": failure_trends,
            "pattern_detection": {
                "patterns_detected": len(detected_patterns),
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "type": p.pattern_type,
                        "description": p.description,
                        "severity": p.severity,
                        "occurrence_count": p.occurrence_count,
                        "frequency_per_hour": p.frequency_per_hour
                    }
                    for p in detected_patterns
                ]
            },
            "voice_workflow_analysis": voice_analytics,
            "dlq_performance": dlq_stats,
            "recommendations": recommendations,
            "performance_impacts": [
                {
                    "impact_id": impact.impact_id,
                    "latency_degradation_percent": impact.latency_degradation_percent,
                    "throughput_reduction_percent": impact.throughput_reduction_percent,
                    "recovery_success_rate": impact.recovery_success_rate
                }
                for impact in self._performance_impacts.values()
            ]
        }

    # Private Methods

    async def _analytics_loop(self) -> None:
        """Background-Task für kontinuierliche Analytics."""
        while self._running:
            try:
                # Führe periodische Analytics durch
                await self._update_analytics_cache()
                await asyncio.sleep(300)  # Alle 5 Minuten
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analytics loop error: {e}")
                await asyncio.sleep(300)

    async def _pattern_detection_loop(self) -> None:
        """Background-Task für Pattern Detection."""
        while self._running:
            try:
                # Führe Pattern Detection durch
                await self.detect_failure_patterns()
                await asyncio.sleep(1800)  # Alle 30 Minuten
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pattern detection loop error: {e}")
                await asyncio.sleep(1800)

    async def _detect_temporal_patterns(self, tasks: list[FailedTask]) -> list[FailurePattern]:
        """Erkennt zeitliche Failure-Patterns."""
        patterns = []

        # Gruppiere Tasks nach Stunden
        hourly_groups = defaultdict(list)
        for task in tasks:
            hour_key = task.failed_at.strftime("%H")
            hourly_groups[hour_key].append(task)

        # Erkenne Spitzen
        for hour, hour_tasks in hourly_groups.items():
            if len(hour_tasks) >= self._pattern_threshold_count:
                pattern = FailurePattern(
                    pattern_id=f"temporal_spike_{hour}",
                    pattern_type="temporal",
                    description=f"Failure spike detected at hour {hour}",
                    occurrence_count=len(hour_tasks),
                    first_seen=min(task.failed_at for task in hour_tasks),
                    last_seen=max(task.failed_at for task in hour_tasks),
                    frequency_per_hour=len(hour_tasks),
                    failure_reasons=list(set(task.failure_reason for task in hour_tasks)),
                    severity="medium" if len(hour_tasks) < 10 else "high"
                )
                patterns.append(pattern)

        return patterns

    async def _detect_frequency_patterns(self, tasks: list[FailedTask]) -> list[FailurePattern]:
        """Erkennt Häufigkeits-Patterns."""
        patterns = []

        # Analysiere Failure Reason Häufigkeiten
        reason_counts = Counter(task.failure_reason for task in tasks)

        for reason, count in reason_counts.most_common():
            if count >= self._pattern_threshold_count:
                pattern = FailurePattern(
                    pattern_id=f"frequency_{reason.value}",
                    pattern_type="frequency",
                    description=f"High frequency of {reason.value} failures",
                    occurrence_count=count,
                    first_seen=min(task.failed_at for task in tasks if task.failure_reason == reason),
                    last_seen=max(task.failed_at for task in tasks if task.failure_reason == reason),
                    frequency_per_hour=count / self._pattern_detection_window_hours,
                    failure_reasons=[reason],
                    severity="high" if count > 20 else "medium"
                )
                patterns.append(pattern)

        return patterns

    async def _detect_cascade_patterns(self, tasks: list[FailedTask]) -> list[FailurePattern]:
        """Erkennt Cascade-Failure-Patterns."""
        patterns = []

        # Gruppiere Tasks nach Workflow
        workflow_groups = defaultdict(list)
        for task in tasks:
            workflow_groups[task.workflow_id].append(task)

        # Erkenne Cascade Failures
        for workflow_id, workflow_tasks in workflow_groups.items():
            if len(workflow_tasks) >= 3:  # Mindestens 3 Failed Tasks im Workflow
                # Sortiere nach Zeit
                workflow_tasks.sort(key=lambda t: t.failed_at)

                # Prüfe zeitliche Nähe
                time_diffs = [
                    (workflow_tasks[i+1].failed_at - workflow_tasks[i].failed_at).total_seconds()
                    for i in range(len(workflow_tasks) - 1)
                ]

                if all(diff < 300 for diff in time_diffs):  # Alle innerhalb 5 Minuten
                    pattern = FailurePattern(
                        pattern_id=f"cascade_{workflow_id}",
                        pattern_type="cascade",
                        description=f"Cascade failure detected in workflow {workflow_id}",
                        occurrence_count=len(workflow_tasks),
                        first_seen=workflow_tasks[0].failed_at,
                        last_seen=workflow_tasks[-1].failed_at,
                        frequency_per_hour=len(workflow_tasks) / (max(time_diffs) / 3600),
                        failure_reasons=list(set(task.failure_reason for task in workflow_tasks)),
                        severity="critical"
                    )
                    patterns.append(pattern)

        return patterns

    async def _detect_resource_patterns(self, tasks: list[FailedTask]) -> list[FailurePattern]:
        """Erkennt Resource-bezogene Patterns."""
        patterns = []

        # Filtere Resource-Failures
        resource_tasks = [
            task for task in tasks
            if task.failure_category == "resource"
        ]

        if len(resource_tasks) >= self._pattern_threshold_count:
            pattern = FailurePattern(
                pattern_id="resource_exhaustion",
                pattern_type="resource",
                description="Resource exhaustion pattern detected",
                occurrence_count=len(resource_tasks),
                first_seen=min(task.failed_at for task in resource_tasks),
                last_seen=max(task.failed_at for task in resource_tasks),
                frequency_per_hour=len(resource_tasks) / self._pattern_detection_window_hours,
                failure_reasons=list(set(task.failure_reason for task in resource_tasks)),
                severity="high",
                recommended_actions=[
                    "Scale resources",
                    "Implement resource limits",
                    "Add resource monitoring"
                ],
                auto_mitigation_possible=True
            )
            patterns.append(pattern)

        return patterns

    def _analyze_recovery_performance(self, tasks: list[FailedTask]) -> dict[str, Any]:
        """Analysiert Recovery-Performance."""
        recovered_tasks = [task for task in tasks if task.status == TaskStatus.RECOVERED]

        if not recovered_tasks:
            return {"recovery_rate": 0.0, "average_recovery_time_ms": 0.0}

        recovery_rate = len(recovered_tasks) / len(tasks)

        recovery_times = []
        for task in recovered_tasks:
            if task.retry_latencies_ms:
                recovery_times.append(sum(task.retry_latencies_ms))

        average_recovery_time = statistics.mean(recovery_times) if recovery_times else 0.0

        return {
            "recovery_rate": recovery_rate,
            "average_recovery_time_ms": average_recovery_time,
            "total_recovered": len(recovered_tasks),
            "recovery_attempts_distribution": Counter(task.retry_count for task in recovered_tasks)
        }

    async def _collect_current_metrics(self) -> dict[str, float]:
        """Sammelt aktuelle Performance-Metriken."""
        # Mock Implementation - in echter Anwendung würde hier
        # Integration mit Monitoring-System stehen
        return {
            "average_latency_ms": 250.0,
            "throughput_rps": 85.0,
            "cpu_overhead_percent": 5.0,
            "memory_overhead_mb": 128.0
        }

    async def _update_analytics_cache(self) -> None:
        """Aktualisiert Analytics-Cache."""
        try:
            # Entferne abgelaufene Cache-Einträge
            now = datetime.utcnow()
            expired_keys = [
                key for key, (_, timestamp) in self._analytics_cache.items()
                if (now - timestamp).total_seconds() > self._cache_ttl_seconds
            ]

            for key in expired_keys:
                del self._analytics_cache[key]

        except Exception as e:
            logger.error(f"Analytics cache update error: {e}")

    def _get_cached_result(self, cache_key: str) -> Any | None:
        """Holt gecachtes Analytics-Result."""
        if cache_key in self._analytics_cache:
            result, timestamp = self._analytics_cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl_seconds:
                return result
            del self._analytics_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cached Analytics-Result."""
        self._analytics_cache[cache_key] = (result, datetime.utcnow())


def create_dlq_analytics_engine(dlq: VoiceWorkflowDLQ) -> DLQAnalyticsEngine:
    """Factory-Funktion für DLQ Analytics Engine."""
    return DLQAnalyticsEngine(dlq)
