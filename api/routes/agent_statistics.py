# backend/api/routes/agent_statistics.py
"""Agent Statistics Endpoints für Keiko Personal Assistant

Implementiert Performance-Metriken und Nutzungsstatistiken für Agenten mit
detaillierter Analyse und Trend-Tracking.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field

from kei_logging import (
    BusinessLogicError,
    LogLinkedError,
    get_logger,
    with_log_links,
)

from .enhanced_agents_management import get_agent_or_404, validate_agent_access

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/agents", tags=["Agent Statistics"])


class TimeRange(BaseModel):
    """Zeitbereich für Statistiken."""
    start_time: datetime = Field(..., description="Start-Zeitpunkt")
    end_time: datetime = Field(..., description="End-Zeitpunkt")

    def validate_range(self):
        """Validiert Zeitbereich."""
        if self.end_time <= self.start_time:
            raise ValueError("end_time muss nach start_time liegen")

        max_range = timedelta(days=90)
        if self.end_time - self.start_time > max_range:
            raise ValueError("Zeitbereich darf maximal 90 Tage betragen")


class StatisticsFilter(BaseModel):
    """Filter für Statistiken."""
    include_performance: bool = Field(default=True, description="Performance-Metriken einschließen")
    include_usage: bool = Field(default=True, description="Nutzungsstatistiken einschließen")
    include_errors: bool = Field(default=True, description="Fehlerstatistiken einschließen")
    include_trends: bool = Field(default=False, description="Trend-Analyse einschließen")
    granularity: str = Field(default="hour", description="Granularität (minute, hour, day)")


class PerformanceMetrics(BaseModel):
    """Performance-Metriken."""
    total_requests: int = Field(default=0, description="Gesamtanzahl Requests")
    successful_requests: int = Field(default=0, description="Erfolgreiche Requests")
    failed_requests: int = Field(default=0, description="Fehlgeschlagene Requests")
    success_rate: float = Field(default=0.0, description="Erfolgsrate")
    error_rate: float = Field(default=0.0, description="Fehlerrate")
    average_response_time_ms: float = Field(default=0.0, description="Durchschnittliche Antwortzeit")
    median_response_time_ms: float = Field(default=0.0, description="Median Antwortzeit")
    p95_response_time_ms: float = Field(default=0.0, description="95. Perzentil Antwortzeit")
    p99_response_time_ms: float = Field(default=0.0, description="99. Perzentil Antwortzeit")
    min_response_time_ms: float = Field(default=0.0, description="Minimale Antwortzeit")
    max_response_time_ms: float = Field(default=0.0, description="Maximale Antwortzeit")
    requests_per_second: float = Field(default=0.0, description="Requests pro Sekunde")
    peak_rps: float = Field(default=0.0, description="Peak RPS")


class UsageStatistics(BaseModel):
    """Nutzungsstatistiken."""
    total_uptime_seconds: float = Field(default=0.0, description="Gesamte Uptime")
    uptime_percentage: float = Field(default=0.0, description="Verfügbarkeit in Prozent")
    active_sessions: int = Field(default=0, description="Aktive Sessions")
    unique_users: int = Field(default=0, description="Eindeutige Benutzer")
    total_data_processed_mb: float = Field(default=0.0, description="Verarbeitete Daten in MB")
    average_session_duration_minutes: float = Field(default=0.0, description="Durchschnittliche Session-Dauer")
    peak_concurrent_users: int = Field(default=0, description="Peak gleichzeitige Benutzer")
    resource_utilization: dict[str, float] = Field(default_factory=dict, description="Ressourcenauslastung")


class ErrorStatistics(BaseModel):
    """Fehlerstatistiken."""
    total_errors: int = Field(default=0, description="Gesamtanzahl Fehler")
    error_rate: float = Field(default=0.0, description="Fehlerrate")
    errors_by_type: dict[str, int] = Field(default_factory=dict, description="Fehler nach Typ")
    errors_by_category: dict[str, int] = Field(default_factory=dict, description="Fehler nach Kategorie")
    most_common_error: str | None = Field(None, description="Häufigster Fehler")
    error_trend: str = Field(default="stable", description="Fehler-Trend")
    critical_errors: int = Field(default=0, description="Kritische Fehler")
    resolved_errors: int = Field(default=0, description="Behobene Fehler")


class TrendAnalysis(BaseModel):
    """Trend-Analyse."""
    performance_trend: str = Field(default="stable", description="Performance-Trend")
    usage_trend: str = Field(default="stable", description="Nutzungs-Trend")
    error_trend: str = Field(default="stable", description="Fehler-Trend")
    capacity_trend: str = Field(default="stable", description="Kapazitäts-Trend")
    predictions: dict[str, Any] = Field(default_factory=dict, description="Vorhersagen")
    recommendations: list[str] = Field(default_factory=list, description="Empfehlungen")


class DetailedStatistics(BaseModel):
    """Detaillierte Statistiken."""
    agent_id: str = Field(..., description="Agent-ID")
    time_range: TimeRange = Field(..., description="Zeitbereich")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Generierungszeitpunkt")
    performance_metrics: PerformanceMetrics | None = Field(None, description="Performance-Metriken")
    usage_statistics: UsageStatistics | None = Field(None, description="Nutzungsstatistiken")
    error_statistics: ErrorStatistics | None = Field(None, description="Fehlerstatistiken")
    trend_analysis: TrendAnalysis | None = Field(None, description="Trend-Analyse")
    summary: dict[str, Any] = Field(default_factory=dict, description="Zusammenfassung")


def _get_performance_metrics(agent_id: str, _: TimeRange) -> PerformanceMetrics:
    """Holt Performance-Metriken für Agent im Zeitbereich."""
    try:
        # Integration mit Observability-System
        from observability import get_agent_metrics_collector

        collector = get_agent_metrics_collector(agent_id)
        metrics = collector.get_comprehensive_metrics()

        # Extrahiere Task-Metriken
        task_metrics = metrics.get("task_metrics", {})
        latency_metrics = task_metrics.get("latency", {})
        rate_metrics = task_metrics.get("rate", {})

        return PerformanceMetrics(
            total_requests=task_metrics.get("success_count", 0) + task_metrics.get("failure_count", 0),
            successful_requests=task_metrics.get("success_count", 0),
            failed_requests=task_metrics.get("failure_count", 0),
            success_rate=task_metrics.get("success_rate", 0.0),
            error_rate=task_metrics.get("error_rate", 0.0),
            average_response_time_ms=latency_metrics.get("mean", 0.0),
            median_response_time_ms=latency_metrics.get("p50", 0.0),
            p95_response_time_ms=latency_metrics.get("p95", 0.0),
            p99_response_time_ms=latency_metrics.get("p99", 0.0),
            min_response_time_ms=latency_metrics.get("min", 0.0),
            max_response_time_ms=latency_metrics.get("max", 0.0),
            requests_per_second=rate_metrics.get("current_rate", 0.0),
            peak_rps=rate_metrics.get("peak_rate", 0.0)
        )

    except ImportError:
        logger.warning("Observability-System nicht verfügbar - verwende Standard-Performance-Metriken")
        return PerformanceMetrics()


def _get_usage_statistics(agent_id: str, time_range: TimeRange) -> UsageStatistics:
    """Holt Nutzungsstatistiken für Agent im Zeitbereich."""
    try:
        # Integration mit Observability-System
        from observability import get_agent_metrics_collector

        collector = get_agent_metrics_collector(agent_id)
        metrics = collector.get_comprehensive_metrics()

        # Berechne Uptime
        uptime_seconds = metrics.get("uptime_seconds", 0)
        total_period = (time_range.end_time - time_range.start_time).total_seconds()
        uptime_percentage = min(100.0, (uptime_seconds / total_period) * 100) if total_period > 0 else 0.0

        # Placeholder für weitere Nutzungsstatistiken
        return UsageStatistics(
            total_uptime_seconds=uptime_seconds,
            uptime_percentage=uptime_percentage,
            active_sessions=0,  # Placeholder
            unique_users=0,  # Placeholder
            total_data_processed_mb=0.0,  # Placeholder
            average_session_duration_minutes=0.0,  # Placeholder
            peak_concurrent_users=0,  # Placeholder
            resource_utilization={
                "cpu": 0.0,  # Placeholder
                "memory": 0.0,  # Placeholder
                "disk": 0.0,  # Placeholder
                "network": 0.0  # Placeholder
            }
        )

    except ImportError:
        logger.warning("Observability-System nicht verfügbar - verwende Standard-Nutzungsstatistiken")
        return UsageStatistics()


def _get_error_statistics(agent_id: str, _: TimeRange) -> ErrorStatistics:
    """Holt Fehlerstatistiken für Agent im Zeitbereich."""
    try:
        # Integration mit Observability-System
        from observability import get_agent_metrics_collector

        collector = get_agent_metrics_collector(agent_id)
        metrics = collector.get_comprehensive_metrics()

        # Extrahiere Error-Metriken
        error_metrics = metrics.get("error_metrics", {})
        errors_by_category = error_metrics.get("errors_by_category", {})

        total_errors = error_metrics.get("total_errors", 0)
        error_rate = error_metrics.get("error_rate", 0.0)

        # Bestimme häufigsten Fehler
        most_common_error = None
        if errors_by_category:
            most_common_error = max(errors_by_category.items(), key=lambda x: x[1])[0]

        return ErrorStatistics(
            total_errors=total_errors,
            error_rate=error_rate,
            errors_by_type={},  # Placeholder
            errors_by_category=errors_by_category,
            most_common_error=most_common_error,
            error_trend="stable",  # Placeholder
            critical_errors=0,  # Placeholder
            resolved_errors=0  # Placeholder
        )

    except ImportError:
        logger.warning("Observability-System nicht verfügbar - verwende Standard-Fehlerstatistiken")
        return ErrorStatistics()


def _perform_trend_analysis(
    performance: PerformanceMetrics,
    usage: UsageStatistics,
    errors: ErrorStatistics
) -> TrendAnalysis:
    """Führt Trend-Analyse durch."""
    # Einfache Trend-Analyse basierend auf aktuellen Metriken
    performance_trend = "stable"
    if performance.success_rate > 0.95:
        performance_trend = "improving"
    elif performance.success_rate < 0.8:
        performance_trend = "declining"

    usage_trend = "stable"
    if usage.uptime_percentage > 99.0:
        usage_trend = "excellent"
    elif usage.uptime_percentage < 95.0:
        usage_trend = "concerning"

    error_trend = "stable"
    if errors.error_rate > 0.1:
        error_trend = "increasing"
    elif errors.error_rate < 0.01:
        error_trend = "decreasing"

    # Generiere Empfehlungen
    recommendations = []

    if performance.success_rate < 0.9:
        recommendations.append("Verbessere Erfolgsrate durch Fehleranalyse und -behebung")

    if performance.p95_response_time_ms > 1000:
        recommendations.append("Optimiere Performance - hohe P95-Latenz erkannt")

    if usage.uptime_percentage < 99.0:
        recommendations.append("Verbessere Verfügbarkeit durch Redundanz und Monitoring")

    if errors.error_rate > 0.05:
        recommendations.append("Reduziere Fehlerrate durch bessere Validierung und Testing")

    # Einfache Vorhersagen
    predictions = {
        "expected_requests_next_hour": performance.requests_per_second * 3600,
        "estimated_error_count_next_day": errors.total_errors * 1.1,  # 10% Puffer
        "predicted_uptime_next_week": usage.uptime_percentage
    }

    return TrendAnalysis(
        performance_trend=performance_trend,
        usage_trend=usage_trend,
        error_trend=error_trend,
        capacity_trend="stable",  # Placeholder
        predictions=predictions,
        recommendations=recommendations
    )


def _generate_statistics_summary(
    performance: PerformanceMetrics | None,
    usage: UsageStatistics | None,
    errors: ErrorStatistics | None,
    _: TrendAnalysis | None
) -> dict[str, Any]:
    """Generiert Statistiken-Zusammenfassung."""
    summary = {
        "overall_health": "unknown",
        "key_insights": [],
        "critical_issues": [],
        "performance_score": 0.0,
        "reliability_score": 0.0
    }

    if performance:
        # Performance-Score berechnen
        performance_score = (
            performance.success_rate * 0.4 +
            (1.0 - min(1.0, performance.p95_response_time_ms / 2000.0)) * 0.3 +
            min(1.0, performance.requests_per_second / 100.0) * 0.3
        ) * 100

        summary["performance_score"] = performance_score

        if performance.success_rate > 0.95:
            summary["key_insights"].append("Hohe Erfolgsrate - Agent arbeitet zuverlässig")
        elif performance.success_rate < 0.8:
            summary["critical_issues"].append("Niedrige Erfolgsrate - sofortige Aufmerksamkeit erforderlich")

    if usage:
        # Reliability-Score berechnen
        reliability_score = usage.uptime_percentage
        summary["reliability_score"] = reliability_score

        if usage.uptime_percentage > 99.5:
            summary["key_insights"].append("Exzellente Verfügbarkeit")
        elif usage.uptime_percentage < 95.0:
            summary["critical_issues"].append("Niedrige Verfügbarkeit - Infrastruktur prüfen")

    if errors:
        if errors.error_rate > 0.1:
            summary["critical_issues"].append("Hohe Fehlerrate - Fehleranalyse erforderlich")
        elif errors.error_rate < 0.01:
            summary["key_insights"].append("Sehr niedrige Fehlerrate")

    # Gesamt-Gesundheit bestimmen
    if summary["critical_issues"]:
        summary["overall_health"] = "critical"
    elif summary["performance_score"] > 80 and summary["reliability_score"] > 95:
        summary["overall_health"] = "excellent"
    elif summary["performance_score"] > 60 and summary["reliability_score"] > 90:
        summary["overall_health"] = "good"
    else:
        summary["overall_health"] = "needs_attention"

    return summary


# Statistics Endpoints

@router.get("/{agent_id}/stats", response_model=DetailedStatistics)
@with_log_links(component="statistics", operation="get_statistics")
async def get_agent_statistics(
    agent_id: str = Path(..., description="Agent-ID"),
    start_time: datetime | None = Query(None, description="Start-Zeitpunkt (ISO format)"),
    end_time: datetime | None = Query(None, description="End-Zeitpunkt (ISO format)"),
    include_performance: bool = Query(default=True, description="Performance-Metriken einschließen"),
    include_usage: bool = Query(default=True, description="Nutzungsstatistiken einschließen"),
    include_errors: bool = Query(default=True, description="Fehlerstatistiken einschließen"),
    include_trends: bool = Query(default=False, description="Trend-Analyse einschließen")
) -> DetailedStatistics:
    """Holt detaillierte Statistiken für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        start_time: Start-Zeitpunkt für Statistiken
        end_time: End-Zeitpunkt für Statistiken
        include_performance: Performance-Metriken einschließen
        include_usage: Nutzungsstatistiken einschließen
        include_errors: Fehlerstatistiken einschließen
        include_trends: Trend-Analyse einschließen

    Returns:
        Detaillierte Agent-Statistiken

    Raises:
        ValidationError: Bei ungültigen Parametern
        AuthorizationError: Bei fehlenden Berechtigungen
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="read")

    try:
        # Standard-Zeitbereich: letzte 24 Stunden
        if not start_time:
            start_time = datetime.now(UTC) - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now(UTC)

        # Erstelle und validiere Zeitbereich
        time_range = TimeRange(start_time=start_time, end_time=end_time)
        time_range.validate_range()

        # Sammle Statistiken basierend auf Filtern
        performance_metrics = None
        usage_statistics = None
        error_statistics = None
        trend_analysis = None

        if include_performance:
            performance_metrics = _get_performance_metrics(agent_id, time_range)

        if include_usage:
            usage_statistics = _get_usage_statistics(agent_id, time_range)

        if include_errors:
            error_statistics = _get_error_statistics(agent_id, time_range)

        if include_trends and (performance_metrics or usage_statistics or error_statistics):
            trend_analysis = _perform_trend_analysis(
                performance_metrics or PerformanceMetrics(),
                usage_statistics or UsageStatistics(),
                error_statistics or ErrorStatistics()
            )

        # Generiere Zusammenfassung
        summary = _generate_statistics_summary(
            performance_metrics,
            usage_statistics,
            error_statistics,
            trend_analysis
        )

        # Erstelle Response
        statistics = DetailedStatistics(
            agent_id=agent_id,
            time_range=time_range,
            performance_metrics=performance_metrics,
            usage_statistics=usage_statistics,
            error_statistics=error_statistics,
            trend_analysis=trend_analysis,
            summary=summary
        )

        logger.info(
            f"Statistiken für Agent {agent_id} generiert",
            extra={
                "agent_id": agent_id,
                "time_range_hours": (time_range.end_time - time_range.start_time).total_seconds() / 3600,
                "overall_health": summary["overall_health"],
                "performance_score": summary["performance_score"],
                "correlation_id": f"stats_{uuid.uuid4().hex[:8]}"
            }
        )

        return statistics

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Statistiken-Generierung fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="statistics",
            operation="get_statistics",
            cause=e
        )
