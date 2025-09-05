# backend/quotas_limits/quota_analytics.py
"""Quota Analytics für Keiko Personal Assistant

Implementiert Usage-Pattern-Analyse, Predictive Analytics,
Quota-Reporting und automatische Quota-Anpassung.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

if TYPE_CHECKING:
    import asyncio

logger = get_logger(__name__)


class UsagePatternType(str, Enum):
    """Typen von Usage-Patterns."""
    STEADY = "steady"
    BURSTY = "bursty"
    PERIODIC = "periodic"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    IRREGULAR = "irregular"


class TrendDirection(str, Enum):
    """Richtung von Trends."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class AlertSeverity(str, Enum):
    """Schweregrad von Alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ReportType(str, Enum):
    """Typen von Reports."""
    USAGE_SUMMARY = "usage_summary"
    TREND_ANALYSIS = "trend_analysis"
    QUOTA_EFFICIENCY = "quota_efficiency"
    COST_ANALYSIS = "cost_analysis"
    PREDICTIVE_FORECAST = "predictive_forecast"


@dataclass
class UsageDataPoint:
    """Einzelner Usage-Datenpunkt."""
    timestamp: datetime
    value: float
    quota_id: str
    scope_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsagePattern:
    """Erkanntes Usage-Pattern."""
    pattern_id: str
    pattern_type: UsagePatternType
    quota_id: str
    scope_id: str

    # Pattern-Details
    confidence: float  # 0.0 - 1.0
    description: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Statistische Eigenschaften
    mean_usage: float = 0.0
    std_deviation: float = 0.0
    peak_usage: float = 0.0
    min_usage: float = 0.0

    # Zeitliche Eigenschaften
    period_hours: float | None = None  # Für periodische Patterns
    trend_slope: float | None = None   # Für Trend-Patterns

    # Metadaten
    data_points_analyzed: int = 0
    analysis_window_hours: float = 24.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageTrend:
    """Usage-Trend-Analyse."""
    trend_id: str
    quota_id: str
    scope_id: str

    # Trend-Details
    direction: TrendDirection
    slope: float  # Änderung pro Zeiteinheit
    r_squared: float  # Güte der linearen Regression

    # Zeitraum
    start_time: datetime
    end_time: datetime

    # Prognose
    predicted_exhaustion: datetime | None = None
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    # Metadaten
    data_points: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PredictiveAnalysis:
    """Predictive Analytics-Ergebnis."""
    analysis_id: str
    quota_id: str
    scope_id: str

    # Vorhersagen
    predicted_usage_24h: float
    predicted_usage_7d: float
    predicted_usage_30d: float

    # Exhaustion-Prognose
    exhaustion_probability_24h: float
    exhaustion_probability_7d: float
    exhaustion_probability_30d: float

    # Empfehlungen
    recommended_quota_adjustment: float | None = None
    recommended_action: str | None = None

    # Konfidenz
    prediction_confidence: float = 0.0
    model_accuracy: float = 0.0

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    model_version: str = "1.0"
    features_used: list[str] = field(default_factory=list)


@dataclass
class AlertRule:
    """Regel für automatische Alerts."""
    rule_id: str
    name: str
    description: str

    # Bedingungen
    quota_id_pattern: str  # Regex-Pattern
    scope_pattern: str

    # Trigger-Bedingungen
    usage_threshold: float | None = None  # Prozent
    trend_threshold: float | None = None  # Änderungsrate
    exhaustion_hours: int | None = None   # Stunden bis Erschöpfung

    # Alert-Konfiguration
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_minutes: int = 60  # Mindestabstand zwischen Alerts

    # Aktionen
    notify_emails: list[str] = field(default_factory=list)
    webhook_url: str | None = None
    auto_adjust_quota: bool = False
    adjustment_factor: float = 1.2  # 20% Erhöhung

    # Gültigkeit
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class QuotaMetrics:
    """Quota-Metriken für Zeitraum."""
    quota_id: str
    scope_id: str

    # Zeitraum
    start_time: datetime
    end_time: datetime

    # Usage-Metriken
    total_usage: float
    average_usage: float
    peak_usage: float
    min_usage: float

    # Effizienz-Metriken
    quota_utilization: float  # Prozent
    waste_percentage: float   # Ungenutzte Quota
    efficiency_score: float   # 0.0 - 1.0

    # Trend-Metriken
    usage_trend: TrendDirection
    growth_rate: float  # Prozent pro Tag

    # Violation-Metriken
    violations_count: int
    violation_rate: float  # Violations pro Stunde

    # Metadaten
    data_points: int
    calculated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class QuotaReport:
    """Quota-Report."""
    report_id: str
    report_type: ReportType
    title: str

    # Scope
    quota_ids: list[str]
    scope_ids: list[str]

    # Zeitraum
    start_time: datetime
    end_time: datetime

    # Report-Daten
    metrics: list[QuotaMetrics] = field(default_factory=list)
    patterns: list[UsagePattern] = field(default_factory=list)
    trends: list[UsageTrend] = field(default_factory=list)
    predictions: list[PredictiveAnalysis] = field(default_factory=list)

    # Zusammenfassung
    summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    # Metadaten
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    generated_by: str | None = None


class PatternDetector:
    """Detektor für Usage-Patterns."""

    def __init__(self):
        """Initialisiert Pattern Detector."""
        self._min_data_points = 10
        self._pattern_cache: dict[str, UsagePattern] = {}

    def detect_patterns(
        self,
        data_points: list[UsageDataPoint],
        quota_id: str,
        scope_id: str
    ) -> list[UsagePattern]:
        """Detektiert Usage-Patterns in Daten."""
        if len(data_points) < self._min_data_points:
            return []

        patterns = []
        values = [dp.value for dp in data_points]

        # Statistische Eigenschaften berechnen
        mean_usage = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        peak_usage = max(values)
        min_usage = min(values)

        # Steady Pattern
        if std_dev / mean_usage < 0.2:  # Niedrige Variabilität
            pattern = UsagePattern(
                pattern_id=f"steady_{quota_id}_{scope_id}",
                pattern_type=UsagePatternType.STEADY,
                quota_id=quota_id,
                scope_id=scope_id,
                confidence=0.9,
                description="Konstante, gleichmäßige Nutzung",
                mean_usage=mean_usage,
                std_deviation=std_dev,
                peak_usage=peak_usage,
                min_usage=min_usage,
                data_points_analyzed=len(data_points)
            )
            patterns.append(pattern)

        # Bursty Pattern
        elif std_dev / mean_usage > 1.0:  # Hohe Variabilität
            pattern = UsagePattern(
                pattern_id=f"bursty_{quota_id}_{scope_id}",
                pattern_type=UsagePatternType.BURSTY,
                quota_id=quota_id,
                scope_id=scope_id,
                confidence=0.8,
                description="Unregelmäßige Nutzungsspitzen",
                mean_usage=mean_usage,
                std_deviation=std_dev,
                peak_usage=peak_usage,
                min_usage=min_usage,
                data_points_analyzed=len(data_points)
            )
            patterns.append(pattern)

        # Trend Pattern
        trend_slope = self._calculate_trend_slope(data_points)
        if abs(trend_slope) > 0.1:  # Signifikanter Trend
            if trend_slope > 0:
                pattern_type = UsagePatternType.TRENDING_UP
                description = "Steigende Nutzung über Zeit"
            else:
                pattern_type = UsagePatternType.TRENDING_DOWN
                description = "Fallende Nutzung über Zeit"

            pattern = UsagePattern(
                pattern_id=f"trend_{quota_id}_{scope_id}",
                pattern_type=pattern_type,
                quota_id=quota_id,
                scope_id=scope_id,
                confidence=0.7,
                description=description,
                mean_usage=mean_usage,
                std_deviation=std_dev,
                peak_usage=peak_usage,
                min_usage=min_usage,
                trend_slope=trend_slope,
                data_points_analyzed=len(data_points)
            )
            patterns.append(pattern)

        # Periodic Pattern (vereinfacht)
        period = self._detect_periodicity(values)
        if period:
            pattern = UsagePattern(
                pattern_id=f"periodic_{quota_id}_{scope_id}",
                pattern_type=UsagePatternType.PERIODIC,
                quota_id=quota_id,
                scope_id=scope_id,
                confidence=0.6,
                description=f"Periodische Nutzung mit {period:.1f}h Zyklus",
                mean_usage=mean_usage,
                std_deviation=std_dev,
                peak_usage=peak_usage,
                min_usage=min_usage,
                period_hours=period,
                data_points_analyzed=len(data_points)
            )
            patterns.append(pattern)

        return patterns

    def _calculate_trend_slope(self, data_points: list[UsageDataPoint]) -> float:
        """Berechnet Trend-Steigung mit linearer Regression."""
        if len(data_points) < 2:
            return 0.0

        # Konvertiere Timestamps zu numerischen Werten
        x_values = [(dp.timestamp - data_points[0].timestamp).total_seconds() / 3600 for dp in data_points]
        y_values = [dp.value for dp in data_points]

        # Einfache lineare Regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _detect_periodicity(self, values: list[float]) -> float | None:
        """Detektiert Periodizität in Werten (vereinfacht)."""
        if len(values) < 24:  # Mindestens 24 Datenpunkte
            return None

        # Vereinfachte Periodizitäts-Detection
        # In Produktion würde hier FFT oder Autokorrelation verwendet

        # Prüfe auf tägliche Periodizität (24h)
        if len(values) >= 48:  # 2 Tage Daten
            daily_correlation = self._calculate_correlation(values[:24], values[24:48])
            if daily_correlation > 0.7:
                return 24.0

        # Prüfe auf wöchentliche Periodizität (168h)
        if len(values) >= 336:  # 2 Wochen Daten
            weekly_correlation = self._calculate_correlation(values[:168], values[168:336])
            if weekly_correlation > 0.6:
                return 168.0

        return None

    def _calculate_correlation(self, series1: list[float], series2: list[float]) -> float:
        """Berechnet Korrelation zwischen zwei Zeitreihen."""
        if len(series1) != len(series2) or len(series1) == 0:
            return 0.0

        mean1 = statistics.mean(series1)
        mean2 = statistics.mean(series2)

        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(series1, series2, strict=False))

        sum_sq1 = sum((x - mean1) ** 2 for x in series1)
        sum_sq2 = sum((y - mean2) ** 2 for y in series2)

        denominator = (sum_sq1 * sum_sq2) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator


class PredictiveModel:
    """Predictive Model für Quota-Usage."""

    def __init__(self):
        """Initialisiert Predictive Model."""
        self._model_accuracy = 0.75  # Vereinfachte Genauigkeit
        self._prediction_cache: dict[str, PredictiveAnalysis] = {}

    def predict_usage(
        self,
        data_points: list[UsageDataPoint],
        quota_id: str,
        scope_id: str,
        current_quota: float
    ) -> PredictiveAnalysis:
        """Erstellt Usage-Vorhersage."""
        import uuid

        if len(data_points) < 5:
            # Fallback für wenige Daten
            return PredictiveAnalysis(
                analysis_id=str(uuid.uuid4()),
                quota_id=quota_id,
                scope_id=scope_id,
                predicted_usage_24h=0.0,
                predicted_usage_7d=0.0,
                predicted_usage_30d=0.0,
                exhaustion_probability_24h=0.0,
                exhaustion_probability_7d=0.0,
                exhaustion_probability_30d=0.0,
                prediction_confidence=0.1,
                model_accuracy=self._model_accuracy
            )

        # Berechne Trend
        values = [dp.value for dp in data_points]
        recent_values = values[-24:] if len(values) >= 24 else values  # Letzte 24 Stunden

        current_usage = recent_values[-1] if recent_values else 0.0
        mean_usage = statistics.mean(recent_values)

        # Einfache lineare Extrapolation
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        else:
            trend = 0.0

        # Vorhersagen
        predicted_24h = current_usage + (trend * 24)
        predicted_7d = current_usage + (trend * 24 * 7)
        predicted_30d = current_usage + (trend * 24 * 30)

        # Exhaustion-Wahrscheinlichkeiten
        exhaustion_24h = min(1.0, max(0.0, predicted_24h / current_quota)) if current_quota > 0 else 0.0
        exhaustion_7d = min(1.0, max(0.0, predicted_7d / current_quota)) if current_quota > 0 else 0.0
        exhaustion_30d = min(1.0, max(0.0, predicted_30d / current_quota)) if current_quota > 0 else 0.0

        # Empfehlungen
        recommended_adjustment = None
        recommended_action = None

        if exhaustion_24h > 0.8:
            recommended_adjustment = current_quota * 1.5  # 50% Erhöhung
            recommended_action = "Immediate quota increase recommended"
        elif exhaustion_7d > 0.7:
            recommended_adjustment = current_quota * 1.3  # 30% Erhöhung
            recommended_action = "Quota increase recommended within 7 days"
        elif exhaustion_30d > 0.6:
            recommended_adjustment = current_quota * 1.2  # 20% Erhöhung
            recommended_action = "Monitor usage and consider quota adjustment"

        # Konfidenz basierend auf Datenmenge und Variabilität
        confidence = min(0.9, len(data_points) / 100.0)  # Mehr Daten = höhere Konfidenz
        if len(recent_values) > 1:
            variability = statistics.stdev(recent_values) / mean_usage if mean_usage > 0 else 1.0
            confidence *= max(0.3, 1.0 - variability)  # Weniger Variabilität = höhere Konfidenz

        return PredictiveAnalysis(
            analysis_id=str(uuid.uuid4()),
            quota_id=quota_id,
            scope_id=scope_id,
            predicted_usage_24h=max(0.0, predicted_24h),
            predicted_usage_7d=max(0.0, predicted_7d),
            predicted_usage_30d=max(0.0, predicted_30d),
            exhaustion_probability_24h=exhaustion_24h,
            exhaustion_probability_7d=exhaustion_7d,
            exhaustion_probability_30d=exhaustion_30d,
            recommended_quota_adjustment=recommended_adjustment,
            recommended_action=recommended_action,
            prediction_confidence=confidence,
            model_accuracy=self._model_accuracy,
            features_used=["trend", "mean_usage", "variability"]
        )


class QuotaAnalytics:
    """Analytics-Engine für Quota-Management."""

    def __init__(self):
        """Initialisiert Quota Analytics."""
        self._usage_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._patterns: dict[str, list[UsagePattern]] = {}
        self._trends: dict[str, list[UsageTrend]] = {}
        self._predictions: dict[str, PredictiveAnalysis] = {}
        self._alert_rules: dict[str, AlertRule] = {}
        self._reports: dict[str, QuotaReport] = {}

        self.pattern_detector = PatternDetector()
        self.predictive_model = PredictiveModel()

        # Statistiken
        self._analyses_performed = 0
        self._patterns_detected = 0
        self._predictions_made = 0
        self._alerts_triggered = 0

        # Background-Tasks
        self._analysis_task: asyncio.Task | None = None
        self._analysis_interval = 3600  # 1 Stunde

    def record_usage(
        self,
        quota_id: str,
        scope_id: str,
        usage_value: float,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Zeichnet Usage-Datenpunkt auf."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        data_point = UsageDataPoint(
            timestamp=timestamp,
            value=usage_value,
            quota_id=quota_id,
            scope_id=scope_id,
            metadata=metadata or {}
        )

        key = f"{quota_id}:{scope_id}"
        self._usage_data[key].append(data_point)

    @trace_function("quota_analytics.analyze")
    async def analyze_usage_patterns(
        self,
        quota_id: str,
        scope_id: str,
        hours_back: int = 24
    ) -> list[UsagePattern]:
        """Analysiert Usage-Patterns."""
        self._analyses_performed += 1

        key = f"{quota_id}:{scope_id}"
        data_points = list(self._usage_data[key])

        if not data_points:
            return []

        # Filtere Daten nach Zeitraum
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)
        filtered_data = [dp for dp in data_points if dp.timestamp >= cutoff_time]

        # Detektiere Patterns
        patterns = self.pattern_detector.detect_patterns(filtered_data, quota_id, scope_id)

        # Cache Patterns
        self._patterns[key] = patterns
        self._patterns_detected += len(patterns)

        return patterns

    async def generate_predictions(
        self,
        quota_id: str,
        scope_id: str,
        current_quota: float,
        hours_back: int = 168  # 7 Tage
    ) -> PredictiveAnalysis:
        """Generiert Vorhersagen."""
        self._predictions_made += 1

        key = f"{quota_id}:{scope_id}"
        data_points = list(self._usage_data[key])

        # Filtere Daten nach Zeitraum
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)
        filtered_data = [dp for dp in data_points if dp.timestamp >= cutoff_time]

        # Generiere Vorhersage
        prediction = self.predictive_model.predict_usage(
            filtered_data, quota_id, scope_id, current_quota
        )

        # Cache Vorhersage
        self._predictions[key] = prediction

        return prediction

    def calculate_metrics(
        self,
        quota_id: str,
        scope_id: str,
        start_time: datetime,
        end_time: datetime,
        current_quota: float
    ) -> QuotaMetrics:
        """Berechnet Quota-Metriken."""
        key = f"{quota_id}:{scope_id}"
        data_points = list(self._usage_data[key])

        # Filtere Daten nach Zeitraum
        filtered_data = [
            dp for dp in data_points
            if start_time <= dp.timestamp <= end_time
        ]

        if not filtered_data:
            return QuotaMetrics(
                quota_id=quota_id,
                scope_id=scope_id,
                start_time=start_time,
                end_time=end_time,
                total_usage=0.0,
                average_usage=0.0,
                peak_usage=0.0,
                min_usage=0.0,
                quota_utilization=0.0,
                waste_percentage=100.0,
                efficiency_score=0.0,
                usage_trend=TrendDirection.STABLE,
                growth_rate=0.0,
                violations_count=0,
                violation_rate=0.0,
                data_points=0
            )

        values = [dp.value for dp in filtered_data]

        # Basis-Metriken
        total_usage = sum(values)
        average_usage = statistics.mean(values)
        peak_usage = max(values)
        min_usage = min(values)

        # Effizienz-Metriken
        quota_utilization = (average_usage / current_quota * 100) if current_quota > 0 else 0.0
        waste_percentage = max(0.0, 100.0 - quota_utilization)
        efficiency_score = min(1.0, quota_utilization / 80.0)  # Optimal bei 80% Auslastung

        # Trend-Analyse
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if second_avg > first_avg * 1.1:
                usage_trend = TrendDirection.INCREASING
            elif second_avg < first_avg * 0.9:
                usage_trend = TrendDirection.DECREASING
            else:
                usage_trend = TrendDirection.STABLE

            # Wachstumsrate pro Tag
            time_span_days = (end_time - start_time).total_seconds() / 86400
            if time_span_days > 0 and first_avg > 0:
                growth_rate = ((second_avg - first_avg) / first_avg) * (1 / time_span_days) * 100
            else:
                growth_rate = 0.0
        else:
            usage_trend = TrendDirection.STABLE
            growth_rate = 0.0

        # Violations (vereinfacht)
        violations_count = sum(1 for value in values if value > current_quota)
        time_span_hours = (end_time - start_time).total_seconds() / 3600
        violation_rate = violations_count / time_span_hours if time_span_hours > 0 else 0.0

        return QuotaMetrics(
            quota_id=quota_id,
            scope_id=scope_id,
            start_time=start_time,
            end_time=end_time,
            total_usage=total_usage,
            average_usage=average_usage,
            peak_usage=peak_usage,
            min_usage=min_usage,
            quota_utilization=quota_utilization,
            waste_percentage=waste_percentage,
            efficiency_score=efficiency_score,
            usage_trend=usage_trend,
            growth_rate=growth_rate,
            violations_count=violations_count,
            violation_rate=violation_rate,
            data_points=len(filtered_data)
        )

    def generate_report(
        self,
        report_type: ReportType,
        quota_ids: list[str],
        scope_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        title: str | None = None
    ) -> QuotaReport:
        """Generiert Quota-Report."""
        import uuid

        report_id = str(uuid.uuid4())

        if title is None:
            title = f"{report_type.value.replace('_', ' ').title()} Report"

        report = QuotaReport(
            report_id=report_id,
            report_type=report_type,
            title=title,
            quota_ids=quota_ids,
            scope_ids=scope_ids,
            start_time=start_time,
            end_time=end_time
        )

        # Sammle Daten für Report
        for quota_id in quota_ids:
            for scope_id in scope_ids:
                key = f"{quota_id}:{scope_id}"

                # Patterns
                if key in self._patterns:
                    report.patterns.extend(self._patterns[key])

                # Predictions
                if key in self._predictions:
                    report.predictions.append(self._predictions[key])

        # Generiere Zusammenfassung
        report.summary = self._generate_report_summary(report)

        # Generiere Empfehlungen
        report.recommendations = self._generate_recommendations(report)

        self._reports[report_id] = report
        return report

    def _generate_report_summary(self, report: QuotaReport) -> dict[str, Any]:
        """Generiert Report-Zusammenfassung."""
        return {
            "total_quotas_analyzed": len(report.quota_ids),
            "total_scopes_analyzed": len(report.scope_ids),
            "patterns_detected": len(report.patterns),
            "predictions_generated": len(report.predictions),
            "time_span_hours": (report.end_time - report.start_time).total_seconds() / 3600,
            "analysis_timestamp": datetime.now(UTC).isoformat()
        }

    def _generate_recommendations(self, report: QuotaReport) -> list[str]:
        """Generiert Empfehlungen basierend auf Report."""
        recommendations = []

        # Analysiere Patterns
        bursty_patterns = [p for p in report.patterns if p.pattern_type == UsagePatternType.BURSTY]
        if bursty_patterns:
            recommendations.append("Consider implementing burst limits for irregular usage patterns")

        trending_up_patterns = [p for p in report.patterns if p.pattern_type == UsagePatternType.TRENDING_UP]
        if trending_up_patterns:
            recommendations.append("Monitor increasing usage trends and consider quota adjustments")

        # Analysiere Predictions
        high_exhaustion_predictions = [
            p for p in report.predictions
            if p.exhaustion_probability_24h > 0.8
        ]
        if high_exhaustion_predictions:
            recommendations.append("Immediate attention required: High probability of quota exhaustion within 24h")

        return recommendations

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Analytics-Statistiken zurück."""
        return {
            "analyses_performed": self._analyses_performed,
            "patterns_detected": self._patterns_detected,
            "predictions_made": self._predictions_made,
            "alerts_triggered": self._alerts_triggered,
            "tracked_quotas": len(self._usage_data),
            "cached_patterns": sum(len(patterns) for patterns in self._patterns.values()),
            "cached_predictions": len(self._predictions),
            "generated_reports": len(self._reports)
        }


# Globale Quota Analytics Instanz
quota_analytics = QuotaAnalytics()
