# backend/agents/slo_sla/capacity_planner.py
"""Capacity Planning System

Implementiert SLO-basierte Performance-Budgets, Trend-Analysis,
automatische Scaling-Recommendations und proaktives Load-Shedding.
"""

import asyncio
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

from .config import SLOSLAConfig
from .models import SLOMetrics

logger = get_logger(__name__)


class ScalingDirection(str, Enum):
    """Scaling-Richtungen."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_CHANGE = "no_change"


class CapacityMetric(str, Enum):
    """Capacity-Metriken."""

    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"


@dataclass
class PerformanceBudget:
    """Performance-Budget basierend auf SLOs."""

    capability: str
    agent_id: str | None = None

    # Budget-Limits basierend auf SLOs
    max_response_time_p95: float = 0.5  # 500ms
    max_error_rate: float = 0.01  # 1%
    min_availability: float = 0.999  # 99.9%
    max_cpu_utilization: float = 0.8  # 80%
    max_memory_utilization: float = 0.85  # 85%

    # Budget-Tracking
    current_response_time_p95: float = 0.0
    current_error_rate: float = 0.0
    current_availability: float = 1.0
    current_cpu_utilization: float = 0.0
    current_memory_utilization: float = 0.0

    # Budget-Status
    budget_exhausted: bool = False
    exhausted_metrics: list[str] = field(default_factory=list)

    # Time-Window
    window_hours: float = 24.0
    last_updated: float = field(default_factory=time.time)

    def update_metrics(
        self,
        response_time_p95: float | None = None,
        error_rate: float | None = None,
        availability: float | None = None,
        cpu_utilization: float | None = None,
        memory_utilization: float | None = None,
    ):
        """Aktualisiert Performance-Metriken."""
        if response_time_p95 is not None:
            self.current_response_time_p95 = response_time_p95
        if error_rate is not None:
            self.current_error_rate = error_rate
        if availability is not None:
            self.current_availability = availability
        if cpu_utilization is not None:
            self.current_cpu_utilization = cpu_utilization
        if memory_utilization is not None:
            self.current_memory_utilization = memory_utilization

        self.last_updated = time.time()
        self._check_budget_exhaustion()

    def _check_budget_exhaustion(self):
        """Prüft Budget-Exhaustion."""
        exhausted_metrics = []

        if self.current_response_time_p95 > self.max_response_time_p95:
            exhausted_metrics.append("response_time_p95")

        if self.current_error_rate > self.max_error_rate:
            exhausted_metrics.append("error_rate")

        if self.current_availability < self.min_availability:
            exhausted_metrics.append("availability")

        if self.current_cpu_utilization > self.max_cpu_utilization:
            exhausted_metrics.append("cpu_utilization")

        if self.current_memory_utilization > self.max_memory_utilization:
            exhausted_metrics.append("memory_utilization")

        self.exhausted_metrics = exhausted_metrics
        self.budget_exhausted = len(exhausted_metrics) > 0

    def get_budget_utilization(self) -> dict[str, float]:
        """Berechnet Budget-Auslastung (0.0 - 1.0+)."""
        return {
            "response_time_p95": (
                self.current_response_time_p95 / self.max_response_time_p95
                if self.max_response_time_p95 > 0
                else 0.0
            ),
            "error_rate": (
                self.current_error_rate / self.max_error_rate if self.max_error_rate > 0 else 0.0
            ),
            "availability": (
                (self.min_availability - self.current_availability) / (1.0 - self.min_availability)
                if self.min_availability < 1.0
                else 0.0
            ),
            "cpu_utilization": (
                self.current_cpu_utilization / self.max_cpu_utilization
                if self.max_cpu_utilization > 0
                else 0.0
            ),
            "memory_utilization": (
                self.current_memory_utilization / self.max_memory_utilization
                if self.max_memory_utilization > 0
                else 0.0
            ),
        }

    def get_worst_metric(self) -> tuple[str, float]:
        """Holt schlechteste Metrik (höchste Auslastung)."""
        utilization = self.get_budget_utilization()

        if not utilization:
            return "none", 0.0

        worst_metric = max(utilization.items(), key=lambda x: x[1])
        return worst_metric

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "capability": self.capability,
            "agent_id": self.agent_id,
            "budget_limits": {
                "max_response_time_p95": self.max_response_time_p95,
                "max_error_rate": self.max_error_rate,
                "min_availability": self.min_availability,
                "max_cpu_utilization": self.max_cpu_utilization,
                "max_memory_utilization": self.max_memory_utilization,
            },
            "current_metrics": {
                "response_time_p95": self.current_response_time_p95,
                "error_rate": self.current_error_rate,
                "availability": self.current_availability,
                "cpu_utilization": self.current_cpu_utilization,
                "memory_utilization": self.current_memory_utilization,
            },
            "budget_utilization": self.get_budget_utilization(),
            "budget_exhausted": self.budget_exhausted,
            "exhausted_metrics": self.exhausted_metrics,
            "worst_metric": self.get_worst_metric(),
            "last_updated": self.last_updated,
        }


@dataclass
class ScalingRecommendation:
    """Scaling-Empfehlung basierend auf Performance-Trends."""

    capability: str
    agent_id: str | None = None

    # Recommendation-Details
    direction: ScalingDirection = ScalingDirection.NO_CHANGE
    confidence: float = 0.0  # 0.0 - 1.0
    urgency: str = "low"  # low, medium, high, critical

    # Scaling-Parameter
    recommended_scale_factor: float = 1.0  # 1.2 = 20% increase, 0.8 = 20% decrease
    recommended_instance_count: int | None = None
    recommended_resource_adjustment: dict[str, float] = field(default_factory=dict)

    # Reasoning
    primary_reason: str = ""
    contributing_factors: list[str] = field(default_factory=list)
    expected_impact: str = ""

    # Timing
    recommended_execution_time: float = field(default_factory=time.time)
    estimated_completion_time: float = 300.0  # 5 Minuten

    # Risk-Assessment
    risk_level: str = "low"  # low, medium, high
    rollback_plan: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "capability": self.capability,
            "agent_id": self.agent_id,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "recommended_scale_factor": self.recommended_scale_factor,
            "recommended_instance_count": self.recommended_instance_count,
            "recommended_resource_adjustment": self.recommended_resource_adjustment,
            "primary_reason": self.primary_reason,
            "contributing_factors": self.contributing_factors,
            "expected_impact": self.expected_impact,
            "recommended_execution_time": self.recommended_execution_time,
            "estimated_completion_time": self.estimated_completion_time,
            "risk_level": self.risk_level,
            "rollback_plan": self.rollback_plan,
        }


class TrendAnalyzer:
    """Trend-Analyzer für Performance-Metriken."""

    def __init__(self, window_size: int = 100):
        """Initialisiert Trend-Analyzer.

        Args:
            window_size: Anzahl Datenpunkte für Trend-Analysis
        """
        self.window_size = window_size
        self.metric_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.RLock()

    def add_data_point(self, metric_name: str, value: float, timestamp: float | None = None):
        """Fügt Datenpunkt hinzu.

        Args:
            metric_name: Metrik-Name
            value: Metrik-Wert
            timestamp: Optional Timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        data_point = {"value": value, "timestamp": timestamp}

        with self._lock:
            self.metric_history[metric_name].append(data_point)

    def calculate_trend(self, metric_name: str) -> dict[str, Any]:
        """Berechnet Trend für Metrik.

        Args:
            metric_name: Metrik-Name

        Returns:
            Trend-Informationen
        """
        with self._lock:
            history = self.metric_history.get(metric_name, deque())

            if len(history) < 3:
                return {
                    "trend_direction": "unknown",
                    "trend_strength": 0.0,
                    "slope": 0.0,
                    "r_squared": 0.0,
                    "prediction_next": 0.0,
                    "data_points": len(history),
                }

            # Extrahiere Werte und Timestamps
            values = [point["value"] for point in history]
            timestamps = [point["timestamp"] for point in history]

            # Normalisiere Timestamps (relative Zeit)
            min_timestamp = min(timestamps)
            normalized_times = [(t - min_timestamp) for t in timestamps]

            # Lineare Regression
            slope, r_squared = self._linear_regression(normalized_times, values)

            # Trend-Richtung
            if abs(slope) < 0.001:  # Sehr kleiner Slope
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(1.0, abs(slope) * 100)  # Normalisiert
            else:
                trend_direction = "decreasing"
                trend_strength = min(1.0, abs(slope) * 100)

            # Prediction für nächsten Zeitpunkt
            next_time = (
                normalized_times[-1] + (normalized_times[-1] - normalized_times[-2])
                if len(normalized_times) >= 2
                else 0
            )
            prediction_next = values[-1] + (slope * (next_time - normalized_times[-1]))

            return {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "slope": slope,
                "r_squared": r_squared,
                "prediction_next": prediction_next,
                "data_points": len(history),
                "current_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "mean_value": statistics.mean(values),
            }

    def _linear_regression(
        self, x_values: list[float], y_values: list[float]
    ) -> tuple[float, float]:
        """Berechnet lineare Regression.

        Args:
            x_values: X-Werte (Zeit)
            y_values: Y-Werte (Metrik)

        Returns:
            Tuple von (slope, r_squared)
        """
        n = len(x_values)
        if n < 2:
            return 0.0, 0.0

        # Berechne Mittelwerte
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        # Berechne Slope
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=False))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0, 0.0

        slope = numerator / denominator

        # Berechne R-squared
        y_pred = [slope * (x - x_mean) + y_mean for x in x_values]
        ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_values, y_pred, strict=False))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return slope, max(0.0, r_squared)  # R-squared sollte nicht negativ sein

    def get_all_trends(self) -> dict[str, dict[str, Any]]:
        """Holt Trends für alle Metriken."""
        with self._lock:
            return {
                metric_name: self.calculate_trend(metric_name)
                for metric_name in self.metric_history.keys()
            }


class CapacityPlanner:
    """Haupt-Capacity-Planner für SLO-basierte Performance-Budgets."""

    def __init__(self, config: SLOSLAConfig):
        """Initialisiert Capacity-Planner.

        Args:
            config: SLO/SLA-Konfiguration
        """
        self.config = config

        # Performance-Budgets
        self.performance_budgets: dict[str, PerformanceBudget] = {}

        # Trend-Analyzer
        self.trend_analyzer = TrendAnalyzer()

        # Scaling-Recommendations
        self.scaling_recommendations: dict[str, ScalingRecommendation] = {}
        self.recommendation_history: list[ScalingRecommendation] = []

        # Load-Shedding-State
        self.load_shedding_active: dict[str, bool] = defaultdict(bool)
        self.load_shedding_thresholds: dict[str, float] = defaultdict(
            lambda: 0.9
        )  # 90% Budget-Auslastung

        # Thread-Safety
        self._lock = threading.RLock()

        # Planning-Task
        self._planning_task: asyncio.Task | None = None

        # Metrics
        self._metrics_collector = MetricsCollector()

        # Callbacks
        from collections.abc import Awaitable, Callable

        self._scaling_callbacks: list[Callable[[ScalingRecommendation], Awaitable[None]]] = []
        self._load_shedding_callbacks: list[Callable[[str, bool], Awaitable[None]]] = []

    def register_scaling_callback(
        self, callback: Callable[[ScalingRecommendation], Awaitable[None]]
    ):
        """Registriert Callback für Scaling-Recommendations."""
        self._scaling_callbacks.append(callback)

    def register_load_shedding_callback(self, callback: Callable[[str, bool], Awaitable[None]]):
        """Registriert Callback für Load-Shedding-Events."""
        self._load_shedding_callbacks.append(callback)

    def create_performance_budget(
        self,
        capability: str,
        agent_id: str | None = None,
        _slo_metrics: SLOMetrics | None = None,
    ) -> PerformanceBudget:
        """Erstellt Performance-Budget basierend auf SLOs.

        Args:
            capability: Capability-Name
            agent_id: Optional Agent-ID
            _slo_metrics: Optional SLO-Metriken für Budget-Ableitung

        Returns:
            Performance-Budget
        """
        budget_key = f"{agent_id or 'global'}.{capability}"

        # Ableitung von Budget-Limits aus SLO-Konfiguration
        slo_config = self.config.slo_config.get_capability_slo_config(capability)

        budget = PerformanceBudget(
            capability=capability,
            agent_id=agent_id,
            max_response_time_p95=(
                slo_config.latency_p95_ms / 1000.0 if slo_config.latency_p95_ms else 0.5
            ),
            max_error_rate=(
                slo_config.error_rate_percent / 100.0 if slo_config.error_rate_percent else 0.01
            ),
            min_availability=(
                slo_config.availability_percent / 100.0
                if slo_config.availability_percent
                else 0.999
            ),
            window_hours=self.config.performance_budget_window_hours,
        )

        with self._lock:
            self.performance_budgets[budget_key] = budget

        return budget

    def update_performance_budget(self, capability: str, agent_id: str | None = None, **metrics):
        """Aktualisiert Performance-Budget mit neuen Metriken.

        Args:
            capability: Capability-Name
            agent_id: Optional Agent-ID
            **metrics: Performance-Metriken
        """
        budget_key = f"{agent_id or 'global'}.{capability}"

        with self._lock:
            if budget_key not in self.performance_budgets:
                self.create_performance_budget(capability, agent_id)

            budget = self.performance_budgets[budget_key]
            budget.update_metrics(**metrics)

            # Trend-Daten hinzufügen
            for metric_name, value in metrics.items():
                if value is not None:
                    trend_key = f"{budget_key}.{metric_name}"
                    self.trend_analyzer.add_data_point(trend_key, value)

            # Prüfe Load-Shedding
            self._check_load_shedding(budget_key, budget)

    def _check_load_shedding(self, budget_key: str, budget: PerformanceBudget):
        """Prüft ob Load-Shedding aktiviert werden soll."""
        worst_metric, worst_utilization = budget.get_worst_metric()
        threshold = self.load_shedding_thresholds[budget_key]

        should_shed_load = worst_utilization >= threshold
        currently_shedding = self.load_shedding_active[budget_key]

        if should_shed_load and not currently_shedding:
            # Aktiviere Load-Shedding
            self.load_shedding_active[budget_key] = True

            logger.warning(
                f"Load-Shedding aktiviert für {budget_key}: "
                f"{worst_metric} bei {worst_utilization:.1%} Auslastung"
            )

            # Metrics
            self._metrics_collector.increment_counter(
                "capacity.load_shedding.activated",
                tags={
                    "capability": budget.capability,
                    "agent_id": budget.agent_id or "global",
                    "trigger_metric": worst_metric,
                },
            )

            # Callbacks
            asyncio.create_task(self._notify_load_shedding_callbacks(budget_key, True))

        elif not should_shed_load and currently_shedding:
            # Deaktiviere Load-Shedding
            self.load_shedding_active[budget_key] = False

            logger.info(f"Load-Shedding deaktiviert für {budget_key}")

            # Metrics
            self._metrics_collector.increment_counter(
                "capacity.load_shedding.deactivated",
                tags={"capability": budget.capability, "agent_id": budget.agent_id or "global"},
            )

            # Callbacks
            asyncio.create_task(self._notify_load_shedding_callbacks(budget_key, False))

    async def _notify_load_shedding_callbacks(self, budget_key: str, active: bool):
        """Benachrichtigt Load-Shedding-Callbacks."""
        for callback in self._load_shedding_callbacks:
            try:
                await callback(budget_key, active)
            except Exception as e:
                logger.error(f"Fehler in Load-Shedding-Callback: {e}")

    async def generate_scaling_recommendations(self) -> list[ScalingRecommendation]:
        """Generiert Scaling-Empfehlungen basierend auf Trends.

        Returns:
            Liste von Scaling-Empfehlungen
        """
        recommendations = []

        with self._lock:
            for budget_key, budget in self.performance_budgets.items():
                recommendation = await self._analyze_scaling_need(budget_key, budget)

                if recommendation.direction != ScalingDirection.NO_CHANGE:
                    recommendations.append(recommendation)
                    self.scaling_recommendations[budget_key] = recommendation
                    self.recommendation_history.append(recommendation)

                    # Behalte nur letzte 1000 Recommendations
                    if len(self.recommendation_history) > 1000:
                        self.recommendation_history = self.recommendation_history[-1000:]

        return recommendations

    async def _analyze_scaling_need(
        self, budget_key: str, budget: PerformanceBudget
    ) -> ScalingRecommendation:
        """Analysiert Scaling-Bedarf für Budget.

        Args:
            budget_key: Budget-Key
            budget: Performance-Budget

        Returns:
            Scaling-Empfehlung
        """
        recommendation = ScalingRecommendation(
            capability=budget.capability, agent_id=budget.agent_id
        )

        # Analysiere Trends für alle Metriken
        utilization = budget.get_budget_utilization()
        trends = {}

        for metric_name in utilization.keys():
            trend_key = f"{budget_key}.{metric_name}"
            trends[metric_name] = self.trend_analyzer.calculate_trend(trend_key)

        # Finde kritischste Metrik
        worst_metric, worst_utilization = budget.get_worst_metric()
        worst_trend = trends.get(worst_metric, {})

        # Scaling-Entscheidung basierend auf Utilization und Trend
        if worst_utilization >= 1.0:  # Budget überschritten
            recommendation.direction = ScalingDirection.SCALE_OUT
            recommendation.urgency = "critical"
            recommendation.confidence = 0.9
            recommendation.primary_reason = (
                f"{worst_metric} budget exceeded ({worst_utilization:.1%})"
            )

        elif worst_utilization >= 0.8 and worst_trend.get("trend_direction") == "increasing":
            # Hohe Auslastung mit steigendem Trend
            recommendation.direction = ScalingDirection.SCALE_OUT
            recommendation.urgency = "high"
            recommendation.confidence = 0.8
            recommendation.primary_reason = f"{worst_metric} trending up ({worst_utilization:.1%})"

        elif worst_utilization <= 0.3 and worst_trend.get("trend_direction") == "decreasing":
            # Niedrige Auslastung mit fallendem Trend
            recommendation.direction = ScalingDirection.SCALE_IN
            recommendation.urgency = "low"
            recommendation.confidence = 0.6
            recommendation.primary_reason = (
                f"{worst_metric} trending down ({worst_utilization:.1%})"
            )

        else:
            # Keine Änderung nötig
            recommendation.direction = ScalingDirection.NO_CHANGE
            recommendation.confidence = 0.5
            recommendation.primary_reason = "Performance within acceptable range"

        # Berechne Scaling-Factor
        if recommendation.direction in [ScalingDirection.SCALE_OUT, ScalingDirection.SCALE_UP]:
            # Scale-Out: Basierend auf Überschreitung
            excess = max(0.0, worst_utilization - 0.7)  # Target 70% Utilization
            recommendation.recommended_scale_factor = 1.0 + (excess * 0.5)  # Max 50% Increase

        elif recommendation.direction in [ScalingDirection.SCALE_IN, ScalingDirection.SCALE_DOWN]:
            # Scale-In: Basierend auf Unterauslastung
            underutilization = max(0.0, 0.5 - worst_utilization)  # Target min 50%
            recommendation.recommended_scale_factor = 1.0 - (
                underutilization * 0.3
            )  # Max 30% Decrease

        # Contributing Factors
        contributing_factors = []
        for metric_name, util in utilization.items():
            if util >= 0.7:
                contributing_factors.append(f"{metric_name}: {util:.1%}")

        recommendation.contributing_factors = contributing_factors

        # Expected Impact
        if recommendation.direction != ScalingDirection.NO_CHANGE:
            recommendation.expected_impact = (
                f"Reduce {worst_metric} utilization by "
                f"{abs(1.0 - recommendation.recommended_scale_factor) * 100:.0f}%"
            )

        return recommendation

    def start_capacity_planning(self):
        """Startet Capacity-Planning."""
        if self._planning_task is None or self._planning_task.done():
            self._planning_task = asyncio.create_task(self._planning_loop())

    def stop_capacity_planning(self):
        """Stoppt Capacity-Planning."""
        if self._planning_task and not self._planning_task.done():
            self._planning_task.cancel()

    async def _planning_loop(self):
        """Capacity-Planning-Loop."""
        while True:
            try:
                interval_minutes = self.config.capacity_planning_interval_minutes
                await asyncio.sleep(interval_minutes * 60)

                # Generiere Scaling-Recommendations
                recommendations = await self.generate_scaling_recommendations()

                # Benachrichtige Callbacks
                for recommendation in recommendations:
                    for callback in self._scaling_callbacks:
                        try:
                            await callback(recommendation)
                        except Exception as e:
                            logger.error(f"Fehler in Scaling-Callback: {e}")

                logger.info(f"Capacity-Planning abgeschlossen: {len(recommendations)} Empfehlungen")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Capacity-Planning: {e}")

    def get_performance_budgets(self) -> dict[str, PerformanceBudget]:
        """Holt alle Performance-Budgets."""
        with self._lock:
            return self.performance_budgets.copy()

    def get_scaling_recommendations(self) -> dict[str, ScalingRecommendation]:
        """Holt aktuelle Scaling-Empfehlungen."""
        with self._lock:
            return self.scaling_recommendations.copy()

    def get_load_shedding_status(self) -> dict[str, bool]:
        """Holt Load-Shedding-Status."""
        with self._lock:
            return dict(self.load_shedding_active)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt Capacity-Planner-Metriken-Zusammenfassung."""
        with self._lock:
            return {
                "performance_budgets": len(self.performance_budgets),
                "budgets_exhausted": sum(
                    1 for budget in self.performance_budgets.values() if budget.budget_exhausted
                ),
                "active_load_shedding": sum(
                    1 for active in self.load_shedding_active.values() if active
                ),
                "scaling_recommendations": len(self.scaling_recommendations),
                "recommendation_history_24h": len(
                    [
                        r
                        for r in self.recommendation_history
                        if r.recommended_execution_time >= time.time() - 86400
                    ]
                ),
                "trends": self.trend_analyzer.get_all_trends(),
            }
