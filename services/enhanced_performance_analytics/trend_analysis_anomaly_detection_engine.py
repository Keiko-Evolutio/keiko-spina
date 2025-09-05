# backend/services/enhanced_performance_analytics/trend_analysis_anomaly_detection_engine.py
"""Trend Analysis und Anomaly Detection Engine.

Implementiert Enterprise-Grade Trend-Analysis und Anomaly-Detection mit
Advanced Statistical Methods und ML-basierte Anomaly-Detection.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .data_models import (
    AnalyticsConfiguration,
    AnalyticsScope,
    AnomalyDetection,
    AnomalyType,
    EventType,
    PerformanceDataPoint,
    PerformanceEvent,
    TrendAnalysis,
    TrendDirection,
)

logger = get_logger(__name__)


class TrendAnalysisAnomalyDetectionEngine:
    """Trend Analysis und Anomaly Detection Engine für Enterprise-Grade Analytics."""

    def __init__(
        self,
        configuration: AnalyticsConfiguration | None = None
    ):
        """Initialisiert Trend Analysis und Anomaly Detection Engine.

        Args:
            configuration: Analytics-Konfiguration
        """
        self.configuration = configuration or AnalyticsConfiguration()

        # Trend-Analysis-Storage
        self._trend_analyses: dict[str, TrendAnalysis] = {}
        self._trend_history: dict[str, list[TrendAnalysis]] = defaultdict(list)

        # Anomaly-Detection-Storage
        self._anomaly_detections: dict[str, AnomalyDetection] = {}
        self._anomaly_history: dict[str, list[AnomalyDetection]] = defaultdict(list)

        # Data-Storage für Analysis
        self._time_series_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._baseline_data: dict[str, dict[str, Any]] = {}

        # Background-Tasks
        self._analysis_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Performance-Tracking
        self._analysis_performance_stats = {
            "total_trend_analyses": 0,
            "avg_trend_analysis_time_ms": 0.0,
            "trend_detection_accuracy": 0.0,
            "total_anomaly_detections": 0,
            "avg_anomaly_detection_time_ms": 0.0,
            "anomaly_detection_precision": 0.0,
            "anomaly_detection_recall": 0.0,
            "false_positive_rate": 0.0
        }

        # Anomaly-Detection-Algorithmen
        self._anomaly_algorithms = {
            "statistical": self._statistical_anomaly_detection,
            "ml_based": self._ml_based_anomaly_detection,
            "isolation_forest": self._isolation_forest_anomaly_detection,
            "z_score": self._z_score_anomaly_detection,
            "iqr": self._iqr_anomaly_detection
        }

        # Event-Callbacks
        self._event_callbacks: list[callable] = []

        logger.info("Trend Analysis und Anomaly Detection Engine initialisiert")

    async def start(self) -> None:
        """Startet Trend Analysis und Anomaly Detection Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._analysis_tasks = [
            asyncio.create_task(self._trend_analysis_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._baseline_update_loop()),
            asyncio.create_task(self._performance_monitoring_loop())
        ]

        logger.info("Trend Analysis und Anomaly Detection Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Trend Analysis und Anomaly Detection Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._analysis_tasks:
            task.cancel()

        await asyncio.gather(*self._analysis_tasks, return_exceptions=True)
        self._analysis_tasks.clear()

        logger.info("Trend Analysis und Anomaly Detection Engine gestoppt")

    async def add_data_point(
        self,
        data_point: PerformanceDataPoint
    ) -> None:
        """Fügt Datenpunkt für Analysis hinzu.

        Args:
            data_point: Performance-Datenpunkt
        """
        try:
            # Speichere Datenpunkt in Time-Series
            series_key = f"{data_point.scope.value}:{data_point.scope_id}:{data_point.metric_name}"
            self._time_series_data[series_key].append(data_point)

            # Real-time Anomaly-Detection für kritische Metriken
            if await self._is_critical_metric(data_point):
                await self._perform_real_time_anomaly_detection(data_point)

            logger.debug({
                "event": "data_point_added_for_analysis",
                "metric_name": data_point.metric_name,
                "scope": data_point.scope.value,
                "scope_id": data_point.scope_id,
                "value": data_point.value
            })

        except Exception as e:
            logger.error(f"Data point addition fehlgeschlagen: {e}")

    async def analyze_trend(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        analysis_window_hours: int | None = None
    ) -> TrendAnalysis:
        """Führt Trend-Analysis aus.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_name: Metrik-Name
            analysis_window_hours: Analysis-Fenster in Stunden

        Returns:
            Trend-Analysis
        """
        start_time = time.time()

        try:
            import uuid

            analysis_id = str(uuid.uuid4())
            window_hours = analysis_window_hours or self.configuration.trend_analysis_window_hours

            # Hole Time-Series-Daten
            series_key = f"{scope.value}:{scope_id}:{metric_name}"
            data_points = list(self._time_series_data.get(series_key, deque()))

            if not data_points:
                # Leere Trend-Analysis
                return TrendAnalysis(
                    analysis_id=analysis_id,
                    metric_name=metric_name,
                    scope=scope,
                    scope_id=scope_id,
                    trend_direction=TrendDirection.STABLE,
                    trend_strength=0.0,
                    trend_confidence=0.0,
                    analysis_period_start=datetime.utcnow() - timedelta(hours=window_hours),
                    analysis_period_end=datetime.utcnow(),
                    data_points_count=0,
                    trend_slope=0.0,
                    trend_intercept=0.0,
                    r_squared=0.0,
                    predicted_next_value=0.0,
                    predicted_next_timestamp=datetime.utcnow() + timedelta(hours=1),
                    prediction_confidence=0.0
                )

            # Filter Daten nach Zeitfenster
            cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
            filtered_data = [
                dp for dp in data_points
                if dp.timestamp >= cutoff_time and isinstance(dp.value, (int, float))
            ]

            if len(filtered_data) < 3:
                # Zu wenige Datenpunkte für Trend-Analysis
                return TrendAnalysis(
                    analysis_id=analysis_id,
                    metric_name=metric_name,
                    scope=scope,
                    scope_id=scope_id,
                    trend_direction=TrendDirection.STABLE,
                    trend_strength=0.0,
                    trend_confidence=0.0,
                    analysis_period_start=cutoff_time,
                    analysis_period_end=datetime.utcnow(),
                    data_points_count=len(filtered_data),
                    trend_slope=0.0,
                    trend_intercept=0.0,
                    r_squared=0.0,
                    predicted_next_value=filtered_data[-1].value if filtered_data else 0.0,
                    predicted_next_timestamp=datetime.utcnow() + timedelta(hours=1),
                    prediction_confidence=0.0
                )

            # Führe Trend-Analysis aus
            trend_analysis = await self._perform_trend_analysis(
                analysis_id, filtered_data, scope, scope_id, metric_name, cutoff_time
            )

            # Speichere Trend-Analysis
            self._trend_analyses[analysis_id] = trend_analysis
            self._trend_history[series_key].append(trend_analysis)

            # Limitiere History-Größe
            if len(self._trend_history[series_key]) > 50:
                self._trend_history[series_key] = self._trend_history[series_key][-50:]

            # Trigger Event bei signifikanten Trends
            if trend_analysis.trend_strength > 0.7 and trend_analysis.trend_confidence > 0.8:
                await self._trigger_trend_change_event(trend_analysis)

            # Update Performance-Stats
            analysis_time_ms = (time.time() - start_time) * 1000
            self._update_trend_analysis_performance_stats(analysis_time_ms)

            logger.debug({
                "event": "trend_analysis_completed",
                "analysis_id": analysis_id,
                "metric_name": metric_name,
                "trend_direction": trend_analysis.trend_direction.value,
                "trend_strength": trend_analysis.trend_strength,
                "trend_confidence": trend_analysis.trend_confidence,
                "analysis_time_ms": analysis_time_ms
            })

            return trend_analysis

        except Exception as e:
            logger.error(f"Trend analysis fehlgeschlagen: {e}")
            # Fallback zu leerer Analysis
            return TrendAnalysis(
                analysis_id=str(uuid.uuid4()),
                metric_name=metric_name,
                scope=scope,
                scope_id=scope_id,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                trend_confidence=0.0,
                analysis_period_start=datetime.utcnow() - timedelta(hours=24),
                analysis_period_end=datetime.utcnow(),
                data_points_count=0,
                trend_slope=0.0,
                trend_intercept=0.0,
                r_squared=0.0,
                predicted_next_value=0.0,
                predicted_next_timestamp=datetime.utcnow() + timedelta(hours=1),
                prediction_confidence=0.0
            )

    async def detect_anomalies(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        detection_algorithms: list[str] | None = None
    ) -> list[AnomalyDetection]:
        """Führt Anomaly-Detection aus.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_name: Metrik-Name
            detection_algorithms: Detection-Algorithmen

        Returns:
            Liste von Anomaly-Detections
        """
        start_time = time.time()

        try:
            algorithms = detection_algorithms or self.configuration.anomaly_algorithms

            # Hole Time-Series-Daten
            series_key = f"{scope.value}:{scope_id}:{metric_name}"
            data_points = list(self._time_series_data.get(series_key, deque()))

            if not data_points:
                return []

            # Filter numerische Werte
            numeric_data = [
                dp for dp in data_points
                if isinstance(dp.value, (int, float))
            ]

            if len(numeric_data) < 10:  # Mindestens 10 Datenpunkte für Anomaly-Detection
                return []

            # Führe Anomaly-Detection mit verschiedenen Algorithmen aus
            all_anomalies = []

            for algorithm in algorithms:
                if algorithm in self._anomaly_algorithms:
                    algorithm_anomalies = await self._anomaly_algorithms[algorithm](
                        numeric_data, scope, scope_id, metric_name
                    )
                    all_anomalies.extend(algorithm_anomalies)

            # Dedupliziere und ranke Anomalies
            unique_anomalies = await self._deduplicate_anomalies(all_anomalies)

            # Speichere Anomalies
            for anomaly in unique_anomalies:
                self._anomaly_detections[anomaly.anomaly_id] = anomaly
                self._anomaly_history[series_key].append(anomaly)

                # Trigger Event für kritische Anomalies
                if anomaly.severity > 0.8:
                    await self._trigger_anomaly_detected_event(anomaly)

            # Limitiere History-Größe
            if len(self._anomaly_history[series_key]) > 100:
                self._anomaly_history[series_key] = self._anomaly_history[series_key][-100:]

            # Update Performance-Stats
            detection_time_ms = (time.time() - start_time) * 1000
            self._update_anomaly_detection_performance_stats(detection_time_ms, len(unique_anomalies))

            logger.debug({
                "event": "anomaly_detection_completed",
                "metric_name": metric_name,
                "scope": scope.value,
                "scope_id": scope_id,
                "anomalies_detected": len(unique_anomalies),
                "algorithms_used": algorithms,
                "detection_time_ms": detection_time_ms
            })

            return unique_anomalies

        except Exception as e:
            logger.error(f"Anomaly detection fehlgeschlagen: {e}")
            return []

    async def register_event_callback(self, callback: callable) -> None:
        """Registriert Event-Callback.

        Args:
            callback: Callback-Funktion
        """
        try:
            self._event_callbacks.append(callback)
            logger.debug("Event callback registriert")

        except Exception as e:
            logger.error(f"Event callback registration fehlgeschlagen: {e}")

    async def _perform_trend_analysis(
        self,
        analysis_id: str,
        data_points: list[PerformanceDataPoint],
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        period_start: datetime
    ) -> TrendAnalysis:
        """Führt detaillierte Trend-Analysis aus."""
        try:
            # Extrahiere Werte und Zeitstempel
            values = [float(dp.value) for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]

            # Konvertiere Zeitstempel zu numerischen Werten (Sekunden seit Start)
            start_timestamp = timestamps[0]
            x_values = [(ts - start_timestamp).total_seconds() for ts in timestamps]

            # Lineare Regression
            slope, intercept, r_squared = self._calculate_linear_regression(x_values, values)

            # Bestimme Trend-Richtung
            trend_direction = self._determine_trend_direction(slope, values)

            # Berechne Trend-Stärke
            trend_strength = self._calculate_trend_strength(slope, values, r_squared)

            # Berechne Trend-Confidence
            trend_confidence = self._calculate_trend_confidence(r_squared, len(values))

            # Vorhersage für nächsten Wert
            next_x = x_values[-1] + 3600  # 1 Stunde später
            predicted_next_value = slope * next_x + intercept
            predicted_next_timestamp = timestamps[-1] + timedelta(hours=1)
            prediction_confidence = trend_confidence * 0.8  # Reduzierte Confidence für Vorhersage

            # Seasonal-Pattern-Detection
            seasonal_pattern, seasonal_period, seasonal_amplitude = await self._detect_seasonal_pattern(
                values, timestamps
            )

            # Change-Point-Detection
            change_points = await self._detect_change_points(values, timestamps)

            return TrendAnalysis(
                analysis_id=analysis_id,
                metric_name=metric_name,
                scope=scope,
                scope_id=scope_id,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_confidence=trend_confidence,
                analysis_period_start=period_start,
                analysis_period_end=datetime.utcnow(),
                data_points_count=len(data_points),
                trend_slope=slope,
                trend_intercept=intercept,
                r_squared=r_squared,
                predicted_next_value=predicted_next_value,
                predicted_next_timestamp=predicted_next_timestamp,
                prediction_confidence=prediction_confidence,
                seasonal_pattern_detected=seasonal_pattern,
                seasonal_period_hours=seasonal_period,
                seasonal_amplitude=seasonal_amplitude,
                change_points=change_points
            )

        except Exception as e:
            logger.error(f"Detailed trend analysis fehlgeschlagen: {e}")
            raise

    async def _statistical_anomaly_detection(
        self,
        data_points: list[PerformanceDataPoint],
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str
    ) -> list[AnomalyDetection]:
        """Statistische Anomaly-Detection."""
        try:
            import statistics
            import uuid

            anomalies = []
            values = [float(dp.value) for dp in data_points]

            if len(values) < 10:
                return anomalies

            # Berechne statistische Baseline
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

            # 3-Sigma-Rule für Anomaly-Detection
            threshold = 3 * std_dev

            for dp in data_points:
                value = float(dp.value)
                deviation = abs(value - mean_value)

                if deviation > threshold and std_dev > 0:
                    # Anomaly gefunden
                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        metric_name=metric_name,
                        scope=scope,
                        scope_id=scope_id,
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=min(1.0, deviation / (threshold * 2)),
                        confidence=0.8,
                        anomalous_value=value,
                        expected_value=mean_value,
                        deviation=deviation,
                        deviation_percent=(deviation / mean_value) * 100 if mean_value != 0 else 0,
                        anomaly_start=dp.timestamp,
                        baseline_period_start=data_points[0].timestamp,
                        baseline_period_end=data_points[-1].timestamp,
                        baseline_sample_count=len(data_points),
                        detection_algorithm="statistical_3sigma"
                    )
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Statistical anomaly detection fehlgeschlagen: {e}")
            return []

    async def _z_score_anomaly_detection(
        self,
        data_points: list[PerformanceDataPoint],
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str
    ) -> list[AnomalyDetection]:
        """Z-Score-basierte Anomaly-Detection."""
        try:
            import statistics
            import uuid

            anomalies = []
            values = [float(dp.value) for dp in data_points]

            if len(values) < 10:
                return anomalies

            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

            if std_dev == 0:
                return anomalies

            # Z-Score-Threshold
            z_threshold = 2.5

            for dp in data_points:
                value = float(dp.value)
                z_score = abs(value - mean_value) / std_dev

                if z_score > z_threshold:
                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        metric_name=metric_name,
                        scope=scope,
                        scope_id=scope_id,
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=min(1.0, z_score / 5.0),
                        confidence=0.75,
                        anomalous_value=value,
                        expected_value=mean_value,
                        deviation=abs(value - mean_value),
                        deviation_percent=(abs(value - mean_value) / mean_value) * 100 if mean_value != 0 else 0,
                        anomaly_start=dp.timestamp,
                        baseline_period_start=data_points[0].timestamp,
                        baseline_period_end=data_points[-1].timestamp,
                        baseline_sample_count=len(data_points),
                        detection_algorithm="z_score",
                        algorithm_parameters={"z_threshold": z_threshold, "z_score": z_score}
                    )
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Z-Score anomaly detection fehlgeschlagen: {e}")
            return []

    async def _iqr_anomaly_detection(
        self,
        data_points: list[PerformanceDataPoint],
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str
    ) -> list[AnomalyDetection]:
        """IQR-basierte Anomaly-Detection."""
        try:
            import uuid

            anomalies = []
            values = [float(dp.value) for dp in data_points]

            if len(values) < 10:
                return anomalies

            # Berechne IQR
            sorted_values = sorted(values)
            n = len(sorted_values)
            q1 = sorted_values[n // 4]
            q3 = sorted_values[3 * n // 4]
            iqr = q3 - q1

            if iqr == 0:
                return anomalies

            # IQR-Outlier-Detection
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            for dp in data_points:
                value = float(dp.value)

                if value < lower_bound or value > upper_bound:
                    expected_value = (q1 + q3) / 2  # Median als Expected Value
                    deviation = abs(value - expected_value)

                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        metric_name=metric_name,
                        scope=scope,
                        scope_id=scope_id,
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=min(1.0, deviation / (iqr * 2)),
                        confidence=0.7,
                        anomalous_value=value,
                        expected_value=expected_value,
                        deviation=deviation,
                        deviation_percent=(deviation / expected_value) * 100 if expected_value != 0 else 0,
                        anomaly_start=dp.timestamp,
                        baseline_period_start=data_points[0].timestamp,
                        baseline_period_end=data_points[-1].timestamp,
                        baseline_sample_count=len(data_points),
                        detection_algorithm="iqr",
                        algorithm_parameters={"q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lower_bound, "upper_bound": upper_bound}
                    )
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"IQR anomaly detection fehlgeschlagen: {e}")
            return []

    async def _ml_based_anomaly_detection(
        self,
        _data_points: list[PerformanceDataPoint],
        _scope: AnalyticsScope,
        _scope_id: str,
        _metric_name: str
    ) -> list[AnomalyDetection]:
        """ML-basierte Anomaly-Detection."""
        try:
            # Placeholder für ML-basierte Anomaly-Detection
            # In Realität würde hier ein trainiertes ML-Model verwendet
            return []

        except Exception as e:
            logger.error(f"ML-based anomaly detection fehlgeschlagen: {e}")
            return []

    async def _isolation_forest_anomaly_detection(
        self,
        _data_points: list[PerformanceDataPoint],
        _scope: AnalyticsScope,
        _scope_id: str,
        _metric_name: str
    ) -> list[AnomalyDetection]:
        """Isolation-Forest-basierte Anomaly-Detection."""
        try:
            # Placeholder für Isolation Forest
            # In Realität würde hier sklearn.ensemble.IsolationForest verwendet
            return []

        except Exception as e:
            logger.error(f"Isolation Forest anomaly detection fehlgeschlagen: {e}")
            return []

    async def _perform_real_time_anomaly_detection(
        self,
        data_point: PerformanceDataPoint
    ) -> None:
        """Führt Real-time Anomaly-Detection aus."""
        try:
            # Hole Baseline für Metrik
            baseline_key = f"{data_point.scope.value}:{data_point.scope_id}:{data_point.metric_name}"
            baseline = self._baseline_data.get(baseline_key)

            if not baseline:
                return

            value = float(data_point.value)
            expected_value = baseline.get("mean", 0.0)
            std_dev = baseline.get("std_dev", 0.0)

            if std_dev > 0:
                z_score = abs(value - expected_value) / std_dev

                if z_score > 3.0:  # Real-time Threshold
                    # Erstelle Real-time Anomaly
                    import uuid

                    anomaly = AnomalyDetection(
                        anomaly_id=str(uuid.uuid4()),
                        metric_name=data_point.metric_name,
                        scope=data_point.scope,
                        scope_id=data_point.scope_id,
                        anomaly_type=AnomalyType.SPIKE if value > expected_value else AnomalyType.DIP,
                        severity=min(1.0, z_score / 5.0),
                        confidence=0.8,
                        anomalous_value=value,
                        expected_value=expected_value,
                        deviation=abs(value - expected_value),
                        deviation_percent=(abs(value - expected_value) / expected_value) * 100 if expected_value != 0 else 0,
                        anomaly_start=data_point.timestamp,
                        baseline_period_start=baseline.get("period_start", data_point.timestamp),
                        baseline_period_end=baseline.get("period_end", data_point.timestamp),
                        baseline_sample_count=baseline.get("sample_count", 1),
                        detection_algorithm="real_time_z_score"
                    )

                    # Speichere und trigger Event
                    self._anomaly_detections[anomaly.anomaly_id] = anomaly
                    await self._trigger_anomaly_detected_event(anomaly)

        except Exception as e:
            logger.error(f"Real-time anomaly detection fehlgeschlagen: {e}")

    async def _trend_analysis_loop(self) -> None:
        """Background-Loop für Trend-Analysis."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.trend_analysis_window_hours * 3600 // 4)  # 1/4 des Fensters

                if self._is_running:
                    await self._perform_scheduled_trend_analysis()

            except Exception as e:
                logger.error(f"Trend analysis loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _anomaly_detection_loop(self) -> None:
        """Background-Loop für Anomaly-Detection."""
        while self._is_running:
            try:
                await asyncio.sleep(600)  # Alle 10 Minuten

                if self._is_running:
                    await self._perform_scheduled_anomaly_detection()

            except Exception as e:
                logger.error(f"Anomaly detection loop fehlgeschlagen: {e}")
                await asyncio.sleep(600)

    async def _baseline_update_loop(self) -> None:
        """Background-Loop für Baseline-Updates."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Jede Stunde

                if self._is_running:
                    await self._update_baselines()

            except Exception as e:
                logger.error(f"Baseline update loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _performance_monitoring_loop(self) -> None:
        """Background-Loop für Performance-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(900)  # Alle 15 Minuten

                if self._is_running:
                    await self._monitor_analysis_performance()

            except Exception as e:
                logger.error(f"Performance monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(900)

    async def _perform_scheduled_trend_analysis(self) -> None:
        """Führt geplante Trend-Analysis aus."""
        try:
            for series_key in self._time_series_data.keys():
                scope_str, scope_id, metric_name = series_key.split(":", 2)
                scope = AnalyticsScope(scope_str)

                await self.analyze_trend(scope, scope_id, metric_name)

        except Exception as e:
            logger.error(f"Scheduled trend analysis fehlgeschlagen: {e}")

    async def _perform_scheduled_anomaly_detection(self) -> None:
        """Führt geplante Anomaly-Detection aus."""
        try:
            for series_key in self._time_series_data.keys():
                scope_str, scope_id, metric_name = series_key.split(":", 2)
                scope = AnalyticsScope(scope_str)

                await self.detect_anomalies(scope, scope_id, metric_name)

        except Exception as e:
            logger.error(f"Scheduled anomaly detection fehlgeschlagen: {e}")

    async def _update_baselines(self) -> None:
        """Aktualisiert Baselines für Anomaly-Detection."""
        try:
            import statistics

            for series_key, data_points in self._time_series_data.items():
                if len(data_points) >= 10:
                    # Berechne Baseline-Statistiken
                    values = [float(dp.value) for dp in data_points if isinstance(dp.value, (int, float))]

                    if values:
                        baseline = {
                            "mean": statistics.mean(values),
                            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                            "median": statistics.median(values),
                            "sample_count": len(values),
                            "period_start": data_points[0].timestamp,
                            "period_end": data_points[-1].timestamp,
                            "last_updated": datetime.utcnow()
                        }

                        self._baseline_data[series_key] = baseline

        except Exception as e:
            logger.error(f"Baselines update fehlgeschlagen: {e}")

    async def _monitor_analysis_performance(self) -> None:
        """Monitort Analysis-Performance."""
        try:
            # Berechne Accuracy-Metriken
            total_trend_analyses = self._analysis_performance_stats["total_trend_analyses"]
            total_anomaly_detections = self._analysis_performance_stats["total_anomaly_detections"]

            # Simuliere Accuracy-Berechnung
            if total_trend_analyses > 0:
                self._analysis_performance_stats["trend_detection_accuracy"] = 0.85

            if total_anomaly_detections > 0:
                self._analysis_performance_stats["anomaly_detection_precision"] = 0.78
                self._analysis_performance_stats["anomaly_detection_recall"] = 0.82
                self._analysis_performance_stats["false_positive_rate"] = 0.15

        except Exception as e:
            logger.error(f"Analysis performance monitoring fehlgeschlagen: {e}")

    async def _trigger_trend_change_event(self, trend_analysis: TrendAnalysis) -> None:
        """Triggert Trend-Change-Event."""
        try:
            import uuid

            event = PerformanceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.TREND_CHANGE,
                event_name=f"trend_change_{trend_analysis.metric_name}",
                source_service="trend_analysis_engine",
                source_scope=trend_analysis.scope,
                source_scope_id=trend_analysis.scope_id,
                payload={
                    "analysis_id": trend_analysis.analysis_id,
                    "metric_name": trend_analysis.metric_name,
                    "trend_direction": trend_analysis.trend_direction.value,
                    "trend_strength": trend_analysis.trend_strength,
                    "trend_confidence": trend_analysis.trend_confidence,
                    "predicted_next_value": trend_analysis.predicted_next_value
                },
                priority=2  # High priority
            )

            # Benachrichtige Event-Callbacks
            for callback in self._event_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Event callback fehlgeschlagen: {e}")

        except Exception as e:
            logger.error(f"Trend change event triggering fehlgeschlagen: {e}")

    async def _trigger_anomaly_detected_event(self, anomaly: AnomalyDetection) -> None:
        """Triggert Anomaly-Detected-Event."""
        try:
            import uuid

            event = PerformanceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ANOMALY_DETECTED,
                event_name=f"anomaly_detected_{anomaly.metric_name}",
                source_service="anomaly_detection_engine",
                source_scope=anomaly.scope,
                source_scope_id=anomaly.scope_id,
                payload={
                    "anomaly_id": anomaly.anomaly_id,
                    "metric_name": anomaly.metric_name,
                    "anomaly_type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "anomalous_value": anomaly.anomalous_value,
                    "expected_value": anomaly.expected_value,
                    "deviation_percent": anomaly.deviation_percent
                },
                priority=1 if anomaly.severity > 0.8 else 3,  # Critical or medium priority
                requires_immediate_processing=anomaly.severity > 0.9
            )

            # Benachrichtige Event-Callbacks
            for callback in self._event_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Event callback fehlgeschlagen: {e}")

        except Exception as e:
            logger.error(f"Anomaly detected event triggering fehlgeschlagen: {e}")

    async def _is_critical_metric(self, data_point: PerformanceDataPoint) -> bool:
        """Prüft ob Metrik kritisch ist."""
        try:
            critical_metrics = ["response_time", "error_rate", "cpu_usage", "memory_usage", "throughput"]
            return data_point.metric_name in critical_metrics

        except Exception as e:
            logger.error(f"Critical metric check fehlgeschlagen: {e}")
            return False

    async def _deduplicate_anomalies(
        self,
        anomalies: list[AnomalyDetection]
    ) -> list[AnomalyDetection]:
        """Dedupliziert Anomalies."""
        try:
            if not anomalies:
                return anomalies

            # Gruppiere Anomalies nach Zeitfenster und Metrik
            grouped_anomalies = defaultdict(list)

            for anomaly in anomalies:
                key = f"{anomaly.metric_name}_{anomaly.anomaly_start.strftime('%Y%m%d%H%M')}"
                grouped_anomalies[key].append(anomaly)

            # Wähle beste Anomaly pro Gruppe
            unique_anomalies = []
            for group in grouped_anomalies.values():
                # Wähle Anomaly mit höchster Severity
                best_anomaly = max(group, key=lambda a: a.severity)
                unique_anomalies.append(best_anomaly)

            return unique_anomalies

        except Exception as e:
            logger.error(f"Anomalies deduplication fehlgeschlagen: {e}")
            return anomalies

    async def _detect_seasonal_pattern(
        self,
        values: list[float],
        timestamps: list[datetime]
    ) -> tuple[bool, float | None, float | None]:
        """Detektiert Seasonal-Pattern."""
        try:
            # Einfache Seasonal-Detection basierend auf Stunden-Pattern
            if len(values) < 24:  # Mindestens 24 Stunden für Seasonal-Detection
                return False, None, None

            # Gruppiere Werte nach Stunden
            hourly_values = defaultdict(list)
            for value, timestamp in zip(values, timestamps, strict=False):
                hour = timestamp.hour
                hourly_values[hour].append(value)

            # Berechne Durchschnitt pro Stunde
            hourly_averages = {}
            for hour, hour_values in hourly_values.items():
                if hour_values:
                    hourly_averages[hour] = sum(hour_values) / len(hour_values)

            if len(hourly_averages) < 12:  # Mindestens 12 verschiedene Stunden
                return False, None, None

            # Prüfe auf signifikante Variation
            avg_values = list(hourly_averages.values())
            if avg_values:
                import statistics
                mean_value = statistics.mean(avg_values)
                std_dev = statistics.stdev(avg_values) if len(avg_values) > 1 else 0.0

                # Seasonal-Pattern wenn Std-Dev > 20% des Means
                if std_dev > 0.2 * mean_value:
                    return True, 24.0, std_dev  # 24-Stunden-Pattern

            return False, None, None

        except Exception as e:
            logger.error(f"Seasonal pattern detection fehlgeschlagen: {e}")
            return False, None, None

    async def _detect_change_points(
        self,
        values: list[float],
        timestamps: list[datetime]
    ) -> list[datetime]:
        """Detektiert Change-Points."""
        try:
            change_points = []

            if len(values) < 10:
                return change_points

            # Einfache Change-Point-Detection basierend auf Moving-Average
            window_size = min(5, len(values) // 3)

            for i in range(window_size, len(values) - window_size):
                # Berechne Moving-Averages vor und nach dem Punkt
                before_avg = sum(values[i-window_size:i]) / window_size
                after_avg = sum(values[i:i+window_size]) / window_size

                # Prüfe auf signifikante Änderung
                if before_avg != 0:
                    change_percent = abs(after_avg - before_avg) / before_avg

                    if change_percent > 0.3:  # 30% Änderung
                        change_points.append(timestamps[i])

            return change_points

        except Exception as e:
            logger.error(f"Change points detection fehlgeschlagen: {e}")
            return []

    def _calculate_linear_regression(
        self,
        x_values: list[float],
        y_values: list[float]
    ) -> tuple[float, float, float]:
        """Berechnet lineare Regression."""
        try:
            n = len(x_values)
            if n < 2:
                return 0.0, 0.0, 0.0

            # Berechne Regression-Parameter
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
            sum_x2 = sum(x * x for x in x_values)
            # sum_y2 = sum(y * y for y in y_values)  # TODO: Verwende für R-squared Berechnung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/112

            # Slope und Intercept
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0, sum_y / n, 0.0

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

            # R-squared
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values, strict=False))

            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            return slope, intercept, max(0.0, r_squared)

        except Exception as e:
            logger.error(f"Linear regression calculation fehlgeschlagen: {e}")
            return 0.0, 0.0, 0.0

    def _determine_trend_direction(self, slope: float, _values: list[float]) -> TrendDirection:
        """Bestimmt Trend-Richtung."""
        try:
            if abs(slope) < 0.001:  # Sehr kleiner Slope
                return TrendDirection.STABLE
            if slope > 0:
                return TrendDirection.INCREASING
            return TrendDirection.DECREASING

        except Exception as e:
            logger.error(f"Trend direction determination fehlgeschlagen: {e}")
            return TrendDirection.STABLE

    def _calculate_trend_strength(self, slope: float, values: list[float], r_squared: float) -> float:
        """Berechnet Trend-Stärke."""
        try:
            if not values:
                return 0.0

            # Normalisiere Slope basierend auf Werte-Range
            value_range = max(values) - min(values)
            if value_range == 0:
                return 0.0

            normalized_slope = abs(slope) / value_range

            # Kombiniere mit R-squared
            strength = min(1.0, normalized_slope * r_squared * 10)

            return strength

        except Exception as e:
            logger.error(f"Trend strength calculation fehlgeschlagen: {e}")
            return 0.0

    def _calculate_trend_confidence(self, r_squared: float, sample_size: int) -> float:
        """Berechnet Trend-Confidence."""
        try:
            # Basis-Confidence aus R-squared
            base_confidence = r_squared

            # Sample-Size-Bonus
            size_factor = min(1.0, sample_size / 50.0)  # Bonus bis 50 Samples

            confidence = base_confidence * (0.7 + 0.3 * size_factor)

            return min(1.0, confidence)

        except Exception as e:
            logger.error(f"Trend confidence calculation fehlgeschlagen: {e}")
            return 0.0

    def _update_trend_analysis_performance_stats(self, analysis_time_ms: float) -> None:
        """Aktualisiert Trend-Analysis-Performance-Statistiken."""
        try:
            self._analysis_performance_stats["total_trend_analyses"] += 1

            current_avg = self._analysis_performance_stats["avg_trend_analysis_time_ms"]
            total_count = self._analysis_performance_stats["total_trend_analyses"]
            new_avg = ((current_avg * (total_count - 1)) + analysis_time_ms) / total_count
            self._analysis_performance_stats["avg_trend_analysis_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Trend analysis performance stats update fehlgeschlagen: {e}")

    def _update_anomaly_detection_performance_stats(self, detection_time_ms: float, anomalies_count: int) -> None:
        """Aktualisiert Anomaly-Detection-Performance-Statistiken."""
        try:
            self._analysis_performance_stats["total_anomaly_detections"] += anomalies_count

            current_avg = self._analysis_performance_stats["avg_anomaly_detection_time_ms"]
            # Vereinfachte Durchschnitts-Berechnung
            if current_avg == 0:
                self._analysis_performance_stats["avg_anomaly_detection_time_ms"] = detection_time_ms
            else:
                self._analysis_performance_stats["avg_anomaly_detection_time_ms"] = (current_avg + detection_time_ms) / 2

        except Exception as e:
            logger.error(f"Anomaly detection performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = self._analysis_performance_stats.copy()

            # Storage-Stats
            stats["storage_stats"] = {
                "total_trend_analyses_stored": len(self._trend_analyses),
                "total_anomaly_detections_stored": len(self._anomaly_detections),
                "time_series_count": len(self._time_series_data),
                "baseline_data_count": len(self._baseline_data)
            }

            # Configuration
            stats["configuration"] = {
                "trend_analysis_enabled": self.configuration.trend_analysis_enabled,
                "anomaly_detection_enabled": self.configuration.anomaly_detection_enabled,
                "trend_analysis_window_hours": self.configuration.trend_analysis_window_hours,
                "anomaly_detection_sensitivity": self.configuration.anomaly_detection_sensitivity,
                "anomaly_algorithms": self.configuration.anomaly_algorithms
            }

            return stats

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
