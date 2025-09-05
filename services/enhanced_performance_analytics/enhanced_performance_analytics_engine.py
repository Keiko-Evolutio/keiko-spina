# backend/services/enhanced_performance_analytics/enhanced_performance_analytics_engine.py
"""Enhanced Performance Analytics Engine.

Hauptengine für Enterprise-Grade Performance Analytics mit Event-driven Integration,
ML-basierte Performance-Vorhersagen und automatische Optimization-Empfehlungen.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_dependency_resolution import EnhancedDependencyResolutionEngine
from services.enhanced_quotas_limits_management import EnhancedQuotaManagementEngine
from services.enhanced_real_time_monitoring import EnhancedRealTimeMonitoringEngine
from services.enhanced_security_integration import (
    EnhancedSecurityIntegrationEngine,
    SecurityContext,
)
from services.ml.performance_prediction import PerformancePredictor

from .analytics_service_integration_layer import AnalyticsServiceIntegrationLayer
from .data_models import (
    AdvancedPerformanceMetrics,
    AnalyticsConfiguration,
    AnalyticsPerformanceMetrics,
    AnalyticsScope,
    AnomalyDetection,
    MLPerformancePrediction,
    PerformanceDataPoint,
    PerformanceOptimizationRecommendation,
    TrendAnalysis,
)
from .event_driven_analytics_engine import EventDrivenAnalyticsEngine
from .ml_performance_prediction_engine import MLPerformancePredictionEngine
from .performance_optimization_engine import PerformanceOptimizationEngine
from .trend_analysis_anomaly_detection_engine import TrendAnalysisAnomalyDetectionEngine

logger = get_logger(__name__)


class EnhancedPerformanceAnalyticsEngine:
    """Enhanced Performance Analytics Engine für Enterprise-Grade Performance Analytics."""

    def __init__(
        self,
        monitoring_engine: EnhancedRealTimeMonitoringEngine | None = None,
        dependency_resolution_engine: EnhancedDependencyResolutionEngine | None = None,
        quota_management_engine: EnhancedQuotaManagementEngine | None = None,
        security_integration_engine: EnhancedSecurityIntegrationEngine | None = None,
        performance_predictor: PerformancePredictor | None = None,
        configuration: AnalyticsConfiguration | None = None
    ):
        """Initialisiert Enhanced Performance Analytics Engine.

        Args:
            monitoring_engine: Real-time Monitoring Engine
            dependency_resolution_engine: Dependency Resolution Engine
            quota_management_engine: Quota Management Engine
            security_integration_engine: Security Integration Engine
            performance_predictor: Performance Predictor
            configuration: Analytics-Konfiguration
        """
        self.configuration = configuration or AnalyticsConfiguration()

        # Erstelle Analytics-Engines
        self.event_driven_analytics_engine = EventDrivenAnalyticsEngine(
            monitoring_engine=monitoring_engine,
            configuration=self.configuration
        )

        self.ml_prediction_engine = MLPerformancePredictionEngine(
            performance_predictor=performance_predictor,
            configuration=self.configuration
        )

        self.trend_anomaly_engine = TrendAnalysisAnomalyDetectionEngine(
            configuration=self.configuration
        )

        self.optimization_engine = PerformanceOptimizationEngine(
            configuration=self.configuration
        )

        # Erstelle Service Integration Layer
        self.service_integration_layer = AnalyticsServiceIntegrationLayer(
            event_driven_analytics_engine=self.event_driven_analytics_engine,
            ml_prediction_engine=self.ml_prediction_engine,
            trend_anomaly_engine=self.trend_anomaly_engine,
            optimization_engine=self.optimization_engine,
            monitoring_engine=monitoring_engine,
            dependency_resolution_engine=dependency_resolution_engine,
            quota_management_engine=quota_management_engine,
            security_integration_engine=security_integration_engine,
            performance_predictor=performance_predictor,
            configuration=self.configuration
        )

        # Performance-Tracking
        self._analytics_performance_metrics = AnalyticsPerformanceMetrics()
        self._is_running = False

        logger.info("Enhanced Performance Analytics Engine initialisiert")

    async def start(self) -> None:
        """Startet Enhanced Performance Analytics Engine."""
        if self._is_running:
            return

        try:
            self._is_running = True

            # Starte Service Integration Layer (startet alle Sub-Engines)
            await self.service_integration_layer.start()

            logger.info("Enhanced Performance Analytics Engine gestartet")

        except Exception as e:
            logger.error(f"Enhanced Performance Analytics Engine start fehlgeschlagen: {e}")
            self._is_running = False
            raise

    async def stop(self) -> None:
        """Stoppt Enhanced Performance Analytics Engine."""
        try:
            self._is_running = False

            # Stoppe Service Integration Layer (stoppt alle Sub-Engines)
            await self.service_integration_layer.stop()

            logger.info("Enhanced Performance Analytics Engine gestoppt")

        except Exception as e:
            logger.error(f"Enhanced Performance Analytics Engine stop fehlgeschlagen: {e}")

    async def analyze_service_performance(
        self,
        service_name: str,
        operation: str,
        request_data: dict[str, Any],
        result_data: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analysiert Service-Performance mit vollständiger Analytics-Pipeline.

        Args:
            service_name: Service-Name
            operation: Operation-Name
            request_data: Request-Daten
            result_data: Result-Daten
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
            metadata: Metadata

        Returns:
            Vollständige Analytics-Ergebnisse
        """
        start_time = time.time()

        try:
            # Delegiere an Service Integration Layer
            analytics_results = await self.service_integration_layer.analyze_service_performance(
                service_name=service_name,
                operation=operation,
                request_data=request_data,
                result_data=result_data,
                response_time_ms=response_time_ms,
                success=success,
                security_context=security_context,
                metadata=metadata
            )

            # Update Performance-Metriken
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_analytics_performance_metrics(processing_time_ms, success)

            # Füge Engine-Level-Informationen hinzu
            analytics_results["analytics_engine"] = {
                "processing_time_ms": processing_time_ms,
                "analytics_enabled": self.configuration.analytics_enabled,
                "meets_analytics_sla": processing_time_ms < self.configuration.analytics_processing_timeout_ms
            }

            return analytics_results

        except Exception as e:
            logger.error(f"Service performance analysis fehlgeschlagen: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_analytics_performance_metrics(processing_time_ms, False)

            return {
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "analytics_engine": {
                    "processing_time_ms": processing_time_ms,
                    "meets_analytics_sla": False
                }
            }

    async def analyze_llm_performance(
        self,
        operation: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analysiert LLM-Performance mit speziellen LLM-Metriken.

        Args:
            operation: LLM-Operation
            model_name: Model-Name
            prompt_tokens: Prompt-Tokens
            completion_tokens: Completion-Tokens
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
            metadata: Metadata

        Returns:
            LLM-Analytics-Ergebnisse
        """
        try:
            return await self.service_integration_layer.analyze_llm_performance(
                operation=operation,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time_ms=response_time_ms,
                success=success,
                security_context=security_context,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"LLM performance analysis fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def predict_performance(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        prediction_horizon_minutes: int,
        context_data: dict[str, Any] | None = None,
        security_context: SecurityContext | None = None
    ) -> MLPerformancePrediction:
        """Erstellt ML-basierte Performance-Vorhersage.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_name: Metrik-Name
            prediction_horizon_minutes: Vorhersage-Horizont
            context_data: Kontext-Daten
            security_context: Security-Context

        Returns:
            ML Performance-Vorhersage
        """
        try:
            return await self.ml_prediction_engine.predict_performance(
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                prediction_horizon_minutes=prediction_horizon_minutes,
                context_data=context_data,
                security_context=security_context
            )

        except Exception as e:
            logger.error(f"Performance prediction fehlgeschlagen: {e}")
            raise

    async def analyze_trends(
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
            analysis_window_hours: Analysis-Fenster

        Returns:
            Trend-Analysis
        """
        try:
            return await self.trend_anomaly_engine.analyze_trend(
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                analysis_window_hours=analysis_window_hours
            )

        except Exception as e:
            logger.error(f"Trend analysis fehlgeschlagen: {e}")
            raise

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
        try:
            return await self.trend_anomaly_engine.detect_anomalies(
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                detection_algorithms=detection_algorithms
            )

        except Exception as e:
            logger.error(f"Anomaly detection fehlgeschlagen: {e}")
            return []

    async def get_optimization_recommendations(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        status_filter: str | None = None,
        priority_threshold: float = 0.0
    ) -> list[PerformanceOptimizationRecommendation]:
        """Holt Performance-Optimization-Empfehlungen.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            status_filter: Status-Filter
            priority_threshold: Priority-Threshold

        Returns:
            Liste von Optimization-Empfehlungen
        """
        try:
            return await self.optimization_engine.get_optimization_recommendations(
                scope=scope,
                scope_id=scope_id,
                status_filter=status_filter,
                priority_threshold=priority_threshold
            )

        except Exception as e:
            logger.error(f"Optimization recommendations retrieval fehlgeschlagen: {e}")
            return []

    async def implement_optimization_recommendation(
        self,
        recommendation_id: str,
        implementation_notes: str | None = None,
        security_context: SecurityContext | None = None
    ) -> dict[str, Any]:
        """Implementiert Optimization-Empfehlung.

        Args:
            recommendation_id: Recommendation-ID
            implementation_notes: Implementation-Notes
            security_context: Security-Context

        Returns:
            Implementation-Result
        """
        try:
            return await self.optimization_engine.implement_optimization_recommendation(
                recommendation_id=recommendation_id,
                implementation_notes=implementation_notes,
                security_context=security_context
            )

        except Exception as e:
            logger.error(f"Optimization recommendation implementation fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    async def generate_advanced_metrics(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_names: list[str],
        period_start: datetime,
        period_end: datetime
    ) -> AdvancedPerformanceMetrics:
        """Generiert Advanced Performance-Metriken.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_names: Metrik-Namen
            period_start: Periode-Start
            period_end: Periode-Ende

        Returns:
            Advanced Performance-Metriken
        """
        try:
            return await self.event_driven_analytics_engine.generate_advanced_metrics(
                scope=scope,
                scope_id=scope_id,
                metric_names=metric_names,
                period_start=period_start,
                period_end=period_end
            )

        except Exception as e:
            logger.error(f"Advanced metrics generation fehlgeschlagen: {e}")
            raise

    async def collect_performance_data_point(
        self,
        data_point: PerformanceDataPoint
    ) -> None:
        """Sammelt Performance-Datenpunkt.

        Args:
            data_point: Performance-Datenpunkt
        """
        try:
            await self.event_driven_analytics_engine.collect_performance_data_point(data_point)
            await self.trend_anomaly_engine.add_data_point(data_point)

        except Exception as e:
            logger.error(f"Performance data point collection fehlgeschlagen: {e}")

    async def validate_prediction(
        self,
        prediction_id: str,
        actual_value: float
    ) -> dict[str, float]:
        """Validiert Prediction mit tatsächlichem Wert.

        Args:
            prediction_id: Prediction-ID
            actual_value: Tatsächlicher Wert

        Returns:
            Validation-Metriken
        """
        try:
            return await self.ml_prediction_engine.validate_prediction(
                prediction_id=prediction_id,
                actual_value=actual_value
            )

        except Exception as e:
            logger.error(f"Prediction validation fehlgeschlagen: {e}")
            return {}

    async def get_prediction_trends(
        self,
        scope: AnalyticsScope,
        scope_id: str,
        metric_name: str,
        time_range_hours: int = 24
    ) -> dict[str, Any]:
        """Holt Prediction-Trends.

        Args:
            scope: Analytics-Scope
            scope_id: Scope-ID
            metric_name: Metrik-Name
            time_range_hours: Zeitraum

        Returns:
            Prediction-Trends
        """
        try:
            return await self.ml_prediction_engine.get_prediction_trends(
                scope=scope,
                scope_id=scope_id,
                metric_name=metric_name,
                time_range_hours=time_range_hours
            )

        except Exception as e:
            logger.error(f"Prediction trends retrieval fehlgeschlagen: {e}")
            return {"trends": [], "summary": {}}

    def _update_analytics_performance_metrics(
        self,
        processing_time_ms: float,
        success: bool
    ) -> None:
        """Aktualisiert Analytics-Performance-Metriken."""
        try:
            self._analytics_performance_metrics.total_data_points_processed += 1

            # Update Average Processing Time
            current_avg = self._analytics_performance_metrics.avg_data_processing_time_ms
            total_count = self._analytics_performance_metrics.total_data_points_processed
            new_avg = ((current_avg * (total_count - 1)) + processing_time_ms) / total_count
            self._analytics_performance_metrics.avg_data_processing_time_ms = new_avg

            # Update Error Rate
            if not success:
                error_count = self._analytics_performance_metrics.data_processing_error_rate * (total_count - 1) + 1
                self._analytics_performance_metrics.data_processing_error_rate = error_count / total_count
            else:
                error_count = self._analytics_performance_metrics.data_processing_error_rate * (total_count - 1)
                self._analytics_performance_metrics.data_processing_error_rate = error_count / total_count

            # Update SLA-Compliance
            self._analytics_performance_metrics.meets_analytics_sla = (
                self._analytics_performance_metrics.avg_data_processing_time_ms <
                self.configuration.analytics_processing_timeout_ms
            )

        except Exception as e:
            logger.error(f"Analytics performance metrics update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt umfassende Performance-Statistiken zurück."""
        try:
            # Hole Statistiken von Service Integration Layer
            integration_stats = self.service_integration_layer.get_performance_stats()

            # Füge Engine-Level-Statistiken hinzu
            engine_stats = {
                "enhanced_performance_analytics_engine": {
                    "is_running": self._is_running,
                    "analytics_performance_metrics": {
                        "total_data_points_processed": self._analytics_performance_metrics.total_data_points_processed,
                        "avg_data_processing_time_ms": self._analytics_performance_metrics.avg_data_processing_time_ms,
                        "data_processing_error_rate": self._analytics_performance_metrics.data_processing_error_rate,
                        "meets_analytics_sla": self._analytics_performance_metrics.meets_analytics_sla,
                        "analytics_sla_threshold_ms": self.configuration.analytics_processing_timeout_ms
                    },
                    "configuration": {
                        "analytics_enabled": self.configuration.analytics_enabled,
                        "real_time_analytics_enabled": self.configuration.real_time_analytics_enabled,
                        "ml_predictions_enabled": self.configuration.ml_predictions_enabled,
                        "trend_analysis_enabled": self.configuration.trend_analysis_enabled,
                        "anomaly_detection_enabled": self.configuration.anomaly_detection_enabled,
                        "optimization_recommendations_enabled": self.configuration.optimization_recommendations_enabled,
                        "analytics_processing_timeout_ms": self.configuration.analytics_processing_timeout_ms
                    }
                }
            }

            # Kombiniere alle Statistiken
            combined_stats = {**engine_stats, **integration_stats}

            return combined_stats

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}

    def get_analytics_summary(self) -> dict[str, Any]:
        """Gibt Analytics-Summary zurück."""
        try:
            stats = self.get_performance_stats()

            # Extrahiere Key-Metriken
            engine_stats = stats.get("enhanced_performance_analytics_engine", {})
            analytics_metrics = engine_stats.get("analytics_performance_metrics", {})

            event_driven_stats = stats.get("event_driven_analytics", {})
            ml_prediction_stats = stats.get("ml_performance_prediction", {})
            trend_anomaly_stats = stats.get("trend_analysis_anomaly_detection", {})
            optimization_stats = stats.get("performance_optimization", {})

            summary = {
                "analytics_engine": {
                    "status": "running" if self._is_running else "stopped",
                    "total_data_points_processed": analytics_metrics.get("total_data_points_processed", 0),
                    "avg_processing_time_ms": analytics_metrics.get("avg_data_processing_time_ms", 0.0),
                    "meets_sla": analytics_metrics.get("meets_analytics_sla", False),
                    "error_rate": analytics_metrics.get("data_processing_error_rate", 0.0)
                },
                "event_driven_analytics": {
                    "total_events_processed": event_driven_stats.get("event_processing", {}).get("total_events_processed", 0),
                    "avg_event_processing_time_ms": event_driven_stats.get("event_processing", {}).get("avg_event_processing_time_ms", 0.0)
                },
                "ml_predictions": {
                    "total_predictions_made": ml_prediction_stats.get("total_predictions_made", 0),
                    "avg_prediction_time_ms": ml_prediction_stats.get("avg_prediction_time_ms", 0.0),
                    "prediction_accuracy": ml_prediction_stats.get("prediction_accuracy", 0.0)
                },
                "trend_anomaly_detection": {
                    "total_trend_analyses": trend_anomaly_stats.get("total_trend_analyses", 0),
                    "total_anomaly_detections": trend_anomaly_stats.get("total_anomaly_detections", 0),
                    "trend_detection_accuracy": trend_anomaly_stats.get("trend_detection_accuracy", 0.0),
                    "anomaly_detection_precision": trend_anomaly_stats.get("anomaly_detection_precision", 0.0)
                },
                "optimization": {
                    "total_recommendations_generated": optimization_stats.get("total_recommendations_generated", 0),
                    "total_optimizations_implemented": optimization_stats.get("total_optimizations_implemented", 0),
                    "recommendation_effectiveness": optimization_stats.get("recommendation_effectiveness", 0.0)
                }
            }

            return summary

        except Exception as e:
            logger.error(f"Analytics summary generation fehlgeschlagen: {e}")
            return {}

    @property
    def is_running(self) -> bool:
        """Gibt Running-Status zurück."""
        return self._is_running

    @property
    def analytics_configuration(self) -> AnalyticsConfiguration:
        """Gibt Analytics-Konfiguration zurück."""
        return self.configuration
