# backend/services/enhanced_performance_analytics/analytics_service_integration_layer.py
"""Analytics Service Integration Layer.

Integriert Enhanced Performance Analytics mit allen bestehenden Services
und erweitert sie um Enterprise-Grade Performance Analytics Features.
"""

from __future__ import annotations

import time
from datetime import timedelta
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

from .data_models import (
    AnalyticsConfiguration,
    AnalyticsScope,
    MetricDimension,
    PerformanceDataPoint,
)
from .event_driven_analytics_engine import EventDrivenAnalyticsEngine
from .ml_performance_prediction_engine import MLPerformancePredictionEngine
from .performance_optimization_engine import PerformanceOptimizationEngine
from .trend_analysis_anomaly_detection_engine import TrendAnalysisAnomalyDetectionEngine

logger = get_logger(__name__)


class AnalyticsServiceIntegrationLayer:
    """Service Integration Layer für Enhanced Performance Analytics."""

    def __init__(
        self,
        event_driven_analytics_engine: EventDrivenAnalyticsEngine,
        ml_prediction_engine: MLPerformancePredictionEngine,
        trend_anomaly_engine: TrendAnalysisAnomalyDetectionEngine,
        optimization_engine: PerformanceOptimizationEngine,
        monitoring_engine: EnhancedRealTimeMonitoringEngine | None = None,
        dependency_resolution_engine: EnhancedDependencyResolutionEngine | None = None,
        quota_management_engine: EnhancedQuotaManagementEngine | None = None,
        security_integration_engine: EnhancedSecurityIntegrationEngine | None = None,
        performance_predictor: PerformancePredictor | None = None,
        configuration: AnalyticsConfiguration | None = None
    ):
        """Initialisiert Analytics Service Integration Layer.

        Args:
            event_driven_analytics_engine: Event-driven Analytics Engine
            ml_prediction_engine: ML Performance Prediction Engine
            trend_anomaly_engine: Trend Analysis und Anomaly Detection Engine
            optimization_engine: Performance Optimization Engine
            monitoring_engine: Real-time Monitoring Engine
            dependency_resolution_engine: Dependency Resolution Engine
            quota_management_engine: Quota Management Engine
            security_integration_engine: Security Integration Engine
            performance_predictor: Performance Predictor
            configuration: Analytics-Konfiguration
        """
        self.event_driven_analytics_engine = event_driven_analytics_engine
        self.ml_prediction_engine = ml_prediction_engine
        self.trend_anomaly_engine = trend_anomaly_engine
        self.optimization_engine = optimization_engine
        self.monitoring_engine = monitoring_engine
        self.dependency_resolution_engine = dependency_resolution_engine
        self.quota_management_engine = quota_management_engine
        self.security_integration_engine = security_integration_engine
        self.performance_predictor = performance_predictor
        self.configuration = configuration or AnalyticsConfiguration()

        # Integration-Status
        self._service_integrations = {
            "real_time_monitoring": monitoring_engine is not None,
            "dependency_resolution": dependency_resolution_engine is not None,
            "quota_management": quota_management_engine is not None,
            "security_integration": security_integration_engine is not None,
            "performance_prediction": performance_predictor is not None
        }

        # Performance-Tracking
        self._integration_performance_stats = {
            "total_analytics_operations": 0,
            "avg_analytics_processing_time_ms": 0.0,
            "service_integration_count": 0,
            "analytics_throughput_ops": 0.0,
            "analytics_error_rate": 0.0
        }

        # Service-Monitoring-Status
        self._service_monitoring_status: dict[str, dict[str, Any]] = {
            "llm_client": {"enabled": False, "metrics": {}},
            "performance_prediction": {"enabled": performance_predictor is not None, "metrics": {}},
            "task_decomposition": {"enabled": False, "metrics": {}},
            "orchestrator": {"enabled": False, "metrics": {}},
            "agent_selection": {"enabled": False, "metrics": {}},
            "security_integration": {"enabled": security_integration_engine is not None, "metrics": {}},
            "quota_management": {"enabled": quota_management_engine is not None, "metrics": {}},
            "dependency_resolution": {"enabled": dependency_resolution_engine is not None, "metrics": {}},
            "real_time_monitoring": {"enabled": monitoring_engine is not None, "metrics": {}}
        }

        logger.info("Analytics Service Integration Layer initialisiert")

    async def start(self) -> None:
        """Startet Analytics Service Integration Layer."""
        try:
            # Starte alle Analytics-Engines
            await self.event_driven_analytics_engine.start()
            await self.ml_prediction_engine.start()
            await self.trend_anomaly_engine.start()
            await self.optimization_engine.start()

            # Registriere Event-Callbacks
            await self._register_event_callbacks()

            # Initialisiere Service-Integrations
            await self._initialize_service_integrations()

            logger.info("Analytics Service Integration Layer gestartet")

        except Exception as e:
            logger.error(f"Analytics Service Integration Layer start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Analytics Service Integration Layer."""
        try:
            await self.event_driven_analytics_engine.stop()
            await self.ml_prediction_engine.stop()
            await self.trend_anomaly_engine.stop()
            await self.optimization_engine.stop()

            logger.info("Analytics Service Integration Layer gestoppt")

        except Exception as e:
            logger.error(f"Analytics Service Integration Layer stop fehlgeschlagen: {e}")

    async def analyze_service_performance(
        self,
        service_name: str,
        operation: str,
        _request_data: dict[str, Any],
        _result_data: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analysiert Service-Performance mit vollständiger Analytics-Pipeline.

        Args:
            service_name: Service-Name
            operation: Operation-Name
            _request_data: Request-Daten
            _result_data: Result-Daten
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
            metadata: Metadata

        Returns:
            Analytics-Ergebnisse
        """
        start_time = time.time()

        try:
            # Erstelle Performance-Datenpunkt
            data_point = PerformanceDataPoint(
                data_point_id=f"{service_name}_{operation}_{int(time.time() * 1000)}",
                metric_name="response_time",
                scope=AnalyticsScope.SERVICE,
                scope_id=service_name,
                value=response_time_ms,
                unit="ms",
                dimensions={
                    MetricDimension.SERVICE: service_name,
                    MetricDimension.OPERATION: operation,
                    MetricDimension.USER: security_context.user_id,
                    MetricDimension.TENANT: security_context.tenant_id
                },
                service_name=service_name,
                user_id=security_context.user_id,
                tenant_id=security_context.tenant_id,
                labels={
                    "operation": operation,
                    "success": str(success)
                },
                metadata=metadata or {}
            )

            # Event-driven Analytics
            await self.event_driven_analytics_engine.collect_performance_data_point(data_point)

            # Trend-Analysis und Anomaly-Detection
            await self.trend_anomaly_engine.add_data_point(data_point)

            # ML-basierte Performance-Vorhersage
            prediction = await self.ml_prediction_engine.predict_performance(
                scope=AnalyticsScope.SERVICE,
                scope_id=service_name,
                metric_name="response_time",
                prediction_horizon_minutes=60,
                context_data={
                    "operation": operation,
                    "current_response_time": response_time_ms,
                    "success": success
                },
                security_context=security_context
            )

            # Advanced Performance-Metriken generieren
            advanced_metrics = await self.event_driven_analytics_engine.generate_advanced_metrics(
                scope=AnalyticsScope.SERVICE,
                scope_id=service_name,
                metric_names=["response_time", "error_rate", "throughput"],
                period_start=data_point.timestamp - timedelta(hours=1),
                period_end=data_point.timestamp
            )

            # Trend-Analysis
            trend_analysis = await self.trend_anomaly_engine.analyze_trend(
                scope=AnalyticsScope.SERVICE,
                scope_id=service_name,
                metric_name="response_time"
            )

            # Anomaly-Detection
            anomalies = await self.trend_anomaly_engine.detect_anomalies(
                scope=AnalyticsScope.SERVICE,
                scope_id=service_name,
                metric_name="response_time"
            )

            # Performance-Optimization-Analysis
            optimization_recommendations = await self.optimization_engine.analyze_performance_for_optimization(
                scope=AnalyticsScope.SERVICE,
                scope_id=service_name,
                performance_metrics=advanced_metrics,
                anomalies=anomalies,
                trends=[trend_analysis],
                security_context=security_context
            )

            # Update Service-Monitoring-Status
            await self._update_service_monitoring_status(
                service_name, operation, response_time_ms, success
            )

            # Update Performance-Stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_integration_performance_stats(processing_time_ms)

            analytics_results = {
                "data_point": {
                    "metric_name": data_point.metric_name,
                    "value": data_point.value,
                    "timestamp": data_point.timestamp.isoformat()
                },
                "prediction": {
                    "predicted_value": prediction.predicted_value,
                    "confidence": prediction.prediction_confidence,
                    "horizon_minutes": prediction.prediction_horizon_minutes
                },
                "advanced_metrics": {
                    "avg_response_time_ms": advanced_metrics.avg_response_time_ms,
                    "p95_response_time_ms": advanced_metrics.p95_response_time_ms,
                    "error_rate": advanced_metrics.error_rate,
                    "success_rate": advanced_metrics.success_rate
                },
                "trend_analysis": {
                    "trend_direction": trend_analysis.trend_direction.value,
                    "trend_strength": trend_analysis.trend_strength,
                    "trend_confidence": trend_analysis.trend_confidence
                },
                "anomalies": [
                    {
                        "anomaly_type": anomaly.anomaly_type.value,
                        "severity": anomaly.severity,
                        "confidence": anomaly.confidence
                    }
                    for anomaly in anomalies
                ],
                "optimization_recommendations": [
                    {
                        "optimization_type": rec.optimization_type.value,
                        "priority": rec.priority,
                        "estimated_improvement": rec.estimated_improvement_percent
                    }
                    for rec in optimization_recommendations
                ],
                "processing_time_ms": processing_time_ms
            }

            logger.debug({
                "event": "service_performance_analyzed",
                "service_name": service_name,
                "operation": operation,
                "response_time_ms": response_time_ms,
                "success": success,
                "anomalies_detected": len(anomalies),
                "recommendations_generated": len(optimization_recommendations),
                "processing_time_ms": processing_time_ms
            })

            return analytics_results

        except Exception as e:
            logger.error(f"Service performance analysis fehlgeschlagen: {e}")
            return {"error": str(e), "processing_time_ms": (time.time() - start_time) * 1000}

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
            # LLM-spezifische Metriken
            total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = total_tokens / (response_time_ms / 1000.0) if response_time_ms > 0 else 0

            # Erstelle LLM-Performance-Datenpunkte
            data_points = [
                PerformanceDataPoint(
                    data_point_id=f"llm_response_time_{int(time.time() * 1000)}",
                    metric_name="llm_response_time",
                    scope=AnalyticsScope.LLM,
                    scope_id=model_name,
                    value=response_time_ms,
                    unit="ms",
                    dimensions={
                        MetricDimension.SERVICE: "llm_client",
                        MetricDimension.OPERATION: operation,
                        MetricDimension.USER: security_context.user_id
                    },
                    service_name="llm_client",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "model_name": model_name,
                        "operation": operation,
                        "success": str(success)
                    },
                    metadata={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": tokens_per_second,
                        **(metadata or {})
                    }
                ),
                PerformanceDataPoint(
                    data_point_id=f"llm_tokens_per_second_{int(time.time() * 1000)}",
                    metric_name="llm_tokens_per_second",
                    scope=AnalyticsScope.LLM,
                    scope_id=model_name,
                    value=tokens_per_second,
                    unit="tokens/s",
                    dimensions={
                        MetricDimension.SERVICE: "llm_client",
                        MetricDimension.OPERATION: operation
                    },
                    service_name="llm_client",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "model_name": model_name,
                        "operation": operation
                    }
                )
            ]

            # Analysiere alle LLM-Datenpunkte
            analytics_results = []
            for data_point in data_points:
                await self.event_driven_analytics_engine.collect_performance_data_point(data_point)
                await self.trend_anomaly_engine.add_data_point(data_point)

                # LLM-spezifische Vorhersagen
                prediction = await self.ml_prediction_engine.predict_performance(
                    scope=AnalyticsScope.LLM,
                    scope_id=model_name,
                    metric_name=data_point.metric_name,
                    prediction_horizon_minutes=30,
                    context_data={
                        "model_name": model_name,
                        "operation": operation,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens
                    },
                    security_context=security_context
                )

                analytics_results.append({
                    "metric_name": data_point.metric_name,
                    "value": data_point.value,
                    "prediction": {
                        "predicted_value": prediction.predicted_value,
                        "confidence": prediction.prediction_confidence
                    }
                })

            # Update Service-Status
            await self._update_service_monitoring_status(
                "llm_client", operation, response_time_ms, success
            )

            return {
                "llm_metrics": analytics_results,
                "tokens_per_second": tokens_per_second,
                "total_tokens": total_tokens,
                "model_name": model_name
            }

        except Exception as e:
            logger.error(f"LLM performance analysis fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def analyze_dependency_resolution_performance(
        self,
        operation: str,
        graph_id: str,
        resolved_nodes: list[str],
        resolution_time_ms: float,
        success: bool,
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert Dependency Resolution Performance.

        Args:
            operation: Operation-Name
            graph_id: Graph-ID
            resolved_nodes: Aufgelöste Nodes
            resolution_time_ms: Resolution-Zeit
            success: Erfolg-Status
            security_context: Security-Context

        Returns:
            Dependency-Analytics-Ergebnisse
        """
        try:
            # Dependency-spezifische Metriken
            nodes_count = len(resolved_nodes)
            nodes_per_second = nodes_count / (resolution_time_ms / 1000.0) if resolution_time_ms > 0 else 0

            # Erstelle Dependency-Performance-Datenpunkte
            data_points = [
                PerformanceDataPoint(
                    data_point_id=f"dependency_resolution_time_{int(time.time() * 1000)}",
                    metric_name="dependency_resolution_time",
                    scope=AnalyticsScope.DEPENDENCY,
                    scope_id=graph_id,
                    value=resolution_time_ms,
                    unit="ms",
                    dimensions={
                        MetricDimension.SERVICE: "dependency_resolution",
                        MetricDimension.OPERATION: operation
                    },
                    service_name="dependency_resolution",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "graph_id": graph_id,
                        "operation": operation,
                        "success": str(success)
                    },
                    metadata={
                        "nodes_count": nodes_count,
                        "nodes_per_second": nodes_per_second,
                        "resolved_nodes": resolved_nodes
                    }
                ),
                PerformanceDataPoint(
                    data_point_id=f"dependency_nodes_per_second_{int(time.time() * 1000)}",
                    metric_name="dependency_nodes_per_second",
                    scope=AnalyticsScope.DEPENDENCY,
                    scope_id=graph_id,
                    value=nodes_per_second,
                    unit="nodes/s",
                    dimensions={
                        MetricDimension.SERVICE: "dependency_resolution",
                        MetricDimension.OPERATION: operation
                    },
                    service_name="dependency_resolution",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "graph_id": graph_id,
                        "operation": operation
                    }
                )
            ]

            # Analysiere Dependency-Datenpunkte
            for data_point in data_points:
                await self.event_driven_analytics_engine.collect_performance_data_point(data_point)
                await self.trend_anomaly_engine.add_data_point(data_point)

            # Update Service-Status
            await self._update_service_monitoring_status(
                "dependency_resolution", operation, resolution_time_ms, success
            )

            return {
                "resolution_time_ms": resolution_time_ms,
                "nodes_count": nodes_count,
                "nodes_per_second": nodes_per_second,
                "graph_id": graph_id
            }

        except Exception as e:
            logger.error(f"Dependency resolution performance analysis fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def analyze_quota_management_performance(
        self,
        operation: str,
        resource_type: str,
        quota_check_result: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert Quota Management Performance.

        Args:
            operation: Operation-Name
            resource_type: Resource-Type
            quota_check_result: Quota-Check-Result
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context

        Returns:
            Quota-Analytics-Ergebnisse
        """
        try:
            # Quota-spezifische Metriken
            usage = quota_check_result.get("usage", 0)
            limit = quota_check_result.get("limit", 1)
            usage_percentage = (usage / limit) * 100 if limit > 0 else 0

            # Erstelle Quota-Performance-Datenpunkte
            data_points = [
                PerformanceDataPoint(
                    data_point_id=f"quota_check_time_{int(time.time() * 1000)}",
                    metric_name="quota_check_time",
                    scope=AnalyticsScope.RESOURCE,
                    scope_id=resource_type,
                    value=response_time_ms,
                    unit="ms",
                    dimensions={
                        MetricDimension.SERVICE: "quota_management",
                        MetricDimension.OPERATION: operation,
                        MetricDimension.RESOURCE_TYPE: resource_type,
                        MetricDimension.TENANT: security_context.tenant_id
                    },
                    service_name="quota_management",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "resource_type": resource_type,
                        "operation": operation,
                        "success": str(success)
                    },
                    metadata=quota_check_result
                ),
                PerformanceDataPoint(
                    data_point_id=f"quota_usage_percentage_{int(time.time() * 1000)}",
                    metric_name="quota_usage_percentage",
                    scope=AnalyticsScope.RESOURCE,
                    scope_id=resource_type,
                    value=usage_percentage,
                    unit="percent",
                    dimensions={
                        MetricDimension.RESOURCE_TYPE: resource_type,
                        MetricDimension.TENANT: security_context.tenant_id
                    },
                    service_name="quota_management",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "resource_type": resource_type
                    }
                )
            ]

            # Analysiere Quota-Datenpunkte
            for data_point in data_points:
                await self.event_driven_analytics_engine.collect_performance_data_point(data_point)
                await self.trend_anomaly_engine.add_data_point(data_point)

            # Update Service-Status
            await self._update_service_monitoring_status(
                "quota_management", operation, response_time_ms, success
            )

            return {
                "check_time_ms": response_time_ms,
                "usage_percentage": usage_percentage,
                "resource_type": resource_type,
                "quota_result": quota_check_result
            }

        except Exception as e:
            logger.error(f"Quota management performance analysis fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def analyze_security_integration_performance(
        self,
        operation: str,
        security_check_result: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Analysiert Security Integration Performance.

        Args:
            operation: Operation-Name
            security_check_result: Security-Check-Result
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context

        Returns:
            Security-Analytics-Ergebnisse
        """
        try:
            # Security-spezifische Metriken
            security_score = security_check_result.get("security_score", 0.0)

            # Erstelle Security-Performance-Datenpunkte
            data_points = [
                PerformanceDataPoint(
                    data_point_id=f"security_check_time_{int(time.time() * 1000)}",
                    metric_name="security_check_time",
                    scope=AnalyticsScope.SECURITY,
                    scope_id="security_integration",
                    value=response_time_ms,
                    unit="ms",
                    dimensions={
                        MetricDimension.SERVICE: "security_integration",
                        MetricDimension.OPERATION: operation,
                        MetricDimension.SECURITY_LEVEL: security_context.security_level.value
                    },
                    service_name="security_integration",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "operation": operation,
                        "success": str(success),
                        "security_level": security_context.security_level.value
                    },
                    metadata=security_check_result
                ),
                PerformanceDataPoint(
                    data_point_id=f"security_score_{int(time.time() * 1000)}",
                    metric_name="security_score",
                    scope=AnalyticsScope.SECURITY,
                    scope_id="security_integration",
                    value=security_score,
                    unit="score",
                    dimensions={
                        MetricDimension.OPERATION: operation,
                        MetricDimension.USER: security_context.user_id
                    },
                    service_name="security_integration",
                    user_id=security_context.user_id,
                    tenant_id=security_context.tenant_id,
                    labels={
                        "operation": operation
                    }
                )
            ]

            # Analysiere Security-Datenpunkte
            for data_point in data_points:
                await self.event_driven_analytics_engine.collect_performance_data_point(data_point)
                await self.trend_anomaly_engine.add_data_point(data_point)

            # Update Service-Status
            await self._update_service_monitoring_status(
                "security_integration", operation, response_time_ms, success
            )

            return {
                "check_time_ms": response_time_ms,
                "security_score": security_score,
                "security_result": security_check_result
            }

        except Exception as e:
            logger.error(f"Security integration performance analysis fehlgeschlagen: {e}")
            return {"error": str(e)}

    async def _register_event_callbacks(self) -> None:
        """Registriert Event-Callbacks zwischen Engines."""
        try:
            # Registriere Trend-Analysis Event-Callback
            await self.trend_anomaly_engine.register_event_callback(
                self._handle_trend_anomaly_event
            )

            # Registriere Optimization Event-Callback
            await self.optimization_engine.register_event_callback(
                self._handle_optimization_event
            )

        except Exception as e:
            logger.error(f"Event callbacks registration fehlgeschlagen: {e}")

    async def _initialize_service_integrations(self) -> None:
        """Initialisiert Service-Integrations."""
        try:
            # Initialisiere Integration mit Real-time Monitoring
            if self.monitoring_engine:
                await self._setup_monitoring_integration()

            # Initialisiere Integration mit Performance Predictor
            if self.performance_predictor:
                await self._setup_performance_predictor_integration()

            logger.info(f"Service integrations initialisiert: {self._service_integrations}")

        except Exception as e:
            logger.error(f"Service integrations initialization fehlgeschlagen: {e}")

    async def _setup_monitoring_integration(self) -> None:
        """Setup Integration mit Real-time Monitoring."""
        try:
            # Hole Performance-Daten vom Monitoring-Engine
            monitoring_stats = self.monitoring_engine.get_performance_stats()

            # Konvertiere zu Analytics-Format
            if monitoring_stats:
                logger.debug(f"Monitoring integration setup: {len(monitoring_stats)} stats")

        except Exception as e:
            logger.error(f"Monitoring integration setup fehlgeschlagen: {e}")

    async def _setup_performance_predictor_integration(self) -> None:
        """Setup Integration mit Performance Predictor."""
        try:
            # Initialisiere Performance Predictor für Analytics
            logger.debug("Performance predictor integration setup")

        except Exception as e:
            logger.error(f"Performance predictor integration setup fehlgeschlagen: {e}")

    async def _handle_trend_anomaly_event(self, event) -> None:
        """Behandelt Trend/Anomaly-Events."""
        try:
            logger.debug({
                "event": "trend_anomaly_event_received",
                "event_type": event.event_type.value,
                "source_scope": event.source_scope.value,
                "source_scope_id": event.source_scope_id
            })

        except Exception as e:
            logger.error(f"Trend anomaly event handling fehlgeschlagen: {e}")

    async def _handle_optimization_event(self, event) -> None:
        """Behandelt Optimization-Events."""
        try:
            logger.debug({
                "event": "optimization_event_received",
                "event_type": event.event_type.value,
                "source_scope": event.source_scope.value,
                "source_scope_id": event.source_scope_id
            })

        except Exception as e:
            logger.error(f"Optimization event handling fehlgeschlagen: {e}")

    async def _update_service_monitoring_status(
        self,
        service_name: str,
        operation: str,
        response_time_ms: float,
        success: bool
    ) -> None:
        """Aktualisiert Service-Monitoring-Status."""
        try:
            if service_name in self._service_monitoring_status:
                status = self._service_monitoring_status[service_name]
                metrics = status["metrics"]

                # Update Metriken
                if "total_operations" not in metrics:
                    metrics["total_operations"] = 0
                    metrics["successful_operations"] = 0
                    metrics["failed_operations"] = 0
                    metrics["avg_response_time_ms"] = 0.0

                metrics["total_operations"] += 1

                if success:
                    metrics["successful_operations"] += 1
                else:
                    metrics["failed_operations"] += 1

                # Update Average Response Time
                total_ops = metrics["total_operations"]
                current_avg = metrics["avg_response_time_ms"]
                new_avg = ((current_avg * (total_ops - 1)) + response_time_ms) / total_ops
                metrics["avg_response_time_ms"] = new_avg

                # Update Success Rate
                metrics["success_rate"] = metrics["successful_operations"] / metrics["total_operations"]

                # Update Last Operation
                metrics["last_operation"] = operation
                metrics["last_operation_time"] = time.time()

        except Exception as e:
            logger.error(f"Service monitoring status update fehlgeschlagen: {e}")

    def _update_integration_performance_stats(self, processing_time_ms: float) -> None:
        """Aktualisiert Integration-Performance-Statistiken."""
        try:
            self._integration_performance_stats["total_analytics_operations"] += 1

            current_avg = self._integration_performance_stats["avg_analytics_processing_time_ms"]
            total_count = self._integration_performance_stats["total_analytics_operations"]
            new_avg = ((current_avg * (total_count - 1)) + processing_time_ms) / total_count
            self._integration_performance_stats["avg_analytics_processing_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Integration performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            # Hole Statistiken von allen Engines
            event_driven_stats = self.event_driven_analytics_engine.get_performance_stats()
            ml_prediction_stats = self.ml_prediction_engine.get_performance_stats()
            trend_anomaly_stats = self.trend_anomaly_engine.get_performance_stats()
            optimization_stats = self.optimization_engine.get_performance_stats()

            return {
                "analytics_service_integration": {
                    "integration_performance": self._integration_performance_stats,
                    "service_integrations": self._service_integrations,
                    "service_monitoring_status": self._service_monitoring_status
                },
                "event_driven_analytics": event_driven_stats,
                "ml_performance_prediction": ml_prediction_stats,
                "trend_analysis_anomaly_detection": trend_anomaly_stats,
                "performance_optimization": optimization_stats
            }

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
