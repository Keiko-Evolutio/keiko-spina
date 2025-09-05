# backend/services/enhanced_real_time_monitoring/service_integration_layer.py
"""Service Integration Layer für Enhanced Real-time Monitoring.

Integriert Enhanced Real-time Monitoring mit allen bestehenden Services
und erweitert sie um Enterprise-Grade Monitoring-Features.
"""

from __future__ import annotations

import time
from typing import Any

from kei_logging import get_logger
from services.enhanced_dependency_resolution import EnhancedDependencyResolutionEngine
from services.enhanced_quotas_limits_management import EnhancedQuotaManagementEngine
from services.enhanced_security_integration import (
    EnhancedSecurityIntegrationEngine,
    SecurityContext,
)

from .data_models import AlertSeverity, MetricType, MonitoringScope
from .distributed_tracing_engine import DistributedTracingEngine
from .live_dashboard_engine import LiveDashboardEngine
from .real_time_monitoring_engine import EnhancedRealTimeMonitoringEngine
from .saga_coordinator_engine import SagaCoordinatorEngine

logger = get_logger(__name__)


class MonitoringServiceIntegrationLayer:
    """Service Integration Layer für Enhanced Real-time Monitoring."""

    def __init__(
        self,
        monitoring_engine: EnhancedRealTimeMonitoringEngine,
        saga_coordinator: SagaCoordinatorEngine,
        tracing_engine: DistributedTracingEngine,
        dashboard_engine: LiveDashboardEngine,
        dependency_resolution_engine: EnhancedDependencyResolutionEngine | None = None,
        quota_management_engine: EnhancedQuotaManagementEngine | None = None,
        security_integration_engine: EnhancedSecurityIntegrationEngine | None = None
    ):
        """Initialisiert Monitoring Service Integration Layer.

        Args:
            monitoring_engine: Real-time Monitoring Engine
            saga_coordinator: Saga Coordinator Engine
            tracing_engine: Distributed Tracing Engine
            dashboard_engine: Live Dashboard Engine
            dependency_resolution_engine: Dependency Resolution Engine
            quota_management_engine: Quota Management Engine
            security_integration_engine: Security Integration Engine
        """
        self.monitoring_engine = monitoring_engine
        self.saga_coordinator = saga_coordinator
        self.tracing_engine = tracing_engine
        self.dashboard_engine = dashboard_engine
        self.dependency_resolution_engine = dependency_resolution_engine
        self.quota_management_engine = quota_management_engine
        self.security_integration_engine = security_integration_engine

        # Integration-Konfiguration
        self.enable_dependency_monitoring = True
        self.enable_quota_monitoring = True
        self.enable_security_monitoring = True
        self.enable_saga_monitoring = True
        self.enable_distributed_tracing = True
        self.enable_live_dashboards = True

        # Performance-Tracking
        self._integration_count = 0
        self._total_integration_time_ms = 0.0
        self._monitored_operations = 0

        # Service-Monitoring-Status
        self._service_monitoring_status: dict[str, dict[str, Any]] = {
            "dependency_resolution": {"enabled": dependency_resolution_engine is not None, "metrics": {}},
            "quota_management": {"enabled": quota_management_engine is not None, "metrics": {}},
            "security_integration": {"enabled": security_integration_engine is not None, "metrics": {}},
            "task_decomposition": {"enabled": False, "metrics": {}},
            "orchestrator": {"enabled": False, "metrics": {}},
            "agent_selection": {"enabled": False, "metrics": {}},
            "performance_prediction": {"enabled": False, "metrics": {}},
            "llm_client": {"enabled": False, "metrics": {}}
        }

        logger.info("Monitoring Service Integration Layer initialisiert")

    async def start(self) -> None:
        """Startet Monitoring Service Integration Layer."""
        try:
            # Starte alle Monitoring-Engines
            await self.monitoring_engine.start()
            await self.saga_coordinator.start()
            await self.tracing_engine.start()
            await self.dashboard_engine.start()

            # Erstelle Standard-Dashboard
            await self._create_default_dashboards()

            # Erstelle Standard-Alert-Rules
            await self._create_default_alert_rules()

            logger.info("Monitoring Service Integration Layer gestartet")

        except Exception as e:
            logger.error(f"Monitoring Service Integration Layer start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Monitoring Service Integration Layer."""
        try:
            await self.monitoring_engine.stop()
            await self.saga_coordinator.stop()
            await self.tracing_engine.stop()
            await self.dashboard_engine.stop()

            logger.info("Monitoring Service Integration Layer gestoppt")

        except Exception as e:
            logger.error(f"Monitoring Service Integration Layer stop fehlgeschlagen: {e}")

    async def monitor_dependency_resolution(
        self,
        operation: str,
        request_data: dict[str, Any],
        _result_data: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext
    ) -> None:
        """Monitort Dependency Resolution Operations.

        Args:
            operation: Operation-Name
            request_data: Request-Daten
            _result_data: Result-Daten
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
        """
        try:
            if not self.enable_dependency_monitoring:
                return

            # Erstelle Trace
            trace_id = await self.tracing_engine.create_trace(
                trace_name=f"dependency_resolution_{operation}",
                service_name="dependency_resolution",
                operation_name=operation,
                user_id=security_context.user_id,
                tenant_id=security_context.tenant_id,
                metadata={"request_data": request_data}
            )

            if trace_id:
                span_id = await self.tracing_engine.create_span(
                    trace_id=trace_id,
                    operation_name=operation,
                    service_name="dependency_resolution",
                    component="resolution_engine",
                    tags={
                        "success": str(success),
                        "user_id": security_context.user_id,
                        "tenant_id": security_context.tenant_id
                    }
                )

                await self.tracing_engine.finish_span(
                    span_id=span_id,
                    status_code=200 if success else 500,
                    error=None if success else "Dependency resolution failed"
                )

            # Sammle Metriken
            await self.monitoring_engine.collect_metric(
                metric_name="dependency_resolution_time",
                value=response_time_ms,
                metric_type=MetricType.TIMER,
                scope=MonitoringScope.SERVICE,
                scope_id="dependency_resolution",
                labels={
                    "operation": operation,
                    "success": str(success),
                    "user_id": security_context.user_id
                }
            )

            # Update Performance-Metriken
            await self.monitoring_engine.update_performance_metrics(
                scope=MonitoringScope.SERVICE,
                scope_id="dependency_resolution",
                response_time_ms=response_time_ms,
                success=success,
                metadata={"operation": operation}
            )

            # Update Service-Status
            await self._update_service_monitoring_status(
                "dependency_resolution", operation, response_time_ms, success
            )

            logger.debug({
                "event": "dependency_resolution_monitored",
                "operation": operation,
                "response_time_ms": response_time_ms,
                "success": success,
                "trace_id": trace_id
            })

        except Exception as e:
            logger.error(f"Dependency resolution monitoring fehlgeschlagen: {e}")

    async def monitor_quota_management(
        self,
        operation: str,
        resource_type: str,
        quota_check_result: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext
    ) -> None:
        """Monitort Quota Management Operations.

        Args:
            operation: Operation-Name
            resource_type: Resource-Type
            quota_check_result: Quota-Check-Result
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
        """
        try:
            if not self.enable_quota_monitoring:
                return

            # Sammle Quota-Metriken
            await self.monitoring_engine.collect_metric(
                metric_name="quota_check_time",
                value=response_time_ms,
                metric_type=MetricType.TIMER,
                scope=MonitoringScope.RESOURCE,
                scope_id=resource_type,
                labels={
                    "operation": operation,
                    "success": str(success),
                    "tenant_id": security_context.tenant_id
                }
            )

            # Sammle Quota-Usage-Metriken
            if quota_check_result.get("usage"):
                usage = quota_check_result["usage"]
                limit = quota_check_result.get("limit", 1)
                usage_percentage = (usage / limit) * 100 if limit > 0 else 0

                await self.monitoring_engine.collect_metric(
                    metric_name="quota_usage_percentage",
                    value=usage_percentage,
                    metric_type=MetricType.PERCENTAGE,
                    scope=MonitoringScope.RESOURCE,
                    scope_id=resource_type,
                    labels={
                        "tenant_id": security_context.tenant_id,
                        "resource_type": resource_type
                    }
                )

                # Alert bei hoher Quota-Usage
                if usage_percentage > 90:
                    await self.monitoring_engine.trigger_alert(
                        rule_id="quota_high_usage",
                        severity=AlertSeverity.WARNING if usage_percentage < 95 else AlertSeverity.CRITICAL,
                        metric_value=usage_percentage,
                        threshold_value=90,
                        scope_id=resource_type,
                        message=f"High quota usage for {resource_type}: {usage_percentage:.1f}%"
                    )

            # Update Service-Status
            await self._update_service_monitoring_status(
                "quota_management", operation, response_time_ms, success
            )

        except Exception as e:
            logger.error(f"Quota management monitoring fehlgeschlagen: {e}")

    async def monitor_security_integration(
        self,
        operation: str,
        security_check_result: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext
    ) -> None:
        """Monitort Security Integration Operations.

        Args:
            operation: Operation-Name
            security_check_result: Security-Check-Result
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
        """
        try:
            if not self.enable_security_monitoring:
                return

            # Sammle Security-Metriken
            await self.monitoring_engine.collect_metric(
                metric_name="security_check_time",
                value=response_time_ms,
                metric_type=MetricType.TIMER,
                scope=MonitoringScope.SECURITY,
                scope_id="security_integration",
                labels={
                    "operation": operation,
                    "success": str(success),
                    "security_level": security_context.security_level.value
                }
            )

            # Sammle Security-Score-Metriken
            if security_check_result.get("security_score"):
                security_score = security_check_result["security_score"]

                await self.monitoring_engine.collect_metric(
                    metric_name="security_score",
                    value=security_score,
                    metric_type=MetricType.GAUGE,
                    scope=MonitoringScope.SECURITY,
                    scope_id="security_integration",
                    labels={
                        "operation": operation,
                        "user_id": security_context.user_id
                    }
                )

                # Alert bei niedrigem Security-Score
                if security_score < 0.7:
                    await self.monitoring_engine.trigger_alert(
                        rule_id="security_low_score",
                        severity=AlertSeverity.WARNING if security_score > 0.5 else AlertSeverity.CRITICAL,
                        metric_value=security_score,
                        threshold_value=0.7,
                        scope_id="security_integration",
                        message=f"Low security score for {operation}: {security_score:.2f}"
                    )

            # Update Service-Status
            await self._update_service_monitoring_status(
                "security_integration", operation, response_time_ms, success
            )

        except Exception as e:
            logger.error(f"Security integration monitoring fehlgeschlagen: {e}")

    async def create_saga_for_orchestration(
        self,
        orchestration_id: str,
        orchestration_steps: list[dict[str, Any]],
        security_context: SecurityContext
    ) -> str:
        """Erstellt Saga für Orchestration.

        Args:
            orchestration_id: Orchestration-ID
            orchestration_steps: Orchestration-Steps
            security_context: Security-Context

        Returns:
            Saga-ID
        """
        try:
            if not self.enable_saga_monitoring:
                return ""

            # Erstelle Saga
            saga_id = await self.saga_coordinator.create_saga(
                saga_name=f"orchestration_{orchestration_id}",
                description=f"Saga for orchestration {orchestration_id}",
                steps=orchestration_steps,
                compensation_strategy="reverse_order",
                security_context=security_context
            )

            # Sammle Saga-Creation-Metrik
            await self.monitoring_engine.collect_metric(
                metric_name="saga_created",
                value=1,
                metric_type=MetricType.COUNTER,
                scope=MonitoringScope.ORCHESTRATION,
                scope_id=orchestration_id,
                labels={
                    "saga_id": saga_id,
                    "steps_count": str(len(orchestration_steps))
                }
            )

            logger.info({
                "event": "saga_created_for_orchestration",
                "orchestration_id": orchestration_id,
                "saga_id": saga_id,
                "steps_count": len(orchestration_steps)
            })

            return saga_id

        except Exception as e:
            logger.error(f"Saga creation for orchestration fehlgeschlagen: {e}")
            return ""

    async def execute_saga_with_monitoring(
        self,
        saga_id: str,
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Führt Saga mit Monitoring aus.

        Args:
            saga_id: Saga-ID
            security_context: Security-Context

        Returns:
            Execution-Result
        """
        try:
            start_time = time.time()

            # Führe Saga aus
            result = await self.saga_coordinator.execute_saga(
                saga_id=saga_id,
                security_context=security_context
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Sammle Saga-Execution-Metriken
            await self.monitoring_engine.collect_metric(
                metric_name="saga_execution_time",
                value=execution_time_ms,
                metric_type=MetricType.TIMER,
                scope=MonitoringScope.ORCHESTRATION,
                scope_id=saga_id,
                labels={
                    "success": str(result["success"]),
                    "completed_steps": str(result.get("completed_steps", 0))
                }
            )

            # Alert bei Saga-Failure
            if not result["success"]:
                await self.monitoring_engine.trigger_alert(
                    rule_id="saga_execution_failed",
                    severity=AlertSeverity.CRITICAL,
                    metric_value=0,
                    threshold_value=1,
                    scope_id=saga_id,
                    message=f"Saga execution failed: {result.get('error', 'Unknown error')}"
                )

            return result

        except Exception as e:
            logger.error(f"Saga execution with monitoring fehlgeschlagen: {e}")
            return {"success": False, "error": str(e)}

    async def monitor_service_operation(
        self,
        service_name: str,
        operation: str,
        _request_data: dict[str, Any],
        _result_data: dict[str, Any],
        response_time_ms: float,
        success: bool,
        security_context: SecurityContext,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Monitort allgemeine Service-Operation.

        Args:
            service_name: Service-Name
            operation: Operation-Name
            _request_data: Request-Daten
            _result_data: Result-Daten
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            security_context: Security-Context
            metadata: Metadata
        """
        try:
            # Erstelle Trace
            trace_id = await self.tracing_engine.create_trace(
                trace_name=f"{service_name}_{operation}",
                service_name=service_name,
                operation_name=operation,
                user_id=security_context.user_id,
                tenant_id=security_context.tenant_id,
                metadata=metadata or {}
            )

            if trace_id:
                span_id = await self.tracing_engine.create_span(
                    trace_id=trace_id,
                    operation_name=operation,
                    service_name=service_name,
                    component="service_operation",
                    tags={
                        "success": str(success),
                        "user_id": security_context.user_id
                    }
                )

                await self.tracing_engine.finish_span(
                    span_id=span_id,
                    status_code=200 if success else 500,
                    error=None if success else f"{service_name} operation failed"
                )

            # Sammle Service-Metriken
            await self.monitoring_engine.collect_metric(
                metric_name=f"{service_name}_operation_time",
                value=response_time_ms,
                metric_type=MetricType.TIMER,
                scope=MonitoringScope.SERVICE,
                scope_id=service_name,
                labels={
                    "operation": operation,
                    "success": str(success)
                }
            )

            # Update Performance-Metriken
            await self.monitoring_engine.update_performance_metrics(
                scope=MonitoringScope.SERVICE,
                scope_id=service_name,
                response_time_ms=response_time_ms,
                success=success,
                metadata={"operation": operation}
            )

            # Update Service-Status
            await self._update_service_monitoring_status(
                service_name, operation, response_time_ms, success
            )

            self._monitored_operations += 1

        except Exception as e:
            logger.error(f"Service operation monitoring fehlgeschlagen: {e}")

    async def _create_default_dashboards(self) -> None:
        """Erstellt Standard-Dashboards."""
        try:
            # Main System Dashboard
            main_dashboard_id = await self.dashboard_engine.create_dashboard(
                dashboard_name="Enhanced Real-time Monitoring Dashboard",
                dashboard_config={
                    "type": "main_system",
                    "refresh_interval": 5,
                    "widgets": [
                        "system_health",
                        "active_alerts",
                        "service_performance",
                        "saga_status",
                        "trace_overview"
                    ]
                }
            )

            logger.debug(f"Main dashboard created with ID: {main_dashboard_id}")

            # Service-specific Dashboards
            for service_name in self._service_monitoring_status.keys():
                if self._service_monitoring_status[service_name]["enabled"]:
                    await self.dashboard_engine.create_dashboard(
                        dashboard_name=f"{service_name.title()} Service Dashboard",
                        dashboard_config={
                            "type": "service_specific",
                            "service_name": service_name,
                            "refresh_interval": 10
                        }
                    )

            logger.info("Standard-Dashboards erstellt")

        except Exception as e:
            logger.error(f"Default dashboards creation fehlgeschlagen: {e}")

    async def _create_default_alert_rules(self) -> None:
        """Erstellt Standard-Alert-Rules."""
        try:
            # High Response Time Alert
            await self.monitoring_engine.create_alert_rule(
                rule_name="High Response Time",
                metric_name="response_time",
                scope=MonitoringScope.SERVICE,
                scope_pattern=".*",
                warning_threshold=1000.0,  # 1 Sekunde
                critical_threshold=5000.0,  # 5 Sekunden
                emergency_threshold=10000.0  # 10 Sekunden
            )

            # High Error Rate Alert
            await self.monitoring_engine.create_alert_rule(
                rule_name="High Error Rate",
                metric_name="error_rate",
                scope=MonitoringScope.SERVICE,
                scope_pattern=".*",
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
                emergency_threshold=0.25   # 25%
            )

            # System Resource Alerts
            await self.monitoring_engine.create_alert_rule(
                rule_name="High CPU Usage",
                metric_name="cpu_usage",
                scope=MonitoringScope.SYSTEM,
                scope_pattern="system",
                warning_threshold=80.0,
                critical_threshold=90.0,
                emergency_threshold=95.0
            )

            logger.info("Standard-Alert-Rules erstellt")

        except Exception as e:
            logger.error(f"Default alert rules creation fehlgeschlagen: {e}")

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

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            # Hole Statistiken von allen Komponenten
            monitoring_stats = self.monitoring_engine.get_performance_stats()
            saga_stats = self.saga_coordinator.get_performance_stats()
            tracing_stats = self.tracing_engine.get_performance_stats()
            dashboard_stats = self.dashboard_engine.get_performance_stats()

            return {
                "monitoring_service_integration": {
                    "integration_count": self._integration_count,
                    "monitored_operations": self._monitored_operations,
                    "service_monitoring_status": self._service_monitoring_status,
                    "dependency_monitoring_enabled": self.enable_dependency_monitoring,
                    "quota_monitoring_enabled": self.enable_quota_monitoring,
                    "security_monitoring_enabled": self.enable_security_monitoring,
                    "saga_monitoring_enabled": self.enable_saga_monitoring,
                    "distributed_tracing_enabled": self.enable_distributed_tracing,
                    "live_dashboards_enabled": self.enable_live_dashboards
                },
                "real_time_monitoring": monitoring_stats,
                "saga_coordinator": saga_stats,
                "distributed_tracing": tracing_stats,
                "live_dashboard": dashboard_stats
            }

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
