# backend/services/enhanced_real_time_monitoring/real_time_monitoring_engine.py
"""Enhanced Real-time Monitoring Engine.

Implementiert Enterprise-Grade Real-time Monitoring mit Live-Dashboards,
Performance-Tracking und Integration aller bestehenden Services.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

from kei_logging import get_logger

from ..enhanced_dependency_resolution import EnhancedDependencyResolutionEngine
from ..enhanced_quotas_limits_management import EnhancedQuotaManagementEngine
from ..enhanced_security_integration import EnhancedSecurityIntegrationEngine
from .data_models import (
    AlertRule,
    AlertSeverity,
    LiveDashboardData,
    MetricType,
    MonitoringAlert,
    MonitoringConfiguration,
    MonitoringMetric,
    MonitoringScope,
    PerformanceMetrics,
)

logger = get_logger(__name__)


class EnhancedRealTimeMonitoringEngine:
    """Enhanced Real-time Monitoring Engine für Enterprise-Grade Monitoring."""

    def __init__(
        self,
        dependency_resolution_engine: EnhancedDependencyResolutionEngine | None = None,
        quota_management_engine: EnhancedQuotaManagementEngine | None = None,
        security_integration_engine: EnhancedSecurityIntegrationEngine | None = None,
        configuration: MonitoringConfiguration | None = None
    ):
        """Initialisiert Enhanced Real-time Monitoring Engine.

        Args:
            dependency_resolution_engine: Dependency Resolution Engine
            quota_management_engine: Quota Management Engine
            security_integration_engine: Security Integration Engine
            configuration: Monitoring-Konfiguration
        """
        self.dependency_resolution_engine = dependency_resolution_engine
        self.quota_management_engine = quota_management_engine
        self.security_integration_engine = security_integration_engine
        self.configuration = configuration or MonitoringConfiguration()

        # Monitoring-Storage
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._performance_metrics: dict[str, PerformanceMetrics] = {}
        self._alert_rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, MonitoringAlert] = {}

        # Real-time Data
        self._live_dashboard_data: LiveDashboardData | None = None
        self._service_health_status: dict[str, str] = {}
        self._system_health_score: float = 1.0

        # Performance-Tracking
        self._monitoring_performance_stats = {
            "total_metrics_collected": 0,
            "total_alerts_processed": 0,
            "avg_metric_collection_time_ms": 0.0,
            "avg_alert_processing_time_ms": 0.0,
            "monitoring_overhead_ms": 0.0
        }

        # Background-Tasks
        self._monitoring_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Integration-Tracking
        self._service_integrations: dict[str, dict[str, Any]] = {
            "dependency_resolution": {"enabled": dependency_resolution_engine is not None, "metrics": {}},
            "quota_management": {"enabled": quota_management_engine is not None, "metrics": {}},
            "security_integration": {"enabled": security_integration_engine is not None, "metrics": {}},
            "task_decomposition": {"enabled": False, "metrics": {}},
            "orchestrator": {"enabled": False, "metrics": {}},
            "agent_selection": {"enabled": False, "metrics": {}},
            "performance_prediction": {"enabled": False, "metrics": {}},
            "llm_client": {"enabled": False, "metrics": {}}
        }

        logger.info("Enhanced Real-time Monitoring Engine initialisiert")

    async def start(self) -> None:
        """Startet Real-time Monitoring Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._alert_evaluation_loop()),
            asyncio.create_task(self._dashboard_update_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._service_health_monitoring_loop())
        ]

        logger.info("Enhanced Real-time Monitoring Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Real-time Monitoring Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        self._monitoring_tasks.clear()

        logger.info("Enhanced Real-time Monitoring Engine gestoppt")

    async def collect_metric(
        self,
        metric_name: str,
        value: Any,
        metric_type: MetricType,
        scope: MonitoringScope,
        scope_id: str,
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Sammelt Metrik für Real-time Monitoring.

        Args:
            metric_name: Name der Metrik
            value: Metrik-Wert
            metric_type: Typ der Metrik
            scope: Monitoring-Scope
            scope_id: Scope-ID
            labels: Labels für Metrik
            metadata: Metadata für Metrik
        """
        start_time = time.time()

        try:
            metric = MonitoringMetric(
                metric_id=str(uuid.uuid4()),
                metric_name=metric_name,
                metric_type=metric_type,
                scope=scope,
                value=value,
                unit=self._determine_metric_unit(metric_name, metric_type),
                service_name=labels.get("service_name") if labels else None,
                agent_id=labels.get("agent_id") if labels else None,
                task_id=labels.get("task_id") if labels else None,
                orchestration_id=labels.get("orchestration_id") if labels else None,
                labels=labels or {},
                metadata=metadata or {}
            )

            # Speichere Metrik
            metric_key = f"{scope.value}:{scope_id}:{metric_name}"
            self._metrics[metric_key].append(metric)

            # Update Performance-Stats
            collection_time_ms = (time.time() - start_time) * 1000
            self._update_monitoring_performance_stats("metric_collection", collection_time_ms)

            logger.debug({
                "event": "metric_collected",
                "metric_name": metric_name,
                "scope": scope.value,
                "scope_id": scope_id,
                "value": value,
                "collection_time_ms": collection_time_ms
            })

        except Exception as e:
            logger.error(f"Metric collection fehlgeschlagen: {e}")

    async def update_performance_metrics(
        self,
        scope: MonitoringScope,
        scope_id: str,
        response_time_ms: float,
        success: bool,
        error_details: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Aktualisiert Performance-Metriken.

        Args:
            scope: Monitoring-Scope
            scope_id: Scope-ID
            response_time_ms: Response-Zeit in ms
            success: Erfolg-Status
            error_details: Fehler-Details
            metadata: Metadata
        """
        try:
            metrics_key = f"{scope.value}:{scope_id}"

            if metrics_key not in self._performance_metrics:
                self._performance_metrics[metrics_key] = PerformanceMetrics(
                    metrics_id=metrics_key,
                    scope=scope,
                    scope_id=scope_id
                )

            metrics = self._performance_metrics[metrics_key]

            # Update Response-Time-Metriken
            metrics.total_requests += 1
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

            # Berechne neue Durchschnittswerte
            total_response_time = metrics.avg_response_time_ms * (metrics.total_requests - 1) + response_time_ms
            metrics.avg_response_time_ms = total_response_time / metrics.total_requests

            # Update Max Response Time
            metrics.max_response_time_ms = max(metrics.max_response_time_ms, response_time_ms)

            # Berechne Error-Rate
            metrics.error_rate = metrics.failed_requests / metrics.total_requests
            metrics.success_rate = metrics.successful_requests / metrics.total_requests

            # Update Zeitstempel
            metrics.measurement_end = datetime.now(UTC)

            # Sammle als Metrik
            await self.collect_metric(
                metric_name="response_time",
                value=response_time_ms,
                metric_type=MetricType.TIMER,
                scope=scope,
                scope_id=scope_id,
                labels={"success": str(success)},
                metadata=metadata
            )

            await self.collect_metric(
                metric_name="error_rate",
                value=metrics.error_rate,
                metric_type=MetricType.PERCENTAGE,
                scope=scope,
                scope_id=scope_id,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Performance metrics update fehlgeschlagen: {e}")

    async def create_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        scope: MonitoringScope,
        scope_pattern: str,
        warning_threshold: float | None = None,
        critical_threshold: float | None = None,
        emergency_threshold: float | None = None,
        notification_channels: list[str] | None = None
    ) -> str:
        """Erstellt Alert-Regel.

        Args:
            rule_name: Name der Regel
            metric_name: Name der Metrik
            scope: Monitoring-Scope
            scope_pattern: Scope-Pattern (Regex)
            warning_threshold: Warning-Schwellwert
            critical_threshold: Critical-Schwellwert
            emergency_threshold: Emergency-Schwellwert
            notification_channels: Notification-Channels

        Returns:
            Rule-ID
        """
        try:
            rule_id = str(uuid.uuid4())

            alert_rule = AlertRule(
                rule_id=rule_id,
                rule_name=rule_name,
                description=f"Alert rule for {metric_name} in {scope.value}",
                metric_name=metric_name,
                scope=scope,
                scope_pattern=scope_pattern,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                emergency_threshold=emergency_threshold,
                notification_channels=notification_channels or []
            )

            self._alert_rules[rule_id] = alert_rule

            logger.info({
                "event": "alert_rule_created",
                "rule_id": rule_id,
                "rule_name": rule_name,
                "metric_name": metric_name,
                "scope": scope.value
            })

            return rule_id

        except Exception as e:
            logger.error(f"Alert rule creation fehlgeschlagen: {e}")
            raise

    async def trigger_alert(
        self,
        rule_id: str,
        severity: AlertSeverity,
        metric_value: Any,
        threshold_value: Any,
        scope_id: str,
        message: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Triggert Alert.

        Args:
            rule_id: Rule-ID
            severity: Alert-Schweregrad
            metric_value: Aktueller Metrik-Wert
            threshold_value: Schwellwert
            scope_id: Scope-ID
            message: Alert-Message
            metadata: Metadata

        Returns:
            Alert-ID
        """
        start_time = time.time()

        try:
            rule = self._alert_rules.get(rule_id)
            if not rule:
                raise ValueError(f"Alert rule {rule_id} nicht gefunden")

            alert_id = str(uuid.uuid4())

            alert = MonitoringAlert(
                alert_id=alert_id,
                rule_id=rule_id,
                alert_name=f"{rule.rule_name} - {severity.value.upper()}",
                severity=severity,
                message=message or f"{rule.metric_name} threshold exceeded",
                description=f"{rule.description} - Value: {metric_value}, Threshold: {threshold_value}",
                scope=rule.scope,
                scope_id=scope_id,
                metric_name=rule.metric_name,
                metric_value=metric_value,
                threshold_value=threshold_value,
                metadata=metadata or {}
            )

            self._active_alerts[alert_id] = alert

            # Update Performance-Stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_monitoring_performance_stats("alert_processing", processing_time_ms)

            logger.warning({
                "event": "alert_triggered",
                "alert_id": alert_id,
                "rule_id": rule_id,
                "severity": severity.value,
                "metric_name": rule.metric_name,
                "metric_value": metric_value,
                "threshold_value": threshold_value,
                "scope_id": scope_id
            })

            return alert_id

        except Exception as e:
            logger.error(f"Alert triggering fehlgeschlagen: {e}")
            raise

    async def get_live_dashboard_data(self) -> LiveDashboardData:
        """Holt Live-Dashboard-Daten.

        Returns:
            Live-Dashboard-Daten
        """
        try:
            if not self._live_dashboard_data:
                await self._generate_dashboard_data()

            return self._live_dashboard_data

        except Exception as e:
            logger.error(f"Live dashboard data retrieval fehlgeschlagen: {e}")
            raise

    async def monitor_service_integration(
        self,
        service_name: str,
        operation: str,
        response_time_ms: float,
        success: bool,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Monitort Service-Integration.

        Args:
            service_name: Service-Name
            operation: Operation-Name
            response_time_ms: Response-Zeit
            success: Erfolg-Status
            metadata: Metadata
        """
        try:
            # Update Service-Integration-Metriken
            if service_name in self._service_integrations:
                integration = self._service_integrations[service_name]

                if "total_operations" not in integration["metrics"]:
                    integration["metrics"]["total_operations"] = 0
                    integration["metrics"]["successful_operations"] = 0
                    integration["metrics"]["failed_operations"] = 0
                    integration["metrics"]["avg_response_time_ms"] = 0.0

                integration["metrics"]["total_operations"] += 1

                if success:
                    integration["metrics"]["successful_operations"] += 1
                else:
                    integration["metrics"]["failed_operations"] += 1

                # Update Average Response Time
                total_ops = integration["metrics"]["total_operations"]
                current_avg = integration["metrics"]["avg_response_time_ms"]
                new_avg = ((current_avg * (total_ops - 1)) + response_time_ms) / total_ops
                integration["metrics"]["avg_response_time_ms"] = new_avg

            # Sammle als Performance-Metrik
            await self.update_performance_metrics(
                scope=MonitoringScope.SERVICE,
                scope_id=service_name,
                response_time_ms=response_time_ms,
                success=success,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Service integration monitoring fehlgeschlagen: {e}")

    async def _metrics_collection_loop(self) -> None:
        """Background-Loop für Metrics-Collection."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.metrics_collection_interval_seconds)

                if self._is_running:
                    await self._collect_system_metrics()

            except Exception as e:
                logger.error(f"Metrics collection loop fehlgeschlagen: {e}")
                await asyncio.sleep(self.configuration.metrics_collection_interval_seconds)

    async def _alert_evaluation_loop(self) -> None:
        """Background-Loop für Alert-Evaluation."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.alert_evaluation_interval_seconds)

                if self._is_running:
                    await self._evaluate_alert_rules()

            except Exception as e:
                logger.error(f"Alert evaluation loop fehlgeschlagen: {e}")
                await asyncio.sleep(self.configuration.alert_evaluation_interval_seconds)

    async def _dashboard_update_loop(self) -> None:
        """Background-Loop für Dashboard-Updates."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.dashboard_refresh_interval_seconds)

                if self._is_running:
                    await self._generate_dashboard_data()

            except Exception as e:
                logger.error(f"Dashboard update loop fehlgeschlagen: {e}")
                await asyncio.sleep(self.configuration.dashboard_refresh_interval_seconds)

    async def _performance_monitoring_loop(self) -> None:
        """Background-Loop für Performance-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._monitor_system_performance()

            except Exception as e:
                logger.error(f"Performance monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _service_health_monitoring_loop(self) -> None:
        """Background-Loop für Service-Health-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Alle 30 Sekunden

                if self._is_running:
                    await self._monitor_service_health()

            except Exception as e:
                logger.error(f"Service health monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(30)

    async def _collect_system_metrics(self) -> None:
        """Sammelt System-Metriken."""
        try:
            if psutil is None:
                logger.warning("psutil nicht verfügbar - System-Metriken werden übersprungen")
                return

            # CPU-Metriken
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.collect_metric(
                metric_name="cpu_usage",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                scope=MonitoringScope.SYSTEM,
                scope_id="system",
                labels={"unit": "percent"}
            )

            # Memory-Metriken
            memory = psutil.virtual_memory()
            await self.collect_metric(
                metric_name="memory_usage",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                scope=MonitoringScope.SYSTEM,
                scope_id="system",
                labels={"unit": "percent"}
            )

            # Disk-Metriken
            disk = psutil.disk_usage("/")
            await self.collect_metric(
                metric_name="disk_usage",
                value=disk.percent,
                metric_type=MetricType.GAUGE,
                scope=MonitoringScope.SYSTEM,
                scope_id="system",
                labels={"unit": "percent"}
            )

        except Exception as e:
            logger.error(f"System metrics collection fehlgeschlagen: {e}")

    async def _evaluate_alert_rules(self) -> None:
        """Evaluiert Alert-Regeln."""
        try:
            for rule_id, rule in self._alert_rules.items():
                if not rule.enabled:
                    continue

                # Prüfe Cooldown
                if rule.last_triggered:
                    time_since_last = (datetime.now(UTC) - rule.last_triggered).total_seconds()
                    if time_since_last < rule.cooldown_seconds:
                        continue

                # Hole aktuelle Metrik-Werte
                await self._evaluate_single_alert_rule(rule)

        except Exception as e:
            logger.error(f"Alert rules evaluation fehlgeschlagen: {e}")

    async def _evaluate_single_alert_rule(self, rule: AlertRule) -> None:
        """Evaluiert einzelne Alert-Regel."""
        try:
            # Finde passende Metriken
            matching_metrics = []

            for metric_key, metric_deque in self._metrics.items():
                scope_str, scope_id, metric_name = metric_key.split(":", 2)

                if (metric_name == rule.metric_name and
                    scope_str == rule.scope.value and
                    self._matches_pattern(scope_id, rule.scope_pattern)):

                    if metric_deque:
                        latest_metric = metric_deque[-1]
                        matching_metrics.append((latest_metric, scope_id))

            # Evaluiere Schwellwerte
            for metric, scope_id in matching_metrics:
                metric_value = float(metric.value) if isinstance(metric.value, (int, float)) else 0.0

                severity = None
                threshold_value = None

                if rule.emergency_threshold and metric_value >= rule.emergency_threshold:
                    severity = AlertSeverity.EMERGENCY
                    threshold_value = rule.emergency_threshold
                elif rule.critical_threshold and metric_value >= rule.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = rule.critical_threshold
                elif rule.warning_threshold and metric_value >= rule.warning_threshold:
                    severity = AlertSeverity.WARNING
                    threshold_value = rule.warning_threshold

                if severity:
                    await self.trigger_alert(
                        rule_id=rule.rule_id,
                        severity=severity,
                        metric_value=metric_value,
                        threshold_value=threshold_value,
                        scope_id=scope_id
                    )

                    rule.last_triggered = datetime.now(UTC)

        except Exception as e:
            logger.error(f"Single alert rule evaluation fehlgeschlagen: {e}")

    async def _generate_dashboard_data(self) -> None:
        """Generiert Dashboard-Daten."""
        try:
            dashboard_data = LiveDashboardData(
                dashboard_id="main_dashboard",
                dashboard_name="Enhanced Real-time Monitoring Dashboard"
            )

            # System-Health
            dashboard_data.system_health = self._calculate_system_health()
            dashboard_data.active_alerts = len(self._active_alerts)

            # Service-Metriken
            dashboard_data.total_services = len(self._service_integrations)
            dashboard_data.healthy_services = len([
                s for s in self._service_integrations.values()
                if s.get("enabled", False)
            ])

            # Performance-Übersicht
            if self._performance_metrics:
                all_metrics = list(self._performance_metrics.values())
                dashboard_data.avg_response_time_ms = sum(m.avg_response_time_ms for m in all_metrics) / len(all_metrics)
                dashboard_data.overall_error_rate = sum(m.error_rate for m in all_metrics) / len(all_metrics)
                dashboard_data.overall_success_rate = sum(m.success_rate for m in all_metrics) / len(all_metrics)

            # Service-Details
            dashboard_data.service_metrics = self._performance_metrics.copy()

            self._live_dashboard_data = dashboard_data

        except Exception as e:
            logger.error(f"Dashboard data generation fehlgeschlagen: {e}")

    async def _monitor_system_performance(self) -> None:
        """Monitort System-Performance."""
        try:
            # Berechne Monitoring-Overhead
            start_time = time.time()

            # Simuliere Monitoring-Arbeit
            await asyncio.sleep(0.001)

            monitoring_overhead_ms = (time.time() - start_time) * 1000
            self._monitoring_performance_stats["monitoring_overhead_ms"] = monitoring_overhead_ms

            # Sammle als Metrik
            await self.collect_metric(
                metric_name="monitoring_overhead",
                value=monitoring_overhead_ms,
                metric_type=MetricType.TIMER,
                scope=MonitoringScope.SYSTEM,
                scope_id="monitoring_engine"
            )

        except Exception as e:
            logger.error(f"System performance monitoring fehlgeschlagen: {e}")

    async def _monitor_service_health(self) -> None:
        """Monitort Service-Health."""
        try:
            for service_name, integration in self._service_integrations.items():
                if integration["enabled"]:
                    # Simuliere Health-Check
                    health_status = "healthy"  # In Realität würde hier ein echter Health-Check stattfinden
                    self._service_health_status[service_name] = health_status

                    await self.collect_metric(
                        metric_name="service_health",
                        value=1.0 if health_status == "healthy" else 0.0,
                        metric_type=MetricType.GAUGE,
                        scope=MonitoringScope.SERVICE,
                        scope_id=service_name,
                        labels={"status": health_status}
                    )

        except Exception as e:
            logger.error(f"Service health monitoring fehlgeschlagen: {e}")

    def _determine_metric_unit(self, metric_name: str, metric_type: MetricType) -> str:
        """Bestimmt Metrik-Unit basierend auf Name und Typ."""
        if "time" in metric_name.lower() or "duration" in metric_name.lower():
            return "ms"
        if "rate" in metric_name.lower() or "percent" in metric_name.lower():
            return "percent"
        if "count" in metric_name.lower() or metric_type == MetricType.COUNTER:
            return "count"
        if "bytes" in metric_name.lower():
            return "bytes"
        return "unit"

    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Prüft ob Wert Pattern entspricht."""
        try:
            return bool(re.match(pattern, value))
        except re.error:
            return value == pattern

    def _calculate_system_health(self) -> str:
        """Berechnet System-Health-Status."""
        try:
            if len(self._active_alerts) == 0:
                return "healthy"

            critical_alerts = len([
                alert for alert in self._active_alerts.values()
                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ])

            if critical_alerts > 0:
                return "unhealthy"
            if len(self._active_alerts) > 5:
                return "degraded"
            return "healthy"

        except Exception as e:
            logger.error(f"System health calculation fehlgeschlagen: {e}")
            return "unknown"

    def _update_monitoring_performance_stats(self, operation: str, duration_ms: float) -> None:
        """Aktualisiert Monitoring-Performance-Statistiken."""
        try:
            if operation == "metric_collection":
                self._monitoring_performance_stats["total_metrics_collected"] += 1
                current_avg = self._monitoring_performance_stats["avg_metric_collection_time_ms"]
                total_count = self._monitoring_performance_stats["total_metrics_collected"]
                new_avg = ((current_avg * (total_count - 1)) + duration_ms) / total_count
                self._monitoring_performance_stats["avg_metric_collection_time_ms"] = new_avg

            elif operation == "alert_processing":
                self._monitoring_performance_stats["total_alerts_processed"] += 1
                current_avg = self._monitoring_performance_stats["avg_alert_processing_time_ms"]
                total_count = self._monitoring_performance_stats["total_alerts_processed"]
                new_avg = ((current_avg * (total_count - 1)) + duration_ms) / total_count
                self._monitoring_performance_stats["avg_alert_processing_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Monitoring performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = self._monitoring_performance_stats.copy()

            # SLA-Compliance
            stats["meets_monitoring_sla"] = stats["monitoring_overhead_ms"] < self.configuration.performance_monitoring_overhead_ms

            # Service-Integration-Stats
            stats["service_integrations"] = {
                name: {
                    "enabled": integration["enabled"],
                    "metrics": integration["metrics"]
                }
                for name, integration in self._service_integrations.items()
            }

            # Dashboard-Stats
            stats["dashboard_stats"] = {
                "total_metrics": sum(len(deque_obj) for deque_obj in self._metrics.values()),
                "active_alerts": len(self._active_alerts),
                "alert_rules": len(self._alert_rules),
                "performance_metrics": len(self._performance_metrics)
            }

            return stats

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
