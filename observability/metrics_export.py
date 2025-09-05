# backend/observability/metrics_export.py
"""Metriken-Export und -Visualisierung für Keiko Personal Assistant

Implementiert Prometheus-kompatible Metriken-Endpoints, JSON-API für Metriken-Abfrage,
Dashboard-Integration und Alerting-Integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from kei_logging import get_logger
from observability import trace_function

from .agent_metrics import get_agent_metrics_collector, get_all_agent_metrics
from .base_metrics import MetricsConstants
from .metrics_aggregator import AggregationType, AggregationWindow, metrics_aggregator
from .system_integration_metrics import system_integration_metrics

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class MetricsFormat(str, Enum):
    """Metriken-Export-Formate."""
    PROMETHEUS = MetricsConstants.PROMETHEUS_FORMAT
    JSON = MetricsConstants.JSON_FORMAT
    CSV = MetricsConstants.CSV_FORMAT
    INFLUXDB = MetricsConstants.INFLUXDB_FORMAT


class AlertSeverity(str, Enum):
    """Alert-Schweregrade."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Alert-Regel-Definition."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    description: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "description": self.description,
            "enabled": self.enabled
        }


@dataclass
class AlertEvent:
    """Alert-Event."""
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


# Pydantic Models für API
class MetricsQueryRequest(BaseModel):
    """Request für Metriken-Abfrage."""
    metric_names: list[str] | None = Field(default=None, description="Liste der Metrik-Namen")
    agent_ids: list[str] | None = Field(default=None, description="Liste der Agent-IDs")
    start_time: datetime | None = Field(default=None, description="Start-Zeit")
    end_time: datetime | None = Field(default=None, description="End-Zeit")
    aggregation_window: AggregationWindow | None = Field(default=None, description="Aggregations-Zeitfenster")
    aggregation_types: list[AggregationType] | None = Field(default=None, description="Aggregations-Typen")
    tags: dict[str, str] | None = Field(default=None, description="Filter-Tags")
    limit: int | None = Field(default=1000, ge=1, le=10000, description="Maximale Anzahl Ergebnisse")


class MetricsResponse(BaseModel):
    """Response für Metriken-Abfrage."""
    metrics: dict[str, Any] = Field(description="Metriken-Daten")
    metadata: dict[str, Any] = Field(description="Metadata")
    query_time_ms: float = Field(description="Query-Zeit in Millisekunden")
    total_results: int = Field(description="Gesamtanzahl Ergebnisse")


class AlertRuleRequest(BaseModel):
    """Request für Alert-Regel."""
    name: str = Field(description="Name der Alert-Regel")
    metric_name: str = Field(description="Name der Metrik")
    condition: str = Field(description="Bedingung (gt, lt, eq, ne)")
    threshold: float = Field(description="Schwellwert")
    severity: AlertSeverity = Field(description="Schweregrad")
    description: str = Field(description="Beschreibung")
    enabled: bool = Field(default=True, description="Aktiviert")


class DashboardConfig(BaseModel):
    """Dashboard-Konfiguration."""
    name: str = Field(description="Dashboard-Name")
    panels: list[dict[str, Any]] = Field(description="Dashboard-Panels")
    refresh_interval_seconds: int = Field(default=30, description="Refresh-Intervall")
    time_range_hours: int = Field(default=24, description="Zeit-Range in Stunden")


class PrometheusExporter:
    """Prometheus-Metriken-Exporter."""

    def __init__(self):
        """Initialisiert Prometheus Exporter."""
        self._metric_cache: dict[str, str] = {}
        self._cache_ttl_seconds = 30
        self._last_cache_update = 0

    @trace_function("prometheus_exporter.export_metrics")
    async def export_metrics(self) -> str:
        """Exportiert Metriken im Prometheus-Format.

        Returns:
            Prometheus-formatierte Metriken
        """
        current_time = time.time()

        # Cache-Check
        if (current_time - self._last_cache_update) < self._cache_ttl_seconds:
            return self._get_cached_metrics()

        # Sammle alle Metriken
        prometheus_lines = []

        # Agent-Metriken
        agent_metrics = get_all_agent_metrics()
        prometheus_lines.extend(self._format_agent_metrics(agent_metrics))

        # System-Metriken
        system_stats = system_integration_metrics.get_integration_statistics()
        prometheus_lines.extend(self._format_system_metrics(system_stats))

        # Aggregierte Metriken
        aggregated_metrics = await metrics_aggregator.get_aggregated_metrics()
        prometheus_lines.extend(self._format_aggregated_metrics(aggregated_metrics))

        # Cache aktualisieren
        prometheus_output = "\n".join(prometheus_lines)
        self._metric_cache["prometheus"] = prometheus_output
        self._last_cache_update = current_time

        return prometheus_output

    def _get_cached_metrics(self) -> str:
        """Gibt gecachte Metriken zurück."""
        return self._metric_cache.get("prometheus", "")

    def _format_agent_metrics(self, agent_metrics: dict[str, dict[str, Any]]) -> list[str]:
        """Formatiert Agent-Metriken für Prometheus.

        Args:
            agent_metrics: Agent-Metriken

        Returns:
            Liste von Prometheus-Zeilen
        """
        # Optimierte List-Creation mit statischen Header-Zeilen
        lines = [
            # HELP und TYPE Definitionen
            "# HELP kei_agent_task_duration_ms Task execution duration in milliseconds",
            "# TYPE kei_agent_task_duration_ms histogram",
            "# HELP kei_agent_tasks_total Total number of tasks processed",
            "# TYPE kei_agent_tasks_total counter",
            "# HELP kei_agent_error_rate Agent error rate",
            "# TYPE kei_agent_error_rate gauge",
        ]

        for agent_id, metrics in agent_metrics.items():
            task_metrics = metrics.get("task_metrics", {})
            latency = task_metrics.get("latency", {})

            # Task-Latenz-Perzentile
            for percentile in ["p50", "p95", "p99"]:
                if percentile in latency:
                    lines.append(
                        f'kei_agent_task_duration_ms{{agent_id="{agent_id}",quantile="{percentile[1:]}"}} {latency[percentile]}'
                    )

            # Task-Counts
            success_count = task_metrics.get("success_count", 0)
            failure_count = task_metrics.get("failure_count", 0)

            lines.append(f'kei_agent_tasks_total{{agent_id="{agent_id}",status="success"}} {success_count}')
            lines.append(f'kei_agent_tasks_total{{agent_id="{agent_id}",status="failure"}} {failure_count}')

            # Error-Rate
            error_rate = task_metrics.get("success_rate", 1.0)
            lines.append(f'kei_agent_error_rate{{agent_id="{agent_id}"}} {1.0 - error_rate}')

        return lines

    def _format_system_metrics(self, system_stats: dict[str, Any]) -> list[str]:
        """Formatiert System-Metriken für Prometheus.

        Args:
            system_stats: System-Statistiken

        Returns:
            Liste von Prometheus-Zeilen
        """
        # Optimierte List-Creation mit statischen Header-Zeilen und bedingten Einträgen
        lines = [
            # System-Status Header
            "# HELP kei_system_status System component status",
            "# TYPE kei_system_status gauge",
            # System-Status Wert (bedingt)
            'kei_system_status{component="integration_metrics"} 1' if system_stats.get("is_running") else 'kei_system_status{component="integration_metrics"} 0',
            # Metriken-Collection-Statistiken Header
            "# HELP kei_metrics_collected_total Total metrics collected",
            "# TYPE kei_metrics_collected_total counter",
            # Metriken-Collection-Wert
            f"kei_metrics_collected_total {system_stats.get('total_metrics_collected', 0)}",
        ]

        return lines

    def _format_aggregated_metrics(self, aggregated_metrics: dict[str, Any]) -> list[str]:
        """Formatiert aggregierte Metriken für Prometheus.

        Args:
            aggregated_metrics: Aggregierte Metriken

        Returns:
            Liste von Prometheus-Zeilen
        """
        lines = []

        for metric_name, windows in aggregated_metrics.items():
            # Sanitize metric name für Prometheus
            prometheus_name = metric_name.replace(".", "_").replace("-", "_")

            lines.append(f"# HELP kei_{prometheus_name} {metric_name}")
            lines.append(f"# TYPE kei_{prometheus_name} gauge")

            for window, aggregations in windows.items():
                for agg_type, metric_data in aggregations.items():
                    value = metric_data.get("value", 0)
                    timestamp = metric_data.get("timestamp", time.time())

                    lines.append(
                        f'kei_{prometheus_name}{{window="{window}",aggregation="{agg_type}"}} {value} {int(timestamp * 1000)}'
                    )

        return lines


class AlertManager:
    """Alert-Manager für Metriken-basierte Alerts."""

    def __init__(self):
        """Initialisiert Alert Manager."""
        self._alert_rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, AlertEvent] = {}
        self._alert_history: list[AlertEvent] = []
        self._alert_callbacks: list[Callable[[AlertEvent], None]] = []

        # Standard-Alert-Regeln
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self) -> None:
        """Setzt Standard-Alert-Regeln auf."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric_name="agent_error_rate",
                condition="gt",
                threshold=0.1,  # 10% Error-Rate
                severity=AlertSeverity.WARNING,
                description="Agent error rate is above 10%"
            ),
            AlertRule(
                name="critical_error_rate",
                metric_name="agent_error_rate",
                condition="gt",
                threshold=0.25,  # 25% Error-Rate
                severity=AlertSeverity.CRITICAL,
                description="Agent error rate is above 25%"
            ),
            AlertRule(
                name="high_task_latency",
                metric_name="task_latency_p95",
                condition="gt",
                threshold=5000,  # 5 Sekunden
                severity=AlertSeverity.WARNING,
                description="Task P95 latency is above 5 seconds"
            ),
            AlertRule(
                name="queue_depth_high",
                metric_name="queue_depth",
                condition="gt",
                threshold=1000,
                severity=AlertSeverity.WARNING,
                description="Task queue depth is above 1000"
            )
        ]

        for rule in default_rules:
            self._alert_rules[rule.name] = rule

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Fügt Alert-Regel hinzu.

        Args:
            rule: Alert-Regel
        """
        self._alert_rules[rule.name] = rule
        logger.info(f"Alert-Regel hinzugefügt: {rule.name}")

    def remove_alert_rule(self, rule_name: str) -> bool:
        """Entfernt Alert-Regel.

        Args:
            rule_name: Name der Alert-Regel

        Returns:
            True wenn entfernt
        """
        if rule_name in self._alert_rules:
            del self._alert_rules[rule_name]
            logger.info(f"Alert-Regel entfernt: {rule_name}")
            return True
        return False

    def add_alert_callback(self, callback: Callable[[AlertEvent], None]) -> None:
        """Fügt Alert-Callback hinzu.

        Args:
            callback: Callback-Funktion
        """
        self._alert_callbacks.append(callback)

    @trace_function("alert_manager.evaluate_alerts")
    async def evaluate_alerts(self) -> list[AlertEvent]:
        """Evaluiert alle Alert-Regeln.

        Returns:
            Liste neuer Alert-Events
        """
        new_alerts = []

        # Hole aktuelle Metriken
        agent_metrics = get_all_agent_metrics()
        aggregated_metrics = await metrics_aggregator.get_aggregated_metrics()

        for rule_name, rule in self._alert_rules.items():
            if not rule.enabled:
                continue

            try:
                current_value = await self._get_metric_value(rule.metric_name, agent_metrics, aggregated_metrics)

                if current_value is not None and self._evaluate_condition(current_value, rule):
                    # Alert ausgelöst
                    alert_event = AlertEvent(
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        message=f"{rule.description} (current: {current_value}, threshold: {rule.threshold})"
                    )

                    # Prüfe, ob Alert bereits aktiv
                    if rule_name not in self._active_alerts:
                        self._active_alerts[rule_name] = alert_event
                        self._alert_history.append(alert_event)
                        new_alerts.append(alert_event)

                        # Trigger Callbacks
                        for callback in self._alert_callbacks:
                            try:
                                callback(alert_event)
                            except Exception as e:
                                logger.exception(f"Alert-Callback-Fehler: {e}")

                # Alert resolved
                elif rule_name in self._active_alerts:
                    del self._active_alerts[rule_name]
                    logger.info(f"Alert resolved: {rule_name}")

            except Exception as e:
                logger.exception(f"Alert-Evaluation-Fehler für {rule_name}: {e}")

        return new_alerts

    async def _get_metric_value(
        self,
        metric_name: str,
        agent_metrics: dict[str, Any],
        aggregated_metrics: dict[str, Any]
    ) -> float | None:
        """Holt Metrik-Wert für Alert-Evaluation.

        Args:
            metric_name: Name der Metrik
            agent_metrics: Agent-Metriken
            aggregated_metrics: Aggregierte Metriken

        Returns:
            Metrik-Wert oder None
        """
        # Spezielle Metrik-Mappings
        if metric_name == "agent_error_rate":
            # Berechne durchschnittliche Error-Rate über alle Agenten
            error_rates = []
            for metrics in agent_metrics.values():
                task_metrics = metrics.get("task_metrics", {})
                success_rate = task_metrics.get("success_rate", 1.0)
                error_rates.append(1.0 - success_rate)

            return sum(error_rates) / len(error_rates) if error_rates else 0.0

        if metric_name == "task_latency_p95":
            # Berechne durchschnittliche P95-Latenz über alle Agenten
            p95_values = []
            for metrics in agent_metrics.values():
                task_metrics = metrics.get("task_metrics", {})
                latency = task_metrics.get("latency", {})
                if "p95" in latency:
                    p95_values.append(latency["p95"])

            return sum(p95_values) / len(p95_values) if p95_values else 0.0

        if metric_name == "queue_depth":
            # Hole Queue-Depth aus aggregierten Metriken
            queue_metrics = aggregated_metrics.get("task_management.queue_depth", {})
            latest_window = queue_metrics.get("1m", {})
            avg_metric = latest_window.get("avg", {})
            return avg_metric.get("value", 0.0)

        # Fallback: Suche in aggregierten Metriken
        metric_data = aggregated_metrics.get(metric_name, {})
        if metric_data:
            # Verwende neueste 1-Minuten-Durchschnitt
            latest_window = metric_data.get("1m", {})
            avg_metric = latest_window.get("avg", {})
            return avg_metric.get("value")

        return None

    def _evaluate_condition(self, current_value: float, rule: AlertRule) -> bool:
        """Evaluiert Alert-Bedingung.

        Args:
            current_value: Aktueller Wert
            rule: Alert-Regel

        Returns:
            True wenn Bedingung erfüllt
        """
        if rule.condition == "gt":
            return current_value > rule.threshold
        if rule.condition == "lt":
            return current_value < rule.threshold
        if rule.condition == "eq":
            return abs(current_value - rule.threshold) < 0.001
        if rule.condition == "ne":
            return abs(current_value - rule.threshold) >= 0.001
        return False

    def get_active_alerts(self) -> list[AlertEvent]:
        """Gibt aktive Alerts zurück.

        Returns:
            Liste aktiver Alert-Events
        """
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[AlertEvent]:
        """Gibt Alert-Historie zurück.

        Args:
            limit: Maximale Anzahl Alerts

        Returns:
            Liste von Alert-Events
        """
        return self._alert_history[-limit:]

    def get_alert_rules(self) -> list[AlertRule]:
        """Gibt alle Alert-Regeln zurück.

        Returns:
            Liste von Alert-Regeln
        """
        return list(self._alert_rules.values())


class MetricsExportAPI:
    """Metriken-Export-API."""

    def __init__(self, app: FastAPI):
        """Initialisiert Metrics Export API.

        Args:
            app: FastAPI-Anwendung
        """
        self.app = app
        self.prometheus_exporter = PrometheusExporter()
        self.alert_manager = AlertManager()

        # Registriere API-Endpoints
        self._register_endpoints()

        # Starte Background-Tasks
        self._background_task: asyncio.Task | None = None

    def _register_endpoints(self) -> None:
        """Registriert API-Endpoints."""

        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics():
            """Prometheus-Metriken-Endpoint."""
            return await self.prometheus_exporter.export_metrics()

        @self.app.post("/api/v1/metrics/query", response_model=MetricsResponse)
        async def query_metrics(request: MetricsQueryRequest):
            """Metriken-Abfrage-Endpoint."""
            start_time = time.time()

            try:
                # Sammle Metriken basierend auf Request
                metrics_data = {}

                # Agent-Metriken
                if not request.agent_ids or "agents" in (request.metric_names or []):
                    agent_metrics = get_all_agent_metrics()
                    if request.agent_ids:
                        agent_metrics = {
                            agent_id: metrics for agent_id, metrics in agent_metrics.items()
                            if agent_id in request.agent_ids
                        }
                    metrics_data["agents"] = agent_metrics

                # Aggregierte Metriken
                if request.aggregation_window and request.aggregation_types:
                    aggregated = await metrics_aggregator.get_aggregated_metrics(
                        window=request.aggregation_window,
                        aggregation_type=request.aggregation_types[0] if request.aggregation_types else None
                    )
                    metrics_data["aggregated"] = aggregated

                # System-Metriken
                if not request.metric_names or "system" in request.metric_names:
                    system_stats = system_integration_metrics.get_integration_statistics()
                    metrics_data["system"] = system_stats

                query_time_ms = (time.time() - start_time) * 1000

                return MetricsResponse(
                    metrics=metrics_data,
                    metadata={
                        "query_time": datetime.now(UTC).isoformat(),
                        "filters_applied": {
                            "metric_names": request.metric_names,
                            "agent_ids": request.agent_ids,
                            "start_time": request.start_time.isoformat() if request.start_time and isinstance(request.start_time, datetime) else None,
                            "end_time": request.end_time.isoformat() if request.end_time and isinstance(request.end_time, datetime) else None
                        }
                    },
                    query_time_ms=query_time_ms,
                    total_results=len(metrics_data)
                )

            except Exception as e:
                logger.exception(f"Metrics-Query-Fehler: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/metrics/agents/{agent_id}")
        async def get_agent_metrics(agent_id: str):
            """Agent-spezifische Metriken."""
            try:
                agent_collector = get_agent_metrics_collector(agent_id)
                return agent_collector.get_comprehensive_metrics()
            except Exception as e:
                logger.exception(f"Agent-Metrics-Fehler: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/alerts")
        async def get_alerts():
            """Aktive Alerts."""
            try:
                active_alerts = self.alert_manager.get_active_alerts()
                return {
                    "active_alerts": [alert.to_dict() for alert in active_alerts],
                    "total_active": len(active_alerts)
                }
            except Exception as e:
                logger.exception(f"Alerts-Abfrage-Fehler: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/alerts/rules")
        async def create_alert_rule(request: AlertRuleRequest):
            """Erstellt Alert-Regel."""
            try:
                rule = AlertRule(
                    name=request.name,
                    metric_name=request.metric_name,
                    condition=request.condition,
                    threshold=request.threshold,
                    severity=request.severity,
                    description=request.description,
                    enabled=request.enabled
                )

                self.alert_manager.add_alert_rule(rule)

                return {"message": f"Alert-Regel '{request.name}' erstellt"}
            except Exception as e:
                logger.exception(f"Alert-Regel-Erstellung-Fehler: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/dashboard/config")
        async def get_dashboard_config():
            """Dashboard-Konfiguration."""
            return {
                "dashboards": [
                    {
                        "name": "Agent Overview",
                        "panels": [
                            {
                                "title": "Active Agents",
                                "type": "stat",
                                "metric": "agents.total_active"
                            },
                            {
                                "title": "Task Success Rate",
                                "type": "gauge",
                                "metric": "agents.success_rate"
                            },
                            {
                                "title": "Task Latency P95",
                                "type": "graph",
                                "metric": "agents.task_latency_p95"
                            }
                        ]
                    }
                ]
            }

    async def start_background_tasks(self) -> None:
        """Startet Background-Tasks."""
        self._background_task = asyncio.create_task(self._alert_evaluation_loop())

    async def stop_background_tasks(self) -> None:
        """Stoppt Background-Tasks."""
        if self._background_task:
            self._background_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._background_task

    async def _alert_evaluation_loop(self) -> None:
        """Background-Loop für Alert-Evaluation."""
        while True:
            try:
                await self.alert_manager.evaluate_alerts()
                await asyncio.sleep(60)  # Evaluiere alle 60 Sekunden
            except Exception as e:
                logger.exception(f"Alert-Evaluation-Loop-Fehler: {e}")
                await asyncio.sleep(60)


# Globale Metriken-Export-API
metrics_export_api: MetricsExportAPI | None = None


def setup_metrics_export_api(app: FastAPI) -> MetricsExportAPI:
    """Setzt Metriken-Export-API auf.

    Args:
        app: FastAPI-Anwendung

    Returns:
        Metriken-Export-API
    """
    global metrics_export_api
    metrics_export_api = MetricsExportAPI(app)
    return metrics_export_api
