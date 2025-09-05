# backend/services/enhanced_performance_analytics/event_driven_analytics_engine.py
"""Event-driven Performance Analytics Engine.

Implementiert Enterprise-Grade Event-driven Performance Analytics mit
Real-time Data-Processing und asynchroner Analytics-Pipeline.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_real_time_monitoring import EnhancedRealTimeMonitoringEngine

from .data_models import (
    AdvancedPerformanceMetrics,
    AnalyticsConfiguration,
    AnalyticsScope,
    EventType,
    MetricDimension,
    PerformanceDataPoint,
    PerformanceEvent,
)

logger = get_logger(__name__)


class EventDrivenAnalyticsEngine:
    """Event-driven Performance Analytics Engine für Enterprise-Grade Analytics."""

    def __init__(
        self,
        monitoring_engine: EnhancedRealTimeMonitoringEngine | None = None,
        configuration: AnalyticsConfiguration | None = None
    ):
        """Initialisiert Event-driven Analytics Engine.

        Args:
            monitoring_engine: Real-time Monitoring Engine
            configuration: Analytics-Konfiguration
        """
        self.monitoring_engine = monitoring_engine
        self.configuration = configuration or AnalyticsConfiguration()

        # Event-Storage
        self._event_queue: deque = deque(maxlen=10000)
        self._processed_events: dict[str, PerformanceEvent] = {}
        self._event_handlers: dict[EventType, list[callable]] = defaultdict(list)

        # Data-Storage
        self._performance_data_points: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._advanced_metrics: dict[str, AdvancedPerformanceMetrics] = {}
        self._aggregated_data: dict[str, dict[str, Any]] = defaultdict(dict)

        # Real-time Processing
        self._real_time_processors: list[asyncio.Task] = []
        self._is_running = False
        self._processing_stats = {
            "total_events_processed": 0,
            "total_data_points_processed": 0,
            "avg_event_processing_time_ms": 0.0,
            "avg_data_processing_time_ms": 0.0,
            "processing_error_rate": 0.0
        }

        # Event-Subscriptions
        self._event_subscriptions: dict[str, set[str]] = defaultdict(set)  # event_type -> subscriber_ids
        self._subscribers: dict[str, dict[str, Any]] = {}  # subscriber_id -> subscription_info

        # Performance-Tracking
        self._analytics_performance_stats = {
            "total_analytics_operations": 0,
            "avg_analytics_processing_time_ms": 0.0,
            "analytics_throughput_ops": 0.0,
            "analytics_error_rate": 0.0
        }

        logger.info("Event-driven Analytics Engine initialisiert")

    async def start(self) -> None:
        """Startet Event-driven Analytics Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Real-time Processors
        self._real_time_processors = [
            asyncio.create_task(self._event_processing_loop()),
            asyncio.create_task(self._data_aggregation_loop()),
            asyncio.create_task(self._real_time_analytics_loop()),
            asyncio.create_task(self._performance_monitoring_loop())
        ]

        # Registriere Standard-Event-Handler
        await self._register_default_event_handlers()

        logger.info("Event-driven Analytics Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Event-driven Analytics Engine."""
        self._is_running = False

        # Stoppe Real-time Processors
        for processor in self._real_time_processors:
            processor.cancel()

        await asyncio.gather(*self._real_time_processors, return_exceptions=True)
        self._real_time_processors.clear()

        logger.info("Event-driven Analytics Engine gestoppt")

    async def publish_event(
        self,
        event: PerformanceEvent
    ) -> None:
        """Publiziert Performance-Event.

        Args:
            event: Performance-Event
        """
        try:
            # Füge Event zur Queue hinzu
            self._event_queue.append(event)

            # Immediate Processing für High-Priority Events
            if event.requires_immediate_processing or event.priority <= 2:
                await self._process_event_immediately(event)

            logger.debug({
                "event": "performance_event_published",
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "priority": event.priority,
                "immediate_processing": event.requires_immediate_processing
            })

        except Exception as e:
            logger.error(f"Event publishing fehlgeschlagen: {e}")

    async def subscribe_to_events(
        self,
        subscriber_id: str,
        event_types: list[EventType],
        callback: callable,
        filter_criteria: dict[str, Any] | None = None
    ) -> bool:
        """Abonniert Events.

        Args:
            subscriber_id: Subscriber-ID
            event_types: Event-Typen
            callback: Callback-Funktion
            filter_criteria: Filter-Kriterien

        Returns:
            Erfolg-Status
        """
        try:
            # Registriere Subscriber
            self._subscribers[subscriber_id] = {
                "event_types": event_types,
                "callback": callback,
                "filter_criteria": filter_criteria or {},
                "subscribed_at": datetime.utcnow(),
                "events_received": 0
            }

            # Füge zu Event-Subscriptions hinzu
            for event_type in event_types:
                self._event_subscriptions[event_type.value].add(subscriber_id)

            logger.debug({
                "event": "event_subscription_created",
                "subscriber_id": subscriber_id,
                "event_types": [et.value for et in event_types]
            })

            return True

        except Exception as e:
            logger.error(f"Event subscription fehlgeschlagen: {e}")
            return False

    async def collect_performance_data_point(
        self,
        data_point: PerformanceDataPoint
    ) -> None:
        """Sammelt Performance-Datenpunkt.

        Args:
            data_point: Performance-Datenpunkt
        """
        start_time = time.time()

        try:
            # Speichere Datenpunkt
            data_key = f"{data_point.scope.value}:{data_point.scope_id}:{data_point.metric_name}"
            self._performance_data_points[data_key].append(data_point)

            # Erstelle Performance-Event
            event = PerformanceEvent(
                event_id=f"data_point_{data_point.data_point_id}",
                event_type=EventType.PERFORMANCE_METRIC,
                event_name=f"performance_metric_{data_point.metric_name}",
                source_service="analytics_engine",
                source_scope=data_point.scope,
                source_scope_id=data_point.scope_id,
                payload={
                    "data_point": {
                        "metric_name": data_point.metric_name,
                        "value": data_point.value,
                        "unit": data_point.unit,
                        "timestamp": data_point.timestamp.isoformat(),
                        "dimensions": {dim.value: val for dim, val in data_point.dimensions.items()},
                        "labels": data_point.labels,
                        "metadata": data_point.metadata
                    }
                },
                user_id=data_point.user_id,
                tenant_id=data_point.tenant_id
            )

            # Publiziere Event
            await self.publish_event(event)

            # Update Performance-Stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_analytics_performance_stats("data_collection", processing_time_ms)

            logger.debug({
                "event": "performance_data_point_collected",
                "metric_name": data_point.metric_name,
                "scope": data_point.scope.value,
                "scope_id": data_point.scope_id,
                "value": data_point.value,
                "processing_time_ms": processing_time_ms
            })

        except Exception as e:
            logger.error(f"Performance data point collection fehlgeschlagen: {e}")

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
        start_time = time.time()

        try:
            import statistics
            import uuid

            metrics_id = str(uuid.uuid4())

            # Sammle relevante Datenpunkte
            relevant_data_points = []
            for metric_name in metric_names:
                data_key = f"{scope.value}:{scope_id}:{metric_name}"
                data_points = self._performance_data_points.get(data_key, deque())

                for dp in data_points:
                    if period_start <= dp.timestamp <= period_end:
                        relevant_data_points.append(dp)

            if not relevant_data_points:
                # Leere Metriken zurückgeben
                return AdvancedPerformanceMetrics(
                    metrics_id=metrics_id,
                    scope=scope,
                    scope_id=scope_id,
                    period_start=period_start,
                    period_end=period_end
                )

            # Extrahiere Response-Time-Werte
            response_times = [
                float(dp.value) for dp in relevant_data_points
                if dp.metric_name in ["response_time", "execution_time", "processing_time"]
                and isinstance(dp.value, (int, float))
            ]

            # Extrahiere Request-Counts
            request_counts = [
                int(dp.value) for dp in relevant_data_points
                if dp.metric_name in ["request_count", "total_requests"]
                and isinstance(dp.value, (int, float))
            ]

            # Extrahiere Error-Counts
            error_counts = [
                int(dp.value) for dp in relevant_data_points
                if dp.metric_name in ["error_count", "failed_requests"]
                and isinstance(dp.value, (int, float))
            ]

            # Berechne Advanced Metrics
            advanced_metrics = AdvancedPerformanceMetrics(
                metrics_id=metrics_id,
                scope=scope,
                scope_id=scope_id,
                period_start=period_start,
                period_end=period_end,
                sample_count=len(relevant_data_points)
            )

            # Response-Time-Metriken
            if response_times:
                response_times.sort()
                advanced_metrics.avg_response_time_ms = statistics.mean(response_times)
                advanced_metrics.median_response_time_ms = statistics.median(response_times)
                advanced_metrics.min_response_time_ms = min(response_times)
                advanced_metrics.max_response_time_ms = max(response_times)
                advanced_metrics.std_dev_response_time_ms = statistics.stdev(response_times) if len(response_times) > 1 else 0.0

                # Percentiles
                n = len(response_times)
                advanced_metrics.p50_response_time_ms = response_times[int(0.50 * n)]
                advanced_metrics.p75_response_time_ms = response_times[int(0.75 * n)]
                advanced_metrics.p90_response_time_ms = response_times[int(0.90 * n)]
                advanced_metrics.p95_response_time_ms = response_times[int(0.95 * n)]
                advanced_metrics.p99_response_time_ms = response_times[int(0.99 * n)]
                advanced_metrics.p999_response_time_ms = response_times[int(0.999 * n)]

            # Request-Metriken
            if request_counts:
                advanced_metrics.total_requests = sum(request_counts)
                advanced_metrics.avg_throughput_rps = advanced_metrics.total_requests / ((period_end - period_start).total_seconds() or 1)
                advanced_metrics.peak_throughput_rps = max(request_counts) if request_counts else 0
                advanced_metrics.min_throughput_rps = min(request_counts) if request_counts else 0

            # Error-Metriken
            if error_counts and request_counts:
                total_errors = sum(error_counts)
                total_requests = sum(request_counts)
                advanced_metrics.failed_requests = total_errors
                advanced_metrics.successful_requests = total_requests - total_errors
                advanced_metrics.error_rate = total_errors / total_requests if total_requests > 0 else 0.0
                advanced_metrics.success_rate = 1.0 - advanced_metrics.error_rate

            # Multi-dimensional Breakdowns
            advanced_metrics.dimension_breakdowns = await self._calculate_dimension_breakdowns(
                relevant_data_points, scope, scope_id
            )

            # Speichere Advanced Metrics
            self._advanced_metrics[metrics_id] = advanced_metrics

            # Update Performance-Stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_analytics_performance_stats("advanced_metrics_generation", processing_time_ms)

            logger.debug({
                "event": "advanced_metrics_generated",
                "metrics_id": metrics_id,
                "scope": scope.value,
                "scope_id": scope_id,
                "sample_count": len(relevant_data_points),
                "processing_time_ms": processing_time_ms
            })

            return advanced_metrics

        except Exception as e:
            logger.error(f"Advanced metrics generation fehlgeschlagen: {e}")
            # Fallback zu leeren Metriken
            return AdvancedPerformanceMetrics(
                metrics_id=str(uuid.uuid4()),
                scope=scope,
                scope_id=scope_id,
                period_start=period_start,
                period_end=period_end
            )

    async def _process_event_immediately(self, event: PerformanceEvent) -> None:
        """Verarbeitet Event sofort."""
        try:
            # Finde passende Event-Handler
            handlers = self._event_handlers.get(event.event_type, [])

            # Führe Handler aus
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler fehlgeschlagen: {e}")

            # Benachrichtige Subscribers
            await self._notify_event_subscribers(event)

        except Exception as e:
            logger.error(f"Immediate event processing fehlgeschlagen: {e}")

    async def _event_processing_loop(self) -> None:
        """Background-Loop für Event-Processing."""
        while self._is_running:
            try:
                # Verarbeite Events in Batches
                batch_size = min(self.configuration.event_batch_size, len(self._event_queue))

                if batch_size > 0:
                    events_to_process = []
                    for _ in range(batch_size):
                        if self._event_queue:
                            events_to_process.append(self._event_queue.popleft())

                    # Verarbeite Event-Batch
                    await self._process_event_batch(events_to_process)

                await asyncio.sleep(self.configuration.event_processing_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Event processing loop fehlgeschlagen: {e}")
                await asyncio.sleep(1)

    async def _data_aggregation_loop(self) -> None:
        """Background-Loop für Data-Aggregation."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.aggregation_interval_seconds)

                if self._is_running:
                    await self._aggregate_performance_data()

            except Exception as e:
                logger.error(f"Data aggregation loop fehlgeschlagen: {e}")
                await asyncio.sleep(self.configuration.aggregation_interval_seconds)

    async def _real_time_analytics_loop(self) -> None:
        """Background-Loop für Real-time Analytics."""
        while self._is_running:
            try:
                await asyncio.sleep(self.configuration.data_collection_interval_seconds)

                if self._is_running:
                    await self._perform_real_time_analytics()

            except Exception as e:
                logger.error(f"Real-time analytics loop fehlgeschlagen: {e}")
                await asyncio.sleep(self.configuration.data_collection_interval_seconds)

    async def _performance_monitoring_loop(self) -> None:
        """Background-Loop für Performance-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._monitor_analytics_performance()

            except Exception as e:
                logger.error(f"Performance monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _process_event_batch(self, events: list[PerformanceEvent]) -> None:
        """Verarbeitet Event-Batch."""
        try:
            start_time = time.time()

            for event in events:
                # Speichere Event
                self._processed_events[event.event_id] = event

                # Führe Event-Handler aus
                handlers = self._event_handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Event handler fehlgeschlagen: {e}")

                # Benachrichtige Subscribers
                await self._notify_event_subscribers(event)

            # Update Performance-Stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._processing_stats["total_events_processed"] += len(events)

            # Update Average Processing Time
            total_events = self._processing_stats["total_events_processed"]
            current_avg = self._processing_stats["avg_event_processing_time_ms"]
            new_avg = ((current_avg * (total_events - len(events))) + processing_time_ms) / total_events
            self._processing_stats["avg_event_processing_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Event batch processing fehlgeschlagen: {e}")

    async def _notify_event_subscribers(self, event: PerformanceEvent) -> None:
        """Benachrichtigt Event-Subscribers."""
        try:
            subscribers = self._event_subscriptions.get(event.event_type.value, set())

            for subscriber_id in subscribers:
                subscriber_info = self._subscribers.get(subscriber_id)
                if not subscriber_info:
                    continue

                # Prüfe Filter-Kriterien
                if self._matches_filter_criteria(event, subscriber_info.get("filter_criteria", {})):
                    try:
                        callback = subscriber_info["callback"]
                        await callback(event)
                        subscriber_info["events_received"] += 1
                    except Exception as e:
                        logger.error(f"Subscriber callback fehlgeschlagen: {e}")

        except Exception as e:
            logger.error(f"Event subscriber notification fehlgeschlagen: {e}")

    async def _register_default_event_handlers(self) -> None:
        """Registriert Standard-Event-Handler."""
        try:
            # Performance-Metric-Handler
            self._event_handlers[EventType.PERFORMANCE_METRIC].append(
                self._handle_performance_metric_event
            )

            # Anomaly-Detection-Handler
            self._event_handlers[EventType.ANOMALY_DETECTED].append(
                self._handle_anomaly_detected_event
            )

            # Trend-Change-Handler
            self._event_handlers[EventType.TREND_CHANGE].append(
                self._handle_trend_change_event
            )

            # Optimization-Trigger-Handler
            self._event_handlers[EventType.OPTIMIZATION_TRIGGER].append(
                self._handle_optimization_trigger_event
            )

        except Exception as e:
            logger.error(f"Default event handlers registration fehlgeschlagen: {e}")

    async def _handle_performance_metric_event(self, event: PerformanceEvent) -> None:
        """Behandelt Performance-Metric-Event."""
        try:
            # Extrahiere Datenpunkt aus Event
            data_point_data = event.payload.get("data_point", {})

            # Verarbeite Datenpunkt für Real-time Analytics
            await self._process_performance_data_point(data_point_data, event)

        except Exception as e:
            logger.error(f"Performance metric event handling fehlgeschlagen: {e}")

    async def _handle_anomaly_detected_event(self, event: PerformanceEvent) -> None:
        """Behandelt Anomaly-Detection-Event."""
        try:
            # Log Anomaly
            logger.warning({
                "event": "anomaly_detected_via_event",
                "event_id": event.event_id,
                "source_scope": event.source_scope.value,
                "source_scope_id": event.source_scope_id,
                "payload": event.payload
            })

        except Exception as e:
            logger.error(f"Anomaly detected event handling fehlgeschlagen: {e}")

    async def _handle_trend_change_event(self, event: PerformanceEvent) -> None:
        """Behandelt Trend-Change-Event."""
        try:
            # Log Trend Change
            logger.info({
                "event": "trend_change_detected_via_event",
                "event_id": event.event_id,
                "source_scope": event.source_scope.value,
                "source_scope_id": event.source_scope_id,
                "payload": event.payload
            })

        except Exception as e:
            logger.error(f"Trend change event handling fehlgeschlagen: {e}")

    async def _handle_optimization_trigger_event(self, event: PerformanceEvent) -> None:
        """Behandelt Optimization-Trigger-Event."""
        try:
            # Log Optimization Trigger
            logger.info({
                "event": "optimization_trigger_via_event",
                "event_id": event.event_id,
                "source_scope": event.source_scope.value,
                "source_scope_id": event.source_scope_id,
                "payload": event.payload
            })

        except Exception as e:
            logger.error(f"Optimization trigger event handling fehlgeschlagen: {e}")

    async def _process_performance_data_point(
        self,
        data_point_data: dict[str, Any],
        event: PerformanceEvent
    ) -> None:
        """Verarbeitet Performance-Datenpunkt."""
        try:
            # Extrahiere Metrik-Informationen
            metric_name = data_point_data.get("metric_name")
            value = data_point_data.get("value")

            if not metric_name or value is None:
                return

            # Update Real-time Aggregation
            aggregation_key = f"{event.source_scope.value}:{event.source_scope_id}:{metric_name}"

            if aggregation_key not in self._aggregated_data:
                self._aggregated_data[aggregation_key] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "last_updated": datetime.utcnow()
                }

            aggregation = self._aggregated_data[aggregation_key]
            aggregation["count"] += 1
            aggregation["sum"] += float(value)
            aggregation["min"] = min(aggregation["min"], float(value))
            aggregation["max"] = max(aggregation["max"], float(value))
            aggregation["last_updated"] = datetime.utcnow()

        except Exception as e:
            logger.error(f"Performance data point processing fehlgeschlagen: {e}")

    async def _aggregate_performance_data(self) -> None:
        """Aggregiert Performance-Daten."""
        try:
            # Aggregiere Daten für alle Scopes
            for aggregation_key, aggregation in self._aggregated_data.items():
                if aggregation["count"] > 0:
                    # Berechne Durchschnitt (für zukünftige Verwendung)
                    # avg_value = aggregation["sum"] / aggregation["count"]  # TODO: Verwende für erweiterte Aggregation - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/112

                    # Reset Aggregation
                    aggregation["count"] = 0
                    aggregation["sum"] = 0.0
                    aggregation["min"] = float("inf")
                    aggregation["max"] = float("-inf")

        except Exception as e:
            logger.error(f"Performance data aggregation fehlgeschlagen: {e}")

    async def _perform_real_time_analytics(self) -> None:
        """Führt Real-time Analytics aus."""
        try:
            # Sammle aktuelle Performance-Daten
            current_time = datetime.utcnow()

            # Analysiere Performance-Trends
            await self._analyze_performance_trends(current_time)

        except Exception as e:
            logger.error(f"Real-time analytics fehlgeschlagen: {e}")

    async def _analyze_performance_trends(self, current_time: datetime) -> None:
        """Analysiert Performance-Trends."""
        try:
            # Analysiere Trends für alle aktiven Metriken
            for data_key, data_points in self._performance_data_points.items():
                if len(data_points) >= 10:  # Mindestens 10 Datenpunkte für Trend-Analysis
                    # Einfache Trend-Detection
                    recent_points = list(data_points)[-10:]
                    values = [float(dp.value) for dp in recent_points if isinstance(dp.value, (int, float))]

                    if len(values) >= 5:
                        # Berechne Trend-Slope
                        x_values = list(range(len(values)))
                        slope = self._calculate_trend_slope(x_values, values)

                        # Prüfe auf signifikante Trends
                        if abs(slope) > 0.1:  # Signifikanter Trend
                            await self._trigger_trend_change_event(data_key, slope, current_time)

        except Exception as e:
            logger.error(f"Performance trends analysis fehlgeschlagen: {e}")

    async def _trigger_trend_change_event(
        self,
        data_key: str,
        slope: float,
        timestamp: datetime
    ) -> None:
        """Triggert Trend-Change-Event."""
        try:
            import uuid

            scope_str, scope_id, metric_name = data_key.split(":", 2)
            scope = AnalyticsScope(scope_str)

            event = PerformanceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.TREND_CHANGE,
                event_name=f"trend_change_{metric_name}",
                source_service="analytics_engine",
                source_scope=scope,
                source_scope_id=scope_id,
                payload={
                    "metric_name": metric_name,
                    "trend_slope": slope,
                    "trend_direction": "increasing" if slope > 0 else "decreasing",
                    "timestamp": timestamp.isoformat()
                }
            )

            await self.publish_event(event)

        except Exception as e:
            logger.error(f"Trend change event triggering fehlgeschlagen: {e}")

    async def _monitor_analytics_performance(self) -> None:
        """Monitort Analytics-Performance."""
        try:
            # Berechne Analytics-Throughput
            # current_time = time.time()  # TODO: Verwende für zeitbasierte Berechnungen - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/112 - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/112
            time_window = 60  # 1 Minute

            operations_in_window = self._analytics_performance_stats["total_analytics_operations"]
            throughput = operations_in_window / time_window
            self._analytics_performance_stats["analytics_throughput_ops"] = throughput

        except Exception as e:
            logger.error(f"Analytics performance monitoring fehlgeschlagen: {e}")

    async def _calculate_dimension_breakdowns(
        self,
        data_points: list[PerformanceDataPoint],
        _scope: AnalyticsScope,
        _scope_id: str
    ) -> dict[MetricDimension, dict[str, float]]:
        """Berechnet Multi-dimensional Breakdowns."""
        try:
            breakdowns = {}

            # Service-Dimension
            service_breakdown = defaultdict(list)
            for dp in data_points:
                if dp.service_name and isinstance(dp.value, (int, float)):
                    service_breakdown[dp.service_name].append(float(dp.value))

            if service_breakdown:
                breakdowns[MetricDimension.SERVICE] = {
                    service: sum(values) / len(values)
                    for service, values in service_breakdown.items()
                }

            # User-Dimension
            user_breakdown = defaultdict(list)
            for dp in data_points:
                if dp.user_id and isinstance(dp.value, (int, float)):
                    user_breakdown[dp.user_id].append(float(dp.value))

            if user_breakdown:
                breakdowns[MetricDimension.USER] = {
                    user: sum(values) / len(values)
                    for user, values in user_breakdown.items()
                }

            return breakdowns

        except Exception as e:
            logger.error(f"Dimension breakdowns calculation fehlgeschlagen: {e}")
            return {}

    def _calculate_trend_slope(self, x_values: list[int], y_values: list[float]) -> float:
        """Berechnet Trend-Slope."""
        try:
            n = len(x_values)
            if n < 2:
                return 0.0

            # Einfache lineare Regression
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
            sum_x2 = sum(x * x for x in x_values)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope

        except Exception as e:
            logger.error(f"Trend slope calculation fehlgeschlagen: {e}")
            return 0.0

    def _matches_filter_criteria(
        self,
        event: PerformanceEvent,
        filter_criteria: dict[str, Any]
    ) -> bool:
        """Prüft ob Event Filter-Kriterien entspricht."""
        try:
            for key, expected_value in filter_criteria.items():
                if (key == "source_scope" and event.source_scope.value != expected_value) or (key == "source_scope_id" and event.source_scope_id != expected_value):
                    return False
                if key == "priority" and event.priority > expected_value:
                    return False

            return True

        except Exception as e:
            logger.error(f"Filter criteria matching fehlgeschlagen: {e}")
            return False

    def _update_analytics_performance_stats(self, _operation: str, duration_ms: float) -> None:
        """Aktualisiert Analytics-Performance-Statistiken."""
        try:
            self._analytics_performance_stats["total_analytics_operations"] += 1

            current_avg = self._analytics_performance_stats["avg_analytics_processing_time_ms"]
            total_count = self._analytics_performance_stats["total_analytics_operations"]
            new_avg = ((current_avg * (total_count - 1)) + duration_ms) / total_count
            self._analytics_performance_stats["avg_analytics_processing_time_ms"] = new_avg

        except Exception as e:
            logger.error(f"Analytics performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = {
                "event_processing": self._processing_stats.copy(),
                "analytics_performance": self._analytics_performance_stats.copy(),
                "data_storage": {
                    "total_data_points": sum(len(deque_obj) for deque_obj in self._performance_data_points.values()),
                    "total_advanced_metrics": len(self._advanced_metrics),
                    "total_processed_events": len(self._processed_events),
                    "active_subscribers": len(self._subscribers)
                },
                "configuration": {
                    "analytics_enabled": self.configuration.analytics_enabled,
                    "real_time_analytics_enabled": self.configuration.real_time_analytics_enabled,
                    "event_processing_enabled": self.configuration.event_processing_enabled,
                    "analytics_processing_timeout_ms": self.configuration.analytics_processing_timeout_ms
                }
            }

            # SLA-Compliance
            avg_processing_time = stats["analytics_performance"]["avg_analytics_processing_time_ms"]
            stats["meets_analytics_sla"] = avg_processing_time < self.configuration.analytics_processing_timeout_ms

            return stats

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
