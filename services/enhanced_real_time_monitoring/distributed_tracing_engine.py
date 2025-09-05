# backend/services/enhanced_real_time_monitoring/distributed_tracing_engine.py
"""Distributed Tracing Engine für Observability.

Implementiert Enterprise-Grade Distributed Tracing mit Span-Management,
Trace-Correlation und Performance-Tracking für komplexe Service-Landschaften.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger

from .data_models import DistributedTrace, TraceSpan, TraceStatus

logger = get_logger(__name__)


class DistributedTracingEngine:
    """Distributed Tracing Engine für Enterprise-Grade Observability."""

    def __init__(self):
        """Initialisiert Distributed Tracing Engine."""
        # Trace-Storage
        self._active_traces: dict[str, DistributedTrace] = {}
        self._completed_traces: dict[str, DistributedTrace] = {}
        self._trace_index: dict[str, set[str]] = defaultdict(set)  # service_name -> trace_ids

        # Span-Storage
        self._active_spans: dict[str, TraceSpan] = {}
        self._span_relationships: dict[str, list[str]] = defaultdict(list)  # parent -> children

        # Tracing-Konfiguration
        self.sampling_rate = 1.0  # 100% Sampling
        self.trace_retention_hours = 24
        self.span_timeout_seconds = 300
        self.enable_performance_tracking = True

        # Performance-Tracking
        self._tracing_performance_stats = {
            "total_traces_created": 0,
            "total_spans_created": 0,
            "avg_trace_duration_ms": 0.0,
            "avg_span_duration_ms": 0.0,
            "trace_completion_rate": 0.0,
            "span_completion_rate": 0.0
        }

        # Background-Tasks
        self._tracing_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Correlation-Tracking
        self._correlation_map: dict[str, dict[str, Any]] = {}  # trace_id -> correlation_data

        logger.info("Distributed Tracing Engine initialisiert")

    async def start(self) -> None:
        """Startet Distributed Tracing Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._tracing_tasks = [
            asyncio.create_task(self._trace_monitoring_loop()),
            asyncio.create_task(self._span_timeout_loop()),
            asyncio.create_task(self._trace_cleanup_loop()),
            asyncio.create_task(self._correlation_analysis_loop())
        ]

        logger.info("Distributed Tracing Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Distributed Tracing Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._tracing_tasks:
            task.cancel()

        await asyncio.gather(*self._tracing_tasks, return_exceptions=True)
        self._tracing_tasks.clear()

        logger.info("Distributed Tracing Engine gestoppt")

    async def create_trace(
        self,
        trace_name: str,
        service_name: str,
        operation_name: str,
        orchestration_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Erstellt neuen Distributed Trace.

        Args:
            trace_name: Name des Traces
            service_name: Service-Name
            operation_name: Operation-Name
            orchestration_id: Orchestration-ID
            user_id: User-ID
            tenant_id: Tenant-ID
            metadata: Metadata

        Returns:
            Trace-ID
        """
        try:
            import uuid

            # Sampling-Check
            if not self._should_sample():
                return ""

            trace_id = str(uuid.uuid4())
            root_span_id = str(uuid.uuid4())

            # Erstelle Root-Span
            root_span = TraceSpan(
                span_id=root_span_id,
                trace_id=trace_id,
                operation_name=operation_name,
                service_name=service_name,
                component="root",
                orchestration_id=orchestration_id,
                user_id=user_id,
                metadata=metadata or {}
            )

            # Erstelle Trace
            trace = DistributedTrace(
                trace_id=trace_id,
                trace_name=trace_name,
                root_span_id=root_span_id,
                orchestration_id=orchestration_id,
                user_id=user_id,
                tenant_id=tenant_id,
                metadata=metadata or {}
            )

            trace.spans[root_span_id] = root_span
            trace.service_names.add(service_name)
            trace.total_spans = 1

            # Speichere Trace und Span
            self._active_traces[trace_id] = trace
            self._active_spans[root_span_id] = root_span
            self._trace_index[service_name].add(trace_id)

            # Correlation-Tracking
            self._correlation_map[trace_id] = {
                "orchestration_id": orchestration_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "service_names": [service_name],
                "created_at": datetime.utcnow().isoformat()
            }

            # Performance-Tracking
            self._tracing_performance_stats["total_traces_created"] += 1

            logger.debug({
                "event": "trace_created",
                "trace_id": trace_id,
                "trace_name": trace_name,
                "service_name": service_name,
                "root_span_id": root_span_id
            })

            return trace_id

        except Exception as e:
            logger.error(f"Trace creation fehlgeschlagen: {e}")
            return ""

    async def create_span(
        self,
        trace_id: str,
        operation_name: str,
        service_name: str,
        component: str,
        parent_span_id: str | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Erstellt neuen Span.

        Args:
            trace_id: Trace-ID
            operation_name: Operation-Name
            service_name: Service-Name
            component: Component-Name
            parent_span_id: Parent-Span-ID
            tags: Span-Tags
            metadata: Metadata

        Returns:
            Span-ID
        """
        try:
            import uuid

            trace = self._active_traces.get(trace_id)
            if not trace:
                logger.warning(f"Trace {trace_id} nicht gefunden für Span creation")
                return ""

            span_id = str(uuid.uuid4())

            # Erstelle Span
            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                service_name=service_name,
                component=component,
                tags=tags or {},
                orchestration_id=trace.orchestration_id,
                user_id=trace.user_id,
                metadata=metadata or {}
            )

            # Füge Span zu Trace hinzu
            trace.spans[span_id] = span
            trace.service_names.add(service_name)
            trace.total_spans += 1

            # Speichere Span
            self._active_spans[span_id] = span

            # Update Span-Hierarchie
            if parent_span_id:
                self._span_relationships[parent_span_id].append(span_id)
                if parent_span_id not in trace.span_hierarchy:
                    trace.span_hierarchy[parent_span_id] = []
                trace.span_hierarchy[parent_span_id].append(span_id)

            # Update Correlation-Map
            if trace_id in self._correlation_map:
                correlation_data = self._correlation_map[trace_id]
                if service_name not in correlation_data["service_names"]:
                    correlation_data["service_names"].append(service_name)

            # Performance-Tracking
            self._tracing_performance_stats["total_spans_created"] += 1

            logger.debug({
                "event": "span_created",
                "trace_id": trace_id,
                "span_id": span_id,
                "operation_name": operation_name,
                "service_name": service_name,
                "parent_span_id": parent_span_id
            })

            return span_id

        except Exception as e:
            logger.error(f"Span creation fehlgeschlagen: {e}")
            return ""

    async def finish_span(
        self,
        span_id: str,
        status_code: int | None = None,
        error: str | None = None,
        logs: list[dict[str, Any]] | None = None
    ) -> None:
        """Beendet Span.

        Args:
            span_id: Span-ID
            status_code: Status-Code
            error: Fehler-Message
            logs: Span-Logs
        """
        try:
            span = self._active_spans.get(span_id)
            if not span:
                logger.warning(f"Span {span_id} nicht gefunden für finish")
                return

            # Update Span
            span.end_time = datetime.utcnow()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            span.status_code = status_code
            span.error = error

            if logs:
                span.logs.extend(logs)

            # Update Status
            if error:
                span.status = TraceStatus.ERROR
            else:
                span.status = TraceStatus.COMPLETED

            # Update Trace
            trace = self._active_traces.get(span.trace_id)
            if trace:
                trace.completed_spans += 1

                if error:
                    trace.error_spans += 1

                # Prüfe ob Trace komplett ist
                if trace.completed_spans == trace.total_spans:
                    await self._complete_trace(trace.trace_id)

            # Entferne aus aktiven Spans
            del self._active_spans[span_id]

            # Performance-Tracking
            self._update_span_performance_stats(span.duration_ms)

            logger.debug({
                "event": "span_finished",
                "trace_id": span.trace_id,
                "span_id": span_id,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "error": error is not None
            })

        except Exception as e:
            logger.error(f"Span finish fehlgeschlagen: {e}")

    async def add_span_log(
        self,
        span_id: str,
        level: str,
        message: str,
        fields: dict[str, Any] | None = None
    ) -> None:
        """Fügt Log zu Span hinzu.

        Args:
            span_id: Span-ID
            level: Log-Level
            message: Log-Message
            fields: Log-Fields
        """
        try:
            span = self._active_spans.get(span_id)
            if not span:
                # Versuche in Trace zu finden
                for trace in self._active_traces.values():
                    if span_id in trace.spans:
                        span = trace.spans[span_id]
                        break

            if span:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": level,
                    "message": message,
                    "fields": fields or {}
                }
                span.logs.append(log_entry)

                logger.debug({
                    "event": "span_log_added",
                    "span_id": span_id,
                    "level": level,
                    "message": message
                })

        except Exception as e:
            logger.error(f"Span log addition fehlgeschlagen: {e}")

    async def add_span_tags(
        self,
        span_id: str,
        tags: dict[str, str]
    ) -> None:
        """Fügt Tags zu Span hinzu.

        Args:
            span_id: Span-ID
            tags: Tags
        """
        try:
            span = self._active_spans.get(span_id)
            if not span:
                # Versuche in Trace zu finden
                for trace in self._active_traces.values():
                    if span_id in trace.spans:
                        span = trace.spans[span_id]
                        break

            if span:
                span.tags.update(tags)

                logger.debug({
                    "event": "span_tags_added",
                    "span_id": span_id,
                    "tags": tags
                })

        except Exception as e:
            logger.error(f"Span tags addition fehlgeschlagen: {e}")

    async def get_trace(self, trace_id: str) -> DistributedTrace | None:
        """Holt Trace.

        Args:
            trace_id: Trace-ID

        Returns:
            Trace oder None
        """
        try:
            trace = self._active_traces.get(trace_id) or self._completed_traces.get(trace_id)
            return trace

        except Exception as e:
            logger.error(f"Trace retrieval fehlgeschlagen: {e}")
            return None

    async def get_traces_by_service(self, service_name: str) -> list[DistributedTrace]:
        """Holt Traces für Service.

        Args:
            service_name: Service-Name

        Returns:
            Liste von Traces
        """
        try:
            trace_ids = self._trace_index.get(service_name, set())
            traces = []

            for trace_id in trace_ids:
                trace = await self.get_trace(trace_id)
                if trace:
                    traces.append(trace)

            return traces

        except Exception as e:
            logger.error(f"Traces by service retrieval fehlgeschlagen: {e}")
            return []

    async def get_traces_by_orchestration(self, orchestration_id: str) -> list[DistributedTrace]:
        """Holt Traces für Orchestration.

        Args:
            orchestration_id: Orchestration-ID

        Returns:
            Liste von Traces
        """
        try:
            traces = []

            # Suche in aktiven Traces
            for trace in self._active_traces.values():
                if trace.orchestration_id == orchestration_id:
                    traces.append(trace)

            # Suche in completed Traces
            for trace in self._completed_traces.values():
                if trace.orchestration_id == orchestration_id:
                    traces.append(trace)

            return traces

        except Exception as e:
            logger.error(f"Traces by orchestration retrieval fehlgeschlagen: {e}")
            return []

    async def _complete_trace(self, trace_id: str) -> None:
        """Vervollständigt Trace."""
        try:
            trace = self._active_traces.get(trace_id)
            if not trace:
                return

            # Update Trace-Status
            trace.end_time = datetime.utcnow()
            trace.total_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

            if trace.error_spans > 0:
                trace.status = TraceStatus.ERROR
            else:
                trace.status = TraceStatus.COMPLETED

            # Move zu completed traces
            self._completed_traces[trace_id] = trace
            del self._active_traces[trace_id]

            # Performance-Tracking
            self._update_trace_performance_stats(trace.total_duration_ms)

            logger.debug({
                "event": "trace_completed",
                "trace_id": trace_id,
                "total_duration_ms": trace.total_duration_ms,
                "total_spans": trace.total_spans,
                "error_spans": trace.error_spans,
                "status": trace.status.value
            })

        except Exception as e:
            logger.error(f"Trace completion fehlgeschlagen: {e}")

    async def _trace_monitoring_loop(self) -> None:
        """Background-Loop für Trace-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._monitor_active_traces()

            except Exception as e:
                logger.error(f"Trace monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _span_timeout_loop(self) -> None:
        """Background-Loop für Span-Timeouts."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Alle 30 Sekunden

                if self._is_running:
                    await self._check_span_timeouts()

            except Exception as e:
                logger.error(f"Span timeout loop fehlgeschlagen: {e}")
                await asyncio.sleep(30)

    async def _trace_cleanup_loop(self) -> None:
        """Background-Loop für Trace-Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Jede Stunde

                if self._is_running:
                    await self._cleanup_old_traces()

            except Exception as e:
                logger.error(f"Trace cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(3600)

    async def _correlation_analysis_loop(self) -> None:
        """Background-Loop für Correlation-Analysis."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Alle 5 Minuten

                if self._is_running:
                    await self._analyze_trace_correlations()

            except Exception as e:
                logger.error(f"Correlation analysis loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _monitor_active_traces(self) -> None:
        """Monitort aktive Traces."""
        try:
            current_time = datetime.utcnow()

            for trace_id, trace in self._active_traces.items():
                # Prüfe Trace-Alter
                trace_age = (current_time - trace.start_time).total_seconds()

                if trace_age > 3600:  # 1 Stunde
                    logger.warning({
                        "event": "long_running_trace",
                        "trace_id": trace_id,
                        "age_seconds": trace_age,
                        "completed_spans": trace.completed_spans,
                        "total_spans": trace.total_spans
                    })

        except Exception as e:
            logger.error(f"Active traces monitoring fehlgeschlagen: {e}")

    async def _check_span_timeouts(self) -> None:
        """Prüft Span-Timeouts."""
        try:
            current_time = datetime.utcnow()
            timeout_spans = []

            for span_id, span in self._active_spans.items():
                span_age = (current_time - span.start_time).total_seconds()

                if span_age > self.span_timeout_seconds:
                    timeout_spans.append(span_id)

            # Beende Timeout-Spans
            for span_id in timeout_spans:
                await self.finish_span(
                    span_id=span_id,
                    status_code=408,  # Request Timeout
                    error="Span timeout exceeded"
                )

            if timeout_spans:
                logger.warning({
                    "event": "span_timeouts_detected",
                    "timeout_spans": len(timeout_spans)
                })

        except Exception as e:
            logger.error(f"Span timeouts check fehlgeschlagen: {e}")

    async def _cleanup_old_traces(self) -> None:
        """Bereinigt alte Traces."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.trace_retention_hours)
            old_trace_ids = []

            for trace_id, trace in self._completed_traces.items():
                if trace.end_time and trace.end_time < cutoff_time:
                    old_trace_ids.append(trace_id)

            # Entferne alte Traces
            for trace_id in old_trace_ids:
                trace = self._completed_traces[trace_id]

                # Entferne aus Index
                for service_name in trace.service_names:
                    self._trace_index[service_name].discard(trace_id)

                # Entferne aus Correlation-Map
                if trace_id in self._correlation_map:
                    del self._correlation_map[trace_id]

                del self._completed_traces[trace_id]

            if old_trace_ids:
                logger.debug({
                    "event": "old_traces_cleaned",
                    "cleaned_traces": len(old_trace_ids)
                })

        except Exception as e:
            logger.error(f"Old traces cleanup fehlgeschlagen: {e}")

    async def _analyze_trace_correlations(self) -> None:
        """Analysiert Trace-Correlations."""
        try:
            # Analysiere Service-Interaktionen
            service_interactions = defaultdict(set)

            for trace in self._active_traces.values():
                services = list(trace.service_names)
                for i, service_a in enumerate(services):
                    for service_b in services[i+1:]:
                        service_interactions[service_a].add(service_b)
                        service_interactions[service_b].add(service_a)

            # Log interessante Patterns
            for service, interactions in service_interactions.items():
                if len(interactions) > 5:  # Service mit vielen Interaktionen
                    logger.debug({
                        "event": "high_interaction_service",
                        "service": service,
                        "interactions": len(interactions),
                        "connected_services": list(interactions)
                    })

        except Exception as e:
            logger.error(f"Trace correlations analysis fehlgeschlagen: {e}")

    def _should_sample(self) -> bool:
        """Prüft ob Trace gesampelt werden soll."""
        import random
        return random.random() < self.sampling_rate

    def _update_trace_performance_stats(self, duration_ms: float) -> None:
        """Aktualisiert Trace-Performance-Statistiken."""
        try:
            completed_traces = len(self._completed_traces)
            if completed_traces > 0:
                current_avg = self._tracing_performance_stats["avg_trace_duration_ms"]
                new_avg = ((current_avg * (completed_traces - 1)) + duration_ms) / completed_traces
                self._tracing_performance_stats["avg_trace_duration_ms"] = new_avg

            # Berechne Completion-Rate
            total_traces = self._tracing_performance_stats["total_traces_created"]
            if total_traces > 0:
                self._tracing_performance_stats["trace_completion_rate"] = completed_traces / total_traces

        except Exception as e:
            logger.error(f"Trace performance stats update fehlgeschlagen: {e}")

    def _update_span_performance_stats(self, duration_ms: float) -> None:
        """Aktualisiert Span-Performance-Statistiken."""
        try:
            total_spans = self._tracing_performance_stats["total_spans_created"]
            current_avg = self._tracing_performance_stats["avg_span_duration_ms"]
            new_avg = ((current_avg * (total_spans - 1)) + duration_ms) / total_spans
            self._tracing_performance_stats["avg_span_duration_ms"] = new_avg

            # Berechne Span-Completion-Rate
            completed_spans = sum(trace.completed_spans for trace in self._completed_traces.values())
            if total_spans > 0:
                self._tracing_performance_stats["span_completion_rate"] = completed_spans / total_spans

        except Exception as e:
            logger.error(f"Span performance stats update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            stats = self._tracing_performance_stats.copy()

            # Current State
            stats["active_traces"] = len(self._active_traces)
            stats["completed_traces"] = len(self._completed_traces)
            stats["active_spans"] = len(self._active_spans)

            # Configuration
            stats["sampling_rate"] = self.sampling_rate
            stats["trace_retention_hours"] = self.trace_retention_hours
            stats["span_timeout_seconds"] = self.span_timeout_seconds

            # Service-Index
            stats["indexed_services"] = len(self._trace_index)
            stats["correlation_entries"] = len(self._correlation_map)

            return stats

        except Exception as e:
            logger.error(f"Tracing performance stats retrieval fehlgeschlagen: {e}")
            return {}
