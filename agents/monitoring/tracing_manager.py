# backend/agents/monitoring/tracing_manager.py
"""Tracing Manager für das Agent-Framework.

Distributed Tracing mit:
- Span-Management
- Trace-Kontext
- Performance-Tracking
- Correlation-IDs
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..logging_utils import StructuredLogger

logger = StructuredLogger("tracing_manager")


class SpanStatus(Enum):
    """Span-Status-Werte."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TracingConfig:
    """Konfiguration für Tracing Manager."""

    # Tracing
    enable_tracing: bool = True
    sample_rate: float = 1.0

    # Storage
    max_spans: int = 10000
    retention_hours: int = 24

    # Context
    enable_context_propagation: bool = True


@dataclass
class TraceContext:
    """Trace-Kontext für Correlation."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None

    # Metadaten
    operation_name: str = ""
    component: str = ""
    agent_id: str | None = None
    user_id: str | None = None

    # Baggage
    baggage: dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed Tracing Span."""

    span_id: str
    trace_id: str
    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Hierarchie
    parent_span_id: str | None = None
    child_spans: list[str] = field(default_factory=list)

    # Status
    status: SpanStatus = SpanStatus.OK

    # Kontext
    component: str = ""
    agent_id: str | None = None
    user_id: str | None = None

    # Daten
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Berechnet Span-Dauer."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def add_log(self, message: str, level: str = "info", **kwargs) -> None:
        """Fügt Log-Eintrag zu Span hinzu."""
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "level": level,
            **kwargs
        }
        self.logs.append(log_entry)

    def set_tag(self, key: str, value: str) -> None:
        """Setzt Span-Tag."""
        self.tags[key] = value

    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Beendet Span."""
        self.end_time = time.time()
        self.status = status


class TracingManager:
    """Tracing Manager für das Agent-Framework."""

    def __init__(self, config: TracingConfig):
        """Initialisiert Tracing Manager.

        Args:
            config: Tracing-Konfiguration
        """
        self.config = config

        # Span-Speicher
        self._spans: dict[str, Span] = {}
        self._active_spans: dict[str, Span] = {}

        # Kontext-Stack
        self._context_stack: list[TraceContext] = []

        logger.info("Tracing Manager initialisiert")

    @staticmethod
    def create_trace_context(
        operation_name: str,
        component: str = "",
        agent_id: str | None = None,
        user_id: str | None = None,
        parent_context: TraceContext | None = None
    ) -> TraceContext:
        """Erstellt neuen Trace-Kontext.

        Args:
            operation_name: Name der Operation
            component: Komponente
            agent_id: Agent-ID
            user_id: Benutzer-ID
            parent_context: Parent-Kontext

        Returns:
            Neuer Trace-Kontext
        """
        context = TraceContext(
            operation_name=operation_name,
            component=component,
            agent_id=agent_id,
            user_id=user_id
        )

        # Parent-Kontext übernehmen
        if parent_context:
            context.trace_id = parent_context.trace_id
            context.parent_span_id = parent_context.span_id
            context.baggage = parent_context.baggage.copy()

        return context

    def start_span(
        self,
        operation_name: str,
        context: TraceContext | None = None,
        component: str = "",
        agent_id: str | None = None
    ) -> Span:
        """Startet neuen Span.

        Args:
            operation_name: Name der Operation
            context: Trace-Kontext
            component: Komponente
            agent_id: Agent-ID

        Returns:
            Neuer Span
        """
        if not self.config.enable_tracing:
            # Dummy-Span für deaktiviertes Tracing
            return Span(
                span_id="dummy",
                trace_id="dummy",
                operation_name=operation_name
            )

        # Sampling prüfen
        if not self._should_sample():
            return Span(
                span_id="sampled_out",
                trace_id="sampled_out",
                operation_name=operation_name
            )

        # Kontext verwenden oder erstellen
        if context:
            trace_id = context.trace_id
            parent_span_id = context.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None

        # Span erstellen
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            operation_name=operation_name,
            parent_span_id=parent_span_id,
            component=component,
            agent_id=agent_id
        )

        # Span speichern
        self._spans[span.span_id] = span
        self._active_spans[span.span_id] = span

        # Parent-Child-Beziehung
        if parent_span_id and parent_span_id in self._spans:
            self._spans[parent_span_id].child_spans.append(span.span_id)

        logger.debug(f"Span gestartet: {operation_name} ({span.span_id})")

        return span

    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """Beendet Span.

        Args:
            span: Zu beendender Span
            status: Span-Status
        """
        if span.span_id in ["dummy", "sampled_out"]:
            return

        span.finish(status)

        # Aus aktiven Spans entfernen
        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]

        logger.debug(
            f"Span beendet: {span.operation_name} ({span.span_id}) - "
            f"Dauer: {span.duration:.3f}s"
        )

    @asynccontextmanager
    async def trace_context(
        self,
        operation_name: str,
        component: str = "",
        agent_id: str | None = None,
        parent_context: TraceContext | None = None
    ):
        """Context Manager für Tracing.

        Args:
            operation_name: Name der Operation
            component: Komponente
            agent_id: Agent-ID
            parent_context: Parent-Kontext
        """
        # Trace-Kontext erstellen
        context = TracingManager.create_trace_context(
            operation_name=operation_name,
            component=component,
            agent_id=agent_id,
            parent_context=parent_context
        )

        # Span starten
        span = self.start_span(
            operation_name=operation_name,
            context=context,
            component=component,
            agent_id=agent_id
        )

        # Kontext auf Stack
        self._context_stack.append(context)

        try:
            yield context, span

            # Erfolgreicher Abschluss
            self.finish_span(span, SpanStatus.OK)

        except Exception as e:
            # Fehler-Status setzen
            span.add_log(f"Exception: {e!s}", level="error")
            span.set_tag("error", "true")
            span.set_tag("error.type", type(e).__name__)

            self.finish_span(span, SpanStatus.ERROR)
            raise

        finally:
            # Kontext vom Stack entfernen
            if self._context_stack and self._context_stack[-1] == context:
                self._context_stack.pop()

    def get_current_context(self) -> TraceContext | None:
        """Gibt aktuellen Trace-Kontext zurück."""
        return self._context_stack[-1] if self._context_stack else None

    def get_span(self, span_id: str) -> Span | None:
        """Gibt Span zurück.

        Args:
            span_id: Span-ID

        Returns:
            Span oder None
        """
        return self._spans.get(span_id)

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Gibt alle Spans eines Traces zurück.

        Args:
            trace_id: Trace-ID

        Returns:
            Liste von Spans
        """
        return [
            span for span in self._spans.values()
            if span.trace_id == trace_id
        ]

    def get_active_spans(self) -> list[Span]:
        """Gibt aktive Spans zurück."""
        return list(self._active_spans.values())

    def get_tracing_statistics(self) -> dict[str, Any]:
        """Gibt Tracing-Statistiken zurück."""
        spans = list(self._spans.values())
        finished_spans = [s for s in spans if s.end_time is not None]

        if not finished_spans:
            return {
                "total_spans": len(spans),
                "active_spans": len(self._active_spans),
                "finished_spans": 0
            }

        durations = [s.duration for s in finished_spans if s.duration is not None]

        stats = {
            "total_spans": len(spans),
            "active_spans": len(self._active_spans),
            "finished_spans": len(finished_spans),
            "unique_traces": len(set(s.trace_id for s in spans)),
            "unique_operations": len(set(s.operation_name for s in spans))
        }

        if durations:
            stats.update({
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations)
            })

        # Status-Verteilung
        status_counts: dict[str, int] = {}
        for span in finished_spans:
            status = span.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        stats["status_distribution"] = status_counts

        return stats

    def _should_sample(self) -> bool:
        """Prüft ob Trace gesampelt werden soll."""
        import random
        return random.random() <= self.config.sample_rate

    def cleanup_old_spans(self) -> int:
        """Bereinigt alte Spans.

        Returns:
            Anzahl der bereinigten Spans
        """
        if not self.config.retention_hours:
            return 0

        cutoff_time = time.time() - (self.config.retention_hours * 3600)
        old_span_ids = [
            span_id for span_id, span in self._spans.items()
            if span.start_time < cutoff_time
        ]

        for span_id in old_span_ids:
            del self._spans[span_id]
            # Auch aus aktiven Spans entfernen falls vorhanden
            if span_id in self._active_spans:
                del self._active_spans[span_id]

        if old_span_ids:
            logger.info(f"{len(old_span_ids)} alte Spans bereinigt")

        return len(old_span_ids)
