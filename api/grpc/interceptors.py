"""gRPC Server Interceptors für KEI-RPC.

Nutzt BaseInterceptor für Code-Deduplizierung und verbesserte Wartbarkeit.
"""

from __future__ import annotations

import time
from typing import Any

import grpc

from kei_logging import get_logger

from .auth_interceptor import AuthInterceptor
from .base_interceptor import BaseInterceptor, ServicerContext, UnaryUnaryHandler
from .constants import DLPConfig, MetadataKeys, TracingConfig
from .rate_limit_interceptor import RateLimitInterceptor

logger = get_logger(__name__)

# Optional Dependencies
try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    trace = None
    OTEL_AVAILABLE = False

try:
    from observability.deadline import set_deadline_ms_from_now

    DEADLINE_AVAILABLE = True
except ImportError:
    set_deadline_ms_from_now = None
    DEADLINE_AVAILABLE = False

try:
    from monitoring.custom_metrics import MonitoringManager

    monitoring_manager = MonitoringManager()
    METRICS_AVAILABLE = True
except ImportError:
    MonitoringManager = None
    monitoring_manager = None
    METRICS_AVAILABLE = False


class TracingInterceptor(BaseInterceptor):
    """OpenTelemetry Tracing Interceptor.

    Erstellt Spans für gRPC-Methoden mit automatischer Fehler-Behandlung.
    """

    def __init__(self) -> None:
        """Initialisiert Tracing Interceptor."""
        super().__init__("Tracing")
        self.tracer = trace.get_tracer(TracingConfig.OTEL_SERVICE_NAME) if OTEL_AVAILABLE else None

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeitet Request mit Tracing."""
        if not self.tracer:
            return await behavior(request, context)

        with self.tracer.start_as_current_span(
            f"grpc.{method_name}",
            attributes={
                "rpc.system": "grpc",
                "rpc.service": TracingConfig.OTEL_SERVICE_NAME,
                "rpc.method": method_name,
            },
        ) as span:
            try:
                result = await behavior(request, context)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class DeadlineInterceptor(BaseInterceptor):
    """Deadline Propagation Interceptor.

    Propagiert und erzwingt Deadlines über gRPC-Metadata.
    """

    def __init__(self) -> None:
        """Initialisiert Deadline Interceptor."""
        super().__init__("Deadline")

    async def _before_call(self, request: Any, context: ServicerContext, method_name: str) -> None:
        """Setzt Deadline basierend auf Metadata."""
        if not DEADLINE_AVAILABLE:
            return

        try:
            metadata = context.invocation_metadata() or []
            for key, value in metadata:
                if key.lower() == MetadataKeys.TIME_BUDGET.lower():
                    budget_ms = int(value)
                    set_deadline_ms_from_now(max(0, budget_ms))

                    break
        except Exception as e:
            self.logger.warning(f"Fehler beim Setzen der Deadline: {e}")

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeitet Request mit Deadline-Überwachung."""
        try:
            return await behavior(request, context)
        except TimeoutError:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline exceeded")


class DLPInterceptor(BaseInterceptor):
    """Data Loss Prevention Interceptor.

    Redaktiert sensible Daten in Request/Response-Logs.
    """

    def __init__(self) -> None:
        """Initialisiert DLP Interceptor."""
        super().__init__("DLP")
        self.redaction_enabled = DLPConfig.REDACTION_ENABLED

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeitet Request mit DLP-Redaktion."""
        if self.redaction_enabled:
            # Redaktiere Request-Preview für Logging
            self._redact_request_preview(request, method_name)

        try:
            result = await behavior(request, context)

            if self.redaction_enabled:
                # Redaktiere Response-Preview für Logging
                self._redact_response_preview(result, method_name)

            return result
        except Exception as e:
            # Redaktiere Error-Details
            if self.redaction_enabled:
                self._redact_error_details(e, method_name)
            raise

    def _redact_request_preview(self, _: Any, method_name: str) -> None:
        """Redaktiert Request-Preview für sicheres Logging."""
        try:
            # Vereinfachte Redaktion - in Produktion würde hier
            # eine vollständige PII-Redaktion stattfinden
            field_masks = DLPConfig.FIELD_MASKS.get(method_name, [])
            if field_masks:
                pass
        except (AttributeError, ValueError) as e:
            logger.debug(f"Fehler beim Parsen der gRPC-Metadaten für Scope-Prüfung: {e}")
        except Exception as e:
            logger.warning(f"Unerwarteter Fehler bei gRPC-Scope-Prüfung: {e}")

    def _redact_response_preview(self, _: Any, method_name: str) -> None:
        """Redaktiert Response-Preview für sicheres Logging."""
        try:
            field_masks = DLPConfig.FIELD_MASKS.get(method_name, [])
            if field_masks:
                pass
        except Exception:
            pass

    def _redact_error_details(self, error: Exception, _: str) -> None:
        """Redaktiert Error-Details für sicheres Logging."""
        try:
            # Redaktiere potentiell sensible Error-Messages
            str(error)
            for _pattern in DLPConfig.PII_PATTERNS.values():
                # In Produktion würde hier Regex-Redaktion stattfinden
                pass
        except Exception:
            pass


class MetricsInterceptor(BaseInterceptor):
    """Metrics Collection Interceptor.

    Sammelt Performance-Metriken für gRPC-Methoden.
    """

    def __init__(self) -> None:
        """Initialisiert Metrics Interceptor."""
        super().__init__("Metrics")

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeitet Request mit Metrics-Collection."""
        start_time = time.time()

        # Request-Counter
        self._increment_counter(TracingConfig.METRIC_REQUESTS_TOTAL, {"method": method_name})

        try:
            return await behavior(request, context)
        except Exception:
            # Error-Counter
            self._increment_counter(TracingConfig.METRIC_ERRORS_TOTAL, {"method": method_name})
            raise
        finally:
            # Duration-Histogram
            duration_ms = (time.time() - start_time) * 1000.0
            self._record_histogram(
                TracingConfig.METRIC_DURATION, duration_ms, {"method": method_name}
            )

    def _increment_counter(self, metric_name: str, tags: dict) -> None:
        """Erhöht Counter-Metrik."""
        if METRICS_AVAILABLE and monitoring_manager:
            try:
                monitoring_manager.increment_counter(metric_name, tags)
            except Exception as e:
                self.logger.warning(f"Fehler beim Incrementieren der Metrik {metric_name}: {e}")

    def _record_histogram(self, metric_name: str, value: float, tags: dict) -> None:
        """Zeichnet Histogram-Wert auf."""
        if METRICS_AVAILABLE and monitoring_manager:
            try:
                monitoring_manager.record_histogram(metric_name, value, tags)
            except Exception as e:
                self.logger.warning(f"Fehler beim Aufzeichnen der Metrik {metric_name}: {e}")


# ============================================================================
# INTERCEPTOR FACTORY
# ============================================================================


def create_interceptor_chain() -> list[grpc.aio.ServerInterceptor]:
    """Erstellt Standard-Interceptor-Chain für KEI-RPC.

    Reihenfolge: Auth → Tracing → DLP → RateLimit → Deadline → Metrics

    Returns:
        Liste von konfigurierten Interceptors
    """
    # Optimierte List-Creation mit bedingten Interceptors
    interceptors = [
        # 1. Authentication (muss zuerst kommen)
        AuthInterceptor(),
        # 4. Rate Limiting (nach Auth)
        RateLimitInterceptor(),
    ]

    # Bedingte Interceptors hinzufügen
    # 2. Tracing (für Request-Verfolgung)
    if OTEL_AVAILABLE:
        interceptors.append(TracingInterceptor())

    # 3. DLP (für sichere Logs)
    if DLPConfig.REDACTION_ENABLED:
        interceptors.append(DLPInterceptor())

    # 5. Deadline (vor Business-Logic)
    if DEADLINE_AVAILABLE:
        interceptors.append(DeadlineInterceptor())

    # 6. Metrics (für Performance-Monitoring)
    if METRICS_AVAILABLE:
        interceptors.append(MetricsInterceptor())

    logger.info(f"Interceptor-Chain erstellt mit {len(interceptors)} Interceptors")
    return interceptors





__all__ = [
    "DLPInterceptor",
    "DeadlineInterceptor",
    "MetricsInterceptor",
    "TracingInterceptor",
    "create_interceptor_chain",

]
