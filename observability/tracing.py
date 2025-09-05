# backend/observability/tracing.py
"""OpenTelemetry Tracing Modul."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar, cast

from kei_logging import get_logger

# Logfire-Integration (optional)
try:
    from .logfire_config import LogfireSettings, get_logfire_settings
    from .logfire_integration import (
        LogfireManager,
        get_logfire_manager,
        initialize_logfire,
        shutdown_logfire,
    )
    LOGFIRE_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Logfire-Integration
    initialize_logfire = None
    shutdown_logfire = None
    get_logfire_manager = None
    LogfireManager = None
    LogfireSettings = None
    get_logfire_settings = None
    LOGFIRE_INTEGRATION_AVAILABLE = False

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])

# ============================================================================
# OPENTELEMETRY IMPORTS
# ============================================================================

try:
    from opentelemetry import metrics, propagate, trace
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OPENTELEMETRY_AVAILABLE = True
    logger.debug("✅ OpenTelemetry Core verfügbar")
except ImportError as import_error:
    # Fallback-Definitionen für fehlende OpenTelemetry-Komponenten
    metrics = None
    propagate = None
    trace = None
    W3CBaggagePropagator = None
    OTLPMetricExporter = None
    OTLPSpanExporter = None
    FastAPIInstrumentor = None
    HTTPXClientInstrumentor = None
    RequestsInstrumentor = None
    CompositePropagator = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    TraceIdRatioBased = None
    TraceContextTextMapPropagator = None
    OPENTELEMETRY_AVAILABLE = False
    logger.info(f"ℹ️ OpenTelemetry nicht verfügbar - Fallback-Modus aktiv. "
               f"Für vollständige Observability installieren Sie: pip install 'keiko-backend[observability]'. "
               f"Details: {import_error}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ObservabilityConfig:
    """OpenTelemetry Konfiguration."""
    service_name: str = "keiko-backend"
    service_version: str = "1.0.0"
    environment: str = "development"
    otlp_endpoint: str | None = None  # z.B. http://localhost:4317
    enable_tracing: bool = True
    enable_metrics: bool = True
    sampling_ratio: float = 1.0


# ============================================================================
# GLOBALE VARIABLEN
# ============================================================================

_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None
_tracer: Any | None = None
_is_initialized: bool = False
_initialization_in_progress: bool = False
_shutdown_in_progress: bool = False
_shutdown_completed: bool = False


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def _apply_span_attributes(span: Any, func_name: str | None, attributes: dict[str, Any] | None) -> None:
    """Setzt gemeinsame Span-Attribute.

    - Setzt optional den Funktionsnamen
    - Überträgt zusätzliche Attribute
    """
    # Funktionsname setzen (falls vorhanden)
    if func_name:
        span.set_attribute("function.name", func_name)

    # Benutzerdefinierte Attribute setzen
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

def setup_opentelemetry(app=None, config: ObservabilityConfig | None = None) -> None:
    """OpenTelemetry Setup mit vollständiger Trace-Propagation und optionaler Logfire-Integration."""
    if not OPENTELEMETRY_AVAILABLE:
        logger.info("OpenTelemetry nicht verfügbar - Setup übersprungen")
        return

    # Konfiguration laden
    cfg = config or _config_from_env()

    # Logfire-Integration initialisieren (falls verfügbar)
    if LOGFIRE_INTEGRATION_AVAILABLE:
        try:
            logfire_settings = get_logfire_settings()
            if initialize_logfire(logfire_settings):
                logger.info("✅ Logfire-Integration erfolgreich initialisiert")
            else:
                logger.warning("⚠️ Logfire-Integration fehlgeschlagen, verwende nur OpenTelemetry")
        except Exception as e:
            logger.warning(f"⚠️ Fehler bei Logfire-Initialisierung: {e}")
    else:
        logger.debug("Logfire-Integration nicht verfügbar")

    # Basis-Setup
    init_tracing(cfg)

    # FastAPI-spezifische Instrumentierung
    if app is not None:
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("✅ FastAPI-Instrumentierung aktiviert")
        except Exception as e:
            logger.warning(f"FastAPI-Instrumentierung fehlgeschlagen: {e}")

    logger.info("✅ OpenTelemetry Setup mit Trace-Propagation abgeschlossen")


def extract_traceparent(headers: dict[str, Any]) -> str | None:
    """Extrahiert W3C traceparent aus Header-Dict.

    Args:
        headers: Header-Dictionary eines Frames oder Requests

    Returns:
        Traceparent-String oder None
    """
    try:
        value = headers.get("traceparent")
        if isinstance(value, str) and value:
            return value
    except Exception:
        return None
    return None


def ensure_traceparent(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Erzwingt Server-seitige Traceparent-Injektion.

    - Wenn bereits vorhanden, wird bestehender Wert beibehalten
    - Wenn nicht vorhanden und OTEL aktiv, wird aktueller Kontext injiziert
    - Fallback: generiert Dummy-Header, falls OTEL fehlt

    Args:
        headers: Optionale vorhandene Header

    Returns:
        Header-Dict mit garantiertem `traceparent`
    """
    hdrs: dict[str, Any] = dict(headers or {})
    if "traceparent" in hdrs and isinstance(hdrs["traceparent"], str) and hdrs["traceparent"]:
        return hdrs
    if not OPENTELEMETRY_AVAILABLE:
        # Minimaler Dummy für Korrelation in Logs
        hdrs["traceparent"] = "00-00000000000000000000000000000000-0000000000000000-01"
        return hdrs
    try:
        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        if "traceparent" in carrier:
            hdrs["traceparent"] = carrier["traceparent"]
        else:
            hdrs["traceparent"] = "00-00000000000000000000000000000000-0000000000000000-01"
    except Exception:
        hdrs["traceparent"] = "00-00000000000000000000000000000000-0000000000000000-01"
    return hdrs


# ============================================================================
# MCP-SPEZIFISCHE TRACING-FUNKTIONEN
# ============================================================================

def create_mcp_span(
    operation_name: str,
    server_name: str | None = None,
    tool_name: str | None = None,
    resource_uri: str | None = None,
    operation_type: str | None = None,
    attributes: dict[str, Any] | None = None
) -> trace.Span:
    """Erstellt einen MCP-spezifischen Span mit standardisierten Attributen.

    Args:
        operation_name: Name der Operation (z.B. "mcp.tool.invoke")
        server_name: Name des MCP-Servers
        tool_name: Name des Tools
        resource_uri: URI der Resource
        operation_type: Typ der Operation (tool_invocation, resource_access, etc.)
        attributes: Zusätzliche Attribute

    Returns:
        OpenTelemetry Span
    """
    if not OPENTELEMETRY_AVAILABLE or not _tracer:
        return trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)

    span = _tracer.start_span(operation_name)

    # Standard MCP-Attribute setzen
    if server_name:
        span.set_attribute("mcp.server.name", server_name)
    if tool_name:
        span.set_attribute("mcp.tool.name", tool_name)
    if resource_uri:
        span.set_attribute("mcp.resource.uri", resource_uri)
    if operation_type:
        span.set_attribute("mcp.operation.type", operation_type)

    # Service-Attribute
    span.set_attribute("service.name", "kei-mcp-api")
    span.set_attribute("service.component", "mcp-client")

    # Zusätzliche Attribute
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    return span


def trace_mcp_tool_invocation(server_name: str, tool_name: str):
    """Decorator für MCP Tool-Invocation-Tracing.

    Args:
        server_name: Name des MCP-Servers
        tool_name: Name des Tools
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with create_mcp_span(
                operation_name="mcp.tool.invoke",
                server_name=server_name,
                tool_name=tool_name,
                operation_type="tool_invocation"
            ) as span:
                try:
                    # Parameter-Anzahl als Attribut
                    if "parameters" in kwargs:
                        param_count = len(kwargs["parameters"]) if isinstance(kwargs["parameters"], dict) else 0
                        span.set_attribute("mcp.tool.parameter_count", param_count)

                    result = func(*args, **kwargs)

                    # Erfolg markieren
                    span.set_attribute("mcp.operation.success", True)
                    span.set_status(trace.Status(trace.StatusCode.OK))

                    return result

                except Exception as e:
                    # Fehler markieren
                    span.set_attribute("mcp.operation.success", False)
                    span.set_attribute("mcp.error.type", type(e).__name__)
                    span.set_attribute("mcp.error.message", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper
    return decorator


def trace_mcp_resource_access(server_name: str, resource_uri: str):
    """Decorator für MCP Resource-Access-Tracing.

    Args:
        server_name: Name des MCP-Servers
        resource_uri: URI der Resource
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with create_mcp_span(
                operation_name="mcp.resource.access",
                server_name=server_name,
                resource_uri=resource_uri,
                operation_type="resource_access"
            ) as span:
                try:
                    result = func(*args, **kwargs)

                    # Resource-Größe als Attribut (falls verfügbar)
                    if hasattr(result, "content") and result.content:
                        content_size = len(str(result.content))
                        span.set_attribute("mcp.resource.content_size", content_size)

                    span.set_attribute("mcp.operation.success", True)
                    span.set_status(trace.Status(trace.StatusCode.OK))

                    return result

                except Exception as e:
                    span.set_attribute("mcp.operation.success", False)
                    span.set_attribute("mcp.error.type", type(e).__name__)
                    span.set_attribute("mcp.error.message", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper
    return decorator


def get_trace_context_headers() -> dict[str, str]:
    """Extrahiert Trace-Context-Header für ausgehende HTTP-Requests.

    Returns:
        Dictionary mit Trace-Context-Headern (traceparent, tracestate)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return {}

    headers = {}

    # Aktuellen Trace-Context extrahieren
    carrier = {}
    propagate.inject(carrier)

    # W3C Trace Context Header extrahieren
    if "traceparent" in carrier:
        headers["traceparent"] = carrier["traceparent"]
    if "tracestate" in carrier:
        headers["tracestate"] = carrier["tracestate"]
    if "baggage" in carrier:
        headers["baggage"] = carrier["baggage"]

    return headers


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Injiziert Trace-Context in HTTP-Headers.

    Args:
        headers: Bestehende HTTP-Headers

    Returns:
        Headers mit injiziertem Trace-Context
    """
    if not OPENTELEMETRY_AVAILABLE:
        return headers

    # Kopie der Headers erstellen
    enriched_headers = headers.copy()

    # Trace-Context injizieren
    trace_headers = get_trace_context_headers()
    enriched_headers.update(trace_headers)

    return enriched_headers


def _config_from_env() -> ObservabilityConfig:
    """Erzeugt ObservabilityConfig aus Umgebungsvariablen."""
    return ObservabilityConfig(
        service_name=os.getenv("OTEL_SERVICE_NAME", os.getenv("APPLICATION_NAME", "keiko-backend")),
        service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        environment=os.getenv("ENVIRONMENT", os.getenv("OTEL_ENVIRONMENT", "development")),
        otlp_endpoint=(
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or os.getenv("OTLP_ENDPOINT")
        ),
        enable_tracing=os.getenv("OTEL_TRACES_ENABLED", "true").lower() == "true",
        enable_metrics=os.getenv("OTEL_METRICS_ENABLED", "true").lower() == "true",
        sampling_ratio=float(os.getenv("OTEL_SAMPLING_RATIO", os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))),
    )


def init_tracing(config: ObservabilityConfig | None = None) -> None:
    """Initialisiert OpenTelemetry-Tracing.

    Args:
        config: Optionale Observability-Konfiguration. Wenn nicht angegeben,
            wird eine Standardkonfiguration verwendet.
    """
    global _tracer_provider, _meter_provider, _tracer, _is_initialized, _initialization_in_progress

    # Verhindere Rekursion und mehrfache Initialisierung
    if _is_initialized or _initialization_in_progress:
        return

    _initialization_in_progress = True

    try:
        setup_opentelemetry()
    except (ImportError, AttributeError) as e:
        # Graceful Degradation im Fehlerfall
        logger.warning(f"Tracing Initialisierung übersprungen - Import-/Attribut-Fehler: {e}")
        _initialization_in_progress = False
        return
    except Exception as e:
        # Graceful Degradation im Fehlerfall
        logger.warning(f"Tracing Initialisierung übersprungen - Unerwarteter Fehler: {e}")
        _initialization_in_progress = False
        return

    if _is_initialized:
        logger.debug("OpenTelemetry bereits initialisiert")
        return

    cfg = config or _config_from_env()

    # Prüfung der OpenTelemetry-Verfügbarkeit mit informativen Meldungen
    if not OPENTELEMETRY_AVAILABLE:
        if cfg.enable_tracing or cfg.enable_metrics:
            logger.info("ℹ️ OpenTelemetry-Features in Konfiguration aktiviert, aber Pakete nicht verfügbar. "
                       "System läuft im Fallback-Modus ohne Tracing/Metrics. "
                       "Installieren Sie 'keiko-backend[observability]' für vollständige Observability.")
        return  # Früher Ausstieg wenn OpenTelemetry nicht verfügbar


    try:
        # Resource erstellen
        resource = Resource.create({
            "service.name": cfg.service_name,
            "service.version": cfg.service_version,
            "environment": cfg.environment
        })

        # Tracing setup (nur wenn OTEL verfügbar)
        if cfg.enable_tracing and OPENTELEMETRY_AVAILABLE:
            sampler = TraceIdRatioBased(max(0.0, min(1.0, cfg.sampling_ratio)))
            _tracer_provider = TracerProvider(resource=resource, sampler=sampler)
            if cfg.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                _tracer_provider.add_span_processor(span_processor)

            trace.set_tracer_provider(_tracer_provider)
            _tracer = trace.get_tracer(cfg.service_name, cfg.service_version)

            # W3C Trace Context Propagation konfigurieren (kompatibel mit verschiedenen OTEL-Versionen)
            try:
                # Neuere OTEL-Versionen verwenden CompositeTextMapPropagator
                from opentelemetry.propagators.composite import (
                    CompositeTextMapPropagator as _Composite,  # type: ignore
                )
            except Exception:
                try:
                    # Ältere OTEL-Versionen verwenden CompositePropagator
                    from opentelemetry.propagators.composite import (
                        CompositePropagator as _Composite,  # type: ignore
                    )
                except Exception:
                    _Composite = None  # type: ignore

            try:
                if _Composite is not None:  # type: ignore[truthy-bool]
                    propagate.set_global_textmap(
                        _Composite([
                            TraceContextTextMapPropagator(),
                            W3CBaggagePropagator(),
                        ])
                    )
                else:
                    # Fallback: Mindestens W3C Trace Context aktivieren
                    propagate.set_global_textmap(TraceContextTextMapPropagator())
            except (ImportError, AttributeError) as e:
                logger.debug(f"Erweiterte Trace-Propagation fehlgeschlagen - Import-/Attribut-Fehler: {e}")
                # Defensiver Fallback: Nur W3C Trace Context
                propagate.set_global_textmap(TraceContextTextMapPropagator())
            except Exception as e:
                logger.warning(f"Erweiterte Trace-Propagation fehlgeschlagen - Unerwarteter Fehler: {e}")
                # Defensiver Fallback: Nur W3C Trace Context
                propagate.set_global_textmap(TraceContextTextMapPropagator())

            # Automatische Instrumentierung für HTTP-Clients
            HTTPXClientInstrumentor().instrument()
            RequestsInstrumentor().instrument()

            logger.info("✅ OpenTelemetry Trace-Propagation konfiguriert")

        # Metrics setup (nur wenn OTEL verfügbar)
        if cfg.enable_metrics and OPENTELEMETRY_AVAILABLE:
            metric_readers = []
            if cfg.otlp_endpoint:
                metric_exporter = OTLPMetricExporter(endpoint=cfg.otlp_endpoint)
                # Shorter export interval to prevent deadline exceeded errors
                metric_readers = [PeriodicExportingMetricReader(
                    exporter=metric_exporter,
                    export_interval_millis=5000,  # 5 seconds instead of default 60
                    export_timeout_millis=2000    # 2 seconds timeout
                )]

            _meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
            metrics.set_meter_provider(_meter_provider)

        _is_initialized = True
        logger.info("✅ OpenTelemetry erfolgreich initialisiert")

    except Exception as e:
        logger.exception(f"❌ OpenTelemetry Setup fehlgeschlagen: {e}")
    finally:
        _initialization_in_progress = False


def trace_function(name: str | None = None, attributes: dict[str, Any] | None = None) -> \
        Callable[[F], F]:
    """Decorator für Funktions-Tracing."""

    def decorator(func: F) -> F:
        if not OPENTELEMETRY_AVAILABLE or not _tracer:
            return func

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            # Extrahiere Modul-Name mit expliziter Behandlung für generische Typen
            module_name = getattr(cast("Any", func), "__module__", "unknown")
            span_name = name or f"{module_name}.{func.__name__}"
            with _tracer.start_as_current_span(span_name) as span:
                _apply_span_attributes(span, func.__name__, attributes)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.result", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.result", "error")
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            # Extrahiere Modul-Name mit expliziter Behandlung für generische Typen
            module_name = getattr(cast("Any", func), "__module__", "unknown")
            span_name = name or f"{module_name}.{func.__name__}"
            with _tracer.start_as_current_span(span_name) as span:
                _apply_span_attributes(span, func.__name__, attributes)
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.result", "error")
                    span.record_exception(e)
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextmanager
def trace_span(span_name: str, attributes: dict[str, Any] | None = None) -> Generator[
    Any, None, None]:
    """Context Manager für Spans."""
    if not OPENTELEMETRY_AVAILABLE or not _tracer:
        yield None
        return

    with _tracer.start_as_current_span(span_name) as span:
        _apply_span_attributes(span, None, attributes)
        yield span


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Fügt Attribute zum aktuellen Span hinzu."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    current_span = trace.get_current_span()
    if current_span.is_recording():
        _apply_span_attributes(current_span, None, attributes)


def get_current_trace_id() -> str | None:
    """Gibt aktuelle Trace-ID zurück."""
    if not OPENTELEMETRY_AVAILABLE:
        return None

    current_span = trace.get_current_span()
    if current_span.is_recording():
        return format(current_span.get_span_context().trace_id, "032x")
    return None


def record_exception_in_span(exception: Exception, span_name: str | None = None,
                             attributes: dict[str, Any] | None = None) -> None:
    """Zeichnet Exception in Span auf."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.record_exception(exception)
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)


async def shutdown_tracing() -> None:
    """Beendet Tracing sauber, inklusive Logfire-Integration."""
    global _tracer_provider, _meter_provider, _is_initialized, _shutdown_in_progress, _shutdown_completed

    # Prevent multiple shutdown attempts
    if _shutdown_in_progress or _shutdown_completed:
        logger.debug("⚠️ Shutdown bereits in Bearbeitung oder abgeschlossen - überspringe")
        return

    _shutdown_in_progress = True

    try:
        # Logfire-Integration beenden (falls verfügbar)
        if LOGFIRE_INTEGRATION_AVAILABLE and shutdown_logfire:
            try:
                shutdown_logfire()
                logger.debug("✅ Logfire-Integration von Tracing beendet")
            except Exception as e:
                logger.debug(f"⚠️ Fehler beim Beenden von Logfire (erwartet bei mehrfachem Aufruf): {e}")

        if not OPENTELEMETRY_AVAILABLE or not _is_initialized:
            return

        try:
            # Shutdown MeterProvider first with timeout protection
            if _meter_provider and hasattr(_meter_provider, 'shutdown'):
                try:
                    import asyncio
                    # Use asyncio timeout to prevent hanging
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, _meter_provider.shutdown
                        ),
                        timeout=2.0  # Reduziertes Timeout
                    )
                    logger.debug("✅ MeterProvider shutdown erfolgreich")
                except asyncio.TimeoutError:
                    logger.debug("⚠️ MeterProvider shutdown timeout - forciere shutdown (erwartet)")
                except Exception as meter_e:
                    # Spezielle Behandlung für "deadline already exceeded" Fehler
                    if "deadline already exceeded" in str(meter_e).lower():
                        logger.debug("⚠️ MeterProvider shutdown deadline exceeded (erwartet bei mehrfachem Aufruf)")
                    elif "shutdown can only be called once" in str(meter_e).lower():
                        logger.debug("⚠️ MeterProvider bereits beendet (erwartet bei mehrfachem Aufruf)")
                    else:
                        logger.warning(f"⚠️ MeterProvider shutdown fehlgeschlagen: {meter_e}")
            
            # Then shutdown TracerProvider
            if _tracer_provider and hasattr(_tracer_provider, 'shutdown'):
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, _tracer_provider.shutdown
                        ),
                        timeout=1.5  # Reduziertes Timeout
                    )
                    logger.debug("✅ TracerProvider shutdown erfolgreich")
                except asyncio.TimeoutError:
                    logger.debug("⚠️ TracerProvider shutdown timeout - forciere shutdown (erwartet)")
                except Exception as tracer_e:
                    # Spezielle Behandlung für bekannte Shutdown-Fehler
                    if "deadline already exceeded" in str(tracer_e).lower():
                        logger.debug("⚠️ TracerProvider shutdown deadline exceeded (erwartet bei mehrfachem Aufruf)")
                    elif "shutdown can only be called once" in str(tracer_e).lower():
                        logger.debug("⚠️ TracerProvider bereits beendet (erwartet bei mehrfachem Aufruf)")
                    else:
                        logger.warning(f"⚠️ TracerProvider shutdown fehlgeschlagen: {tracer_e}")

            # Clean up references
            _tracer_provider = None
            _meter_provider = None
            _is_initialized = False

            logger.info("✅ Tracing erfolgreich beendet")
        except (AttributeError, TypeError) as e:
            logger.warning(f"❌ Fehler beim Tracing-Shutdown - Attribut-/Typ-Fehler: {e}")
        except Exception as e:
            logger.exception(f"❌ Fehler beim Tracing-Shutdown - Unerwarteter Fehler: {e}")
            
    finally:
        _shutdown_completed = True
        _shutdown_in_progress = False


# ============================================================================
# STATUS/HEALTH FUNCTIONS
# ============================================================================

def is_tracing_healthy() -> bool:
    """Prüft Tracing-Gesundheit."""
    return OPENTELEMETRY_AVAILABLE and _is_initialized and _tracer is not None


def get_tracing_status() -> dict[str, Any]:
    """Gibt Tracing-Status zurück."""
    return {
        "available": OPENTELEMETRY_AVAILABLE,
        "enabled": _is_initialized,
        "initialized": _is_initialized,
        "healthy": is_tracing_healthy(),
        "tracer_available": _tracer is not None,
        "fallback_mode": not OPENTELEMETRY_AVAILABLE
    }


# ============================================================================
# OBSERVABILITY STATUS FUNCTIONS
# ============================================================================

def is_observability_available() -> bool:
    """Prüft ob grundlegende Observability verfügbar ist."""
    return OPENTELEMETRY_AVAILABLE and _is_initialized


def get_observability_status() -> dict[str, Any]:
    """Status der Observability-Infrastruktur."""
    return {
        "available": is_observability_available(),
        "opentelemetry_core": OPENTELEMETRY_AVAILABLE,
        "tracing_module": True,  # Dieses Modul ist immer verfügbar
        "fallback_mode": not OPENTELEMETRY_AVAILABLE,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "service_name": os.getenv("OTEL_SERVICE_NAME", "keiko-backend"),
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Availability Flags
    "OPENTELEMETRY_AVAILABLE",
    # Configuration
    "ObservabilityConfig",
    "add_span_attributes",
    "get_current_trace_id",
    "get_observability_status",
    "get_tracing_status",
    "is_observability_available",
    # Status Functions
    "is_tracing_healthy",
    "record_exception_in_span",
    # Core Tracing Functions
    "setup_opentelemetry",
    "shutdown_tracing",
    "trace_function",
    "trace_span"
]
