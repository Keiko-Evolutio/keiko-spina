"""
Advanced Observability Setup für Keiko Platform
Distributed Tracing, Correlation-IDs und Business-Metriken
"""

import uuid
import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, UTC
from contextvars import ContextVar
from functools import wraps

# OpenTelemetry Imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

# Context Variables für Request-Tracking
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


@dataclass
class TraceContext:
    """Trace-Kontext für Request-Tracking"""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service_name: str = "keiko-service"
    operation_name: str = "unknown"
    start_time: float = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()


@dataclass
class BusinessMetric:
    """Business-Metrik für Observability"""
    name: str
    value: float
    unit: str
    tags: Dict[str, str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat()
        }


class ObservabilityManager:
    """Zentrale Observability-Verwaltung"""
    
    def __init__(self, service_name: str, jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        
        # Setup OpenTelemetry
        self._setup_tracing()
        self._setup_metrics()
        
        # Business Metrics Storage
        self.business_metrics: List[BusinessMetric] = []
        
        logger.info(f"Observability Manager initialisiert für Service: {service_name}")
    
    def _setup_tracing(self):
        """Setup Distributed Tracing"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # Tracer Provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Jaeger Exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.jaeger_endpoint,
            collector_endpoint=self.jaeger_endpoint.replace("/api/traces", "/api/traces"),
        )
        
        # Span Processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Auto-Instrumentation
        FastAPIInstrumentor.instrument()
        HTTPXClientInstrumentor.instrument()
        GrpcInstrumentorClient().instrument()
        GrpcInstrumentorServer().instrument()
    
    def _setup_metrics(self):
        """Setup Metrics Collection"""
        resource = Resource.create({
            "service.name": self.service_name
        })
        
        # Prometheus Metric Reader
        prometheus_reader = PrometheusMetricReader()
        
        # Meter Provider
        metrics.set_meter_provider(MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        ))
        
        # Meter
        self.meter = metrics.get_meter(__name__)
        
        # Standard Metrics
        self.request_counter = self.meter.create_counter(
            name="http_requests_total",
            description="Total HTTP requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="http_request_duration_seconds",
            description="HTTP request duration",
            unit="s"
        )
        
        self.business_metric_counter = self.meter.create_counter(
            name="business_events_total",
            description="Total business events",
            unit="1"
        )
        
        # Keiko-spezifische Metrics
        self.agent_registrations = self.meter.create_counter(
            name="keiko_agent_registrations_total",
            description="Total agent registrations",
            unit="1"
        )
        
        self.function_calls = self.meter.create_counter(
            name="keiko_function_calls_total",
            description="Total function calls",
            unit="1"
        )
        
        self.function_duration = self.meter.create_histogram(
            name="keiko_function_execution_duration_seconds",
            description="Function execution duration",
            unit="s"
        )
    
    def start_trace(self, operation_name: str, **kwargs) -> TraceContext:
        """Startet neuen Trace"""
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
        
        # Baggage für Cross-Service-Propagation
        baggage.set_baggage("correlation_id", correlation_id)
        baggage.set_baggage("service_name", self.service_name)
        
        context = TraceContext(
            correlation_id=correlation_id,
            service_name=self.service_name,
            operation_name=operation_name,
            **kwargs
        )
        
        # OpenTelemetry Span
        span = self.tracer.start_span(operation_name)
        span.set_attribute("correlation_id", correlation_id)
        span.set_attribute("service_name", self.service_name)
        
        for key, value in kwargs.items():
            if value is not None:
                span.set_attribute(key, str(value))
        
        logger.debug(f"Trace gestartet: {operation_name} (ID: {correlation_id})")
        return context
    
    def add_span_event(self, name: str, attributes: Dict[str, Any] = None):
        """Fügt Event zu aktuellem Span hinzu"""
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes or {})
    
    def set_span_attribute(self, key: str, value: Any):
        """Setzt Attribut für aktuellen Span"""
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, str(value))
    
    def record_business_metric(
        self, 
        name: str, 
        value: float, 
        unit: str = "count",
        tags: Dict[str, str] = None
    ):
        """Zeichnet Business-Metrik auf"""
        metric = BusinessMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            timestamp=datetime.now(UTC)
        )
        
        self.business_metrics.append(metric)
        
        # OpenTelemetry Metric
        self.business_metric_counter.add(1, {
            "metric_name": name,
            "unit": unit,
            **metric.tags
        })
        
        logger.debug(f"Business-Metrik aufgezeichnet: {name} = {value} {unit}")
    
    def record_agent_registration(self, agent_id: str, agent_type: str, success: bool):
        """Zeichnet Agent-Registrierung auf"""
        self.agent_registrations.add(1, {
            "agent_type": agent_type,
            "status": "success" if success else "failure"
        })
        
        self.record_business_metric(
            "agent_registration",
            1,
            "count",
            {"agent_id": agent_id, "agent_type": agent_type, "success": str(success)}
        )
    
    def record_function_call(
        self, 
        function_name: str, 
        agent_id: str, 
        duration: float, 
        success: bool
    ):
        """Zeichnet Function-Call auf"""
        self.function_calls.add(1, {
            "function_name": function_name,
            "agent_id": agent_id,
            "status": "success" if success else "failure"
        })
        
        self.function_duration.record(duration, {
            "function_name": function_name,
            "agent_id": agent_id
        })
        
        self.record_business_metric(
            "function_call",
            duration,
            "seconds",
            {
                "function_name": function_name,
                "agent_id": agent_id,
                "success": str(success)
            }
        )
    
    def get_correlation_id(self) -> str:
        """Holt aktuelle Correlation-ID"""
        return correlation_id_var.get()
    
    def get_business_metrics(self, since: datetime = None) -> List[Dict[str, Any]]:
        """Holt Business-Metriken"""
        if since:
            filtered_metrics = [
                m for m in self.business_metrics 
                if m.timestamp >= since
            ]
        else:
            filtered_metrics = self.business_metrics
        
        return [metric.to_dict() for metric in filtered_metrics]


def trace_function(operation_name: str = None):
    """Decorator für Function-Tracing"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with observability_manager.tracer.start_as_current_span(op_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error", str(e))
                    span.record_exception(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("function.duration", duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with observability_manager.tracer.start_as_current_span(op_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error", str(e))
                    span.record_exception(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("function.duration", duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class CorrelationIDMiddleware:
    """Middleware für Correlation-ID-Propagation"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extrahiere Correlation-ID aus Headers
            headers = dict(scope["headers"])
            correlation_id = headers.get(b"x-correlation-id", b"").decode()
            
            if not correlation_id:
                correlation_id = str(uuid.uuid4())
            
            correlation_id_var.set(correlation_id)
            
            # Füge Correlation-ID zu Response-Headers hinzu
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append([b"x-correlation-id", correlation_id.encode()])
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Global Observability Manager
observability_manager: Optional[ObservabilityManager] = None


def init_observability(service_name: str, jaeger_endpoint: str = "http://localhost:14268/api/traces"):
    """Initialisiert Observability für Service"""
    global observability_manager
    observability_manager = ObservabilityManager(service_name, jaeger_endpoint)
    return observability_manager


# Beispiel-Usage
if __name__ == "__main__":
    # Observability initialisieren
    obs = init_observability("keiko-backend")
    
    # Trace starten
    context = obs.start_trace("test_operation", user_id="user123")
    
    # Business-Metriken aufzeichnen
    obs.record_agent_registration("agent-001", "code_assistant", True)
    obs.record_function_call("generate_code", "agent-001", 1.5, True)
    
    # Span-Events hinzufügen
    obs.add_span_event("processing_started", {"input_size": 1024})
    obs.set_span_attribute("processing_result", "success")
    
    print(f"Correlation ID: {obs.get_correlation_id()}")
    print(f"Business Metrics: {len(obs.get_business_metrics())}")
    
    # Decorated Function Example
    @trace_function("example_function")
    async def example_function():
        await asyncio.sleep(0.1)
        return "success"
    
    # Run example
    asyncio.run(example_function())
