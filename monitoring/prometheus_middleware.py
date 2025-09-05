"""
Prometheus Monitoring Middleware for Keiko Backend
Provides metrics collection for the separated backend service
"""

import time
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code', 'repository']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint', 'repository']
)

API_ERRORS = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type', 'repository']
)

AGENT_METRICS = Counter(
    'agent_operations_total',
    'Total agent operations',
    ['operation', 'agent_type', 'status']
)

REGISTERED_AGENTS = Counter(
    'registered_agents_total',
    'Total registered agents',
    ['agent_type', 'status']
)

async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    """Prometheus metrics collection middleware"""
    start_time = time.time()

    # Extract request information
    method = request.method
    endpoint = request.url.path
    repository = "keiko-backend"

    try:
        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time

        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint,
            repository=repository
        ).observe(duration)

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=response.status_code,
            repository=repository
        ).inc()

        return response

    except Exception as e:
        # Record error metrics
        API_ERRORS.labels(
            endpoint=endpoint,
            error_type=type(e).__name__,
            repository=repository
        ).inc()
        raise

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Agent-specific metrics functions
def record_agent_operation(operation: str, agent_type: str, status: str):
    """Record agent operation metrics"""
    AGENT_METRICS.labels(
        operation=operation,
        agent_type=agent_type,
        status=status
    ).inc()

def record_agent_registration(agent_type: str, status: str):
    """Record agent registration metrics"""
    REGISTERED_AGENTS.labels(
        agent_type=agent_type,
        status=status
    ).inc()
