"""Prometheus Middleware für HTTP‑Metriken.

Erfasst Request‑Zähler und Latenzen pro Endpoint und Statuscode.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from starlette.responses import Response

# --- Prometheus Metriken -----------------------------------------------------

# Zähler der HTTP Requests nach Methode/Endpoint/Status
HTTP_REQUESTS_TOTAL = Counter(
    "keiko_http_requests_total",
    "Anzahl HTTP Requests",
    ["method", "endpoint", "status_code"],
)

# Latenz‑Histogramm (Sekunden) nach Methode/Endpoint/Status
HTTP_REQUEST_DURATION = Histogram(
    "keiko_http_request_duration_seconds",
    "HTTP Antwortzeiten (s)",
    ["method", "endpoint", "status_code"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)


def _endpoint_from_request(request: Request) -> str:
    """Liefert einen stabilen Endpoint‑Namen aus dem Request.

    Bevorzugt die Route Template Path, fällt zurück auf den Path.
    """
    try:
        route = request.scope.get("route")
        if route and getattr(route, "path_format", None):
            return str(route.path_format)
        if route and getattr(route, "path", None):
            return str(route.path)
    except Exception:
        pass
    return request.url.path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware zur Erfassung von HTTP‑Metriken.

    - Inkrementiert Request‑Zähler
    - Beobachtet Antwortzeit im Histogramm
    """

    def __init__(self, app) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        method = request.method
        endpoint = _endpoint_from_request(request)
        status_code: int | None = None

        try:
            response = await call_next(request)
            status_code = int(response.status_code)
            return response
        finally:
            # Dauer und Zähler nach Abschluss messen
            duration = max(0.0, time.perf_counter() - start)
            code_label = str(status_code or 500)
            try:
                HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=code_label).inc()
                HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint, status_code=code_label).observe(duration)
            except Exception:
                # Metrics sind best‑effort
                pass


__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION",
    "PrometheusMiddleware",
]
