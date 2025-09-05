"""Health- und Metrics-Endpunkte für Keiko.

Dieses Modul stellt Endpunkt-Registrierungsfunktionen bereit, um Health- und
Metrics-Routen an eine FastAPI‑Anwendung zu binden.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .common.health_service import create_health_response
from .common.logger_utils import get_module_logger, safe_log_exception

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_module_logger(__name__)


def register_health_and_metrics(app: FastAPI) -> None:
    """Registriert Health- und Metrics‑Routen auf der App."""

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus Metrics Endpoint."""
        try:
            payload = generate_latest()
            # Content-Length explizit entfernen, um ASGI-Probleme zu vermeiden
            response = Response(content=payload, media_type=CONTENT_TYPE_LATEST)
            # Content-Length wird automatisch von FastAPI gesetzt, aber kann Probleme verursachen
            if "content-length" in response.headers:
                del response.headers["content-length"]
            return response
        except Exception as exc:  # pragma: no cover
            safe_log_exception(logger, exc, "Metrics Fehler", endpoint="metrics")
            response = Response(content=b"", media_type=CONTENT_TYPE_LATEST, status_code=500)
            if "content-length" in response.headers:
                del response.headers["content-length"]
            return response

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health Check Endpoint."""
        return await create_health_response(include_additional_info=False)


__all__ = [
    "register_health_and_metrics",
]
