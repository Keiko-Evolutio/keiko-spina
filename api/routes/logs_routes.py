"""Client Logging Routes.

Diese Routen nehmen strukturierte Client-Logs entgegen und integrieren sie in den bestehenden
Observability-Stack (OpenTelemetry, Prometheus, strukturierte Logs).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request, Response, status

# Rate limiting removed for public telemetry endpoint
from prometheus_client import Counter
from pydantic import BaseModel, Field

from monitoring import get_tracer

router = APIRouter(prefix="/logs", tags=["logs"])


CLIENT_LOGS_RECEIVED = Counter(
    "client_logs_received_total",
    "Anzahl empfangener Client-Logs",
    ["category", "severity"],
)


class ClientLogContext(BaseModel):
    """Kontextinformationen zum Client-Log."""

    traceId: str | None = Field(default=None, alias="traceId")
    userId: str | None = None
    route: str | None = None
    userAgent: str | None = None
    timestamp: str


class ClientLogEntry(BaseModel):
    """Strukturiertes Client-Log Model."""

    message: str
    category: str
    name: str | None = None
    status: int | None = None
    severity: str | None = None
    context: ClientLogContext
    raw: dict[str, Any] | None = None


@router.options("/client", operation_id="logs_client_logs_options")
async def client_logs_options(response: Response) -> dict[str, str]:
    """Handle CORS preflight requests for client logs."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Tenant-Id, X-Trace-Id"
    return {"status": "ok"}


@router.post("/client", status_code=status.HTTP_202_ACCEPTED, operation_id="logs_ingest_client_log")
async def ingest_client_log(
    entry: ClientLogEntry,
    _: Request,
    response: Response,
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> dict[str, str]:
    """Empfängt Client-Logs und integriert sie in Observability.

    - Verwendet OpenTelemetry Tracing für Korrelation
    - Erhöht Prometheus Counter
    - Schreibt strukturiertes Log mit PII-Redaktion
    """
    # Trace- und Correlation-IDs setzen
    trace_id = entry.context.traceId or x_trace_id

    try:
        tracer = get_tracer("client_logs")
        with tracer.start_as_current_span("client_log_ingest") as span:  # type: ignore[call-arg]
            if trace_id:
                span.set_attribute("client.trace_id", trace_id)

            span.set_attribute("client.category", entry.category)
            span.set_attribute("client.severity", entry.severity or "info")
            span.set_attribute("client.user_id", entry.context.userId or "")
            span.set_attribute("client.route", entry.context.route or "")
    except Exception:
        # Tracing ist optional - bei Fehlern nicht blockieren
        pass

    # Prometheus Metrik aktualisieren
    try:
        CLIENT_LOGS_RECEIVED.labels(
            category=entry.category, severity=(entry.severity or "info")
        ).inc()
    except Exception:
        # Metriken sind optional - bei Fehlern nicht blockieren
        pass

    # Strukturierte Logs ausgeben
    try:
        # Einfache PII-Redaktion ohne externe Abhängigkeiten
        safe_raw = {k: v for k, v in (entry.raw or {}).items() if not any(
            sensitive in str(k).lower() for sensitive in ["password", "token", "secret", "key"]
        )}

        from kei_logging import get_logger
        logger = get_logger(__name__)
        logger.info(
            "Client log received",
            extra={
                "source": "client",
                "message": entry.message,
                "category": entry.category,
                "name": entry.name,
                "status": entry.status,
                "severity": entry.severity,
                "context": entry.context.model_dump(by_alias=True),
                "raw": safe_raw,
                "trace_id": trace_id,
            }
        )
    except Exception:
        # Logging ist optional - bei Fehlern nicht blockieren
        pass

    # Set response headers safely
    try:
        response.headers["X-Trace-Id"] = trace_id or ""
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "X-Trace-Id"
    except Exception:
        pass

    return {"status": "accepted"}
