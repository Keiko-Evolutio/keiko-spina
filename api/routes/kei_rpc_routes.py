# backend/api/routes/kei_rpc_routes.py
"""KEI-RPC API-Routen für standardisierte Agent-Operationen.

Implementiert plan/act/observe/explain Endpunkte mit W3C Trace-Propagation,
Idempotenz-Unterstützung und standardisierter Fehlerbehandlung.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from api.grpc.models import (
    ActRequest,
    ActResponse,
    ExplainRequest,
    ExplainResponse,
    ObserveRequest,
    ObserveResponse,
    OperationMetadata,
    PlanRequest,
    PlanResponse,
    TraceContext,
)
from api.grpc import kei_rpc_service as grpc_service
from kei_logging import get_logger
from observability import trace_function, trace_span
from security.kei_mcp_auth import require_auth
from services.limits.rate_limiter import check_agent_capability_quota

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/api-grpc", tags=["api-grpc"])

# Service-Instanz
_rpc_service = grpc_service.KEIRPCService()


# ============================================================================
# MIDDLEWARE UND DEPENDENCIES
# ============================================================================

async def extract_trace_context(
    request: Request,
    traceparent: str | None = Header(None, alias="traceparent"),
    tracestate: str | None = Header(None, alias="tracestate"),
) -> TraceContext:
    """Extrahiert W3C Trace Context aus Request Headers."""
    correlation_id = getattr(request.state, "trace_id", None)
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())

    return TraceContext(
        traceparent=traceparent,
        tracestate=tracestate,
        correlation_id=correlation_id
    )


async def extract_operation_metadata(
    _: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    x_priority: str | None = Header(None, alias="X-Priority"),
    x_timeout: int | None = Header(None, alias="X-Timeout"),
) -> OperationMetadata:
    """Extrahiert Operation-Metadaten aus Request Headers."""
    from api.grpc.models import PriorityLevel

    # Priorität validieren
    priority = PriorityLevel.NORMAL
    if x_priority:
        try:
            priority = PriorityLevel(x_priority.lower())
        except ValueError:
            logger.warning(f"Ungültige Priorität: {x_priority}")

    # Timeout validieren
    timeout = 60
    if x_timeout:
        timeout = max(1, min(x_timeout, 300))  # 1-300 Sekunden

    return OperationMetadata(
        idempotency_key=idempotency_key,
        priority=priority,
        timeout_seconds=timeout
    )


async def check_rpc_quota(
    request: Request,
    operation_type: str,
    _: str = Depends(require_auth)
) -> None:
    """Prüft RPC-Operation-Quotas."""
    try:
        # Tenant aus Request State extrahieren
        tenant_id = getattr(request.state, "tenant_id", "default")

        # Quota für RPC-Operation prüfen
        await check_agent_capability_quota(
            agent_id=f"kei_rpc_agent",
            capability_id=f"kei_rpc_{operation_type}",
            tenant_id=tenant_id
        )

    except Exception as e:
        logger.exception(f"RPC-Quota-Prüfung fehlgeschlagen: {e}")
        raise HTTPException(
            status_code=429,
            detail="RPC-Operation-Quota überschritten"
        )


def create_quota_dependency(operation_type: str):
    """Erstellt eine Quota-Dependency für eine spezifische Operation."""
    async def quota_check(request: Request, auth: str = Depends(require_auth)) -> None:
        await check_rpc_quota(request, operation_type, auth)
    return quota_check


def handle_rpc_error(operation_id: str, correlation_id: str, error: Exception) -> JSONResponse:
    """Behandelt RPC-Fehler mit standardisiertem Format."""
    error_code = "INTERNAL_ERROR"
    status_code = 500

    if isinstance(error, HTTPException):
        status_code = error.status_code
        error_code = f"HTTP_{error.status_code}"
    elif "timeout" in str(error).lower():
        status_code = 408
        error_code = "OPERATION_TIMEOUT"
    elif "not found" in str(error).lower():
        status_code = 404
        error_code = "AGENT_NOT_FOUND"
    elif "quota" in str(error).lower() or "rate limit" in str(error).lower():
        status_code = 429
        error_code = "QUOTA_EXCEEDED"

    error_response = {
        "operation_id": operation_id,
        "correlation_id": correlation_id,
        "status": "failed",
        "error": {
            "error_code": error_code,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now(UTC).isoformat()
        }
    }

    return JSONResponse(
        status_code=status_code,
        content=error_response,
        headers={
            "X-Correlation-ID": correlation_id,
            "X-Operation-ID": operation_id
        }
    )


# ============================================================================
# RPC OPERATION ENDPOINTS
# ============================================================================

@router.post("/plan", response_model=PlanResponse)
@trace_function("api.grpc.plan")
async def plan_operation(
    request: PlanRequest,
    _: Request,
    trace_context: TraceContext = Depends(extract_trace_context),
    metadata: OperationMetadata = Depends(extract_operation_metadata),
    __: str = Depends(require_auth),
    _quota: None = Depends(create_quota_dependency("plan"))
) -> PlanResponse:
    """Führt Plan-Operation aus.

    Erstellt einen strukturierten Plan basierend auf der Zielbeschreibung
    und verfügbaren Ressourcen.
    """
    # Trace Context und Metadaten in Request einbetten
    request.trace_context = trace_context
    request.metadata = metadata

    try:
        with trace_span("api.grpc.plan.execute", {
            "operation_id": metadata.operation_id,
            "correlation_id": trace_context.correlation_id,
            "objective": request.objective[:100]  # Gekürzt für Tracing
        }):
            response = await _rpc_service.plan(request)

            # Response Headers setzen
            return Response(
                content=response.json(),
                media_type="application/json",
                headers={
                    "X-Correlation-ID": trace_context.correlation_id,
                    "X-Operation-ID": metadata.operation_id,
                    "X-Agent-ID": response.agent_id or "unknown",
                    "X-Duration-MS": str(response.timing.duration_ms or 0)
                }
            )

    except Exception as e:
        logger.exception(f"Plan-Operation fehlgeschlagen: {e}")
        return handle_rpc_error(metadata.operation_id, trace_context.correlation_id, e)


@router.post("/act", response_model=ActResponse)
@trace_function("api.grpc.act")
async def act_operation(
    request: ActRequest,
    _: Request,
    trace_context: TraceContext = Depends(extract_trace_context),
    metadata: OperationMetadata = Depends(extract_operation_metadata),
    __: str = Depends(require_auth),
    _quota: None = Depends(create_quota_dependency("act"))
) -> ActResponse:
    """Führt Act-Operation aus.

    Führt eine spezifische Aktion basierend auf der Aktions-Beschreibung
    und den bereitgestellten Parametern aus.
    """
    # Trace Context und Metadaten in Request einbetten
    request.trace_context = trace_context
    request.metadata = metadata

    try:
        with trace_span("api.grpc.act.execute", {
            "operation_id": metadata.operation_id,
            "correlation_id": trace_context.correlation_id,
            "action": request.action[:100]  # Gekürzt für Tracing
        }):
            response = await _rpc_service.act(request)

            return Response(
                content=response.json(),
                media_type="application/json",
                headers={
                    "X-Correlation-ID": trace_context.correlation_id,
                    "X-Operation-ID": metadata.operation_id,
                    "X-Agent-ID": response.agent_id or "unknown",
                    "X-Duration-MS": str(response.timing.duration_ms or 0)
                }
            )

    except Exception as e:
        logger.exception(f"Act-Operation fehlgeschlagen: {e}")
        return handle_rpc_error(metadata.operation_id, trace_context.correlation_id, e)


@router.post("/observe", response_model=ObserveResponse)
@trace_function("api.grpc.observe")
async def observe_operation(
    request: ObserveRequest,
    _: Request,
    trace_context: TraceContext = Depends(extract_trace_context),
    metadata: OperationMetadata = Depends(extract_operation_metadata),
    __: str = Depends(require_auth),
    _quota: None = Depends(create_quota_dependency("observe"))
) -> ObserveResponse:
    """Führt Observe-Operation aus.

    Beobachtet und analysiert den angegebenen Zustand oder Prozess
    und liefert strukturierte Beobachtungen zurück.
    """
    # Trace Context und Metadaten in Request einbetten
    request.trace_context = trace_context
    request.metadata = metadata

    try:
        with trace_span("api.grpc.observe.execute", {
            "operation_id": metadata.operation_id,
            "correlation_id": trace_context.correlation_id,
            "observation_target": request.observation_target[:100]
        }):
            response = await _rpc_service.observe(request)

            return Response(
                content=response.json(),
                media_type="application/json",
                headers={
                    "X-Correlation-ID": trace_context.correlation_id,
                    "X-Operation-ID": metadata.operation_id,
                    "X-Agent-ID": response.agent_id or "unknown",
                    "X-Duration-MS": str(response.timing.duration_ms or 0)
                }
            )

    except Exception as e:
        logger.exception(f"Observe-Operation fehlgeschlagen: {e}")
        return handle_rpc_error(metadata.operation_id, trace_context.correlation_id, e)


@router.post("/explain", response_model=ExplainResponse)
@trace_function("api.grpc.explain")
async def explain_operation(
    request: ExplainRequest,
    _: Request,
    trace_context: TraceContext = Depends(extract_trace_context),
    metadata: OperationMetadata = Depends(extract_operation_metadata),
    __: str = Depends(require_auth),
    _quota: None = Depends(create_quota_dependency("explain"))
) -> ExplainResponse:
    """Führt Explain-Operation aus.

    Generiert eine strukturierte Erklärung für das angegebene Subjekt
    basierend auf dem gewünschten Detailgrad und der Zielgruppe.
    """
    # Trace Context und Metadaten in Request einbetten
    request.trace_context = trace_context
    request.metadata = metadata

    try:
        with trace_span("api.grpc.explain.execute", {
            "operation_id": metadata.operation_id,
            "correlation_id": trace_context.correlation_id,
            "subject": request.subject[:100],
            "detail_level": request.detail_level
        }):
            response = await _rpc_service.explain(request)

            return Response(
                content=response.json(),
                media_type="application/json",
                headers={
                    "X-Correlation-ID": trace_context.correlation_id,
                    "X-Operation-ID": metadata.operation_id,
                    "X-Agent-ID": response.agent_id or "unknown",
                    "X-Duration-MS": str(response.timing.duration_ms or 0)
                }
            )

    except Exception as e:
        logger.exception(f"Explain-Operation fehlgeschlagen: {e}")
        return handle_rpc_error(metadata.operation_id, trace_context.correlation_id, e)


# ============================================================================
# HEALTH UND STATUS ENDPOINTS
# ============================================================================

@router.get("/health")
@trace_function("api.grpc.health")
async def health_check() -> dict[str, Any]:
    """Gibt KEI-RPC Service Health-Status zurück."""
    try:
        # Service-Initialisierung prüfen
        await _rpc_service.ensure_initialized()

        health_status = _rpc_service.get_health_status()
        return {
            "status": "healthy",
            "service": "api-grpc",
            "version": "1.0.0",
            "timestamp": datetime.now(UTC).isoformat(),
            "components": health_status
        }

    except Exception as e:
        logger.exception(f"KEI-RPC Health-Check fehlgeschlagen: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "api-grpc",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }
        )


@router.get("/status")
@trace_function("api.grpc.status")
async def service_status() -> dict[str, Any]:
    """Gibt detaillierten KEI-RPC Service-Status zurück."""
    try:
        await _rpc_service.ensure_initialized()

        status_info = _rpc_service.get_status_info()
        return {
            "service": "api-grpc",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.now(UTC).isoformat(),
            "statistics": {
                "operation_cache_size": status_info["operation_cache_size"],
                "available_agents": status_info["available_agents"],
            },
            "capabilities": [
                "plan", "act", "observe", "explain"
            ]
        }

    except Exception as e:
        logger.exception(f"KEI-RPC Status-Abruf fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Status-Abruf fehlgeschlagen")
