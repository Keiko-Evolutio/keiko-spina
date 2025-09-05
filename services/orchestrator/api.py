# backend/services/orchestrator/api.py
"""REST API für Orchestrator Service.

Implementiert HTTP-Endpoints für Task-Orchestration,
Progress-Monitoring und Service-Management.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from kei_logging import get_logger
from task_management.core_task_manager import TaskPriority, TaskType

from .data_models import ExecutionMode, OrchestrationRequest
from .orchestrator_service import OrchestratorService

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)

# Pydantic Models für API
class OrchestrationRequestModel(BaseModel):
    """API Model für Orchestration-Request."""

    task_id: str = Field(..., description="Eindeutige Task-ID")
    task_type: str = Field(..., description="Task-Type (data_processing, nlp_analysis, etc.)")
    task_name: str = Field(..., description="Task-Name")
    task_description: str = Field(..., description="Task-Beschreibung")
    task_payload: dict[str, Any] = Field(default_factory=dict, description="Task-Payload")
    priority: str = Field(
        default="normal", description="Task-Priorität (low, normal, high, urgent)"
    )

    execution_mode: str = Field(
        default="optimized",
        description="Execution-Mode (sequential, parallel, optimized, manual)"
    )
    max_parallel_tasks: int = Field(default=5, ge=1, le=20, description="Maximale parallele Tasks")
    timeout_seconds: int = Field(default=3600, ge=60, le=7200, description="Timeout in Sekunden")

    required_capabilities: list[str] = Field(
        default_factory=list, description="Erforderliche Agent-Capabilities"
    )
    preferred_agents: list[str] = Field(default_factory=list, description="Bevorzugte Agent-IDs")
    excluded_agents: list[str] = Field(
        default_factory=list, description="Ausgeschlossene Agent-IDs"
    )
    resource_constraints: dict[str, Any] = Field(
        default_factory=dict, description="Resource-Constraints"
    )

    user_id: str | None = Field(None, description="User-ID")
    session_id: str | None = Field(None, description="Session-ID")
    tenant_id: str | None = Field(None, description="Tenant-ID")
    correlation_id: str | None = Field(None, description="Correlation-ID")

    enable_decomposition: bool = Field(default=True, description="Task-Decomposition aktivieren")
    enable_performance_prediction: bool = Field(
        default=True, description="Performance-Prediction aktivieren"
    )
    enable_monitoring: bool = Field(default=True, description="Monitoring aktivieren")
    enable_recovery: bool = Field(default=True, description="Recovery aktivieren")


class OrchestrationResponseModel(BaseModel):
    """API Model für Orchestration-Response."""

    success: bool
    orchestration_id: str
    state: str
    message: str | None = None

    results: dict[str, Any] = Field(default_factory=dict)
    aggregated_result: dict[str, Any] | None = None

    total_execution_time_ms: float = 0.0
    orchestration_overhead_ms: float = 0.0
    parallelization_achieved: float = 0.0

    subtasks_count: int = 0
    failed_subtasks: list[str] = Field(default_factory=list)

    error_message: str | None = None
    error_details: dict[str, Any] = Field(default_factory=dict)

    completed_at: datetime


class ProgressResponseModel(BaseModel):
    """API Model für Progress-Response."""

    orchestration_id: str
    state: str
    completion_percentage: float

    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    running_subtasks: int
    pending_subtasks: int

    start_time: datetime | None = None
    estimated_completion_time: datetime | None = None
    execution_efficiency: float = 0.0
    resource_utilization: float = 0.0

    error_count: int = 0
    last_error: str | None = None
    last_updated: datetime


class HealthResponseModel(BaseModel):
    """API Model für Health-Response."""

    service_healthy: bool
    service_version: str
    uptime_seconds: float

    components: dict[str, bool] = Field(default_factory=dict)

    active_orchestrations: int
    total_orchestrations: int
    avg_orchestration_time_ms: float
    success_rate: float

    memory_usage_mb: float
    cpu_usage_percent: float

    check_timestamp: datetime


class MetricsResponseModel(BaseModel):
    """API Model für Metrics-Response."""

    service: dict[str, Any] = Field(default_factory=dict)
    orchestrations: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    monitoring: dict[str, Any] = Field(default_factory=dict)


# Global Service-Instanz
_orchestrator_service: OrchestratorService | None = None


def get_orchestrator_service() -> OrchestratorService:
    """Holt Orchestrator Service Instanz."""
    global _orchestrator_service

    if _orchestrator_service is None:
        _orchestrator_service = OrchestratorService()

    return _orchestrator_service


async def initialize_orchestrator_service() -> None:
    """Initialisiert Orchestrator Service."""
    service = get_orchestrator_service()
    await service.start()
    logger.info("Orchestrator Service API initialisiert")


async def shutdown_orchestrator_service() -> None:
    """Fährt Orchestrator Service herunter."""
    global _orchestrator_service

    if _orchestrator_service:
        await _orchestrator_service.stop()
        _orchestrator_service = None

    logger.info("Orchestrator Service API heruntergefahren")


# Router für Orchestrator API
router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])


@router.post("/orchestrate", response_model=OrchestrationResponseModel, status_code=HTTP_201_CREATED)
async def orchestrate_task(request: OrchestrationRequestModel) -> OrchestrationResponseModel:
    """Startet Task-Orchestration.

    Args:
        request: Orchestration-Request

    Returns:
        Orchestration-Response

    Raises:
        HTTPException: Bei Validierungs- oder Service-Fehlern
    """
    try:
        service = get_orchestrator_service()

        # Konvertiere API-Model zu Service-Model
        orchestration_request = OrchestrationRequest(
            task_id=request.task_id,
            task_type=TaskType(request.task_type),
            task_name=request.task_name,
            task_description=request.task_description,
            task_payload=request.task_payload,
            priority=TaskPriority(request.priority),
            execution_mode=ExecutionMode(request.execution_mode),
            max_parallel_tasks=request.max_parallel_tasks,
            timeout_seconds=request.timeout_seconds,
            required_capabilities=request.required_capabilities,
            preferred_agents=request.preferred_agents,
            excluded_agents=request.excluded_agents,
            resource_constraints=request.resource_constraints,
            user_id=request.user_id,
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            correlation_id=request.correlation_id,
            enable_decomposition=request.enable_decomposition,
            enable_performance_prediction=request.enable_performance_prediction,
            enable_monitoring=request.enable_monitoring,
            enable_recovery=request.enable_recovery
        )

        # Führe Orchestration aus
        result = await service.orchestrate_task(orchestration_request)

        # Konvertiere zu API-Response
        return OrchestrationResponseModel(
            success=result.success,
            orchestration_id=result.orchestration_id,
            state=result.state.value,
            message="Orchestration erfolgreich abgeschlossen" if result.success else "Orchestration fehlgeschlagen",
            results=result.results,
            aggregated_result=result.aggregated_result,
            total_execution_time_ms=result.total_execution_time_ms,
            orchestration_overhead_ms=result.orchestration_overhead_ms,
            parallelization_achieved=result.parallelization_achieved,
            subtasks_count=len(result.subtask_results),
            failed_subtasks=result.failed_subtasks,
            error_message=result.error_message,
            error_details=result.error_details,
            completed_at=result.completed_at
        )

    except ValueError as e:
        logger.exception("Orchestration-Request-Validierung fehlgeschlagen")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e)) from e

    except Exception as e:
        logger.exception("Orchestration fehlgeschlagen")
        raise HTTPException(status_code=500, detail="Interne Service-Fehler") from e


@router.get("/orchestrations/{orchestration_id}/progress", response_model=ProgressResponseModel)
async def get_orchestration_progress(orchestration_id: str) -> ProgressResponseModel:
    """Holt Orchestration-Progress.

    Args:
        orchestration_id: Orchestration-ID

    Returns:
        Progress-Informationen

    Raises:
        HTTPException: Wenn Orchestration nicht gefunden
    """
    try:
        service = get_orchestrator_service()
        progress = await service.get_orchestration_progress(orchestration_id)

        if not progress:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Orchestration {orchestration_id} nicht gefunden"
            )

        return ProgressResponseModel(**progress)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Progress-Abfrage fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Interne Service-Fehler")


@router.delete("/orchestrations/{orchestration_id}")
async def cancel_orchestration(orchestration_id: str) -> dict[str, Any]:
    """Bricht Orchestration ab.

    Args:
        orchestration_id: Orchestration-ID

    Returns:
        Cancellation-Status
    """
    try:
        service = get_orchestrator_service()
        success = await service.cancel_orchestration(orchestration_id)

        return {
            "success": success,
            "orchestration_id": orchestration_id,
            "message": "Orchestration abgebrochen" if success else "Orchestration-Abbruch fehlgeschlagen"
        }

    except Exception as e:
        logger.exception(f"Orchestration-Abbruch fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Interne Service-Fehler")


@router.get("/health", response_model=HealthResponseModel)
async def get_service_health() -> HealthResponseModel:
    """Führt Service-Health-Check durch.

    Returns:
        Health-Check-Ergebnis
    """
    try:
        service = get_orchestrator_service()
        health = await service.get_service_health()

        return HealthResponseModel(
            service_healthy=health.service_healthy,
            service_version=health.service_version,
            uptime_seconds=health.uptime_seconds,
            components={
                "task_decomposition": health.task_decomposition_healthy,
                "performance_prediction": health.performance_prediction_healthy,
                "agent_registry": health.agent_registry_healthy,
                "task_manager": health.task_manager_healthy
            },
            active_orchestrations=health.active_orchestrations,
            total_orchestrations=health.total_orchestrations,
            avg_orchestration_time_ms=health.avg_orchestration_time_ms,
            success_rate=health.success_rate,
            memory_usage_mb=health.memory_usage_mb,
            cpu_usage_percent=health.cpu_usage_percent,
            check_timestamp=health.check_timestamp
        )

    except Exception as e:
        logger.exception(f"Health-Check fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Health-Check fehlgeschlagen")


@router.get("/metrics", response_model=MetricsResponseModel)
async def get_service_metrics() -> MetricsResponseModel:
    """Holt Service-Metriken.

    Returns:
        Service-Metriken
    """
    try:
        service = get_orchestrator_service()
        metrics = await service.get_service_metrics()

        return MetricsResponseModel(**metrics)

    except Exception as e:
        logger.exception(f"Metriken-Abfrage fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Metriken-Abfrage fehlgeschlagen")


@router.websocket("/ws/orchestrations/{orchestration_id}")
async def websocket_orchestration_monitoring(websocket: WebSocket, orchestration_id: str):
    """WebSocket für Real-time Orchestration-Monitoring.

    Args:
        websocket: WebSocket-Connection
        orchestration_id: Orchestration-ID für Monitoring
    """
    await websocket.accept()

    try:
        service = get_orchestrator_service()

        # Registriere WebSocket-Connection
        await service.monitor.add_websocket_connection(websocket)

        logger.info(f"WebSocket-Connection für Orchestration {orchestration_id} etabliert")

        # Halte Connection offen
        while True:
            try:
                # Warte auf Messages vom Client (Ping/Pong)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                # Sende Ping
                await websocket.send_text('{"type": "ping"}')

    except WebSocketDisconnect:
        logger.info(f"WebSocket-Connection für Orchestration {orchestration_id} getrennt")
    except Exception as e:
        logger.exception(f"WebSocket-Fehler: {e}")
    finally:
        # Entferne WebSocket-Connection
        try:
            service = get_orchestrator_service()
            await service.monitor.remove_websocket_connection(websocket)
        except Exception:
            pass
