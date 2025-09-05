"""Dead Letter Queue Admin API Routes.
Management APIs für Voice-Workflow Failed Task Management.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from core.container import get_container
from kei_logging import get_logger

# Import Pydantic Models
from voice_performance.models import (
    AnalyticsRequest,
    CreateFailedTaskRequest,
    CreateFailedTaskResponse,
    DLQStatusResponse,
    FailureReason,
    RecoveryRequest,
    RecoveryStrategy,
    TaskStatus,
)
from voice_performance.service import VoicePerformanceService

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/dlq-admin", tags=["dlq-admin"])


# Note: Pydantic Models sind jetzt in voice_performance.models definiert


def get_voice_performance_service() -> VoicePerformanceService:
    """Dependency für Voice Performance Service."""
    try:
        container = get_container()
        return container.resolve(VoicePerformanceService)
    except Exception as e:
        logger.error(f"Failed to resolve voice performance service: {e}")
        raise HTTPException(status_code=503, detail="Voice performance service not available")


@router.get("/status", summary="DLQ Status", response_model=DLQStatusResponse)
async def get_dlq_status(
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> DLQStatusResponse:
    """Gibt Status der Dead Letter Queue zurück.

    Returns:
        Dict mit DLQ-Status und Statistiken
    """
    try:
        if not service._voice_dlq:
            raise HTTPException(status_code=503, detail="DLQ not initialized")

        dlq_stats = await service._voice_dlq.get_statistics()
        voice_analytics = await service._voice_dlq.get_voice_failure_analytics()

        return DLQStatusResponse(
            success=True,
            dlq_status="operational",
            statistics=dlq_stats,
            voice_analytics=voice_analytics
        )

    except Exception as e:
        logger.error(f"Failed to get DLQ status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get DLQ status")


@router.get("/failed-tasks", summary="List Failed Tasks")
async def list_failed_tasks(
    task_type: str | None = Query(None, description="Filter by task type"),
    status: str | None = Query(None, description="Filter by status"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    limit: int = Query(100, description="Maximum number of tasks to return"),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt Liste der Failed Tasks zurück.

    Args:
        task_type: Optional task type filter
        status: Optional status filter
        user_id: Optional user ID filter
        limit: Maximum number of tasks
        service: Voice Performance Service für DLQ-Zugriff

    Returns:
        Dict mit Failed Tasks
    """
    try:
        if not service._voice_dlq:
            raise HTTPException(status_code=503, detail="DLQ not initialized")

        # Konvertiere Status String zu Enum
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        # Hole Failed Tasks
        failed_tasks = await service._voice_dlq.get_failed_tasks(
            task_type=task_type,
            status=status_filter,
            limit=limit
        )

        # Filtere nach User ID falls angegeben
        if user_id:
            failed_tasks = [task for task in failed_tasks if task.user_id == user_id]

        # Konvertiere zu API-Format
        tasks_data = []
        for task in failed_tasks:
            task_data = {
                "task_id": task.task_id,
                "workflow_id": task.workflow_id,
                "task_type": task.task_type,
                "failure_reason": task.failure_reason.value,
                "failure_category": task.failure_category,
                "error_message": task.error_message,
                "status": task.status.value,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "priority": task.priority,
                "criticality": task.criticality,
                "user_id": task.user_id,
                "session_id": task.session_id,
                "failed_at": task.failed_at.isoformat(),
                "next_retry_at": task.next_retry_at.isoformat() if task.next_retry_at else None,
                "voice_context": {
                    "voice_input": task.voice_context.voice_input if task.voice_context else None,
                    "language": task.voice_context.language if task.voice_context else None,
                    "intent": task.voice_context.intent if task.voice_context else None,
                    "confidence_score": task.voice_context.confidence_score if task.voice_context else 0.0
                } if task.voice_context else None
            }
            tasks_data.append(task_data)

        return {
            "total_tasks": len(tasks_data),
            "tasks": tasks_data,
            "filters": {
                "task_type": task_type,
                "status": status,
                "user_id": user_id,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list failed tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to list failed tasks")


@router.post("/failed-tasks", summary="Add Failed Task", response_model=CreateFailedTaskResponse)
async def add_failed_task(
    request: CreateFailedTaskRequest,
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> CreateFailedTaskResponse:
    """Fügt Failed Task zur DLQ hinzu.

    Args:
        request: Failed Task Request
        service: Voice Performance Service für DLQ-Zugriff

    Returns:
        Dict mit Task-ID und Status
    """
    try:
        if not service._voice_dlq:
            raise HTTPException(status_code=503, detail="DLQ not initialized")

        # Konvertiere Failure Reason String zu Enum
        try:
            failure_reason = FailureReason(request.failure_reason)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid failure reason: {request.failure_reason}")

        # Erstelle Voice Context
        from voice_performance.interfaces import VoiceWorkflowContext
        voice_context = VoiceWorkflowContext(
            workflow_id=str(uuid.uuid4()),
            user_id=request.user_id or "unknown",
            session_id=request.session_id or "unknown",
            text_input=request.voice_input or "",
            language=request.language
        )

        # Generiere Task ID
        task_id = str(uuid.uuid4())

        # Füge Failed Task hinzu
        await service._voice_dlq.add_voice_failed_task(
            task_id=task_id,
            workflow_id=request.workflow_id,
            task_type=request.task_type,
            failure_reason=failure_reason,
            error_message=request.error_message,
            voice_context=voice_context,
            original_request={
                "voice_input": request.voice_input,
                "language": request.language
            },
            user_id=request.user_id,
            session_id=request.session_id,
            priority=request.priority,
            criticality=request.criticality
        )

        return CreateFailedTaskResponse(
            success=True,
            task_id=task_id,
            workflow_id=request.workflow_id,
            message="Failed task added to DLQ",
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add failed task: {e}")
        raise HTTPException(status_code=500, detail="Failed to add failed task")


@router.post("/recovery", summary="Recover Failed Tasks")
async def recover_failed_tasks(
    request: RecoveryRequest,
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Führt Recovery für Failed Tasks durch.

    Args:
        request: Recovery Request
        service: Voice Performance Service für Recovery-Operationen

    Returns:
        Dict mit Recovery-Ergebnissen
    """
    try:
        if not service._recovery_system:
            raise HTTPException(status_code=503, detail="Recovery system not initialized")

        # Single Task Recovery
        if request.task_id:
            strategy_override = None
            if request.recovery_strategy:
                try:
                    strategy_override = RecoveryStrategy(request.recovery_strategy)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid recovery strategy: {request.recovery_strategy}")

            success = await service._recovery_system.recover_failed_task(
                request.task_id,
                strategy_override
            )

            return {
                "success": success,
                "task_id": request.task_id,
                "recovery_strategy": request.recovery_strategy,
                "message": "Task recovery completed",
                "timestamp": datetime.utcnow().isoformat()
            }

        # Workflow Recovery
        if request.workflow_id:
            result = await service._recovery_system.recover_voice_workflow(
                request.workflow_id,
                request.recovery_mode
            )

            return {
                "success": result.get("success", False),
                "workflow_id": request.workflow_id,
                "recovery_mode": request.recovery_mode,
                "recovery_details": result,
                "timestamp": datetime.utcnow().isoformat()
            }

        # Session Recovery
        if request.session_id:
            success = await service._recovery_system.implement_session_continuity(
                request.session_id,
                "seamless"
            )

            return {
                "success": success,
                "session_id": request.session_id,
                "continuity_strategy": "seamless",
                "message": "Session continuity implemented",
                "timestamp": datetime.utcnow().isoformat()
            }

        raise HTTPException(status_code=400, detail="Either task_id, workflow_id, or session_id must be provided")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        raise HTTPException(status_code=500, detail="Recovery failed")


@router.get("/analytics", summary="DLQ Analytics")
async def get_dlq_analytics(
    request: AnalyticsRequest = Depends(),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt DLQ Analytics zurück.

    Args:
        request: Analytics Request
        service: Voice Performance Service für Analytics-Zugriff

    Returns:
        Dict mit Analytics-Daten
    """
    try:
        if not service._dlq_analytics:
            raise HTTPException(status_code=503, detail="DLQ analytics not initialized")

        # Failure Trends
        failure_trends = await service._dlq_analytics.analyze_failure_trends(
            request.time_window_hours
        )

        result = {
            "failure_trends": failure_trends,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Pattern Detection
        if request.include_patterns:
            patterns = await service._dlq_analytics.detect_failure_patterns()
            result["failure_patterns"] = [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "severity": p.severity,
                    "occurrence_count": p.occurrence_count,
                    "frequency_per_hour": p.frequency_per_hour,
                    "first_seen": p.first_seen.isoformat(),
                    "last_seen": p.last_seen.isoformat()
                }
                for p in patterns
            ]

        # Performance Impact
        if request.include_performance_impact:
            baseline_metrics = {
                "average_latency_ms": 200.0,
                "throughput_rps": 100.0
            }
            impact = await service._dlq_analytics.analyze_performance_impact(baseline_metrics)
            result["performance_impact"] = {
                "latency_degradation_percent": impact.latency_degradation_percent,
                "throughput_reduction_percent": impact.throughput_reduction_percent,
                "cpu_overhead_percent": impact.cpu_overhead_percent,
                "memory_overhead_mb": impact.memory_overhead_mb
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DLQ analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get DLQ analytics")


@router.get("/recommendations", summary="Recovery Recommendations")
async def get_recovery_recommendations(
    task_id: str | None = Query(None, description="Task ID for specific recommendations"),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt Recovery-Empfehlungen zurück.

    Args:
        task_id: Optional task ID für spezifische Empfehlungen
        service: Voice Performance Service für Empfehlungs-Zugriff

    Returns:
        Dict mit Empfehlungen
    """
    try:
        if not service._dlq_analytics or not service._recovery_system:
            raise HTTPException(status_code=503, detail="Analytics or recovery system not initialized")

        if task_id:
            # Spezifische Task-Empfehlungen
            recommendations = await service._recovery_system.get_recovery_recommendations(task_id)
        else:
            # Allgemeine proaktive Empfehlungen
            recommendations = await service._dlq_analytics.get_proactive_recommendations()

        return {
            "recommendations": recommendations,
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@router.get("/report", summary="Comprehensive Failure Report")
async def generate_failure_report(
    time_window_hours: int = Query(24, description="Time window for report"),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Generiert umfassenden Failure-Report.

    Args:
        time_window_hours: Zeitfenster für Report
        service: Voice Performance Service für Report-Generierung

    Returns:
        Dict mit umfassendem Report
    """
    try:
        if not service._dlq_analytics:
            raise HTTPException(status_code=503, detail="DLQ analytics not initialized")

        report = await service._dlq_analytics.generate_failure_report(time_window_hours)

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate failure report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate failure report")


@router.delete("/failed-tasks/{task_id}", summary="Delete Failed Task")
async def delete_failed_task(
    task_id: str = Path(..., description="Task ID to delete"),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Löscht Failed Task aus DLQ.

    Args:
        task_id: Task ID zum Löschen
        service: Voice Performance Service für DLQ-Zugriff

    Returns:
        Dict mit Lösch-Status
    """
    try:
        if not service._voice_dlq:
            raise HTTPException(status_code=503, detail="DLQ not initialized")

        # Prüfe ob Task existiert
        if task_id not in service._voice_dlq._failed_tasks:
            raise HTTPException(status_code=404, detail="Failed task not found")

        # Lösche Task
        del service._voice_dlq._failed_tasks[task_id]

        return {
            "success": True,
            "task_id": task_id,
            "message": "Failed task deleted",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete failed task: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete failed task")


@router.get("/statistics", summary="DLQ Statistics")
async def get_dlq_statistics(
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt detaillierte DLQ-Statistiken zurück.

    Returns:
        Dict mit DLQ-Statistiken
    """
    try:
        if not service._voice_dlq or not service._recovery_system:
            raise HTTPException(status_code=503, detail="DLQ or recovery system not initialized")

        # DLQ Statistics
        dlq_stats = await service._voice_dlq.get_statistics()

        # Voice Analytics
        voice_analytics = await service._voice_dlq.get_voice_failure_analytics()

        # Recovery Statistics
        recovery_stats = await service._recovery_system.get_recovery_statistics()

        return {
            "dlq_statistics": dlq_stats,
            "voice_analytics": voice_analytics,
            "recovery_statistics": recovery_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DLQ statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get DLQ statistics")
