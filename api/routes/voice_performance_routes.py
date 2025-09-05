"""Voice Performance API Routes.
Admin-APIs für Voice Performance Optimization Management und Monitoring.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.container import get_container
from kei_logging import get_logger
from voice_performance.interfaces import ProcessingStage, VoiceWorkflowContext
from voice_performance.service import VoicePerformanceService

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/voice-performance", tags=["voice-performance"])


# Pydantic Models für API
class VoiceOptimizationRequest(BaseModel):
    """Request Model für Voice Optimization."""
    text: str | None = None
    audio_data: bytes | None = None
    user_id: str = "unknown"
    session_id: str = "unknown"
    language: str = "de-DE"
    max_latency_ms: int = 500
    parallel_processing: bool = True
    cache_enabled: bool = True
    priority: int = 0


class VoiceWorkflowRequest(BaseModel):
    """Request Model für Voice Workflow."""
    workflow_id: str | None = None
    text_input: str | None = None
    user_id: str = "unknown"
    session_id: str = "unknown"
    language: str = "de-DE"
    max_latency_ms: int = 500
    parallel_processing: bool = True
    cache_enabled: bool = True
    min_confidence: float = 0.7
    max_agents: int = 5
    timeout_seconds: float = 30.0
    priority: int = 0


class ConcurrentVoiceRequest(BaseModel):
    """Request Model für Concurrent Voice Processing."""
    requests: list[VoiceOptimizationRequest]
    max_concurrent: int | None = None


class OrchestratorOptimizationRequest(BaseModel):
    """Request Model für Orchestrator Optimization."""
    task_id: str | None = None
    text: str
    user_id: str = "unknown"
    session_id: str = "unknown"
    framework: str = "default"
    agent_id: str = "orchestrator"


def get_voice_performance_service() -> VoicePerformanceService:
    """Dependency für Voice Performance Service."""
    try:
        container = get_container()
        return container.resolve(VoicePerformanceService)
    except Exception as e:
        logger.error(f"Failed to resolve voice performance service: {e}")
        raise HTTPException(status_code=503, detail="Voice performance service not available")


@router.get("/status", summary="Voice Performance Service Status")
async def get_service_status(
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt Status des Voice Performance Services zurück.

    Returns:
        Dict mit Service-Status und Statistiken
    """
    try:
        return await service.get_service_statistics()
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@router.post("/optimize", summary="Optimize Voice Request")
async def optimize_voice_request(
    request: VoiceOptimizationRequest,
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Optimiert Voice Request mit Performance-Optimierung.

    Args:
        request: Voice Optimization Request
        service: Voice Performance Service für Optimierung

    Returns:
        Dict mit optimierter Response
    """
    try:
        # Konvertiere zu Voice Request Format
        voice_request = {
            "text": request.text,
            "audio_data": request.audio_data,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "language": request.language,
            "max_latency_ms": request.max_latency_ms,
            "parallel_processing": request.parallel_processing,
            "cache_enabled": request.cache_enabled,
            "priority": request.priority
        }

        # Führe Optimization durch
        result = await service.optimize_voice_request(voice_request)

        return result

    except Exception as e:
        logger.error(f"Voice request optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Voice request optimization failed")


@router.post("/workflow", summary="Process Voice Workflow")
async def process_voice_workflow(
    request: VoiceWorkflowRequest,
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Verarbeitet Voice Workflow mit vollständiger Performance-Optimierung.

    Args:
        request: Voice Workflow Request
        service: Voice Performance Service für Workflow-Verarbeitung

    Returns:
        Dict mit Workflow-Ergebnis
    """
    try:
        # Erstelle VoiceWorkflowContext
        context = VoiceWorkflowContext(
            workflow_id=request.workflow_id or str(uuid.uuid4()),
            user_id=request.user_id,
            session_id=request.session_id,
            text_input=request.text_input,
            language=request.language,
            max_latency_ms=request.max_latency_ms,
            parallel_processing=request.parallel_processing,
            cache_enabled=request.cache_enabled,
            min_confidence=request.min_confidence,
            max_agents=request.max_agents,
            timeout_seconds=request.timeout_seconds,
            priority=request.priority
        )

        # Führe Workflow Optimization durch
        result = await service.optimize_voice_workflow(context)

        return {
            "workflow_id": result.workflow_id,
            "stage": result.stage.value,
            "success": result.success,
            "results": result.results,
            "errors": result.errors,
            "performance_metrics": {
                "total_time_ms": result.total_time_ms,
                "parallel_time_ms": result.parallel_time_ms,
                "sequential_time_ms": result.sequential_time_ms,
                "speedup_factor": result.speedup_factor,
                "max_concurrent_tasks": result.max_concurrent_tasks,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "success_rate": result.success_rate,
                "average_confidence": result.average_confidence
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice workflow processing failed: {e}")
        raise HTTPException(status_code=500, detail="Voice workflow processing failed")


@router.post("/concurrent", summary="Handle Concurrent Voice Requests")
async def handle_concurrent_requests(
    request: ConcurrentVoiceRequest,
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Behandelt mehrere Voice Requests concurrent.

    Args:
        request: Concurrent Voice Request
        service: Voice Performance Service für Concurrent Processing

    Returns:
        Dict mit Concurrent Processing Results
    """
    try:
        # Konvertiere zu Voice Request Format
        voice_requests = []
        for i, voice_req in enumerate(request.requests):
            voice_request = {
                "workflow_id": f"concurrent_{i}_{uuid.uuid4()}",
                "text": voice_req.text,
                "audio_data": voice_req.audio_data,
                "user_id": voice_req.user_id,
                "session_id": voice_req.session_id,
                "language": voice_req.language,
                "max_latency_ms": voice_req.max_latency_ms,
                "parallel_processing": voice_req.parallel_processing,
                "cache_enabled": voice_req.cache_enabled,
                "priority": voice_req.priority
            }
            voice_requests.append(voice_request)

        # Führe Concurrent Processing durch
        results = await service.handle_concurrent_voice_requests(voice_requests)

        return {
            "total_requests": len(request.requests),
            "successful_requests": len([r for r in results if r.get("success", False)]),
            "failed_requests": len([r for r in results if not r.get("success", False)]),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Concurrent request handling failed: {e}")
        raise HTTPException(status_code=500, detail="Concurrent request handling failed")


@router.post("/orchestrator", summary="Optimize Orchestrator Request")
async def optimize_orchestrator_request(
    request: OrchestratorOptimizationRequest,
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Optimiert Orchestrator Request mit Performance-Optimierung.

    Args:
        request: Orchestrator Optimization Request
        service: Voice Performance Service für Orchestrator-Optimierung

    Returns:
        Dict mit optimierter Orchestrator Response
    """
    try:
        # Konvertiere zu Orchestrator Request Format
        orchestrator_request = {
            "task_id": request.task_id or str(uuid.uuid4()),
            "text": request.text,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "framework": request.framework,
            "agent_id": request.agent_id
        }

        # Führe Orchestrator Optimization durch
        result = await service.optimize_orchestrator_request(orchestrator_request)

        return result

    except Exception as e:
        logger.error(f"Orchestrator request optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Orchestrator request optimization failed")


@router.get("/health", summary="Voice Performance Health Check")
async def health_check(
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Führt Health Check für Voice Performance Service durch.

    Returns:
        Dict mit Health-Status
    """
    try:
        health = await service.health_check()

        if not health["healthy"]:
            raise HTTPException(status_code=503, detail="Voice performance service unhealthy")

        return health
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/metrics", summary="Voice Performance Metrics")
async def get_performance_metrics(
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt detaillierte Performance-Metriken zurück.

    Returns:
        Dict mit Performance-Metriken
    """
    try:
        return await service.get_performance_metrics()
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@router.get("/statistics", summary="Voice Performance Statistics")
async def get_service_statistics(
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Gibt Service-Statistiken zurück.

    Returns:
        Dict mit Service-Statistiken
    """
    try:
        return await service.get_service_statistics()
    except Exception as e:
        logger.error(f"Failed to get service statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service statistics")


@router.get("/processing-stages", summary="List Processing Stages")
async def list_processing_stages() -> list[dict[str, str]]:
    """Gibt Liste aller Processing Stages zurück.

    Returns:
        Liste mit Processing Stages
    """
    return [
        {"value": stage.value, "name": stage.name}
        for stage in ProcessingStage
    ]


@router.post("/test/latency", summary="Test Voice Latency")
async def test_voice_latency(
    text: str = Query(..., description="Text to process"),
    iterations: int = Query(10, description="Number of test iterations"),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Testet Voice Processing Latency.

    Args:
        text: Text für Processing
        iterations: Anzahl Test-Iterationen

    Returns:
        Dict mit Latency-Test-Ergebnissen
    """
    try:
        import time

        latencies = []

        for i in range(iterations):
            # Erstelle Test-Context
            context = VoiceWorkflowContext(
                workflow_id=f"latency_test_{i}",
                user_id="test_user",
                session_id="test_session",
                text_input=text,
                parallel_processing=True,
                cache_enabled=True
            )

            # Messe Latency
            start_time = time.time()
            result = await service.optimize_voice_workflow(context)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append({
                "iteration": i + 1,
                "latency_ms": latency_ms,
                "success": result.success,
                "speedup_factor": result.speedup_factor
            })

        # Berechne Statistiken
        successful_latencies = [l["latency_ms"] for l in latencies if l["success"]]

        if successful_latencies:
            avg_latency = sum(successful_latencies) / len(successful_latencies)
            min_latency = min(successful_latencies)
            max_latency = max(successful_latencies)
        else:
            avg_latency = min_latency = max_latency = 0.0

        return {
            "test_parameters": {
                "text": text,
                "iterations": iterations
            },
            "results": {
                "total_iterations": iterations,
                "successful_iterations": len(successful_latencies),
                "failed_iterations": iterations - len(successful_latencies),
                "average_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "success_rate": len(successful_latencies) / iterations if iterations > 0 else 0.0
            },
            "detailed_results": latencies,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Latency test failed: {e}")
        raise HTTPException(status_code=500, detail="Latency test failed")


@router.post("/test/throughput", summary="Test Voice Throughput")
async def test_voice_throughput(
    text: str = Query(..., description="Text to process"),
    concurrent_requests: int = Query(10, description="Number of concurrent requests"),
    service: VoicePerformanceService = Depends(get_voice_performance_service)
) -> dict[str, Any]:
    """Testet Voice Processing Throughput.

    Args:
        text: Text für Processing
        concurrent_requests: Anzahl concurrent Requests

    Returns:
        Dict mit Throughput-Test-Ergebnissen
    """
    try:
        import time

        # Erstelle Concurrent Requests
        voice_requests = []
        for i in range(concurrent_requests):
            voice_request = {
                "workflow_id": f"throughput_test_{i}",
                "text": text,
                "user_id": f"test_user_{i}",
                "session_id": f"test_session_{i}",
                "parallel_processing": True,
                "cache_enabled": True
            }
            voice_requests.append(voice_request)

        # Messe Throughput
        start_time = time.time()
        results = await service.handle_concurrent_voice_requests(voice_requests)
        end_time = time.time()

        total_time_seconds = end_time - start_time
        successful_requests = len([r for r in results if r.get("success", False)])
        throughput_rps = successful_requests / total_time_seconds if total_time_seconds > 0 else 0.0

        return {
            "test_parameters": {
                "text": text,
                "concurrent_requests": concurrent_requests
            },
            "results": {
                "total_requests": concurrent_requests,
                "successful_requests": successful_requests,
                "failed_requests": concurrent_requests - successful_requests,
                "total_time_seconds": total_time_seconds,
                "throughput_rps": throughput_rps,
                "success_rate": successful_requests / concurrent_requests if concurrent_requests > 0 else 0.0
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Throughput test failed: {e}")
        raise HTTPException(status_code=500, detail="Throughput test failed")
