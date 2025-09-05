"""Agent Circuit Breaker API Routes.
Admin-APIs für Agent Circuit Breaker Management und Monitoring.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from agents.circuit_breaker.interfaces import (
    AgentCallContext,
    AgentType,
    FailureType,
    IAgentCircuitBreakerService,
)
from core.container import get_container
from kei_logging import get_logger

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/agent-circuit-breaker", tags=["agent-circuit-breaker"])


# Pydantic Models für API
class CircuitBreakerStatusRequest(BaseModel):
    """Request Model für Circuit Breaker Status."""
    agent_id: str
    agent_type: AgentType = AgentType.CUSTOM_AGENT


class CircuitBreakerStateRequest(BaseModel):
    """Request Model für Circuit Breaker State Change."""
    agent_id: str
    agent_type: AgentType = AgentType.CUSTOM_AGENT
    state: str  # "open", "closed", "reset"
    reason: str | None = ""


class AgentExecutionTestRequest(BaseModel):
    """Request Model für Agent Execution Test."""
    agent_id: str
    agent_type: AgentType = AgentType.CUSTOM_AGENT
    framework: str
    task: str
    voice_workflow_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None


def get_agent_circuit_breaker_service() -> IAgentCircuitBreakerService:
    """Dependency für Agent Circuit Breaker Service."""
    try:
        container = get_container()
        return container.resolve(IAgentCircuitBreakerService)
    except Exception as e:
        logger.error(f"Failed to resolve agent circuit breaker service: {e}")
        raise HTTPException(status_code=503, detail="Agent circuit breaker service not available")


@router.get("/status", summary="Agent Circuit Breaker Service Status")
async def get_service_status(
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Gibt Status des Agent Circuit Breaker Services zurück.

    Returns:
        Dict mit Service-Status und Statistiken
    """
    try:
        return await service.get_service_statistics()
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@router.post("/status/agent", summary="Agent-specific Circuit Breaker Status")
async def get_agent_circuit_breaker_status(
    request: CircuitBreakerStatusRequest,
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Gibt Circuit Breaker Status für spezifischen Agent zurück.

    Args:
        request: Agent-Kontext für Status-Abfrage
        service: Agent Circuit Breaker Service für Status-Abfrage

    Returns:
        Dict mit Circuit Breaker Status
    """
    try:
        return await service.get_circuit_breaker_status(request.agent_id, request.agent_type.value)
    except Exception as e:
        logger.error(f"Failed to get agent circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent circuit breaker status")


@router.get("/status/all", summary="All Circuit Breakers Status")
async def get_all_circuit_breakers_status(
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Gibt Status aller Circuit Breaker zurück.

    Returns:
        Dict mit Status aller Circuit Breaker
    """
    try:
        return await service.get_all_circuit_breakers_status()
    except Exception as e:
        logger.error(f"Failed to get all circuit breakers status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get all circuit breakers status")


@router.post("/state/change", summary="Change Circuit Breaker State")
async def change_circuit_breaker_state(
    request: CircuitBreakerStateRequest,
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Ändert Circuit Breaker State (Admin-Funktion).

    Args:
        request: State Change Request
        service: Agent Circuit Breaker Service für State-Änderung

    Returns:
        Dict mit State Change Bestätigung
    """
    try:
        return await service.force_circuit_breaker_state(
            request.agent_id,
            request.agent_type.value,
            request.state,
            request.reason
        )
    except Exception as e:
        logger.error(f"Failed to change circuit breaker state: {e}")
        raise HTTPException(status_code=500, detail="Failed to change circuit breaker state")


@router.post("/test/execution", summary="Test Agent Execution with Circuit Breaker")
async def test_agent_execution(
    request: AgentExecutionTestRequest,
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Testet Agent-Execution mit Circuit Breaker Protection.

    Args:
        request: Test-Parameter

    Returns:
        Dict mit Test-Ergebnis
    """
    try:
        context = AgentCallContext(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            framework=request.framework,
            task=request.task,
            voice_workflow_id=request.voice_workflow_id,
            user_id=request.user_id,
            session_id=request.session_id
        )

        # Mock Agent Execution für Test
        async def mock_agent_execution():
            # Simuliere Agent-Execution
            import asyncio
            await asyncio.sleep(0.1)  # Simuliere Verarbeitungszeit
            return {"result": f"Mock execution for {request.task}", "success": True}

        # Führe mit Circuit Breaker Protection aus
        execution_result = await service.execute_agent_with_protection(
            context, mock_agent_execution
        )

        return {
            "agent_id": request.agent_id,
            "agent_type": request.agent_type.value,
            "framework": request.framework,
            "task": request.task,
            "execution_result": {
                "success": execution_result.success,
                "result": execution_result.result,
                "error": execution_result.error,
                "execution_time_ms": execution_result.execution_time_ms,
                "fallback_used": execution_result.fallback_used,
                "fallback_agent_id": execution_result.fallback_agent_id,
                "failure_type": execution_result.failure_type.value if execution_result.failure_type else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to test agent execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to test agent execution")


@router.get("/health", summary="Agent Circuit Breaker Health Check")
async def health_check(
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Führt Health Check für Agent Circuit Breaker Service durch.

    Returns:
        Dict mit Health-Status
    """
    try:
        health = await service.health_check()

        if not health["healthy"]:
            raise HTTPException(status_code=503, detail="Agent circuit breaker service unhealthy")

        return health
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/statistics", summary="Global Circuit Breaker Statistics")
async def get_global_statistics(
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Gibt globale Circuit Breaker Statistiken zurück.

    Returns:
        Dict mit globalen Statistiken
    """
    try:
        return await service.get_service_statistics()
    except Exception as e:
        logger.error(f"Failed to get global statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get global statistics")


@router.get("/agent-types", summary="List Agent Types")
async def list_agent_types() -> list[dict[str, str]]:
    """Gibt Liste aller Agent-Typen zurück.

    Returns:
        Liste mit Agent-Typen
    """
    return [
        {"value": agent_type.value, "name": agent_type.name}
        for agent_type in AgentType
    ]


@router.get("/failure-types", summary="List Failure Types")
async def list_failure_types() -> list[dict[str, str]]:
    """Gibt Liste aller Failure-Typen zurück.

    Returns:
        Liste mit Failure-Typen
    """
    return [
        {"value": failure_type.value, "name": failure_type.name}
        for failure_type in FailureType
    ]


@router.get("/metrics/{agent_id}", summary="Agent-specific Metrics")
async def get_agent_metrics(
    agent_id: str = Path(..., description="Agent ID"),
    agent_type: str = Query("custom_agent", description="Agent Type"),
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Gibt Agent-spezifische Metriken zurück.

    Args:
        agent_id: Agent ID
        agent_type: Agent Type
        service: Agent Circuit Breaker Service für Metriken-Abfrage

    Returns:
        Dict mit Agent-Metriken
    """
    try:
        status = await service.get_circuit_breaker_status(agent_id, agent_type)

        # Extrahiere relevante Metriken
        metrics = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "state": status.get("state", "unknown"),
            "failure_count": status.get("failure_count", 0),
            "success_count": status.get("success_count", 0),
            "recent_failure_rate": status.get("recent_failure_rate", 0.0),
            "avg_response_time_ms": status.get("avg_response_time_ms", 0.0),
            "uptime_seconds": status.get("uptime_seconds", 0),
            "last_failure_time": status.get("last_failure_time"),
            "last_success_time": status.get("last_success_time"),
            "config": status.get("config", {}),
            "timestamp": datetime.utcnow().isoformat()
        }

        return metrics
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent metrics")


@router.post("/reset/all", summary="Reset All Circuit Breakers")
async def reset_all_circuit_breakers(
    reason: str = Query("Admin reset", description="Reason for reset"),
    service: IAgentCircuitBreakerService = Depends(get_agent_circuit_breaker_service)
) -> dict[str, Any]:
    """Setzt alle Circuit Breaker zurück (Admin-Funktion).

    Args:
        reason: Grund für Reset
        service: Agent Circuit Breaker Service für Reset-Operation

    Returns:
        Dict mit Reset-Bestätigung
    """
    try:
        all_status = await service.get_all_circuit_breakers_status()
        circuit_breakers = all_status.get("circuit_breakers", {})

        reset_results = []
        for cb_name in circuit_breakers.keys():
            try:
                # Extrahiere Agent-ID und Type aus Name
                if ":" in cb_name:
                    agent_type_str, agent_id = cb_name.split(":", 1)

                    result = await service.force_circuit_breaker_state(
                        agent_id, agent_type_str, "reset", reason
                    )
                    reset_results.append({"name": cb_name, "status": "success", "result": result})
                else:
                    reset_results.append({"name": cb_name, "status": "skipped", "reason": "Invalid name format"})
            except Exception as e:
                reset_results.append({"name": cb_name, "status": "error", "error": str(e)})

        return {
            "message": "Circuit breaker reset completed",
            "total_circuit_breakers": len(circuit_breakers),
            "reset_results": reset_results,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset all circuit breakers: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset all circuit breakers")
