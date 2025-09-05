# backend/services/orchestrator/__init__.py
"""Orchestrator Service Package.

Eigenständiger Service für intelligente Task-Orchestration mit
LLM-Integration, Performance-Optimierung und Real-time Monitoring.
"""

from __future__ import annotations

# Kommentiere problematische Imports aus bis sklearn-Problem gelöst ist
# from .api import (
#     HealthResponseModel,
#     MetricsResponseModel,
#     OrchestrationRequestModel,
#     OrchestrationResponseModel,
#     ProgressResponseModel,
#     get_orchestrator_service,
#     initialize_orchestrator_service,
#     router as orchestrator_router,
#     shutdown_orchestrator_service
# )
# Importiere nur funktionsfähige Data Models
from .data_models import (
    AgentAssignmentStatus,
    ExecutionMode,
    # Auskommentiert bis sklearn-Problem gelöst:
    # AgentLoadInfo,
    # ExecutionPlan,
    # HealthCheckResult,
    # OrchestrationEvent,
    # OrchestrationProgress,
    # OrchestrationRequest,
    # OrchestrationResult,
    # OrchestrationState,
    # SubtaskExecution
)

# Importiere nur den funktionsfähigen EnterpriseOrchestratorService
from .enterprise_orchestrator_service import EnterpriseOrchestratorService

# Auskommentiert bis sklearn-Problem gelöst:
from .execution_engine import ExecutionEngine
from .monitoring import OrchestrationMonitor
from .orchestrator_service import OrchestratorService

__all__ = [
    "AgentAssignmentStatus",
    # Enterprise Orchestrator Service (funktionsfähig)
    "EnterpriseOrchestratorService",
    "ExecutionEngine",
    # Data Models (funktionsfähig)
    "ExecutionMode",
    "OrchestrationMonitor",
    # Core Service
    "OrchestratorService",

    # Data Models
    # "OrchestrationRequest",
    # "OrchestrationResult",
    # "OrchestrationProgress",
    # "OrchestrationEvent",
    # "SubtaskExecution",
    # "ExecutionPlan",
    # "AgentLoadInfo",
    # "HealthCheckResult",
    # "OrchestrationState",

    # API Models
    # "OrchestrationRequestModel",
    # "OrchestrationResponseModel",
    # "ProgressResponseModel",
    # "HealthResponseModel",
    # "MetricsResponseModel",

    # API Functions
    # "get_orchestrator_service",
    # "initialize_orchestrator_service",
    # "shutdown_orchestrator_service",
    # "orchestrator_router",

    # Factory Functions
    # "create_orchestrator_service",
    # "create_execution_engine",
    # "create_orchestration_monitor",
]

__version__ = "1.0.0"


def create_orchestrator_service(
    task_manager=None,
    agent_registry=None,
    decomposition_engine=None,
    performance_predictor=None
) -> OrchestratorService:
    """Factory-Funktion für Orchestrator Service.

    Args:
        task_manager: Task Manager Instanz
        agent_registry: Agent Registry Instanz
        decomposition_engine: Task Decomposition Engine aus TASK 3
        performance_predictor: Performance Predictor aus TASK 2

    Returns:
        Konfigurierter Orchestrator Service
    """
    return OrchestratorService(
        task_manager=task_manager,
        agent_registry=agent_registry,
        decomposition_engine=decomposition_engine,
        performance_predictor=performance_predictor
    )


def create_execution_engine(
    task_manager,
    agent_registry,
    decomposition_engine,
    performance_predictor=None,
    monitor=None
) -> ExecutionEngine:
    """Factory-Funktion für Execution Engine.

    Args:
        task_manager: Task Manager Instanz
        agent_registry: Agent Registry Instanz
        decomposition_engine: Task Decomposition Engine
        performance_predictor: Performance Predictor (optional)
        monitor: Orchestration Monitor (optional)

    Returns:
        Konfigurierte Execution Engine
    """
    return ExecutionEngine(
        task_manager=task_manager,
        agent_registry=agent_registry,
        decomposition_engine=decomposition_engine,
        performance_predictor=performance_predictor,
        monitor=monitor
    )


def create_orchestration_monitor(bus_service=None) -> OrchestrationMonitor:
    """Factory-Funktion für Orchestration Monitor.

    Args:
        bus_service: Message Bus Service (optional)

    Returns:
        Konfigurierter Orchestration Monitor
    """
    return OrchestrationMonitor(bus_service=bus_service)
