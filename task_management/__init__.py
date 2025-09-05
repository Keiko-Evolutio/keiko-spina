# backend/task_management/__init__.py
"""Vollständige Task Management API für Keiko Personal Assistant

Implementiert Enterprise-Grade Task-Management mit Idempotenz, Korrelation,
vollständigem Lifecycle-Management und Integration mit bestehenden Systemen.
"""

from __future__ import annotations

from kei_logging import get_logger

# Core Task Management
from .core_task_manager import (
    Task,
    TaskDependency,
    TaskExecutionContext,
    TaskManager,
    TaskPriority,
    TaskResult,
    TaskSchedule,
    TaskState,
    TaskType,
    task_manager,
)

# Idempotenz und Korrelation
from .idempotency_manager import (
    CorrelationID,
    DuplicateDetectionResult,
    IdempotencyConfig,
    IdempotencyKey,
    IdempotencyManager,
    RequestCache,
    idempotency_manager,
)

# KEI-Bus Integration
from .kei_bus_integration import (
    DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_QUEUE_VERSION,
    KEIBusTaskWorker,
    create_kei_bus_task_worker,
)

# Task API Implementation
from .task_api import (
    CancelTaskRequest,
    RetryTaskRequest,
    SubmitTaskRequest,
    SubmitTaskResponse,
    TaskAPI,
    TaskFilter,
    TaskListResponse,
    TaskStatusResponse,
    UpdateTaskRequest,
    task_api,
)

# Task Execution Engine
from .task_execution_engine import (
    ExecutionResult,
    TaskExecutionEngine,
    TaskQueue,
    TaskScheduler,
    TaskWorker,
    TaskWorkerPool,
    WorkerMetrics,
    task_execution_engine,
)

# Integration Layer
from .task_integration import (
    AuditIntegration,
    IntegrationConfig,
    PolicyIntegration,
    QuotaIntegration,
    SecurityIntegration,
    TaskIntegrationManager,
    task_integration_manager,
)

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "DEFAULT_MAX_IN_FLIGHT",
    "DEFAULT_QUEUE_VERSION",
    "AuditIntegration",
    "CancelTaskRequest",
    "CorrelationID",
    "DuplicateDetectionResult",
    "ExecutionResult",
    "IdempotencyConfig",
    "IdempotencyKey",
    # Idempotenz und Korrelation
    "IdempotencyManager",
    "IntegrationConfig",
    # KEI-Bus Integration
    "KEIBusTaskWorker",
    "PolicyIntegration",
    "QuotaIntegration",
    "RequestCache",
    "RetryTaskRequest",
    "SecurityIntegration",
    "SubmitTaskRequest",
    "SubmitTaskResponse",
    "Task",
    # Task API Implementation
    "TaskAPI",
    "TaskDependency",
    "TaskExecutionContext",
    # Task Execution Engine
    "TaskExecutionEngine",
    "TaskFilter",
    # Integration Layer
    "TaskIntegrationManager",
    "TaskListResponse",
    # Core Task Management
    "TaskManager",
    "TaskPriority",
    "TaskQueue",
    "TaskResult",
    "TaskSchedule",
    "TaskScheduler",
    "TaskState",
    "TaskStatusResponse",
    "TaskType",
    "TaskWorker",
    "TaskWorkerPool",
    "UpdateTaskRequest",
    "WorkerMetrics",
    "create_kei_bus_task_worker",
    "idempotency_manager",
    "task_api",
    "task_execution_engine",
    "task_integration_manager",
    "task_manager",


]

# Task-Management-System Status
def get_task_management_status() -> dict:
    """Gibt Status des Task-Management-Systems zurück."""
    return {
        "package": "backend.task_management",
        "version": "1.0.0",
        "components": {
            "core_task_manager": True,
            "task_api": True,
            "idempotency_manager": True,
            "task_lifecycle": True,
            "task_execution_engine": True,
            "task_performance": True,
            "task_integration": True,
            "task_monitoring": True,
        },
        "features": {
            "idempotent_api_calls": True,
            "correlation_tracking": True,
            "duplicate_detection": True,
            "task_lifecycle_management": True,
            "state_transitions": True,
            "task_dependencies": True,
            "task_scheduling": True,
            "timeout_management": True,
            "automatic_cleanup": True,
            "async_execution": True,
            "worker_pools": True,
            "load_balancing": True,
            "priority_queues": True,
            "horizontal_scaling": True,
            "security_integration": True,
            "policy_integration": True,
            "quota_integration": True,
            "audit_integration": True,
            "real_time_monitoring": True,
            "performance_analytics": True,
        },
        "task_states": [
            "pending",
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
            "timeout",
            "retrying"
        ],
        "task_priorities": [
            "low",
            "normal",
            "high",
            "critical",
            "urgent"
        ],
        "task_types": [
            "agent_execution",
            "data_processing",
            "nlp_analysis",
            "tool_call",
            "workflow",
            "batch_job",
            "scheduled_task",
            "system_maintenance"
        ],
        "integration_points": [
            "enhanced_security",
            "policy_engine",
            "quotas_limits",
            "audit_system",
            "observability",
            "kei_bus",
            "agent_lifecycle"
        ],
        "api_endpoints": [
            "POST /api/v1/tasks/submit",
            "POST /api/v1/tasks/{task_id}/cancel",
            "GET /api/v1/tasks/{task_id}/status",
            "GET /api/v1/tasks",
            "PUT /api/v1/tasks/{task_id}",
            "POST /api/v1/tasks/{task_id}/retry"
        ],
        "performance_features": [
            "async_processing",
            "worker_pools",
            "load_balancing",
            "priority_queues",
            "batch_processing",
            "horizontal_scaling",
            "circuit_breakers",
            "backpressure_handling"
        ]
    }

logger.info(f"Task Management System geladen - Status: {get_task_management_status()}")
