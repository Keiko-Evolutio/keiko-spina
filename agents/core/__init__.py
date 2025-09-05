# backend/agents/core/__init__.py
"""Core Agent Framework Module.

Stellt die grundlegenden Komponenten für das Agent-Framework bereit:
- BaseAgent: Basis-Klasse für alle Agents
- AgentConfig: Konfiguration für Agents
- ComponentManager: Verwaltung von Framework-Komponenten
- TaskExecutor: Task-Ausführung und -Management
"""

from __future__ import annotations

from .base_agent import (
    AgentConfig,
    AgentMetrics,
    BaseAgent,
)
from .component_manager import (
    ComponentManager,
    ComponentRegistry,
)

# Global component manager instance
_global_component_manager: ComponentManager | None = None

def get_component_manager() -> ComponentManager:
    """Gibt den globalen ComponentManager zurück."""
    global _global_component_manager
    if _global_component_manager is None:
        _global_component_manager = ComponentManager()
    return _global_component_manager
from .task_executor import (
    BaseTaskExecutor,
    CircuitBreakerTaskExecutor,
    RetryTaskExecutor,
    TaskContext,
    TaskExecutionError,
    TaskExecutor,
    TaskResult,
    TaskStatus,
)
from .utils import (
    # Constants
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    # Utilities
    MetricsCollector,
    # Validation
    ValidationError,
    async_error_context,
    async_timeout_context,
    generate_task_id,
    get_module_logger,
    validate_non_empty_string,
    validate_positive_number,
    validate_required_field,
    with_async_error_handling,
    with_error_handling,
)

__all__ = [
    # Base Agent
    "BaseAgent",
    "AgentConfig",
    "AgentMetrics",
    # Component Management
    "ComponentManager",
    "ComponentRegistry",
    "get_component_manager",
    # Task Execution
    "BaseTaskExecutor",
    "CircuitBreakerTaskExecutor",
    "RetryTaskExecutor",
    "TaskContext",
    "TaskExecutionError",
    "TaskExecutor",
    "TaskResult",
    "TaskStatus",
    # Constants
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_MAX_CONCURRENT_TASKS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RECOVERY_TIMEOUT_SECONDS",
    "DEFAULT_TASK_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    # Validation
    "ValidationError",
    "validate_non_empty_string",
    "validate_positive_number",
    "validate_required_field",
    # Utilities
    "MetricsCollector",
    "async_error_context",
    "async_timeout_context",
    "generate_task_id",
    "get_module_logger",
    "with_async_error_handling",
    "with_error_handling",
]

# Version Information
__version__ = "1.0.0"
__author__ = "Agent-Framework Core Team"
