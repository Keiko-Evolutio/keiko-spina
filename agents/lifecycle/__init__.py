# backend/agents/lifecycle/__init__.py
"""Agent-Lifecycle-Management.

Implementiert vollständiges Lifecycle-Management mit State-Machine,
Task-Handling, Event-Processing und Suspend/Resume-Mechanismen.
"""

from __future__ import annotations

from kei_logging import get_logger

from .agent_lifecycle_manager import (
    AgentLifecycleManager,
    agent_lifecycle_manager,
)
from .enhanced_base_agent import EnhancedBaseAgent
from .models import (
    AgentEvent,
    AgentLifecycleState,
    AgentState,
    AgentTask,
    BackpressureStrategy,
    EventHandler,
    EventSubscription,
    EventType,
    LifecycleCallback,
    LifecycleTransition,
    PriorityTask,
    TaskExecutionResult,
    TaskHandler,
    TaskPriority,
    TaskQueueConfig,
    TaskStatus,
)
from .task_event_handler import (
    EventBus,
    TaskQueue,
)

logger = get_logger(__name__)

__all__ = [
    "AgentEvent",
    "AgentLifecycleManager",
    "AgentLifecycleState",
    "AgentState",
    "AgentTask",
    "BackpressureStrategy",
    "EnhancedBaseAgent",
    "EventBus",
    "EventHandler",
    "EventSubscription",
    "EventType",
    "LifecycleCallback",
    "LifecycleTransition",
    "PriorityTask",
    "TaskExecutionResult",
    "TaskHandler",
    "TaskPriority",
    "TaskQueue",
    "TaskQueueConfig",
    "TaskStatus",
    "agent_lifecycle_manager",
]


def get_lifecycle_system_status() -> dict:
    """Gibt Status des Lifecycle-Systems zurück."""
    return {
        "package": "agents.lifecycle",
        "version": "1.0.0",
        "components": {
            "lifecycle_manager": True,
            "task_queue": True,
            "event_bus": True,
            "enhanced_base_agent": True,
        },
        "features": {
            "state_machine": True,
            "task_handling": True,
            "event_processing": True,
            "suspend_resume": True,
            "capability_advertisement": True,
            "heartbeat_monitoring": True,
            "graceful_termination": True,
            "backpressure_handling": True,
            "priority_queuing": True,
            "retry_logic": True,
        },
        "lifecycle_states": [state.value for state in AgentLifecycleState],
        "task_priorities": [priority.value for priority in TaskPriority],
        "event_types": [event_type.value for event_type in EventType],
    }


logger.info(f"Agent-Lifecycle-System geladen - Status: {get_lifecycle_system_status()}")
