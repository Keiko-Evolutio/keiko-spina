# backend/agents/lifecycle/models.py
"""Gemeinsame Datenmodelle für Agent Lifecycle Management.

Enthält alle Enums, Dataclasses und Type-Definitionen, die von
mehreren Lifecycle-Komponenten verwendet werden.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AgentLifecycleState(Enum):
    """Agent-Lifecycle-Zustände."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SUSPENDED = "suspended"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"


class TaskPriority(Enum):
    """Task-Prioritäten."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class EventType(Enum):
    """Event-Typen."""
    LIFECYCLE_CHANGED = "lifecycle_changed"
    TASK_STARTED = "task_started"
    TASK_RECEIVED = "task_received"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    CAPABILITY_ADVERTISED = "capability_advertised"
    HEARTBEAT = "heartbeat"
    ERROR_OCCURRED = "error_occurred"


class TaskStatus(Enum):
    """Task-Status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class BackpressureStrategy(Enum):
    """Backpressure-Strategien."""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    REJECT_NEW = "reject_new"
    BLOCK = "block"
    QUEUE_UNLIMITED = "queue_unlimited"


@dataclass
class AgentTask:
    """Repräsentiert eine Agent-Task."""
    task_id: str
    task_type: str
    agent_id: str
    priority: TaskPriority = TaskPriority.NORMAL
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    error: str | None = None

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Task abgelaufen ist."""
        if not self.started_at:
            return False
        elapsed = datetime.now(UTC) - self.started_at
        return elapsed.total_seconds() > self.timeout_seconds

    @property
    def can_retry(self) -> bool:
        """Prüft, ob Task wiederholt werden kann."""
        return self.retry_count < self.max_retries


@dataclass
class AgentEvent:
    """Repräsentiert ein Agent-Event."""
    event_id: str
    event_type: EventType
    agent_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = None


@dataclass
class LifecycleTransition:
    """Repräsentiert eine Lifecycle-Transition."""
    from_state: AgentLifecycleState
    to_state: AgentLifecycleState
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Repräsentiert den State eines Agents."""
    agent_id: str
    current_state: AgentLifecycleState = AgentLifecycleState.UNREGISTERED
    previous_state: AgentLifecycleState | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    state_changed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    pending_tasks: list[AgentTask] = field(default_factory=list)
    active_tasks: dict[str, AgentTask] = field(default_factory=dict)
    completed_tasks: list[AgentTask] = field(default_factory=list)
    advertised_capabilities: set[str] = field(default_factory=set)
    configuration: dict[str, Any] = field(default_factory=dict)
    suspended_at: datetime | None = None
    suspend_reason: str | None = None
    persisted_state: dict[str, Any] = field(default_factory=dict)
    total_tasks_processed: int = 0
    total_errors: int = 0
    last_heartbeat: datetime | None = None
    transition_history: list[LifecycleTransition] = field(default_factory=list)

    def add_transition(self, to_state: AgentLifecycleState, reason: str | None = None) -> None:
        """Fügt Lifecycle-Transition hinzu."""
        transition = LifecycleTransition(
            from_state=self.current_state, to_state=to_state, reason=reason
        )
        self.transition_history.append(transition)
        self.previous_state = self.current_state
        self.current_state = to_state
        self.state_changed_at = datetime.now(UTC)


@dataclass
class TaskExecutionResult:
    """Ergebnis einer Task-Ausführung."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PriorityTask:
    """Task mit Priorität für Priority-Queue."""
    priority: int
    timestamp: float
    task: AgentTask

    def __lt__(self, other: PriorityTask) -> bool:
        """Vergleichsoperator für Priority-Queue (niedrigere Zahl = höhere Priorität)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class EventSubscription:
    """Event-Subscription."""
    subscription_id: str
    event_types: set[EventType]
    callback: Callable[[AgentEvent], Awaitable[None]]
    agent_id_filter: str | None = None
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class TaskQueueConfig:
    """Konfiguration für Task-Queue."""
    max_queue_size: int = 1000
    max_concurrent_tasks: int = 10
    default_timeout_seconds: int = 300
    retry_delay_seconds: int = 5
    max_retry_delay_seconds: int = 60
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST
    enable_priority_queue: bool = True
    enable_task_retry: bool = True
    max_retry_attempts: int = 3


LifecycleCallback = Callable[[str, AgentLifecycleState, AgentLifecycleState], Awaitable[None]]
TaskHandler = Callable[[AgentTask], Awaitable[Any]]
EventHandler = Callable[[AgentEvent], Awaitable[None]]
