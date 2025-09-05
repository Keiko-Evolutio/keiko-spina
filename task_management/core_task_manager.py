# backend/task_management/core_task_manager.py
"""Core Task Manager für Keiko Personal Assistant

Implementiert zentrale Task-Verwaltung mit vollständigem Lifecycle-Management,
State-Transitions, Dependencies und Scheduling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TASK_TIMEOUT_SECONDS,
)
from .utils import (
    generate_uuid,
    get_current_utc_datetime,
)

logger = get_logger(__name__)


class TaskState(str, Enum):
    """Task-Status-Definitionen."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    SUSPENDED = "suspended"


class TaskPriority(str, Enum):
    """Task-Prioritäten."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class TaskType(str, Enum):
    """Task-Typen."""
    AGENT_EXECUTION = "agent_execution"
    DATA_PROCESSING = "data_processing"
    NLP_ANALYSIS = "nlp_analysis"
    TOOL_CALL = "tool_call"
    WORKFLOW = "workflow"
    BATCH_JOB = "batch_job"
    SCHEDULED_TASK = "scheduled_task"
    SYSTEM_MAINTENANCE = "system_maintenance"
    # Orchestrator-specific task types
    IMAGE_GENERATION = "image_generation"
    WEB_SEARCH = "web_search"
    CONVERSATION = "conversation"


@dataclass
class TaskExecutionContext:
    """Kontext für Task-Ausführung."""
    user_id: str | None = None
    agent_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
    request_id: str | None = None

    # Execution-Environment
    environment: str = "production"
    resource_limits: dict[str, Any] = field(default_factory=dict)
    security_context: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "environment": self.environment,
            "resource_limits": self.resource_limits,
            "security_context": self.security_context,
            "metadata": self.metadata
        }


@dataclass
class TaskDependency:
    """Task-Abhängigkeit."""
    task_id: str
    dependency_type: str = "completion"  # completion, data, resource
    condition: str | None = None
    timeout_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "task_id": self.task_id,
            "dependency_type": self.dependency_type,
            "condition": self.condition,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class TaskSchedule:
    """Task-Scheduling-Konfiguration."""
    scheduled_at: datetime | None = None
    delay_seconds: int | None = None
    cron_expression: str | None = None
    repeat_count: int | None = None
    repeat_interval_seconds: int | None = None

    # Scheduling-Constraints
    earliest_start: datetime | None = None
    latest_start: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "delay_seconds": self.delay_seconds,
            "cron_expression": self.cron_expression,
            "repeat_count": self.repeat_count,
            "repeat_interval_seconds": self.repeat_interval_seconds,
            "earliest_start": self.earliest_start.isoformat() if self.earliest_start else None,
            "latest_start": self.latest_start.isoformat() if self.latest_start else None
        }


@dataclass
class TaskResult:
    """Task-Ausführungsergebnis."""
    task_id: str
    state: TaskState

    # Execution-Details
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time_ms: float | None = None

    # Result-Data
    result_data: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None

    # Metrics
    cpu_time_ms: float | None = None
    memory_usage_mb: float | None = None
    network_calls: int | None = None

    # Retry-Information
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "network_calls": self.network_calls,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None
        }


@dataclass
class Task:
    """Zentrale Task-Definition."""
    task_id: str
    task_type: TaskType
    state: TaskState
    priority: TaskPriority

    # Task-Definition
    name: str
    description: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    # Execution-Context
    context: TaskExecutionContext = field(default_factory=TaskExecutionContext)

    # Timing
    created_at: datetime = field(default_factory=get_current_utc_datetime)
    updated_at: datetime = field(default_factory=get_current_utc_datetime)

    # Scheduling
    schedule: TaskSchedule | None = None
    timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS

    # Dependencies
    dependencies: list[TaskDependency] = field(default_factory=list)

    # Retry-Configuration
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay_seconds: int = 60
    retry_backoff_multiplier: float = 2.0

    # Result
    result: TaskResult | None = None

    # Tags und Labels
    tags: set[str] = field(default_factory=set)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def is_terminal_state(self) -> bool:
        """Prüft, ob Task in terminalem Zustand ist."""
        return self.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMEOUT}

    @property
    def is_active(self) -> bool:
        """Prüft, ob Task aktiv ist."""
        return self.state in {TaskState.QUEUED, TaskState.RUNNING, TaskState.RETRYING}

    @property
    def can_retry(self) -> bool:
        """Prüft, ob Task wiederholt werden kann."""
        if not self.result:
            return False
        return (self.state == TaskState.FAILED and
                self.result.retry_count < self.max_retries)

    @property
    def is_overdue(self) -> bool:
        """Prüft, ob Task überfällig ist."""
        if self.state != TaskState.RUNNING or not self.result or not self.result.started_at:
            return False

        elapsed = datetime.now(UTC) - self.result.started_at
        return elapsed.total_seconds() > self.timeout_seconds

    def update_state(self, new_state: TaskState) -> None:
        """Aktualisiert Task-State mit Timestamp."""
        self.state = new_state
        self.updated_at = datetime.now(UTC)

    def add_dependency(self, dependency: TaskDependency) -> None:
        """Fügt Task-Abhängigkeit hinzu."""
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)
            self.updated_at = datetime.now(UTC)

    def add_tag(self, tag: str) -> None:
        """Fügt Tag hinzu."""
        self.tags.add(tag)
        self.updated_at = datetime.now(UTC)

    def set_label(self, key: str, value: str) -> None:
        """Setzt Label."""
        self.labels[key] = value
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "state": self.state.value,
            "priority": self.priority.value,
            "name": self.name,
            "description": self.description,
            "payload": self.payload,
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "timeout_seconds": self.timeout_seconds,
            "dependencies": [dep.to_dict() for dep in self.dependencies],
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_backoff_multiplier": self.retry_backoff_multiplier,
            "result": self.result.to_dict() if self.result else None,
            "tags": list(self.tags),
            "labels": self.labels,
            "is_terminal_state": self.is_terminal_state,
            "is_active": self.is_active,
            "can_retry": self.can_retry,
            "is_overdue": self.is_overdue
        }


class TaskManager:
    """Zentraler Task Manager."""

    def __init__(self):
        """Initialisiert Task Manager."""
        # Task-Storage
        self._tasks: dict[str, Task] = {}
        self._task_lock = asyncio.Lock()

        # Indexing für Performance
        self._tasks_by_state: dict[TaskState, set[str]] = {state: set() for state in TaskState}
        self._tasks_by_priority: dict[TaskPriority, set[str]] = {priority: set() for priority in TaskPriority}
        self._tasks_by_type: dict[TaskType, set[str]] = {task_type: set() for task_type in TaskType}
        self._tasks_by_user: dict[str, set[str]] = {}
        self._tasks_by_agent: dict[str, set[str]] = {}

        # Dependency-Tracking
        self._dependency_graph: dict[str, set[str]] = {}  # task_id -> dependent_task_ids
        self._reverse_dependencies: dict[str, set[str]] = {}  # task_id -> dependency_task_ids

        # Background-Tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._is_running = False

        # Statistiken
        self._tasks_created = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_cancelled = 0

    async def start(self) -> None:
        """Startet Task Manager."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        timeout_task = asyncio.create_task(self._timeout_check_loop())
        dependency_task = asyncio.create_task(self._dependency_resolution_loop())

        self._background_tasks.update([cleanup_task, timeout_task, dependency_task])

        logger.info("Task Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Task Manager."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Task Manager gestoppt")

    @trace_function("task_manager.create_task")
    async def create_task(
        self,
        task_type: TaskType,
        name: str,
        payload: dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        context: TaskExecutionContext | None = None,
        schedule: TaskSchedule | None = None,
        dependencies: list[TaskDependency] | None = None,
        timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        tags: set[str] | None = None,
        labels: dict[str, str] | None = None,
        description: str | None = None
    ) -> Task:
        """Erstellt neue Task.

        Args:
            task_type: Task-Typ
            name: Task-Name
            payload: Task-Payload
            priority: Task-Priorität
            context: Execution-Context
            schedule: Scheduling-Konfiguration
            dependencies: Task-Abhängigkeiten
            timeout_seconds: Timeout in Sekunden
            max_retries: Maximale Wiederholungen
            tags: Task-Tags
            labels: Task-Labels
            description: Task-Beschreibung

        Returns:
            Erstellte Task
        """
        # Parameter validieren
        if not name or not name.strip():
            raise ValueError("Task-Name darf nicht leer sein")

        if not isinstance(payload, dict):
            raise ValueError("Payload muss ein Dictionary sein")

        task_id = generate_uuid()

        task = Task(
            task_id=task_id,
            task_type=task_type,
            state=TaskState.PENDING,
            priority=priority,
            name=name,
            description=description,
            payload=payload,
            context=context or TaskExecutionContext(),
            schedule=schedule,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies or [],
            max_retries=max_retries,
            tags=tags or set(),
            labels=labels or {}
        )

        async with self._task_lock:
            # Task speichern
            self._tasks[task_id] = task

            # Indizes aktualisieren
            self._update_indices_for_task(task)

            # Dependencies verwalten
            await self._update_dependency_graph(task)

            # Statistiken
            self._tasks_created += 1

        logger.info(f"Task erstellt: {task_id} ({task_type.value}) - {name}")

        return task

    async def get_task(self, task_id: str) -> Task | None:
        """Holt Task nach ID."""
        return self._tasks.get(task_id)

    async def update_task_state(self, task_id: str, new_state: TaskState) -> bool:
        """Aktualisiert Task-State.

        Args:
            task_id: Task-ID
            new_state: Neuer State

        Returns:
            True wenn erfolgreich
        """
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            old_state = task.state

            # State-Transition validieren
            if not self._is_valid_state_transition(old_state, new_state):
                logger.warning(f"Ungültige State-Transition für Task {task_id}: {old_state.value} -> {new_state.value}")
                return False

            # State aktualisieren
            task.update_state(new_state)

            # Indizes aktualisieren
            self._tasks_by_state[old_state].discard(task_id)
            self._tasks_by_state[new_state].add(task_id)

            # Statistiken aktualisieren
            if new_state == TaskState.COMPLETED:
                self._tasks_completed += 1
            elif new_state == TaskState.FAILED:
                self._tasks_failed += 1
            elif new_state == TaskState.CANCELLED:
                self._tasks_cancelled += 1

            logger.debug(f"Task {task_id} State aktualisiert: {old_state.value} -> {new_state.value}")

            return True

    def _is_valid_state_transition(self, from_state: TaskState, to_state: TaskState) -> bool:
        """Validiert State-Transition."""
        valid_transitions = {
            TaskState.PENDING: {TaskState.QUEUED, TaskState.CANCELLED},
            TaskState.QUEUED: {TaskState.RUNNING, TaskState.CANCELLED, TaskState.SUSPENDED},
            TaskState.RUNNING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMEOUT, TaskState.SUSPENDED},
            TaskState.FAILED: {TaskState.RETRYING, TaskState.CANCELLED},
            TaskState.RETRYING: {TaskState.QUEUED, TaskState.CANCELLED},
            TaskState.SUSPENDED: {TaskState.QUEUED, TaskState.CANCELLED},
            TaskState.COMPLETED: set(),  # Terminal
            TaskState.CANCELLED: set(),  # Terminal
            TaskState.TIMEOUT: {TaskState.RETRYING, TaskState.CANCELLED}
        }

        return to_state in valid_transitions.get(from_state, set())

    def _update_indices_for_task(self, task: Task) -> None:
        """Aktualisiert Indizes für Task."""
        task_id = task.task_id

        # State-Index
        self._tasks_by_state[task.state].add(task_id)

        # Priority-Index
        self._tasks_by_priority[task.priority].add(task_id)

        # Type-Index
        self._tasks_by_type[task.task_type].add(task_id)

        # User-Index
        if task.context.user_id:
            if task.context.user_id not in self._tasks_by_user:
                self._tasks_by_user[task.context.user_id] = set()
            self._tasks_by_user[task.context.user_id].add(task_id)

        # Agent-Index
        if task.context.agent_id:
            if task.context.agent_id not in self._tasks_by_agent:
                self._tasks_by_agent[task.context.agent_id] = set()
            self._tasks_by_agent[task.context.agent_id].add(task_id)

    async def _update_dependency_graph(self, task: Task) -> None:
        """Aktualisiert Dependency-Graph."""
        task_id = task.task_id

        for dependency in task.dependencies:
            dep_task_id = dependency.task_id

            # Forward-Dependencies
            if dep_task_id not in self._dependency_graph:
                self._dependency_graph[dep_task_id] = set()
            self._dependency_graph[dep_task_id].add(task_id)

            # Reverse-Dependencies
            if task_id not in self._reverse_dependencies:
                self._reverse_dependencies[task_id] = set()
            self._reverse_dependencies[task_id].add(dep_task_id)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Task-Cleanup."""
        while self._is_running:
            try:
                await self._cleanup_completed_tasks()
                await asyncio.sleep(300)  # 5 Minuten
            except Exception as e:
                logger.exception(f"Cleanup-Loop-Fehler: {e}")
                await asyncio.sleep(60)

    async def _timeout_check_loop(self) -> None:
        """Background-Loop für Timeout-Checks."""
        while self._is_running:
            try:
                await self._check_task_timeouts()
                await asyncio.sleep(30)  # 30 Sekunden
            except Exception as e:
                logger.exception(f"Timeout-Check-Loop-Fehler: {e}")
                await asyncio.sleep(30)

    async def _dependency_resolution_loop(self) -> None:
        """Background-Loop für Dependency-Resolution."""
        while self._is_running:
            try:
                await self._resolve_dependencies()
                await asyncio.sleep(10)  # 10 Sekunden
            except Exception as e:
                logger.exception(f"Dependency-Resolution-Loop-Fehler: {e}")
                await asyncio.sleep(10)

    async def _cleanup_completed_tasks(self) -> None:
        """Bereinigt abgeschlossene Tasks."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        async with self._task_lock:
            tasks_to_remove = []

            for task_id, task in self._tasks.items():
                if (task.is_terminal_state and
                    task.updated_at < cutoff_time):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                await self._remove_task(task_id)

        if tasks_to_remove:
            logger.info(f"Cleanup: {len(tasks_to_remove)} Tasks entfernt")

    async def _check_task_timeouts(self) -> None:
        """Prüft auf Task-Timeouts."""
        running_task_ids = list(self._tasks_by_state[TaskState.RUNNING])

        for task_id in running_task_ids:
            task = self._tasks.get(task_id)
            if task and task.is_overdue:
                await self.update_task_state(task_id, TaskState.TIMEOUT)
                logger.warning(f"Task {task_id} wegen Timeout beendet")

    async def _resolve_dependencies(self) -> None:
        """Löst Task-Dependencies auf."""
        pending_task_ids = list(self._tasks_by_state[TaskState.PENDING])

        for task_id in pending_task_ids:
            task = self._tasks.get(task_id)
            if task and await self._are_dependencies_satisfied(task):
                await self.update_task_state(task_id, TaskState.QUEUED)

    async def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Prüft, ob Task-Dependencies erfüllt sind."""
        for dependency in task.dependencies:
            dep_task = self._tasks.get(dependency.task_id)
            if not dep_task:
                continue

            if dependency.dependency_type == "completion":
                if dep_task.state != TaskState.COMPLETED:
                    return False
            # Weitere Dependency-Typen können hier implementiert werden

        return True

    async def _remove_task(self, task_id: str) -> None:
        """Entfernt Task aus allen Indizes."""
        task = self._tasks.get(task_id)
        if not task:
            return

        # Aus Hauptspeicher entfernen
        del self._tasks[task_id]

        # Aus Indizes entfernen
        self._tasks_by_state[task.state].discard(task_id)
        self._tasks_by_priority[task.priority].discard(task_id)
        self._tasks_by_type[task.task_type].discard(task_id)

        if task.context.user_id:
            self._tasks_by_user.get(task.context.user_id, set()).discard(task_id)

        if task.context.agent_id:
            self._tasks_by_agent.get(task.context.agent_id, set()).discard(task_id)

        # Aus Dependency-Graph entfernen
        self._dependency_graph.pop(task_id, None)
        self._reverse_dependencies.pop(task_id, None)

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Task-Manager-Statistiken zurück."""
        return {
            "total_tasks": len(self._tasks),
            "tasks_created": self._tasks_created,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "tasks_cancelled": self._tasks_cancelled,
            "tasks_by_state": {state.value: len(task_ids) for state, task_ids in self._tasks_by_state.items()},
            "tasks_by_priority": {priority.value: len(task_ids) for priority, task_ids in self._tasks_by_priority.items()},
            "tasks_by_type": {task_type.value: len(task_ids) for task_type, task_ids in self._tasks_by_type.items()},
            "is_running": self._is_running,
            "background_tasks": len(self._background_tasks)
        }


# Globale Task Manager Instanz
task_manager = TaskManager()
