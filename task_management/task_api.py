# backend/task_management/task_api.py
"""Task Management API für Keiko Personal Assistant

Implementiert vollständige Task-API mit submit_task(), cancel_task(), get_task_status(),
list_tasks(), update_task() und retry_task() mit Idempotenz-Mechanismen.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, field_validator

from kei_logging import get_logger
from observability import trace_function

from .constants import (
    MAX_TASK_DESCRIPTION_LENGTH,
    MAX_TASK_NAME_LENGTH,
    MAX_TASKS_PER_LIST_REQUEST,
    VALIDATION_RULES,
)
from .core_task_manager import (
    Task,
    TaskDependency,
    TaskExecutionContext,
    TaskPriority,
    TaskResult,
    TaskSchedule,
    TaskState,
    TaskType,
    task_manager,
)
from .idempotency_manager import DuplicateDetectionStrategy, idempotency_manager
from .utils import (
    generate_uuid,
    sanitize_string,
)

logger = get_logger(__name__)


class TaskAPIValidator:
    """Validierungs-Utilities für Task API."""

    @staticmethod
    def validate_submit_request(request: SubmitTaskRequest) -> None:
        """Validiert Submit-Request.

        Args:
            request: Zu validierender Request

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        # Name validieren
        if not request.name or not request.name.strip():
            raise ValueError("Task-Name darf nicht leer sein")

        if len(request.name) > MAX_TASK_NAME_LENGTH:
            raise ValueError(f"Task-Name zu lang (max. {MAX_TASK_NAME_LENGTH} Zeichen)")

        # Description validieren
        if request.description and len(request.description) > MAX_TASK_DESCRIPTION_LENGTH:
            raise ValueError(f"Task-Beschreibung zu lang (max. {MAX_TASK_DESCRIPTION_LENGTH} Zeichen)")

        # Timeout validieren
        timeout_rules = VALIDATION_RULES["timeout_seconds"]
        if request.timeout_seconds < timeout_rules["min_value"] or request.timeout_seconds > timeout_rules["max_value"]:
            raise ValueError(f"Timeout muss zwischen {timeout_rules['min_value']} und {timeout_rules['max_value']} Sekunden liegen")

        # Max-Retries validieren
        retry_rules = VALIDATION_RULES["max_retries"]
        if request.max_retries < retry_rules["min_value"] or request.max_retries > retry_rules["max_value"]:
            raise ValueError(f"Max-Retries muss zwischen {retry_rules['min_value']} und {retry_rules['max_value']} liegen")

    @staticmethod
    def validate_list_request(request: TaskListRequest) -> None:
        """Validiert List-Request.

        Args:
            request: Zu validierender Request

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        if request.page < 1:
            raise ValueError("Page muss >= 1 sein")

        if request.page_size < 1 or request.page_size > MAX_TASKS_PER_LIST_REQUEST:
            raise ValueError(f"Page-Size muss zwischen 1 und {MAX_TASKS_PER_LIST_REQUEST} liegen")

    @staticmethod
    def sanitize_submit_request(request: SubmitTaskRequest) -> SubmitTaskRequest:
        """Bereinigt Submit-Request.

        Args:
            request: Zu bereinigender Request

        Returns:
            Bereinigter Request
        """
        # Name bereinigen
        request.name = sanitize_string(request.name, MAX_TASK_NAME_LENGTH)

        # Description bereinigen
        if request.description:
            request.description = sanitize_string(request.description, MAX_TASK_DESCRIPTION_LENGTH)

        return request


class TaskFilterType(str, Enum):
    """Filter-Typen für Task-Listen."""
    STATE = "state"
    PRIORITY = "priority"
    TYPE = "type"
    USER = "user"
    AGENT = "agent"
    TAG = "tag"
    CREATED_AFTER = "created_after"
    CREATED_BEFORE = "created_before"


class SortOrder(str, Enum):
    """Sortier-Reihenfolge."""
    ASC = "asc"
    DESC = "desc"


# Request/Response Models
class SubmitTaskRequest(BaseModel):
    """Request für Task-Submission."""
    task_type: TaskType
    name: str
    description: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    # Execution-Context
    user_id: str | None = None
    agent_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None

    # Scheduling
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    scheduled_at: datetime | None = None
    delay_seconds: int | None = Field(default=None, ge=0)

    # Dependencies
    dependencies: list[str] = Field(default_factory=list)

    # Retry-Configuration
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=1)

    # Tags und Labels
    tags: set[str] = Field(default_factory=set)
    labels: dict[str, str] = Field(default_factory=dict)

    # Idempotenz
    idempotency_key: str | None = None
    correlation_id: str | None = None

    @field_validator("name")
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Task-Name darf nicht leer sein")
        if len(v) > 200:
            raise ValueError("Task-Name darf maximal 200 Zeichen haben")
        return v.strip()

    @field_validator("dependencies")
    def validate_dependencies(cls, v):
        if len(v) > 10:
            raise ValueError("Maximal 10 Dependencies erlaubt")
        return v


class SubmitTaskResponse(BaseModel):
    """Response für Task-Submission."""
    task_id: str
    state: TaskState
    correlation_id: str | None = None
    queued_at: datetime
    estimated_start_time: datetime | None = None

    # Idempotenz-Information
    is_duplicate: bool = False
    original_task_id: str | None = None


class TaskStatusResponse(BaseModel):
    """Response für Task-Status."""
    task_id: str
    task_type: TaskType
    state: TaskState
    priority: TaskPriority
    name: str
    description: str | None = None

    # Timing
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Execution-Details
    execution_time_ms: float | None = None
    timeout_seconds: int

    # Result
    result_data: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: str | None = None

    # Retry-Information
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: datetime | None = None

    # Context
    user_id: str | None = None
    agent_id: str | None = None
    correlation_id: str | None = None

    # Dependencies
    dependencies: list[str] = Field(default_factory=list)
    dependent_tasks: list[str] = Field(default_factory=list)

    # Tags und Labels
    tags: set[str] = Field(default_factory=set)
    labels: dict[str, str] = Field(default_factory=dict)

    # Status-Flags
    is_terminal_state: bool
    is_active: bool
    can_retry: bool
    is_overdue: bool


class TaskFilter(BaseModel):
    """Filter für Task-Listen."""
    filter_type: TaskFilterType
    value: str | list[str] | datetime
    operator: str = "eq"  # eq, ne, in, not_in, gt, lt, gte, lte


class TaskListRequest(BaseModel):
    """Request für Task-Listen."""
    # Pagination
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)

    # Filtering
    filters: list[TaskFilter] = Field(default_factory=list)

    # Sorting
    sort_by: str = "created_at"
    sort_order: SortOrder = SortOrder.DESC

    # Includes
    include_result_data: bool = False
    include_dependencies: bool = False


class TaskListResponse(BaseModel):
    """Response für Task-Listen."""
    tasks: list[TaskStatusResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class UpdateTaskRequest(BaseModel):
    """Request für Task-Updates."""
    name: str | None = None
    description: str | None = None
    priority: TaskPriority | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=3600)

    # Tags und Labels
    add_tags: set[str] = Field(default_factory=set)
    remove_tags: set[str] = Field(default_factory=set)
    set_labels: dict[str, str] = Field(default_factory=dict)
    remove_labels: set[str] = Field(default_factory=set)

    # Dependencies
    add_dependencies: list[str] = Field(default_factory=list)
    remove_dependencies: list[str] = Field(default_factory=list)


class RetryTaskRequest(BaseModel):
    """Request für Task-Retry."""
    reset_retry_count: bool = False
    new_max_retries: int | None = Field(default=None, ge=0, le=10)
    delay_seconds: int | None = Field(default=None, ge=0)


class CancelTaskRequest(BaseModel):
    """Request für Task-Cancellation."""
    reason: str | None = None
    force: bool = False


class TaskAPI:
    """Task Management API Implementation."""

    def __init__(self):
        """Initialisiert Task API."""
        self._api_lock = asyncio.Lock()

        # Statistiken
        self._requests_processed = 0
        self._tasks_submitted = 0
        self._tasks_cancelled = 0
        self._duplicates_handled = 0

    @trace_function("task_api.submit_task")
    async def submit_task(
        self,
        request: SubmitTaskRequest,
        request_id: str | None = None
    ) -> SubmitTaskResponse:
        """Submits neue Task mit Idempotenz-Mechanismen.

        Args:
            request: Task-Submit-Request
            request_id: Request-ID für Idempotenz

        Returns:
            Task-Submit-Response

        Raises:
            HTTPException: Bei Validierungs- oder Verarbeitungsfehlern
        """
        self._requests_processed += 1

        try:
            # Request validieren und bereinigen
            TaskAPIValidator.validate_submit_request(request)
            request = TaskAPIValidator.sanitize_submit_request(request)

            # Generiere Request-ID falls nicht vorhanden
            if not request_id:
                request_id = generate_uuid()

            # Generiere Correlation-ID falls nicht vorhanden
            correlation_id = request.correlation_id
            if not correlation_id:
                corr_obj = await idempotency_manager.create_correlation_id(
                    created_by=request.user_id,
                    context={"api_call": "submit_task", "task_type": request.task_type.value}
                )
                correlation_id = corr_obj.correlation_id

            # Prüfe auf Duplicates
            duplicate_result = await idempotency_manager.check_duplicate(
                request_data=request.dict(),
                idempotency_key=request.idempotency_key,
                strategy=DuplicateDetectionStrategy.CONTENT_HASH
            )

            if duplicate_result.is_duplicate and duplicate_result.original_response:
                self._duplicates_handled += 1
                logger.info(f"Duplicate-Request erkannt: {request_id}")

                # Gebe Original-Response zurück
                return SubmitTaskResponse(
                    **duplicate_result.original_response,
                    is_duplicate=True,
                    original_task_id=duplicate_result.original_request_id
                )

            # Cache Request für Idempotenz
            await idempotency_manager.cache_request(
                request_id=request_id,
                idempotency_key=request.idempotency_key or request_id,
                request_data=request.dict(),
                correlation_id=correlation_id,
                user_id=request.user_id,
                agent_id=request.agent_id
            )

            # Erstelle Task-Execution-Context
            context = TaskExecutionContext(
                user_id=request.user_id,
                agent_id=request.agent_id,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                correlation_id=correlation_id,
                request_id=request_id
            )

            # Erstelle Task-Schedule falls erforderlich
            schedule = None
            if request.scheduled_at or request.delay_seconds:
                schedule = TaskSchedule(
                    scheduled_at=request.scheduled_at,
                    delay_seconds=request.delay_seconds
                )

            # Erstelle Task-Dependencies
            dependencies = []
            for dep_task_id in request.dependencies:
                dependencies.append(TaskDependency(
                    task_id=dep_task_id,
                    dependency_type="completion"
                ))

            # Erstelle Task
            task = await task_manager.create_task(
                task_type=request.task_type,
                name=request.name,
                payload=request.payload,
                priority=request.priority,
                context=context,
                schedule=schedule,
                dependencies=dependencies,
                timeout_seconds=request.timeout_seconds,
                max_retries=request.max_retries,
                tags=request.tags,
                labels=request.labels,
                description=request.description
            )

            # Bestimme geschätzte Startzeit
            estimated_start_time = None
            if schedule and schedule.scheduled_at:
                estimated_start_time = schedule.scheduled_at
            elif schedule and schedule.delay_seconds:
                estimated_start_time = datetime.now(UTC) + timedelta(seconds=schedule.delay_seconds)

            # Erstelle Response
            response = SubmitTaskResponse(
                task_id=task.task_id,
                state=task.state,
                correlation_id=correlation_id,
                queued_at=task.created_at,
                estimated_start_time=estimated_start_time
            )

            # Update Response im Cache
            await idempotency_manager.update_response(
                request_id=request_id,
                response_data=response.dict()
            )

            self._tasks_submitted += 1
            logger.info(f"Task submitted: {task.task_id} ({request.task_type.value})")

            return response

        except Exception as e:
            logger.exception(f"Task-Submission fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task-Submission fehlgeschlagen: {e!s}"
            )

    @trace_function("task_api.cancel_task")
    async def cancel_task(
        self,
        task_id: str,
        request: CancelTaskRequest | None = None
    ) -> dict[str, Any]:
        """Cancelt Task mit graceful Cancellation.

        Args:
            task_id: Task-ID
            request: Cancel-Request

        Returns:
            Cancellation-Result

        Raises:
            HTTPException: Bei Fehlern
        """
        try:
            task = await task_manager.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} nicht gefunden"
                )

            # Prüfe, ob Task cancellable ist
            if task.is_terminal_state:
                return {
                    "task_id": task_id,
                    "status": "already_terminal",
                    "current_state": task.state.value,
                    "message": f"Task ist bereits in terminalem Zustand: {task.state.value}"
                }

            # Force-Cancel oder graceful Cancel
            force = request.force if request else False

            if force or task.state in {TaskState.PENDING, TaskState.QUEUED}:
                # Direkte Cancellation
                success = await task_manager.update_task_state(task_id, TaskState.CANCELLED)

                if success:
                    self._tasks_cancelled += 1

                    # Update Task-Result
                    if not task.result:
                        task.result = TaskResult(task_id=task_id, state=TaskState.CANCELLED)
                    else:
                        task.result.state = TaskState.CANCELLED

                    task.result.completed_at = datetime.now(UTC)
                    task.result.error_message = request.reason if request else "Task cancelled by user"

                    logger.info(f"Task cancelled: {task_id}")

                    return {
                        "task_id": task_id,
                        "status": "cancelled",
                        "cancelled_at": datetime.now(UTC).isoformat(),
                        "reason": request.reason if request else "User cancellation"
                    }
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Task-Cancellation fehlgeschlagen"
                )

            if task.state == TaskState.RUNNING:
                # Graceful Cancellation für laufende Tasks
                success = await task_manager.update_task_state(task_id, TaskState.CANCELLED)

                if success:
                    self._tasks_cancelled += 1
                    logger.info(f"Running task cancelled: {task_id}")

                    return {
                        "task_id": task_id,
                        "status": "cancellation_requested",
                        "message": "Cancellation-Signal an Worker gesendet"
                    }
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Task-Cancellation fehlgeschlagen"
                )

            return {
                "task_id": task_id,
                "status": "not_cancellable",
                "current_state": task.state.value,
                "message": f"Task in State {task.state.value} kann nicht cancelled werden"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Task-Cancellation fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task-Cancellation fehlgeschlagen: {e!s}"
            )

    @trace_function("task_api.get_task_status")
    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        """Holt detaillierten Task-Status.

        Args:
            task_id: Task-ID

        Returns:
            Task-Status-Response

        Raises:
            HTTPException: Bei Fehlern
        """
        try:
            task = await task_manager.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} nicht gefunden"
                )

            # Bestimme dependent Tasks
            dependent_tasks = []

            # Erstelle Response
            response = TaskStatusResponse(
                task_id=task.task_id,
                task_type=task.task_type,
                state=task.state,
                priority=task.priority,
                name=task.name,
                description=task.description,
                created_at=task.created_at,
                updated_at=task.updated_at,
                timeout_seconds=task.timeout_seconds,
                max_retries=task.max_retries,
                user_id=task.context.user_id,
                agent_id=task.context.agent_id,
                correlation_id=task.context.correlation_id,
                dependencies=[dep.task_id for dep in task.dependencies],
                dependent_tasks=dependent_tasks,
                tags=task.tags,
                labels=task.labels,
                is_terminal_state=task.is_terminal_state,
                is_active=task.is_active,
                can_retry=task.can_retry,
                is_overdue=task.is_overdue
            )

            # Füge Result-Details hinzu falls vorhanden
            if task.result:
                response.started_at = task.result.started_at
                response.completed_at = task.result.completed_at
                response.execution_time_ms = task.result.execution_time_ms
                response.result_data = task.result.result_data
                response.error_message = task.result.error_message
                response.error_code = task.result.error_code
                response.retry_count = task.result.retry_count
                response.next_retry_at = task.result.next_retry_at

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Task-Status-Abfrage fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task-Status-Abfrage fehlgeschlagen: {e!s}"
            )

    @trace_function("task_api.list_tasks")
    async def list_tasks(self, request: TaskListRequest) -> TaskListResponse:
        """Listet Tasks mit Filtering und Pagination.

        Args:
            request: List-Request

        Returns:
            Task-List-Response

        Raises:
            HTTPException: Bei Fehlern
        """
        try:
            # Vereinfachte Implementierung für Task-Filtering und Pagination
            all_tasks = []
            for task in task_manager._tasks.values():
                # Anwende Filter
                if self._apply_filters(task, request.filters):
                    task_response = await self._convert_task_to_response(task, request)
                    all_tasks.append(task_response)

            # Sortierung
            all_tasks = self._sort_tasks(all_tasks, request.sort_by, request.sort_order)

            # Pagination
            total_count = len(all_tasks)
            start_idx = (request.page - 1) * request.page_size
            end_idx = start_idx + request.page_size
            page_tasks = all_tasks[start_idx:end_idx]

            total_pages = (total_count + request.page_size - 1) // request.page_size

            return TaskListResponse(
                tasks=page_tasks,
                total_count=total_count,
                page=request.page,
                page_size=request.page_size,
                total_pages=total_pages,
                has_next=request.page < total_pages,
                has_previous=request.page > 1
            )

        except Exception as e:
            logger.exception(f"Task-Listen-Abfrage fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task-Listen-Abfrage fehlgeschlagen: {e!s}"
            )

    @trace_function("task_api.update_task")
    async def update_task(
        self,
        task_id: str,
        request: UpdateTaskRequest
    ) -> TaskStatusResponse:
        """Aktualisiert Task-Eigenschaften.

        Args:
            task_id: Task-ID
            request: Update-Request

        Returns:
            Aktualisierte Task-Status-Response

        Raises:
            HTTPException: Bei Fehlern
        """
        try:
            task = await task_manager.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} nicht gefunden"
                )

            # Prüfe, ob Task updatebar ist
            if task.is_terminal_state:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Task in terminalem Zustand {task.state.value} kann nicht aktualisiert werden"
                )

            # Aktualisiere Task-Eigenschaften
            updated = False

            if request.name is not None:
                task.name = request.name
                updated = True

            if request.description is not None:
                task.description = request.description
                updated = True

            if request.priority is not None:
                task.priority = request.priority
                updated = True

            if request.timeout_seconds is not None:
                task.timeout_seconds = request.timeout_seconds
                updated = True

            # Tags aktualisieren
            if request.add_tags:
                task.tags.update(request.add_tags)
                updated = True

            if request.remove_tags:
                task.tags.difference_update(request.remove_tags)
                updated = True

            # Labels aktualisieren
            if request.set_labels:
                task.labels.update(request.set_labels)
                updated = True

            if request.remove_labels:
                for label_key in request.remove_labels:
                    task.labels.pop(label_key, None)
                updated = True

            # Dependencies aktualisieren
            if request.add_dependencies:
                for dep_task_id in request.add_dependencies:
                    dependency = TaskDependency(task_id=dep_task_id, dependency_type="completion")
                    task.add_dependency(dependency)
                updated = True

            if request.remove_dependencies:
                task.dependencies = [
                    dep for dep in task.dependencies
                    if dep.task_id not in request.remove_dependencies
                ]
                updated = True

            if updated:
                task.updated_at = datetime.now(UTC)
                logger.info(f"Task aktualisiert: {task_id}")

            # Gebe aktualisierte Task-Status zurück
            return await self.get_task_status(task_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Task-Update fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task-Update fehlgeschlagen: {e!s}"
            )

    @trace_function("task_api.retry_task")
    async def retry_task(
        self,
        task_id: str,
        request: RetryTaskRequest | None = None
    ) -> dict[str, Any]:
        """Wiederholt fehlgeschlagene Task.

        Args:
            task_id: Task-ID
            request: Retry-Request

        Returns:
            Retry-Result

        Raises:
            HTTPException: Bei Fehlern
        """
        try:
            task = await task_manager.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} nicht gefunden"
                )

            # Prüfe, ob Task retryable ist
            if not task.can_retry and task.state != TaskState.FAILED:
                return {
                    "task_id": task_id,
                    "status": "not_retryable",
                    "current_state": task.state.value,
                    "message": f"Task in State {task.state.value} kann nicht wiederholt werden"
                }

            # Aktualisiere Retry-Konfiguration falls angegeben
            if request:
                if request.reset_retry_count and task.result:
                    task.result.retry_count = 0

                if request.new_max_retries is not None:
                    task.max_retries = request.new_max_retries
                    if task.result:
                        task.result.max_retries = request.new_max_retries

            # Setze Task auf RETRYING
            success = await task_manager.update_task_state(task_id, TaskState.RETRYING)

            if success:
                # Aktualisiere Result
                if task.result:
                    task.result.retry_count += 1

                    # Berechne nächste Retry-Zeit
                    delay = task.retry_delay_seconds * (task.retry_backoff_multiplier ** task.result.retry_count)
                    if request and request.delay_seconds is not None:
                        delay = request.delay_seconds

                    task.result.next_retry_at = datetime.now(UTC) + timedelta(seconds=delay)

                logger.info(f"Task retry initiiert: {task_id}")

                return {
                    "task_id": task_id,
                    "status": "retry_scheduled",
                    "retry_count": task.result.retry_count if task.result else 0,
                    "next_retry_at": task.result.next_retry_at.isoformat() if task.result and task.result.next_retry_at else None
                }
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Task-Retry fehlgeschlagen"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Task-Retry fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task-Retry fehlgeschlagen: {e!s}"
            )

    def _apply_filters(self, task: Task, filters: list[TaskFilter]) -> bool:
        """Wendet Filter auf Task an."""
        return all(self._apply_single_filter(task, filter_obj) for filter_obj in filters)

    def _apply_single_filter(self, task: Task, filter_obj: TaskFilter) -> bool:
        """Wendet einzelnen Filter an."""
        if filter_obj.filter_type == TaskFilterType.STATE:
            return task.state.value == filter_obj.value
        if filter_obj.filter_type == TaskFilterType.PRIORITY:
            return task.priority.value == filter_obj.value
        if filter_obj.filter_type == TaskFilterType.TYPE:
            return task.task_type.value == filter_obj.value
        if filter_obj.filter_type == TaskFilterType.USER:
            return task.context.user_id == filter_obj.value
        if filter_obj.filter_type == TaskFilterType.AGENT:
            return task.context.agent_id == filter_obj.value
        if filter_obj.filter_type == TaskFilterType.TAG:
            return filter_obj.value in task.tags
        if filter_obj.filter_type == TaskFilterType.CREATED_AFTER:
            return task.created_at >= filter_obj.value
        if filter_obj.filter_type == TaskFilterType.CREATED_BEFORE:
            return task.created_at <= filter_obj.value

        # Fallback: Alle Tasks durchlassen wenn Filter unbekannt
        return True

    def _sort_tasks(self, tasks: list[TaskStatusResponse], sort_by: str, sort_order: SortOrder) -> list[TaskStatusResponse]:
        """Sortiert Tasks."""
        reverse = sort_order == SortOrder.DESC

        if sort_by == "created_at":
            return sorted(tasks, key=lambda t: t.created_at, reverse=reverse)
        if sort_by == "updated_at":
            return sorted(tasks, key=lambda t: t.updated_at, reverse=reverse)
        if sort_by == "priority":
            priority_order = {TaskPriority.LOW: 1, TaskPriority.NORMAL: 2, TaskPriority.HIGH: 3, TaskPriority.CRITICAL: 4, TaskPriority.URGENT: 5}
            return sorted(tasks, key=lambda t: priority_order.get(t.priority, 0), reverse=reverse)
        if sort_by == "name":
            return sorted(tasks, key=lambda t: t.name, reverse=reverse)
        return tasks

    async def _convert_task_to_response(self, task: Task, request: TaskListRequest) -> TaskStatusResponse:
        """Konvertiert Task zu Response."""
        response = TaskStatusResponse(
            task_id=task.task_id,
            task_type=task.task_type,
            state=task.state,
            priority=task.priority,
            name=task.name,
            description=task.description,
            created_at=task.created_at,
            updated_at=task.updated_at,
            timeout_seconds=task.timeout_seconds,
            max_retries=task.max_retries,
            user_id=task.context.user_id,
            agent_id=task.context.agent_id,
            correlation_id=task.context.correlation_id,
            tags=task.tags,
            labels=task.labels,
            is_terminal_state=task.is_terminal_state,
            is_active=task.is_active,
            can_retry=task.can_retry,
            is_overdue=task.is_overdue
        )

        # Conditional Includes
        if request.include_dependencies:
            response.dependencies = [dep.task_id for dep in task.dependencies]

        if request.include_result_data and task.result:
            response.result_data = task.result.result_data
            response.error_message = task.result.error_message
            response.error_code = task.result.error_code
            response.started_at = task.result.started_at
            response.completed_at = task.result.completed_at
            response.execution_time_ms = task.result.execution_time_ms
            response.retry_count = task.result.retry_count
            response.next_retry_at = task.result.next_retry_at

        return response

    def get_api_statistics(self) -> dict[str, Any]:
        """Gibt API-Statistiken zurück."""
        return {
            "requests_processed": self._requests_processed,
            "tasks_submitted": self._tasks_submitted,
            "tasks_cancelled": self._tasks_cancelled,
            "duplicates_handled": self._duplicates_handled,
            "task_manager_stats": task_manager.get_statistics(),
            "idempotency_stats": idempotency_manager.get_statistics()
        }


# Globale Task API Instanz
task_api = TaskAPI()
