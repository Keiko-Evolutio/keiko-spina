"""Backward-Compatibility-Layer für services.tasks.

Dieses Modul wurde konsolidiert mit dem vollständigen Task-Management-System
in backend.task_management. Alle Funktionalitäten sind dort verfügbar.

DEPRECATED: Verwende stattdessen backend.task_management für neue Implementierungen.
"""

from __future__ import annotations

import warnings

from kei_logging import get_logger

# Service-Imports
try:
    from services.messaging import get_messaging_service
    from services.messaging.naming import subject_for_tasks
    MESSAGING_SERVICES_AVAILABLE = True
except ImportError:
    subject_for_tasks = None
    get_messaging_service = None
    MESSAGING_SERVICES_AVAILABLE = False

# Deprecation-Warnung beim Import
warnings.warn(
    "services.tasks ist deprecated. Verwende backend.task_management für neue Implementierungen.",
    DeprecationWarning,
    stacklevel=2
)

# Import der konsolidierten Task-Management-Komponenten
from task_management import (
    ExecutionResult,
    Task,
    TaskExecutionEngine,
    TaskPriority,
    TaskState,
    TaskType,
    TaskWorkerPool,
    WorkerMetrics,
    task_execution_engine,
)

logger = get_logger(__name__)


class TaskWorker:
    """Backward-Compatibility-Wrapper für KEI-Bus-spezifischen TaskWorker.

    DEPRECATED: Diese Klasse ist ein Compatibility-Layer.
    Verwende backend.task_management.TaskWorker für neue Implementierungen.

    Diese Implementierung kombiniert die KEI-Bus-Integration der ursprünglichen
    services.tasks.TaskWorker mit der Enterprise-Grade-Funktionalität der
    task_management.TaskWorker.
    """

    def __init__(
        self,
        *,
        queue: str,
        tenant: str | None = None,
        max_in_flight: int = 8
    ) -> None:
        """Initialisiert KEI-Bus-TaskWorker.

        Args:
            queue: Queue-Name für KEI-Bus
            tenant: Tenant-ID (optional)
            max_in_flight: Maximale parallele Tasks (Standard: 8)
        """
        warnings.warn(
            "services.tasks.TaskWorker ist deprecated. "
            "Verwende backend.task_management.TaskWorker für neue Implementierungen.",
            DeprecationWarning,
            stacklevel=2
        )

        self.queue = queue
        self.tenant = tenant
        self.max_in_flight = max(1, max_in_flight)

        # Lazy Import um zirkuläre Abhängigkeiten zu vermeiden
        self._bus = None
        self._stop_event = None
        self._initialized = False



    async def _ensure_initialized(self) -> None:
        """Initialisiert KEI-Bus-Abhängigkeiten lazy."""
        if self._initialized:
            return

        try:
            import asyncio

            if not MESSAGING_SERVICES_AVAILABLE:
                raise ImportError("Messaging services not available")

            self._bus = get_messaging_service()
            self._stop_event = asyncio.Event()
            self._subject_for_tasks = subject_for_tasks
            self._initialized = True

        except ImportError as e:
            logger.exception(f"KEI-Bus-Abhängigkeiten nicht verfügbar: {e}")
            raise RuntimeError(
                "KEI-Bus-Integration nicht verfügbar. "
                "Verwende task_management.TaskWorker für Bus-unabhängige Tasks."
            ) from e

    async def start(self) -> None:
        """Startet den Worker-Loop im Hintergrund."""
        await self._ensure_initialized()

        await self._bus.initialize()
        subject = self._subject_for_tasks(
            queue=self.queue,
            version=1,
            tenant=self.tenant
        )

        logger.info(f"Starte TaskWorker für {subject} (max_in_flight={self.max_in_flight})")

        import asyncio
        asyncio.create_task(self._pull_loop(subject))

    async def stop(self) -> None:
        """Stoppt den Worker asynchron."""
        if self._stop_event:
            self._stop_event.set()

    async def _pull_loop(self, subject: str) -> None:
        """Einfacher Pull-Loop mit Backpressure-Steuerung."""
        import asyncio
        from typing import Any

        sem = asyncio.Semaphore(self.max_in_flight)

        async def _handle(env: Any) -> None:
            """Verarbeitet einen Task (Logging/No-Op)."""
            try:
                payload: dict[str, Any] = getattr(env, "payload", None) or {}
                task_id = payload.get("task_id", getattr(env, "id", "unknown"))
                logger.info(f"Verarbeite Task: {task_id} auf {subject}")
            except Exception as e:
                logger.exception(f"Task-Verarbeitung fehlgeschlagen: {e}")

        async def _on_message(env: Any) -> None:
            """Message-Handler mit Semaphore-Steuerung."""
            await sem.acquire()
            try:
                await _handle(env)
            finally:
                sem.release()

        # Subscribe zu KEI-Bus-Subject
        await self._bus.subscribe(subject, queue=None, handler=_on_message)


# Backward-Compatibility-Exports
__all__ = [
    "ExecutionResult",
    "Task",
    "TaskExecutionEngine",
    "TaskPriority",
    "TaskState",
    "TaskType",
    "TaskWorker",
    # Re-exports aus task_management für Convenience
    "TaskWorkerPool",
    "WorkerMetrics",
    "task_execution_engine",
]


def get_migration_guide() -> str:
    """Gibt Migrations-Guide für Umstellung auf task_management zurück.

    Returns:
        Migrations-Anleitung als String
    """
    return """
    MIGRATION GUIDE: services.tasks → task_management

    Alt (deprecated):
        from services.tasks import TaskWorker
        worker = TaskWorker(queue="my_queue")

    Neu (empfohlen):
        from task_management import TaskWorker, TaskExecutionEngine

        # Für KEI-Bus-Integration:
        engine = TaskExecutionEngine()
        await engine.start()

        # Für generische Task-Verarbeitung:
        worker = TaskWorker("worker_001", my_executor_function)
        await worker.start()

    Vorteile der Migration:
    - Enterprise-Grade-Features (Metriken, Lifecycle-Management)
    - Bessere Error-Handling und Retry-Logik
    - Vollständige Test-Coverage
    - Performance-Optimierungen
    - Zukunftssichere API
    """



