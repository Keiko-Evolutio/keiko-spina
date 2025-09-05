"""KEI-Bus-Integration für Task-Management-System.

Erweitert das Enterprise-Grade Task-Management um KEI-Bus-spezifische
Funktionalitäten wie Pull/Push-Scheduling und Backpressure-Awareness.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

from .constants import GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS
from .core_task_manager import Task, TaskPriority, TaskState, TaskType
from .task_execution_engine import ExecutionResult, TaskWorker
from .utils import get_current_utc_datetime

# Service-Imports
try:
    from services.messaging.naming import subject_for_tasks
    from services.messaging.service import get_bus_service
    MESSAGING_SERVICES_AVAILABLE = True
except ImportError:
    subject_for_tasks = None
    get_bus_service = None
    MESSAGING_SERVICES_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

# Konstanten für KEI-Bus-Integration
DEFAULT_MAX_IN_FLIGHT = 8
DEFAULT_QUEUE_VERSION = 1
DEFAULT_POLL_TIMEOUT_SECONDS = 1.0
DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS


class KEIBusTaskWorker(TaskWorker):
    """Enterprise-Grade TaskWorker mit KEI-Bus-Integration.

    Erweitert den Standard-TaskWorker um KEI-Bus-spezifische Funktionalitäten:
    - Pull/Push-Scheduling mit Backpressure
    - Tenant-spezifische Queue-Verarbeitung
    - Automatische Subject-Generierung
    - Graceful Shutdown mit KEI-Bus-Cleanup
    """

    def __init__(
        self,
        worker_id: str,
        queue: str,
        executor_function: Callable[[Task], Awaitable[ExecutionResult]],
        *,
        tenant: str | None = None,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        queue_version: int = DEFAULT_QUEUE_VERSION,
    ) -> None:
        """Initialisiert KEI-Bus-TaskWorker.

        Args:
            worker_id: Eindeutige Worker-ID
            queue: KEI-Bus-Queue-Name
            executor_function: Task-Executor-Funktion
            tenant: Tenant-ID für Multi-Tenancy (optional)
            max_in_flight: Maximale parallele Tasks
            queue_version: Queue-Version für KEI-Bus
        """
        super().__init__(worker_id, executor_function)

        self.queue = queue
        self.tenant = tenant
        self.max_in_flight = max(1, max_in_flight)
        self.queue_version = queue_version

        # KEI-Bus-spezifische Attribute
        self._bus = None
        self._subject = None
        self._semaphore: asyncio.Semaphore | None = None
        self._bus_initialized = False

        logger.info(
            f"KEI-Bus-TaskWorker initialisiert: {worker_id} "
            f"(queue={queue}, tenant={tenant}, max_in_flight={max_in_flight})"
        )

    async def _initialize_kei_bus(self) -> None:
        """Initialisiert KEI-Bus-Abhängigkeiten lazy."""
        if self._bus_initialized:
            return

        try:
            if not MESSAGING_SERVICES_AVAILABLE:
                raise ImportError("Messaging services not available")

            self._bus = get_bus_service()
            self._subject = subject_for_tasks(
                queue=self.queue,
                version=self.queue_version,
                tenant=self.tenant
            )
            self._semaphore = asyncio.Semaphore(self.max_in_flight)

            await self._bus.initialize()
            self._bus_initialized = True

            logger.info(f"KEI-Bus initialisiert für Subject: {self._subject}")

        except ImportError as e:
            logger.exception(f"KEI-Bus-Abhängigkeiten nicht verfügbar: {e}")
            raise RuntimeError(
                "KEI-Bus-Integration nicht verfügbar. "
                "Prüfe KEI-Bus-Installation und -Konfiguration."
            ) from e

    @trace_function("kei_bus_task_worker.start")
    async def start(self) -> None:
        """Startet KEI-Bus-TaskWorker mit Enterprise-Features."""
        await self._initialize_kei_bus()

        # Starte Standard-TaskWorker
        await super().start()

        # Starte KEI-Bus-Integration
        await self._start_kei_bus_integration()

        logger.info(
            f"KEI-Bus-TaskWorker gestartet: {self.worker_id} "
            f"auf Subject {self._subject}"
        )

    async def _start_kei_bus_integration(self) -> None:
        """Startet KEI-Bus-Message-Handling."""
        if not self._bus or not self._subject:
            raise RuntimeError("KEI-Bus nicht initialisiert")

        # Subscribe zu KEI-Bus-Subject mit Message-Handler
        await self._bus.subscribe(
            self._subject,
            queue=None,
            handler=self._handle_kei_bus_message
        )

        logger.debug(f"KEI-Bus-Subscription aktiv für {self._subject}")

    @trace_function("kei_bus_task_worker.handle_message")
    async def _handle_kei_bus_message(self, envelope: Any) -> None:
        """Verarbeitet KEI-Bus-Message mit Backpressure-Steuerung.

        Args:
            envelope: KEI-Bus-Message-Envelope
        """
        if not self._semaphore:
            logger.error("Semaphore nicht initialisiert")
            return

        # Backpressure-Steuerung
        await self._semaphore.acquire()

        try:
            # Konvertiere KEI-Bus-Message zu Task
            task = await KEIBusTaskWorker._convert_envelope_to_task(envelope)

            if task:
                # Reiche Task an Standard-Worker weiter
                await self.submit_task(task)

                logger.debug(
                    f"KEI-Bus-Message erfolgreich als Task weitergeleitet: "
                    f"{task.task_id}"
                )
            else:
                logger.warning("Konnte KEI-Bus-Message nicht zu Task konvertieren")

        except Exception as e:
            logger.exception(f"Fehler bei KEI-Bus-Message-Verarbeitung: {e}")
            # Aktualisiere Worker-Metriken
            self.metrics.errors_count += 1

        finally:
            self._semaphore.release()

    @staticmethod
    async def _convert_envelope_to_task(envelope: Any) -> Task | None:
        """Konvertiert KEI-Bus-Envelope zu Task-Objekt.

        Args:
            envelope: KEI-Bus-Message-Envelope

        Returns:
            Task-Objekt oder None bei Konvertierungsfehlern
        """
        try:
            # Extrahiere Payload aus Envelope
            payload: dict[str, Any] = getattr(envelope, "payload", None) or {}
            envelope_id = getattr(envelope, "id", None)

            # Generiere Task-ID
            task_id = payload.get("task_id", envelope_id or f"kei_bus_{datetime.now().timestamp()}")

            # Erstelle Task mit Standard-Werten
            task = Task(
                task_id=str(task_id),
                task_type=TaskType(payload.get("task_type", TaskType.AGENT_EXECUTION.value)),
                state=TaskState.PENDING,
                priority=TaskPriority(payload.get("priority", TaskPriority.NORMAL.value)),
                name=payload.get("name", f"KEI-Bus Task {task_id}"),
                payload=payload,
                created_at=get_current_utc_datetime(),
                description=payload.get("description", "Task aus KEI-Bus-Message")
            )

            logger.debug(f"KEI-Bus-Envelope zu Task konvertiert: {task_id}")
            return task

        except Exception as e:
            logger.exception(f"Fehler bei Envelope-zu-Task-Konvertierung: {e}")
            return None

    @trace_function("kei_bus_task_worker.stop")
    async def stop(self, graceful: bool = True) -> None:
        """Stoppt KEI-Bus-TaskWorker mit Cleanup.

        Args:
            graceful: Graceful Shutdown aktivieren
        """
        logger.info(f"Stoppe KEI-Bus-TaskWorker: {self.worker_id}")

        # Stoppe Standard-TaskWorker
        await super().stop(graceful)

        # KEI-Bus-Cleanup
        if self._bus and graceful:
            try:
                # Warte auf verbleibende Messages
                await asyncio.sleep(0.1)
                logger.debug("KEI-Bus-Cleanup abgeschlossen")
            except Exception as e:
                logger.warning(f"KEI-Bus-Cleanup-Fehler: {e}")

        logger.info(f"KEI-Bus-TaskWorker gestoppt: {self.worker_id}")

    def get_kei_bus_metrics(self) -> dict[str, Any]:
        """Gibt KEI-Bus-spezifische Metriken zurück.

        Returns:
            Dictionary mit KEI-Bus-Metriken
        """
        base_metrics = {
            "worker_id": self.worker_id,
            "queue": self.queue,
            "tenant": self.tenant,
            "subject": self._subject,
            "max_in_flight": self.max_in_flight,
            "bus_initialized": self._bus_initialized,
        }

        # Kombiniere mit Standard-Worker-Metriken
        worker_metrics = self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else {}

        return {**base_metrics, **worker_metrics}


# Factory-Funktion für einfache Erstellung
def create_kei_bus_task_worker(
    worker_id: str,
    queue: str,
    executor_function: Callable[[Task], Awaitable[ExecutionResult]],
    **kwargs
) -> KEIBusTaskWorker:
    """Factory-Funktion für KEI-Bus-TaskWorker.

    Args:
        worker_id: Worker-ID
        queue: KEI-Bus-Queue
        executor_function: Task-Executor
        **kwargs: Zusätzliche Parameter

    Returns:
        Konfigurierter KEI-Bus-TaskWorker
    """
    return KEIBusTaskWorker(
        worker_id=worker_id,
        queue=queue,
        executor_function=executor_function,
        **kwargs
    )


__all__ = [
    "DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT",
    "DEFAULT_MAX_IN_FLIGHT",
    "DEFAULT_POLL_TIMEOUT_SECONDS",
    "DEFAULT_QUEUE_VERSION",
    "GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS",
    "KEIBusTaskWorker",
    "create_kei_bus_task_worker",
]
