# backend/agents/operations/operations_coordinator.py
"""Operations-Coordinator für zentrale Agent-Operations."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from ..core.base_agent import AgentConfig, BaseAgent
from ..core.component_manager import ComponentLifecycle, ComponentManager
from ..core.task_executor import TaskContext
from ..enhanced_security import SecurityManager
from ..resilience import ResilienceCoordinator
from ..slo_sla import SLOSLACoordinator
from .constants import (
    COMPONENT_RESILIENCE_COORDINATOR,
    COMPONENT_SECURITY_MANAGER,
    COMPONENT_SLO_SLA_COORDINATOR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_METRICS_COLLECTION_INTERVAL,
    ERROR_COMPONENT_MANAGER_INIT_FAILED,
    ERROR_TASK_QUEUE_FULL,
    ERROR_TASK_QUEUE_NOT_ENABLED,
    HEALTH_QUEUE_WARNING_FULL,
    LOG_PERFORMANCE_MONITORING_ERROR,
    LOG_PERFORMANCE_MONITORING_STARTED,
    LOG_PERFORMANCE_MONITORING_STOPPED,
    LOG_QUEUED_TASK_FAILED,
    LOG_QUEUED_TASK_SUCCESS,
    LOG_TASK_ADDED_TO_QUEUE,
    LOG_TASK_QUEUE_FULL_REJECTED,
    LOG_TASK_QUEUE_PROCESSOR_ERROR,
    LOG_TASK_QUEUE_PROCESSOR_STARTED,
    LOG_TASK_QUEUE_PROCESSOR_STOPPED,
    MONITORING_SLEEP_INTERVAL,
    PERFORMANCE_LOG_FORMAT,
    QUEUE_JOIN_TIMEOUT,
    QUEUE_WARNING_THRESHOLD,
    STANDARD_TASK_SIMULATION_DELAY,
    TASK_ID_PREFIX_QUEUED,
    TASK_ID_SEPARATOR,
)

logger = get_logger(__name__)


@dataclass
class OperationsConfig(AgentConfig):
    """Konfiguration für Operations-Coordinator."""

    # Operations-spezifische Einstellungen
    enable_task_queuing: bool = True
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    enable_task_prioritization: bool = True
    enable_batch_processing: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE

    # Monitoring-Einstellungen
    enable_performance_monitoring: bool = True
    metrics_collection_interval: float = DEFAULT_METRICS_COLLECTION_INTERVAL
    enable_detailed_logging: bool = False


class OperationsCoordinator(BaseAgent[dict[str, Any]], ComponentLifecycle):
    """Operations-Coordinator für zentrale Agent-Operations.

    Koordiniert:
    - Task-Execution mit Resilience
    - Performance-Monitoring
    - Component-Management
    - Security-Integration
    """

    def __init__(self, config: OperationsConfig) -> None:
        """Initialisiert Operations-Coordinator.

        Args:
            config: Operations-Konfiguration
        """
        super().__init__(config)
        self.operations_config: OperationsConfig = config

        # Component-Manager
        self.component_manager: ComponentManager = ComponentManager()

        # Task-Queue
        self._task_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._queue_processor_task: asyncio.Task[None] | None = None

        # Performance-Monitoring
        self._monitoring_task: asyncio.Task[None] | None = None
        self._last_metrics_collection: float = time.time()

    async def _initialize_agent_specific(self) -> None:
        """Operations-spezifische Initialisierung."""
        await self._register_framework_components()

        success = await self.component_manager.initialize_all()
        if not success:
            raise RuntimeError(ERROR_COMPONENT_MANAGER_INIT_FAILED)

        if self.operations_config.enable_task_queuing:
            self._task_queue = asyncio.Queue(maxsize=self.operations_config.max_queue_size)
            self._queue_processor_task = asyncio.create_task(self._process_task_queue())

        if self.operations_config.enable_performance_monitoring:
            self._monitoring_task = asyncio.create_task(self._performance_monitoring_loop())

    async def _register_framework_components(self) -> None:
        """Registriert Framework-Komponenten."""
        if self.config.resilience_config:
            self.component_manager.register_singleton(
                COMPONENT_RESILIENCE_COORDINATOR,
                ResilienceCoordinator,
                lambda: self._create_resilience_coordinator(),
                dependencies=[],
            )

        if self.config.slo_sla_config:
            self.component_manager.register_singleton(
                COMPONENT_SLO_SLA_COORDINATOR,
                SLOSLACoordinator,
                lambda: self._create_slo_sla_coordinator(),
                dependencies=[],
            )

        if self.config.security_config:
            self.component_manager.register_singleton(
                COMPONENT_SECURITY_MANAGER,
                SecurityManager,
                lambda: self._create_security_manager(),
                dependencies=[],
            )

    @staticmethod
    async def _create_coordinator_with_initialization(
        coordinator_class: type[Any],
        config: Any,
        requires_initialization: bool = True
    ) -> Any:
        """Factory-Methode für Coordinator-Erstellung.

        Args:
            coordinator_class: Klasse des Coordinators
            config: Konfiguration für den Coordinator
            requires_initialization: Ob initialize() aufgerufen werden soll

        Returns:
            Initialisierter Coordinator
        """
        coordinator = coordinator_class(config)
        if requires_initialization and hasattr(coordinator, "initialize"):
            await coordinator.initialize()
        return coordinator

    async def _create_resilience_coordinator(self) -> ResilienceCoordinator:
        """Erstellt Resilience-Coordinator."""
        return await self._create_coordinator_with_initialization(
            ResilienceCoordinator,
            self.config.resilience_config
        )

    async def _create_slo_sla_coordinator(self) -> SLOSLACoordinator:
        """Erstellt SLO/SLA-Coordinator."""
        return await self._create_coordinator_with_initialization(
            SLOSLACoordinator,
            self.config.slo_sla_config
        )

    async def _create_security_manager(self) -> SecurityManager:
        """Erstellt Security-Manager."""
        return await self._create_coordinator_with_initialization(
            SecurityManager,
            self.config.security_config,
            requires_initialization=False
        )

    @staticmethod
    def _create_task_context(agent_id: str, task: str, capability: str, timeout: float) -> TaskContext:
        """Erstellt Task-Context für Agent-Task.

        Args:
            agent_id: Agent-ID
            task: Task-Beschreibung
            capability: Capability-Name
            timeout: Timeout in Sekunden

        Returns:
            Task-Context
        """
        return TaskContext(
            task_id=f"{agent_id}_{int(time.time())}",
            task_name=task,
            agent_id=agent_id,
            capability=capability,
            timeout_seconds=timeout,
        )

    @staticmethod
    def _create_task_data(
        framework: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        custom_executor: Callable[..., Awaitable[Any]] | None = None,
        custom_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Erstellt Task-Data-Dictionary.

        Args:
            framework: Framework-Name
            model_name: Model-Name
            temperature: Temperature-Parameter
            max_tokens: Max-Tokens-Parameter
            custom_executor: Custom-Executor
            custom_params: Custom-Parameter

        Returns:
            Task-Data-Dictionary
        """
        return {
            "framework": framework,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "custom_executor": custom_executor,
            "custom_params": custom_params or {},
        }

    async def _execute_with_resilience_if_available(
        self,
        agent_id: str,
        capability: str,
        task: str,
        task_data: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Führt Task mit Resilience aus, falls verfügbar.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            task: Task-Beschreibung
            task_data: Task-Daten
            timeout: Timeout in Sekunden

        Returns:
            Task-Ergebnis
        """
        if self.component_manager.is_initialized(COMPONENT_RESILIENCE_COORDINATOR):
            resilience_coordinator = await self.component_manager.get_component(
                COMPONENT_RESILIENCE_COORDINATOR
            )
            return await resilience_coordinator.execute_with_resilience(
                agent_id=agent_id,
                capability=capability,
                task_name=task,
                executor=self._execute_task_with_custom_executor,
                task_data=task_data,
                timeout=timeout,
            )
        return await self.execute_task(task, task_data, capability, timeout)

    async def execute_agent_task_with_resilience(
        self,
        agent_id: str,
        task: str,
        capability: str,
        timeout: float,
        framework: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        custom_executor: Callable[..., Awaitable[Any]] | None = None,
        custom_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Führt Agent-Task mit vollständiger Resilience aus.

        Args:
            agent_id: Agent-ID
            task: Task-Beschreibung
            capability: Capability-Name
            timeout: Timeout in Sekunden
            framework: Framework-Name
            model_name: Model-Name
            temperature: Temperature-Parameter
            max_tokens: Max-Tokens-Parameter
            custom_executor: Optional Custom-Executor
            custom_params: Optional Custom-Parameter

        Returns:
            Task-Ergebnis
        """
        task_data = OperationsCoordinator._create_task_data(
            framework, model_name, temperature, max_tokens, custom_executor, custom_params
        )

        return await self._execute_with_resilience_if_available(
            agent_id, capability, task, task_data, timeout
        )

    async def _execute_task_with_custom_executor(self, task_data: dict[str, Any]) -> Any:
        """Führt Task mit Custom-Executor aus."""
        custom_executor = task_data.get("custom_executor")
        custom_params = task_data.get("custom_params", {})

        if custom_executor:
            return await custom_executor(**custom_params)
        return await OperationsCoordinator._execute_standard_task(task_data)

    @staticmethod
    async def _execute_standard_task(task_data: dict[str, Any]) -> Any:
        """Führt Standard-Task aus.

        Args:
            task_data: Task-Daten mit Framework und Model-Informationen

        Returns:
            Task-Ergebnis mit Status und Metadaten
        """
        await asyncio.sleep(STANDARD_TASK_SIMULATION_DELAY)

        return {
            "framework": task_data.get("framework"),
            "model_name": task_data.get("model_name"),
            "result": "Standard-Task erfolgreich ausgeführt",
        }

    async def _execute_agent_task(self, task_data: dict[str, Any]) -> Any:
        """Agent-spezifische Task-Ausführung."""
        return await self._execute_task_with_custom_executor(task_data)

    async def queue_task(self, task_name: str, task_data: dict[str, Any], priority: int = 0) -> str:
        """Fügt Task zur Queue hinzu.

        Args:
            task_name: Task-Name
            task_data: Task-Daten
            priority: Task-Priorität (höher = wichtiger)

        Returns:
            Task-ID
        """
        if not self._task_queue:
            raise RuntimeError(ERROR_TASK_QUEUE_NOT_ENABLED)

        task_context = TaskContext(
            task_id=f"{TASK_ID_PREFIX_QUEUED}{TASK_ID_SEPARATOR}{int(time.time())}",
            task_name=task_name,
            agent_id=self.config.agent_id
        )

        queue_item = {
            "task_context": task_context,
            "task_data": task_data,
            "priority": priority,
            "queued_at": time.time(),
        }

        try:
            await self._task_queue.put(queue_item)
            logger.debug(LOG_TASK_ADDED_TO_QUEUE.format(task_name=task_name, task_id=task_context.task_id))
            return task_context.task_id

        except asyncio.QueueFull:
            logger.error(LOG_TASK_QUEUE_FULL_REJECTED.format(task_name=task_name))
            raise RuntimeError(ERROR_TASK_QUEUE_FULL)

    async def _execute_queued_task(self, queue_item: dict[str, Any]) -> None:
        """Führt einen einzelnen Task aus der Queue aus.

        Args:
            queue_item: Queue-Item mit task_context und task_data
        """
        task_context = queue_item["task_context"]
        task_data = queue_item["task_data"]

        try:
            await self.execute_task(
                task_context.task_name,
                task_data,
                task_context.capability,
                task_context.timeout_seconds,
            )
            logger.debug(LOG_QUEUED_TASK_SUCCESS.format(task_name=task_context.task_name))

        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError) as e:
            logger.error(LOG_QUEUED_TASK_FAILED.format(task_name=task_context.task_name, error=f"Validierungsfehler: {e}"))
        except (ConnectionError, TimeoutError) as e:
            logger.error(LOG_QUEUED_TASK_FAILED.format(task_name=task_context.task_name, error=f"Verbindungsproblem: {e}"))
        except Exception as e:
            logger.error(LOG_QUEUED_TASK_FAILED.format(task_name=task_context.task_name, error=f"Unerwarteter Fehler: {e}"))

        finally:
            self._task_queue.task_done()

    async def _process_task_queue(self) -> None:
        """Verarbeitet Task-Queue asynchron."""
        logger.info(LOG_TASK_QUEUE_PROCESSOR_STARTED)

        while True:
            try:
                queue_item = await self._task_queue.get()
                await self._execute_queued_task(queue_item)

            except asyncio.CancelledError:
                logger.info(LOG_TASK_QUEUE_PROCESSOR_STOPPED)
                break
            except (ValueError, TypeError) as e:
                logger.error(LOG_TASK_QUEUE_PROCESSOR_ERROR.format(error=f"Validierungsfehler: {e}"))
                await asyncio.sleep(MONITORING_SLEEP_INTERVAL)
            except (ConnectionError, TimeoutError) as e:
                logger.error(LOG_TASK_QUEUE_PROCESSOR_ERROR.format(error=f"Verbindungsproblem: {e}"))
                await asyncio.sleep(MONITORING_SLEEP_INTERVAL)
            except Exception as e:
                logger.error(LOG_TASK_QUEUE_PROCESSOR_ERROR.format(error=f"Unerwarteter Fehler: {e}"))
                await asyncio.sleep(MONITORING_SLEEP_INTERVAL)

    async def _performance_monitoring_loop(self) -> None:
        """Performance-Monitoring-Loop."""
        logger.info(LOG_PERFORMANCE_MONITORING_STARTED)

        while True:
            try:
                await asyncio.sleep(self.operations_config.metrics_collection_interval)

                metrics = await self.get_performance_metrics()

                if self.operations_config.enable_detailed_logging:
                    agent_metrics = metrics["agent_metrics"]
                    logger.info(PERFORMANCE_LOG_FORMAT.format(
                        total_tasks=agent_metrics["total_tasks"],
                        success_rate=agent_metrics["success_rate"],
                        avg_execution_time=agent_metrics["avg_execution_time"]
                    ))

                self._last_metrics_collection = time.time()

            except asyncio.CancelledError:
                logger.info(LOG_PERFORMANCE_MONITORING_STOPPED)
                break
            except (ValueError, TypeError) as e:
                logger.error(LOG_PERFORMANCE_MONITORING_ERROR.format(error=f"Validierungsfehler: {e}"))
            except (ConnectionError, TimeoutError) as e:
                logger.error(LOG_PERFORMANCE_MONITORING_ERROR.format(error=f"Verbindungsproblem: {e}"))
            except Exception as e:
                logger.error(LOG_PERFORMANCE_MONITORING_ERROR.format(error=f"Unerwarteter Fehler: {e}"))

    def _get_task_queue_health(self) -> dict[str, Any]:
        """Erstellt Task-Queue-Health-Status."""
        return {
            "enabled": self._task_queue is not None,
            "size": self._task_queue.qsize() if self._task_queue else 0,
            "max_size": self.operations_config.max_queue_size,
        }

    def _get_monitoring_health(self) -> dict[str, Any]:
        """Erstellt Monitoring-Health-Status."""
        return {
            "enabled": self._monitoring_task is not None,
            "last_collection": self._last_metrics_collection,
            "collection_interval": self.operations_config.metrics_collection_interval,
        }

    def _check_task_queue_health_warnings(self, health_status: dict[str, Any]) -> None:
        """Prüft Task-Queue auf Health-Warnings.

        Args:
            health_status: Health-Status-Dictionary zum Aktualisieren
        """
        if (
            self._task_queue
            and self._task_queue.qsize() >= self.operations_config.max_queue_size * QUEUE_WARNING_THRESHOLD
        ):
            health_status["healthy"] = False
            health_status["task_queue"]["warning"] = HEALTH_QUEUE_WARNING_FULL

    async def _agent_specific_health_check(self) -> dict[str, Any]:
        """Operations-spezifische Health-Checks."""
        component_health = await self.component_manager.health_check_all()

        health_status = {
            "healthy": True,
            "component_manager": component_health,
            "task_queue": self._get_task_queue_health(),
            "monitoring": self._get_monitoring_health(),
        }

        if not component_health["healthy"]:
            health_status["healthy"] = False

        self._check_task_queue_health_warnings(health_status)

        return health_status

    async def _get_agent_specific_metrics(self) -> dict[str, Any]:
        """Operations-spezifische Metriken."""
        metrics = {
            "component_metrics": await self.component_manager.get_component_metrics(),
            "task_queue_metrics": {
                "enabled": self._task_queue is not None,
                "current_size": self._task_queue.qsize() if self._task_queue else 0,
                "max_size": self.operations_config.max_queue_size,
                "utilization": (
                    (self._task_queue.qsize() / self.operations_config.max_queue_size)
                    if self._task_queue
                    else 0
                ),
            },
        }

        return metrics

    async def _get_component_metrics(self, component_name: str) -> dict[str, Any]:
        """Holt Metriken von einer Komponente.

        Args:
            component_name: Name der Komponente

        Returns:
            Metriken der Komponente oder leeres Dict
        """
        if self.component_manager.is_initialized(component_name):
            component = await self.component_manager.get_component(component_name)
            if hasattr(component, "get_metrics"):
                return await component.get_metrics()
        return {}

    async def get_resilience_metrics(self) -> dict[str, Any]:
        """Holt Resilience-Metriken."""
        return await self._get_component_metrics(COMPONENT_RESILIENCE_COORDINATOR)

    async def get_slo_sla_metrics(self) -> dict[str, Any]:
        """Holt SLO/SLA-Metriken."""
        return await self._get_component_metrics(COMPONENT_SLO_SLA_COORDINATOR)

    async def get_security_metrics(self) -> dict[str, Any]:
        """Holt Security-Metriken."""
        return await self._get_component_metrics(COMPONENT_SECURITY_MANAGER)

    async def _agent_specific_cleanup(self) -> None:
        """Operations-spezifisches Cleanup."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass

        if self._task_queue:
            try:
                await asyncio.wait_for(self._task_queue.join(), timeout=QUEUE_JOIN_TIMEOUT)
            except TimeoutError:
                try:
                    while not self._task_queue.empty():
                        _ = self._task_queue.get_nowait()
                        self._task_queue.task_done()
                except Exception:
                    pass

        await self.component_manager.close_all()
