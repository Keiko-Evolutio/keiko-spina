# backend/agents/core/base_agent.py
"""Basis-Agent-Klasse für Keiko Personal Assistant

Implementiert gemeinsame Funktionalität für alle Agents
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..enhanced_security import SecurityConfig, SecurityManager
from ..monitoring import HealthConfig, HealthMonitor, PerformanceConfig, PerformanceMonitor
from ..resilience import ResilienceConfig, ResilienceCoordinator
from ..slo_sla import SLOSLAConfig, SLOSLACoordinator
from .utils import (
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    MetricsCollector,
    ValidationError,
    async_error_context,
    generate_task_id,
    get_module_logger,
    validate_positive_number,
    validate_required_field,
    with_async_error_handling,
)

logger = get_module_logger(__name__)

T = TypeVar("T")


class AgentMetrics(MetricsCollector):
    """Agent-Performance-Metriken basierend auf gemeinsamer MetricsCollector-Basis."""

    def update_task_result(self, success: bool, execution_time: float) -> None:
        """Aktualisiert Task-Metriken.

        Args:
            success: Ob Task erfolgreich war
            execution_time: Ausführungszeit in Sekunden
        """
        self.record_operation(success, execution_time)

    @property
    def total_tasks(self) -> int:
        """Alias für total_operations für Backward-Compatibility."""
        return self.total_operations

    @property
    def successful_tasks(self) -> int:
        """Alias für successful_operations für Backward-Compatibility."""
        return self.successful_operations

    @property
    def failed_tasks(self) -> int:
        """Alias für failed_operations für Backward-Compatibility."""
        return self.failed_operations

    @property
    def avg_execution_time(self) -> float:
        """Alias für average_execution_time für Backward-Compatibility."""
        return self.average_execution_time


@dataclass
class AgentConfig:
    """Basis-Konfiguration für alle Agents."""

    agent_id: str
    framework: str = "foundry"
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1000

    resilience_config: ResilienceConfig | None = None
    slo_sla_config: SLOSLAConfig | None = None
    security_config: SecurityConfig | None = None
    performance_config: PerformanceConfig | None = None
    health_config: HealthConfig | None = None
    max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    log_level: str = "INFO"
    enable_metrics: bool = True

    def __post_init__(self) -> None:
        """Post-Initialisierung mit Validierung."""
        try:
            validate_required_field(self.agent_id, "agent_id")
            validate_positive_number(self.max_concurrent_tasks, "max_concurrent_tasks")
            validate_positive_number(self.task_timeout_seconds, "task_timeout_seconds")
        except ValidationError as e:
            raise ValueError(str(e)) from e


class BaseAgent(ABC, Generic[T]):
    """Abstrakte Basis-Klasse für alle Agents.

    Implementiert gemeinsame Funktionalität:
    - Lifecycle-Management (initialize, close)
    - Task-Execution mit Resilience
    - Performance-Monitoring
    - Error-Handling
    - Health-Checks
    """

    def __init__(self, config: AgentConfig):
        """Initialisiert Base-Agent.

        Args:
            config: Agent-Konfiguration
        """
        self.config = config
        self.metrics = AgentMetrics()
        self.is_initialized = False
        self.start_time = time.time()

        self._resilience_coordinator: ResilienceCoordinator | None = None
        self._slo_sla_coordinator: SLOSLACoordinator | None = None
        self._security_manager: SecurityManager | None = None
        self._performance_monitor: PerformanceMonitor | None = None
        self._health_monitor: HealthMonitor | None = None
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def initialize(self) -> bool:
        """Initialisiert Agent mit allen Komponenten.

        Returns:
            True wenn erfolgreich initialisiert
        """
        if self.is_initialized:
            logger.warning(f"Agent {self.config.agent_id} bereits initialisiert")
            return True

        try:
            await self._initialize_framework_components()
            await self._initialize_agent_specific()
            health_status = await self.health_check()
            if not health_status["healthy"]:
                raise RuntimeError(f"Health-Check fehlgeschlagen: {health_status}")

            self.is_initialized = True
            logger.info(f"Agent {self.config.agent_id} erfolgreich initialisiert")
            return True

        except Exception as e:
            logger.error(f"Agent-Initialisierung fehlgeschlagen: {e}")
            return False

    async def _initialize_framework_components(self) -> None:
        """Initialisiert Framework-Komponenten."""
        if self.config.resilience_config:
            self._resilience_coordinator = ResilienceCoordinator(self.config.resilience_config)
            await self._resilience_coordinator.initialize()

        if self.config.slo_sla_config:
            self._slo_sla_coordinator = SLOSLACoordinator(self.config.slo_sla_config)
            await self._slo_sla_coordinator.initialize()

        if self.config.security_config:
            self._security_manager = SecurityManager(self.config.security_config)
            logger.debug(f"Security Manager für Agent {self.config.agent_id} initialisiert")

        if self.config.performance_config:
            self._performance_monitor = PerformanceMonitor(self.config.performance_config)
            logger.debug(f"Performance Monitor für Agent {self.config.agent_id} initialisiert")

        if self.config.health_config:
            self._health_monitor = HealthMonitor(self.config.health_config)
            self._health_monitor.register_component_health_check(
                name=f"agent_{self.config.agent_id}", component=self, critical=True
            )
            await self._health_monitor.start_monitoring()
            logger.debug(f"Health Monitor für Agent {self.config.agent_id} initialisiert")

    @abstractmethod
    async def _initialize_agent_specific(self) -> None:
        """Agent-spezifische Initialisierung (zu implementieren von Subklassen)."""

    @with_async_error_handling(
        error_message="Task-Ausführung fehlgeschlagen",
        raise_on_error=True
    )
    async def execute_task(
        self,
        task_name: str,
        task_data: dict[str, Any],
        capability: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Führt Task mit Resilience und Monitoring aus.

        Args:
            task_name: Name des Tasks
            task_data: Task-Daten
            capability: Optional Capability-Name
            timeout: Optional Timeout

        Returns:
            Task-Ergebnis
        """
        if not self.is_initialized:
            raise RuntimeError(f"Agent {self.config.agent_id} nicht initialisiert")

        task_id = generate_task_id()
        timeout = timeout or self.config.task_timeout_seconds

        async with self._task_semaphore:
            return await self._execute_task_with_monitoring(
                task_id, task_name, task_data, capability, timeout
            )

    async def _execute_task_with_monitoring(
        self,
        task_id: str,
        task_name: str,
        task_data: dict[str, Any],
        capability: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Führt Task mit vollständigem Monitoring aus."""
        async with async_error_context(
            operation_name=f"Task {task_name}",
            log_errors=True,
            raise_on_error=False
        ) as context:
            if self._resilience_coordinator:
                result = await self._execute_with_resilience(
                    task_id, task_name, task_data, capability, timeout
                )
            else:
                result = await self._execute_direct(task_id, task_name, task_data, timeout)

            context["result"] = result

        self.metrics.update_task_result(context["success"], context["duration"])

        return {
            "success": context["success"],
            "result": context["result"],
            "error": str(context["error"]) if context["error"] else None,
            "execution_time": context["duration"],
            "task_id": task_id,
        }

    async def _execute_with_resilience(
        self,
        task_id: str,
        _task_name: str,
        task_data: dict[str, Any],
        capability: str | None,
        timeout: float,
    ) -> Any:
        """Führt Task mit Resilience-Features aus."""
        async with self._resilience_coordinator.execute_with_resilience(
            self.config.agent_id,
            capability or "default",
            self._execute_agent_task,
            task_data,
            request_id=task_id,
            timeout=timeout,
        ) as result:
            return result

    async def _execute_direct(
        self, _task_id: str, _task_name: str, task_data: dict[str, Any], timeout: float
    ) -> Any:
        """Führt Task direkt ohne Resilience aus."""
        return await asyncio.wait_for(self._execute_agent_task(task_data), timeout=timeout)

    @abstractmethod
    async def _execute_agent_task(self, task_data: dict[str, Any]) -> Any:
        """Agent-spezifische Task-Ausführung (zu implementieren von Subklassen)."""

    async def health_check(self) -> dict[str, Any]:
        """Führt umfassenden Health-Check durch.

        Returns:
            Health-Status
        """
        health_status = self._create_base_health_status()

        await self._check_framework_components_health(health_status)
        await self._check_agent_specific_health(health_status)

        return health_status

    def _create_base_health_status(self) -> dict[str, Any]:
        """Erstellt Basis-Health-Status."""
        return {
            "healthy": True,
            "agent_id": self.config.agent_id,
            "initialized": self.is_initialized,
            "uptime_seconds": time.time() - self.start_time,
            "components": {},
            "metrics": self.metrics.get_metrics_dict(),
        }

    async def _check_framework_components_health(self, health_status: dict[str, Any]) -> None:
        """Prüft Health-Status der Framework-Komponenten."""
        components_to_check = [
            ("resilience", self._resilience_coordinator),
            ("slo_sla", self._slo_sla_coordinator),
            ("security", self._security_manager),
        ]

        for component_name, component in components_to_check:
            if component:
                component_health = await component.health_check()
                health_status["components"][component_name] = component_health
                if not component_health["healthy"]:
                    health_status["healthy"] = False

    async def _check_agent_specific_health(self, health_status: dict[str, Any]) -> None:
        """Prüft Agent-spezifische Health-Checks."""
        agent_health = await self._agent_specific_health_check()
        health_status["components"]["agent_specific"] = agent_health

        if not agent_health["healthy"]:
            health_status["healthy"] = False

        if "example_type" in agent_health:
            health_status["example_type"] = agent_health["example_type"]

    @abstractmethod
    async def _agent_specific_health_check(self) -> dict[str, Any]:
        """Agent-spezifische Health-Checks (zu implementieren von Subklassen)."""

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Holt umfassende Performance-Metriken.

        Returns:
            Performance-Metriken
        """
        metrics = {"agent_metrics": self._get_base_agent_metrics()}

        await self._collect_framework_metrics(metrics)
        await self._collect_agent_specific_metrics(metrics)

        return metrics

    def _get_base_agent_metrics(self) -> dict[str, Any]:
        """Sammelt Basis-Agent-Metriken."""
        base_metrics = self.metrics.get_metrics_dict()
        base_metrics["uptime_seconds"] = time.time() - self.start_time
        return base_metrics

    async def _collect_framework_metrics(self, metrics: dict[str, Any]) -> None:
        """Sammelt Metriken der Framework-Komponenten."""
        framework_components = [
            ("resilience_metrics", self._resilience_coordinator),
            ("slo_sla_metrics", self._slo_sla_coordinator),
            ("security_metrics", self._security_manager),
        ]

        for metric_key, component in framework_components:
            if component:
                metrics[metric_key] = await component.get_metrics()

    async def _collect_agent_specific_metrics(self, metrics: dict[str, Any]) -> None:
        """Sammelt Agent-spezifische Metriken."""
        agent_metrics = await self._get_agent_specific_metrics()
        metrics["agent_specific_metrics"] = agent_metrics

        if hasattr(self, "example_config") and "example_specific" in agent_metrics:
            metrics["example_specific_metrics"] = agent_metrics["example_specific"]

    @abstractmethod
    async def _get_agent_specific_metrics(self) -> dict[str, Any]:
        """Agent-spezifische Metriken (zu implementieren von Subklassen)."""

    @asynccontextmanager
    async def task_context(self, task_name: str):
        """Context-Manager für Task-Ausführung mit automatischem Cleanup."""
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.debug(f"Starte Task {task_name} (ID: {task_id})")
            yield task_id

        except Exception as e:
            logger.error(f"Task {task_name} fehlgeschlagen: {e}")
            raise

        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Task {task_name} abgeschlossen ({execution_time:.2f}s)")

    async def close(self) -> None:
        """Schließt Agent und alle Komponenten."""
        logger.info(f"Schließe Agent {self.config.agent_id}")

        if self._active_tasks:
            logger.info(f"Warte auf {len(self._active_tasks)} aktive Tasks")
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)

        if self._security_manager:
            await self._security_manager.close()

        if self._slo_sla_coordinator:
            await self._slo_sla_coordinator.close()

        if self._resilience_coordinator:
            await self._resilience_coordinator.close()

        await self._agent_specific_cleanup()

        self.is_initialized = False
        logger.info(f"Agent {self.config.agent_id} geschlossen")

    @abstractmethod
    async def _agent_specific_cleanup(self) -> None:
        """Agent-spezifisches Cleanup (zu implementieren von Subklassen)."""
