# backend/agents/resilience/request_budgets.py
"""Request Deadlines & Budgets für Personal Assistant

Implementiert Request-Deadline-Management mit konfigurierbaren Timeouts
und Request-Budget-Tracking für CPU-Zeit, Memory und Network-Calls.
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)


class ResourceType(str, Enum):
    """Typen von Ressourcen für Budget-Tracking."""

    CPU_TIME = "cpu_time"
    MEMORY = "memory"
    NETWORK_CALLS = "network_calls"
    DISK_IO = "disk_io"
    CUSTOM = "custom"


@dataclass
class BudgetConfig:
    """Konfiguration für Request-Budgets."""

    # Deadline-Konfiguration
    default_timeout: float = 30.0
    capability_timeouts: dict[str, float] = field(default_factory=dict)

    # Resource-Budgets
    max_cpu_time_ms: float = 5000.0  # 5 Sekunden CPU-Zeit
    max_memory_mb: float = 512.0  # 512 MB Memory
    max_network_calls: int = 10  # 10 Network-Calls
    max_disk_io_mb: float = 100.0  # 100 MB Disk-I/O

    # Custom Resource-Budgets
    custom_budgets: dict[str, float] = field(default_factory=dict)

    # Budget-Exhaustion-Handling
    graceful_degradation: bool = True
    budget_warning_threshold: float = 0.8  # 80% Budget-Nutzung

    # Callbacks
    on_budget_warning: Callable[[str, str, float], Awaitable[None]] | None = None
    on_budget_exhausted: Callable[[str, str, float], Awaitable[None]] | None = None
    on_deadline_exceeded: Callable[[str, float], Awaitable[None]] | None = None


@dataclass
class ResourceUsage:
    """Aktuelle Ressourcen-Nutzung."""

    cpu_time_ms: float = 0.0
    memory_mb: float = 0.0
    network_calls: int = 0
    disk_io_mb: float = 0.0
    custom_usage: dict[str, float] = field(default_factory=dict)

    def get_usage(self, resource_type: ResourceType) -> float:
        """Holt Nutzung für Ressourcen-Typ."""
        if resource_type == ResourceType.CPU_TIME:
            return self.cpu_time_ms
        if resource_type == ResourceType.MEMORY:
            return self.memory_mb
        if resource_type == ResourceType.NETWORK_CALLS:
            return float(self.network_calls)
        if resource_type == ResourceType.DISK_IO:
            return self.disk_io_mb
        return 0.0

    def add_usage(self, resource_type: ResourceType, amount: float):
        """Fügt Ressourcen-Nutzung hinzu."""
        if resource_type == ResourceType.CPU_TIME:
            self.cpu_time_ms += amount
        elif resource_type == ResourceType.MEMORY:
            self.memory_mb = max(self.memory_mb, amount)  # Peak Memory
        elif resource_type == ResourceType.NETWORK_CALLS:
            self.network_calls += int(amount)
        elif resource_type == ResourceType.DISK_IO:
            self.disk_io_mb += amount


class ResourceTracker:
    """Tracker für Ressourcen-Nutzung während Request-Ausführung."""

    def __init__(self, request_id: str):
        """Initialisiert Resource-Tracker.

        Args:
            request_id: Eindeutige Request-ID
        """
        self.request_id = request_id
        self.usage = ResourceUsage()

        # Tracking-State
        self._start_time = time.time()
        self._start_cpu_time = time.process_time()
        self._start_memory = self._get_memory_usage()
        self._network_calls = 0
        self._disk_io_start = self._get_disk_io()

        # Thread-Safety
        self._lock = threading.RLock()

        # Metrics
        self._metrics_collector = MetricsCollector()

    def _get_memory_usage(self) -> float:
        """Holt aktuelle Memory-Nutzung in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def _get_disk_io(self) -> float:
        """Holt aktuelle Disk-I/O in MB."""
        try:
            process = psutil.Process()
            io_counters = process.io_counters()
            return (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)
        except Exception:
            return 0.0

    def record_network_call(self):
        """Zeichnet Network-Call auf."""
        with self._lock:
            self._network_calls += 1
            self.usage.network_calls = self._network_calls

    def record_custom_usage(self, resource_name: str, amount: float):
        """Zeichnet Custom-Ressourcen-Nutzung auf."""
        with self._lock:
            if resource_name not in self.usage.custom_usage:
                self.usage.custom_usage[resource_name] = 0.0
            self.usage.custom_usage[resource_name] += amount

    def update_usage(self):
        """Aktualisiert aktuelle Ressourcen-Nutzung."""
        with self._lock:
            _current_time = time.time()
            current_cpu_time = time.process_time()
            current_memory = self._get_memory_usage()
            current_disk_io = self._get_disk_io()

            # CPU-Zeit (in Millisekunden)
            self.usage.cpu_time_ms = (current_cpu_time - self._start_cpu_time) * 1000

            # Memory (Peak-Nutzung)
            memory_diff = current_memory - self._start_memory
            if memory_diff > 0:
                self.usage.memory_mb = max(self.usage.memory_mb, memory_diff)

            # Disk-I/O
            disk_io_diff = current_disk_io - self._disk_io_start
            if disk_io_diff > 0:
                self.usage.disk_io_mb = disk_io_diff

    def get_current_usage(self) -> ResourceUsage:
        """Holt aktuelle Ressourcen-Nutzung.

        Returns:
            Aktuelle Ressourcen-Nutzung
        """
        self.update_usage()
        return self.usage


@dataclass
class RequestBudget:
    """Budget für einzelnen Request."""

    request_id: str
    capability: str
    agent_id: str

    # Deadline
    deadline: float  # Unix-Timestamp
    timeout: float  # Timeout in Sekunden

    # Resource-Budgets
    max_cpu_time_ms: float
    max_memory_mb: float
    max_network_calls: int
    max_disk_io_mb: float
    custom_budgets: dict[str, float] = field(default_factory=dict)

    # Tracking
    tracker: ResourceTracker = field(init=False)
    start_time: float = field(default_factory=time.time)

    def __post_init__(self):
        """Post-Initialisierung."""
        self.tracker = ResourceTracker(self.request_id)

    def is_deadline_exceeded(self) -> bool:
        """Prüft ob Deadline überschritten wurde."""
        return time.time() > self.deadline

    def get_remaining_time(self) -> float:
        """Holt verbleibende Zeit bis Deadline."""
        return max(0.0, self.deadline - time.time())

    def is_budget_exceeded(self, resource_type: ResourceType) -> bool:
        """Prüft ob Budget für Ressourcen-Typ überschritten wurde."""
        current_usage = self.tracker.get_current_usage()

        if resource_type == ResourceType.CPU_TIME:
            return current_usage.cpu_time_ms > self.max_cpu_time_ms
        if resource_type == ResourceType.MEMORY:
            return current_usage.memory_mb > self.max_memory_mb
        if resource_type == ResourceType.NETWORK_CALLS:
            return current_usage.network_calls > self.max_network_calls
        if resource_type == ResourceType.DISK_IO:
            return current_usage.disk_io_mb > self.max_disk_io_mb

        return False

    def get_budget_utilization(self, resource_type: ResourceType) -> float:
        """Holt Budget-Auslastung für Ressourcen-Typ (0.0 - 1.0)."""
        current_usage = self.tracker.get_current_usage()

        if resource_type == ResourceType.CPU_TIME:
            return min(1.0, current_usage.cpu_time_ms / self.max_cpu_time_ms)
        if resource_type == ResourceType.MEMORY:
            return min(1.0, current_usage.memory_mb / self.max_memory_mb)
        if resource_type == ResourceType.NETWORK_CALLS:
            return min(1.0, current_usage.network_calls / self.max_network_calls)
        if resource_type == ResourceType.DISK_IO:
            return min(1.0, current_usage.disk_io_mb / self.max_disk_io_mb)

        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Holt Budget-Metriken."""
        current_usage = self.tracker.get_current_usage()

        return {
            "request_id": self.request_id,
            "capability": self.capability,
            "agent_id": self.agent_id,
            "deadline_exceeded": self.is_deadline_exceeded(),
            "remaining_time": self.get_remaining_time(),
            "usage": {
                "cpu_time_ms": current_usage.cpu_time_ms,
                "memory_mb": current_usage.memory_mb,
                "network_calls": current_usage.network_calls,
                "disk_io_mb": current_usage.disk_io_mb,
                "custom": current_usage.custom_usage,
            },
            "utilization": {
                "cpu_time": self.get_budget_utilization(ResourceType.CPU_TIME),
                "memory": self.get_budget_utilization(ResourceType.MEMORY),
                "network_calls": self.get_budget_utilization(ResourceType.NETWORK_CALLS),
                "disk_io": self.get_budget_utilization(ResourceType.DISK_IO),
            },
            "budget_exceeded": {
                "cpu_time": self.is_budget_exceeded(ResourceType.CPU_TIME),
                "memory": self.is_budget_exceeded(ResourceType.MEMORY),
                "network_calls": self.is_budget_exceeded(ResourceType.NETWORK_CALLS),
                "disk_io": self.is_budget_exceeded(ResourceType.DISK_IO),
            },
        }


class BudgetExhaustionHandler:
    """Handler für Budget-Exhaustion mit graceful Degradation."""

    def __init__(self, config: BudgetConfig):
        """Initialisiert Budget-Exhaustion-Handler.

        Args:
            config: Budget-Konfiguration
        """
        self.config = config
        self._metrics_collector = MetricsCollector()

    async def handle_budget_warning(
        self, budget: RequestBudget, resource_type: ResourceType, utilization: float
    ):
        """Behandelt Budget-Warning.

        Args:
            budget: Request-Budget
            resource_type: Ressourcen-Typ
            utilization: Aktuelle Auslastung (0.0 - 1.0)
        """
        logger.warning(
            f"Budget-Warning für {budget.request_id} ({budget.capability}): "
            f"{resource_type.value} bei {utilization:.1%} Auslastung"
        )

        # Callback
        if self.config.on_budget_warning:
            await self.config.on_budget_warning(budget.request_id, resource_type.value, utilization)

        # Metrics
        self._metrics_collector.increment_counter(
            "budget.warnings",
            tags={
                "capability": budget.capability,
                "agent_id": budget.agent_id,
                "resource_type": resource_type.value,
            },
        )

    async def handle_budget_exhaustion(
        self, budget: RequestBudget, resource_type: ResourceType, utilization: float
    ):
        """Behandelt Budget-Exhaustion.

        Args:
            budget: Request-Budget
            resource_type: Ressourcen-Typ
            utilization: Aktuelle Auslastung (>= 1.0)
        """
        logger.error(
            f"Budget erschöpft für {budget.request_id} ({budget.capability}): "
            f"{resource_type.value} bei {utilization:.1%} Auslastung"
        )

        # Callback
        if self.config.on_budget_exhausted:
            await self.config.on_budget_exhausted(
                budget.request_id, resource_type.value, utilization
            )

        # Metrics
        self._metrics_collector.increment_counter(
            "budget.exhausted",
            tags={
                "capability": budget.capability,
                "agent_id": budget.agent_id,
                "resource_type": resource_type.value,
            },
        )

        # Graceful Degradation
        if self.config.graceful_degradation:
            await self._apply_graceful_degradation(budget, resource_type)
        else:
            raise BudgetExhaustedError(
                f"Budget für {resource_type.value} erschöpft: {utilization:.1%}"
            )

    async def handle_deadline_exceeded(self, budget: RequestBudget):
        """Behandelt überschrittene Deadline.

        Args:
            budget: Request-Budget
        """
        exceeded_by = time.time() - budget.deadline

        logger.error(
            f"Deadline überschritten für {budget.request_id} ({budget.capability}): "
            f"{exceeded_by:.2f}s zu spät"
        )

        # Callback
        if self.config.on_deadline_exceeded:
            await self.config.on_deadline_exceeded(budget.request_id, exceeded_by)

        # Metrics
        self._metrics_collector.increment_counter(
            "budget.deadline_exceeded",
            tags={"capability": budget.capability, "agent_id": budget.agent_id},
        )

        self._metrics_collector.record_histogram(
            "budget.deadline_exceeded_by",
            exceeded_by,
            tags={"capability": budget.capability, "agent_id": budget.agent_id},
        )

        raise DeadlineExceededError(f"Request-Deadline um {exceeded_by:.2f}s überschritten")

    async def _apply_graceful_degradation(self, budget: RequestBudget, resource_type: ResourceType):
        """Wendet graceful Degradation an.

        Args:
            budget: Request-Budget
            resource_type: Erschöpfter Ressourcen-Typ
        """
        # Implementiere capability-spezifische Degradation
        if resource_type == ResourceType.NETWORK_CALLS:
            # Reduziere Network-Calls durch Caching
            logger.info(f"Aktiviere aggressive Caching für {budget.request_id}")

        elif resource_type == ResourceType.MEMORY:
            # Reduziere Memory-Nutzung
            logger.info(f"Aktiviere Memory-Optimierung für {budget.request_id}")

        elif resource_type == ResourceType.CPU_TIME:
            # Reduziere CPU-intensive Operationen
            logger.info(f"Aktiviere CPU-Optimierung für {budget.request_id}")


class BudgetManager:
    """Manager für Request-Budgets und Deadlines."""

    def __init__(self, config: BudgetConfig | None = None):
        """Initialisiert Budget-Manager.

        Args:
            config: Budget-Konfiguration
        """
        self.config = config or BudgetConfig()
        self._active_budgets: dict[str, RequestBudget] = {}
        self._exhaustion_handler = BudgetExhaustionHandler(self.config)
        self._lock = threading.RLock()

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 1.0  # 1 Sekunde

        # Metrics
        self._metrics_collector = MetricsCollector()

    def create_budget(
        self, request_id: str, capability: str, agent_id: str, timeout: float | None = None
    ) -> RequestBudget:
        """Erstellt Budget für Request.

        Args:
            request_id: Eindeutige Request-ID
            capability: Capability-Name
            agent_id: Agent-ID
            timeout: Optional spezifisches Timeout

        Returns:
            Request-Budget
        """
        # Timeout bestimmen
        if timeout is None:
            timeout = self.config.capability_timeouts.get(capability, self.config.default_timeout)

        # Budget erstellen
        budget = RequestBudget(
            request_id=request_id,
            capability=capability,
            agent_id=agent_id,
            deadline=time.time() + timeout,
            timeout=timeout,
            max_cpu_time_ms=self.config.max_cpu_time_ms,
            max_memory_mb=self.config.max_memory_mb,
            max_network_calls=self.config.max_network_calls,
            max_disk_io_mb=self.config.max_disk_io_mb,
            custom_budgets=self.config.custom_budgets.copy(),
        )

        with self._lock:
            self._active_budgets[request_id] = budget

        # Starte Monitoring falls noch nicht aktiv
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        return budget

    def get_budget(self, request_id: str) -> RequestBudget | None:
        """Holt Budget für Request.

        Args:
            request_id: Request-ID

        Returns:
            Request-Budget oder None
        """
        with self._lock:
            return self._active_budgets.get(request_id)

    def remove_budget(self, request_id: str) -> RequestBudget | None:
        """Entfernt Budget für Request.

        Args:
            request_id: Request-ID

        Returns:
            Entferntes Budget oder None
        """
        with self._lock:
            return self._active_budgets.pop(request_id, None)

    @asynccontextmanager
    async def track_request(
        self, request_id: str, capability: str, agent_id: str, timeout: float | None = None
    ):
        """Context Manager für Request-Budget-Tracking.

        Args:
            request_id: Request-ID
            capability: Capability-Name
            agent_id: Agent-ID
            timeout: Optional Timeout

        Yields:
            Request-Budget
        """
        budget = self.create_budget(request_id, capability, agent_id, timeout)

        try:
            yield budget
        finally:
            # Budget entfernen und finale Metriken sammeln
            final_budget = self.remove_budget(request_id)
            if final_budget:
                await self._record_final_metrics(final_budget)

    async def _monitoring_loop(self):
        """Monitoring-Loop für aktive Budgets."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)

                with self._lock:
                    budgets_to_check = list(self._active_budgets.values())

                for budget in budgets_to_check:
                    await self._check_budget(budget)

                # Stoppe Monitoring wenn keine aktiven Budgets
                if not budgets_to_check:
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Budget-Monitoring: {e}")

    async def _check_budget(self, budget: RequestBudget):
        """Prüft Budget auf Überschreitungen.

        Args:
            budget: Zu prüfendes Budget
        """
        # Deadline-Check
        if budget.is_deadline_exceeded():
            await self._exhaustion_handler.handle_deadline_exceeded(budget)
            return

        # Resource-Budget-Checks
        for resource_type in ResourceType:
            if resource_type == ResourceType.CUSTOM:
                continue

            utilization = budget.get_budget_utilization(resource_type)

            # Budget-Warning
            if self.config.budget_warning_threshold <= utilization < 1.0:
                await self._exhaustion_handler.handle_budget_warning(
                    budget, resource_type, utilization
                )

            # Budget-Exhaustion
            elif utilization >= 1.0:
                await self._exhaustion_handler.handle_budget_exhaustion(
                    budget, resource_type, utilization
                )

    async def _record_final_metrics(self, budget: RequestBudget):
        """Zeichnet finale Metriken für Budget auf.

        Args:
            budget: Abgeschlossenes Budget
        """
        _metrics = budget.get_metrics()

        # Duration
        duration = time.time() - budget.start_time
        self._metrics_collector.record_histogram(
            "budget.request_duration",
            duration,
            tags={"capability": budget.capability, "agent_id": budget.agent_id},
        )

        # Resource-Utilization
        for resource_type in ResourceType:
            if resource_type == ResourceType.CUSTOM:
                continue

            utilization = budget.get_budget_utilization(resource_type)
            self._metrics_collector.record_histogram(
                "budget.resource_utilization",
                utilization,
                tags={
                    "capability": budget.capability,
                    "agent_id": budget.agent_id,
                    "resource_type": resource_type.value,
                },
            )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt Zusammenfassung aller Budget-Metriken.

        Returns:
            Budget-Metriken-Zusammenfassung
        """
        with self._lock:
            active_budgets = list(self._active_budgets.values())

        summary = {"active_budgets": len(active_budgets), "budgets": {}}

        for budget in active_budgets:
            summary["budgets"][budget.request_id] = budget.get_metrics()

        return summary


# Exceptions
class BudgetExhaustedError(Exception):
    """Exception wenn Budget erschöpft ist."""



class DeadlineExceededError(Exception):
    """Exception wenn Deadline überschritten wurde."""
