# backend/agents/base_agent.py
"""Base Agent Class für Custom Agents.

Implementiert Mixin-basierte Architektur für bessere Trennung der
Verantwortlichkeiten und verbesserte Testbarkeit. Verwendet konsolidierte
Constants und eliminiert Code-Duplikation.
"""

from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kei_logging import get_logger

# Mixin-Imports für Trennung der Verantwortlichkeiten
from .base_agent_mixins import (
    ErrorHandlingMixin,
    LoggingMixin,
    MetricsMixin,
    PerformanceMonitoringMixin,
    ValidationMixin,
)

# Konsolidierte Imports aus constants.py
from .constants import (
    AgentStatus,
    AgentType,
)

if TYPE_CHECKING:
    from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)

# Type variables für generische Task/Response-Typen
TaskType = TypeVar("TaskType")
ResponseType = TypeVar("ResponseType")


@dataclass(slots=True)
class AgentMetadata:
    """Metadaten für einen Agent."""

    id: str
    name: str
    type: AgentType
    description: str
    status: AgentStatus = AgentStatus.AVAILABLE
    capabilities: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Metadaten zu Dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class AgentPerformanceMetrics:
    """Performance-Metriken für einen Agent."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    last_request_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Berechnet die Erfolgsrate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Berechnet die Fehlerrate."""
        return 1.0 - self.success_rate


class BaseAgent(Generic[TaskType, ResponseType],
    LoggingMixin,
    MetricsMixin,
    ErrorHandlingMixin,
    PerformanceMonitoringMixin,
    ValidationMixin,
    ABC,
):
    """Basis-Klasse für alle Custom Agents.

    Verwendet Mixin-basierte Architektur für klare Trennung der Verantwortlichkeiten:
    - LoggingMixin: Standardisiertes Agent-Logging
    - MetricsMixin: Performance-Metrics-Collection
    - ErrorHandlingMixin: Einheitliches Error-Handling
    - PerformanceMonitoringMixin: Performance-Monitoring und Timing
    - ValidationMixin: Agent- und Task-Validierung
    """

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        name: str,
        agent_type: AgentType,
        description: str,
        capabilities: list[str] | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        """Initialisiert den Base Agent mit Mixin-basierter Architektur.

        Args:
            agent_id: Eindeutige Agent-ID (wird generiert falls None)
            name: Agent-Name
            agent_type: Agent-Typ
            description: Agent-Beschreibung
            capabilities: Liste der Agent-Fähigkeiten
            metrics_collector: Metrics-Collector-Instanz
        """
        # Initialisiere Mixins (wichtig für MetricsMixin)
        super().__init__(metrics_collector=metrics_collector)

        # Agent-Metadaten
        self.metadata = AgentMetadata(
            id=agent_id or self._generate_agent_id(),
            name=name,
            type=agent_type,
            description=description,
            capabilities=capabilities or [],
        )

        # Performance-Tracking
        self.performance_metrics = AgentPerformanceMetrics()

        # Initialisierungs-Logging (verwendet LoggingMixin)
        logger.info(
            {
                "event": "agent.init",  # Aus agents_constants.LogEvents
                "agent_id": self.metadata.id,
                "agent_name": self.metadata.name,
                "agent_type": self.metadata.type.value,
                "capabilities": self.metadata.capabilities,
            }
        )

    @property
    def id(self) -> str:
        """Agent-ID."""
        return self.metadata.id

    @property
    def agent_id(self) -> str:
        """Agent-ID (Alias für id)."""
        return self.metadata.id

    @property
    def name(self) -> str:
        """Agent-Name."""
        return self.metadata.name

    @property
    def type(self) -> str:
        """Agent-Typ als String."""
        return self.metadata.type.value

    @property
    def agent_type(self) -> AgentType:
        """Agent-Typ als Enum."""
        return self.metadata.type

    @property
    def description(self) -> str:
        """Agent-Beschreibung."""
        return self.metadata.description

    @property
    def status(self) -> str:
        """Agent-Status als String."""
        return self.metadata.status.value

    @property
    def capabilities(self) -> list[str]:
        """Agent-Fähigkeiten."""
        return self.metadata.capabilities

    @property
    def is_available(self) -> bool:
        """Prüft ob Agent verfügbar ist."""
        return self.metadata.status == AgentStatus.AVAILABLE

    async def handle(self, task: TaskType) -> ResponseType:
        """Führt eine Task aus mit vollständigem Monitoring.

        Verwendet Mixin-basierte Architektur für klare Trennung der Verantwortlichkeiten.

        Args:
            task: Auszuführende Task

        Returns:
            Task-Ergebnis

        Raises:
            Exception: Bei Task-Ausführungsfehlern
        """
        # Validation (ValidationMixin)
        self._validate_agent_availability()
        self._validate_task(task)

        # Performance-Monitoring (PerformanceMonitoringMixin)
        start_time = time.perf_counter()

        # Logging (LoggingMixin)
        self._log_task_start(task)

        try:
            # Metrics (MetricsMixin)
            self._record_task_start_metrics()

            # Eigentliche Task-Ausführung
            result = await self._execute_task(task)

            # Success-Handling (ErrorHandlingMixin + andere Mixins)
            execution_time_ms = self._calculate_execution_time(start_time)
            await self._handle_success(task, result, execution_time_ms)

            return result

        except Exception as e:
            # Error-Handling (ErrorHandlingMixin + andere Mixins)
            execution_time_ms = self._calculate_execution_time(start_time)
            await self._handle_error(task, e, execution_time_ms)
            raise

        finally:
            # Cleanup (MetricsMixin)
            self._record_metric("active_requests", -1)

    @abstractmethod
    async def _execute_task(self, task: TaskType) -> ResponseType:
        """Führt die eigentliche Task-Logik aus.

        Muss von Subklassen implementiert werden.

        Args:
            task: Auszuführende Task

        Returns:
            Task-Ergebnis
        """

    def update_status(self, status: AgentStatus) -> None:
        """Aktualisiert den Agent-Status.

        Args:
            status: Neuer Status
        """
        old_status = self.metadata.status
        self.metadata.status = status

        logger.info(
            {
                "event": "agent_status_changed",
                "agent_id": self.metadata.id,
                "old_status": old_status.value,
                "new_status": status.value,
            }
        )

    def add_capability(self, capability: str) -> None:
        """Fügt eine Fähigkeit hinzu.

        Args:
            capability: Neue Fähigkeit
        """
        if capability not in self.metadata.capabilities:
            self.metadata.capabilities.append(capability)
            logger.info(f"Capability '{capability}' added to agent {self.metadata.id}")

    def remove_capability(self, capability: str) -> None:
        """Entfernt eine Fähigkeit.

        Args:
            capability: Zu entfernende Fähigkeit
        """
        if capability in self.metadata.capabilities:
            self.metadata.capabilities.remove(capability)
            logger.info(f"Capability '{capability}' removed from agent {self.metadata.id}")

    def get_health_info(self) -> dict[str, Any]:
        """Gibt Health-Informationen zurück.

        Verwendet PerformanceMonitoringMixin für Performance-Zusammenfassung.

        Returns:
            Health-Status und Metriken
        """
        return {
            "agent_id": self.metadata.id,
            "status": self.metadata.status.value,
            "is_available": self.is_available,
            "performance": self._get_performance_summary(),  # Aus PerformanceMonitoringMixin
            "metadata": self.metadata.to_dict(),
        }

    @staticmethod
    def _generate_agent_id() -> str:
        """Generiert eine eindeutige Agent-ID.

        Returns:
            Eindeutige Agent-ID im Format 'agent_<8-stelliger-hex>'
        """
        return f"agent_{uuid4().hex[:8]}"


__all__ = [
    "AgentMetadata",
    "AgentPerformanceMetrics",
    "BaseAgent",
    "ResponseType",
    "TaskType",
]
