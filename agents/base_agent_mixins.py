# backend/agents/base_agent_mixins.py
"""Mixins für BaseAgent zur Trennung der Verantwortlichkeiten.

Aufteilt die BaseAgent-Funktionalität in spezialisierte Mixins für:
- Logging
- Metrics-Collection
- Error-Handling
- Performance-Monitoring

Implementiert Single Responsibility Principle und verbessert Testbarkeit.
"""

from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kei_logging import get_logger

from .constants import (
    PERFORMANCE_ERROR_THRESHOLD_MS,
    PERFORMANCE_WARNING_THRESHOLD_MS,
    ErrorMessages,
    LogEvents,
    MetricsNames,
)

if TYPE_CHECKING:
    from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)

# Type variables für generische Task/Response-Typen
TaskType = TypeVar("TaskType")
ResponseType = TypeVar("ResponseType")


# =============================================================================
# LOGGING MIXIN
# =============================================================================

class LoggingMixin(ABC):
    """Mixin für standardisiertes Agent-Logging."""

    @property
    @abstractmethod
    def metadata(self) -> Any:
        """Agent-Metadaten. Muss von implementierenden Klassen bereitgestellt werden."""

    def _log_task_start(self, task: TaskType) -> None:
        """Loggt den Start einer Task-Ausführung.

        Args:
            task: Auszuführende Task
        """
        logger.info(
            {
                "event": LogEvents.AGENT_HANDLE_START,
                "agent_id": self.metadata.id,
                "agent_name": self.metadata.name,
                "task_type": type(task).__name__,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def _log_task_complete(self, task: TaskType, result: ResponseType, execution_time_ms: float) -> None:
        """Loggt den erfolgreichen Abschluss einer Task.

        Args:
            task: Ausgeführte Task
            result: Task-Ergebnis
            execution_time_ms: Ausführungszeit in Millisekunden
        """
        logger.info(
            {
                "event": LogEvents.AGENT_HANDLE_COMPLETE,
                "agent_id": self.metadata.id,
                "agent_name": self.metadata.name,
                "task_type": type(task).__name__,
                "execution_time_ms": execution_time_ms,
                "result_type": type(result).__name__,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def _log_performance_warning(self, task: TaskType, execution_time_ms: float) -> None:
        """Loggt Performance-Warnungen bei langsamer Ausführung.

        Args:
            task: Ausgeführte Task
            execution_time_ms: Ausführungszeit in Millisekunden
        """
        warning_threshold, _ = self._get_performance_thresholds()
        logger.warning(
            {
                "event": LogEvents.PERFORMANCE_WARNING,
                "agent_id": self.metadata.id,
                "agent_name": self.metadata.name,
                "task_type": type(task).__name__,
                "execution_time_ms": execution_time_ms,
                "threshold_ms": warning_threshold,
                "message": f"Task-Ausführung langsam: {execution_time_ms:.2f}ms",
            }
        )

    def _get_performance_thresholds(self) -> tuple[float, float]:
        """Gibt Performance-Schwellenwerte zurück. Kann von Subklassen überschrieben werden.

        Returns:
            Tuple von (warning_threshold_ms, error_threshold_ms)
        """
        return PERFORMANCE_WARNING_THRESHOLD_MS, PERFORMANCE_ERROR_THRESHOLD_MS

    def _log_performance_error(self, task: TaskType, execution_time_ms: float) -> None:
        """Loggt Performance-Errors bei sehr langsamer Ausführung.

        Args:
            task: Ausgeführte Task
            execution_time_ms: Ausführungszeit in Millisekunden
        """
        _, error_threshold = self._get_performance_thresholds()
        logger.warning(
            {
                "event": LogEvents.PERFORMANCE_ERROR,
                "agent_id": self.metadata.id,
                "agent_name": self.metadata.name,
                "task_type": type(task).__name__,
                "execution_time_ms": execution_time_ms,
                "threshold_ms": error_threshold,
                "message": f"Task-Ausführung kritisch langsam: {execution_time_ms:.2f}ms",
            }
        )


# =============================================================================
# METRICS MIXIN
# =============================================================================

class MetricsMixin(ABC):
    """Mixin für Agent-Metrics-Collection."""

    @property
    @abstractmethod
    def metadata(self) -> Any:
        """Agent-Metadaten. Muss von implementierenden Klassen bereitgestellt werden."""

    def __init__(self, *args, **kwargs):
        """Initialisiert Metrics-Collection."""
        self._metrics_collector: MetricsCollector | None = kwargs.pop("metrics_collector", None)
        super().__init__(*args, **kwargs)

    def _record_task_start_metrics(self) -> None:
        """Zeichnet Metrics für Task-Start auf."""
        self._record_metric(MetricsNames.TOTAL_REQUESTS, 1)
        self._record_metric(MetricsNames.ACTIVE_REQUESTS, 1)

    def _record_task_success_metrics(self, execution_time_ms: float) -> None:
        """Zeichnet Metrics für erfolgreiche Task-Ausführung auf.

        Args:
            execution_time_ms: Ausführungszeit in Millisekunden
        """
        self._record_metric(MetricsNames.SUCCESSFUL_REQUESTS, 1)
        self._record_metric(MetricsNames.AVERAGE_LATENCY, execution_time_ms)

        # Update Performance-Metrics
        if hasattr(self, "performance_metrics"):
            self.performance_metrics.total_requests += 1
            self.performance_metrics.successful_requests += 1
            self.performance_metrics.last_request_at = datetime.now(UTC)

            # Berechne gleitenden Durchschnitt der Latenz
            current_avg = self.performance_metrics.average_latency_ms
            total_requests = self.performance_metrics.total_requests
            self.performance_metrics.average_latency_ms = (
                (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
            )

    def _record_task_error_metrics(self, execution_time_ms: float) -> None:
        """Zeichnet Metrics für fehlgeschlagene Task-Ausführung auf.

        Args:
            execution_time_ms: Ausführungszeit in Millisekunden
        """
        self._record_metric(MetricsNames.FAILED_REQUESTS, 1)
        self._record_metric(MetricsNames.ERROR_RATE, 1)
        self._record_metric(MetricsNames.AVERAGE_LATENCY, execution_time_ms)

        # Update Performance-Metrics
        if hasattr(self, "performance_metrics"):
            self.performance_metrics.total_requests += 1
            self.performance_metrics.failed_requests += 1
            self.performance_metrics.last_request_at = datetime.now(UTC)

            # Berechne gleitenden Durchschnitt der Latenz (auch für Fehler)
            current_avg = self.performance_metrics.average_latency_ms
            total_requests = self.performance_metrics.total_requests
            self.performance_metrics.average_latency_ms = (
                (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
            )

    def _record_metric(self, metric_name: str, value: float) -> None:
        """Zeichnet eine einzelne Metrik auf.

        Args:
            metric_name: Name der Metrik
            value: Metrik-Wert
        """
        if self._metrics_collector:
            try:
                self._metrics_collector.record_metric(
                    metric_name=metric_name,
                    value=value,
                    labels={
                        "agent_id": self.metadata.id,
                        "agent_name": self.metadata.name,
                        "agent_type": self.metadata.type.value,
                    }
                )
            except Exception as e:
                logger.warning(f"{ErrorMessages.METRICS_RECORDING_FAILED}: {e}")


# =============================================================================
# ERROR HANDLING MIXIN
# =============================================================================

class ErrorHandlingMixin(ABC):
    """Mixin für standardisiertes Error-Handling."""

    @property
    @abstractmethod
    def metadata(self) -> Any:
        """Agent-Metadaten. Muss von implementierenden Klassen bereitgestellt werden."""

    @abstractmethod
    def _log_task_complete(self, task: TaskType, result: ResponseType, execution_time_ms: float) -> None:
        """Loggt den erfolgreichen Abschluss einer Task."""

    @abstractmethod
    def _get_performance_thresholds(self) -> tuple[float, float]:
        """Gibt Performance-Schwellenwerte zurück."""

    @abstractmethod
    def _log_performance_error(self, task: TaskType, execution_time_ms: float) -> None:
        """Loggt Performance-Errors."""

    @abstractmethod
    def _log_performance_warning(self, task: TaskType, execution_time_ms: float) -> None:
        """Loggt Performance-Warnungen."""

    @abstractmethod
    def _record_task_success_metrics(self, execution_time_ms: float) -> None:
        """Zeichnet Metrics für erfolgreiche Task-Ausführung auf."""

    @abstractmethod
    def _record_task_error_metrics(self, execution_time_ms: float) -> None:
        """Zeichnet Metrics für fehlgeschlagene Task-Ausführung auf."""

    async def _handle_success(self, task: TaskType, result: ResponseType, execution_time_ms: float) -> None:
        """Behandelt erfolgreiche Task-Ausführung.

        Args:
            task: Ausgeführte Task
            result: Task-Ergebnis
            execution_time_ms: Ausführungszeit in Millisekunden
        """
        # Logging
        self._log_task_complete(task, result, execution_time_ms)

        # Performance-Monitoring mit agent-spezifischen Schwellenwerten
        warning_threshold, error_threshold = self._get_performance_thresholds()
        if execution_time_ms > error_threshold:
            self._log_performance_error(task, execution_time_ms)
        elif execution_time_ms > warning_threshold:
            self._log_performance_warning(task, execution_time_ms)

        # Metrics
        self._record_task_success_metrics(execution_time_ms)

    async def _handle_error(self, task: TaskType, error: Exception, execution_time_ms: float) -> None:
        """Behandelt Fehler bei Task-Ausführung.

        Args:
            task: Fehlgeschlagene Task
            error: Aufgetretener Fehler
            execution_time_ms: Ausführungszeit bis zum Fehler
        """
        # Error-Logging
        logger.error(
            {
                "event": LogEvents.TASK_EXECUTION_ERROR,
                "agent_id": self.metadata.id,
                "agent_name": self.metadata.name,
                "task_type": type(task).__name__,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Metrics
        self._record_task_error_metrics(execution_time_ms)


# =============================================================================
# PERFORMANCE MONITORING MIXIN
# =============================================================================

class PerformanceMonitoringMixin(ABC):
    """Mixin für Performance-Monitoring und Timing."""

    @property
    @abstractmethod
    def performance_metrics(self) -> Any:
        """Performance-Metriken. Muss von implementierenden Klassen bereitgestellt werden."""

    @staticmethod
    def _calculate_execution_time(start_time: float) -> float:
        """Berechnet die Ausführungszeit in Millisekunden.

        Args:
            start_time: Start-Zeit von time.perf_counter()

        Returns:
            Ausführungszeit in Millisekunden
        """
        return (time.perf_counter() - start_time) * 1000

    def _get_performance_summary(self) -> dict[str, Any]:
        """Gibt eine Zusammenfassung der Performance-Metriken zurück.

        Returns:
            Dictionary mit Performance-Metriken
        """
        if not hasattr(self, "performance_metrics"):
            return {}

        metrics = self.performance_metrics
        return {
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "success_rate": metrics.success_rate,
            "error_rate": metrics.error_rate,
            "average_latency_ms": metrics.average_latency_ms,
            "last_request_at": metrics.last_request_at.isoformat() if metrics.last_request_at else None,
        }


# =============================================================================
# VALIDATION MIXIN
# =============================================================================

class ValidationMixin(ABC):
    """Mixin für Agent-Validierung und Status-Checks."""

    @property
    @abstractmethod
    def metadata(self) -> Any:
        """Agent-Metadaten. Muss von implementierenden Klassen bereitgestellt werden."""

    def _validate_agent_availability(self) -> None:
        """Validiert, dass der Agent verfügbar ist.

        Raises:
            RuntimeError: Wenn Agent nicht verfügbar ist
        """
        from .constants import (
            AgentStatus,  # Lokaler Import um zirkuläre Abhängigkeiten zu vermeiden
        )

        # Robuste Status-Prüfung: sowohl Enum als auch String-Werte unterstützen
        current_status = self.metadata.status
        if hasattr(current_status, "value"):
            status_value = current_status.value
        else:
            status_value = str(current_status)

        if status_value != AgentStatus.AVAILABLE.value:
            raise RuntimeError(
                f"{ErrorMessages.AGENT_NOT_AVAILABLE}: Status ist {status_value}"
            )

    def _validate_task(self, task: TaskType) -> None:
        """Validiert eine Task vor der Ausführung.

        Args:
            task: Zu validierende Task

        Raises:
            ValueError: Bei ungültiger Task
        """
        if task is None:
            raise ValueError(f"{ErrorMessages.INVALID_PARAMETERS}: Task darf nicht None sein")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ErrorHandlingMixin",
    "LoggingMixin",
    "MetricsMixin",
    "PerformanceMonitoringMixin",
    "ValidationMixin",
]
