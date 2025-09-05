"""Konfigurationsklassen für Scheduling-Services.

Zentralisiert alle Konfigurationswerte und eliminiert Magic Numbers
aus dem Scheduling-Code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackpressureConfig:
    """Konfiguration für Backpressure-Entscheidungen."""

    # Schwellenwerte für Queue-basierte Entscheidungen
    queue_utilization_threshold: float = 0.8
    """Schwellenwert für Queue-Auslastung (0.0-1.0) ab dem auf Pull-Mode gewechselt wird"""

    # Default-Modus wenn keine Hints verfügbar
    default_mode: str = "push"
    """Standard-Scheduling-Modus ('push' oder 'pull')"""

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte."""
        if not 0.0 <= self.queue_utilization_threshold <= 1.0:
            raise ValueError("queue_utilization_threshold muss zwischen 0.0 und 1.0 liegen")

        if self.default_mode not in ("push", "pull"):
            raise ValueError("default_mode muss 'push' oder 'pull' sein")


@dataclass
class WorkStealingConfig:
    """Konfiguration für Work-Stealing-Verhalten."""

    # Timing-Parameter
    check_interval_seconds: float = 2.0
    """Intervall zwischen Work-Stealing-Checks in Sekunden"""

    # Schwellenwerte
    min_remote_queue_length: int = 5
    """Minimale Queue-Länge bevor Work-Stealing aktiviert wird"""

    max_concurrent_steals: int = 4
    """Maximale Anzahl gleichzeitiger Work-Stealing-Operationen"""

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds muss positiv sein")

        if self.min_remote_queue_length < 1:
            raise ValueError("min_remote_queue_length muss mindestens 1 sein")

        if self.max_concurrent_steals < 1:
            raise ValueError("max_concurrent_steals muss mindestens 1 sein")


@dataclass
class SchedulerConfig:
    """Hauptkonfiguration für Scheduler-Services."""

    # Sub-Konfigurationen
    backpressure: BackpressureConfig
    work_stealing: WorkStealingConfig

    # Service-spezifische Parameter
    default_tenant: str | None = None
    """Standard-Tenant für Scheduling-Operationen"""

    graceful_shutdown_timeout: float = 30.0
    """Timeout für graceful shutdown in Sekunden"""

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte."""
        if self.graceful_shutdown_timeout <= 0:
            raise ValueError("graceful_shutdown_timeout muss positiv sein")

    @classmethod
    def create_default(cls) -> SchedulerConfig:
        """Erstellt Standard-Konfiguration."""
        return cls(
            backpressure=BackpressureConfig(),
            work_stealing=WorkStealingConfig()
        )


# Globale Standard-Konfiguration
DEFAULT_SCHEDULER_CONFIG = SchedulerConfig.create_default()


__all__ = [
    "DEFAULT_SCHEDULER_CONFIG",
    "BackpressureConfig",
    "SchedulerConfig",
    "WorkStealingConfig"
]
