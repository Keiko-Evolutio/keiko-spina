"""Scheduling-Services für Keiko Personal Assistant.

Dieses Modul stellt robuste Scheduling-Funktionalitäten bereit:
- Push/Pull-basiertes Task-Scheduling mit Backpressure-Awareness
- Work-Stealing für automatisches Load-Balancing
- Konfigurierbare Backpressure-Entscheidungen
- AsyncIO-basierte Service-Basis-Klassen

Hauptkomponenten:
    Scheduler: Einheitlicher Scheduler für Push/Direct und Pull/Queue
    WorkStealer: Work-Stealing für Task-Queues
    AsyncServiceBase: Basis-Klasse für AsyncIO-Services
    PeriodicServiceBase: Basis-Klasse für periodische Services
"""

from .backpressure import decide_mode
from .base import AsyncServiceBase, PeriodicServiceBase
from .config import (
    DEFAULT_SCHEDULER_CONFIG,
    BackpressureConfig,
    SchedulerConfig,
    WorkStealingConfig,
)
from .scheduler import Scheduler, SchedulingResult
from .work_stealing import WorkStealer
from .work_stealing import WorkStealingConfig as StealPolicy

__all__ = [
    "DEFAULT_SCHEDULER_CONFIG",
    # Base classes
    "AsyncServiceBase",
    # Configuration
    "BackpressureConfig",
    "PeriodicServiceBase",
    # Core scheduling
    "Scheduler",
    "SchedulerConfig",
    "SchedulingResult",
    "StealPolicy",  # Backward compatibility
    # Work stealing
    "WorkStealer",
    "WorkStealingConfig",
    "decide_mode",
]

__version__ = "1.0.0"
