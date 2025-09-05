# backend/agents/operations/__init__.py
"""Operations-Modul für zentrale Agent-Operations.

Stellt den Operations-Coordinator bereit, der Agent-Operations koordiniert,
einschließlich Task-Execution mit Resilience, Performance-Monitoring,
Component-Management und Security-Integration.

Hauptkomponenten:
- OperationsCoordinator: Zentrale Koordination aller Agent-Operations
- OperationsConfig: Konfiguration für Operations-Coordinator
- Constants: Zentrale Konstanten für das Operations-Modul

Beispiel-Verwendung:
    ```python
    from agents.operations import OperationsCoordinator, OperationsConfig

    config = OperationsConfig(
        agent_id="operations-coordinator",
        enable_task_queuing=True,
        enable_performance_monitoring=True
    )

    coordinator = OperationsCoordinator(config)
    await coordinator.initialize()

    # Agent-Task mit Resilience ausführen
    result = await coordinator.execute_agent_task_with_resilience(
        agent_id="test-agent",
        task="Test-Task",
        capability="test-capability",
        timeout=30.0,
        framework="test-framework",
        model_name="test-model",
        temperature=0.7,
        max_tokens=1000
    )
    ```
"""

from __future__ import annotations

from .constants import (
    # Component-Namen
    COMPONENT_RESILIENCE_COORDINATOR,
    COMPONENT_SECURITY_MANAGER,
    COMPONENT_SLO_SLA_COORDINATOR,
    DEFAULT_BATCH_SIZE,
    # Queue-Konfiguration
    DEFAULT_MAX_QUEUE_SIZE,
    # Monitoring-Konfiguration
    DEFAULT_METRICS_COLLECTION_INTERVAL,
    ERROR_COMPONENT_MANAGER_INIT_FAILED,
    ERROR_TASK_QUEUE_FULL,
    # Error-Messages
    ERROR_TASK_QUEUE_NOT_ENABLED,
    MONITORING_SLEEP_INTERVAL,
    # Timeout-Konfiguration
    QUEUE_JOIN_TIMEOUT,
    QUEUE_WARNING_THRESHOLD,
    STANDARD_TASK_SIMULATION_DELAY,
)
from .operations_coordinator import OperationsConfig, OperationsCoordinator

# Version und Metadaten
__version__ = "1.0.0"
__author__ = "Operations Team"
__description__ = "Operations-Coordinator für zentrale Agent-Operations"

# Public API
__all__ = [
    # Hauptklassen
    "OperationsCoordinator",
    "OperationsConfig",

    # Queue-Konstanten
    "DEFAULT_MAX_QUEUE_SIZE",
    "DEFAULT_BATCH_SIZE",
    "QUEUE_WARNING_THRESHOLD",

    # Monitoring-Konstanten
    "DEFAULT_METRICS_COLLECTION_INTERVAL",
    "MONITORING_SLEEP_INTERVAL",

    # Timeout-Konstanten
    "QUEUE_JOIN_TIMEOUT",
    "STANDARD_TASK_SIMULATION_DELAY",

    # Component-Konstanten
    "COMPONENT_RESILIENCE_COORDINATOR",
    "COMPONENT_SLO_SLA_COORDINATOR",
    "COMPONENT_SECURITY_MANAGER",

    # Error-Konstanten
    "ERROR_TASK_QUEUE_NOT_ENABLED",
    "ERROR_TASK_QUEUE_FULL",
    "ERROR_COMPONENT_MANAGER_INIT_FAILED",

    # Metadaten
    "__version__",
    "__author__",
    "__description__",
]


def get_operations_info() -> dict[str, str]:
    """Gibt Informationen über das Operations-Modul zurück.

    Returns:
        Dictionary mit Modul-Informationen
    """
    return {
        "module": "backend.agents.operations",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "main_class": "OperationsCoordinator",
        "config_class": "OperationsConfig",
    }
