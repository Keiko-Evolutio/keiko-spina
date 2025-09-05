"""Konstanten für Task-Management-System.

Zentrale Definition aller Magic Numbers, Timeouts und Konfigurationswerte
für bessere Wartbarkeit und Clean Code Standards.
"""

from __future__ import annotations

from typing import Any, Final

# Task-Timeouts und Limits
DEFAULT_TASK_TIMEOUT_SECONDS: Final[int] = 300  # 5 Minuten
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 60  # 1 Minute
DEFAULT_RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0
MAX_TASK_TIMEOUT_SECONDS: Final[int] = 7200  # 2 Stunden
MAX_RETRY_COUNT: Final[int] = 10

# Worker-Pool-Konfiguration
DEFAULT_WORKER_POOL_SIZE: Final[int] = 10
MIN_WORKER_POOL_SIZE: Final[int] = 1
MAX_WORKER_POOL_SIZE: Final[int] = 100
DEFAULT_WORKER_QUEUE_SIZE: Final[int] = 1000

# Background-Task-Intervalle
CLEANUP_LOOP_INTERVAL_SECONDS: Final[int] = 300  # 5 Minuten
TIMEOUT_CHECK_INTERVAL_SECONDS: Final[int] = 60  # 1 Minute
DEPENDENCY_RESOLUTION_INTERVAL_SECONDS: Final[int] = 30  # 30 Sekunden
SCHEDULER_LOOP_INTERVAL_SECONDS: Final[int] = 10  # 10 Sekunden

# Task-Retention-Zeiten
COMPLETED_TASK_RETENTION_HOURS: Final[int] = 24  # 1 Tag
FAILED_TASK_RETENTION_HOURS: Final[int] = 72  # 3 Tage
CANCELLED_TASK_RETENTION_HOURS: Final[int] = 24  # 1 Tag

# API-Limits
MAX_TASKS_PER_LIST_REQUEST: Final[int] = 1000
DEFAULT_TASKS_PER_PAGE: Final[int] = 50
MAX_TASK_NAME_LENGTH: Final[int] = 255
MAX_TASK_DESCRIPTION_LENGTH: Final[int] = 2000
MAX_TAGS_PER_TASK: Final[int] = 20
MAX_LABELS_PER_TASK: Final[int] = 50

# Idempotenz-Konfiguration
DEFAULT_IDEMPOTENCY_TTL_HOURS: Final[int] = 24  # 1 Tag
MAX_IDEMPOTENCY_TTL_HOURS: Final[int] = 168  # 1 Woche
IDEMPOTENCY_CACHE_SIZE: Final[int] = 10000

# Performance-Tuning
BATCH_SIZE_DEPENDENCY_RESOLUTION: Final[int] = 100
BATCH_SIZE_CLEANUP: Final[int] = 50
QUEUE_POLL_TIMEOUT_SECONDS: Final[float] = 1.0
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 30.0

# Simulation und Testing
SIMULATED_TASK_EXECUTION_TIME_MS: Final[float] = 100.0
TEST_TASK_TIMEOUT_SECONDS: Final[int] = 5

# Task-Prioritäten (numerische Werte für Sorting)
PRIORITY_WEIGHTS: Final[dict[str, int]] = {
    "critical": 1000,
    "urgent": 800,
    "high": 600,
    "normal": 400,
    "low": 200,
    "background": 100,
}

# Task-Type-spezifische Timeouts
TASK_TYPE_TIMEOUTS: Final[dict[str, int]] = {
    "agent_execution": 1800,  # 30 Minuten
    "data_processing": 3600,  # 1 Stunde
    "nlp_analysis": 900,  # 15 Minuten
    "tool_call": 300,  # 5 Minuten
    "workflow": 7200,  # 2 Stunden
    "batch_job": 14400,  # 4 Stunden
    "scheduled_task": 1800,  # 30 Minuten
    "system_maintenance": 3600,  # 1 Stunde
}

# Error-Messages (für Konsistenz)
ERROR_MESSAGES: Final[dict[str, str]] = {
    "task_not_found": "Task mit ID '{task_id}' nicht gefunden",
    "task_already_running": "Task '{task_id}' läuft bereits",
    "task_already_completed": "Task '{task_id}' bereits abgeschlossen",
    "task_already_cancelled": "Task '{task_id}' bereits abgebrochen",
    "invalid_state_transition": "Ungültiger State-Übergang von '{from_state}' zu '{to_state}'",
    "dependency_cycle_detected": "Zirkuläre Abhängigkeit erkannt für Task '{task_id}'",
    "max_retries_exceeded": "Maximale Anzahl Wiederholungen ({max_retries}) überschritten",
    "timeout_exceeded": "Task-Timeout ({timeout_seconds}s) überschritten",
    "worker_pool_full": "Worker-Pool ist voll ({pool_size} Worker)",
    "invalid_task_type": "Ungültiger Task-Type: '{task_type}'",
    "invalid_priority": "Ungültige Priorität: '{priority}'",
    "duplicate_task_detected": "Duplicate Task erkannt (Idempotency-Key: '{key}')",
}

# Logging-Messages (für Konsistenz)
LOG_MESSAGES: Final[dict[str, str]] = {
    "task_created": "Task erstellt: {task_id} (Type: {task_type}, Priority: {priority})",
    "task_started": "Task gestartet: {task_id} auf Worker {worker_id}",
    "task_completed": "Task abgeschlossen: {task_id} (Dauer: {duration_ms}ms)",
    "task_failed": "Task fehlgeschlagen: {task_id} (Fehler: {error})",
    "task_cancelled": "Task abgebrochen: {task_id} (Grund: {reason})",
    "task_retried": "Task wiederholt: {task_id} (Versuch: {attempt}/{max_retries})",
    "worker_started": "Worker gestartet: {worker_id}",
    "worker_stopped": "Worker gestoppt: {worker_id}",
    "pool_started": "Worker-Pool gestartet: {pool_size} Worker",
    "pool_stopped": "Worker-Pool gestoppt",
    "dependency_resolved": "Abhängigkeit aufgelöst: {task_id} -> {dependent_task_id}",
    "cleanup_completed": "Cleanup abgeschlossen: {cleaned_tasks} Tasks entfernt",
}

# Metriken-Namen (für Observability)
METRIC_NAMES: Final[dict[str, str]] = {
    "tasks_created_total": "task_management_tasks_created_total",
    "tasks_completed_total": "task_management_tasks_completed_total",
    "tasks_failed_total": "task_management_tasks_failed_total",
    "tasks_cancelled_total": "task_management_tasks_cancelled_total",
    "task_execution_duration_seconds": "task_management_execution_duration_seconds",
    "worker_utilization_ratio": "task_management_worker_utilization_ratio",
    "queue_size_current": "task_management_queue_size_current",
    "dependency_resolution_duration_seconds": "task_management_dependency_resolution_duration_seconds",
}

# Trace-Span-Namen (für Observability)
TRACE_SPANS: Final[dict[str, str]] = {
    "task_creation": "task_management.create_task",
    "task_execution": "task_management.execute_task",
    "task_state_transition": "task_management.state_transition",
    "dependency_resolution": "task_management.dependency_resolution",
    "worker_execution": "task_management.worker_execution",
    "api_request": "task_management.api_request",
    "idempotency_check": "task_management.idempotency_check",
    "cleanup_operation": "task_management.cleanup_operation",
}

# Validierungs-Regeln
VALIDATION_RULES: Final[dict[str, dict[str, Any]]] = {
    "task_name": {
        "min_length": 1,
        "max_length": MAX_TASK_NAME_LENGTH,
        "pattern": r"^[a-zA-Z0-9\s\-_\.]+$",
    },
    "task_description": {
        "max_length": MAX_TASK_DESCRIPTION_LENGTH,
    },
    "timeout_seconds": {
        "min_value": 1,
        "max_value": MAX_TASK_TIMEOUT_SECONDS,
    },
    "max_retries": {
        "min_value": 0,
        "max_value": MAX_RETRY_COUNT,
    },
    "pool_size": {
        "min_value": MIN_WORKER_POOL_SIZE,
        "max_value": MAX_WORKER_POOL_SIZE,
    },
}

# Feature-Flags (für A/B-Testing und Rollouts)
FEATURE_FLAGS: Final[dict[str, bool]] = {
    "enable_dependency_resolution": True,
    "enable_automatic_cleanup": True,
    "enable_task_metrics": True,
    "enable_idempotency": True,
    "enable_task_scheduling": True,
    "enable_worker_scaling": False,  # Experimentell
    "enable_circuit_breaker": False,  # Experimentell
    "enable_task_prioritization": True,
}

# Environment-spezifische Konfiguration
ENVIRONMENT_CONFIGS: Final[dict[str, dict[str, Any]]] = {
    "development": {
        "cleanup_interval": CLEANUP_LOOP_INTERVAL_SECONDS * 2,  # Weniger aggressiv
        "log_level": "DEBUG",
        "enable_debug_metrics": True,
    },
    "testing": {
        "cleanup_interval": 10,  # Schneller für Tests
        "task_timeout": TEST_TASK_TIMEOUT_SECONDS,
        "enable_debug_metrics": False,
    },
    "production": {
        "cleanup_interval": CLEANUP_LOOP_INTERVAL_SECONDS,
        "log_level": "INFO",
        "enable_debug_metrics": False,
    },
}


def get_environment_config(environment: str = "production") -> dict[str, Any]:
    """Gibt Environment-spezifische Konfiguration zurück.

    Args:
        environment: Environment-Name (development, testing, production)

    Returns:
        Konfiguration für das angegebene Environment
    """
    return ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS["production"])


def get_task_type_timeout(task_type: str) -> int:
    """Gibt Task-Type-spezifischen Timeout zurück.

    Args:
        task_type: Task-Type

    Returns:
        Timeout in Sekunden
    """
    return TASK_TYPE_TIMEOUTS.get(task_type, DEFAULT_TASK_TIMEOUT_SECONDS)


def get_priority_weight(priority: str) -> int:
    """Gibt numerisches Gewicht für Priorität zurück.

    Args:
        priority: Prioritäts-String

    Returns:
        Numerisches Gewicht für Sorting
    """
    return PRIORITY_WEIGHTS.get(priority.lower(), PRIORITY_WEIGHTS["normal"])


__all__ = [
    # Intervalle
    "CLEANUP_LOOP_INTERVAL_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF_MULTIPLIER",
    "DEFAULT_RETRY_DELAY_SECONDS",
    "DEFAULT_TASKS_PER_PAGE",
    # Timeouts und Limits
    "DEFAULT_TASK_TIMEOUT_SECONDS",
    # Worker-Pool
    "DEFAULT_WORKER_POOL_SIZE",
    "DEFAULT_WORKER_QUEUE_SIZE",
    "DEPENDENCY_RESOLUTION_INTERVAL_SECONDS",
    # Dictionaries
    "ERROR_MESSAGES",
    "FEATURE_FLAGS",
    "LOG_MESSAGES",
    "MAX_RETRY_COUNT",
    # API-Limits
    "MAX_TASKS_PER_LIST_REQUEST",
    "MAX_TASK_DESCRIPTION_LENGTH",
    "MAX_TASK_NAME_LENGTH",
    "MAX_TASK_TIMEOUT_SECONDS",
    "MAX_WORKER_POOL_SIZE",
    "METRIC_NAMES",
    "MIN_WORKER_POOL_SIZE",
    "SCHEDULER_LOOP_INTERVAL_SECONDS",
    "TIMEOUT_CHECK_INTERVAL_SECONDS",
    "TRACE_SPANS",
    "VALIDATION_RULES",
    # Helper-Funktionen
    "get_environment_config",
    "get_priority_weight",
    "get_task_type_timeout",
]
