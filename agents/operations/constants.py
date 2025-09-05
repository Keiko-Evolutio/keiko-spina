# backend/agents/operations/constants.py
"""Konstanten für Operations-Coordinator."""

from __future__ import annotations

from typing import Final

# Queue-Konfiguration
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 1000
DEFAULT_BATCH_SIZE: Final[int] = 10
QUEUE_WARNING_THRESHOLD: Final[float] = 0.9

# Monitoring-Konfiguration
DEFAULT_METRICS_COLLECTION_INTERVAL: Final[float] = 60.0
MONITORING_SLEEP_INTERVAL: Final[float] = 1.0

# Timeout-Konfiguration
QUEUE_JOIN_TIMEOUT: Final[float] = 1.0
STANDARD_TASK_SIMULATION_DELAY: Final[float] = 0.1

# Task-ID-Generierung
TASK_ID_PREFIX_QUEUED: Final[str] = "queued"
TASK_ID_SEPARATOR: Final[str] = "_"

# Component-Namen
COMPONENT_RESILIENCE_COORDINATOR: Final[str] = "resilience_coordinator"
COMPONENT_SLO_SLA_COORDINATOR: Final[str] = "slo_sla_coordinator"
COMPONENT_SECURITY_MANAGER: Final[str] = "security_manager"

# Error-Messages
ERROR_TASK_QUEUE_NOT_ENABLED: Final[str] = "Task-Queue nicht aktiviert"
ERROR_TASK_QUEUE_FULL: Final[str] = "Task-Queue voll"
ERROR_COMPONENT_MANAGER_INIT_FAILED: Final[str] = "Component-Manager-Initialisierung fehlgeschlagen"

# Log-Messages
LOG_TASK_QUEUE_PROCESSOR_STARTED: Final[str] = "Task-Queue-Processor gestartet"
LOG_TASK_QUEUE_PROCESSOR_STOPPED: Final[str] = "Task-Queue-Processor gestoppt"
LOG_PERFORMANCE_MONITORING_STARTED: Final[str] = "Performance-Monitoring gestartet"
LOG_PERFORMANCE_MONITORING_STOPPED: Final[str] = "Performance-Monitoring gestoppt"
LOG_TASK_ADDED_TO_QUEUE: Final[str] = "Task {task_name} zur Queue hinzugefügt (ID: {task_id})"
LOG_TASK_QUEUE_FULL_REJECTED: Final[str] = "Task-Queue voll - Task {task_name} abgelehnt"
LOG_QUEUED_TASK_SUCCESS: Final[str] = "Queued Task {task_name} erfolgreich ausgeführt"
LOG_QUEUED_TASK_FAILED: Final[str] = "Queued Task {task_name} fehlgeschlagen: {error}"
LOG_TASK_QUEUE_PROCESSOR_ERROR: Final[str] = "Task-Queue-Processor-Fehler: {error}"
LOG_PERFORMANCE_MONITORING_ERROR: Final[str] = "Performance-Monitoring-Fehler: {error}"

# Health-Check-Messages
HEALTH_QUEUE_WARNING_FULL: Final[str] = "Queue fast voll"

# Performance-Metrics-Format
PERFORMANCE_LOG_FORMAT: Final[str] = (
    "Performance: {total_tasks} Tasks, "
    "{success_rate:.1%} Success-Rate, "
    "{avg_execution_time:.2f}s Avg-Time"
)
