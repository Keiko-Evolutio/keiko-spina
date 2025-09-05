"""Konstanten für das State-Management-Modul."""

from __future__ import annotations

from typing import Final

DEFAULT_EMPTY_MESSAGE: Final[str] = ""
DEFAULT_STEP_VALUE: Final[int] = 0
DEFAULT_THREAD_ID: Final[str] = "default"

WORKFLOW_NOT_FOUND_ERROR: Final[str] = "Workflow {name} nicht registriert"
WORKFLOW_START_ERROR: Final[str] = "Workflow Start fehlgeschlagen: {error}"
WORKFLOW_RESUME_ERROR: Final[str] = "Workflow Resume fehlgeschlagen: {error}"
INVALID_CONFIG_ERROR: Final[str] = "Ungültige Konfiguration: {details}"
INVALID_STATE_ERROR: Final[str] = "Ungültiger State: {details}"

TRACE_WORKFLOW_START: Final[str] = "workflow.start"
TRACE_WORKFLOW_RESUME: Final[str] = "workflow.resume"
TRACE_STATE_SERIALIZATION: Final[str] = "state.serialization"
TRACE_STATE_DESERIALIZATION: Final[str] = "state.deserialization"

MAX_WORKFLOW_NAME_LENGTH: Final[int] = 100
MAX_THREAD_ID_LENGTH: Final[int] = 255
MAX_MESSAGE_LENGTH: Final[int] = 10000

LOG_WORKFLOW_REGISTRATION: Final[str] = "Workflow registriert: {name}"
LOG_WORKFLOW_START: Final[str] = "Workflow gestartet: {name} (Thread: {thread_id})"
LOG_WORKFLOW_RESUME: Final[str] = "Workflow fortgesetzt: {name} (Thread: {thread_id})"
LOG_STATE_SERIALIZED: Final[str] = "State serialisiert: {step} Schritte"

LOG_STATE_DESERIALIZED: Final[str] = "State deserialisiert: {step} Schritte"
"""Log-Message für State-Deserialisierung."""
