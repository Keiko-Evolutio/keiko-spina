# backend/agents/event_handler/constants.py
"""Konstanten für Event-Handler-Modul.

Zentrale Definition aller Magic Strings, Status-Werte und Konfigurationsparameter
für das Event-Handler-System.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

# =============================================================================
# Azure AI Foundry Status-Konstanten
# =============================================================================

class RunStatus(str, Enum):
    """Status-Werte für ThreadRun-Objekte."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    REQUIRES_ACTION = "requires_action"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


class StepStatus(str, Enum):
    """Status-Werte für RunStep-Objekte."""

    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


class MessageStatus(str, Enum):
    """Status-Werte für ThreadMessage-Objekte."""

    IN_PROGRESS = "in_progress"
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"


class StepType(str, Enum):
    """Typ-Werte für RunStep-Objekte."""

    MESSAGE_CREATION = "message_creation"
    TOOL_CALLS = "tool_calls"


# =============================================================================
# Event-Handler Status-Formatierung
# =============================================================================

class EventType(str, Enum):
    """Event-Typen für Status-Formatierung."""

    RUN = "run"
    STEP = "step"
    MESSAGE = "message"


# =============================================================================
# Display-Status-Konstanten
# =============================================================================

# Formatierte Status-Nachrichten für Client-Anzeige
DISPLAY_STATUS_MESSAGE_COMPLETED: Final[str] = "message completed"

# Informations-Nachrichten
INFO_MESSAGE_RESULT_RECEIVED: Final[str] = "Ergebnis erhalten"

# Error-Nachrichten
ERROR_AGENT_STREAM: Final[str] = "Agent-Stream-Fehler"
ERROR_TOOL_EXECUTION: Final[str] = "Fehler bei Tool-Ausführung"
ERROR_TOOL_OUTPUT_SUBMISSION: Final[str] = "Tool-Output-Submission fehlgeschlagen"

# Warning-Nachrichten
WARNING_TOOL_NOT_FOUND: Final[str] = "Tool nicht in Registry gefunden"

DEBUG_TOOL_EXECUTED: Final[str] = "Tool erfolgreich ausgeführt"
DEBUG_TOOL_OUTPUTS_SENT: Final[str] = "Tool-Outputs erfolgreich gesendet"
DEBUG_UNHANDLED_EVENT: Final[str] = "Unbehandeltes Event"

# =============================================================================
# Content-Type-Konstanten
# =============================================================================

class ContentType(str, Enum):
    """Content-Typen für Message-Content."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"


# =============================================================================
# Tool-Call-Konstanten
# =============================================================================

# JSON-Serialisierung
JSON_ENSURE_ASCII: Final[bool] = False

# =============================================================================
# Event-Deduplication-Konstanten
# =============================================================================

# Cache-Konfiguration für Event-Deduplication
DEFAULT_DEDUPLICATION_CACHE_SIZE: Final[int] = 1000
DEFAULT_DEDUPLICATION_TTL_SECONDS: Final[int] = 3600  # 1 Stunde

# =============================================================================
# Logging-Konstanten
# =============================================================================

# Log-Level-Mapping für verschiedene Event-Typen
LOG_LEVEL_ERROR_EVENTS: Final[list[str]] = [
    "stream_error",
    "tool_execution_error",
    "submission_error"
]

LOG_LEVEL_WARNING_EVENTS: Final[list[str]] = [
    "tool_not_found",
    "invalid_tool_call"
]

LOG_LEVEL_DEBUG_EVENTS: Final[list[str]] = [
    "tool_executed",
    "outputs_sent",
    "unhandled_event"
]

# =============================================================================
# Performance-Konstanten
# =============================================================================

# Timeouts für verschiedene Operationen
TOOL_EXECUTION_TIMEOUT_SECONDS: Final[int] = 30
TOOL_OUTPUT_SUBMISSION_TIMEOUT_SECONDS: Final[int] = 10
EVENT_DISPATCH_TIMEOUT_SECONDS: Final[int] = 5

# Retry-Konfiguration
MAX_TOOL_EXECUTION_RETRIES: Final[int] = 3
MAX_SUBMISSION_RETRIES: Final[int] = 2
RETRY_BACKOFF_FACTOR: Final[float] = 1.5

# =============================================================================
# Validation-Konstanten
# =============================================================================

# Maximale Größen für verschiedene Datentypen
MAX_TOOL_OUTPUT_SIZE_BYTES: Final[int] = 1024 * 1024  # 1 MB
MAX_EVENT_PAYLOAD_SIZE_BYTES: Final[int] = 512 * 1024  # 512 KB
MAX_TOOL_ARGUMENTS_SIZE_BYTES: Final[int] = 64 * 1024  # 64 KB

# String-Längen-Limits
MAX_TOOL_NAME_LENGTH: Final[int] = 100
MAX_STATUS_MESSAGE_LENGTH: Final[int] = 500
MAX_ERROR_MESSAGE_LENGTH: Final[int] = 1000

# =============================================================================
# Feature-Flags
# =============================================================================

# Experimentelle Features
ENABLE_ASYNC_TOOL_EXECUTION: Final[bool] = True
ENABLE_TOOL_EXECUTION_METRICS: Final[bool] = True
ENABLE_EVENT_DEDUPLICATION: Final[bool] = True
ENABLE_PERFORMANCE_MONITORING: Final[bool] = True

ENABLE_VERBOSE_LOGGING: Final[bool] = False
ENABLE_TOOL_CALL_TRACING: Final[bool] = False
ENABLE_EVENT_HISTORY_TRACKING: Final[bool] = False

# =============================================================================
# Error-Codes
# =============================================================================

class EventHandlerErrorCode(str, Enum):
    """Error-Codes für Event-Handler-Fehler."""

    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    EVENT_DISPATCH_FAILED = "EVENT_DISPATCH_FAILED"
    DEDUPLICATION_ERROR = "DEDUPLICATION_ERROR"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"


# =============================================================================
# Utility-Konstanten
# =============================================================================

# Regex-Patterns für Validation
TOOL_NAME_PATTERN: Final[str] = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
EVENT_ID_PATTERN: Final[str] = r"^[a-zA-Z0-9\-_]+$"

# Default-Werte
DEFAULT_EVENT_TYPE: Final[str] = "unknown"
DEFAULT_STATUS: Final[str] = "processing"
DEFAULT_OBJECT_TYPE: Final[str] = "event"

# =============================================================================
# Export-Liste
# =============================================================================

__all__ = [
    # Enums
    "RunStatus",
    "StepStatus",
    "MessageStatus",
    "StepType",
    "EventType",
    "ContentType",
    "EventHandlerErrorCode",

    # Display-Status
    "DISPLAY_STATUS_MESSAGE_COMPLETED",

    # Messages
    "INFO_MESSAGE_RESULT_RECEIVED",
    "ERROR_AGENT_STREAM",
    "ERROR_TOOL_EXECUTION",
    "ERROR_TOOL_OUTPUT_SUBMISSION",
    "WARNING_TOOL_NOT_FOUND",
    "DEBUG_TOOL_EXECUTED",
    "DEBUG_TOOL_OUTPUTS_SENT",
    "DEBUG_UNHANDLED_EVENT",

    # Configuration
    "JSON_ENSURE_ASCII",
    "DEFAULT_DEDUPLICATION_CACHE_SIZE",
    "DEFAULT_DEDUPLICATION_TTL_SECONDS",

    # Performance
    "TOOL_EXECUTION_TIMEOUT_SECONDS",
    "TOOL_OUTPUT_SUBMISSION_TIMEOUT_SECONDS",
    "EVENT_DISPATCH_TIMEOUT_SECONDS",
    "MAX_TOOL_EXECUTION_RETRIES",
    "MAX_SUBMISSION_RETRIES",
    "RETRY_BACKOFF_FACTOR",

    # Validation
    "MAX_TOOL_OUTPUT_SIZE_BYTES",
    "MAX_EVENT_PAYLOAD_SIZE_BYTES",
    "MAX_TOOL_ARGUMENTS_SIZE_BYTES",
    "MAX_TOOL_NAME_LENGTH",
    "MAX_STATUS_MESSAGE_LENGTH",
    "MAX_ERROR_MESSAGE_LENGTH",

    # Feature Flags
    "ENABLE_ASYNC_TOOL_EXECUTION",
    "ENABLE_TOOL_EXECUTION_METRICS",
    "ENABLE_EVENT_DEDUPLICATION",
    "ENABLE_PERFORMANCE_MONITORING",
    "ENABLE_VERBOSE_LOGGING",
    "ENABLE_TOOL_CALL_TRACING",
    "ENABLE_EVENT_HISTORY_TRACKING",

    # Patterns
    "TOOL_NAME_PATTERN",
    "EVENT_ID_PATTERN",

    # Defaults
    "DEFAULT_EVENT_TYPE",
    "DEFAULT_STATUS",
    "DEFAULT_OBJECT_TYPE",
]
