# backend/data_models/constants.py
"""Konstanten für Data Models - Zentrale Definition aller Magic Numbers und Hard-coded Strings."""

from __future__ import annotations

# =====================================================================
# API Constants
# =====================================================================

# Pagination Defaults
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 100
DEFAULT_SORT_FIELD = "created_at"
DEFAULT_SORT_ORDER = "desc"

# Voice Settings Defaults
DEFAULT_VOICE_SPEED = 1.0
MIN_VOICE_SPEED = 0.5
MAX_VOICE_SPEED = 2.0
DEFAULT_VOICE_PITCH = 0.0
MIN_VOICE_PITCH = -1.0
MAX_VOICE_PITCH = 1.0
DEFAULT_VOICE_VOLUME = 1.0
MIN_VOICE_VOLUME = 0.0
MAX_VOICE_VOLUME = 2.0

# Field Length Constraints
MIN_NAME_LENGTH = 3
MAX_NAME_LENGTH = 100
MIN_SYSTEM_MESSAGE_LENGTH = 10

# =====================================================================
# WebSocket Constants
# =====================================================================

# Event ID Prefixes
EVENT_ID_PREFIX = "evt_"
EVENT_ID_LENGTH = 8

# Audio Defaults
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_LANGUAGE = "de-DE"
DEFAULT_SAMPLE_RATE = 16000

# Progress Range
MIN_PROGRESS = 0.0
MAX_PROGRESS = 1.0

# Connection Status Values
CONNECTION_STATUS_CONNECTED = "connected"
CONNECTION_STATUS_DISCONNECTED = "disconnected"
CONNECTION_STATUS_RECONNECTING = "reconnecting"
CONNECTION_STATUS_ERROR = "error"

# =====================================================================
# Core Model Constants
# =====================================================================

# Default Values
DEFAULT_VERSION = "1.0.0"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_STATUS = "unknown"
DEFAULT_CATEGORY = "general"
DEFAULT_SEVERITY = "medium"

# Error Codes
DEFAULT_ERROR_CODE = "UNKNOWN_ERROR"
DEFAULT_ERROR_MESSAGE = "Unbekannter Fehler aufgetreten"

# Update Types (String Values)
UPDATE_TYPE_AGENT = "agent"
UPDATE_TYPE_AUDIO = "audio"
UPDATE_TYPE_CONSOLE = "console"
UPDATE_TYPE_ERROR = "error"
UPDATE_TYPE_FUNCTION = "function"
UPDATE_TYPE_FUNCTION_COMPLETION = "function_completion"
UPDATE_TYPE_MESSAGE = "message"
UPDATE_TYPE_SETTINGS = "settings"
UPDATE_TYPE_INTERRUPT = "interrupt"

# Role Values (String Values)
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_ORCHESTRATOR = "orchestrator"
ROLE_SPECIALIST = "specialist"
ROLE_TOOL = "tool"

# =====================================================================
# A2A Constants
# =====================================================================

# A2A Role Values
A2A_ROLE_USER = "user"
A2A_ROLE_ASSISTANT = "assistant"
A2A_ROLE_SYSTEM = "system"
A2A_ROLE_TOOL = "tool"

# =====================================================================
# Error Severity Levels
# =====================================================================

SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

# =====================================================================
# Agent Configuration Categories
# =====================================================================

CATEGORY_GENERAL = "general"
CATEGORY_TECHNICAL = "technical"
CATEGORY_CREATIVE = "creative"
CATEGORY_BUSINESS = "business"

# =====================================================================
# WebSocket Event Types
# =====================================================================

EVENT_TYPE_AGENT_RESPONSE = "agent_response"
EVENT_TYPE_STATUS_UPDATE = "status_update"
EVENT_TYPE_ERROR = "error"
EVENT_TYPE_FUNCTION_CALL = "function_call"
EVENT_TYPE_FUNCTION_RESULT = "function_result"
EVENT_TYPE_VOICE_INPUT = "voice_input"
EVENT_TYPE_VOICE_OUTPUT = "voice_output"
EVENT_TYPE_CONNECTION_STATUS = "connection_status"

# =====================================================================
# Message Types
# =====================================================================

MESSAGE_TYPE_FUNCTION_CONFIRMATION = "function_confirmation"

# =====================================================================
# Validation Messages
# =====================================================================

# German Error Messages
ERROR_MSG_CONTENT_REQUIRED = "content ist erforderlich"
ERROR_MSG_NAME_REQUIRED = "name ist erforderlich"
ERROR_MSG_DESCRIPTION_REQUIRED = "description ist erforderlich"
ERROR_MSG_TYPE_REQUIRED = "type ist erforderlich"
ERROR_MSG_INSTRUCTIONS_REQUIRED = "instructions ist erforderlich"
ERROR_MSG_EMPTY_NOT_ALLOWED = "darf nicht leer sein"
ERROR_MSG_INVALID_URI = "ungültige URI"

# A2A Specific Error Messages
A2A_ERROR_PREFIX = "A2A: "
A2A_ERROR_CONTENT_REQUIRED = f"{A2A_ERROR_PREFIX}{ERROR_MSG_CONTENT_REQUIRED}"
A2A_ERROR_TOOL_NAME_REQUIRED = f"{A2A_ERROR_PREFIX}tool_call.name ist erforderlich"
A2A_ERROR_INVALID_URI = f"{A2A_ERROR_PREFIX}ungültige Attachment-URI"
A2A_ERROR_CORR_ID_EMPTY = f"{A2A_ERROR_PREFIX}corr_id {ERROR_MSG_EMPTY_NOT_ALLOWED}"
A2A_ERROR_TRACEPARENT_EMPTY = f"{A2A_ERROR_PREFIX}traceparent {ERROR_MSG_EMPTY_NOT_ALLOWED}"

# =====================================================================
# Success Messages
# =====================================================================

SUCCESS_MSG_OPERATION_SUCCESSFUL = "Operation erfolgreich"

# =====================================================================
# Regex Patterns
# =====================================================================

SORT_ORDER_PATTERN = "^(asc|desc)$"

# =====================================================================
# Time and Date Formats
# =====================================================================

ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# =====================================================================
# Package Metadata
# =====================================================================

PACKAGE_VERSION = "2.0.0"
PACKAGE_AUTHOR = "Keiko Development Team"

# =====================================================================
# Logging Messages
# =====================================================================

LOG_MSG_CORE_MODELS_LOADED = "✅ Core Models erfolgreich geladen"
LOG_MSG_CORE_MODELS_INITIALIZED = "📦 Core Models Package initialisiert"
LOG_MSG_OPENAI_SDK_AVAILABLE = "🔧 OpenAI SDK SessionTool verfügbar"
LOG_MSG_EMPTY_CONTENT_WARNING = "Leerer content für {role}-Message mit ID {update_id}"
LOG_MSG_EMPTY_FUNCTION_NAME_WARNING = "Leerer Funktionsname für call_id {call_id}"

# =====================================================================
# HTTP Status Codes (for API errors)
# =====================================================================

HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_CONFLICT = 409
HTTP_STATUS_TOO_MANY_REQUESTS = 429
HTTP_STATUS_INTERNAL_SERVER_ERROR = 500

# =====================================================================
# Default Error Codes
# =====================================================================

ERROR_CODE_VALIDATION_ERROR = "VALIDATION_ERROR"
ERROR_CODE_NOT_FOUND = "NOT_FOUND"
ERROR_CODE_CONFLICT = "CONFLICT"
ERROR_CODE_RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
ERROR_CODE_INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

# =====================================================================
# Field Names (for validation)
# =====================================================================

FIELD_NAME_CONFIGURATION_NAME = "Configuration.name"
FIELD_NAME_FUNCTION_NAME = "Function.name"
FIELD_NAME_FUNCTION_DESCRIPTION = "Function.description"
FIELD_NAME_FUNCTION_PARAMETER_NAME = "FunctionParameter.name"
FIELD_NAME_FUNCTION_PARAMETER_TYPE = "FunctionParameter.type"
FIELD_NAME_DEFAULT_CONFIGURATION_INSTRUCTIONS = "DefaultConfiguration.instructions"
