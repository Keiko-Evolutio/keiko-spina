# backend/data_models/__init__.py
"""Data Models Paket für Domain-Objekte, API und WebSocket-Kommunikation."""

from __future__ import annotations

# Paket-Metadaten
from .constants import PACKAGE_AUTHOR, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__author__ = PACKAGE_AUTHOR

# Pydantic ValidationError Alias
from pydantic import ValidationError as PydanticValidationError

# A2A Models - Agent-to-Agent Kommunikation
from .a2a import (
    A2AAttachment,
    A2AEnvelope,
    A2AMessage,
    A2ARole,
    A2AToolCall,
)

# API Models - Request/Response-Strukturen
from .api import (
    AgentConfigurationCategory,
    # Request Models
    AgentConfigurationCreateRequest,
    AgentConfigurationListResponse,
    AgentConfigurationResponse,
    AgentConfigurationUpdateRequest,
    # Error Models
    APIError,
    # Response Models
    BaseResponse,
    ConfigurationStatsResponse,
    ConflictError,
    ErrorResponse,
    InternalServerError,
    NotFoundError,
    PaginatedResponse,
    PaginationMeta,
    PaginationRequest,
    RateLimitError,
    SuccessResponse,
    ToolConfiguration,
    ValidationDetail,
    ValidationError,
    VoiceSettings,
)
from .constants import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_CATEGORY,
    DEFAULT_LOG_LEVEL,
    # Häufig verwendete Konstanten
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    DEFAULT_STATUS,
    DEFAULT_VERSION,
    DEFAULT_VOICE_SPEED,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)

# Core Models - Domain-Objekte und Update-System
from .core import (
    # Domain-Objekte
    Agent,
    # Update-System
    AgentUpdate,
    # Event-System
    AgentUpdateEvent,
    AudioUpdate,
    Configuration,
    ConsoleUpdate,
    Content,
    DefaultConfiguration,
    ErrorUpdate,
    Function,
    FunctionCompletionUpdate,
    FunctionParameter,
    FunctionUpdate,
    MessageUpdate,
    Role,
    SettingsUpdate,
    Update,
    UpdateFactory,
    UpdateType,
)

# Utility-Funktionen und Konstanten exportieren
from .utils import (
    ValidationMixin,
    generate_short_id,
    generate_uuid,
    utc_now,
    validate_non_empty_string,
)

# WebSocket Models - Event-Strukturen
from .websocket import (
    AgentResponseEvent,
    BaseWebSocketEvent,
    ConnectionStatusEvent,
    ErrorEvent,
    EventType,
    FunctionCallEvent,
    FunctionConfirmationMessage,
    FunctionResultEvent,
    StatusUpdateEvent,
    VoiceInputEvent,
    VoiceOutputEvent,
    WebSocketEvent,
)

__all__ = [
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_CATEGORY",
    "DEFAULT_LOG_LEVEL",
    # Constants (häufig verwendete)
    "DEFAULT_PAGE",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_STATUS",
    "DEFAULT_VERSION",
    "DEFAULT_VOICE_SPEED",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "A2AAttachment",
    "A2AEnvelope",
    "A2AMessage",
    # A2A Models
    "A2ARole",
    "A2AToolCall",
    "APIError",
    # Core Models
    "Agent",
    "AgentConfigurationCategory",
    # API Models
    "AgentConfigurationCreateRequest",
    "AgentConfigurationListResponse",
    "AgentConfigurationResponse",
    "AgentConfigurationUpdateRequest",
    "AgentResponseEvent",
    "AgentUpdate",
    "AgentUpdateEvent",
    "AudioUpdate",
    "BaseResponse",
    "BaseWebSocketEvent",
    "Configuration",
    "ConfigurationStatsResponse",
    "ConflictError",
    "ConnectionStatusEvent",
    "ConsoleUpdate",
    "Content",
    "DefaultConfiguration",
    "ErrorEvent",
    "ErrorResponse",
    "ErrorUpdate",
    # WebSocket Models
    "EventType",
    "Function",
    "FunctionCallEvent",
    "FunctionCompletionUpdate",
    "FunctionConfirmationMessage",
    "FunctionParameter",
    "FunctionResultEvent",
    "FunctionUpdate",
    "InternalServerError",
    "MessageUpdate",
    "NotFoundError",
    "PaginatedResponse",
    "PaginationMeta",
    "PaginationRequest",
    # Validation
    "PydanticValidationError",
    "RateLimitError",
    "Role",
    "SettingsUpdate",
    "StatusUpdateEvent",
    "SuccessResponse",
    "ToolConfiguration",
    "Update",
    "UpdateFactory",
    "UpdateType",
    "ValidationDetail",
    "ValidationError",
    "ValidationMixin",
    "VoiceInputEvent",
    "VoiceOutputEvent",
    "VoiceSettings",
    "WebSocketEvent",
    "generate_short_id",
    # Utility Functions
    "generate_uuid",
    "utc_now",
    "validate_non_empty_string",
]
