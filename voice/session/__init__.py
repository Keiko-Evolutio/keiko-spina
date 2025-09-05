"""Voice Session Management Module.

Refactored Voice Session Management mit verbesserter Architektur,
Separation of Concerns und Enterprise-Grade Code-Qualität.
"""

from __future__ import annotations

# =============================================================================
# Specialized Services
# =============================================================================
from .audio_processor import AudioProcessor
from .content_utils import (
    extract_and_validate_content,
    extract_content_from_item,
    sanitize_content,
    should_send_content,
    validate_content_quality,
)
from .core_operations import (
    SessionOperations,
    ThreadForwarder,
    UpdateSender,
)
from .event_handler import SessionEventHandler

# =============================================================================
# Utilities
# =============================================================================
from .role_utils import (
    is_assistant_role,
    is_system_role,
    is_user_role,
    map_role_to_enum,
    normalize_role_string,
    validate_role_string,
)
from .session_config import (
    AudioTranscriptionConfig,
    SessionConfig,
    TurnDetectionConfig,
    create_default_config,
    create_fast_response_config,
    create_high_quality_config,
)

# =============================================================================
# Constants
# =============================================================================
from .session_constants import (
    # Role Constants
    ASSISTANT_ROLE,
    DEFAULT_DETECTION_TYPE,
    DEFAULT_EAGERNESS,
    DEFAULT_PREFIX_PADDING_MS,
    DEFAULT_SILENCE_DURATION_MS,
    # Threshold Constants
    DEFAULT_THRESHOLD,
    DEFAULT_VOICE,
    FUNCTION_CALL_TYPE,
    HIGH_QUALITY_SILENCE_DURATION_MS,
    HIGH_QUALITY_THRESHOLD,
    # Event Type Constants
    MESSAGE_TYPE,
    SAGE_VOICE,
    # Detection Constants
    SEMANTIC_VAD,
    SYSTEM_ROLE,
    USER_ROLE,
    VALID_DETECTION_TYPES,
    VALID_ROLES,
    # Validation Sets
    VALID_VOICES,
    # Audio Constants
    WHISPER_MODEL,
)

# =============================================================================
# Exceptions
# =============================================================================
from .session_exceptions import (
    AudioProcessingError,
    AudioSendingError,
    ContentProcessingError,
    EventHandlingError,
    EventProcessingError,
    ResponseStartError,
    SessionConfigurationError,
    SessionNotAvailableError,
    SessionUpdateError,
    ThreadForwardingError,
    UpdateSendingError,
    VoiceSessionError,
    create_session_error,
    is_recoverable_error,
)

# =============================================================================
# Core Session Classes
# =============================================================================
from .session_refactored import RealtimeSession

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ASSISTANT_ROLE",
    "DEFAULT_DETECTION_TYPE",
    "DEFAULT_EAGERNESS",
    "DEFAULT_PREFIX_PADDING_MS",
    "DEFAULT_SILENCE_DURATION_MS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_VOICE",
    "FUNCTION_CALL_TYPE",
    "HIGH_QUALITY_SILENCE_DURATION_MS",
    "HIGH_QUALITY_THRESHOLD",
    "MESSAGE_TYPE",
    "SAGE_VOICE",
    "SEMANTIC_VAD",
    "SYSTEM_ROLE",
    "USER_ROLE",
    "VALID_DETECTION_TYPES",
    "VALID_ROLES",
    "VALID_VOICES",
    # Constants
    "WHISPER_MODEL",
    "AudioProcessingError",
    # Services
    "AudioProcessor",
    "AudioSendingError",
    "AudioTranscriptionConfig",
    "ContentProcessingError",
    "EventHandlingError",
    "EventProcessingError",
    # Core Classes
    "RealtimeSession",
    "ResponseStartError",
    "SessionConfig",
    "SessionConfigurationError",
    "SessionEventHandler",
    "SessionNotAvailableError",
    "SessionOperations",
    "SessionUpdateError",
    "ThreadForwarder",
    "ThreadForwardingError",
    "TurnDetectionConfig",
    "UpdateSender",
    "UpdateSendingError",
    # Exceptions
    "VoiceSessionError",
    # Factory Functions
    "create_default_config",
    "create_fast_response_config",
    "create_high_quality_config",
    "create_session_error",
    "extract_and_validate_content",
    "extract_content_from_item",
    "is_assistant_role",
    "is_recoverable_error",
    "is_system_role",
    "is_user_role",
    # Utilities
    "map_role_to_enum",
    "normalize_role_string",
    "sanitize_content",
    "should_send_content",
    "validate_content_quality",
    "validate_role_string",
]

# =============================================================================
# Module Metadata
# =============================================================================

__version__ = "2.0.0"
__author__ = "Keiko Development Team"
__description__ = "Refactored Voice Session Management with Enterprise-Grade Code Quality"

# =============================================================================
# Backward-Compatibility
# =============================================================================

# Legacy-Kompatibilität wurde entfernt - alle Verwendungen sollten
# die refactored RealtimeSession aus session_refactored.py verwenden
