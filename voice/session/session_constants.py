"""Konstanten für Voice Session Management.

Zentrale Definition aller Magic Numbers, Hard-coded Strings und
Konfigurationswerte für das Voice Session System.
"""

from __future__ import annotations

import os
from typing import Final

# =============================================================================
# Audio-Konstanten
# =============================================================================

# Transcription Models
WHISPER_MODEL: Final[str] = "whisper-1"
WHISPER_MODEL_TURBO: Final[str] = "whisper-1-turbo"

# Voice Models
SAGE_VOICE: Final[str] = "sage"
ALLOY_VOICE: Final[str] = "alloy"
ECHO_VOICE: Final[str] = "echo"
FABLE_VOICE: Final[str] = "fable"
ONYX_VOICE: Final[str] = "onyx"
NOVA_VOICE: Final[str] = "nova"
SHIMMER_VOICE: Final[str] = "shimmer"

# Standard Voice für Session
DEFAULT_VOICE: Final[str] = SAGE_VOICE

# =============================================================================
# Detection-Konstanten
# =============================================================================

# Turn Detection Types
SEMANTIC_VAD: Final[str] = "semantic_vad"
SERVER_VAD: Final[str] = "server_vad"
NONE_DETECTION: Final[str] = "none"

# Standard Detection Type
DEFAULT_DETECTION_TYPE: Final[str] = SEMANTIC_VAD

# Eagerness Settings
AUTO_EAGERNESS: Final[str] = "auto"
LOW_EAGERNESS: Final[str] = "low"
MEDIUM_EAGERNESS: Final[str] = "medium"
HIGH_EAGERNESS: Final[str] = "high"

# Standard Eagerness
DEFAULT_EAGERNESS: Final[str] = AUTO_EAGERNESS

# =============================================================================
# Threshold und Timing-Konstanten
# =============================================================================

# Detection Thresholds
DEFAULT_THRESHOLD: Final[float] = 0.8
MIN_THRESHOLD: Final[float] = 0.0
MAX_THRESHOLD: Final[float] = 1.0

# Timing-Konstanten (in Millisekunden)
DEFAULT_SILENCE_DURATION_MS: Final[int] = 500
MIN_SILENCE_DURATION_MS: Final[int] = 100
MAX_SILENCE_DURATION_MS: Final[int] = 5000

DEFAULT_PREFIX_PADDING_MS: Final[int] = 300
MIN_PREFIX_PADDING_MS: Final[int] = 0
MAX_PREFIX_PADDING_MS: Final[int] = 1000

# High-Quality-Konfiguration
HIGH_QUALITY_THRESHOLD: Final[float] = 0.9
HIGH_QUALITY_SILENCE_DURATION_MS: Final[int] = 800

# Fast-Response-Konfiguration
FAST_RESPONSE_THRESHOLD: Final[float] = 0.6
FAST_RESPONSE_SILENCE_DURATION_MS: Final[int] = 200
FAST_RESPONSE_PREFIX_PADDING_MS: Final[int] = 100

# Audio-Processing-Konstanten
MAX_AUDIO_CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024  # 1MB

# =============================================================================
# Role-Konstanten
# =============================================================================

# Role-Strings für API-Kommunikation
ASSISTANT_ROLE: Final[str] = "assistant"
USER_ROLE: Final[str] = "user"
SYSTEM_ROLE: Final[str] = "system"

# Standard Role
DEFAULT_ROLE: Final[str] = ASSISTANT_ROLE

# =============================================================================
# Event-Type-Konstanten
# =============================================================================

# Output Item Types
MESSAGE_TYPE: Final[str] = "message"
FUNCTION_CALL_TYPE: Final[str] = "function_call"
AUDIO_TYPE: Final[str] = "audio"

# Event Types
RESPONSE_DONE_EVENT: Final[str] = "response.done"
OUTPUT_ITEM_DONE_EVENT: Final[str] = "response.output_item.done"
AUDIO_DELTA_EVENT: Final[str] = "response.audio.delta"
SPEECH_STARTED_EVENT: Final[str] = "input_audio_buffer.speech_started"

# =============================================================================
# Error-Message-Konstanten
# =============================================================================

# Session-Errors
SESSION_NOT_AVAILABLE_MSG: Final[str] = "Realtime-Session nicht verfügbar"
SESSION_UPDATE_ERROR_MSG: Final[str] = "Fehler Session-Update"
SESSION_CONFIG_UPDATED_MSG: Final[str] = "Session-Konfiguration aktualisiert"

# Audio-Errors
AUDIO_SENDING_ERROR_MSG: Final[str] = "Fehler Audio-Sending"
RESPONSE_START_ERROR_MSG: Final[str] = "Fehler Response-Start"

# Update-Errors
UPDATE_SENDING_ERROR_MSG: Final[str] = "Fehler beim Senden des Updates"
THREAD_FORWARDING_ERROR_MSG: Final[str] = "Fehler Thread-Weiterleitung"
IMPORT_ERROR_MSG: Final[str] = "Import-Fehler create_thread_message"

# Response-Messages
RESPONSE_COMPLETED_MSG: Final[str] = "Response abgeschlossen"
OUTPUT_ITEM_DONE_MSG: Final[str] = "OutputItemDone: %s - ID: %s"
MESSAGE_FORWARDED_MSG: Final[str] = "Nachricht an Thread weitergeleitet: %s"

# =============================================================================
# Environment-Variable-basierte Konfiguration
# =============================================================================

def get_env_voice() -> str:
    """Liefert Voice-Model aus Environment oder Default."""
    return os.getenv("KEIKO_VOICE_MODEL", DEFAULT_VOICE)


def get_env_detection_type() -> str:
    """Liefert Detection-Type aus Environment oder Default."""
    return os.getenv("KEIKO_DETECTION_TYPE", DEFAULT_DETECTION_TYPE)


def get_env_transcription_model() -> str:
    """Liefert Transcription-Model aus Environment oder Default."""
    return os.getenv("KEIKO_TRANSCRIPTION_MODEL", WHISPER_MODEL)


def get_env_threshold() -> float:
    """Liefert Threshold aus Environment oder Default."""
    try:
        threshold = float(os.getenv("KEIKO_THRESHOLD", str(DEFAULT_THRESHOLD)))
        return max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
    except (ValueError, TypeError):
        return DEFAULT_THRESHOLD


def get_env_silence_duration() -> int:
    """Liefert Silence-Duration aus Environment oder Default."""
    try:
        duration = int(os.getenv("KEIKO_SILENCE_DURATION_MS", str(DEFAULT_SILENCE_DURATION_MS)))
        return max(MIN_SILENCE_DURATION_MS, min(MAX_SILENCE_DURATION_MS, duration))
    except (ValueError, TypeError):
        return DEFAULT_SILENCE_DURATION_MS


def get_env_prefix_padding() -> int:
    """Liefert Prefix-Padding aus Environment oder Default."""
    try:
        padding = int(os.getenv("KEIKO_PREFIX_PADDING_MS", str(DEFAULT_PREFIX_PADDING_MS)))
        return max(MIN_PREFIX_PADDING_MS, min(MAX_PREFIX_PADDING_MS, padding))
    except (ValueError, TypeError):
        return DEFAULT_PREFIX_PADDING_MS


def get_env_eagerness() -> str:
    """Liefert Eagerness aus Environment oder Default."""
    return os.getenv("KEIKO_EAGERNESS", DEFAULT_EAGERNESS)


# =============================================================================
# Validation-Konstanten
# =============================================================================

# Gültige Werte für Validation
VALID_VOICES: Final[set[str]] = {
    SAGE_VOICE, ALLOY_VOICE, ECHO_VOICE, FABLE_VOICE,
    ONYX_VOICE, NOVA_VOICE, SHIMMER_VOICE
}

VALID_DETECTION_TYPES: Final[set[str]] = {
    SEMANTIC_VAD, SERVER_VAD, NONE_DETECTION
}

VALID_TRANSCRIPTION_MODELS: Final[set[str]] = {
    WHISPER_MODEL, WHISPER_MODEL_TURBO
}

VALID_EAGERNESS_LEVELS: Final[set[str]] = {
    AUTO_EAGERNESS, LOW_EAGERNESS, MEDIUM_EAGERNESS, HIGH_EAGERNESS
}

VALID_ROLES: Final[set[str]] = {
    ASSISTANT_ROLE, USER_ROLE, SYSTEM_ROLE
}

VALID_EVENT_TYPES: Final[set[str]] = {
    MESSAGE_TYPE, FUNCTION_CALL_TYPE, AUDIO_TYPE
}

# =============================================================================
# Logging-Level-Konstanten
# =============================================================================

INFO_LEVEL: Final[str] = "info"
WARNING_LEVEL: Final[str] = "warning"
ERROR_LEVEL: Final[str] = "error"
DEBUG_LEVEL: Final[str] = "debug"
