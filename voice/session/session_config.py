"""Session-Konfiguration für Voice Session Management.

Strukturierte Konfiguration für Realtime-Sessions mit Type-Safety,
Validation und Default-Werten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .session_constants import (
    # Default-Werte
    DEFAULT_DETECTION_TYPE,
    DEFAULT_EAGERNESS,
    DEFAULT_PREFIX_PADDING_MS,
    DEFAULT_SILENCE_DURATION_MS,
    DEFAULT_THRESHOLD,
    DEFAULT_VOICE,
    FAST_RESPONSE_PREFIX_PADDING_MS,
    FAST_RESPONSE_SILENCE_DURATION_MS,
    # Fast-Response-Werte
    FAST_RESPONSE_THRESHOLD,
    HIGH_QUALITY_SILENCE_DURATION_MS,
    # High-Quality-Werte
    HIGH_QUALITY_THRESHOLD,
    MAX_PREFIX_PADDING_MS,
    MAX_SILENCE_DURATION_MS,
    MAX_THRESHOLD,
    MIN_PREFIX_PADDING_MS,
    MIN_SILENCE_DURATION_MS,
    # Limits
    MIN_THRESHOLD,
    VALID_DETECTION_TYPES,
    VALID_EAGERNESS_LEVELS,
    VALID_TRANSCRIPTION_MODELS,
    # Validation-Sets
    VALID_VOICES,
    WHISPER_MODEL,
    get_env_detection_type,
    get_env_eagerness,
    get_env_prefix_padding,
    get_env_silence_duration,
    get_env_threshold,
    get_env_transcription_model,
    # Environment-Funktionen
    get_env_voice,
)


@dataclass
class TurnDetectionConfig:
    """Konfiguration für Turn-Detection."""

    type: str = field(default_factory=get_env_detection_type)
    threshold: float = field(default_factory=get_env_threshold)
    silence_duration_ms: int = field(default_factory=get_env_silence_duration)
    prefix_padding_ms: int = field(default_factory=get_env_prefix_padding)
    eagerness: str = field(default_factory=get_env_eagerness)

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte nach Initialisierung."""
        self._validate()

    def _validate(self) -> None:
        """Validiert alle Konfigurationswerte."""
        if self.type not in VALID_DETECTION_TYPES:
            raise ValueError(f"Ungültiger Detection-Type: {self.type}. Gültig: {VALID_DETECTION_TYPES}")

        if not (MIN_THRESHOLD <= self.threshold <= MAX_THRESHOLD):
            raise ValueError(f"Threshold muss zwischen {MIN_THRESHOLD} und {MAX_THRESHOLD} liegen: {self.threshold}")

        if not (MIN_SILENCE_DURATION_MS <= self.silence_duration_ms <= MAX_SILENCE_DURATION_MS):
            raise ValueError(f"Silence-Duration muss zwischen {MIN_SILENCE_DURATION_MS} und {MAX_SILENCE_DURATION_MS} ms liegen: {self.silence_duration_ms}")

        if not (MIN_PREFIX_PADDING_MS <= self.prefix_padding_ms <= MAX_PREFIX_PADDING_MS):
            raise ValueError(f"Prefix-Padding muss zwischen {MIN_PREFIX_PADDING_MS} und {MAX_PREFIX_PADDING_MS} ms liegen: {self.prefix_padding_ms}")

        if self.eagerness not in VALID_EAGERNESS_LEVELS:
            raise ValueError(f"Ungültiger Eagerness-Level: {self.eagerness}. Gültig: {VALID_EAGERNESS_LEVELS}")

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für API-Calls."""
        return {
            "type": self.type,
            "threshold": self.threshold,
            "silence_duration_ms": self.silence_duration_ms,
            "prefix_padding_ms": self.prefix_padding_ms,
            "eagerness": self.eagerness,
        }


@dataclass
class AudioTranscriptionConfig:
    """Konfiguration für Audio-Transcription."""

    model: str = field(default_factory=get_env_transcription_model)

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte nach Initialisierung."""
        self._validate()

    def _validate(self) -> None:
        """Validiert Transcription-Model."""
        if self.model not in VALID_TRANSCRIPTION_MODELS:
            raise ValueError(f"Ungültiges Transcription-Model: {self.model}. Gültig: {VALID_TRANSCRIPTION_MODELS}")

    def to_dict(self) -> dict[str, str]:
        """Konvertiert zu Dictionary für API-Calls."""
        return {"model": self.model}


@dataclass
class SessionConfig:
    """Vollständige Session-Konfiguration für Realtime-Sessions."""

    instructions: str
    voice: str = field(default_factory=get_env_voice)
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)
    input_audio_transcription: AudioTranscriptionConfig = field(default_factory=AudioTranscriptionConfig)
    tools: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Validiert Konfigurationswerte nach Initialisierung."""
        self._validate()

    def _validate(self) -> None:
        """Validiert alle Konfigurationswerte."""
        if not self.instructions or not self.instructions.strip():
            raise ValueError("Instructions dürfen nicht leer sein")

        if self.voice not in VALID_VOICES:
            raise ValueError(f"Ungültige Voice: {self.voice}. Gültig: {VALID_VOICES}")

        if self.tools is None:
            self.tools = []

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für API-Calls."""
        return {
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_transcription": self.input_audio_transcription.to_dict(),
            "turn_detection": self.turn_detection.to_dict(),
            "tools": self.tools or [],
        }

    @classmethod
    def from_legacy_params(
        cls,
        *,
        instructions: str,
        detection_type: str = DEFAULT_DETECTION_TYPE,
        transcription_model: str = WHISPER_MODEL,
        voice: str = DEFAULT_VOICE,
        threshold: float = DEFAULT_THRESHOLD,
        silence_duration_ms: int = DEFAULT_SILENCE_DURATION_MS,
        prefix_padding_ms: int = DEFAULT_PREFIX_PADDING_MS,
        eagerness: str = DEFAULT_EAGERNESS,
        tools: list[dict[str, Any]] | None = None,
    ) -> SessionConfig:
        """Erstellt SessionConfig aus Legacy-Parametern für Backward-Compatibility."""
        turn_detection = TurnDetectionConfig(
            type=detection_type,
            threshold=threshold,
            silence_duration_ms=silence_duration_ms,
            prefix_padding_ms=prefix_padding_ms,
            eagerness=eagerness,
        )

        audio_transcription = AudioTranscriptionConfig(model=transcription_model)

        return cls(
            instructions=instructions,
            voice=voice,
            turn_detection=turn_detection,
            input_audio_transcription=audio_transcription,
            tools=tools,
        )

    def update_voice(self, voice: str) -> SessionConfig:
        """Erstellt neue Konfiguration mit geänderter Voice."""
        if voice not in VALID_VOICES:
            raise ValueError(f"Ungültige Voice: {voice}. Gültig: {VALID_VOICES}")

        return SessionConfig(
            instructions=self.instructions,
            voice=voice,
            turn_detection=self.turn_detection,
            input_audio_transcription=self.input_audio_transcription,
            tools=self.tools,
        )

    def update_threshold(self, threshold: float) -> SessionConfig:
        """Erstellt neue Konfiguration mit geändertem Threshold."""
        new_turn_detection = TurnDetectionConfig(
            type=self.turn_detection.type,
            threshold=threshold,
            silence_duration_ms=self.turn_detection.silence_duration_ms,
            prefix_padding_ms=self.turn_detection.prefix_padding_ms,
            eagerness=self.turn_detection.eagerness,
        )

        return SessionConfig(
            instructions=self.instructions,
            voice=self.voice,
            turn_detection=new_turn_detection,
            input_audio_transcription=self.input_audio_transcription,
            tools=self.tools,
        )

    def update_tools(self, tools: list[dict[str, Any]]) -> SessionConfig:
        """Erstellt neue Konfiguration mit geänderten Tools."""
        return SessionConfig(
            instructions=self.instructions,
            voice=self.voice,
            turn_detection=self.turn_detection,
            input_audio_transcription=self.input_audio_transcription,
            tools=tools,
        )


# =============================================================================
# Factory-Funktionen für häufige Konfigurationen
# =============================================================================

def create_default_config(instructions: str) -> SessionConfig:
    """Erstellt Standard-Konfiguration mit Default-Werten."""
    return SessionConfig(instructions=instructions)


def create_high_quality_config(instructions: str) -> SessionConfig:
    """Erstellt High-Quality-Konfiguration für beste Audio-Qualität."""
    turn_detection = TurnDetectionConfig(
        threshold=HIGH_QUALITY_THRESHOLD,  # Höhere Threshold für weniger false positives
        silence_duration_ms=HIGH_QUALITY_SILENCE_DURATION_MS,  # Längere Pause für sicherere Detection
        prefix_padding_ms=DEFAULT_PREFIX_PADDING_MS,  # Mehr Padding für bessere Qualität
    )

    return SessionConfig(
        instructions=instructions,
        turn_detection=turn_detection,
    )


def create_fast_response_config(instructions: str) -> SessionConfig:
    """Erstellt Konfiguration für schnelle Responses."""
    turn_detection = TurnDetectionConfig(
        threshold=FAST_RESPONSE_THRESHOLD,  # Niedrigere Threshold für schnellere Detection
        silence_duration_ms=FAST_RESPONSE_SILENCE_DURATION_MS,  # Kürzere Pause für schnellere Response
        prefix_padding_ms=FAST_RESPONSE_PREFIX_PADDING_MS,  # Weniger Padding für Speed
        eagerness="high",  # Hohe Eagerness für schnelle Antworten
    )

    return SessionConfig(
        instructions=instructions,
        turn_detection=turn_detection,
    )
