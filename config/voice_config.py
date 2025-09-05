"""Voice Detection und Audio Processing Konfiguration.

Externalisierte Konfiguration für Voice Service mit Environment-Variable-Support
und Validierung für Produktions- und Entwicklungsumgebung.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

from .env_utils import get_env_bool, get_env_float, get_env_int, get_env_str


@dataclass
class VoiceDetectionConfig:
    """Voice Activity Detection Konfiguration mit Environment-Variable-Support.

    Alle Parameter können über Environment-Variablen überschrieben werden.
    Fallback auf sinnvolle Default-Werte für Produktions- und Entwicklungsumgebung.
    """

    # Voice Activity Detection (VAD) Einstellungen
    detection_type: str = field(
        default_factory=lambda: get_env_str("VOICE_DETECTION_TYPE", "server_vad")
    )
    threshold: float = field(
        default_factory=lambda: get_env_float("VOICE_THRESHOLD", 0.8)
    )
    prefix_padding_ms: int = field(
        default_factory=lambda: get_env_int("VOICE_PREFIX_PADDING_MS", 300)
    )
    silence_duration_ms: int = field(
        default_factory=lambda: get_env_int("VOICE_SILENCE_DURATION_MS", 1000)
    )
    create_response: bool = field(
        default_factory=lambda: get_env_bool("VOICE_CREATE_RESPONSE", True)
    )

    # Audio Format Einstellungen
    input_audio_format: str = field(
        default_factory=lambda: get_env_str("VOICE_INPUT_AUDIO_FORMAT", "pcm16")
    )
    output_audio_format: str = field(
        default_factory=lambda: get_env_str("VOICE_OUTPUT_AUDIO_FORMAT", "pcm16")
    )

    # Speech-to-Text Konfiguration
    transcription_model: str = field(
        default_factory=lambda: get_env_str("VOICE_TRANSCRIPTION_MODEL", "whisper-1")
    )
    transcription_language: str = field(
        default_factory=lambda: get_env_str("VOICE_TRANSCRIPTION_LANGUAGE", "de")
    )

    # Voice Synthesis Einstellungen
    voice: str = field(
        default_factory=lambda: get_env_str("VOICE_SYNTHESIS_VOICE", "echo")
    )
    temperature: float = field(
        default_factory=lambda: get_env_float("VOICE_TEMPERATURE", 0.7)
    )
    max_response_output_tokens: int = field(
        default_factory=lambda: get_env_int("VOICE_MAX_RESPONSE_TOKENS", 4096)
    )
    speech_rate: float = field(
        default_factory=lambda: get_env_float("VOICE_SPEECH_RATE", 1.0)
    )

    def __post_init__(self):
        """Validiert Konfigurationswerte nach Initialisierung."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validiert alle Konfigurationswerte.

        Raises:
            ValueError: Bei ungültigen Konfigurationswerten
        """
        # Threshold Validierung (0.0 - 1.0)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Voice threshold muss zwischen 0.0 und 1.0 liegen, erhalten: {self.threshold}")

        # Timing Validierungen (positive Werte)
        if self.prefix_padding_ms < 0:
            raise ValueError(f"Prefix padding muss positiv sein, erhalten: {self.prefix_padding_ms}")

        if self.silence_duration_ms < 100:
            raise ValueError(f"Silence duration muss mindestens 100ms sein, erhalten: {self.silence_duration_ms}")

        # Temperature Validierung (0.0 - 2.0)
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature muss zwischen 0.0 und 2.0 liegen, erhalten: {self.temperature}")

        # Speech Rate Validierung (0.25 - 4.0)
        if not 0.25 <= self.speech_rate <= 4.0:
            raise ValueError(f"Speech rate muss zwischen 0.25 und 4.0 liegen, erhalten: {self.speech_rate}")

        # Token Limit Validierung
        if self.max_response_output_tokens < 1:
            raise ValueError(f"Max response tokens muss positiv sein, erhalten: {self.max_response_output_tokens}")

        # Detection Type Validierung
        valid_detection_types = ["server_vad", "none"]
        if self.detection_type not in valid_detection_types:
            raise ValueError(f"Detection type muss einer von {valid_detection_types} sein, erhalten: {self.detection_type}")

        # Audio Format Validierung
        valid_audio_formats = ["pcm16", "g711_ulaw", "g711_alaw"]
        if self.input_audio_format not in valid_audio_formats:
            raise ValueError(f"Input audio format muss einer von {valid_audio_formats} sein, erhalten: {self.input_audio_format}")

        if self.output_audio_format not in valid_audio_formats:
            raise ValueError(f"Output audio format muss einer von {valid_audio_formats} sein, erhalten: {self.output_audio_format}")

    def to_turn_detection_dict(self) -> dict[str, Any]:
        """Konvertiert Konfiguration zu turn_detection Dictionary für Azure OpenAI.

        Returns:
            Dictionary im Azure OpenAI turn_detection Format
        """
        return {
            "type": self.detection_type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response
        }

    def to_transcription_dict(self) -> dict[str, str]:
        """Konvertiert Konfiguration zu input_audio_transcription Dictionary.

        Returns:
            Dictionary im Azure OpenAI transcription Format
        """
        return {
            "model": self.transcription_model,
            "language": self.transcription_language
        }

    @classmethod
    def for_environment(cls, environment: str = "development") -> "VoiceDetectionConfig":
        """Erstellt umgebungsspezifische Konfiguration.

        Args:
            environment: Zielumgebung (development, staging, production)

        Returns:
            VoiceDetectionConfig für spezifische Umgebung
        """
        if environment == "production":
            # Produktionseinstellungen: Konservativere Werte für Stabilität
            return cls(
                # Voice Activity Detection - Konservative Werte
                detection_type=get_env_str("VOICE_DETECTION_TYPE", "server_vad"),
                threshold=get_env_float("VOICE_THRESHOLD", 0.9),  # Höhere Schwelle
                prefix_padding_ms=get_env_int("VOICE_PREFIX_PADDING_MS", 500),  # Mehr Padding
                silence_duration_ms=get_env_int("VOICE_SILENCE_DURATION_MS", 1500),  # Längere Pause
                create_response=get_env_bool("VOICE_CREATE_RESPONSE", True),

                # Audio Format
                input_audio_format=get_env_str("VOICE_INPUT_AUDIO_FORMAT", "pcm16"),
                output_audio_format=get_env_str("VOICE_OUTPUT_AUDIO_FORMAT", "pcm16"),

                # Speech-to-Text
                transcription_model=get_env_str("VOICE_TRANSCRIPTION_MODEL", "whisper-1"),
                transcription_language=get_env_str("VOICE_TRANSCRIPTION_LANGUAGE", "de"),

                # Voice Synthesis - Konservative Werte
                voice=get_env_str("VOICE_SYNTHESIS_VOICE", "echo"),
                temperature=get_env_float("VOICE_TEMPERATURE", 0.5),  # Niedrigere Temperature
                max_response_output_tokens=get_env_int("VOICE_MAX_RESPONSE_TOKENS", 4096),
                speech_rate=get_env_float("VOICE_SPEECH_RATE", 1.0),
            )
        if environment == "staging":
            # Staging: Ausgewogene Einstellungen
            return cls(
                # Voice Activity Detection - Ausgewogene Werte
                detection_type=get_env_str("VOICE_DETECTION_TYPE", "server_vad"),
                threshold=get_env_float("VOICE_THRESHOLD", 0.85),  # Mittlere Schwelle
                prefix_padding_ms=get_env_int("VOICE_PREFIX_PADDING_MS", 400),  # Moderates Padding
                silence_duration_ms=get_env_int("VOICE_SILENCE_DURATION_MS", 1200),  # Mittlere Pause
                create_response=get_env_bool("VOICE_CREATE_RESPONSE", True),

                # Audio Format
                input_audio_format=get_env_str("VOICE_INPUT_AUDIO_FORMAT", "pcm16"),
                output_audio_format=get_env_str("VOICE_OUTPUT_AUDIO_FORMAT", "pcm16"),

                # Speech-to-Text
                transcription_model=get_env_str("VOICE_TRANSCRIPTION_MODEL", "whisper-1"),
                transcription_language=get_env_str("VOICE_TRANSCRIPTION_LANGUAGE", "de"),

                # Voice Synthesis - Ausgewogene Werte
                voice=get_env_str("VOICE_SYNTHESIS_VOICE", "echo"),
                temperature=get_env_float("VOICE_TEMPERATURE", 0.6),  # Mittlere Temperature
                max_response_output_tokens=get_env_int("VOICE_MAX_RESPONSE_TOKENS", 4096),
                speech_rate=get_env_float("VOICE_SPEECH_RATE", 1.0),
            )
        # Development: Standard-Einstellungen (responsive)
        return cls()


class VoiceServiceSettings(BaseSettings):
    """Pydantic-basierte Voice Service Einstellungen für erweiterte Validierung."""

    # Voice Detection Einstellungen
    voice_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Voice Activity Detection Schwellenwert (0.0-1.0)"
    )

    voice_silence_duration_ms: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Silence Duration in Millisekunden (100-10000)"
    )

    voice_prefix_padding_ms: int = Field(
        default=300,
        ge=0,
        le=2000,
        description="Prefix Padding in Millisekunden (0-2000)"
    )

    voice_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Voice Generation Temperature (0.0-2.0)"
    )

    voice_speech_rate: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech Rate Multiplier (0.25-4.0)"
    )

    voice_max_response_tokens: int = Field(
        default=4096,
        ge=1,
        le=16384,
        description="Maximum Response Tokens (1-16384)"
    )

    class Config:
        """Pydantic Konfiguration."""
        env_prefix = "VOICE_"
        case_sensitive = False


# Globale Konfigurationsinstanz
_voice_config: VoiceDetectionConfig | None = None


def get_voice_config(environment: str | None = None) -> VoiceDetectionConfig:
    """Lädt oder gibt Voice Detection Konfiguration zurück.

    Args:
        environment: Optionale Umgebung für spezifische Konfiguration

    Returns:
        VoiceDetectionConfig Instanz
    """
    global _voice_config

    if _voice_config is None or environment is not None:
        if environment:
            _voice_config = VoiceDetectionConfig.for_environment(environment)
        else:
            current_env = os.getenv("KEIKO_ENVIRONMENT", "development")
            _voice_config = VoiceDetectionConfig.for_environment(current_env)

    return _voice_config


def reload_voice_config(environment: str | None = None) -> VoiceDetectionConfig:
    """Lädt Voice Konfiguration neu.

    Args:
        environment: Optionale Umgebung für spezifische Konfiguration

    Returns:
        Neu geladene VoiceDetectionConfig Instanz
    """
    global _voice_config
    _voice_config = None
    return get_voice_config(environment)
