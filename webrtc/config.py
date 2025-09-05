"""WebRTC Konfiguration für Backend

Zentrale Konfiguration für WebRTC-Services im Backend.
Umgebungsbasierte Konfiguration mit sicheren Defaults.

@version 1.0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from kei_logging import get_logger

from .types import AudioCodec, IceServer, WebRTCConfiguration

logger = get_logger(__name__)

# =============================================================================
# WebRTC Backend Konfiguration
# =============================================================================

@dataclass
class WebRTCConfig:
    """WebRTC Backend Konfiguration."""

    # Server Konfiguration
    signaling_host: str = "0.0.0.0"
    signaling_port: int = 8001
    signaling_path: str = "/webrtc-signaling"

    # STUN/TURN Server
    stun_servers: list[str] = field(default_factory=lambda: [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302"
    ])
    turn_servers: list[dict[str, str]] = field(default_factory=list)

    # Audio Konfiguration
    audio_codecs: list[str] = field(default_factory=lambda: ["opus", "G722", "PCMU"])
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    audio_bitrate: int = 64  # kbps

    # Session Management
    session_timeout: int = 300  # 5 Minuten
    max_sessions_per_user: int = 5
    session_cleanup_interval: int = 60  # 1 Minute

    # Performance und Monitoring
    enable_metrics: bool = True
    metrics_interval: int = 5  # Sekunden
    enable_performance_logging: bool = False

    # Security
    enable_dtls: bool = True
    dtls_fingerprint_algorithm: str = "sha-256"
    require_authentication: bool = True

    # Fallback Konfiguration
    enable_websocket_fallback: bool = True
    fallback_timeout: int = 10  # Sekunden

    # Debug und Logging
    debug_mode: bool = False
    log_signaling_messages: bool = False
    log_ice_candidates: bool = False

# =============================================================================
# Umgebungsbasierte Konfiguration
# =============================================================================

def get_env_str(key: str, default: str) -> str:
    """Holt String-Wert aus Umgebungsvariablen."""
    return os.getenv(key, default)

def get_env_int(key: str, default: int) -> int:
    """Holt Integer-Wert aus Umgebungsvariablen."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Ungültiger Integer-Wert für {key}, verwende Default: {default}")
        return default

def get_env_bool(key: str, default: bool) -> bool:
    """Holt Boolean-Wert aus Umgebungsvariablen."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")

def get_env_list(key: str, default: list[str], separator: str = ",") -> list[str]:
    """Holt Liste aus Umgebungsvariablen."""
    value = os.getenv(key)
    if value:
        return [item.strip() for item in value.split(separator) if item.strip()]
    return default

# =============================================================================
# Konfiguration Factory Functions
# =============================================================================

def create_webrtc_config() -> WebRTCConfig:
    """Erstellt WebRTC-Konfiguration aus Umgebungsvariablen."""
    # TURN Server aus Umgebungsvariablen
    turn_servers = []
    turn_urls = get_env_str("WEBRTC_TURN_URLS", "")
    turn_username = get_env_str("WEBRTC_TURN_USERNAME", "")
    turn_password = get_env_str("WEBRTC_TURN_PASSWORD", "")

    if turn_urls and turn_username and turn_password:
        urls = [url.strip() for url in turn_urls.split(",")]
        turn_servers.append({
            "urls": urls,
            "username": turn_username,
            "credential": turn_password
        })

    config = WebRTCConfig(
        # Server Konfiguration
        signaling_host=get_env_str("WEBRTC_SIGNALING_HOST", "0.0.0.0"),
        signaling_port=get_env_int("WEBRTC_SIGNALING_PORT", 8001),
        signaling_path=get_env_str("WEBRTC_SIGNALING_PATH", "/webrtc-signaling"),

        # STUN/TURN Server
        stun_servers=get_env_list("WEBRTC_STUN_SERVERS", [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302"
        ]),
        turn_servers=turn_servers,

        # Audio Konfiguration
        audio_codecs=get_env_list("WEBRTC_AUDIO_CODECS", ["opus", "G722", "PCMU"]),
        audio_sample_rate=get_env_int("WEBRTC_AUDIO_SAMPLE_RATE", 48000),
        audio_channels=get_env_int("WEBRTC_AUDIO_CHANNELS", 1),
        audio_bitrate=get_env_int("WEBRTC_AUDIO_BITRATE", 64),

        # Session Management
        session_timeout=get_env_int("WEBRTC_SESSION_TIMEOUT", 300),
        max_sessions_per_user=get_env_int("WEBRTC_MAX_SESSIONS_PER_USER", 5),
        session_cleanup_interval=get_env_int("WEBRTC_SESSION_CLEANUP_INTERVAL", 60),

        # Performance und Monitoring
        enable_metrics=get_env_bool("WEBRTC_ENABLE_METRICS", True),
        metrics_interval=get_env_int("WEBRTC_METRICS_INTERVAL", 5),
        enable_performance_logging=get_env_bool("WEBRTC_ENABLE_PERFORMANCE_LOGGING", False),

        # Security
        enable_dtls=get_env_bool("WEBRTC_ENABLE_DTLS", True),
        dtls_fingerprint_algorithm=get_env_str("WEBRTC_DTLS_FINGERPRINT_ALGORITHM", "sha-256"),
        require_authentication=get_env_bool("WEBRTC_REQUIRE_AUTHENTICATION", True),

        # Fallback Konfiguration
        enable_websocket_fallback=get_env_bool("WEBRTC_ENABLE_WEBSOCKET_FALLBACK", True),
        fallback_timeout=get_env_int("WEBRTC_FALLBACK_TIMEOUT", 10),

        # Debug und Logging
        debug_mode=get_env_bool("WEBRTC_DEBUG_MODE", False),
        log_signaling_messages=get_env_bool("WEBRTC_LOG_SIGNALING_MESSAGES", False),
        log_ice_candidates=get_env_bool("WEBRTC_LOG_ICE_CANDIDATES", False)
    )

    logger.info("WebRTC-Konfiguration erstellt", extra={
        "signaling_port": config.signaling_port,
        "stun_servers_count": len(config.stun_servers),
        "turn_servers_count": len(config.turn_servers),
        "audio_codecs": config.audio_codecs,
        "debug_mode": config.debug_mode
    })

    return config

def create_production_webrtc_config() -> WebRTCConfig:
    """Erstellt Production-optimierte WebRTC-Konfiguration."""
    config = create_webrtc_config()

    # Production-spezifische Overrides
    config.debug_mode = False
    config.log_signaling_messages = False
    config.log_ice_candidates = False
    config.enable_performance_logging = True
    config.require_authentication = True

    logger.info("Production WebRTC-Konfiguration erstellt")
    return config

def create_development_webrtc_config() -> WebRTCConfig:
    """Erstellt Development-optimierte WebRTC-Konfiguration."""
    config = create_webrtc_config()

    # Development-spezifische Overrides
    config.debug_mode = True
    config.log_signaling_messages = True
    config.log_ice_candidates = True
    config.enable_performance_logging = True
    config.require_authentication = False
    config.session_timeout = 600  # 10 Minuten für Development

    logger.info("Development WebRTC-Konfiguration erstellt")
    return config

# =============================================================================
# WebRTC Configuration Conversion
# =============================================================================

def webrtc_config_to_rtc_configuration(config: WebRTCConfig) -> WebRTCConfiguration:
    """Konvertiert WebRTCConfig zu WebRTCConfiguration."""
    # ICE Server erstellen
    ice_servers = []

    # STUN Server hinzufügen
    for stun_url in config.stun_servers:
        ice_servers.append(IceServer(urls=[stun_url]))

    # TURN Server hinzufügen
    for turn_config in config.turn_servers:
        ice_servers.append(IceServer(
            urls=turn_config["urls"] if isinstance(turn_config["urls"], list) else [turn_config["urls"]],
            username=turn_config.get("username"),
            credential=turn_config.get("credential")
        ))

    # Audio Codecs erstellen
    audio_codecs = []
    for i, codec_name in enumerate(config.audio_codecs):
        priority = 100 - (i * 10)  # Höhere Priorität für frühere Codecs

        codec = AudioCodec(
            codec=codec_name.lower(),
            priority=priority,
            bitrate=config.audio_bitrate
        )

        # Codec-spezifische Parameter
        if codec_name.lower() == "opus":
            codec.parameters = {
                "useinbandfec": 1,
                "usedtx": 1,
                "maxaveragebitrate": config.audio_bitrate * 1000,
                "maxplaybackrate": config.audio_sample_rate,
                "stereo": 0 if config.audio_channels == 1 else 1
            }

        audio_codecs.append(codec)

    return WebRTCConfiguration(
        ice_servers=ice_servers,
        ice_transport_policy="all",
        bundle_policy="max-bundle",
        audio_codecs=audio_codecs
    )

# =============================================================================
# Global Configuration Instance
# =============================================================================

_webrtc_config: WebRTCConfig | None = None

def get_webrtc_config() -> WebRTCConfig:
    """Gibt die globale WebRTC-Konfiguration zurück."""
    global _webrtc_config

    if _webrtc_config is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            _webrtc_config = create_production_webrtc_config()
        elif environment == "development":
            _webrtc_config = create_development_webrtc_config()
        else:
            _webrtc_config = create_webrtc_config()

    return _webrtc_config

def set_webrtc_config(config: WebRTCConfig) -> None:
    """Setzt die globale WebRTC-Konfiguration."""
    global _webrtc_config
    _webrtc_config = config
    logger.info("WebRTC-Konfiguration aktualisiert")

def reset_webrtc_config() -> None:
    """Setzt die globale WebRTC-Konfiguration zurück."""
    global _webrtc_config
    _webrtc_config = None
    logger.info("WebRTC-Konfiguration zurückgesetzt")

# =============================================================================
# Validation
# =============================================================================

def validate_webrtc_config(config: WebRTCConfig) -> bool:
    """Validiert WebRTC-Konfiguration."""
    errors = []

    # Port Validation
    if not (1024 <= config.signaling_port <= 65535):
        errors.append(f"Ungültiger Signaling Port: {config.signaling_port}")

    # Audio Validation
    if config.audio_sample_rate not in [8000, 16000, 24000, 48000]:
        errors.append(f"Ungültige Audio Sample Rate: {config.audio_sample_rate}")

    if config.audio_channels not in [1, 2]:
        errors.append(f"Ungültige Audio Channel Count: {config.audio_channels}")

    if not (8 <= config.audio_bitrate <= 320):
        errors.append(f"Ungültige Audio Bitrate: {config.audio_bitrate}")

    # Session Validation
    if config.session_timeout < 60:
        errors.append(f"Session Timeout zu niedrig: {config.session_timeout}")

    if config.max_sessions_per_user < 1:
        errors.append(f"Max Sessions per User zu niedrig: {config.max_sessions_per_user}")

    # STUN/TURN Validation
    if not config.stun_servers:
        errors.append("Keine STUN Server konfiguriert")

    if errors:
        for error in errors:
            logger.error(f"WebRTC Config Validation Error: {error}")
        return False

    return True

# =============================================================================
# Configuration Export
# =============================================================================

def export_webrtc_config_for_frontend(config: WebRTCConfig) -> dict[str, Any]:
    """Exportiert WebRTC-Konfiguration für Frontend."""
    ice_servers = []

    # STUN Server
    for stun_url in config.stun_servers:
        ice_servers.append({"urls": stun_url})

    # TURN Server
    for turn_config in config.turn_servers:
        ice_servers.append(turn_config)

    return {
        "iceServers": ice_servers,
        "iceTransportPolicy": "all",
        "bundlePolicy": "max-bundle",
        "audioCodecs": [
            {
                "codec": codec,
                "priority": 100 - (i * 10),
                "bitrate": config.audio_bitrate
            }
            for i, codec in enumerate(config.audio_codecs)
        ],
        "audioConfig": {
            "sampleRate": config.audio_sample_rate,
            "channelCount": config.audio_channels,
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
            "latency": "interactive"
        }
    }
