"""Voice Rate Limiting Configuration für Keiko Personal Assistant.
Konfiguration für das Voice-spezifische Rate Limiting System.
"""

import os

from voice_rate_limiting.interfaces import VoiceRateLimitSettings


def get_voice_rate_limiting_settings() -> VoiceRateLimitSettings:
    """Lädt Voice Rate Limiting Konfiguration aus Environment-Variablen."""

    def get_bool_env(key: str, default: bool) -> bool:
        """Hilfsfunktion für Boolean-Environment-Variablen."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    # Produktionsreife Lösung: Prüfe explizit auf Development-Modus
    # Wenn KEI_VOICE_RATE_LIMITING_ENABLED explizit auf false gesetzt ist, respektiere das
    voice_rate_limiting_enabled = get_bool_env("KEI_VOICE_RATE_LIMITING_ENABLED", False)

    # Development-Override: Wenn die .env.voice_rate_limiting Datei existiert und enabled=false enthält
    env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env.voice_rate_limiting")
    if os.path.exists(env_file_path):
        try:
            with open(env_file_path) as f:
                content = f.read()
                if "KEI_VOICE_RATE_LIMITING_ENABLED=false" in content:
                    voice_rate_limiting_enabled = False
        except Exception:
            pass  # Fallback auf Environment-Variable

    def get_int_env(key: str, default: int) -> int:
        """Hilfsfunktion für Integer-Environment-Variablen."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    def get_float_env(key: str, default: float) -> float:
        """Hilfsfunktion für Float-Environment-Variablen."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    return VoiceRateLimitSettings(
        # Basis-Konfiguration - Verwendet die produktionsreife Lösung
        enabled=voice_rate_limiting_enabled,
        redis_enabled=get_bool_env("KEI_VOICE_RATE_LIMITING_REDIS_ENABLED", True),  # Default: True wenn aioredis verfügbar
        redis_url=os.getenv("KEI_VOICE_RATE_LIMITING_REDIS_URL", "redis://localhost:6379"),

        # Globale Limits
        global_requests_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_GLOBAL_REQUESTS_PER_MINUTE", 1000),
        global_concurrent_connections=get_int_env("KEI_VOICE_RATE_LIMITING_GLOBAL_CONCURRENT_CONNECTIONS", 100),

        # Granulare Zeitfenster-basierte Limits
        # Speech-to-Text Limits
        stt_requests_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_STT_REQUESTS_PER_MINUTE", 60),
        stt_requests_per_hour=get_int_env("KEI_VOICE_RATE_LIMITING_STT_REQUESTS_PER_HOUR", 1000),
        stt_requests_per_day=get_int_env("KEI_VOICE_RATE_LIMITING_STT_REQUESTS_PER_DAY", 10000),

        # Text-to-Speech Limits
        tts_requests_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_TTS_REQUESTS_PER_MINUTE", 30),
        tts_requests_per_hour=get_int_env("KEI_VOICE_RATE_LIMITING_TTS_REQUESTS_PER_HOUR", 500),
        tts_requests_per_day=get_int_env("KEI_VOICE_RATE_LIMITING_TTS_REQUESTS_PER_DAY", 5000),

        # Realtime Streaming Limits
        streaming_requests_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_STREAMING_REQUESTS_PER_MINUTE", 10),
        streaming_requests_per_hour=get_int_env("KEI_VOICE_RATE_LIMITING_STREAMING_REQUESTS_PER_HOUR", 100),
        streaming_requests_per_day=get_int_env("KEI_VOICE_RATE_LIMITING_STREAMING_REQUESTS_PER_DAY", 1000),

        # WebSocket Connection Limits
        websocket_connections_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_WEBSOCKET_CONNECTIONS_PER_MINUTE", 3),
        websocket_connections_per_hour=get_int_env("KEI_VOICE_RATE_LIMITING_WEBSOCKET_CONNECTIONS_PER_HOUR", 20),
        websocket_connections_per_day=get_int_env("KEI_VOICE_RATE_LIMITING_WEBSOCKET_CONNECTIONS_PER_DAY", 100),

        # Agent Execution Limits
        agent_executions_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_AGENT_EXECUTIONS_PER_MINUTE", 20),
        agent_executions_per_hour=get_int_env("KEI_VOICE_RATE_LIMITING_AGENT_EXECUTIONS_PER_HOUR", 200),
        agent_executions_per_day=get_int_env("KEI_VOICE_RATE_LIMITING_AGENT_EXECUTIONS_PER_DAY", 2000),

        # Tool Call Limits
        tool_calls_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_TOOL_CALLS_PER_MINUTE", 50),
        tool_calls_per_hour=get_int_env("KEI_VOICE_RATE_LIMITING_TOOL_CALLS_PER_HOUR", 500),
        tool_calls_per_day=get_int_env("KEI_VOICE_RATE_LIMITING_TOOL_CALLS_PER_DAY", 5000),

        # User-basierte Limits (per Minute)
        user_speech_to_text_limit=get_int_env("KEI_VOICE_RATE_LIMITING_USER_STT_LIMIT", 60),
        user_voice_synthesis_limit=get_int_env("KEI_VOICE_RATE_LIMITING_USER_TTS_LIMIT", 30),
        user_realtime_streaming_limit=get_int_env("KEI_VOICE_RATE_LIMITING_USER_STREAMING_LIMIT", 10),
        user_websocket_connections=get_int_env("KEI_VOICE_RATE_LIMITING_USER_WEBSOCKET_CONNECTIONS", 3),
        user_agent_executions_limit=get_int_env("KEI_VOICE_RATE_LIMITING_USER_AGENT_EXECUTIONS_LIMIT", 20),
        user_tool_calls_limit=get_int_env("KEI_VOICE_RATE_LIMITING_USER_TOOL_CALLS_LIMIT", 50),

        # Session-basierte Limits
        session_workflow_starts_limit=get_int_env("KEI_VOICE_RATE_LIMITING_SESSION_WORKFLOW_STARTS_LIMIT", 100),
        session_audio_uploads_limit=get_int_env("KEI_VOICE_RATE_LIMITING_SESSION_AUDIO_UPLOADS_LIMIT", 200),
        session_text_inputs_limit=get_int_env("KEI_VOICE_RATE_LIMITING_SESSION_TEXT_INPUTS_LIMIT", 500),

        # IP-basierte Limits
        ip_requests_per_minute=get_int_env("KEI_VOICE_RATE_LIMITING_IP_REQUESTS_PER_MINUTE", 200),
        ip_concurrent_connections=get_int_env("KEI_VOICE_RATE_LIMITING_IP_CONCURRENT_CONNECTIONS", 10),

        # User-Tier-spezifische Multiplier
        anonymous_multiplier=get_float_env("KEI_VOICE_RATE_LIMITING_ANONYMOUS_MULTIPLIER", 0.5),
        standard_multiplier=get_float_env("KEI_VOICE_RATE_LIMITING_STANDARD_MULTIPLIER", 1.0),
        premium_multiplier=get_float_env("KEI_VOICE_RATE_LIMITING_PREMIUM_MULTIPLIER", 2.0),
        enterprise_multiplier=get_float_env("KEI_VOICE_RATE_LIMITING_ENTERPRISE_MULTIPLIER", 5.0),

        # Adaptive Limits
        adaptive_enabled=get_bool_env("KEI_VOICE_RATE_LIMITING_ADAPTIVE_ENABLED", True),
        cpu_threshold=get_float_env("KEI_VOICE_RATE_LIMITING_CPU_THRESHOLD", 0.8),
        memory_threshold=get_float_env("KEI_VOICE_RATE_LIMITING_MEMORY_THRESHOLD", 0.8),
        peak_hours_start=get_int_env("KEI_VOICE_RATE_LIMITING_PEAK_HOURS_START", 9),
        peak_hours_end=get_int_env("KEI_VOICE_RATE_LIMITING_PEAK_HOURS_END", 17),
        peak_hours_multiplier=get_float_env("KEI_VOICE_RATE_LIMITING_PEAK_HOURS_MULTIPLIER", 0.7),

        # Burst-Konfiguration
        burst_enabled=get_bool_env("KEI_VOICE_RATE_LIMITING_BURST_ENABLED", True),
        burst_multiplier=get_float_env("KEI_VOICE_RATE_LIMITING_BURST_MULTIPLIER", 1.5),

        # Monitoring
        monitoring_enabled=get_bool_env("KEI_VOICE_RATE_LIMITING_MONITORING_ENABLED", True),
        alert_threshold=get_float_env("KEI_VOICE_RATE_LIMITING_ALERT_THRESHOLD", 0.9)
    )


# Environment Template für .env Datei
VOICE_RATE_LIMITING_ENV_TEMPLATE = """
# =============================================================================
# KEIKO VOICE RATE LIMITING CONFIGURATION
# =============================================================================

# Basis-Konfiguration
KEI_VOICE_RATE_LIMITING_ENABLED=true
KEI_VOICE_RATE_LIMITING_REDIS_ENABLED=true
KEI_VOICE_RATE_LIMITING_REDIS_URL=redis://localhost:6379

# Globale Limits
KEI_VOICE_RATE_LIMITING_GLOBAL_REQUESTS_PER_MINUTE=1000
KEI_VOICE_RATE_LIMITING_GLOBAL_CONCURRENT_CONNECTIONS=100

# User-basierte Limits (per Minute)
KEI_VOICE_RATE_LIMITING_USER_STT_LIMIT=60
KEI_VOICE_RATE_LIMITING_USER_TTS_LIMIT=30
KEI_VOICE_RATE_LIMITING_USER_STREAMING_LIMIT=10
KEI_VOICE_RATE_LIMITING_USER_WEBSOCKET_CONNECTIONS=3
KEI_VOICE_RATE_LIMITING_USER_AGENT_EXECUTIONS_LIMIT=20
KEI_VOICE_RATE_LIMITING_USER_TOOL_CALLS_LIMIT=50

# Session-basierte Limits
KEI_VOICE_RATE_LIMITING_SESSION_WORKFLOW_STARTS_LIMIT=100
KEI_VOICE_RATE_LIMITING_SESSION_AUDIO_UPLOADS_LIMIT=200
KEI_VOICE_RATE_LIMITING_SESSION_TEXT_INPUTS_LIMIT=500

# IP-basierte Limits
KEI_VOICE_RATE_LIMITING_IP_REQUESTS_PER_MINUTE=200
KEI_VOICE_RATE_LIMITING_IP_CONCURRENT_CONNECTIONS=10

# User-Tier Multiplier
KEI_VOICE_RATE_LIMITING_ANONYMOUS_MULTIPLIER=0.5
KEI_VOICE_RATE_LIMITING_STANDARD_MULTIPLIER=1.0
KEI_VOICE_RATE_LIMITING_PREMIUM_MULTIPLIER=2.0
KEI_VOICE_RATE_LIMITING_ENTERPRISE_MULTIPLIER=5.0

# Adaptive Rate Limiting
KEI_VOICE_RATE_LIMITING_ADAPTIVE_ENABLED=true
KEI_VOICE_RATE_LIMITING_CPU_THRESHOLD=0.8
KEI_VOICE_RATE_LIMITING_MEMORY_THRESHOLD=0.8
KEI_VOICE_RATE_LIMITING_PEAK_HOURS_START=9
KEI_VOICE_RATE_LIMITING_PEAK_HOURS_END=17
KEI_VOICE_RATE_LIMITING_PEAK_HOURS_MULTIPLIER=0.7

# Burst-Konfiguration
KEI_VOICE_RATE_LIMITING_BURST_ENABLED=true
KEI_VOICE_RATE_LIMITING_BURST_MULTIPLIER=1.5

# Monitoring
KEI_VOICE_RATE_LIMITING_MONITORING_ENABLED=true
KEI_VOICE_RATE_LIMITING_ALERT_THRESHOLD=0.9

# =============================================================================
# VOICE RATE LIMITING EXAMPLES
# =============================================================================

# Beispiel-Konfiguration für Development (niedrigere Limits)
# KEI_VOICE_RATE_LIMITING_USER_STT_LIMIT=10
# KEI_VOICE_RATE_LIMITING_USER_TTS_LIMIT=5
# KEI_VOICE_RATE_LIMITING_USER_STREAMING_LIMIT=2

# Beispiel-Konfiguration für Production (höhere Limits)
# KEI_VOICE_RATE_LIMITING_USER_STT_LIMIT=120
# KEI_VOICE_RATE_LIMITING_USER_TTS_LIMIT=60
# KEI_VOICE_RATE_LIMITING_USER_STREAMING_LIMIT=20

# Beispiel-Konfiguration für High-Traffic (sehr hohe Limits)
# KEI_VOICE_RATE_LIMITING_GLOBAL_REQUESTS_PER_MINUTE=5000
# KEI_VOICE_RATE_LIMITING_GLOBAL_CONCURRENT_CONNECTIONS=500
# KEI_VOICE_RATE_LIMITING_USER_STT_LIMIT=300
# KEI_VOICE_RATE_LIMITING_USER_TTS_LIMIT=150

# Redis Cluster Konfiguration (für High Availability)
# KEI_VOICE_RATE_LIMITING_REDIS_URL=redis://redis-cluster-1:6379,redis://redis-cluster-2:6379,redis://redis-cluster-3:6379

# Deaktivierung für Testing
# KEI_VOICE_RATE_LIMITING_ENABLED=false
"""


def generate_voice_rate_limiting_env_template(file_path: str = ".env.voice_rate_limiting") -> None:
    """Generiert Environment-Template-Datei für Voice Rate Limiting."""
    from kei_logging import get_logger
    logger = get_logger(__name__)

    with open(file_path, "w") as f:
        f.write(VOICE_RATE_LIMITING_ENV_TEMPLATE)

    logger.info(f"Voice rate limiting environment template generated: {file_path}")


# Vordefinierte Konfigurationen für verschiedene Umgebungen
def get_development_settings() -> VoiceRateLimitSettings:
    """Entwicklungs-Konfiguration mit niedrigeren Limits."""
    settings = get_voice_rate_limiting_settings()

    # Niedrigere Limits für Development
    settings.user_speech_to_text_limit = 10
    settings.user_voice_synthesis_limit = 5
    settings.user_realtime_streaming_limit = 2
    settings.user_websocket_connections = 2
    settings.user_agent_executions_limit = 10
    settings.user_tool_calls_limit = 20

    # Redis optional für Development
    settings.redis_enabled = False

    return settings


def get_production_settings() -> VoiceRateLimitSettings:
    """Production-Konfiguration mit optimierten Limits."""
    settings = get_voice_rate_limiting_settings()

    # Höhere Limits für Production
    settings.user_speech_to_text_limit = 120
    settings.user_voice_synthesis_limit = 60
    settings.user_realtime_streaming_limit = 20
    settings.user_websocket_connections = 5
    settings.user_agent_executions_limit = 40
    settings.user_tool_calls_limit = 100

    # Globale Limits erhöhen
    settings.global_requests_per_minute = 2000
    settings.global_concurrent_connections = 200

    # Redis erforderlich für Production
    settings.redis_enabled = True

    return settings


def get_redis_production_settings() -> VoiceRateLimitSettings:
    """Production-Konfiguration mit Redis aktiviert."""
    settings = get_production_settings()

    # Redis explizit aktivieren
    settings.redis_enabled = True
    settings.redis_url = os.getenv("KEI_VOICE_RATE_LIMITING_REDIS_URL", "redis://localhost:6379")

    return settings


def get_high_traffic_settings() -> VoiceRateLimitSettings:
    """High-Traffic-Konfiguration für große Deployments."""
    settings = get_voice_rate_limiting_settings()

    # Sehr hohe Limits für High-Traffic
    settings.user_speech_to_text_limit = 300
    settings.user_voice_synthesis_limit = 150
    settings.user_realtime_streaming_limit = 50
    settings.user_websocket_connections = 10
    settings.user_agent_executions_limit = 100
    settings.user_tool_calls_limit = 250

    # Globale Limits stark erhöhen
    settings.global_requests_per_minute = 5000
    settings.global_concurrent_connections = 500

    # Enterprise-Multiplier erhöhen
    settings.enterprise_multiplier = 10.0

    return settings


if __name__ == "__main__":
    from kei_logging import get_logger
    logger = get_logger(__name__)

    # Generiere Environment-Template
    generate_voice_rate_limiting_env_template()

    # Zeige aktuelle Konfiguration
    settings = get_voice_rate_limiting_settings()
    logger.info("Current voice rate limiting settings:")
    for field, value in settings.__dict__.items():
        logger.info(f"  {field}: {value}")
