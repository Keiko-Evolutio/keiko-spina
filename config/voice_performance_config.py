"""Voice Performance Configuration für Keiko Personal Assistant.
Konfiguration für das Voice Performance Optimization System.
"""

import os

from voice_performance.interfaces import CacheStrategy, VoicePerformanceSettings


def get_voice_performance_settings() -> VoicePerformanceSettings:
    """Lädt Voice Performance Konfiguration aus Environment-Variablen."""

    def get_bool_env(key: str, default: bool) -> bool:
        """Hilfsfunktion für Boolean-Environment-Variablen."""
        env_value = os.getenv(key, str(default)).lower()
        return env_value in ("true", "1", "yes", "on")

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

    def get_cache_strategy_env(key: str, default: CacheStrategy) -> CacheStrategy:
        """Hilfsfunktion für CacheStrategy-Environment-Variablen."""
        try:
            strategy_value = os.getenv(key, default.value)
            return CacheStrategy(strategy_value)
        except ValueError:
            return default

    return VoicePerformanceSettings(
        # Basis-Konfiguration
        enabled=get_bool_env("KEI_VOICE_PERFORMANCE_ENABLED", True),
        max_concurrent_discoveries=get_int_env("KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_DISCOVERIES", 10),
        max_concurrent_agents=get_int_env("KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_AGENTS", 5),
        max_concurrent_tools=get_int_env("KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_TOOLS", 8),

        # Discovery Timeouts
        agent_discovery_timeout_seconds=get_float_env("KEI_VOICE_PERFORMANCE_AGENT_DISCOVERY_TIMEOUT", 3.0),
        tool_discovery_timeout_seconds=get_float_env("KEI_VOICE_PERFORMANCE_TOOL_DISCOVERY_TIMEOUT", 2.0),
        service_discovery_timeout_seconds=get_float_env("KEI_VOICE_PERFORMANCE_SERVICE_DISCOVERY_TIMEOUT", 5.0),
        capability_discovery_timeout_seconds=get_float_env("KEI_VOICE_PERFORMANCE_CAPABILITY_DISCOVERY_TIMEOUT", 1.0),

        # Performance Targets
        target_latency_ms=get_int_env("KEI_VOICE_PERFORMANCE_TARGET_LATENCY_MS", 500),
        max_latency_ms=get_int_env("KEI_VOICE_PERFORMANCE_MAX_LATENCY_MS", 2000),
        target_throughput_rps=get_int_env("KEI_VOICE_PERFORMANCE_TARGET_THROUGHPUT_RPS", 100),

        # Caching Configuration
        cache_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_CACHE_ENABLED", True),
        cache_strategy=get_cache_strategy_env("KEI_VOICE_PERFORMANCE_CACHE_STRATEGY", CacheStrategy.ADAPTIVE_CACHE),
        cache_ttl_seconds=get_int_env("KEI_VOICE_PERFORMANCE_CACHE_TTL_SECONDS", 300),
        cache_max_size=get_int_env("KEI_VOICE_PERFORMANCE_CACHE_MAX_SIZE", 10000),

        # Predictive Features
        predictive_loading_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_PREDICTIVE_LOADING_ENABLED", True),
        warm_up_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_WARM_UP_ENABLED", True),
        pattern_learning_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_PATTERN_LEARNING_ENABLED", True),

        # Resource Limits
        max_memory_usage_mb=get_int_env("KEI_VOICE_PERFORMANCE_MAX_MEMORY_USAGE_MB", 1024),
        max_cpu_usage_percent=get_float_env("KEI_VOICE_PERFORMANCE_MAX_CPU_USAGE_PERCENT", 80.0),
        max_concurrent_workflows=get_int_env("KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_WORKFLOWS", 50),

        # Quality Settings
        min_discovery_confidence=get_float_env("KEI_VOICE_PERFORMANCE_MIN_DISCOVERY_CONFIDENCE", 0.5),
        min_agent_confidence=get_float_env("KEI_VOICE_PERFORMANCE_MIN_AGENT_CONFIDENCE", 0.7),
        max_discovery_results=get_int_env("KEI_VOICE_PERFORMANCE_MAX_DISCOVERY_RESULTS", 20),

        # Monitoring
        monitoring_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_MONITORING_ENABLED", True),
        metrics_collection_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_METRICS_COLLECTION_ENABLED", True),
        performance_alerts_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_PERFORMANCE_ALERTS_ENABLED", True),

        # Circuit Breaker Integration
        circuit_breaker_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_CIRCUIT_BREAKER_ENABLED", True),
        circuit_breaker_failure_threshold=get_int_env("KEI_VOICE_PERFORMANCE_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5),
        circuit_breaker_timeout_seconds=get_int_env("KEI_VOICE_PERFORMANCE_CIRCUIT_BREAKER_TIMEOUT_SECONDS", 30),

        # Rate Limiting Integration
        rate_limiting_enabled=get_bool_env("KEI_VOICE_PERFORMANCE_RATE_LIMITING_ENABLED", True),
        rate_limiting_coordination=get_bool_env("KEI_VOICE_PERFORMANCE_RATE_LIMITING_COORDINATION", True)
    )


# Environment Template für .env Datei
VOICE_PERFORMANCE_ENV_TEMPLATE = """
# =============================================================================
# KEIKO VOICE PERFORMANCE OPTIMIZATION CONFIGURATION
# =============================================================================

# Basis-Konfiguration
KEI_VOICE_PERFORMANCE_ENABLED=true
KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_DISCOVERIES=10
KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_AGENTS=5
KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_TOOLS=8

# Discovery Timeouts (in Sekunden)
KEI_VOICE_PERFORMANCE_AGENT_DISCOVERY_TIMEOUT=3.0
KEI_VOICE_PERFORMANCE_TOOL_DISCOVERY_TIMEOUT=2.0
KEI_VOICE_PERFORMANCE_SERVICE_DISCOVERY_TIMEOUT=5.0
KEI_VOICE_PERFORMANCE_CAPABILITY_DISCOVERY_TIMEOUT=1.0

# Performance Targets
KEI_VOICE_PERFORMANCE_TARGET_LATENCY_MS=500
KEI_VOICE_PERFORMANCE_MAX_LATENCY_MS=2000
KEI_VOICE_PERFORMANCE_TARGET_THROUGHPUT_RPS=100

# Caching Configuration
KEI_VOICE_PERFORMANCE_CACHE_ENABLED=true
KEI_VOICE_PERFORMANCE_CACHE_STRATEGY=adaptive_cache
KEI_VOICE_PERFORMANCE_CACHE_TTL_SECONDS=300
KEI_VOICE_PERFORMANCE_CACHE_MAX_SIZE=10000

# Predictive Features
KEI_VOICE_PERFORMANCE_PREDICTIVE_LOADING_ENABLED=true
KEI_VOICE_PERFORMANCE_WARM_UP_ENABLED=true
KEI_VOICE_PERFORMANCE_PATTERN_LEARNING_ENABLED=true

# Resource Limits
KEI_VOICE_PERFORMANCE_MAX_MEMORY_USAGE_MB=1024
KEI_VOICE_PERFORMANCE_MAX_CPU_USAGE_PERCENT=80.0
KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_WORKFLOWS=50

# Quality Settings
KEI_VOICE_PERFORMANCE_MIN_DISCOVERY_CONFIDENCE=0.5
KEI_VOICE_PERFORMANCE_MIN_AGENT_CONFIDENCE=0.7
KEI_VOICE_PERFORMANCE_MAX_DISCOVERY_RESULTS=20

# Monitoring
KEI_VOICE_PERFORMANCE_MONITORING_ENABLED=true
KEI_VOICE_PERFORMANCE_METRICS_COLLECTION_ENABLED=true
KEI_VOICE_PERFORMANCE_PERFORMANCE_ALERTS_ENABLED=true

# Circuit Breaker Integration
KEI_VOICE_PERFORMANCE_CIRCUIT_BREAKER_ENABLED=true
KEI_VOICE_PERFORMANCE_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
KEI_VOICE_PERFORMANCE_CIRCUIT_BREAKER_TIMEOUT_SECONDS=30

# Rate Limiting Integration
KEI_VOICE_PERFORMANCE_RATE_LIMITING_ENABLED=true
KEI_VOICE_PERFORMANCE_RATE_LIMITING_COORDINATION=true

# =============================================================================
# VOICE PERFORMANCE OPTIMIZATION EXAMPLES
# =============================================================================

# Beispiel-Konfiguration für Development (weniger aggressiv)
# KEI_VOICE_PERFORMANCE_TARGET_LATENCY_MS=1000
# KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_DISCOVERIES=5
# KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_AGENTS=3

# Beispiel-Konfiguration für Production (aggressiv optimiert)
# KEI_VOICE_PERFORMANCE_TARGET_LATENCY_MS=300
# KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_DISCOVERIES=15
# KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_AGENTS=8

# Beispiel-Konfiguration für High-Load (maximale Performance)
# KEI_VOICE_PERFORMANCE_TARGET_LATENCY_MS=200
# KEI_VOICE_PERFORMANCE_MAX_CONCURRENT_WORKFLOWS=100
# KEI_VOICE_PERFORMANCE_CACHE_MAX_SIZE=50000

# Cache-Strategien
# KEI_VOICE_PERFORMANCE_CACHE_STRATEGY=no_cache          # Kein Caching
# KEI_VOICE_PERFORMANCE_CACHE_STRATEGY=memory_cache      # Einfaches Memory Caching
# KEI_VOICE_PERFORMANCE_CACHE_STRATEGY=distributed_cache # Verteiltes Caching
# KEI_VOICE_PERFORMANCE_CACHE_STRATEGY=predictive_cache  # Predictive Caching
# KEI_VOICE_PERFORMANCE_CACHE_STRATEGY=adaptive_cache    # Adaptive Caching (Standard)

# Deaktivierung für Testing
# KEI_VOICE_PERFORMANCE_ENABLED=false
# KEI_VOICE_PERFORMANCE_CACHE_ENABLED=false
# KEI_VOICE_PERFORMANCE_PREDICTIVE_LOADING_ENABLED=false
"""


def generate_voice_performance_env_template(file_path: str = ".env.voice_performance") -> None:
    """Generiert Environment-Template-Datei für Voice Performance."""
    from kei_logging import get_logger
    logger = get_logger(__name__)

    with open(file_path, "w") as f:
        f.write(VOICE_PERFORMANCE_ENV_TEMPLATE)

    logger.info(f"Voice performance environment template generated: {file_path}")


# Vordefinierte Konfigurationen für verschiedene Umgebungen
def get_development_settings() -> VoicePerformanceSettings:
    """Entwicklungs-Konfiguration mit weniger aggressiven Limits."""
    dev_settings = get_voice_performance_settings()

    # Weniger aggressive Limits für Development
    settings.target_latency_ms = 1000
    settings.max_latency_ms = 3000
    settings.max_concurrent_discoveries = 5
    settings.max_concurrent_agents = 3
    settings.max_concurrent_tools = 5
    settings.max_concurrent_workflows = 20

    # Längere Timeouts für Debugging
    settings.agent_discovery_timeout_seconds = 5.0
    settings.tool_discovery_timeout_seconds = 3.0
    settings.service_discovery_timeout_seconds = 8.0

    # Weniger aggressive Resource Limits
    settings.max_memory_usage_mb = 2048
    settings.max_cpu_usage_percent = 90.0

    return settings


def get_production_settings() -> VoicePerformanceSettings:
    """Production-Konfiguration mit aggressiver Optimierung."""
    settings = get_voice_performance_settings()

    # Aggressive Performance Targets
    settings.target_latency_ms = 300
    settings.max_latency_ms = 1000
    settings.target_throughput_rps = 200

    # Höhere Parallelität
    settings.max_concurrent_discoveries = 15
    settings.max_concurrent_agents = 8
    settings.max_concurrent_tools = 12
    settings.max_concurrent_workflows = 100

    # Kürzere Timeouts
    settings.agent_discovery_timeout_seconds = 2.0
    settings.tool_discovery_timeout_seconds = 1.5
    settings.service_discovery_timeout_seconds = 3.0
    settings.capability_discovery_timeout_seconds = 0.5

    # Erweiterte Caching
    settings.cache_max_size = 50000
    settings.cache_ttl_seconds = 600

    # Strengere Resource Limits
    settings.max_memory_usage_mb = 512
    settings.max_cpu_usage_percent = 70.0

    # Erweiterte Features
    settings.predictive_loading_enabled = True
    settings.warm_up_enabled = True
    settings.pattern_learning_enabled = True

    return settings


def get_high_load_settings() -> VoicePerformanceSettings:
    """High-Load-Konfiguration für maximale Performance."""
    settings = get_voice_performance_settings()

    # Extreme Performance Targets
    settings.target_latency_ms = 200
    settings.max_latency_ms = 500
    settings.target_throughput_rps = 500

    # Maximale Parallelität
    settings.max_concurrent_discoveries = 25
    settings.max_concurrent_agents = 15
    settings.max_concurrent_tools = 20
    settings.max_concurrent_workflows = 200

    # Sehr kurze Timeouts
    settings.agent_discovery_timeout_seconds = 1.0
    settings.tool_discovery_timeout_seconds = 0.8
    settings.service_discovery_timeout_seconds = 2.0
    settings.capability_discovery_timeout_seconds = 0.3

    # Massive Caching
    settings.cache_max_size = 100000
    settings.cache_ttl_seconds = 900
    settings.cache_strategy = CacheStrategy.PREDICTIVE_CACHE

    # Alle Performance Features aktiviert
    settings.predictive_loading_enabled = True
    settings.warm_up_enabled = True
    settings.pattern_learning_enabled = True
    settings.monitoring_enabled = True
    settings.metrics_collection_enabled = True

    return settings


def get_m4_max_settings() -> VoicePerformanceSettings:
    """Optimierte Konfiguration für MacBook Pro M4 Max."""
    settings = get_voice_performance_settings()

    # M4 Max Performance Targets
    settings.target_latency_ms = 150  # Sehr aggressiv für M4 Max
    settings.max_latency_ms = 400
    settings.target_throughput_rps = 1000  # Sehr hoch für M4 Max

    # M4 Max Parallelität (nutzt alle Cores)
    settings.max_concurrent_discoveries = 40  # M4 Max hat viele Cores
    settings.max_concurrent_agents = 20
    settings.max_concurrent_tools = 30
    settings.max_concurrent_workflows = 500  # Sehr hoch für 128GB RAM

    # Sehr kurze Timeouts für schnelle Hardware
    settings.agent_discovery_timeout_seconds = 0.5
    settings.tool_discovery_timeout_seconds = 0.3
    settings.service_discovery_timeout_seconds = 1.0
    settings.capability_discovery_timeout_seconds = 0.2

    # Massive Caching für 128GB RAM
    settings.cache_max_size = 500000  # Sehr groß für 128GB RAM
    settings.cache_ttl_seconds = 1800  # 30 Minuten
    settings.cache_strategy = CacheStrategy.PREDICTIVE_CACHE

    # Höhere Resource Limits für M4 Max
    settings.max_memory_usage_mb = 8192  # 8GB für Voice Performance
    settings.max_cpu_usage_percent = 60.0  # Kann mehr CPU nutzen

    # Alle Features aktiviert
    settings.predictive_loading_enabled = True
    settings.warm_up_enabled = True
    settings.pattern_learning_enabled = True
    settings.monitoring_enabled = True
    settings.metrics_collection_enabled = True

    return settings


def get_minimal_settings() -> VoicePerformanceSettings:
    """Minimale Konfiguration für Testing oder Resource-beschränkte Umgebungen."""
    settings = get_voice_performance_settings()

    # Minimale Performance Targets
    settings.target_latency_ms = 2000
    settings.max_latency_ms = 5000
    settings.target_throughput_rps = 10

    # Minimale Parallelität
    settings.max_concurrent_discoveries = 2
    settings.max_concurrent_agents = 1
    settings.max_concurrent_tools = 2
    settings.max_concurrent_workflows = 5

    # Längere Timeouts
    settings.agent_discovery_timeout_seconds = 10.0
    settings.tool_discovery_timeout_seconds = 8.0
    settings.service_discovery_timeout_seconds = 15.0

    # Minimales Caching
    settings.cache_max_size = 100
    settings.cache_ttl_seconds = 60
    settings.cache_strategy = CacheStrategy.MEMORY_CACHE

    # Deaktivierte Features
    settings.predictive_loading_enabled = False
    settings.warm_up_enabled = False
    settings.pattern_learning_enabled = False

    # Sehr niedrige Resource Limits
    settings.max_memory_usage_mb = 128
    settings.max_cpu_usage_percent = 50.0

    return settings


if __name__ == "__main__":
    from kei_logging import get_logger
    logger = get_logger(__name__)

    # Generiere Environment-Template
    generate_voice_performance_env_template()

    # Zeige aktuelle Konfiguration
    settings = get_voice_performance_settings()
    logger.info("Current voice performance settings:")
    for field, value in settings.__dict__.items():
        logger.info(f"  {field}: {value}")
