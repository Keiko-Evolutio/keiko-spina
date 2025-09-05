"""Agent Circuit Breaker Configuration für Keiko Personal Assistant.
Konfiguration für das Agent-spezifische Circuit Breaker System.
"""

import os

from agents.circuit_breaker.interfaces import AgentCircuitBreakerSettings, RecoveryStrategy


def get_agent_circuit_breaker_settings() -> AgentCircuitBreakerSettings:
    """Lädt Agent Circuit Breaker Konfiguration aus Environment-Variablen."""

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

    def get_recovery_strategy_env(key: str, default: RecoveryStrategy) -> RecoveryStrategy:
        """Hilfsfunktion für RecoveryStrategy-Environment-Variablen."""
        try:
            strategy_value = os.getenv(key, default.value)
            return RecoveryStrategy(strategy_value)
        except ValueError:
            return default

    return AgentCircuitBreakerSettings(
        # Basis-Konfiguration
        enabled=get_bool_env("KEI_AGENT_CIRCUIT_BREAKER_ENABLED", True),
        monitoring_enabled=get_bool_env("KEI_AGENT_CIRCUIT_BREAKER_MONITORING_ENABLED", True),
        fallback_enabled=get_bool_env("KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_ENABLED", True),
        caching_enabled=get_bool_env("KEI_AGENT_CIRCUIT_BREAKER_CACHING_ENABLED", True),

        # Default Circuit Breaker Settings
        default_failure_threshold=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_FAILURE_THRESHOLD", 5),
        default_recovery_timeout_seconds=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_RECOVERY_TIMEOUT", 60),
        default_success_threshold=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_SUCCESS_THRESHOLD", 3),
        default_timeout_seconds=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_TIMEOUT", 30.0),

        # Agent-Type-spezifische Settings
        voice_agent_failure_threshold=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_VOICE_FAILURE_THRESHOLD", 3),
        voice_agent_timeout_seconds=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_VOICE_TIMEOUT", 10.0),
        tool_agent_failure_threshold=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_TOOL_FAILURE_THRESHOLD", 5),
        tool_agent_timeout_seconds=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_TOOL_TIMEOUT", 30.0),
        workflow_agent_failure_threshold=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_WORKFLOW_FAILURE_THRESHOLD", 2),
        workflow_agent_timeout_seconds=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_WORKFLOW_TIMEOUT", 60.0),
        orchestrator_agent_failure_threshold=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_ORCHESTRATOR_FAILURE_THRESHOLD", 2),
        orchestrator_agent_timeout_seconds=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_ORCHESTRATOR_TIMEOUT", 15.0),

        # Recovery-Konfiguration
        recovery_strategy=get_recovery_strategy_env("KEI_AGENT_CIRCUIT_BREAKER_RECOVERY_STRATEGY", RecoveryStrategy.EXPONENTIAL_BACKOFF),
        max_recovery_attempts=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_MAX_RECOVERY_ATTEMPTS", 3),
        backoff_multiplier=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_BACKOFF_MULTIPLIER", 2.0),
        max_backoff_seconds=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_MAX_BACKOFF", 300),

        # Fallback-Konfiguration
        fallback_timeout_seconds=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_TIMEOUT", 15.0),
        max_fallback_attempts=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_MAX_FALLBACK_ATTEMPTS", 2),
        fallback_cache_enabled=get_bool_env("KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_CACHE_ENABLED", True),

        # Cache-Konfiguration
        cache_ttl_seconds=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_CACHE_TTL", 300),
        cache_max_size=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_CACHE_MAX_SIZE", 1000),
        cache_cleanup_interval_seconds=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_CACHE_CLEANUP_INTERVAL", 600),

        # Monitoring-Konfiguration
        metrics_window_seconds=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_METRICS_WINDOW", 300),
        alert_threshold=get_float_env("KEI_AGENT_CIRCUIT_BREAKER_ALERT_THRESHOLD", 0.8),
        statistics_retention_hours=get_int_env("KEI_AGENT_CIRCUIT_BREAKER_STATISTICS_RETENTION", 24)
    )


# Environment Template für .env Datei
AGENT_CIRCUIT_BREAKER_ENV_TEMPLATE = """
# =============================================================================
# KEIKO AGENT CIRCUIT BREAKER CONFIGURATION
# =============================================================================

# Basis-Konfiguration
KEI_AGENT_CIRCUIT_BREAKER_ENABLED=true
KEI_AGENT_CIRCUIT_BREAKER_MONITORING_ENABLED=true
KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_ENABLED=true
KEI_AGENT_CIRCUIT_BREAKER_CACHING_ENABLED=true

# Default Circuit Breaker Settings
KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_FAILURE_THRESHOLD=5
KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_RECOVERY_TIMEOUT=60
KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_SUCCESS_THRESHOLD=3
KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_TIMEOUT=30.0

# Voice Agent Settings (schnelle Response erforderlich)
KEI_AGENT_CIRCUIT_BREAKER_VOICE_FAILURE_THRESHOLD=3
KEI_AGENT_CIRCUIT_BREAKER_VOICE_TIMEOUT=10.0

# Tool Agent Settings (mittlere Toleranz)
KEI_AGENT_CIRCUIT_BREAKER_TOOL_FAILURE_THRESHOLD=5
KEI_AGENT_CIRCUIT_BREAKER_TOOL_TIMEOUT=30.0

# Workflow Agent Settings (niedrige Toleranz für Komplexität)
KEI_AGENT_CIRCUIT_BREAKER_WORKFLOW_FAILURE_THRESHOLD=2
KEI_AGENT_CIRCUIT_BREAKER_WORKFLOW_TIMEOUT=60.0

# Orchestrator Agent Settings (kritisch für System)
KEI_AGENT_CIRCUIT_BREAKER_ORCHESTRATOR_FAILURE_THRESHOLD=2
KEI_AGENT_CIRCUIT_BREAKER_ORCHESTRATOR_TIMEOUT=15.0

# Recovery-Konfiguration
KEI_AGENT_CIRCUIT_BREAKER_RECOVERY_STRATEGY=exponential_backoff
KEI_AGENT_CIRCUIT_BREAKER_MAX_RECOVERY_ATTEMPTS=3
KEI_AGENT_CIRCUIT_BREAKER_BACKOFF_MULTIPLIER=2.0
KEI_AGENT_CIRCUIT_BREAKER_MAX_BACKOFF=300

# Fallback-Konfiguration
KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_TIMEOUT=15.0
KEI_AGENT_CIRCUIT_BREAKER_MAX_FALLBACK_ATTEMPTS=2
KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_CACHE_ENABLED=true

# Cache-Konfiguration
KEI_AGENT_CIRCUIT_BREAKER_CACHE_TTL=300
KEI_AGENT_CIRCUIT_BREAKER_CACHE_MAX_SIZE=1000
KEI_AGENT_CIRCUIT_BREAKER_CACHE_CLEANUP_INTERVAL=600

# Monitoring-Konfiguration
KEI_AGENT_CIRCUIT_BREAKER_METRICS_WINDOW=300
KEI_AGENT_CIRCUIT_BREAKER_ALERT_THRESHOLD=0.8
KEI_AGENT_CIRCUIT_BREAKER_STATISTICS_RETENTION=24

# =============================================================================
# AGENT CIRCUIT BREAKER EXAMPLES
# =============================================================================

# Beispiel-Konfiguration für Development (toleranter)
# KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_FAILURE_THRESHOLD=10
# KEI_AGENT_CIRCUIT_BREAKER_VOICE_FAILURE_THRESHOLD=5
# KEI_AGENT_CIRCUIT_BREAKER_TOOL_FAILURE_THRESHOLD=8

# Beispiel-Konfiguration für Production (strenger)
# KEI_AGENT_CIRCUIT_BREAKER_DEFAULT_FAILURE_THRESHOLD=3
# KEI_AGENT_CIRCUIT_BREAKER_VOICE_FAILURE_THRESHOLD=2
# KEI_AGENT_CIRCUIT_BREAKER_ORCHESTRATOR_FAILURE_THRESHOLD=1

# Beispiel-Konfiguration für High-Performance (sehr schnell)
# KEI_AGENT_CIRCUIT_BREAKER_VOICE_TIMEOUT=5.0
# KEI_AGENT_CIRCUIT_BREAKER_TOOL_TIMEOUT=15.0
# KEI_AGENT_CIRCUIT_BREAKER_ORCHESTRATOR_TIMEOUT=10.0

# Recovery-Strategien
# KEI_AGENT_CIRCUIT_BREAKER_RECOVERY_STRATEGY=exponential_backoff  # Standard
# KEI_AGENT_CIRCUIT_BREAKER_RECOVERY_STRATEGY=linear_backoff       # Linear
# KEI_AGENT_CIRCUIT_BREAKER_RECOVERY_STRATEGY=fixed_interval       # Fest
# KEI_AGENT_CIRCUIT_BREAKER_RECOVERY_STRATEGY=adaptive             # Adaptiv

# Deaktivierung für Testing
# KEI_AGENT_CIRCUIT_BREAKER_ENABLED=false
# KEI_AGENT_CIRCUIT_BREAKER_FALLBACK_ENABLED=false
# KEI_AGENT_CIRCUIT_BREAKER_CACHING_ENABLED=false
"""


def generate_agent_circuit_breaker_env_template(file_path: str = ".env.circuit_breaker") -> None:
    """Generiert Environment-Template-Datei für Agent Circuit Breaker."""
    from kei_logging import get_logger
    logger = get_logger(__name__)

    with open(file_path, "w") as f:
        f.write(AGENT_CIRCUIT_BREAKER_ENV_TEMPLATE)

    logger.info(f"Agent circuit breaker environment template generated: {file_path}")


# Vordefinierte Konfigurationen für verschiedene Umgebungen
def get_development_settings() -> AgentCircuitBreakerSettings:
    """Entwicklungs-Konfiguration mit toleranteren Limits."""
    dev_settings = get_agent_circuit_breaker_settings()

    # Tolerantere Limits für Development
    dev_settings.default_failure_threshold = 10
    dev_settings.voice_agent_failure_threshold = 5
    dev_settings.tool_agent_failure_threshold = 8
    dev_settings.workflow_agent_failure_threshold = 4
    dev_settings.orchestrator_agent_failure_threshold = 3

    # Längere Timeouts für Debugging
    dev_settings.voice_agent_timeout_seconds = 30.0
    dev_settings.tool_agent_timeout_seconds = 60.0
    dev_settings.workflow_agent_timeout_seconds = 120.0
    dev_settings.orchestrator_agent_timeout_seconds = 30.0

    return dev_settings


def get_production_settings() -> AgentCircuitBreakerSettings:
    """Production-Konfiguration mit strengeren Limits."""
    prod_settings = get_agent_circuit_breaker_settings()

    # Strengere Limits für Production
    prod_settings.default_failure_threshold = 3
    prod_settings.voice_agent_failure_threshold = 2
    prod_settings.tool_agent_failure_threshold = 3
    prod_settings.workflow_agent_failure_threshold = 1
    prod_settings.orchestrator_agent_failure_threshold = 1

    # Kürzere Timeouts für bessere Performance
    prod_settings.voice_agent_timeout_seconds = 8.0
    prod_settings.tool_agent_timeout_seconds = 20.0
    prod_settings.workflow_agent_timeout_seconds = 45.0
    prod_settings.orchestrator_agent_timeout_seconds = 10.0

    # Erweiterte Monitoring
    prod_settings.monitoring_enabled = True
    prod_settings.alert_threshold = 0.7  # Frühere Alerts

    return prod_settings


def get_high_performance_settings() -> AgentCircuitBreakerSettings:
    """High-Performance-Konfiguration für schnelle Responses."""
    perf_settings = get_agent_circuit_breaker_settings()

    # Sehr schnelle Timeouts
    perf_settings.voice_agent_timeout_seconds = 5.0
    perf_settings.tool_agent_timeout_seconds = 15.0
    perf_settings.workflow_agent_timeout_seconds = 30.0
    perf_settings.orchestrator_agent_timeout_seconds = 8.0

    # Aggressive Failure-Thresholds
    perf_settings.voice_agent_failure_threshold = 2
    perf_settings.tool_agent_failure_threshold = 3
    perf_settings.workflow_agent_failure_threshold = 1
    perf_settings.orchestrator_agent_failure_threshold = 1

    # Optimierte Recovery
    perf_settings.recovery_strategy = RecoveryStrategy.ADAPTIVE
    perf_settings.max_recovery_attempts = 2

    # Erweiterte Caching
    perf_settings.caching_enabled = True
    perf_settings.cache_ttl_seconds = 600  # Längere Cache-Zeit
    perf_settings.cache_max_size = 2000

    return perf_settings


if __name__ == "__main__":
    from kei_logging import get_logger
    logger = get_logger(__name__)

    # Generiere Environment-Template
    generate_agent_circuit_breaker_env_template()

    # Zeige aktuelle Konfiguration
    settings = get_agent_circuit_breaker_settings()
    logger.info("Current agent circuit breaker settings:")
    for field, value in settings.__dict__.items():
        logger.info(f"  {field}: {value}")
