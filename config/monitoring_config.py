"""Monitoring Configuration für Keiko Personal Assistant.
Konfiguration für das Comprehensive Monitoring System.
"""

import os
from dataclasses import dataclass


@dataclass
class MonitoringSettings:
    """Monitoring-Konfiguration aus Environment-Variablen."""

    # Basis-Konfiguration
    enabled: bool = True
    metrics_enabled: bool = True
    health_checks_enabled: bool = True
    alerts_enabled: bool = True
    voice_monitoring_enabled: bool = True
    performance_monitoring_enabled: bool = True

    # Export-Konfiguration
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    grafana_enabled: bool = True

    # Health-Check-Konfiguration
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5

    # Alert-Konfiguration
    alert_webhook_url: str | None = None
    alert_email_enabled: bool = False
    alert_slack_enabled: bool = False
    alert_cooldown_seconds: int = 300

    # Performance-Thresholds
    response_time_threshold_ms: float = 1000.0
    error_rate_threshold_percent: float = 5.0
    cpu_threshold_percent: float = 80.0
    memory_threshold_mb: float = 1024.0

    # Voice-Workflow-Thresholds
    voice_stt_threshold_ms: float = 2000.0
    voice_orchestrator_threshold_ms: float = 1000.0
    voice_agent_execution_threshold_ms: float = 5000.0
    voice_failure_rate_threshold_percent: float = 10.0

    # Circuit Breaker-Konfiguration
    circuit_breaker_enabled: bool = True
    azure_openai_failure_threshold: int = 5
    azure_openai_recovery_timeout_seconds: int = 60
    redis_failure_threshold: int = 3
    redis_recovery_timeout_seconds: int = 30

    # Retention-Konfiguration
    metrics_retention_hours: int = 24
    alert_history_retention_hours: int = 168  # 1 week
    workflow_history_retention_count: int = 1000


def get_monitoring_settings() -> MonitoringSettings:
    """Lädt Monitoring-Konfiguration aus Environment-Variablen."""

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

    return MonitoringSettings(
        # Basis-Konfiguration
        enabled=get_bool_env("KEI_MONITORING_ENABLED", True),
        metrics_enabled=get_bool_env("KEI_MONITORING_METRICS_ENABLED", True),
        health_checks_enabled=get_bool_env("KEI_MONITORING_HEALTH_CHECKS_ENABLED", True),
        alerts_enabled=get_bool_env("KEI_MONITORING_ALERTS_ENABLED", True),
        voice_monitoring_enabled=get_bool_env("KEI_MONITORING_VOICE_ENABLED", True),
        performance_monitoring_enabled=get_bool_env("KEI_MONITORING_PERFORMANCE_ENABLED", True),

        # Export-Konfiguration
        prometheus_enabled=get_bool_env("KEI_MONITORING_PROMETHEUS_ENABLED", True),
        prometheus_port=get_int_env("KEI_MONITORING_PROMETHEUS_PORT", 9090),
        prometheus_path=os.getenv("KEI_MONITORING_PROMETHEUS_PATH", "/metrics"),
        grafana_enabled=get_bool_env("KEI_MONITORING_GRAFANA_ENABLED", True),

        # Health-Check-Konfiguration
        health_check_interval_seconds=get_int_env("KEI_MONITORING_HEALTH_CHECK_INTERVAL", 30),
        health_check_timeout_seconds=get_int_env("KEI_MONITORING_HEALTH_CHECK_TIMEOUT", 5),

        # Alert-Konfiguration
        alert_webhook_url=os.getenv("KEI_MONITORING_ALERT_WEBHOOK_URL"),
        alert_email_enabled=get_bool_env("KEI_MONITORING_ALERT_EMAIL_ENABLED", False),
        alert_slack_enabled=get_bool_env("KEI_MONITORING_ALERT_SLACK_ENABLED", False),
        alert_cooldown_seconds=get_int_env("KEI_MONITORING_ALERT_COOLDOWN", 300),

        # Performance-Thresholds
        response_time_threshold_ms=get_float_env("KEI_MONITORING_RESPONSE_TIME_THRESHOLD_MS", 1000.0),
        error_rate_threshold_percent=get_float_env("KEI_MONITORING_ERROR_RATE_THRESHOLD", 5.0),
        cpu_threshold_percent=get_float_env("KEI_MONITORING_CPU_THRESHOLD", 80.0),
        memory_threshold_mb=get_float_env("KEI_MONITORING_MEMORY_THRESHOLD_MB", 1024.0),

        # Voice-Workflow-Thresholds
        voice_stt_threshold_ms=get_float_env("KEI_MONITORING_VOICE_STT_THRESHOLD_MS", 2000.0),
        voice_orchestrator_threshold_ms=get_float_env("KEI_MONITORING_VOICE_ORCHESTRATOR_THRESHOLD_MS", 1000.0),
        voice_agent_execution_threshold_ms=get_float_env("KEI_MONITORING_VOICE_AGENT_EXECUTION_THRESHOLD_MS", 5000.0),
        voice_failure_rate_threshold_percent=get_float_env("KEI_MONITORING_VOICE_FAILURE_RATE_THRESHOLD", 10.0),

        # Circuit Breaker-Konfiguration
        circuit_breaker_enabled=get_bool_env("KEI_MONITORING_CIRCUIT_BREAKER_ENABLED", True),
        azure_openai_failure_threshold=get_int_env("KEI_MONITORING_AZURE_OPENAI_FAILURE_THRESHOLD", 5),
        azure_openai_recovery_timeout_seconds=get_int_env("KEI_MONITORING_AZURE_OPENAI_RECOVERY_TIMEOUT", 60),
        redis_failure_threshold=get_int_env("KEI_MONITORING_REDIS_FAILURE_THRESHOLD", 3),
        redis_recovery_timeout_seconds=get_int_env("KEI_MONITORING_REDIS_RECOVERY_TIMEOUT", 30),

        # Retention-Konfiguration
        metrics_retention_hours=get_int_env("KEI_MONITORING_METRICS_RETENTION_HOURS", 24),
        alert_history_retention_hours=get_int_env("KEI_MONITORING_ALERT_HISTORY_RETENTION_HOURS", 168),
        workflow_history_retention_count=get_int_env("KEI_MONITORING_WORKFLOW_HISTORY_RETENTION_COUNT", 1000)
    )


def create_monitoring_config_from_settings():
    """Erstellt MonitoringConfig aus MonitoringSettings."""
    from monitoring.interfaces import MonitoringConfig

    monitoring_settings = get_monitoring_settings()

    return MonitoringConfig(
        enabled=monitoring_settings.enabled,
        metrics_enabled=monitoring_settings.metrics_enabled,
        health_checks_enabled=monitoring_settings.health_checks_enabled,
        alerts_enabled=monitoring_settings.alerts_enabled,
        voice_monitoring_enabled=monitoring_settings.voice_monitoring_enabled,
        performance_monitoring_enabled=monitoring_settings.performance_monitoring_enabled,

        prometheus_enabled=monitoring_settings.prometheus_enabled,
        prometheus_port=monitoring_settings.prometheus_port,
        grafana_enabled=monitoring_settings.grafana_enabled,

        health_check_interval_seconds=monitoring_settings.health_check_interval_seconds,
        health_check_timeout_seconds=monitoring_settings.health_check_timeout_seconds,

        alert_webhook_url=monitoring_settings.alert_webhook_url,
        alert_email_enabled=monitoring_settings.alert_email_enabled,
        alert_slack_enabled=monitoring_settings.alert_slack_enabled,

        response_time_threshold_ms=monitoring_settings.response_time_threshold_ms,
        error_rate_threshold_percent=monitoring_settings.error_rate_threshold_percent,
        cpu_threshold_percent=monitoring_settings.cpu_threshold_percent,
        memory_threshold_mb=monitoring_settings.memory_threshold_mb
    )


# Environment Template für .env Datei
MONITORING_ENV_TEMPLATE = """
# =============================================================================
# KEIKO MONITORING CONFIGURATION
# =============================================================================

# Basis-Monitoring
KEI_MONITORING_ENABLED=true
KEI_MONITORING_METRICS_ENABLED=true
KEI_MONITORING_HEALTH_CHECKS_ENABLED=true
KEI_MONITORING_ALERTS_ENABLED=true
KEI_MONITORING_VOICE_ENABLED=true
KEI_MONITORING_PERFORMANCE_ENABLED=true

# Prometheus Export
KEI_MONITORING_PROMETHEUS_ENABLED=true
KEI_MONITORING_PROMETHEUS_PORT=9090
KEI_MONITORING_PROMETHEUS_PATH=/metrics

# Grafana Integration
KEI_MONITORING_GRAFANA_ENABLED=true

# Health Checks
KEI_MONITORING_HEALTH_CHECK_INTERVAL=30
KEI_MONITORING_HEALTH_CHECK_TIMEOUT=5

# Alerts
KEI_MONITORING_ALERT_WEBHOOK_URL=
KEI_MONITORING_ALERT_EMAIL_ENABLED=false
KEI_MONITORING_ALERT_SLACK_ENABLED=false
KEI_MONITORING_ALERT_COOLDOWN=300

# Performance Thresholds
KEI_MONITORING_RESPONSE_TIME_THRESHOLD_MS=1000.0
KEI_MONITORING_ERROR_RATE_THRESHOLD=5.0
KEI_MONITORING_CPU_THRESHOLD=80.0
KEI_MONITORING_MEMORY_THRESHOLD_MB=1024.0

# Voice Workflow Thresholds
KEI_MONITORING_VOICE_STT_THRESHOLD_MS=2000.0
KEI_MONITORING_VOICE_ORCHESTRATOR_THRESHOLD_MS=1000.0
KEI_MONITORING_VOICE_AGENT_EXECUTION_THRESHOLD_MS=5000.0
KEI_MONITORING_VOICE_FAILURE_RATE_THRESHOLD=10.0

# Circuit Breaker
KEI_MONITORING_CIRCUIT_BREAKER_ENABLED=true
KEI_MONITORING_AZURE_OPENAI_FAILURE_THRESHOLD=5
KEI_MONITORING_AZURE_OPENAI_RECOVERY_TIMEOUT=60
KEI_MONITORING_REDIS_FAILURE_THRESHOLD=3
KEI_MONITORING_REDIS_RECOVERY_TIMEOUT=30

# Data Retention
KEI_MONITORING_METRICS_RETENTION_HOURS=24
KEI_MONITORING_ALERT_HISTORY_RETENTION_HOURS=168
KEI_MONITORING_WORKFLOW_HISTORY_RETENTION_COUNT=1000
"""


def generate_env_template(file_path: str = ".env.monitoring") -> None:
    """Generiert Environment-Template-Datei."""
    from kei_logging import get_logger
    logger = get_logger(__name__)

    with open(file_path, "w") as f:
        f.write(MONITORING_ENV_TEMPLATE)

    logger.info(f"Monitoring environment template generated: {file_path}")


if __name__ == "__main__":
    from kei_logging import get_logger
    logger = get_logger(__name__)

    # Generiere Environment-Template
    generate_env_template()

    # Zeige aktuelle Konfiguration
    settings = get_monitoring_settings()
    logger.info("Current monitoring settings:")
    for field, value in settings.__dict__.items():
        logger.info(f"  {field}: {value}")
