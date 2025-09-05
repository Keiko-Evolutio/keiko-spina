"""Monitoring Settings für Keiko Personal Assistant.

Alle monitoring- und observability-bezogenen Konfigurationen.
Folgt Single Responsibility Principle.
"""


from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

from .constants import (
    DEFAULT_ANOMALY_MODEL_VERSION,
    DEFAULT_ANOMALY_TRAINING_INTERVAL_MINUTES,
    DEFAULT_GRAFANA_URL,
    DEFAULT_REPORTING_INTERVAL_MINUTES,
    DEFAULT_SLA_AVAILABILITY_TARGET_PCT,
    DEFAULT_SLA_ERROR_RATE_TARGET_PCT,
    DEFAULT_SLA_LATENCY_TARGET_MS,
    MAX_ANOMALY_TRAINING_INTERVAL_MINUTES,
    MAX_REPORTING_INTERVAL_MINUTES,
    MIN_ANOMALY_TRAINING_INTERVAL_MINUTES,
    MIN_REPORTING_INTERVAL_MINUTES,
)
from .env_utils import get_env_bool, get_env_float, get_env_int, get_env_list, get_env_str


class MonitoringSettings(BaseSettings):
    """Monitoring- und Observability-spezifische Konfigurationen."""

    # Grafana Integration
    grafana_url: str = Field(
        default=DEFAULT_GRAFANA_URL,
        description="Grafana Dashboard URL"
    )

    grafana_api_token: SecretStr = Field(
        default=SecretStr(""),
        description="Grafana API Token für Exports"
    )

    # Reporting Configuration
    reporting_enabled: bool = Field(
        default=False,
        description="Automatische Reports aktivieren"
    )

    reporting_interval_minutes: int = Field(
        default=DEFAULT_REPORTING_INTERVAL_MINUTES,
        ge=MIN_REPORTING_INTERVAL_MINUTES,
        le=MAX_REPORTING_INTERVAL_MINUTES,
        description="Report-Intervall in Minuten"
    )

    reporting_default_recipients: str = Field(
        default="",
        description="Standard-Empfänger für Reports (CSV E-Mail-Adressen)"
    )

    # Metrics Collection
    enable_prometheus_metrics: bool = Field(
        default=True,
        description="Prometheus-Metriken aktivieren"
    )

    metrics_port: int = Field(
        default=8090,
        description="Port für Metrics-Endpoint"
    )

    metrics_path: str = Field(
        default="/metrics",
        description="Pfad für Metrics-Endpoint"
    )

    # Health Checks
    enable_health_checks: bool = Field(
        default=True,
        description="Health Check Endpoints aktivieren"
    )

    health_check_interval_seconds: int = Field(
        default=30,
        description="Health Check Intervall in Sekunden"
    )

    health_check_timeout_seconds: int = Field(
        default=10,
        description="Health Check Timeout in Sekunden"
    )

    # Tracing
    enable_tracing: bool = Field(
        default=True,
        description="Distributed Tracing aktivieren"
    )

    tracing_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Tracing Sample Rate (0.0-1.0)"
    )

    jaeger_endpoint: str = Field(
        default="",
        description="Jaeger Collector Endpoint"
    )

    # Anomaly Detection
    anomaly_training_enabled: bool = Field(
        default=False,
        description="Anomalie-Erkennung Training aktivieren"
    )

    anomaly_training_interval_minutes: int = Field(
        default=DEFAULT_ANOMALY_TRAINING_INTERVAL_MINUTES,
        ge=MIN_ANOMALY_TRAINING_INTERVAL_MINUTES,
        le=MAX_ANOMALY_TRAINING_INTERVAL_MINUTES,
        description="Anomalie-Training Intervall in Minuten"
    )

    anomaly_model_version: str = Field(
        default=DEFAULT_ANOMALY_MODEL_VERSION,
        description="Version des Anomalie-Erkennungsmodells"
    )

    # SLA Targets
    sla_availability_target_pct: float = Field(
        default=DEFAULT_SLA_AVAILABILITY_TARGET_PCT,
        ge=0.0,
        le=100.0,
        description="SLA Verfügbarkeitsziel in Prozent"
    )

    sla_latency_target_ms: float = Field(
        default=DEFAULT_SLA_LATENCY_TARGET_MS,
        ge=0.0,
        description="SLA Latenz-Ziel in Millisekunden"
    )

    sla_error_rate_target_pct: float = Field(
        default=DEFAULT_SLA_ERROR_RATE_TARGET_PCT,
        ge=0.0,
        le=100.0,
        description="SLA Fehlerrate-Ziel in Prozent"
    )

    # Alerting
    enable_alerting: bool = Field(
        default=True,
        description="Alerting-System aktivieren"
    )

    alert_webhook_url: str = Field(
        default="",
        description="Webhook URL für Alerts"
    )

    alert_channels: list[str] = Field(
        default_factory=lambda: ["email", "webhook"],
        description="Aktivierte Alert-Kanäle"
    )

    # Log Aggregation
    enable_log_aggregation: bool = Field(
        default=True,
        description="Log-Aggregation aktivieren"
    )

    log_retention_days: int = Field(
        default=30,
        description="Log-Aufbewahrung in Tagen"
    )

    elasticsearch_url: str = Field(
        default="",
        description="Elasticsearch URL für Log-Aggregation"
    )

    class Config:
        """Pydantic-Konfiguration."""
        env_prefix = "MONITORING_"
        case_sensitive = False


def load_monitoring_settings() -> MonitoringSettings:
    """Lädt Monitoring Settings aus Umgebungsvariablen.

    Returns:
        MonitoringSettings-Instanz
    """
    return MonitoringSettings(
        grafana_url=get_env_str("GRAFANA_URL", DEFAULT_GRAFANA_URL),
        grafana_api_token=SecretStr(get_env_str("GRAFANA_API_TOKEN")),
        reporting_enabled=get_env_bool("REPORTING_ENABLED", False),
        reporting_interval_minutes=get_env_int(
            "REPORTING_INTERVAL_MINUTES",
            DEFAULT_REPORTING_INTERVAL_MINUTES,
            MIN_REPORTING_INTERVAL_MINUTES,
            MAX_REPORTING_INTERVAL_MINUTES
        ),
        reporting_default_recipients=get_env_str("REPORTING_DEFAULT_RECIPIENTS"),
        enable_prometheus_metrics=get_env_bool("ENABLE_PROMETHEUS_METRICS", True),
        metrics_port=get_env_int("METRICS_PORT", 8090),
        metrics_path=get_env_str("METRICS_PATH", "/metrics"),
        enable_health_checks=get_env_bool("ENABLE_HEALTH_CHECKS", True),
        health_check_interval_seconds=get_env_int("HEALTH_CHECK_INTERVAL_SECONDS", 30),
        health_check_timeout_seconds=get_env_int("HEALTH_CHECK_TIMEOUT_SECONDS", 10),
        enable_tracing=get_env_bool("ENABLE_TRACING", True),
        tracing_sample_rate=get_env_float("TRACING_SAMPLE_RATE", 0.1, 0.0, 1.0),
        jaeger_endpoint=get_env_str("JAEGER_ENDPOINT"),
        anomaly_training_enabled=get_env_bool("ANOMALY_TRAINING_ENABLED", False),
        anomaly_training_interval_minutes=get_env_int(
            "ANOMALY_TRAINING_INTERVAL_MINUTES",
            DEFAULT_ANOMALY_TRAINING_INTERVAL_MINUTES,
            MIN_ANOMALY_TRAINING_INTERVAL_MINUTES,
            MAX_ANOMALY_TRAINING_INTERVAL_MINUTES
        ),
        anomaly_model_version=get_env_str("ANOMALY_MODEL_VERSION", DEFAULT_ANOMALY_MODEL_VERSION),
        sla_availability_target_pct=get_env_float("SLA_AVAILABILITY_TARGET_PCT", DEFAULT_SLA_AVAILABILITY_TARGET_PCT, 0.0, 100.0),
        sla_latency_target_ms=get_env_float("SLA_LATENCY_TARGET_MS", DEFAULT_SLA_LATENCY_TARGET_MS, 0.0),
        sla_error_rate_target_pct=get_env_float("SLA_ERROR_RATE_TARGET_PCT", DEFAULT_SLA_ERROR_RATE_TARGET_PCT, 0.0, 100.0),
        enable_alerting=get_env_bool("ENABLE_ALERTING", True),
        alert_webhook_url=get_env_str("ALERT_WEBHOOK_URL"),
        alert_channels=get_env_list("ALERT_CHANNELS", ["email", "webhook"]),
        enable_log_aggregation=get_env_bool("ENABLE_LOG_AGGREGATION", True),
        log_retention_days=get_env_int("LOG_RETENTION_DAYS", 30),
        elasticsearch_url=get_env_str("ELASTICSEARCH_URL")
    )


# Globale Monitoring Settings Instanz
monitoring_settings = load_monitoring_settings()


__all__ = [
    "MonitoringSettings",
    "load_monitoring_settings",
    "monitoring_settings"
]
