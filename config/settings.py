# backend/config/settings.py
"""Anwendungskonfiguration für Keiko Personal Assistant.

Stellt eine typsichere, auf Umgebungsvariablen basierende Konfiguration bereit.
Sensible Werte werden als `SecretStr` verwaltet.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_CANDIDATES: list[Path] = [
    Path(".env"),
    Path("backend/.env"),
    Path("../.env"),
]


def _load_env_file() -> Path | None:
    """Lädt die erste gefundene .env-Datei aus Standardpfaden.

    Returns:
        Pfad zur geladenen Datei oder None, wenn keine .env gefunden wurde.
    """
    for env_path in _ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return env_path
    return None


class Settings(BaseSettings):
    """Anwendungskonfiguration für Keiko Personal Assistant.

    Alle Felder lassen sich über Umgebungsvariablen überschreiben.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Core Settings
    environment: str = Field(default="development")
    debug_mode: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # Azure Configuration
    azure_subscription_id: str = Field(default="")
    azure_resource_group: str = Field(default="")
    azure_location: str = Field(default="westeurope")

    # Azure Storage
    storage_account_url: str = Field(default="https://defaultstorage.blob.core.windows.net")
    keiko_storage_container_for_img: str = Field(default="keiko-images")

    # Azure AI Services
    cosmosdb_connection: SecretStr = Field(default=SecretStr(""))
    database_name: str = Field(default="cdb_db_id_keikoPersonalAssistant")
    container_name: str = Field(default="configurations")

    # Database
    database_url: str = Field(default="sqlite:///./app.db")

    # CORS Configuration
    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    )

    # Response Cache Configuration
    response_cache_enabled: bool = Field(default=False, description="Aktiviert HTTP Response Caching")
    response_cache_rules: str = Field(default="", description="Cache-Regeln als CSV (pattern:ttl)")
    response_cache_default_ttl_seconds: int = Field(default=300, ge=0, description="Standard-TTL für Response Cache in Sekunden")

    # Tenant Configuration
    default_tenant_id: str = Field(default="public", description="Standard-Tenant für Single-Tenant-Setups")
    tenant_isolation_enabled: bool = Field(default=False, description="Aktiviert Multi-Tenant-Isolation")
    tenant_header_required: bool = Field(default=False, description="Erfordert X-Tenant-Id Header für alle Requests")

    # Authorization Configuration
    allow_anonymous_access: bool = Field(default=True, description="Erlaubt anonymen Zugriff für Development")
    default_user_id: str = Field(default="oscharko", description="Standard-Benutzer für Development")
    default_user_scopes: str = Field(default="webhook:read,webhook:write,api:read,voice:manage,admin:read", description="Standard-Scopes für Default-Benutzer")

    # Error Handling Configuration
    error_include_stack_traces: bool = Field(default=None, description="Inkludiert Stack-Traces in Error Responses (None = auto basierend auf Environment)")
    error_include_debug_info: bool = Field(default=None, description="Inkludiert Debug-Informationen in Error Responses (None = auto basierend auf Environment)")
    error_log_sensitive_data: bool = Field(default=False, description="Loggt sensitive Daten in Error-Logs (nur Development)")
    error_enable_legacy_format: bool = Field(default=True, description="Aktiviert Legacy Error Format für Backward Compatibility")
    error_help_base_url: str = Field(default="https://docs.keiko.dev/errors", description="Basis-URL für Error-Dokumentation")

    # Environment Variables Integration
    project_keiko_model_inference_endpoint: str = Field(default="")
    project_keiko_openai_endpoint: str = Field(default="")
    project_keiko_services_endpoint: str = Field(default="")
    project_keiko_api_key: str = Field(default="")
    project_keiko_project_id: str = Field(default="")
    project_keiko_model_deployment_name: str = Field(default="gpt-4o")
    project_keiko_api_version: str = Field(default="2025-05-01")

    # Image Generation (Azure OpenAI DALL·E-3 via endpoint)
    project_keiko_image_endpoint: str = Field(default="")
    project_keiko_image_api_key: SecretStr = Field(default=SecretStr(""))

    # Azure Content Safety (optional)
    azure_content_safety_endpoint: str = Field(default="")
    azure_content_safety_key: SecretStr = Field(default=SecretStr(""))

    # Azure Key Vault / Secret Management
    azure_key_vault_url: str = Field(default="")
    secret_rotation_enabled: bool = Field(default=True)
    secret_rotation_interval_days: int = Field(default=30)
    secret_grace_period_hours: int = Field(default=24)
    secret_cache_ttl_seconds: int = Field(default=300)
    inbound_hmac_secret_key_name: str = Field(default="")

    # n8n Webhook Sicherheit
    n8n_hmac_secret: SecretStr = Field(default=SecretStr("n8n_webhook_secret"))
    n8n_base_url: str = Field(default="")
    n8n_api_key: SecretStr = Field(default=SecretStr(""))

    # Redis Konfiguration (optional)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    # Webhook Worker‑Pool
    webhook_worker_count: int = Field(default=3, ge=1, le=20)
    webhook_poll_interval: float = Field(default=1.0, ge=0.05)
    # Webhook Health Prober
    webhook_health_prober_enabled: bool = Field(default=True)
    webhook_health_poll_seconds: float = Field(default=300.0)
    webhook_health_timeout_seconds: float = Field(default=5.0)
    # Webhook Circuit Breaker
    webhook_cb_window_size: int = Field(default=10)
    webhook_cb_failure_ratio: float = Field(default=0.5)
    webhook_cb_open_timeout_seconds: float = Field(default=60.0)
    webhook_cb_half_open_timeout_seconds: float = Field(default=30.0)
    webhook_cb_half_open_max_calls: int = Field(default=1)
    # Webhook Sanitization
    sanitizer_allowed_html_tags: str = Field(default="b,i,u,strong,em,br,span")
    sanitizer_max_payload_bytes: int = Field(default=524288)
    sanitizer_allowed_content_types: str = Field(default="application/json,text/json,application/problem+json")
    # Event‑spezifische Regeln (kommagetrennte Paare event:tags)
    sanitizer_event_tag_overrides: str = Field(default="")

    # HTTP/2 Konfiguration für Webhook Outbound
    webhook_http2_enabled: bool = Field(default=False)
    webhook_http2_max_connections: int = Field(default=100, ge=1, le=10000)
    webhook_http2_max_keepalive: int = Field(default=20, ge=0, le=10000)

    # Agent-IDs (aus .env)
    agent_orchestrator_id: str = Field(default="")
    agent_bing_search_id: str = Field(default="")
    agent_image_generator_id: str = Field(default="")

    # Project Keiko Voice Service
    project_keiko_voice_endpoint: str = Field(default="")
    project_keiko_debug_voice_events: bool = Field(default=False)

    # Observability / Monitoring
    app_insights_connection_string: str = Field(default="")

    # LangSmith Observability
    langsmith_enabled: bool = Field(default=True)
    langsmith_api_key: SecretStr = Field(default=SecretStr(""))
    langsmith_project_name: str = Field(default="keiko")
    langsmith_sampling_rate: float = Field(default=1.0)
    langsmith_otel_bridge: bool = Field(default=True)

    # Logging Configuration
    kei_logging_pii_mask: str = Field(default="***REDACTED***")
    kei_logging_pii_rules: str = Field(default="")
    keiko_enable_clickable_links: bool = Field(default=True)

    # OpenTelemetry Configuration
    otel_service_name: str = Field(default="keiko-backend")
    otel_service_version: str = Field(default="1.0.0")
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_exporter_otlp_protocol: str = Field(default="grpc")
    otel_traces_enabled: bool = Field(default=True)
    otel_metrics_enabled: bool = Field(default=True)
    otel_sampling_ratio: float = Field(default=1.0)

    # Webhook Configuration (bereits oben definiert)

    # Rate Limiting (KEI_MCP)
    kei_mcp_rate_limit_backend: str = Field(default="hybrid")
    kei_mcp_redis_host: str = Field(default="localhost")
    kei_mcp_redis_port: int = Field(default=6379)
    kei_mcp_redis_db: int = Field(default=1)
    kei_mcp_redis_password: SecretStr = Field(default=SecretStr(""))
    kei_mcp_redis_ssl: bool = Field(default=False)
    kei_mcp_redis_timeout: float = Field(default=5.0)
    kei_mcp_memory_fallback: bool = Field(default=True)
    kei_mcp_default_tier: str = Field(default="basic")
    kei_mcp_rate_limit_metrics: bool = Field(default=True)
    kei_mcp_rate_limit_logging: bool = Field(default=True)
    kei_mcp_rate_limit_headers: bool = Field(default=True)
    kei_mcp_cleanup_interval: int = Field(default=3600)
    kei_mcp_ip_fallback: bool = Field(default=True)
    kei_mcp_ip_requests_per_minute: int = Field(default=20)

    # Security & RBAC
    rbac_roles_json: str = Field(default="[]")
    rbac_assignments_json: str = Field(default="[]")

    # DDoS Protection
    ddos_enabled: bool = Field(default=True, description="DDoS-Schutz aktivieren")
    ddos_trust_proxy_headers: bool = Field(default=True, description="Proxy-Headers vertrauen")
    ddos_anomaly_requests_per_minute: int = Field(default=600, ge=1, description="Anomalie-Schwellwert für Requests pro Minute")
    ddos_burst_threshold: int = Field(default=120, ge=1, description="Burst-Schwellwert für DDoS-Erkennung")
    ddos_ip_block_duration_seconds: int = Field(default=900, ge=0, description="IP-Block-Dauer bei DDoS in Sekunden")

    # HSTS (HTTP Strict Transport Security)
    hsts_enabled: bool = Field(default=False, description="HSTS aktivieren")
    hsts_https_only: bool = Field(default=True, description="HSTS nur bei HTTPS setzen")
    hsts_max_age: int = Field(default=31536000, ge=0, description="HSTS max-age in Sekunden")
    hsts_include_subdomains: bool = Field(default=True, description="HSTS includeSubDomains")
    hsts_preload: bool = Field(default=True, description="HSTS preload")

    # Alerting (Webhook-basierte Adapter)
    alerting_enabled: bool = Field(default=False)
    alert_slack_webhook_url: SecretStr = Field(default=SecretStr(""))
    alert_teams_webhook_url: SecretStr = Field(default=SecretStr(""))
    alert_rate_limit_per_minute: int = Field(default=30, ge=1, le=1000)
    alert_retry_max_attempts: int = Field(default=3, ge=1, le=10)
    alert_retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=30.0)
    # Schwellwerte
    alert_threshold_dlq_overflow: int = Field(default=1000, ge=1)
    alert_threshold_retry_rate_warning: float = Field(default=0.2, ge=0.0, le=1.0)
    alert_threshold_worker_pool_min_active: int = Field(default=1, ge=0)

    @field_validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validiert Environment-Werte."""
        allowed = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"Environment muss einer von {allowed} sein")
        return v.lower()

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validiert Log-Level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log Level muss einer von {allowed} sein")
        return v.upper()

    @property
    def is_development(self) -> bool:
        """Prüft ob Development-Environment."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Prüft ob Production-Environment."""
        return self.environment == "production"

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Gibt CORS Origins als Liste zurück."""
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    def get_config_summary(self) -> dict[str, Any]:
        """Gibt eine sensible-freundliche Konfigurationsübersicht zurück."""
        return {
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "cors_origins": len(self.cors_allowed_origins_list),
            "azure": {
                "subscription_configured": bool(self.azure_subscription_id and self.azure_resource_group),
                "storage_account_url_set": bool(self.storage_account_url),
                "key_vault_configured": bool(self.azure_key_vault_url),
            },
            "database": {
                "url_set": bool(self.database_url),
                "cosmos_connection_set": bool(self.cosmosdb_connection),
            },
            "alerting": {
                "enabled": self.alerting_enabled,
                "slack_configured": bool(self.alert_slack_webhook_url.get_secret_value()),
                "teams_configured": bool(self.alert_teams_webhook_url.get_secret_value()),
                "rate_limit_per_minute": self.alert_rate_limit_per_minute,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton-Factory für Settings."""
    _load_env_file()
    return Settings()


# Globale Instanz
settings: Settings = get_settings()


def create_test_settings(**overrides: Any) -> Settings:
    """Erstellt Test-Settings mit Overrides.

    Setzt temporär Umgebungsvariablen, um die Settings-Instanz mit
    überschriebenen Werten zu erzeugen, und stellt diese anschließend wieder her.
    """
    original_env = {}
    for key, value in overrides.items():
        env_key = key.upper()
        if env_key in os.environ:
            original_env[env_key] = os.environ[env_key]
        os.environ[env_key] = str(value)

    try:
        get_settings.cache_clear()
        return get_settings()
    finally:
        # Original-Environment wiederherstellen
        for key in overrides:
            env_key = key.upper()
            if env_key in original_env:
                os.environ[env_key] = original_env[env_key]
            elif env_key in os.environ:
                del os.environ[env_key]


__all__ = [
    "Settings",
    "create_test_settings",
    "get_settings",
    "settings",
    "validate_environment_or_raise",
]


# --- NEU: Startup-Validierung für .env und Pflichtvariablen ---
from kei_logging import get_logger  # noqa: E402

_env_logger = get_logger(__name__)

# Pflicht-Keys (abhängig von Features); Voice/Image sind kritisch, wenn verwendet
REQUIRED_ENV_KEYS: list[str] = [
    # Azure OpenAI (Core)
    "PROJECT_KEIKO_API_KEY",
    # Voice Realtime
    "PROJECT_KEIKO_VOICE_ENDPOINT",
    # Images (DALL·E-3)
    "PROJECT_KEIKO_IMAGE_ENDPOINT",
    # Storage für Bilder
    "STORAGE_ACCOUNT_URL",
    # Authentication
    "KEI_API_TOKEN",
    "KEI_MCP_API_TOKEN",
    # Environment
    "ENVIRONMENT",
]


def validate_environment_or_raise() -> None:
    """Validiert, dass eine .env vorhanden ist und Pflicht-Variablen gesetzt sind.

    Raises:
        RuntimeError: wenn .env fehlt oder Pflichtvariablen nicht gesetzt sind
    """
    found_env = _load_env_file()
    if found_env is None:
        raise RuntimeError(
            "Keine .env-Datei gefunden. Bitte erstellen Sie eine .env im Projektstamm oder unter backend/.env."
        )

    missing: list[str] = []
    for key in REQUIRED_ENV_KEYS:
        if not os.environ.get(key, "").strip():
            missing.append(key)

    # LangSmith nur in Production verpflichtend, wenn aktiviert
    if os.environ.get("ENVIRONMENT", "development").lower() == "production":
        if os.environ.get("LANGSMITH_ENABLED", "true").lower() == "true":
            if not os.environ.get("LANGSMITH_API_KEY", "").strip():
                missing.append("LANGSMITH_API_KEY")
            if not os.environ.get("LANGSMITH_PROJECT_NAME", "").strip() and not os.environ.get("LANGSMITH_PROJECT", "").strip():
                missing.append("LANGSMITH_PROJECT_NAME")

    if missing:
        raise RuntimeError(
            "Fehlende erforderliche Umgebungsvariablen: " + ", ".join(missing) +
            ". Bitte prüfen Sie die .env-Datei."
        )
